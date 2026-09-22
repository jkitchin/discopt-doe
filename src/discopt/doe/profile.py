"""Profile likelihood confidence intervals and identifiability.

Implements the Raue et al. (2009) profile-likelihood algorithm on top
of :func:`discopt.estimate.estimate_parameters`. For each parameter to
profile, we step outward from the global estimate, re-solve the
estimation problem with the parameter fixed at each step, and record
the resulting objective.

Likelihood convention
---------------------
``estimate_parameters`` minimizes

    D(theta) = sum_i ((y_i - yhat_i(theta)) / sigma_i)^2

which is the *deviance*, i.e. ``2 * negative log-likelihood`` (up to a
constant) under Gaussian noise. The profile-likelihood confidence
region therefore uses the deviance form of the likelihood-ratio test,

    D(theta_i = c) - D(theta_hat) <= chi2_{1, 1-alpha},

with no factor of 1/2. All thresholds in this module use this
convention, matching ``result.objective`` directly.

References
----------
Raue, A., Kreutz, C., Maiwald, T., et al. Structural and practical
identifiability analysis of partially observed dynamical models by
exploiting the profile likelihood. *Bioinformatics* 25, 1923-1929
(2009).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import chi2

from discopt.doe._estimation import DevianceFunction, estimate_parameters, parameter_bounds
from discopt.doe._logging import quiet_solver
from discopt.estimate import EstimationResult, Experiment

ProfileShape = Literal["bounded", "one_sided_lower", "one_sided_upper", "flat"]


@dataclass
class ProfileLikelihoodResult:
    """Result of a single-parameter profile-likelihood scan.

    Attributes
    ----------
    parameter : str
        Name of the profiled parameter, or of the profiled combination
        (e.g. ``"k*K"``).
    theta_values : numpy.ndarray
        Parameter values visited, sorted ascending.
    neg_log_lik : numpy.ndarray
        Deviance ``D(theta)`` at each ``theta`` value. Despite the name
        (kept for user-facing familiarity) these are deviance values,
        ``2 * NLL``, matching ``estimate_parameters.objective``.
    theta_hat : float
        Maximum-likelihood estimate of the profiled parameter.
    objective_hat : float
        Deviance at ``theta_hat``.
    ci_lower : float or None
        Lower confidence-interval bound (linear interpolation between
        straddling grid points). ``None`` if the profile never crosses
        the threshold on the lower side.
    ci_upper : float or None
        Upper confidence-interval bound. ``None`` if unbounded above.
    confidence_level : float
        Nominal confidence level used to set the threshold.
    threshold : float
        The deviance threshold ``D(theta_hat) + chi2.ppf(confidence_level, 1)``.
    shape : str
        One of ``"bounded"``, ``"one_sided_lower"``,
        ``"one_sided_upper"``, ``"flat"``.
    warnings : list[str]
        Human-readable flags (non-monotone arm, NLP retries, etc.).
    """

    parameter: str
    theta_values: np.ndarray
    neg_log_lik: np.ndarray
    theta_hat: float
    objective_hat: float
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float
    threshold: float
    shape: ProfileShape
    warnings: list[str]


def profile_likelihood(
    experiment: Experiment,
    data: dict,
    parameter_name: str | None = None,
    *,
    expression: str | None = None,
    function: Callable[[Mapping[str, float]], float] | None = None,
    name: str | None = None,
    confidence_level: float = 0.95,
    max_steps: int = 40,
    target_delta_loglik: float = 0.2,
    initial_estimate: EstimationResult | None = None,
    initial_step: float | None = None,
    design: Mapping[str, float] | None = None,
    quiet: bool = True,
) -> ProfileLikelihoodResult:
    """Compute the profile likelihood for one parameter, or for a combination.

    Pass ``parameter_name`` to profile a single parameter, or ``expression``
    (e.g. ``"k*K"``, written in the parameter names) or ``function`` (any
    callable of the parameter dict) to profile a derived quantity. A
    combination is profiled by constrained re-optimization: at each value ``c``
    the deviance is minimized over all parameters subject to ``g(theta) = c``.
    This is how to show that a product is identifiable when its factors are
    not: the profile of ``k*K`` is bounded while those of ``k`` and ``K`` are
    flat.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition; must expose ``parameter_name`` in its
        ``unknown_parameters``.
    data : dict
        Observed response values (same format as
        :func:`estimate_parameters`).
    parameter_name : str
        Parameter to profile.
    confidence_level : float, default 0.95
        Nominal coverage level.
    max_steps : int, default 40
        Maximum number of profile points per direction.
    target_delta_loglik : float, default 0.2
        Target deviance increment per step. The step size is adjusted
        after each solve to approximately achieve this (Raue 2009 Sec 2.3).
    initial_estimate : EstimationResult, optional
        Pre-computed global fit. If omitted, one is computed.
    initial_step : float, optional
        Override the starting step size. Defaults to a curvature-based
        estimate from the FIM diagonal (for a combination, from the
        delta-method variance of ``g``).
    expression : str, optional
        A combination of the parameters to profile instead of one parameter.
    function : callable, optional
        ``function(theta_dict) -> float``, a combination given as code.
    name : str, optional
        Label for a ``function`` in the result (defaults to its ``__name__``).
    design : mapping, optional
        Design conditions, for experiments whose fit needs them (e.g. an
        :class:`~discopt.doe.ODEExperiment` with design inputs).
    quiet : bool, default True
        Hide the base solver's per-solve log messages.

    Returns
    -------
    ProfileLikelihoodResult
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    given = sum(x is not None for x in (parameter_name, expression, function))
    if given != 1:
        raise ValueError("pass exactly one of parameter_name, expression or function")
    extra: dict[str, Any] = {"design": dict(design)} if design else {}

    with quiet_solver(quiet):
        if initial_estimate is None:
            initial_estimate = estimate_parameters(experiment, data, **extra)
        if parameter_name is None:
            return _profile_combination(
                experiment,
                data,
                initial_estimate,
                expression=expression,
                function=function,
                name=name,
                confidence_level=confidence_level,
                max_steps=max_steps,
                target=target_delta_loglik,
                initial_step=initial_step,
                design=design,
            )
        return _profile_parameter(
            experiment,
            data,
            parameter_name,
            initial_estimate,
            confidence_level=confidence_level,
            max_steps=max_steps,
            target_delta_loglik=target_delta_loglik,
            initial_step=initial_step,
            extra=extra,
        )


def _profile_parameter(
    experiment: Experiment,
    data: dict,
    parameter_name: str,
    initial_estimate: EstimationResult,
    *,
    confidence_level: float,
    max_steps: int,
    target_delta_loglik: float,
    initial_step: float | None,
    extra: dict[str, Any],
) -> ProfileLikelihoodResult:
    if parameter_name not in initial_estimate.parameters:
        raise KeyError(
            f"{parameter_name!r} not in estimated parameters ({list(initial_estimate.parameters)})"
        )

    theta_hat = initial_estimate.parameters[parameter_name]
    objective_hat = float(initial_estimate.objective)
    threshold_offset = float(chi2.ppf(confidence_level, df=1))
    threshold = objective_hat + threshold_offset

    lb, ub = parameter_bounds(experiment, initial_estimate.parameters).get(
        parameter_name, (-np.inf, np.inf)
    )

    # Curvature-based initial step size from FIM diagonal.
    if initial_step is None:
        idx = initial_estimate.parameter_names.index(parameter_name)
        fim_diag = float(initial_estimate.fim[idx, idx])
        # Deviance curvature: D(theta_hat + h) ~ D(theta_hat) + fim_diag * h^2
        # (D = 2*NLL and NLL curvature ~ 0.5*fim_diag). The extra factor 2 below
        # makes this a deliberately conservative (smaller) first step; the
        # adaptive loop grows it as needed.
        if fim_diag > 0:
            initial_step = np.sqrt(target_delta_loglik / (2.0 * fim_diag))
        else:
            initial_step = 1e-2 * max(abs(theta_hat), 1.0)

    warnings_out: list[str] = []
    lower_points, lower_obj = _profile_direction(
        experiment,
        data,
        parameter_name,
        theta_hat,
        objective_hat,
        direction=-1,
        bound=lb,
        step0=initial_step,
        threshold=threshold,
        max_steps=max_steps,
        target=target_delta_loglik,
        other_init=dict(initial_estimate.parameters),
        warnings_out=warnings_out,
        extra=extra,
    )
    upper_points, upper_obj = _profile_direction(
        experiment,
        data,
        parameter_name,
        theta_hat,
        objective_hat,
        direction=+1,
        bound=ub,
        step0=initial_step,
        threshold=threshold,
        max_steps=max_steps,
        target=target_delta_loglik,
        other_init=dict(initial_estimate.parameters),
        warnings_out=warnings_out,
        extra=extra,
    )
    return _assemble(
        parameter_name,
        theta_hat,
        objective_hat,
        (lower_points, lower_obj),
        (upper_points, upper_obj),
        lb,
        ub,
        confidence_level,
        threshold,
        threshold_offset,
        warnings_out,
    )


def _assemble(
    parameter_name: str,
    theta_hat: float,
    objective_hat: float,
    lower: tuple[list[float], list[float]],
    upper: tuple[list[float], list[float]],
    lb: float,
    ub: float,
    confidence_level: float,
    threshold: float,
    threshold_offset: float,
    warnings_out: list[str],
) -> ProfileLikelihoodResult:
    """Stitch two profile arms into a result: interval, shape and warnings."""
    lower_points, lower_obj = lower
    upper_points, upper_obj = upper
    # Stitch arms together (lower arm reversed so theta ascends).
    theta_vals = np.concatenate(
        [np.asarray(lower_points[::-1]), [theta_hat], np.asarray(upper_points)]
    )
    obj_vals = np.concatenate([np.asarray(lower_obj[::-1]), [objective_hat], np.asarray(upper_obj)])

    ci_lower = _interp_crossing(lower_points, lower_obj, threshold, theta_hat, objective_hat)
    ci_upper = _interp_crossing(upper_points, upper_obj, threshold, theta_hat, objective_hat)

    crossed_lower = _crossed(lower_obj, threshold)
    crossed_upper = _crossed(upper_obj, threshold)
    hit_lb = bool(lower_points) and abs(lower_points[-1] - lb) < 1e-12
    hit_ub = bool(upper_points) and abs(upper_points[-1] - ub) < 1e-12

    max_delta = float(np.max(obj_vals - objective_hat)) if obj_vals.size else 0.0
    if max_delta < 0.1 * threshold_offset:
        shape: ProfileShape = "flat"
        if not crossed_lower and not crossed_upper:
            warnings_out.append(
                "profile is flat: |D(theta) - D(theta_hat)| stays below "
                f"{0.1 * threshold_offset:.3g}; parameter is not "
                "practically identifiable"
            )
    elif crossed_lower and crossed_upper:
        shape = "bounded"
    elif crossed_upper and not crossed_lower:
        shape = "one_sided_upper"
        if hit_lb:
            warnings_out.append(
                f"lower arm hit parameter bound lb={lb:.6g} without crossing the threshold"
            )
        else:
            warnings_out.append("lower arm exhausted max_steps without crossing the threshold")
    elif crossed_lower and not crossed_upper:
        shape = "one_sided_lower"
        if hit_ub:
            warnings_out.append(
                f"upper arm hit parameter bound ub={ub:.6g} without crossing the threshold"
            )
        else:
            warnings_out.append("upper arm exhausted max_steps without crossing the threshold")
    else:
        shape = "flat"
        warnings_out.append(
            "neither arm crossed the threshold; parameter may be practically non-identifiable"
        )

    for arm_name, arm_obj in (("lower", lower_obj), ("upper", upper_obj)):
        if _non_monotone(arm_obj):
            warnings_out.append(
                f"{arm_name} arm is non-monotone; consider multi-start on the full problem"
            )

    return ProfileLikelihoodResult(
        parameter=parameter_name,
        theta_values=theta_vals,
        neg_log_lik=obj_vals,
        theta_hat=theta_hat,
        objective_hat=objective_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        threshold=threshold,
        shape=shape,
        warnings=warnings_out,
    )


def profile_all(
    experiment: Experiment,
    data: dict,
    *,
    confidence_level: float = 0.95,
    initial_estimate: EstimationResult | None = None,
    quiet: bool = True,
    **kwargs,
) -> dict[str, ProfileLikelihoodResult]:
    """Run :func:`profile_likelihood` for every unknown parameter."""
    with quiet_solver(quiet):
        if initial_estimate is None:
            design = kwargs.get("design")
            initial_estimate = estimate_parameters(
                experiment, data, **({"design": dict(design)} if design else {})
            )
        return {
            name: profile_likelihood(
                experiment,
                data,
                name,
                confidence_level=confidence_level,
                initial_estimate=initial_estimate,
                quiet=quiet,
                **kwargs,
            )
            for name in initial_estimate.parameter_names
        }


def _combination(
    names: list[str],
    expression: str | None,
    function: Callable[[Mapping[str, float]], float] | None,
    name: str | None,
) -> tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray] | None]:
    """``(label, g(theta_vec), grad_g(theta_vec) or None)`` for the combination."""
    if expression is not None:
        import sympy

        from discopt.doe.symbolic import parse_expression

        expr = parse_expression(expression, names)
        # Use the parser's own symbols: it creates them with assumptions
        # (real=True), and a plain Symbol of the same name would differentiate
        # to zero.
        by_name = {sym.name: sym for sym in expr.free_symbols}
        symbols = [by_name.get(n, sympy.Symbol(n, real=True)) for n in names]
        g_fn = sympy.lambdify(symbols, expr, "numpy")
        dg = sympy.lambdify(symbols, [sympy.diff(expr, s) for s in symbols], "numpy")

        def g(v: np.ndarray) -> float:
            return float(g_fn(*v))

        def grad(v: np.ndarray) -> np.ndarray:
            return np.asarray(dg(*v), dtype=float)

        return (name or expression, g, grad)
    assert function is not None

    def g_call(v: np.ndarray) -> float:
        return float(function(dict(zip(names, map(float, v)))))

    return (name or getattr(function, "__name__", "g"), g_call, None)


def _profile_combination(
    experiment: Experiment,
    data: dict,
    estimate: EstimationResult,
    *,
    expression: str | None,
    function: Callable[[Mapping[str, float]], float] | None,
    name: str | None,
    confidence_level: float,
    max_steps: int,
    target: float,
    initial_step: float | None,
    design: Mapping[str, float] | None,
) -> ProfileLikelihoodResult:
    """Profile ``g(theta)`` by minimizing the deviance subject to ``g(theta) = c``."""
    from scipy.optimize import approx_fprime, minimize

    names = list(estimate.parameter_names)
    label, g, grad_g = _combination(names, expression, function, name)
    theta_hat_d = {n: float(estimate.parameters[n]) for n in names}
    dev = DevianceFunction(experiment, data, theta_hat_d, design=design)
    theta_hat = dev.vector(theta_hat_d)
    objective_hat = float(estimate.objective)
    c_hat = g(theta_hat)
    threshold_offset = float(chi2.ppf(confidence_level, df=1))
    threshold = objective_hat + threshold_offset

    # Optimize in log space for parameters bounded away from zero (rate
    # constants, equilibrium constants): along a ridge they move over decades,
    # and a linear scaling leaves SLSQP badly conditioned. Others are scaled
    # by their estimate.
    bnds = parameter_bounds(experiment, theta_hat_d)
    lo = np.array([bnds.get(n, (-np.inf, np.inf))[0] for n in names], dtype=float)
    hi = np.array([bnds.get(n, (-np.inf, np.inf))[1] for n in names], dtype=float)
    logv = np.isfinite(lo) & (lo > 0)
    scale = np.maximum(np.abs(theta_hat), 1e-12)

    def to_theta(u: np.ndarray) -> np.ndarray:
        return np.where(logv, np.exp(np.where(logv, u, 0.0)), u * scale)

    def dtheta(u: np.ndarray) -> np.ndarray:
        return np.where(logv, to_theta(u), scale)

    def to_u(theta: np.ndarray) -> np.ndarray:
        return np.where(logv, np.log(np.where(logv, np.maximum(theta, 1e-300), 1.0)), theta / scale)

    box = []
    for i in range(len(names)):
        if logv[i]:
            box.append((np.log(lo[i]), np.log(hi[i]) if np.isfinite(hi[i]) else None))
        else:
            box.append(
                (
                    lo[i] / scale[i] if np.isfinite(lo[i]) else None,
                    hi[i] / scale[i] if np.isfinite(hi[i]) else None,
                )
            )

    def grad_c(v: np.ndarray) -> np.ndarray:
        if grad_g is not None:
            return grad_g(v)
        h = 1e-6 * np.maximum(np.abs(v), 1e-8)
        return approx_fprime(v, g, h)

    if initial_step is None:
        cov = np.asarray(getattr(estimate, "covariance", None), dtype=float)
        dg = grad_c(theta_hat)
        var_g = float(dg @ cov @ dg) if cov.shape == (len(names), len(names)) else np.nan
        if np.isfinite(var_g) and var_g > 0:
            # D(c) ~ D_hat + (c - c_hat)^2 / var_g: aim for a step of ~target.
            initial_step = 0.5 * float(np.sqrt(target * var_g))
        else:
            initial_step = 1e-2 * max(abs(c_hat), 1.0)

    def f(u: np.ndarray) -> float:
        val = dev(to_theta(u))
        return val if np.isfinite(val) else 1e300

    def df(u: np.ndarray) -> np.ndarray:
        th = to_theta(u)
        gr = dev.gradient(th)
        if gr is None:
            h = 1e-7 * np.maximum(np.abs(u), 1e-6)
            return approx_fprime(u, f, h)
        return np.asarray(gr) * dtheta(u)

    def attempt(c: float, u0: np.ndarray) -> tuple[np.ndarray, float] | None:
        cons = {
            "type": "eq",
            "fun": lambda u: g(to_theta(u)) - c,
            "jac": lambda u: grad_c(to_theta(u)) * dtheta(u),
        }
        res = minimize(
            f,
            u0,
            jac=df,
            method="SLSQP",
            bounds=box,
            constraints=[cons],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        u = np.asarray(res.x, dtype=float)
        if (
            not res.success
            or not np.isfinite(res.fun)
            or abs(g(to_theta(u)) - c) > 1e-6 * max(abs(c), 1.0)
        ):
            return None
        return u, float(res.fun)

    u_hat = to_u(theta_hat)

    def solve(c: float, u0: np.ndarray) -> tuple[np.ndarray, float] | None:
        return attempt(c, u0) or attempt(c, u_hat)

    warnings_out: list[str] = []

    def arm(direction: int) -> tuple[list[float], list[float]]:
        cs: list[float] = []
        objs: list[float] = []
        c_cur, obj_cur, y = c_hat, objective_hat, u_hat
        step = float(initial_step)
        for _ in range(max_steps):
            c_new = c_cur + direction * step
            out = solve(c_new, y)
            if out is None:
                step /= 2.0
                if step < 1e-10 * max(abs(c_hat), 1.0):
                    warnings_out.append(
                        f"{'upper' if direction > 0 else 'lower'} arm stopped near "
                        f"{c_new:.6g}: g = c could not be met within the parameter bounds"
                    )
                    break
                continue
            y, obj_new = out
            cs.append(c_new)
            objs.append(obj_new)
            delta = obj_new - obj_cur
            c_cur, obj_cur = c_new, obj_new
            if obj_new > threshold:
                break
            factor = target / delta if delta > 0 else 10.0
            step *= max(0.1, min(10.0, factor))
        return cs, objs

    lower = arm(-1)
    upper = arm(+1)
    return _assemble(
        label,
        c_hat,
        objective_hat,
        lower,
        upper,
        -np.inf,
        np.inf,
        confidence_level,
        threshold,
        threshold_offset,
        warnings_out,
    )


def _profile_direction(
    experiment: Experiment,
    data: dict,
    name: str,
    theta_hat: float,
    objective_hat: float,
    *,
    direction: int,
    bound: float,
    step0: float,
    threshold: float,
    max_steps: int,
    target: float,
    other_init: dict[str, float],
    warnings_out: list[str],
    extra: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    """Walk out in one direction, re-solving at each step.

    Returns (theta_values, objective_values) in step order (starting
    from the point *after* ``theta_hat``).
    """
    theta_vals: list[float] = []
    obj_vals: list[float] = []
    current_theta = theta_hat
    current_obj = objective_hat
    step = step0
    warm = dict(other_init)

    for _ in range(max_steps):
        proposal = current_theta + direction * step
        # Clip to bound.
        if direction > 0 and proposal >= bound:
            proposal = bound
        if direction < 0 and proposal <= bound:
            proposal = bound

        try:
            warm_no_fixed = {k: v for k, v in warm.items() if k != name}
            res = estimate_parameters(
                experiment,
                data,
                initial_guess=warm_no_fixed,
                fixed_parameters={name: proposal},
                **(extra or {}),
            )
            new_obj = float(res.objective)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            # Treat solver-layer failures (ipopt convergence issues,
            # invalid bounds, numerical breakdown) as retryable with a
            # halved step. Programming errors (KeyError, TypeError) are
            # allowed to propagate so bugs surface immediately.
            step = step / 2.0
            if step < 1e-12 * max(abs(theta_hat), 1.0):
                warnings_out.append(
                    f"profile direction {direction:+d} abandoned near "
                    f"theta={proposal:.6g} after {type(exc).__name__}: {exc}"
                )
                break
            continue

        theta_vals.append(proposal)
        obj_vals.append(new_obj)
        warm = dict(res.parameters)

        delta = new_obj - current_obj
        current_theta = proposal
        current_obj = new_obj

        # Stopping conditions.
        if new_obj > threshold:
            break
        if abs(proposal - bound) < 1e-12:
            break

        # Adaptive step (Raue 2009 Sec 2.3): step *= target / delta, clipped.
        if delta > 0:
            factor = target / delta
            step = step * max(0.1, min(10.0, factor))
        else:
            step = step * 10.0
        # Safety ceiling: never step further than the remaining distance
        # to the bound, and never below a tiny fraction of the estimate.
        if np.isfinite(bound):
            step = min(step, abs(bound - current_theta))
        step = max(step, 1e-12 * max(abs(theta_hat), 1.0))

    return theta_vals, obj_vals


def _interp_crossing(
    theta_vals: list[float],
    obj_vals: list[float],
    threshold: float,
    theta_hat: float,
    objective_hat: float,
) -> float | None:
    """Linearly interpolate the theta at which the objective crosses the threshold.

    The arm's "previous" point before its first entry is ``(theta_hat,
    objective_hat)``; beyond that, consecutive arm points are paired.
    Returns ``None`` if no crossing occurred.
    """
    if not theta_vals or not obj_vals:
        return None

    def lerp(t0: float, t1: float, o0: float, o1: float) -> float:
        if o1 == o0:
            return float(t1)
        frac = (threshold - o0) / (o1 - o0)
        return float(t0 + frac * (t1 - t0))

    prev_t = theta_hat
    prev_o = objective_hat
    for t, o in zip(theta_vals, obj_vals):
        if o >= threshold:
            return lerp(prev_t, t, prev_o, o)
        prev_t, prev_o = t, o
    return None


def _crossed(obj_vals: list[float], threshold: float) -> bool:
    return bool(obj_vals) and max(obj_vals) >= threshold


def _non_monotone(obj_vals: list[float]) -> bool:
    """Detect a non-monotone arm (later points lower than earlier ones by a
    meaningful margin).

    We tolerate small numerical oscillations: a drop of more than
    ``1e-3 * max(obj)`` relative to a previous point counts.
    """
    if len(obj_vals) < 2:
        return False
    arr = np.asarray(obj_vals, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drops = running_max - arr
    scale = max(arr.max(), 1.0)
    return bool(np.any(drops > 1e-3 * scale))
