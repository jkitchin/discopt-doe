"""Parameter-estimation dispatch for the discopt-doe loops.

:func:`discopt.estimate.estimate_parameters` builds and solves a discopt NLP,
then computes the covariance by compiling the response expressions. An
experiment whose responses come from an opaque JAX callable (a
:func:`discopt.modeling.custom` node, as in :mod:`discopt.doe.dynamic`) solves
fine but cannot go through that covariance compiler. Such experiments carry
their own ``estimate`` method, and the loops that fit models repeatedly
(:func:`~discopt.doe.profile_likelihood`, :func:`~discopt.doe.sequential_doe`,
:func:`~discopt.doe.sequential_discrimination`) call this dispatcher instead
of the base function.

The dispatcher also adds two things the base estimator lacks:

* ``n_starts`` for multi-start fitting. A least-squares problem can have more
  than one optimum with the same fit (a mirror solution, e.g. two exponentials
  whose roles can be swapped). The dispatcher fits from ``n_starts`` starting
  points, returns the best, and warns when distinct parameter sets fit equally
  well: the parameters are then not globally identifiable from the data.
* ``quiet``, which hides the base solver's per-solve log messages.

:class:`DevianceFunction` evaluates the fitting objective at arbitrary
parameters without a solve, for the profile of a parameter combination.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from discopt.doe._logging import quiet_solver

_FIT_ERRORS = (RuntimeError, ValueError, ArithmeticError, np.linalg.LinAlgError)


def estimate_parameters(
    experiment: Any,
    data: Any,
    *,
    n_starts: int = 1,
    seed: int | None = 0,
    quiet: bool = True,
    tie_tolerance: float = 1e-3,
    **kwargs: Any,
):
    """Fit ``experiment`` to ``data``: a drop-in for the base estimator.

    Calls ``experiment.estimate(data, **kwargs)`` when the experiment defines
    it, else :func:`discopt.estimate.estimate_parameters`. Takes and returns
    exactly what the base function does, plus the options below.

    Parameters
    ----------
    n_starts : int, default 1
        Number of starting points. The first is ``initial_guess`` (or the
        experiment's default); the rest are drawn at random inside each free
        parameter's bounds (log-uniformly when a positive range spans more than
        two decades). The best fit is returned.
    seed : int, optional
        Seed for the random starting points.
    quiet : bool, default True
        Hide the base solver's per-solve log messages.
    tie_tolerance : float, default 1e-3
        Two optima whose deviances differ by less than
        ``max(tie_tolerance, 1e-6 * deviance)`` fit equally well. If their
        parameters differ by more than 1 %, a :class:`UserWarning` is issued and
        the alternatives are attached to the result as
        ``result.alternative_optima`` (a list of parameter dicts). A single
        optimum gives an empty list.

    Other keyword arguments (``initial_guess``, ``fixed_parameters``,
    ``solver_options``, ``design`` for experiments that take it) are passed
    through unchanged.
    """
    n_starts = max(1, int(n_starts))
    with quiet_solver(quiet):
        if n_starts == 1:
            result = _fit_once(experiment, data, kwargs)
            _attach(result, [])
            return result
        return _multistart(experiment, data, n_starts, seed, tie_tolerance, kwargs)


def _fit_once(experiment: Any, data: Any, kwargs: Mapping[str, Any]):
    own = getattr(experiment, "estimate", None)
    if callable(own):
        return own(data, **kwargs)
    from discopt.estimate import estimate_parameters as _base

    return _base(experiment, data, **kwargs)


def _attach(result: Any, alternatives: list[dict[str, float]]) -> None:
    try:
        result.alternative_optima = alternatives
    except AttributeError:  # pragma: no cover - a frozen result type
        pass


def parameter_bounds(
    experiment: Any, guess: Mapping[str, float] | None = None
) -> dict[str, tuple[float, float]]:
    """``{name: (lb, ub)}`` for every unknown parameter of ``experiment``."""
    specs = getattr(experiment, "parameter_specs", None)
    if isinstance(specs, Mapping) and specs:
        out = {}
        for name, spec in specs.items():
            if isinstance(spec, (tuple, list)) and len(spec) >= 3:
                out[name] = (float(spec[1]), float(spec[2]))
            else:
                out[name] = (-math.inf, math.inf)
        return out
    em = experiment.create_model(**dict(guess or {}))
    out = {}
    for name, var in em.unknown_parameters.items():
        lb = float(np.asarray(var.lb, dtype=float).ravel()[0])
        ub = float(np.asarray(var.ub, dtype=float).ravel()[0])
        out[name] = (lb, ub)
    return out


def _random_start(
    bounds: Mapping[str, tuple[float, float]],
    guess: Mapping[str, float],
    names: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    start = {}
    for name in names:
        lb, ub = bounds.get(name, (-math.inf, math.inf))
        g = float(guess.get(name, 1.0))
        if math.isfinite(lb) and math.isfinite(ub):
            if lb > 0 and ub / lb > 100.0:
                start[name] = float(math.exp(rng.uniform(math.log(lb), math.log(ub))))
            else:
                start[name] = float(rng.uniform(lb, ub))
        else:
            scale = abs(g) if g != 0 else 1.0
            val = g + scale * float(rng.normal())
            start[name] = float(min(max(val, lb), ub))
    return start


def _multistart(
    experiment: Any,
    data: Any,
    n_starts: int,
    seed: int | None,
    tie_tolerance: float,
    kwargs: dict[str, Any],
):
    """Fit from ``n_starts`` points and report every equally good optimum.

    The base estimator builds its model inside the call and cannot be seeded
    from outside, so the extra starts minimize the compiled deviance
    (:class:`DevianceFunction`) directly with bounded L-BFGS-B. The best
    optimum is returned as an ordinary estimation result: re-solved through
    the experiment's own estimator when it accepts a starting point, else
    assembled at the optimum from the FIM.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    guess = dict(kwargs.get("initial_guess") or {})
    fixed = {k: float(v) for k, v in (kwargs.get("fixed_parameters") or {}).items()}
    design = kwargs.get("design")

    first = _fit_once(experiment, data, kwargs)
    theta_hat = {k: float(v) for k, v in first.parameters.items()}
    bounds = parameter_bounds(experiment, {**guess, **fixed})
    free = [n for n in theta_hat if n not in fixed]
    dev = DevianceFunction(experiment, data, theta_hat, design=design)
    pos = {n: i for i, n in enumerate(dev.names)}
    base_vec = dev.vector(theta_hat)
    for k, v in fixed.items():
        base_vec[pos[k]] = v
    free_idx = [pos[n] for n in free]

    def full(z: np.ndarray) -> np.ndarray:
        v = base_vec.copy()
        v[free_idx] = z
        return v

    def f(z: np.ndarray) -> float:
        val = dev(full(z))
        return val if np.isfinite(val) else 1e300

    grad = None
    if dev.gradient(base_vec) is not None:

        def grad(z: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            g = dev.gradient(full(z))
            return np.zeros(len(free_idx)) if g is None else np.asarray(g)[free_idx]

    box = [
        (
            None if not math.isfinite(bounds[n][0]) else bounds[n][0],
            None if not math.isfinite(bounds[n][1]) else bounds[n][1],
        )
        for n in free
    ]
    optima: list[tuple[float, dict[str, float]]] = [(float(first.objective), theta_hat)]
    for _ in range(n_starts - 1):
        start = _random_start(bounds, guess, free, rng)
        z0 = np.asarray([start[n] for n in free], dtype=float)
        try:
            res = minimize(f, z0, jac=grad, method="L-BFGS-B", bounds=box)
        except _FIT_ERRORS:
            continue
        if not np.isfinite(res.fun) or res.fun >= 1e299:
            continue
        theta = {n: float(v) for n, v in zip(dev.names, full(res.x))}
        optima.append((float(res.fun), theta))

    optima.sort(key=lambda t: t[0])
    best_obj, best_theta = optima[0]
    tol = max(float(tie_tolerance), 1e-6 * abs(best_obj))
    seen = [best_theta]
    for obj, theta in optima[1:]:
        if obj - best_obj > tol:
            break
        if all(_distinct(theta, s) for s in seen):
            seen.append(theta)

    if float(first.objective) - best_obj > tol and _distinct(best_theta, theta_hat):
        best = _result_at(experiment, data, best_theta, best_obj, kwargs)
    else:
        best = first
        best_theta = theta_hat
    alternatives = [t for t in seen if _distinct(t, best_theta)]
    _attach(best, alternatives)
    if alternatives:
        shown = "; ".join(
            "(" + ", ".join(f"{k}={v:.4g}" for k, v in p.items()) + ")"
            for p in [best_theta, *alternatives[:3]]
        )
        warnings.warn(
            f"{1 + len(alternatives)} distinct parameter sets fit these data equally well "
            f"(deviance {float(best.objective):.6g}): {shown}. The parameters are not "
            "globally identifiable from these data; the returned estimate is one of them. "
            "See result.alternative_optima.",
            UserWarning,
            stacklevel=3,
        )
    return best


def _result_at(
    experiment: Any,
    data: Any,
    theta: Mapping[str, float],
    objective: float,
    kwargs: Mapping[str, Any],
):
    """An estimation result at ``theta``, found by a start the solver could not take."""
    own = getattr(experiment, "estimate", None)
    if callable(own):
        call = dict(kwargs)
        call["initial_guess"] = dict(theta)
        return own(data, **call)
    from discopt.estimate import EstimationResult

    from discopt.doe.fim import compute_fim

    fr = compute_fim(experiment, dict(theta), kwargs.get("design"))
    J = np.asarray(fr.jacobian, dtype=float)
    em = experiment.create_model(**dict(theta))
    names = list(fr.parameter_names)
    reps, sig = [], []
    for rn in fr.response_names:
        reps.append(float(np.atleast_1d(np.asarray(data[rn])).size) if rn in data else 0.0)
        sig.append(float(em.measurement_error[rn]))
    w = np.asarray(reps) / np.asarray(sig) ** 2
    fim = J.T @ (w[:, None] * J)
    try:
        cov = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fim)
    return EstimationResult(
        parameters={n: float(theta[n]) for n in names},
        covariance=cov,
        fim=fim,
        objective=float(objective),
        solve_result=None,
        parameter_names=names,
        n_observations=int(sum(reps)),
    )


def _distinct(a: Mapping[str, float], b: Mapping[str, float], rtol: float = 1e-2) -> bool:
    for k, va in a.items():
        vb = float(b.get(k, va))
        if abs(float(va) - vb) > rtol * max(abs(float(va)), abs(vb), 1e-12):
            return True
    return False


# --------------------------------------------------------------------------
# Deviance at arbitrary parameters, without a solve
# --------------------------------------------------------------------------


class DevianceFunction:
    """``D(theta) = sum ((y - yhat(theta)) / sigma)^2`` for fixed data.

    The same objective :func:`discopt.estimate.estimate_parameters` minimizes,
    evaluated at any parameter values without solving. It is built once and
    reused, which is what makes a constrained re-optimization (the profile of a
    parameter combination) affordable.

    Three evaluation paths, fastest first:

    1. an experiment with a ``predict(theta, design)`` method (e.g.
       :class:`~discopt.doe.ODEExperiment`);
    2. a pure explicit response model: the responses are compiled once
       with JAX, and the deviance and its gradient are JIT-compiled;
    3. anything else: a fit with every parameter fixed, which evaluates the
       objective through the base estimator (slower, but always available).
    """

    def __init__(
        self,
        experiment: Any,
        data: Mapping[str, Any],
        theta0: Mapping[str, float],
        *,
        design: Mapping[str, float] | None = None,
    ) -> None:
        self.experiment = experiment
        self.data = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in data.items()}
        self.design = dict(design) if design else None
        self.names = list(theta0)
        self.theta0 = {k: float(v) for k, v in theta0.items()}
        self._grad: Callable[[np.ndarray], np.ndarray] | None = None
        self._value: Callable[[np.ndarray], float]
        self.path = ""
        predict = getattr(experiment, "predict", None)
        if callable(predict) and hasattr(experiment, "response_names"):
            self._setup_predict(predict)
        elif not self._setup_compiled():
            self._setup_fallback()

    # -- evaluation paths ---------------------------------------------------

    def _setup_predict(self, predict: Callable) -> None:
        sig = self._sigmas_from_experiment()

        def value(theta: np.ndarray) -> float:
            y = predict(dict(zip(self.names, map(float, theta))), self.design)
            return float(
                sum(np.sum(((obs - float(y[k])) / sig[k]) ** 2) for k, obs in self.data.items())
            )

        self._value = value
        self.path = "predict"
        self._setup_predict_gradient(predict, sig)

    def _setup_predict_gradient(self, predict: Callable, sig: Mapping[str, float]) -> None:
        """Analytic ``dD/dtheta`` from the experiment's sensitivity matrix, if it has one.

        ``jacobian(theta, design)`` has one row per response name and one column
        per parameter in the experiment's own order, so the columns are mapped
        onto :attr:`names` here. Without it the caller falls back to finite
        differences, which is correct but much slower.
        """
        jacobian = getattr(self.experiment, "jacobian", None)
        rows = list(getattr(self.experiment, "response_names", []))
        cols = list(getattr(self.experiment, "parameter_names", []))
        if not callable(jacobian) or set(self.names) - set(cols) or not rows:
            return
        take = [cols.index(n) for n in self.names]
        keys = [k for k in self.data if k in rows]
        if len(keys) != len(self.data):  # a response the Jacobian does not cover
            return
        ridx = [rows.index(k) for k in keys]

        def gradient(theta: np.ndarray) -> np.ndarray:
            th = dict(zip(self.names, map(float, theta)))
            y = predict(th, self.design)
            J = np.asarray(jacobian(th, self.design), dtype=float)[np.ix_(ridx, take)]
            resid = np.array(
                [np.sum(self.data[k] - float(y[k])) / sig[k] ** 2 for k in keys], dtype=float
            )
            return -2.0 * (resid @ J)

        self._grad = gradient

    def _sigmas_from_experiment(self) -> dict[str, float]:
        """Measurement ``sigma`` for every response in the data.

        The deviance is compared against a chi-square threshold, so its scale
        decides every interval built from it: a sigma that is wrong by a factor
        of 100 makes the deviance wrong by 10,000 and the profile look flat.
        Experiments spell the errors differently -- a ``measurement_error``
        mapping or scalar, or (dynamic experiments) a ``sigma`` mapping keyed by
        the measured state behind a ``state@time`` response name -- so all of
        those are tried before falling back to 1.0, which is announced.
        """
        for source in (
            getattr(self.experiment, "measurement_error", None),
            getattr(self.experiment, "sigma", None),
        ):
            if source is None:
                continue
            if not isinstance(source, Mapping):
                return {k: float(source) for k in self.data}
            try:
                return {
                    k: float(source[k] if k in source else source[k.split("@")[0]])
                    for k in self.data
                }
            except (KeyError, TypeError, ValueError):
                continue
        warnings.warn(
            f"{type(self.experiment).__name__} reports no measurement error; using sigma = 1 "
            "for every response. Deviance-based intervals (profile likelihood) are only "
            "meaningful if that is the true measurement scale.",
            UserWarning,
            stacklevel=3,
        )
        return {k: 1.0 for k in self.data}

    def _setup_compiled(self) -> bool:
        try:
            import jax
            import jax.numpy as jnp
            from discopt.parametric import flatten_params, variable_slices

            from discopt.doe.fim import (
                _assemble_x_flat_direct,
                _compile_response,
                _design_source_map,
            )
        except ImportError:  # pragma: no cover - no jax
            return False
        try:
            em = self.experiment.create_model(**self.theta0)
            if _design_source_map(em) is None:
                return False
            x0 = _assemble_x_flat_direct(em, self.theta0, self.design)
            if x0 is None:
                return False
            slices = variable_slices(em.model)
            idx = []
            for name in self.names:
                var = em.unknown_parameters[name]
                sl = slices[var.name]
                if sl.stop - sl.start != 1:
                    return False
                idx.append(sl.start)
            keys = [k for k in em.response_names if k in self.data]
            fns = [_compile_response(em.responses[k], em.model) for k in keys]
            p_flat = flatten_params(em.model)
            sig = [float(em.measurement_error[k]) for k in keys]
            obs = [jnp.asarray(self.data[k]) for k in keys]
            x_base = jnp.asarray(x0)
            index = jnp.asarray(idx)

            def dev(theta):
                x = x_base.at[index].set(theta)
                total = 0.0
                for fn, s, o in zip(fns, sig, obs):
                    yhat = jnp.reshape(fn(x, p_flat), ())
                    total = total + jnp.sum(((o - yhat) / s) ** 2)
                return total

            value_jit = jax.jit(dev)
            grad_jit = jax.jit(jax.grad(dev))
            float(value_jit(jnp.asarray([self.theta0[n] for n in self.names])))
        except Exception:  # noqa: BLE001 - any compile problem: use the fallback
            return False
        self._value = lambda theta: float(value_jit(jnp.asarray(theta, dtype=jnp.float64)))
        self._grad = lambda theta: np.asarray(
            grad_jit(jnp.asarray(theta, dtype=jnp.float64)), dtype=float
        )
        self.path = "compiled"
        return True

    def _setup_fallback(self) -> None:
        def value(theta: np.ndarray) -> float:
            fixed = dict(zip(self.names, map(float, theta)))
            kw: dict[str, Any] = {"initial_guess": fixed, "fixed_parameters": fixed}
            if self.design is not None:
                kw["design"] = self.design
            with quiet_solver():
                return float(_fit_once(self.experiment, self.data_for_fit, kw).objective)

        self._value = value
        self.path = "fixed-fit"

    @property
    def data_for_fit(self) -> dict[str, Any]:
        return {k: (v[0] if v.size == 1 else v) for k, v in self.data.items()}

    # -- public interface -------------------------------------------------

    def vector(self, theta: Mapping[str, float]) -> np.ndarray:
        """Parameter dict to the ordered vector this function takes."""
        return np.asarray([float(theta[n]) for n in self.names], dtype=float)

    def __call__(self, theta: np.ndarray | Mapping[str, float]) -> float:
        if isinstance(theta, Mapping):
            theta = self.vector(theta)
        return self._value(np.asarray(theta, dtype=float))

    def gradient(self, theta: np.ndarray) -> np.ndarray | None:
        """Exact gradient on the compiled path, ``None`` otherwise."""
        return None if self._grad is None else self._grad(np.asarray(theta, dtype=float))


__all__ = ["DevianceFunction", "estimate_parameters", "parameter_bounds"]
