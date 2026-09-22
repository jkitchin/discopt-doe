"""Designs that stay good when the parameter guess is wrong.

A model-based design for a nonlinear model is only *locally* optimal: it is
built for a nominal parameter value, and it can lose most of its efficiency
when the truth is elsewhere (Chapter 17 of the book shows a first-order decay
design at ``t = 1/k`` falling to 4 % efficiency when ``k`` is off by four).
Robust criteria average or guard against that loss over a set of plausible
parameter values, a sample from a prior or a grid over a region:

- **pseudo-Bayesian** (``robust="expected"``): maximize the average
  ``log det FIM(theta_s, design)`` over the samples (ED-optimality;
  Chaloner & Verdinelli 1995, Pronzato & Walter 1985);
- **max-min** (``robust="maximin"``): maximize the *worst* D-efficiency over
  the samples, each relative to that sample's own locally optimal design.

The information for each sample is computed with a response Jacobian
compiled once per sample, so a design evaluation costs one Jacobian call per
sample, not a model rebuild.

References
----------
Chaloner, K. & Verdinelli, I. Bayesian experimental design: a review.
*Statist. Sci.* 10, 273-304 (1995).

Pronzato, L. & Walter, E. Robust experiment design via stochastic
approximation. *Math. Biosci.* 75, 103-120 (1985).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from discopt.doe.simplex import DesignConstraint
from discopt.estimate import Experiment

_SINGULAR = -1e30


@dataclass
class RobustDesignResult:
    """Result of :func:`robust_optimal_experiment`.

    Attributes
    ----------
    designs : list[dict]
        The ``n_experiments`` design points.
    robust : str
        ``"expected"`` or ``"maximin"``.
    criterion_value : float
        Mean ``log det`` (expected) or minimum D-efficiency (maximin).
    log_dets : numpy.ndarray
        ``log det FIM`` of the design at each parameter sample.
    efficiencies : numpy.ndarray or None
        D-efficiency at each sample, relative to that sample's locally
        optimal design (computed for ``maximin``, or when ``reference_log_dets``
        is given).
    reference_log_dets : numpy.ndarray or None
        ``log det`` of each sample's locally optimal design.
    parameter_samples : list[dict]
    n_parameters : int
    """

    designs: list[dict[str, float]]
    robust: str
    criterion_value: float
    log_dets: np.ndarray
    efficiencies: np.ndarray | None
    reference_log_dets: np.ndarray | None
    parameter_samples: list[dict[str, float]]
    n_parameters: int

    def summary(self) -> str:
        lines = [f"robust design ({self.robust}), {len(self.designs)} run(s)"]
        for i, d in enumerate(self.designs):
            lines.append(f"  run {i + 1}: " + ", ".join(f"{k}={v:.6g}" for k, v in d.items()))
        lines.append(
            f"  mean log det over {len(self.log_dets)} samples: {self.log_dets.mean():.4f}"
        )
        if self.efficiencies is not None:
            e = self.efficiencies
            lines.append(
                f"  D-efficiency: mean {e.mean():.3f}, min {e.min():.3f}, "
                f"5th percentile {np.percentile(e, 5):.3f}"
            )
        return "\n".join(lines)


def _as_samples(param_samples: Any) -> list[dict[str, float]]:
    if isinstance(param_samples, Mapping):
        keys = list(param_samples)
        cols = [np.atleast_1d(np.asarray(param_samples[k], dtype=float)) for k in keys]
        n = len(cols[0])
        if any(len(c) != n for c in cols):
            raise ValueError("every parameter needs the same number of samples")
        return [{k: float(c[i]) for k, c in zip(keys, cols)} for i in range(n)]
    out = [{k: float(v) for k, v in s.items()} for s in param_samples]
    if not out:
        raise ValueError("param_samples is empty")
    return out


def _fim_functions(
    experiment: Experiment, samples: Sequence[Mapping[str, float]]
) -> list[Callable[[dict[str, float] | None], np.ndarray]]:
    """One ``design -> FIM`` function per sample, compiled once where possible."""
    from discopt.doe.fim import compute_fim

    try:
        from discopt.doe.fim import _make_direct_fim_evaluator
    except ImportError:  # pragma: no cover - internal helper renamed
        _make_direct_fim_evaluator = None  # type: ignore[assignment]

    fns = []
    for theta in samples:
        ev = None
        if _make_direct_fim_evaluator is not None:
            try:
                ev = _make_direct_fim_evaluator(experiment, dict(theta))
            except Exception:  # noqa: BLE001 - fall back to the per-call path
                ev = None
        if ev is not None:
            fns.append(lambda d, _ev=ev: np.asarray(_ev(d).fim, dtype=float))
        else:
            fns.append(
                lambda d, _t=dict(theta): np.asarray(
                    compute_fim(experiment, _t, d).fim, dtype=float
                )
            )
    return fns


def _logdet(m: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(m)
    return float(val) if sign > 0 and np.isfinite(val) else _SINGULAR


_CRITERION_ALIASES = {
    "D": "D",
    "D_OPTIMAL": "D",
    "DETERMINANT": "D",
    "A": "A",
    "A_OPTIMAL": "A",
    "TRACE": "A",
    "E": "E",
    "E_OPTIMAL": "E",
    "MIN_EIGENVALUE": "E",
}


def _normalize_criterion(criterion: str) -> str:
    key = str(criterion).upper()
    if key not in _CRITERION_ALIASES:
        raise ValueError(
            f"criterion must be one of 'D', 'A', 'E' (got {criterion!r}); "
            "'D_OPTIMAL'/'determinant', 'A_OPTIMAL'/'trace' and "
            "'E_OPTIMAL'/'min_eigenvalue' are accepted too"
        )
    return _CRITERION_ALIASES[key]


def _score(fim: np.ndarray, criterion: str) -> float:
    """The criterion at one FIM, signed so that larger is always better.

    D is ``log det F``; A is ``-tr(F⁻¹)``, the negated total variance; E is
    ``lambda_min(F)``, the worst-determined direction. A singular FIM scores
    ``_SINGULAR`` in every one of them.
    """
    if criterion == "D":
        return _logdet(fim)
    eigenvalues = np.linalg.eigvalsh(np.asarray(fim, dtype=float))
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues.min() <= 0.0:
        return _SINGULAR
    if criterion == "E":
        return float(eigenvalues.min())
    return float(-np.sum(1.0 / eigenvalues))  # -tr(F^-1), from the same decomposition


def _efficiencies(scores: np.ndarray, reference: np.ndarray, criterion: str, p: int) -> np.ndarray:
    """Efficiency of a design at each sample, relative to that sample's best.

    Each is on the usual 0-1 scale, where 1 is the locally optimal design for
    that sample: ``(det F / det F*)^(1/p)`` for D, ``tr(F*⁻¹)/tr(F⁻¹)`` for A,
    and ``lambda_min(F)/lambda_min(F*)`` for E. Note the direction: A's score is
    a negated cost, so its ratio is the other way up from E's.
    """
    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(reference, dtype=float)
    singular = (scores <= _SINGULAR / 2) | (reference <= _SINGULAR / 2)
    if np.any(reference <= _SINGULAR / 2):
        warnings.warn(
            "at least one parameter sample has a singular information matrix even under its "
            "own best design: the model is not identifiable there from any design in these "
            "bounds, so its efficiency is reported as 0. Drop the sample, add runs, or fix "
            "the parameter it cannot see.",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        if criterion == "D":
            eff = np.exp((scores - reference) / p)
        elif criterion == "A":
            eff = reference / scores  # both are negated costs, so this is cost*/cost
        else:
            eff = scores / reference  # lambda_min / lambda_min*
    eff = np.where(singular, 0.0, eff)
    return np.clip(np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _violations(
    designs: Sequence[Mapping[str, float]],
    equality_constraints: Sequence[Any],
    inequality_constraints: Sequence[Any],
    tol: float = 1e-6,
) -> list[str]:
    """Constraint violations of a whole design, as readable strings."""
    out = []
    for i, d in enumerate(designs):
        for j, g in enumerate(equality_constraints):
            value = float(g(dict(d)))
            if abs(value) > tol:
                out.append(f"run {i + 1}, equality {j}: g = {value:.3g} (should be 0)")
        for j, h in enumerate(inequality_constraints):
            value = float(h(dict(d)))
            if value < -tol:
                out.append(f"run {i + 1}, inequality {j}: h = {value:.3g} (should be >= 0)")
    return out


def _maximin_epigraph(
    minimize,
    *,
    scores,
    efficiencies,
    designs_of,
    bounds,
    constraints,
    lbs,
    ubs,
    n_exp: int,
    n_samples: int,
    n_starts: int,
    rng,
    feasible,
    extra_starts: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Max-min in epigraph form: maximize ``t`` s.t. every efficiency >= ``t``.

    The optimization variable is ``[design..., t]``. This replaces the
    non-smooth ``min`` in the objective with one smooth constraint per sample,
    which is what lets SLSQP carry the user's own constraints at the same time.
    """
    cons = list(constraints) + [
        {"type": "ineq", "fun": lambda u, s=s: float(efficiencies(scores(u[:-1]))[s] - u[-1])}
        for s in range(n_samples)
    ]
    box = list(bounds) + [(0.0, None)]
    # SLSQP's default tolerance stops early on this problem: the objective is a
    # single variable t and the information is all in the constraints, so the
    # step test trips while t can still be pushed up.
    options = {"ftol": 1e-10, "maxiter": 400}

    def solve(z0: np.ndarray) -> np.ndarray | None:
        t0 = float(np.min(efficiencies(scores(z0))))
        res = minimize(
            lambda u: -float(u[-1]),
            np.append(z0, t0),
            method="SLSQP",
            bounds=box,
            constraints=cons,
            options=options,
        )
        z = np.clip(np.asarray(res.x)[:-1], np.tile(lbs, n_exp), np.tile(ubs, n_exp))
        return z if feasible(z) else None

    # The worst-case surface is multimodal, and random starts land in the same
    # basin over and over: on the book's reactor schedule, 4, 12 and 24 starts
    # all return the same design. The pseudo-Bayesian optimum is a much better
    # place to start from than any random point, so it is always tried.
    starts = [np.asarray(z) for z in extra_starts]
    starts += [
        np.concatenate([rng.uniform(lbs, ubs) for _ in range(n_exp)])
        for _ in range(max(1, n_starts))
    ]
    best_u, best_f = None, np.inf
    for z0 in starts:
        z = solve(z0)
        if z is None:
            continue
        value = -float(np.min(efficiencies(scores(z))))
        if value < best_f:
            best_f, best_u = value, z
    # Polish: restarting from the best point found lets SLSQP take the steps a
    # stalled run left on the table. Repeat while it keeps paying.
    for _ in range(5):
        if best_u is None:
            break
        z = solve(best_u)
        if z is None:
            break
        value = -float(np.min(efficiencies(scores(z))))
        if value >= best_f - 1e-9:
            break
        best_f, best_u = value, z
    if best_u is None:
        raise ValueError(
            "no feasible max-min design found: every start refined to a point violating "
            "the constraints. Check that they are satisfiable inside design_bounds."
        )
    return best_u


def robust_optimal_experiment(
    experiment: Experiment,
    param_samples: Sequence[Mapping[str, float]] | Mapping[str, Sequence[float]],
    design_bounds: Mapping[str, tuple[float, float]],
    *,
    criterion: str = "D",
    robust: str = "expected",
    n_experiments: int = 1,
    prior_fim: np.ndarray | None = None,
    reference_log_dets: Sequence[float] | None = None,
    reference_scores: Sequence[float] | None = None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    n_starts: int = 10,
    seed: int = 0,
) -> RobustDesignResult:
    """Design experiments that are good across a set of parameter values.

    Parameters
    ----------
    experiment : Experiment
    param_samples : sequence of dict, or dict of arrays
        Plausible parameter values: draws from a prior, or a grid over the
        region you believe the parameters lie in. Tens of samples usually
        suffice; the cost grows linearly with their number.
    design_bounds : mapping name -> (lb, ub)
        Bounds on every design input.
    criterion : {"D", "A", "E"}, default "D"
        What "good" means at each sample: ``"D"`` maximizes ``log det F``
        (the volume of the confidence ellipsoid), ``"A"`` minimizes
        ``tr(F⁻¹)`` (total parameter variance), ``"E"`` maximizes the smallest
        eigenvalue (the worst-determined direction).

        Prefer ``"D"`` unless you have a reason not to: its efficiency is
        invariant to how the parameters are scaled, which is what makes values
        from different samples commensurable. ``A`` and ``E`` are *not* scale
        invariant -- rescaling a parameter (seconds to hours, say) changes
        which design they pick -- so use them only when the parameters are
        already on comparable scales, or when total variance / the worst
        direction is genuinely the quantity you care about.
    robust : {"expected", "maximin"}, default "expected"
        Pseudo-Bayesian (maximize mean ``log det``) or max-min (maximize the
        worst D-efficiency).
    n_experiments : int, default 1
        Runs in the design; their information adds.
    prior_fim : numpy.ndarray, optional
        Information already collected (the same for every sample).
    reference_log_dets, reference_scores : sequence of float, optional
        The criterion at each sample's locally optimal design, which sets the
        efficiency scale for max-min. Computed when omitted (one local
        optimization per sample). ``reference_scores`` is the general name;
        ``reference_log_dets`` is the same thing for ``criterion="D"``, where
        the score *is* the log determinant. Give at most one.
    equality_constraints, inequality_constraints : sequence of callable, optional
        Constraints on each run, called with that run's design dict:
        ``g(design) == 0`` and ``h(design) >= 0``. They apply to every run
        separately, as in :func:`~discopt.doe.batch_optimal_experiment`, so a
        minimum spacing between sampling times (``lambda d: d["t2"] - d["t1"] -
        2.0``) constrains each run's own schedule. The search switches to SLSQP
        when any are given, and the result is checked for feasibility.
    n_starts : int, default 10
        Random starts of the design search.
    seed : int, default 0

    Returns
    -------
    RobustDesignResult

    Notes
    -----
    A max-min design guarantees its worst case only over the samples it was
    built on. With few samples it can over-fit them; check its efficiency on
    fresh draws (``design_efficiencies``) before relying on the guarantee.
    """
    from scipy.optimize import minimize

    crit = _normalize_criterion(criterion)
    if robust not in ("expected", "maximin"):
        raise ValueError(f"robust must be 'expected' or 'maximin', got {robust!r}")
    if reference_log_dets is not None and reference_scores is not None:
        raise ValueError("give reference_scores or reference_log_dets, not both")
    if reference_log_dets is not None:
        if crit != "D":
            raise ValueError(
                f"reference_log_dets is the D-optimal spelling; with criterion={criterion!r} "
                "pass reference_scores"
            )
        reference_scores = reference_log_dets
    equality_constraints = list(equality_constraints or ())
    inequality_constraints = list(inequality_constraints or ())
    constrained = bool(equality_constraints or inequality_constraints)
    samples = _as_samples(param_samples)
    names = list(design_bounds)
    lbs = np.array([float(design_bounds[n][0]) for n in names])
    ubs = np.array([float(design_bounds[n][1]) for n in names])
    n_exp = int(n_experiments)
    fims = _fim_functions(experiment, samples)
    p = fims[0]({n: float((lo + hi) / 2) for n, lo, hi in zip(names, lbs, ubs)}).shape[0]
    prior = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=float)

    def designs_of(z: np.ndarray) -> list[dict[str, float]]:
        z = np.clip(z.reshape(n_exp, len(names)), lbs, ubs)
        return [dict(zip(names, map(float, row))) for row in z]

    def total_fims(z: np.ndarray, which: Sequence[int] | None = None) -> list[np.ndarray]:
        ds = designs_of(z)
        idx = range(len(samples)) if which is None else which
        out = []
        for s in idx:
            total = prior.copy()
            for d in ds:
                total = total + fims[s](d)
            out.append(total)
        return out

    def scores(z: np.ndarray, which: Sequence[int] | None = None) -> np.ndarray:
        return np.asarray([_score(m, crit) for m in total_fims(z, which)])

    def log_dets(z: np.ndarray, which: Sequence[int] | None = None) -> np.ndarray:
        return np.asarray([_logdet(m) for m in total_fims(z, which)])

    rng = np.random.default_rng(seed)
    bounds = [(lo, hi) for _ in range(n_exp) for lo, hi in zip(lbs, ubs)]

    def scipy_constraints(offset: int = 0, n_extra: int = 0) -> list[dict]:
        """The per-run constraints, over the flat design vector.

        ``offset``/``n_extra`` let the same constraints be reused when the
        vector carries extra variables (the max-min epigraph variable).
        """
        cons: list[dict] = []
        for i in range(n_exp):

            def row(z: np.ndarray, i=i) -> dict[str, float]:
                head = z[: len(z) - n_extra] if n_extra else z
                return designs_of(np.asarray(head))[i]

            cons += [
                {"type": "eq", "fun": lambda z, g=g, row=row: float(g(row(z)))}
                for g in equality_constraints
            ]
            cons += [
                {"type": "ineq", "fun": lambda z, h=h, row=row: float(h(row(z)))}
                for h in inequality_constraints
            ]
        return cons

    def best_of(
        obj: Callable[[np.ndarray], float], method: str, *, use_constraints: bool = True
    ) -> np.ndarray:
        """Best of ``n_starts`` random starts. SLSQP whenever there are constraints.

        ``use_constraints=False`` searches the whole box: that is what the
        efficiency references need (see below).
        """
        apply_cons = constrained and use_constraints
        cons = scipy_constraints() if apply_cons else []
        if apply_cons:
            method = "SLSQP"
        best_z, best_f = None, np.inf
        for _ in range(max(1, int(n_starts))):
            z0 = np.concatenate([rng.uniform(lbs, ubs) for _ in range(n_exp)])
            res = minimize(obj, z0, method=method, bounds=bounds, constraints=cons)
            z = np.clip(np.asarray(res.x), np.tile(lbs, n_exp), np.tile(ubs, n_exp))
            if apply_cons and _violations(
                designs_of(z), equality_constraints, inequality_constraints
            ):
                continue  # an infeasible refinement is not a candidate at all
            value = float(obj(z))
            if value < best_f:
                best_f, best_z = value, z
        if best_z is None:
            raise ValueError(
                "no feasible design found: every start refined to a point violating the "
                "constraints. Check that the constraints are satisfiable inside "
                "design_bounds, or raise n_starts."
            )
        return best_z

    ref = None
    if robust == "maximin" or reference_scores is not None:
        if reference_scores is not None:
            ref = np.asarray(reference_scores, dtype=float)
            if ref.shape != (len(samples),):
                raise ValueError("reference scores need one value per parameter sample")
        else:
            # Each sample's best design sets its efficiency scale -- the best
            # design in the *box*, not the best design satisfying the user's
            # constraints. Scaling by the constrained optimum instead would make
            # every efficiency relative to a moving reference: a constrained run
            # would report a higher worst case than an unconstrained one on the
            # same problem, which reads as a constraint improving a design it
            # can only ever restrict. With the box reference, the number says
            # what the constraint costs.
            ref = np.empty(len(samples))
            for s in range(len(samples)):
                z_s = best_of(
                    lambda z, _s=s: -float(scores(z, [_s])[0]),
                    "L-BFGS-B",
                    use_constraints=False,
                )
                ref[s] = scores(z_s, [s])[0]

    if robust == "expected":
        z = best_of(lambda z: -float(np.mean(scores(z))), "L-BFGS-B")
        value = float(np.mean(scores(z)))
    else:
        assert ref is not None

        # The worst case is a min of smooth functions, so it has kinks wherever
        # the argmin changes sample. The epigraph form removes them: maximize t
        # subject to every sample's efficiency being at least t, which is smooth
        # in (z, t) and takes the user's constraints at the same time. It is
        # never worse than the direct min, and on a kinked problem it is better
        # (worst-case efficiency 0.60 -> 0.63 on the book's three-sample reactor
        # schedule).
        # Warm start from the pseudo-Bayesian design: it is optimal for the mean
        # and usually in the right region for the worst case too.
        try:
            warm = [best_of(lambda z: -float(np.mean(scores(z))), "L-BFGS-B")]
        except ValueError:  # no feasible expected design; random starts only
            warm = []
        z = _maximin_epigraph(
            minimize,
            scores=scores,
            efficiencies=lambda sc: _efficiencies(sc, ref, crit, p),
            designs_of=designs_of,
            bounds=bounds,
            constraints=scipy_constraints(n_extra=1),
            lbs=lbs,
            ubs=ubs,
            n_exp=n_exp,
            n_samples=len(samples),
            n_starts=int(n_starts),
            rng=rng,
            feasible=lambda zz: (
                not _violations(designs_of(zz), equality_constraints, inequality_constraints)
            ),
            extra_starts=warm,
        )
        value = float(np.min(_efficiencies(scores(z), ref, crit, p)))

    ld = log_dets(z)
    final_scores = ld if crit == "D" else scores(z)
    eff = None if ref is None else _efficiencies(final_scores, ref, crit, p)
    if constrained:
        broken = _violations(designs_of(z), equality_constraints, inequality_constraints)
        if broken:  # pragma: no cover - best_of already rejects infeasible designs
            raise ValueError("the best design found violates its constraints: " + "; ".join(broken))
    return RobustDesignResult(
        designs=designs_of(z),
        robust=robust,
        criterion_value=value,
        log_dets=ld,
        efficiencies=eff,
        reference_log_dets=ref,
        parameter_samples=samples,
        n_parameters=int(p),
    )


def design_efficiencies(
    experiment: Experiment,
    designs: Sequence[Mapping[str, float]],
    param_samples: Sequence[Mapping[str, float]] | Mapping[str, Sequence[float]],
    reference_designs: Sequence[Sequence[Mapping[str, float]]] | None = None,
    *,
    reference_log_dets: Sequence[float] | None = None,
    prior_fim: np.ndarray | None = None,
) -> np.ndarray:
    """D-efficiency of ``designs`` at each parameter sample.

    Relative to ``reference_log_dets`` (one per sample) or to
    ``reference_designs`` (one design per sample, e.g. each sample's local
    optimum). Use it to check a robust design on fresh parameter draws that it
    was not built on.
    """
    samples = _as_samples(param_samples)
    fims = _fim_functions(experiment, samples)
    p = fims[0](dict(designs[0])).shape[0]
    prior = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=float)

    def ld(s: int, ds: Sequence[Mapping[str, float]]) -> float:
        total = prior.copy()
        for d in ds:
            total = total + fims[s](dict(d))
        return _logdet(total)

    if reference_log_dets is not None:
        ref = np.asarray(reference_log_dets, dtype=float)
    elif reference_designs is not None:
        ref = np.array([ld(s, reference_designs[s]) for s in range(len(samples))])
    else:
        raise ValueError("pass reference_log_dets or reference_designs")
    return np.exp((np.array([ld(s, designs) for s in range(len(samples))]) - ref) / p)


__all__ = ["RobustDesignResult", "design_efficiencies", "robust_optimal_experiment"]
