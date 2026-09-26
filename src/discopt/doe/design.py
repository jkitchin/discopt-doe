"""Optimal experimental design via FIM criterion optimization.

Finds experimental conditions that maximize the information content
of an experiment, as measured by criteria derived from the Fisher
Information Matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from discopt.doe.fim import (
    FIMResult,
    _make_direct_fim_evaluator,
    compute_fim,
    compute_fim_batch,
)
from discopt.doe.linear_design import trace_inverse

# The mixture-constraint geometry lives in discopt.doe.simplex, which has no
# FIM or Experiment dependency — a Scheffé design in a WebAssembly build needs
# these three without being able to import this module at all. Re-exported
# here so the import paths that predate that split keep working.
from discopt.doe.simplex import (
    DesignConstraint,
    project_to_simplex,  # noqa: F401 - re-exported for backwards compatibility
    sample_simplex,  # noqa: F401 - re-exported for backwards compatibility
    sum_constraint,  # noqa: F401 - re-exported for backwards compatibility
)
from discopt.estimate import Experiment

if TYPE_CHECKING:  # DesignRegion is imported lazily where it is built
    from discopt.doe.linear_design import DesignRegion

_SINGULAR_SENTINEL = 1e12


class DesignCriterion:
    """Design optimality criteria constants.

    The first four are about the *parameters*: how small the confidence
    ellipsoid is (D), the total variance (A), the worst-determined direction
    (E), how spherical it is (ME). The last two are about *predictions* over a
    region you name, and need one: I is the average variance of the fitted
    response over the region, G its maximum.
    """

    D_OPTIMAL = "determinant"
    A_OPTIMAL = "trace"
    E_OPTIMAL = "min_eigenvalue"
    ME_OPTIMAL = "condition_number"
    I_OPTIMAL = "average_variance"
    G_OPTIMAL = "max_variance"


class BatchStrategy:
    """Strategies for batch / parallel experimental design."""

    GREEDY = "greedy"
    JOINT = "joint"
    PENALIZED = "penalized"


@dataclass
class DesignResult:
    """Result of optimal experimental design.

    Attributes
    ----------
    design : dict[str, float]
        Optimal values for each design input.
    fim_result : FIMResult
        FIM at the optimal design.
    criterion_value : float
        Value of the optimized design criterion.
    """

    design: dict[str, float]
    fim_result: FIMResult
    criterion_value: float

    @property
    def fim(self) -> np.ndarray:
        """Fisher Information Matrix at optimal design."""
        return self.fim_result.fim

    @property
    def parameter_covariance(self) -> np.ndarray:
        """Predicted parameter covariance if this experiment is run."""
        try:
            return np.linalg.inv(self.fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        """Predicted standard errors for each parameter."""
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics."""
        return self.fim_result.metrics

    def summary(self) -> str:
        """Human-readable summary of the optimal design."""
        lines = ["Optimal Experimental Design", "=" * 50]
        for name, val in self.design.items():
            lines.append(f"  {name:>15s} = {val:.6g}")
        lines.append("")
        m = self.metrics
        lines.append(f"  D-opt (log det FIM) = {m['log_det_fim']:.4g}")
        lines.append(f"  A-opt (trace FIM⁻¹) = {m['trace_fim_inv']:.4g}")
        lines.append(f"  E-opt (min eigenval) = {m['min_eigenvalue']:.4g}")
        lines.append(f"  Condition number     = {m['condition_number']:.4g}")
        lines.append("")
        se = self.predicted_standard_errors
        for i, name in enumerate(self.fim_result.parameter_names):
            lines.append(f"  SE({name}) = {se[i]:.4g}")
        return "\n".join(lines)


@dataclass
class BatchDesignResult:
    """Result of joint / batch optimal experimental design.

    Attributes
    ----------
    designs : list[dict[str, float]]
        Optimal values for each of the ``N`` designs in the batch.
    fim_results : list[FIMResult]
        Per-experiment FIMs, each computed without the prior or other
        batch members folded in (so ``sum(r.fim for r in fim_results)
        + prior_fim == joint_fim``).
    joint_fim : numpy.ndarray
        Sum of per-experiment FIMs plus ``prior_fim`` (if supplied).
        This is the FIM that the chosen criterion was evaluated on.
    criterion_value : float
        Value of the optimized design criterion on ``joint_fim``.
    strategy : str
        Name of the batch strategy used (see :class:`BatchStrategy`).
    per_round_criterion : list[float] or None
        For greedy / penalized strategies, the criterion value after
        each successive pick. ``None`` for joint strategy.
    """

    designs: list[dict[str, float]]
    fim_results: list[FIMResult]
    joint_fim: np.ndarray
    criterion_value: float
    strategy: str
    per_round_criterion: list[float] | None = None

    @property
    def n_experiments(self) -> int:
        return len(self.designs)

    @property
    def parameter_covariance(self) -> np.ndarray:
        """Predicted parameter covariance after running the full batch."""
        try:
            return np.linalg.inv(self.joint_fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.joint_fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        """Predicted standard errors for each parameter."""
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics evaluated on the joint FIM."""
        return _metrics_from_fim(self.joint_fim)

    @property
    def parameter_names(self) -> list[str]:
        return self.fim_results[0].parameter_names if self.fim_results else []

    def to_design_result(self) -> DesignResult:
        """Return a single-experiment ``DesignResult`` view (requires N=1)."""
        if self.n_experiments != 1:
            raise ValueError(
                f"to_design_result requires n_experiments == 1, got {self.n_experiments}"
            )
        return DesignResult(
            design=self.designs[0],
            fim_result=self.fim_results[0],
            criterion_value=self.criterion_value,
        )

    def summary(self) -> str:
        """Human-readable summary of the batch design."""
        lines = [
            f"Batch Optimal Design (N={self.n_experiments}, strategy={self.strategy})",
            "=" * 60,
        ]
        for i, design in enumerate(self.designs):
            lines.append(f"  Experiment {i + 1}:")
            for name, val in design.items():
                lines.append(f"    {name:>15s} = {val:.6g}")
        lines.append("")
        m = self.metrics
        lines.append(f"  D-opt (log det joint FIM) = {m['log_det_fim']:.4g}")
        lines.append(f"  A-opt (trace joint FIM⁻¹) = {m['trace_fim_inv']:.4g}")
        lines.append(f"  E-opt (min eigenval)      = {m['min_eigenvalue']:.4g}")
        lines.append(f"  Condition number          = {m['condition_number']:.4g}")
        lines.append("")
        se = self.predicted_standard_errors
        for i, name in enumerate(self.parameter_names):
            lines.append(f"  SE({name}) = {se[i]:.4g}")
        return "\n".join(lines)


def _metrics_from_fim(fim: np.ndarray) -> dict[str, float]:
    """All optimality metrics computed from a raw FIM matrix."""
    # slogdet is stable for badly-scaled FIMs where det over/underflows.
    sign, slog = np.linalg.slogdet(fim)
    log_det = float(slog) if sign > 0 and np.isfinite(slog) else float("-inf")
    tr_inv = trace_inverse(fim)
    return {
        "log_det_fim": log_det,
        "trace_fim_inv": tr_inv,
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(fim))),
        "condition_number": float(np.linalg.cond(fim)),
    }


PREDICTION_CRITERIA = (DesignCriterion.I_OPTIMAL, DesignCriterion.G_OPTIMAL)


def _criterion_from_fim(
    fim: np.ndarray,
    criterion: str,
    region: "DesignRegion | None" = None,
    *,
    smooth: bool = False,
) -> float:
    """Evaluate a design criterion directly on a FIM matrix.

    ``region`` is required for the prediction criteria and ignored by the
    others. ``smooth=True`` asks G for its soft maximum, a differentiable
    stand-in the local optimizer can follow; the exact maximum is piecewise and
    a gradient search stalls on its kinks. Report the exact one.
    """
    if criterion in PREDICTION_CRITERIA:
        if region is None:
            raise ValueError(
                f"criterion {criterion!r} predicts over a region, so it needs one: pass "
                "prediction_bounds or prediction_points (see experiment_region)"
            )
        if criterion == DesignCriterion.I_OPTIMAL:
            return region.average_variance(fim)
        return region.soft_max_variance(fim) if smooth else region.max_variance(fim)
    metrics = _metrics_from_fim(fim)
    if criterion == DesignCriterion.D_OPTIMAL:
        return metrics["log_det_fim"]
    elif criterion == DesignCriterion.A_OPTIMAL:
        return metrics["trace_fim_inv"]
    elif criterion == DesignCriterion.E_OPTIMAL:
        return metrics["min_eigenvalue"]
    elif criterion == DesignCriterion.ME_OPTIMAL:
        return metrics["condition_number"]
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")


def experiment_region(
    experiment: Experiment,
    param_values: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    points: Sequence[Mapping[str, float]] | None = None,
    n_points: int = 256,
    seed: int = 0,
) -> "DesignRegion":
    """Where a design has to predict well, for the I and G criteria.

    For a model nonlinear in its parameters the row ``f(x)`` that the
    prediction variance is built from is the *sensitivity* row
    ``dy/dtheta`` at ``x``, evaluated at the nominal parameters -- the same
    rows the FIM is assembled from. This computes them once for every point in
    the region, which is what makes an I- or G-optimal search affordable: they
    depend on the nominals and the region, never on the design being searched,
    so each candidate costs one solve against a precomputed matrix.

    An experiment with several responses (a dynamic experiment measuring a
    state at four times has four) contributes one row per response per point,
    so I averages over points *and* responses, and G takes the worst of them.

    Parameters
    ----------
    experiment, param_values
        The experiment and the nominal parameters the sensitivities are taken
        at, as for :func:`~discopt.doe.compute_fim`.
    bounds : mapping, optional
        Box to sample the region from, usually the design bounds.
    points : sequence of mapping, optional
        Explicit points, used as given instead of sampling.
    n_points, seed
        Sample size and seed when sampling from ``bounds``.
    """
    from discopt.doe.linear_design import DesignRegion
    from discopt.doe.prediction import region_points

    if points is None:
        if not bounds:
            raise ValueError("give either bounds to sample the region from, or points")
        names = list(bounds)
        points = region_points(names, int(n_points), bounds=bounds, seed=seed)
    rows = [dict(r) for r in points]
    if not rows:
        raise ValueError("the prediction region has no points")
    names = list(rows[0])

    results = compute_fim_batch(experiment, dict(param_values), rows)
    jac = np.stack([np.atleast_2d(np.asarray(r.jacobian, dtype=float)) for r in results])
    n_points_actual, n_responses, n_par = jac.shape
    flat = jac.reshape(n_points_actual * n_responses, n_par)
    coords = np.array([[float(r[n]) for n in names] for r in rows], dtype=float)
    repeated = np.repeat(coords, n_responses, axis=0)
    weights = np.full(flat.shape[0], 1.0 / flat.shape[0])
    return DesignRegion(flat, weights, flat, repeated, repeated, "sensitivity")


def optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[DesignConstraint] | None = None,
    inequality_constraints: Sequence[DesignConstraint] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    prediction_region: "DesignRegion | None" = None,
    prediction_points: Sequence[Mapping[str, float]] | None = None,
    n_prediction_points: int = 256,
    n_starts: int = 10,
    local_refine: bool = True,
    n_refine: int = 4,
    initial_designs: Sequence[Mapping[str, float]] | None = None,
    seed: int = 42,
) -> DesignResult:
    """Find optimal experimental conditions by maximizing information gain.

    Evaluates the FIM criterion at multiple starting points within the
    design bounds and refines the best candidate with a bounded local
    solver. When constraints are supplied, the refiner switches to SLSQP.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Current best parameter estimates (nominal values).
    design_bounds : dict[str, tuple[float, float]]
        Bounds on each design input variable.
    criterion : str, default DesignCriterion.D_OPTIMAL
        Design criterion: ``"determinant"`` (D), ``"trace"`` (A),
        ``"min_eigenvalue"`` (E), ``"condition_number"`` (ME),
        ``"average_variance"`` (I) or ``"max_variance"`` (G). The last two are
        about predictions rather than parameters, so they need a region to
        predict over: by default the design bounds, sampled at
        ``n_prediction_points``, or give ``prediction_points`` or a
        ``prediction_region`` built with :func:`experiment_region`.
    prediction_region : DesignRegion, optional
        A region built once with :func:`experiment_region`, to reuse across
        calls. Only the I and G criteria read it.
    prediction_points : sequence of mapping, optional
        Explicit points to predict at, instead of sampling the design bounds.
    n_prediction_points : int, default 256
        Sample size when the region is sampled from the design bounds.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previous experiments.
    equality_constraints : sequence of callable, optional
        Constraints ``g(design) == 0``. For mixture designs use
        :func:`sum_constraint`.
    inequality_constraints : sequence of callable, optional
        Constraints ``h(design) >= 0`` (scipy SLSQP convention).
    feasible_projection : callable, optional
        Maps a raw candidate to a constraint-feasible candidate. Used to
        seed the multi-start with feasible points. For mixture designs
        pass a partial of :func:`project_to_simplex`.
    n_starts : int, default 10
        Number of random starting points to evaluate.
    local_refine : bool, default True
        If True, refine the best multi-start candidate with
        scipy.optimize.minimize (L-BFGS-B without constraints, SLSQP
        when ``equality_constraints`` or ``inequality_constraints`` are
        supplied).
    initial_designs : sequence of mapping, optional
        Designs to add to the multi-start candidates -- the current operating
        point, a design from a previous round, a literature design. In a
        high-dimensional design space random starts rarely land in the basin
        of a good design that is already known. Values are clipped to
        ``design_bounds``; every design input must be given.
    n_refine : int, default 4
        Number of the best-scoring multi-start candidates the local solver is
        started from (unconstrained problems); the best refined design wins.
        Refining only the single best candidate lands in a local optimum
        whenever the criterion is multimodal in the design, and can stall
        outright when the first step reaches a singular region.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DesignResult
        Optimal design, FIM, and metrics.
    """
    design_names = list(design_bounds.keys())
    eq = list(equality_constraints) if equality_constraints else []
    ineq = list(inequality_constraints) if inequality_constraints else []
    constrained = bool(eq or ineq)
    candidates = _multi_start_candidates(
        design_bounds, n_starts, seed, projection=feasible_projection
    )
    initial_designs = list(initial_designs or ())
    if criterion in (DesignCriterion.E_OPTIMAL, DesignCriterion.ME_OPTIMAL):
        # E and ME are nonsmooth (eigenvalues cross) and near-flat around the
        # nearly singular designs random candidates tend to be, so a gradient
        # refinement from them does not move (bi-exponential sampling times:
        # E = 1e-7 against an optimum of 120). The D-optimal design spreads
        # information over every direction and sits in the right basin, so it
        # joins the candidates.
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d_seed = optimal_experiment(
                    experiment,
                    param_values,
                    design_bounds,
                    criterion=DesignCriterion.D_OPTIMAL,
                    prior_fim=prior_fim,
                    equality_constraints=eq,
                    inequality_constraints=ineq,
                    feasible_projection=feasible_projection,
                    n_starts=n_starts,
                    local_refine=local_refine,
                    n_refine=n_refine,
                    initial_designs=initial_designs,
                    seed=seed,
                )
            initial_designs.append(d_seed.design)
        except Exception:  # noqa: BLE001 - the E/ME search runs without the seed
            pass
    for given in initial_designs:
        missing = [n for n in design_names if n not in given]
        if missing:
            raise ValueError(f"initial design {dict(given)} is missing design inputs {missing}")
        candidates.append(
            {
                n: float(np.clip(float(given[n]), design_bounds[n][0], design_bounds[n][1]))
                for n in design_names
            }
        )
    if constrained and feasible_projection is None:
        # Random points in the box rarely satisfy the constraints (four
        # sampling times summing to <= 5 in [0.05, 30]^4: essentially never), so
        # add each infeasible candidate's projection onto the feasible set.
        # Without feasible seeds the refinement starts from one infeasible point
        # and whatever feasible corner SLSQP reaches first is accepted.
        candidates = candidates + _project_candidates(candidates, design_bounds, eq, ineq)

    # Best-by-criterion candidate, ignoring constraints — used only as the
    # SLSQP refinement seed. The scan ranks purely by the criterion, so when
    # constraints are present this incumbent is typically infeasible and must
    # not be returned as-is.
    region = prediction_region
    if criterion in PREDICTION_CRITERIA and region is None:
        # The sensitivity rows depend on the nominals and the region only, so
        # they are computed once here and reused by every candidate evaluation.
        region = experiment_region(
            experiment,
            param_values,
            bounds=None if prediction_points is not None else design_bounds,
            points=prediction_points,
            n_points=n_prediction_points,
            seed=seed,
        )

    seed_design, scan_criterion, scan_fim_result = _scan_candidates(
        experiment, param_values, candidates, criterion, prior_fim, region
    )
    if seed_design is None or scan_fim_result is None:
        # Re-evaluate one candidate to surface the underlying failure (a bad
        # parameter name, a shape mismatch, ...) instead of a generic message.
        # _scan_candidates swallows the batch exception to stay robust.
        try:
            compute_fim(experiment, param_values, candidates[0], prior_fim=prior_fim)
        except Exception as e:  # noqa: BLE001 -- surfaced as the cause below
            raise RuntimeError(
                "No feasible design point found; the FIM could not be evaluated "
                "at any candidate (see the chained error for the cause)."
            ) from e
        raise RuntimeError("No feasible design point found")

    if not constrained:
        best_design, best_criterion, best_fim_result = (
            seed_design,
            scan_criterion,
            scan_fim_result,
        )
    else:
        # Restrict the incumbent result to constraint-feasible candidates.
        feasible = [c for c in candidates if _is_feasible(c, eq, ineq)]
        if feasible:
            best_design, best_criterion, best_fim_result = _scan_candidates(
                experiment, param_values, feasible, criterion, prior_fim, region
            )
        else:
            best_design = None
            best_criterion = -np.inf if _is_maximization(criterion) else np.inf
            best_fim_result = None

    if local_refine:
        seeds = [seed_design]
        if n_refine > 1:
            pool = (
                [c for c in candidates if _is_feasible(c, eq, ineq)] if constrained else candidates
            )
            top = _top_candidates(
                experiment, param_values, pool, criterion, prior_fim, region, n_refine
            )
            if constrained:
                seeds = top + ([seed_design] if seed_design not in top else [])
            else:
                seeds = top or [seed_design]
        refined = None
        for start in seeds:
            attempt = _refine_single_design(
                experiment,
                param_values,
                start,
                design_names,
                design_bounds,
                criterion,
                prior_fim,
                region=region,
                equality_constraints=eq,
                inequality_constraints=ineq,
            )
            if attempt is not None and (
                refined is None or _is_better(attempt[1], refined[1], criterion)
            ):
                refined = attempt
        if refined is not None:
            r_design, r_criterion, r_fim_result = refined
            # Unconstrained: accept any improvement. Constrained: SLSQP does
            # not guarantee exact feasibility, so accept only feasible refined
            # designs — and prefer one whenever we have no feasible incumbent.
            accept = (
                _is_better(r_criterion, best_criterion, criterion)
                if best_design is not None
                else True
            )
            if constrained:
                accept = accept and _is_feasible(r_design, eq, ineq)
            if accept:
                best_design, best_criterion, best_fim_result = refined

        # Constrained fallback: refining only the best-by-criterion seed can
        # fail when that seed is far from the feasible set — SLSQP may report
        # "incompatible constraints" and diverge to a singular corner (observed
        # on some BLAS/LAPACK builds, e.g. macOS/Accelerate), leaving no
        # feasible design even though one exists. Before giving up, retry SLSQP
        # from every candidate and keep the best feasible result.
        if constrained and best_design is None:
            fallback = _best_feasible_refinement(
                experiment,
                param_values,
                candidates,
                design_names,
                design_bounds,
                criterion,
                prior_fim,
                eq,
                ineq,
                region,
            )
            if fallback is not None:
                best_design, best_criterion, best_fim_result = fallback

    if best_design is None or best_fim_result is None:
        raise RuntimeError(
            "No constraint-feasible design point found. Pass "
            "feasible_projection (e.g. a partial of project_to_simplex) to "
            "seed the multi-start with constraint-satisfying candidates."
        )

    _warn_if_singular(best_fim_result)

    return DesignResult(
        design=best_design,
        fim_result=best_fim_result,
        criterion_value=best_criterion,
    )


# A FIM whose correlation form (unit diagonal) is this ill-conditioned is
# singular to working precision: its smallest eigenvalue is round-off, so the
# D/A/E criterion values, and the design that optimizes them, are noise.
_SINGULAR_CORRELATION_COND = 1e12


def _warn_if_singular(fim_result: FIMResult) -> None:
    """Warn when the optimal design's FIM cannot identify the parameters.

    Uses the diagonal-normalized FIM, so parameters in very different units
    (a pre-exponential factor and an activation energy) do not trigger it.
    """
    import warnings

    fim = np.asarray(fim_result.fim, dtype=float)
    diag = np.diag(fim)
    names = list(fim_result.parameter_names)
    if np.any(diag <= 0) or not np.all(np.isfinite(fim)):
        blind = [n for n, d in zip(names, diag) if not d > 0]
        warnings.warn(
            f"the FIM at the optimal design carries no information about {blind}; "
            "the design criterion is degenerate and the returned design is arbitrary.",
            stacklevel=3,
        )
        return
    scale = 1.0 / np.sqrt(diag)
    corr = fim * scale[:, None] * scale[None, :]
    eig, vec = np.linalg.eigh(0.5 * (corr + corr.T))
    if eig[-1] <= 0 or eig[0] <= eig[-1] / _SINGULAR_CORRELATION_COND:
        v = vec[:, 0]
        involved = [n for n, c in zip(names, v) if abs(c) > 0.1]
        warnings.warn(
            "the FIM at the optimal design is numerically singular (condition number "
            f"of its correlation form {eig[-1] / max(eig[0], 1e-300):.1e}): the "
            f"combination of {involved} is not identifiable from this experiment, so "
            "the criterion value and the design are dominated by round-off. Add a "
            "prior_fim from other experiments, more responses, or fix/reparameterize "
            "those parameters (see diagnose_identifiability).",
            stacklevel=3,
        )


def _multi_start_candidates(
    design_bounds: dict[str, tuple[float, float]],
    n_starts: int,
    seed: int,
    projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Generate random interior + boundary design candidates.

    When ``projection`` is supplied, each candidate is passed through it
    before being returned (used to seed multi-start with feasible points
    for problems with equality constraints, e.g. mixture designs).
    """
    rng = np.random.default_rng(seed)
    design_names = list(design_bounds.keys())

    candidates: list[dict[str, float]] = []
    for _ in range(n_starts):
        candidates.append({name: rng.uniform(*design_bounds[name]) for name in design_names})

    for name in design_names:
        lo, hi = design_bounds[name]
        for val in (lo, hi):
            point = {n: (design_bounds[n][0] + design_bounds[n][1]) / 2 for n in design_names}
            point[name] = val
            candidates.append(point)

    if projection is not None:
        candidates = [projection(c) for c in candidates]
    return candidates


def _scan_candidates(
    experiment: Experiment,
    param_values: dict[str, float],
    candidates: list[dict[str, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
    region: "DesignRegion | None" = None,
) -> tuple[dict[str, float] | None, float, FIMResult | None]:
    """Evaluate each candidate and return the best."""
    best_design: dict[str, float] | None = None
    best_criterion = -np.inf if _is_maximization(criterion) else np.inf
    best_fim_result: FIMResult | None = None

    # Multi-RHS: evaluate every candidate's FIM in one batched pass (no
    # per-candidate solve / Jacobian re-trace for explicit response models).
    fim_results: list[FIMResult | None]
    try:
        fim_results = list(
            compute_fim_batch(experiment, param_values, candidates, prior_fim=prior_fim)
        )
    except Exception:
        fim_results = [None] * len(candidates)

    for design_point, fim_result in zip(candidates, fim_results):
        if fim_result is None:
            continue
        try:
            crit_val = _evaluate_criterion(fim_result, criterion, region)
        except Exception:
            continue

        if _is_better(crit_val, best_criterion, criterion):
            best_criterion = crit_val
            best_design = design_point
            best_fim_result = fim_result

    return best_design, best_criterion, best_fim_result


def _project_candidates(
    candidates: list[dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    equality_constraints: Sequence[DesignConstraint],
    inequality_constraints: Sequence[DesignConstraint],
) -> list[dict[str, float]]:
    """Constraint-feasible seeds derived from infeasible candidates.

    With inequality constraints only, find one interior point that maximizes
    the smallest constraint slack, then pull each infeasible candidate toward it
    along the connecting segment (bisection) until it is feasible. Seeds stay
    interior and keep the candidates' spread. A nearest-point projection would
    instead pile them onto the constraint and the box faces -- for sampling
    times, several at the same bound, which is a singular design SLSQP cannot
    leave. Equality constraints fall back to that projection, the only
    construction that keeps them satisfied. Work is in bounds-normalized
    coordinates; candidates that cannot be made feasible are dropped.
    """
    names = list(design_bounds)
    lo = np.array([design_bounds[n][0] for n in names], dtype=float)
    span = np.array([design_bounds[n][1] - design_bounds[n][0] for n in names], dtype=float)
    span[span == 0] = 1.0
    eq, ineq = list(equality_constraints), list(inequality_constraints)

    def to_design(z: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(names, lo + np.clip(z, 0.0, 1.0) * span)}

    def feasible(z: np.ndarray) -> bool:
        return _is_feasible(to_design(z), eq, ineq)

    infeasible = [
        (np.array([c[n] for n in names], dtype=float) - lo) / span
        for c in candidates
        if not _is_feasible(c, eq, ineq)
    ]
    if not infeasible:
        return []
    bounds01 = [(0.0, 1.0)] * len(names)

    if not eq:
        # max s  s.t.  h_i(x) >= s, x in the box
        z0 = np.append(np.full(len(names), 0.5), 0.0)
        cons = [
            {"type": "ineq", "fun": (lambda v, h=h: float(h(to_design(v[:-1]))) - v[-1])}
            for h in ineq
        ]
        try:
            res = minimize(
                lambda v: -v[-1],
                z0,
                jac=lambda v: np.append(np.zeros(len(names)), -1.0),
                method="SLSQP",
                bounds=bounds01 + [(None, None)],
                constraints=cons,
            )
            center = np.clip(res.x[:-1], 0.0, 1.0)
        except Exception:
            center = None
        if center is not None and feasible(center):
            out = [to_design(center)]
            for z in infeasible:
                a, b = 0.0, 1.0  # fraction of the way from the center to z
                for _ in range(30):
                    mid = 0.5 * (a + b)
                    if feasible(center + mid * (z - center)):
                        a = mid
                    else:
                        b = mid
                out.append(to_design(center + a * (z - center)))
            return out

    cons = [{"type": "eq", "fun": (lambda z, g=g: float(g(to_design(z))))} for g in eq] + [
        {"type": "ineq", "fun": (lambda z, h=h: float(h(to_design(z))))} for h in ineq
    ]
    out = []
    for z0 in infeasible:
        try:
            res = minimize(
                lambda z: float(np.sum((z - z0) ** 2)),
                z0,
                jac=lambda z: 2.0 * (z - z0),
                method="SLSQP",
                bounds=bounds01,
                constraints=cons,
            )
        except Exception:
            continue
        if feasible(res.x):
            out.append(to_design(res.x))
    return out


# Criteria refined on a log scale. A, E and ME span many decades across a design
# box (A: 1e-2 at the optimum, 1e7 near a singular corner; E: ~1e-6 in the
# units of an activation energy), which stalls a gradient search -- L-BFGS-B's
# gradient tolerance is absolute, so a criterion of size 1e-6 looks converged
# at the first step. log() is monotone, so the optimum is unchanged.
_LOG_SCALED_CRITERIA = (
    DesignCriterion.A_OPTIMAL,
    DesignCriterion.E_OPTIMAL,
    DesignCriterion.ME_OPTIMAL,
)


def _log_objective(crit: float, maximize: bool) -> float:
    """Minimization objective ``±log(crit)``; a non-positive value is singular."""
    if not crit > 0:
        return _SINGULAR_SENTINEL
    value = float(np.log(crit))
    return -value if maximize else value


def _top_candidates(
    experiment: Experiment,
    param_values: dict[str, float],
    candidates: list[dict[str, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
    region: "DesignRegion | None",
    k: int,
) -> list[dict[str, float]]:
    """The ``k`` best candidates with a finite criterion, best first."""
    try:
        fims = compute_fim_batch(experiment, param_values, candidates, prior_fim=prior_fim)
    except Exception:
        return []
    scored = []
    for design_point, fim_result in zip(candidates, fims):
        try:
            value = _evaluate_criterion(fim_result, criterion, region)
        except Exception:
            continue
        if np.isfinite(value):
            scored.append((value, design_point))
    scored.sort(key=lambda t: -t[0] if _is_maximization(criterion) else t[0])
    return [d for _, d in scored[:k]]


def _refine_single_design(
    experiment: Experiment,
    param_values: dict[str, float],
    seed_design: dict[str, float],
    design_names: list[str],
    design_bounds: dict[str, tuple[float, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    *,
    region: "DesignRegion | None" = None,
) -> tuple[dict[str, float], float, FIMResult] | None:
    """Local refinement of a single design via scipy L-BFGS-B or SLSQP."""
    maximize = _is_maximization(criterion)
    bounds = [design_bounds[n] for n in design_names]
    x0 = np.array([seed_design[n] for n in design_names], dtype=float)

    # The scipy refiner evaluates the FIM many times (each L-BFGS-B / SLSQP step
    # plus its finite-difference gradient). For a pure explicit response model,
    # build the model and JIT-compile the response Jacobian ONCE here and reuse
    # it across every evaluation, instead of rebuilding + re-tracing per call
    # inside compute_fim. Falls back to per-call compute_fim (which solves) for
    # constrained / implicit-state models, where the evaluator is None.
    evaluator = _make_direct_fim_evaluator(experiment, param_values, prior_fim=prior_fim)

    def eval_fim(design: dict[str, float]) -> FIMResult:
        if evaluator is not None:
            return evaluator(design)
        return compute_fim(experiment, param_values, design, prior_fim=prior_fim)

    def to_design(x: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(design_names, x)}

    log_scale = criterion in _LOG_SCALED_CRITERIA

    def objective(x: np.ndarray) -> float:
        design = to_design(x)
        try:
            fim_result = eval_fim(design)
            crit = _evaluate_criterion(fim_result, criterion, region, smooth=True)
        except Exception:
            return _SINGULAR_SENTINEL
        if not np.isfinite(crit):
            return _SINGULAR_SENTINEL
        if log_scale:
            return _log_objective(crit, maximize)
        return -crit if maximize else crit

    has_constraints = bool(equality_constraints) or bool(inequality_constraints)
    if has_constraints:
        scipy_cons = [
            {"type": "eq", "fun": (lambda x, g=g: float(g(to_design(x))))}
            for g in equality_constraints
        ] + [
            {"type": "ineq", "fun": (lambda x, h=h: float(h(to_design(x))))}
            for h in inequality_constraints
        ]
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=scipy_cons)
        except Exception:
            return None
    else:
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            return None

    if not res.success and not np.isfinite(res.fun):
        return None

    design = to_design(np.asarray(res.x))
    try:
        fim_result = eval_fim(design)
    except Exception:
        return None
    crit_val = _evaluate_criterion(fim_result, criterion, region)
    if not np.isfinite(crit_val):
        return None
    return design, crit_val, fim_result


def _best_feasible_refinement(
    experiment: Experiment,
    param_values: dict[str, float],
    seeds: Sequence[dict[str, float]],
    design_names: list[str],
    design_bounds: dict[str, tuple[float, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint],
    inequality_constraints: Sequence[DesignConstraint],
    region: "DesignRegion | None" = None,
) -> tuple[dict[str, float], float, FIMResult] | None:
    """Refine from each seed via SLSQP; return the best feasible refinement.

    Used as a robustness fallback for constrained problems: refining a single
    seed can diverge to an infeasible/singular point on some platforms, so we
    scan every candidate seed and keep the constraint-feasible refined design
    with the best criterion. Returns ``None`` if no seed yields a feasible
    refinement.
    """
    best: tuple[dict[str, float], float, FIMResult] | None = None
    for seed in seeds:
        refined = _refine_single_design(
            experiment,
            param_values,
            seed,
            design_names,
            design_bounds,
            criterion,
            prior_fim,
            region=region,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )
        if refined is None:
            continue
        r_design, r_criterion, _ = refined
        if not _is_feasible(r_design, equality_constraints, inequality_constraints):
            continue
        if best is None or _is_better(r_criterion, best[1], criterion):
            best = refined
    return best


def _evaluate_criterion(
    fim_result: FIMResult,
    criterion: str,
    region: "DesignRegion | None" = None,
    *,
    smooth: bool = False,
) -> float:
    """Evaluate a design criterion from a FIM result."""
    if criterion in PREDICTION_CRITERIA:
        return _criterion_from_fim(np.asarray(fim_result.fim), criterion, region, smooth=smooth)
    if criterion == DesignCriterion.D_OPTIMAL:
        return fim_result.d_optimal
    elif criterion == DesignCriterion.A_OPTIMAL:
        return fim_result.a_optimal
    elif criterion == DesignCriterion.E_OPTIMAL:
        return fim_result.e_optimal
    elif criterion == DesignCriterion.ME_OPTIMAL:
        return fim_result.me_optimal
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")


def _is_maximization(criterion: str) -> bool:
    """Return True if the criterion should be maximized."""
    return criterion in (DesignCriterion.D_OPTIMAL, DesignCriterion.E_OPTIMAL)


def _is_better(new_val: float, best_val: float, criterion: str) -> bool:
    """Check if new_val is better than best_val for the given criterion."""
    if _is_maximization(criterion):
        return new_val > best_val
    return new_val < best_val


def _is_feasible(
    design: dict[str, float],
    equality_constraints: Sequence[DesignConstraint],
    inequality_constraints: Sequence[DesignConstraint],
    tol: float = 1e-6,
) -> bool:
    """True if ``design`` satisfies all constraints within ``tol``.

    Equality constraints ``g`` require ``|g(design)| <= tol``; inequality
    constraints ``h`` require ``h(design) >= -tol`` (scipy SLSQP convention).
    """
    for g in equality_constraints:
        if abs(float(g(design))) > tol:
            return False
    for h in inequality_constraints:
        if float(h(design)) < -tol:
            return False
    return True


def batch_optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    strategy: str = BatchStrategy.GREEDY,
    prior_fim: np.ndarray | None = None,
    prediction_region: "DesignRegion | None" = None,
    prediction_points: Sequence[Mapping[str, float]] | None = None,
    n_prediction_points: int = 256,
    equality_constraints: Sequence[DesignConstraint] | None = None,
    inequality_constraints: Sequence[DesignConstraint] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    min_distance: float | None = None,
    seed: int = 42,
    exchange_passes: int = 2,
) -> BatchDesignResult:
    """Design a batch of ``N`` experiments to run in parallel.

    Experiments are independent, so their Fisher information adds:
    ``FIM_total = Σ_i FIM(d_i) + FIM_prior``. Strategies differ in how
    they search the joint design space.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_bounds : dict[str, tuple[float, float]]
        Bounds on each design input variable.
    n_experiments : int
        Number of experiments in the batch (must be ``>= 1``).
    criterion : str, default ``DesignCriterion.D_OPTIMAL``
        Design criterion evaluated on the joint FIM.
    strategy : str, default ``BatchStrategy.GREEDY``
        Batch selection strategy. One of ``"greedy"``, ``"joint"``,
        ``"penalized"``.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previously collected data.
    n_starts : int, default 10
        Multi-start budget used by each internal single-design search
        (greedy / penalized) or the joint search.
    local_refine : bool, default True
        If True, apply scipy L-BFGS-B refinement at the relevant stage
        (single-design refinement for greedy / penalized, joint-vector
        refinement for joint).
    min_distance : float, optional
        Minimum normalised distance between selected designs. Only used
        by ``"penalized"``; ignored otherwise.
    seed : int, default 42
        Random seed for reproducibility.
    exchange_passes : int, default 2
        ``"greedy"`` only. Greedy selection never revisits an early pick, so
        the batch is polished by up to this many exchange sweeps: each run in
        turn is re-optimized given all the others and replaced if that
        improves the criterion. ``0`` gives pure greedy selection.

    Returns
    -------
    BatchDesignResult
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")

    eq = list(equality_constraints) if equality_constraints else []
    ineq = list(inequality_constraints) if inequality_constraints else []

    region = prediction_region
    if criterion in PREDICTION_CRITERIA and region is None:
        region = experiment_region(
            experiment,
            param_values,
            bounds=None if prediction_points is not None else design_bounds,
            points=prediction_points,
            n_points=n_prediction_points,
            seed=seed,
        )

    if strategy == BatchStrategy.GREEDY:
        return _greedy_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            region=region,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed,
            exchange_passes=exchange_passes,
        )
    elif strategy == BatchStrategy.JOINT:
        # Seed the joint search with the greedy batch: the joint optimum is at
        # least as good by definition, and a joint search from random stacks
        # alone can end below it (replicated optima are hard to reach from
        # distinct random points).
        try:
            greedy = _greedy_batch(
                experiment,
                param_values,
                design_bounds,
                n_experiments,
                criterion=criterion,
                prior_fim=prior_fim,
                region=region,
                equality_constraints=eq,
                inequality_constraints=ineq,
                feasible_projection=feasible_projection,
                n_starts=n_starts,
                local_refine=local_refine,
                seed=seed,
                exchange_passes=exchange_passes,
            )
            extra_starts = [greedy.designs]
        except Exception:  # noqa: BLE001 - the joint search runs without it
            extra_starts = []
        return _joint_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            region=region,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed,
            extra_starts=extra_starts,
        )
    elif strategy == BatchStrategy.PENALIZED:
        return _penalized_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            region=region,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            min_distance=min_distance,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown batch strategy: {strategy!r}")


def _greedy_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    region: "DesignRegion | None" = None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    seed: int,
    exchange_passes: int = 0,
) -> BatchDesignResult:
    """Greedy batch: pick one design at a time, folding each FIM into the prior.

    Optionally polished by exchange sweeps (see :func:`batch_optimal_experiment`).
    """
    running_prior = prior_fim.copy() if prior_fim is not None else None
    designs: list[dict[str, float]] = []
    fim_results: list[FIMResult] = []
    per_round: list[float] = []

    for i in range(n_experiments):
        picked = optimal_experiment(
            experiment,
            param_values,
            design_bounds,
            criterion=criterion,
            prior_fim=running_prior,
            equality_constraints=equality_constraints or None,
            inequality_constraints=inequality_constraints or None,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed + i,
        )
        per_fim = compute_fim(experiment, param_values, picked.design, prior_fim=None)
        designs.append(picked.design)
        fim_results.append(per_fim)
        running_prior = per_fim.fim.copy() if running_prior is None else running_prior + per_fim.fim
        per_round.append(_criterion_from_fim(running_prior, criterion, region))

    assert running_prior is not None  # n_experiments >= 1

    # Exchange refinement: re-optimize each run given all the others.
    maximize = criterion in (DesignCriterion.D_OPTIMAL, DesignCriterion.E_OPTIMAL)
    current = _criterion_from_fim(running_prior, criterion, region)
    for sweep in range(max(0, int(exchange_passes)) if np.isfinite(current) else 0):
        improved = False
        for i in range(len(designs)):
            others = running_prior - fim_results[i].fim
            try:
                picked = optimal_experiment(
                    experiment,
                    param_values,
                    design_bounds,
                    criterion=criterion,
                    prior_fim=others,
                    equality_constraints=equality_constraints or None,
                    inequality_constraints=inequality_constraints or None,
                    feasible_projection=feasible_projection,
                    n_starts=n_starts,
                    local_refine=local_refine,
                    seed=seed + n_experiments + sweep * n_experiments + i,
                )
            except Exception:  # noqa: BLE001 - a failed re-search keeps the greedy pick
                continue
            per_fim = compute_fim(experiment, param_values, picked.design, prior_fim=None)
            trial = others + per_fim.fim
            value = _criterion_from_fim(trial, criterion, region)
            gain = (value - current) if maximize else (current - value)
            if np.isfinite(value) and gain > 1e-10 * max(1.0, abs(current)):
                designs[i] = picked.design
                fim_results[i] = per_fim
                running_prior = trial
                current = value
                improved = True
        if not improved:
            break

    # Per-round criterion in final run order.
    per_round = []
    acc = prior_fim.copy() if prior_fim is not None else None
    for r in fim_results:
        acc = r.fim.copy() if acc is None else acc + r.fim
        per_round.append(_criterion_from_fim(acc, criterion, region))

    return BatchDesignResult(
        designs=designs,
        fim_results=fim_results,
        joint_fim=running_prior,
        criterion_value=per_round[-1],
        strategy=BatchStrategy.GREEDY,
        per_round_criterion=per_round,
    )


def _joint_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    region: "DesignRegion | None" = None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    seed: int,
    extra_starts: Sequence[list[dict[str, float]]] = (),
    n_refine: int = 4,
) -> BatchDesignResult:
    """Galvanina-style joint batch: optimize N design vectors simultaneously."""
    design_names = list(design_bounds.keys())
    d = len(design_names)
    maximize = _is_maximization(criterion)
    flat_bounds = [design_bounds[n] for n in design_names] * n_experiments
    lows = np.array([design_bounds[n][0] for n in design_names])
    highs = np.array([design_bounds[n][1] for n in design_names])

    def unpack(z: np.ndarray) -> list[dict[str, float]]:
        stacks = z.reshape(n_experiments, d)
        return [
            {name: float(stacks[i, j]) for j, name in enumerate(design_names)}
            for i in range(n_experiments)
        ]

    def joint_fim_and_pieces(
        designs: list[dict[str, float]],
    ) -> tuple[np.ndarray, list[FIMResult]] | None:
        if not designs:
            return None
        pieces: list[FIMResult] = []
        try:
            first = compute_fim(experiment, param_values, designs[0], prior_fim=None)
        except Exception:
            return None
        pieces.append(first)
        total: np.ndarray = first.fim.copy()
        for design in designs[1:]:
            try:
                piece = compute_fim(experiment, param_values, design, prior_fim=None)
            except Exception:
                return None
            pieces.append(piece)
            total = total + piece.fim
        if prior_fim is not None:
            total = total + prior_fim
        return total, pieces

    log_scale = criterion in _LOG_SCALED_CRITERIA

    def objective(z: np.ndarray) -> float:
        designs = unpack(z)
        result = joint_fim_and_pieces(designs)
        if result is None:
            return _SINGULAR_SENTINEL
        fim, _ = result
        crit = _criterion_from_fim(fim, criterion, region)
        if not np.isfinite(crit):
            return _SINGULAR_SENTINEL
        if log_scale:
            return _log_objective(crit, maximize)
        return -crit if maximize else crit

    rng = np.random.default_rng(seed)

    def project_stack(z: np.ndarray) -> np.ndarray:
        if feasible_projection is None:
            return z
        stacks = z.reshape(n_experiments, d)
        out = np.empty_like(z)
        for i in range(n_experiments):
            point = {name: float(stacks[i, j]) for j, name in enumerate(design_names)}
            projected = feasible_projection(point)
            for j, name in enumerate(design_names):
                out[i * d + j] = projected[name]
        return out

    # Multi-start: pure random stacks + a few structured ones.
    starts: list[np.ndarray] = []
    for _ in range(n_starts):
        z = rng.uniform(np.tile(lows, n_experiments), np.tile(highs, n_experiments))
        starts.append(project_stack(z))
    # Structured: one stack with each row at a distinct quantile of each bound.
    if n_experiments >= 2:
        quantiles = np.linspace(0.0, 1.0, n_experiments)
        structured = np.concatenate([lows + q * (highs - lows) for q in quantiles])
        starts.append(project_stack(structured))

    has_constraints = bool(equality_constraints) or bool(inequality_constraints)
    scipy_cons = []
    if has_constraints:

        def slot_dict(z: np.ndarray, i: int) -> dict[str, float]:
            return {name: float(z[i * d + j]) for j, name in enumerate(design_names)}

        for i in range(n_experiments):
            for g in equality_constraints:
                scipy_cons.append(
                    {"type": "eq", "fun": (lambda z, ii=i, gg=g: float(gg(slot_dict(z, ii))))}
                )
            for h in inequality_constraints:
                scipy_cons.append(
                    {"type": "ineq", "fun": (lambda z, ii=i, hh=h: float(hh(slot_dict(z, ii))))}
                )

    def _refine_joint(z0: np.ndarray) -> tuple[np.ndarray, float]:
        """Local refinement of one stacked start: SLSQP with constraints, else L-BFGS-B."""
        try:
            if has_constraints:
                res = minimize(
                    objective, z0, method="SLSQP", bounds=flat_bounds, constraints=scipy_cons
                )
                # SLSQP need not end feasible: accept only a feasible stack.
                if not all(
                    _is_feasible(p, equality_constraints, inequality_constraints)
                    for p in unpack(np.asarray(res.x))
                ):
                    return z0, _SINGULAR_SENTINEL
            else:
                res = minimize(objective, z0, method="L-BFGS-B", bounds=flat_bounds)
        except Exception:
            return z0, _SINGULAR_SENTINEL
        if not np.isfinite(res.fun):
            return z0, _SINGULAR_SENTINEL
        return np.asarray(res.x), float(res.fun)

    for stack in extra_starts:
        starts.append(np.array([[p[n] for n in design_names] for p in stack], dtype=float).ravel())

    scored = [(objective(z0), i) for i, z0 in enumerate(starts)]
    scored = [(v, i) for v, i in scored if np.isfinite(v) and v < _SINGULAR_SENTINEL]
    if not scored:
        raise RuntimeError("joint batch: no feasible starting point found")
    scored.sort()
    best_val, best_i = scored[0]
    final_z: np.ndarray = starts[best_i].copy()
    # Refine the best few starts, not only the best: the joint criterion is
    # multimodal in the stacked designs (every permutation is an optimum, and
    # replicate structure separates basins).
    refine_from = [starts[i] for _, i in scored[:n_refine]]
    for extra in starts[len(starts) - len(extra_starts) :]:
        if not any(extra is z for z in refine_from):
            refine_from.append(extra)
    if local_refine:
        for z_start in refine_from:
            z_ref, v_ref = _refine_joint(z_start)
            if v_ref < best_val:
                final_z, best_val = z_ref, v_ref
    designs = unpack(final_z)
    final = joint_fim_and_pieces(designs)
    if final is None:
        raise RuntimeError("joint batch: final FIM evaluation failed")
    joint_fim, pieces = final
    criterion_value = _criterion_from_fim(joint_fim, criterion, region)

    return BatchDesignResult(
        designs=designs,
        fim_results=pieces,
        joint_fim=joint_fim,
        criterion_value=criterion_value,
        strategy=BatchStrategy.JOINT,
        per_round_criterion=None,
    )


def _penalized_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    region: "DesignRegion | None" = None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    min_distance: float | None,
    seed: int,
) -> BatchDesignResult:
    """Distance-penalized greedy: pick one at a time, reject picks too close."""
    min_dist = float(min_distance) if min_distance is not None else 0.0
    design_names = list(design_bounds.keys())

    running_prior = prior_fim.copy() if prior_fim is not None else None
    designs: list[dict[str, float]] = []
    fim_results: list[FIMResult] = []
    per_round: list[float] = []

    for i in range(n_experiments):
        candidates = _multi_start_candidates(
            design_bounds, n_starts, seed + i, projection=feasible_projection
        )
        filtered = _filter_by_min_distance(candidates, designs, design_bounds, min_dist)
        if not filtered:
            # Retry with a bigger pool before giving up.
            candidates = _multi_start_candidates(
                design_bounds, n_starts * 5, seed + 1000 + i, projection=feasible_projection
            )
            filtered = _filter_by_min_distance(candidates, designs, design_bounds, min_dist)
        if not filtered:
            raise RuntimeError(
                f"penalized batch: no candidate respects min_distance={min_dist} "
                f"from the {len(designs)} already-selected designs."
            )

        best, best_crit, best_fim = _scan_candidates(
            experiment, param_values, filtered, criterion, running_prior
        )

        if best is None or best_fim is None:
            raise RuntimeError("penalized batch: no feasible candidate found")

        if local_refine:
            refined = _refine_single_design(
                experiment,
                param_values,
                best,
                design_names,
                design_bounds,
                criterion,
                running_prior,
                equality_constraints=equality_constraints,
                inequality_constraints=inequality_constraints,
            )
            if (
                refined is not None
                and _min_normalized_distance(refined[0], designs, design_bounds) >= min_dist
                and _is_better(refined[1], best_crit, criterion)
            ):
                best, best_crit, best_fim = refined

        per_fim = compute_fim(experiment, param_values, best, prior_fim=None)
        designs.append(best)
        fim_results.append(per_fim)
        running_prior = per_fim.fim.copy() if running_prior is None else running_prior + per_fim.fim
        per_round.append(_criterion_from_fim(running_prior, criterion, region))

    assert running_prior is not None
    return BatchDesignResult(
        designs=designs,
        fim_results=fim_results,
        joint_fim=running_prior,
        criterion_value=per_round[-1],
        strategy=BatchStrategy.PENALIZED,
        per_round_criterion=per_round,
    )


def _normalized_distance(
    a: dict[str, float],
    b: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
) -> float:
    """Euclidean distance between two designs after normalising each axis to [0, 1]."""
    total = 0.0
    for name, (lo, hi) in design_bounds.items():
        scale = hi - lo
        if scale <= 0:
            continue
        total += ((a[name] - b[name]) / scale) ** 2
    return float(np.sqrt(total))


def _min_normalized_distance(
    candidate: dict[str, float],
    existing: list[dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
) -> float:
    """Minimum normalised distance from candidate to any existing design."""
    if not existing:
        return float("inf")
    return min(_normalized_distance(candidate, e, design_bounds) for e in existing)


def _filter_by_min_distance(
    candidates: list[dict[str, float]],
    existing: list[dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    min_dist: float,
) -> list[dict[str, float]]:
    """Drop candidates within min_dist (normalised) of any existing design."""
    if min_dist <= 0.0 or not existing:
        return list(candidates)
    return [
        c for c in candidates if _min_normalized_distance(c, existing, design_bounds) >= min_dist
    ]
