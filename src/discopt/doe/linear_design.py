"""Optimal design for models that are linear in their parameters.

Every built-in template in :mod:`discopt.doe.templates` — ``linear``,
``polynomial-1d``, ``response-surface-2d``/``-3d``, and the three ``scheffe-*``
mixture models — has the form

    y(x, θ) = f(x)ᵀ θ

for a basis-function vector ``f(x)`` that does not involve θ. The sensitivity
Jacobian ∂y/∂θ is therefore exactly ``f(x)``, independent of the parameter
values, and the Fisher information collapses to

    FIM = Xᵀ X / σ²        where  X = [f(x₁); …; f(x_n)]

with no differentiation of any kind. That matters for two reasons:

* **It is exact.** This is not an approximation of what
  :func:`~discopt.doe.design.optimal_experiment` computes by autodiff — it is
  the same matrix, reached in closed form. ``tests/test_linear_design.py``
  pins the two together to machine precision.
* **It has no heavy dependencies.** The autodiff path needs jax, which has no
  WebAssembly build, so it cannot run in a browser. This module needs only
  numpy and ``scipy.optimize``, both of which do.

The design *is* still a numerical optimization — the criterion is a nonlinear
function of the design point — so ``scipy.optimize.minimize`` does the same
multi-start search that :mod:`discopt.doe.design` runs. Only the FIM
evaluation inside the loop differs, and it is far cheaper here.

For models that are *not* linear in their parameters (anything built through
``discopt doe new module``), use :func:`~discopt.doe.design.optimal_experiment`
instead; this module will refuse an unknown template rather than silently
computing the wrong Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

# Criterion names. These are the same string values as
# ``discopt.doe.design.DesignCriterion`` — deliberately duplicated rather than
# imported, because that module reaches jax through discopt.doe.fim and this
# one must stay importable without it. The two are pinned together in
# tests/test_linear_design.py.
D_OPTIMAL = "determinant"
A_OPTIMAL = "trace"
E_OPTIMAL = "min_eigenvalue"
ME_OPTIMAL = "condition_number"

CRITERIA = (D_OPTIMAL, A_OPTIMAL, E_OPTIMAL, ME_OPTIMAL)

# Criteria where a larger value is a better design.
_MAXIMIZED = frozenset({D_OPTIMAL, E_OPTIMAL})

# Returned by the objective when the FIM is singular or the criterion is
# non-finite, so a failed evaluation loses to every real one without letting
# NaN propagate into the optimizer.
_SINGULAR_SENTINEL = 1e12

# Templates whose response is linear in the parameters, and thus whose
# Jacobian is the basis row returned by `design_row`.
LINEAR_TEMPLATES = frozenset(
    {
        "linear",
        "polynomial-1d",
        "response-surface-2d",
        "response-surface-3d",
        "scheffe-linear",
        "scheffe-quadratic",
        "scheffe-special-cubic",
        # The classical generators are design recipes, not models: each one
        # records the basis it is meant to be analysed with in
        # ``template_args["basis"]``.
        "latin-hypercube",
        "central-composite",
        "box-behnken",
    }
)

# Templates whose fitted model comes from template_args["basis"].
CLASSICAL_TEMPLATES = frozenset({"latin-hypercube", "central-composite", "box-behnken"})

# Bases a classical design can be analysed with.
BASES = ("linear", "quadratic")


def basis_parameter_names(basis: str, n_inputs: int) -> list[str]:
    """Return the coefficient names for a k-factor ``basis``.

    ``"linear"`` gives ``b0`` plus one main effect per factor. ``"quadratic"``
    gives the full quadratic — intercept, main effects, pure squares, then
    two-factor interactions — generalising ``response-surface-2d``/``-3d`` to
    any factor count, which is what lets a 5-factor Box-Behnken design be
    fitted at all.
    """
    if basis == "linear":
        return ["b0"] + [f"b{i + 1}" for i in range(n_inputs)]
    if basis == "quadratic":
        main = [f"b{i + 1}" for i in range(n_inputs)]
        squares = [f"b{i + 1}{i + 1}" for i in range(n_inputs)]
        cross = [f"b{i + 1}{j + 1}" for i in range(n_inputs) for j in range(i + 1, n_inputs)]
        return ["b0", *main, *squares, *cross]
    raise ValueError(f"unknown basis {basis!r}; expected one of {list(BASES)}")


def design_row(
    template: str,
    template_args: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    row: Mapping[str, Any],
) -> np.ndarray:
    """Return the design-matrix row ``f(x)`` for one run.

    The entries are the basis functions multiplying each parameter,
    evaluated at this run's inputs, ordered to match ``parameter_names``.
    Because the built-in templates are linear in their parameters, this row
    is simultaneously the regression basis used to fit them and the
    sensitivity Jacobian ∂y/∂θ used to design them.

    Parameters
    ----------
    template : str
        Template name; must be one of :data:`LINEAR_TEMPLATES`.
    template_args : mapping
        Template metadata. Only ``degree`` is read, for ``polynomial-1d``.
    parameter_names : sequence of str
        Ordered parameter names, used to infer the polynomial degree when
        ``template_args`` omits it.
    input_names : sequence of str
        Ordered factor names.
    row : mapping
        Factor values for this run, keyed by name.

    Returns
    -------
    numpy.ndarray
        Basis vector of length ``len(parameter_names)``.
    """
    xs = {name: float(row[name]) for name in input_names}

    if template in CLASSICAL_TEMPLATES:
        # A design recipe, not a model: the analysis basis travels in the
        # workbook metadata alongside the generator's own options.
        basis = str(template_args.get("basis", "quadratic"))
        if basis == "linear":
            template = "linear"
        elif basis == "quadratic":
            vals = [xs[n] for n in input_names]
            k = len(input_names)
            cross = [vals[i] * vals[j] for i in range(k) for j in range(i + 1, k)]
            return np.array([1.0, *vals, *[v * v for v in vals], *cross], dtype=np.float64)
        else:
            raise ValueError(f"unknown basis {basis!r}; expected one of {list(BASES)}")

    if template == "linear":
        return np.array([1.0] + [xs[n] for n in input_names], dtype=np.float64)

    if template == "polynomial-1d":
        x = xs[input_names[0]]
        degree = int(template_args.get("degree", len(parameter_names) - 1))
        return np.array([x**j for j in range(degree + 1)], dtype=np.float64)

    if template in ("response-surface-2d", "response-surface-3d"):
        n = len(input_names)
        vals = [xs[n_] for n_ in input_names]
        cross = [vals[i] * vals[j] for i in range(n) for j in range(i + 1, n)]
        return np.array([1.0, *vals, *[v * v for v in vals], *cross], dtype=np.float64)

    if template in ("scheffe-linear", "scheffe-quadratic", "scheffe-special-cubic"):
        vals = [xs[n_] for n_ in input_names]
        q = len(input_names)
        terms: list[float] = list(vals)
        if template == "scheffe-linear":
            return np.array(terms, dtype=np.float64)
        terms.extend(vals[i] * vals[j] for i in range(q) for j in range(i + 1, q))
        if template == "scheffe-quadratic":
            return np.array(terms, dtype=np.float64)
        terms.extend(
            vals[i] * vals[j] * vals[k]
            for i in range(q)
            for j in range(i + 1, q)
            for k in range(j + 1, q)
        )
        return np.array(terms, dtype=np.float64)

    raise ValueError(
        f"unknown template {template!r}; linear-design supports {sorted(LINEAR_TEMPLATES)}"
    )


def design_matrix(
    template: str,
    template_args: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Stack :func:`design_row` over ``rows`` into an ``(n_runs, n_params)`` matrix."""
    if not rows:
        return np.zeros((0, len(parameter_names)), dtype=np.float64)
    return np.array(
        [design_row(template, template_args, parameter_names, input_names, r) for r in rows],
        dtype=np.float64,
    )


def linear_fim(X: np.ndarray, sigma: float) -> np.ndarray:
    """Return ``Xᵀ X / σ²``, the Fisher information of a linear model.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix, shape ``(n_runs, n_params)``.
    sigma : float
        Measurement-error standard deviation; must be positive. A zero would
        make the information infinite, which silently poisons every criterion.
    """
    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"measurement error must be positive and finite, got {sigma!r}")
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"design matrix must be 2-D, got shape {Xa.shape}")
    return Xa.T @ Xa / (s * s)


def evaluate_criterion(fim: np.ndarray, criterion: str) -> float:
    """Evaluate a design criterion on a FIM.

    Matches :func:`discopt.doe.design._evaluate_criterion` term for term:
    D is ``log det`` (via ``slogdet``, which stays finite for badly scaled
    FIMs where ``det`` would overflow), A is ``trace(FIM⁻¹)``, E is the
    minimum eigenvalue, and ME is the condition number.
    """
    if criterion == D_OPTIMAL:
        sign, logdet = np.linalg.slogdet(fim)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        return float(logdet)
    if criterion == A_OPTIMAL:
        try:
            return float(np.trace(np.linalg.inv(fim)))
        except np.linalg.LinAlgError:
            return np.inf
    if criterion == E_OPTIMAL:
        return float(np.min(np.linalg.eigvalsh(fim)))
    if criterion == ME_OPTIMAL:
        return float(np.linalg.cond(fim))
    raise ValueError(f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}")


def is_maximized(criterion: str) -> bool:
    """Return True when a larger criterion value means a better design."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}")
    return criterion in _MAXIMIZED


def _metrics(fim: np.ndarray) -> dict[str, float]:
    return {
        "log_det_fim": evaluate_criterion(fim, D_OPTIMAL),
        "trace_fim_inv": evaluate_criterion(fim, A_OPTIMAL),
        "min_eigenvalue": evaluate_criterion(fim, E_OPTIMAL),
        "condition_number": evaluate_criterion(fim, ME_OPTIMAL),
    }


@dataclass
class LinearDesignResult:
    """One optimal design point for a linear-in-parameters model.

    Attribute names mirror :class:`discopt.doe.design.DesignResult` so the two
    are interchangeable at the call site, but this one carries a plain FIM
    array instead of a ``FIMResult`` (that type lives in the jax-bound module).
    """

    design: dict[str, float]
    fim: np.ndarray
    criterion_value: float
    criterion: str
    parameter_names: list[str] = field(default_factory=list)

    @property
    def parameter_covariance(self) -> np.ndarray:
        """Predicted parameter covariance if this experiment is run."""
        try:
            return np.linalg.inv(self.fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        """Predicted standard error for each parameter."""
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics at this design."""
        return _metrics(self.fim)

    def summary(self) -> str:
        """Human-readable summary of the optimal design."""
        lines = ["Optimal Experimental Design (linear model)", "=" * 50]
        for name, val in self.design.items():
            lines.append(f"  {name:>15s} = {val:.6g}")
        lines.append("")
        m = self.metrics
        lines.append(f"  D-opt (log det FIM)  = {m['log_det_fim']:.4g}")
        lines.append(f"  A-opt (trace FIM⁻¹)  = {m['trace_fim_inv']:.4g}")
        lines.append(f"  E-opt (min eigenval) = {m['min_eigenvalue']:.4g}")
        lines.append(f"  Condition number     = {m['condition_number']:.4g}")
        if self.parameter_names:
            lines.append("")
            se = self.predicted_standard_errors
            for i, name in enumerate(self.parameter_names):
                lines.append(f"  SE({name}) = {se[i]:.4g}")
        return "\n".join(lines)


@dataclass
class LinearBatchDesignResult:
    """A batch of optimal design points for a linear-in-parameters model."""

    designs: list[dict[str, float]]
    joint_fim: np.ndarray
    criterion_value: float
    criterion: str
    per_round_criterion: list[float]
    parameter_names: list[str] = field(default_factory=list)

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
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        return _metrics(self.joint_fim)


def _basis_evaluator(
    template: str,
    template_args: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(x_vector) -> basis row``, closing over the template metadata."""
    names = list(input_names)

    def basis(x: np.ndarray) -> np.ndarray:
        row = {n: float(v) for n, v in zip(names, x)}
        return design_row(template, template_args, parameter_names, names, row)

    return basis


def _search_one_point(
    basis: Callable[[np.ndarray], np.ndarray],
    accumulated: np.ndarray,
    sigma: float,
    lbs: np.ndarray,
    ubs: np.ndarray,
    criterion: str,
    n_starts: int,
    rng: np.random.Generator,
    input_names: Sequence[str],
    equality_constraints: Sequence[Callable[[dict[str, float]], float]],
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]],
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None,
) -> tuple[np.ndarray, float] | None:
    """Multi-start search for the point that best improves ``accumulated``."""
    maximize = is_maximized(criterion)
    inv_var = 1.0 / (float(sigma) ** 2)

    def to_design(x: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(input_names, x)}

    def project(x: np.ndarray) -> np.ndarray:
        if feasible_projection is None:
            return x
        projected = feasible_projection(to_design(x))
        return np.array([projected[n] for n in input_names], dtype=np.float64)

    def objective(x: np.ndarray) -> float:
        try:
            f = basis(x)
            # Rank-1 update: one more run adds f fᵀ/σ² to the information.
            trial = accumulated + np.outer(f, f) * inv_var
            value = evaluate_criterion(trial, criterion)
        except Exception:
            return _SINGULAR_SENTINEL
        if not np.isfinite(value):
            return _SINGULAR_SENTINEL
        return -value if maximize else value

    constraints = [
        {"type": "eq", "fun": (lambda x, g=g: float(g(to_design(x))))} for g in equality_constraints
    ] + [
        {"type": "ineq", "fun": (lambda x, h=h: float(h(to_design(x))))}
        for h in inequality_constraints
    ]
    bounds = list(zip(lbs, ubs))

    best_x: np.ndarray | None = None
    best_obj = np.inf

    for _ in range(max(1, int(n_starts))):
        x0 = project(rng.uniform(lbs, ubs))
        try:
            if constraints:
                res = minimize(
                    objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
                )
            else:
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        x = project(np.clip(np.asarray(res.x, dtype=np.float64), lbs, ubs))
        value = objective(x)
        if np.isfinite(value) and value < best_obj:
            best_obj = value
            best_x = x

    if best_x is None or best_obj >= _SINGULAR_SENTINEL:
        return None
    return best_x, (-best_obj if maximize else best_obj)


def linear_optimal_design(
    template: str,
    *,
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    design_bounds: Mapping[str, tuple[float, float]],
    template_args: Mapping[str, Any] | None = None,
    measurement_error: float = 1.0,
    criterion: str = D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    seed: int = 42,
) -> LinearDesignResult:
    """Find the single most informative next experiment.

    Parameters
    ----------
    template : str
        One of :data:`LINEAR_TEMPLATES`.
    parameter_names, input_names : sequence of str
        Ordered parameter and factor names. ``parameter_names`` fixes the
        FIM row/column order.
    design_bounds : mapping name -> (lb, ub)
        Feasible range per factor.
    template_args : mapping, optional
        Template metadata; ``degree`` is required for ``polynomial-1d``.
    measurement_error : float, default 1.0
        Response standard deviation σ.
    criterion : str, default ``"determinant"``
        One of :data:`CRITERIA`.
    prior_fim : numpy.ndarray, optional
        Information already in hand, from completed runs. **Usually
        required**: a single run contributes a rank-1 matrix, so with more
        than one parameter the D-criterion of one experiment alone is
        ``-inf`` regardless of where you put it. Pass the FIM of the runs so
        far, or use :func:`linear_batch_design` to design several at once.
    equality_constraints, inequality_constraints : sequence of callable, optional
        Constraints on the design dict, ``g(d) == 0`` and ``h(d) >= 0``.
        Their presence switches the refiner to SLSQP.
    feasible_projection : callable, optional
        Maps an arbitrary design dict onto the feasible set — for mixtures,
        :func:`~discopt.doe.design.project_to_simplex`.
    n_starts : int, default 10
        Random restarts. The criterion surface is multi-modal, so more starts
        buy robustness at linear cost.
    seed : int, default 42
        Seed for the restart points.

    Returns
    -------
    LinearDesignResult
        ``fim`` is the total information, ``prior_fim`` included.
    """
    if template not in LINEAR_TEMPLATES:
        raise ValueError(
            f"{template!r} is not linear in its parameters; use "
            "discopt.doe.design.optimal_experiment (which needs jax) instead. "
            f"Linear templates: {sorted(LINEAR_TEMPLATES)}"
        )
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}")

    names = list(input_names)
    missing = [n for n in names if n not in design_bounds]
    if missing:
        raise ValueError(f"design_bounds missing entries for {missing}")
    lbs = np.array([float(design_bounds[n][0]) for n in names], dtype=np.float64)
    ubs = np.array([float(design_bounds[n][1]) for n in names], dtype=np.float64)
    if np.any(ubs <= lbs):
        bad = [n for n, lo, hi in zip(names, lbs, ubs) if hi <= lo]
        raise ValueError(f"design_bounds must satisfy ub > lb; offending factors: {bad}")

    n_p = len(parameter_names)
    accumulated = (
        np.zeros((n_p, n_p), dtype=np.float64)
        if prior_fim is None
        else np.asarray(prior_fim, dtype=np.float64).copy()
    )
    if accumulated.shape != (n_p, n_p):
        raise ValueError(
            f"prior_fim shape {accumulated.shape} does not match "
            f"{n_p} parameters {list(parameter_names)}"
        )

    # A rank-deficient result must be rejected up front rather than at the end:
    # one run adds a rank-1 matrix, so with p parameters and no prior the total
    # can never exceed rank 1. Left alone, slogdet happily reports a finite
    # log-determinant for such a matrix (its determinant underflows to ~1e-81
    # rather than to exactly zero), so the search would return a confident
    # number for a design that cannot estimate anything.
    prior_rank = int(np.linalg.matrix_rank(accumulated)) if accumulated.any() else 0
    if prior_rank + 1 < n_p:
        raise RuntimeError(
            f"a single experiment adds rank 1 to a prior of rank {prior_rank}, which "
            f"cannot reach the {n_p} parameters {list(parameter_names)}: the FIM is "
            "singular everywhere in the design box, and every criterion is degenerate. "
            "Supply a prior_fim from completed runs, or design a batch of at least "
            f"{n_p - prior_rank} experiments with linear_batch_design."
        )

    basis = _basis_evaluator(template, template_args or {}, parameter_names, names)
    found = _search_one_point(
        basis,
        accumulated,
        measurement_error,
        lbs,
        ubs,
        criterion,
        n_starts,
        np.random.default_rng(seed),
        names,
        tuple(equality_constraints or ()),
        tuple(inequality_constraints or ()),
        feasible_projection,
    )
    if found is None:
        raise RuntimeError(
            "optimal design search failed to find a finite criterion value within "
            "the design bounds; check the bounds and any constraints."
        )

    x, value = found
    f = basis(x)
    total = accumulated + np.outer(f, f) / (float(measurement_error) ** 2)
    return LinearDesignResult(
        design={n: float(v) for n, v in zip(names, x)},
        fim=total,
        criterion_value=float(value),
        criterion=criterion,
        parameter_names=list(parameter_names),
    )


def linear_batch_design(
    template: str,
    n_experiments: int,
    *,
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    design_bounds: Mapping[str, tuple[float, float]],
    template_args: Mapping[str, Any] | None = None,
    measurement_error: float = 1.0,
    criterion: str = D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    seed: int = 42,
) -> LinearBatchDesignResult:
    """Design ``n_experiments`` runs greedily, accumulating information.

    Experiments are independent, so their information adds:
    ``FIM_total = Σᵢ f(xᵢ) f(xᵢ)ᵀ / σ² + prior``. Each round picks the point
    that most improves the criterion given everything chosen so far, which is
    what makes a batch escape the rank-1 degeneracy of a single run.

    Parameters are as :func:`linear_optimal_design`, plus:

    Parameters
    ----------
    n_experiments : int
        Number of runs to design. To make the joint FIM non-singular from
        scratch this must be at least ``len(parameter_names)``.

    Returns
    -------
    LinearBatchDesignResult
        ``per_round_criterion`` records the criterion after each pick, which
        shows how fast the batch is buying information.

    Notes
    -----
    The first pick of a from-scratch D-optimal batch is degenerate — every
    point gives ``log det = -inf`` — so early rounds fall back to maximizing
    the smallest non-zero information direction (E-criterion) until the
    accumulated FIM is full rank, then revert to the requested criterion.
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")
    if template not in LINEAR_TEMPLATES:
        raise ValueError(
            f"{template!r} is not linear in its parameters; use "
            "discopt.doe.design.batch_optimal_experiment (which needs jax) instead. "
            f"Linear templates: {sorted(LINEAR_TEMPLATES)}"
        )
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}")

    names = list(input_names)
    missing = [n for n in names if n not in design_bounds]
    if missing:
        raise ValueError(f"design_bounds missing entries for {missing}")
    lbs = np.array([float(design_bounds[n][0]) for n in names], dtype=np.float64)
    ubs = np.array([float(design_bounds[n][1]) for n in names], dtype=np.float64)
    if np.any(ubs <= lbs):
        bad = [n for n, lo, hi in zip(names, lbs, ubs) if hi <= lo]
        raise ValueError(f"design_bounds must satisfy ub > lb; offending factors: {bad}")

    n_p = len(parameter_names)
    accumulated = (
        np.zeros((n_p, n_p), dtype=np.float64)
        if prior_fim is None
        else np.asarray(prior_fim, dtype=np.float64).copy()
    )
    if accumulated.shape != (n_p, n_p):
        raise ValueError(
            f"prior_fim shape {accumulated.shape} does not match "
            f"{n_p} parameters {list(parameter_names)}"
        )

    basis = _basis_evaluator(template, template_args or {}, parameter_names, names)
    return batch_design_from_basis(
        basis,
        n_experiments,
        parameter_names=parameter_names,
        input_names=names,
        design_bounds=design_bounds,
        measurement_error=measurement_error,
        criterion=criterion,
        prior_fim=accumulated,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        feasible_projection=feasible_projection,
        n_starts=n_starts,
        seed=seed,
    )


def batch_design_from_basis(
    basis: Callable[[np.ndarray], np.ndarray],
    n_experiments: int,
    *,
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    design_bounds: Mapping[str, tuple[float, float]],
    measurement_error: float = 1.0,
    criterion: str = D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    seed: int = 42,
) -> LinearBatchDesignResult:
    """Greedy batch design driven by an arbitrary Jacobian-row provider.

    The search never needs to know *where* the row ``∂y/∂θ`` came from — only
    its value at a candidate point. Splitting that out is what lets a
    user-defined nonlinear model reuse this machinery unchanged: the linear
    templates supply a basis-function row, and
    :func:`discopt.doe.symbolic.basis_evaluator` supplies one differentiated by
    sympy at fixed nominal parameters.

    Parameters
    ----------
    basis : callable
        ``f(x_vector) -> ndarray`` of length ``len(parameter_names)``, where
        ``x_vector`` is ordered by ``input_names``.

    Other parameters are as :func:`linear_batch_design`.
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}")

    names = list(input_names)
    missing = [n for n in names if n not in design_bounds]
    if missing:
        raise ValueError(f"design_bounds missing entries for {missing}")
    lbs = np.array([float(design_bounds[n][0]) for n in names], dtype=np.float64)
    ubs = np.array([float(design_bounds[n][1]) for n in names], dtype=np.float64)
    if np.any(ubs <= lbs):
        bad = [n for n, lo, hi in zip(names, lbs, ubs) if hi <= lo]
        raise ValueError(f"design_bounds must satisfy ub > lb; offending factors: {bad}")

    n_p = len(parameter_names)
    accumulated = (
        np.zeros((n_p, n_p), dtype=np.float64)
        if prior_fim is None
        else np.asarray(prior_fim, dtype=np.float64).copy()
    )
    if accumulated.shape != (n_p, n_p):
        raise ValueError(
            f"prior_fim shape {accumulated.shape} does not match "
            f"{n_p} parameters {list(parameter_names)}"
        )

    rng = np.random.default_rng(seed)
    inv_var = 1.0 / (float(measurement_error) ** 2)

    designs: list[dict[str, float]] = []
    per_round: list[float] = []

    for _ in range(int(n_experiments)):
        # While the accumulated FIM is rank-deficient the requested criterion
        # is degenerate for every candidate (log det = -inf, trace of a
        # singular inverse = inf). Grow the weakest direction instead, which
        # is exactly what fills in the missing rank.
        rank = int(np.linalg.matrix_rank(accumulated)) if accumulated.any() else 0
        active = criterion if rank >= n_p else E_OPTIMAL

        found = _search_one_point(
            basis,
            accumulated,
            measurement_error,
            lbs,
            ubs,
            active,
            n_starts,
            rng,
            names,
            tuple(equality_constraints or ()),
            tuple(inequality_constraints or ()),
            feasible_projection,
        )
        if found is None:
            raise RuntimeError(
                "batch design search failed to find a finite criterion value "
                f"on round {len(designs) + 1}; check the design bounds and constraints."
            )
        x, _ = found
        f = basis(x)
        accumulated = accumulated + np.outer(f, f) * inv_var
        designs.append({n: float(v) for n, v in zip(names, x)})
        # Report the degenerate value while the FIM is still rank-deficient.
        # evaluate_criterion would otherwise hand back a finite-looking log-det
        # (the determinant underflows rather than reaching exactly zero), which
        # reads as a real score for a design that identifies nothing yet.
        if int(np.linalg.matrix_rank(accumulated)) >= n_p:
            per_round.append(evaluate_criterion(accumulated, criterion))
        else:
            per_round.append(-np.inf if is_maximized(criterion) else np.inf)

    return LinearBatchDesignResult(
        designs=designs,
        joint_fim=accumulated,
        criterion_value=per_round[-1],
        criterion=criterion,
        per_round_criterion=per_round,
        parameter_names=list(parameter_names),
    )


__all__ = [
    "A_OPTIMAL",
    "CRITERIA",
    "D_OPTIMAL",
    "E_OPTIMAL",
    "LINEAR_TEMPLATES",
    "LinearBatchDesignResult",
    "LinearDesignResult",
    "ME_OPTIMAL",
    "batch_design_from_basis",
    "design_matrix",
    "design_row",
    "evaluate_criterion",
    "is_maximized",
    "linear_batch_design",
    "linear_fim",
    "linear_optimal_design",
]
