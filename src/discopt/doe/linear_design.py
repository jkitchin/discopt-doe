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
from functools import cached_property
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

# Prediction-oriented criteria. Unlike the four above they are not functions of
# the FIM alone: they also need the region over which predictions matter (a
# :class:`DesignRegion`). I (also called V or IV) is the average variance of the
# fitted response over the region, G its maximum; both are minimized.
I_OPTIMAL = "average_prediction_variance"
G_OPTIMAL = "max_prediction_variance"
REGION_CRITERIA = (I_OPTIMAL, G_OPTIMAL)
ALL_CRITERIA = CRITERIA + REGION_CRITERIA

# Short names accepted wherever a criterion is: the textbook letters.
CRITERION_ALIASES = {
    "D": D_OPTIMAL,
    "A": A_OPTIMAL,
    "E": E_OPTIMAL,
    "ME": ME_OPTIMAL,
    "I": I_OPTIMAL,
    "V": I_OPTIMAL,
    "IV": I_OPTIMAL,
    "G": G_OPTIMAL,
}

# Criteria where a larger value is a better design.
_MAXIMIZED = frozenset({D_OPTIMAL, E_OPTIMAL})

# Returned by the objective when the FIM is singular or the criterion is
# non-finite, so a failed evaluation loses to every real one without letting
# NaN propagate into the optimizer.
_SINGULAR_SENTINEL = 1e12

# Optimal-design criteria are flat near their optimum, and the default
# tolerances stop well short of it (a D-optimal quadratic centre point landed at
# 4.86 instead of 5.0). Tight tolerances cost a few extra iterations per start.
_LBFGSB_OPTIONS = {"ftol": 1e-13, "gtol": 1e-10, "maxiter": 2000}
_SLSQP_OPTIONS = {"ftol": 1e-13, "maxiter": 500}

# Relative size of the ridge used while the accumulated FIM is rank-deficient:
# small enough never to outvote real information, large enough to make the
# criterion non-degenerate.
_RANK_RIDGE = 1e-8

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
        "definitive-screening",
    }
)

# Templates whose fitted model comes from template_args["basis"].
CLASSICAL_TEMPLATES = frozenset(
    {"latin-hypercube", "central-composite", "box-behnken", "definitive-screening"}
)

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


def basis_terms(
    template: str,
    template_args: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
) -> list[dict[str, int]]:
    """Return the basis function multiplying each parameter, as factor powers.

    Every basis here is a monomial in the factors, so a term is fully described
    by the power of each factor in it: ``{}`` is the intercept, ``{"T": 1}`` is
    a main effect, ``{"T": 2}`` a pure square, ``{"T": 1, "P": 1}`` a two-factor
    interaction. The i-th entry is the term :func:`design_row` evaluates into
    column i, so the two must be read together — a change to one is wrong
    without the same change to the other, and
    ``tests/test_linear_design.py::test_basis_terms_reproduce_the_design_row``
    fails if they disagree.

    This is what lets a caller *write the model down* — ``y = b0 + b1·T +
    b11·T²`` — from workbook metadata alone, without a symbolic layer. The
    arguments are exactly :func:`design_row`'s, minus the run.

    Returns
    -------
    list of dict
        One mapping of factor name to positive integer power per parameter, in
        ``parameter_names`` order.
    """
    names = list(input_names)

    if template in CLASSICAL_TEMPLATES:
        # A design recipe, not a model: the basis travels in the metadata.
        basis = str(template_args.get("basis", "quadratic"))
        if basis == "linear":
            template = "linear"
        elif basis == "quadratic":
            return [{}, *_main_terms(names), *_square_terms(names), *_cross_terms(names)]
        else:
            raise ValueError(f"unknown basis {basis!r}; expected one of {list(BASES)}")

    if template == "linear":
        return [{}, *_main_terms(names)]

    if template == "polynomial-1d":
        x = names[0]
        degree = int(template_args.get("degree", len(parameter_names) - 1))
        return [{} if j == 0 else {x: j} for j in range(degree + 1)]

    if template in ("response-surface-2d", "response-surface-3d"):
        return [{}, *_main_terms(names), *_square_terms(names), *_cross_terms(names)]

    if template in ("scheffe-linear", "scheffe-quadratic", "scheffe-special-cubic"):
        # Mixture models have no intercept: the factors sum to a constant, so
        # one would be perfectly collinear with the main effects.
        terms = _main_terms(names)
        if template == "scheffe-linear":
            return terms
        terms += _cross_terms(names)
        if template == "scheffe-quadratic":
            return terms
        q = len(names)
        terms += [
            {names[i]: 1, names[j]: 1, names[k]: 1}
            for i in range(q)
            for j in range(i + 1, q)
            for k in range(j + 1, q)
        ]
        return terms

    raise ValueError(
        f"unknown template {template!r}; linear-design supports {sorted(LINEAR_TEMPLATES)}"
    )


def _main_terms(names: Sequence[str]) -> list[dict[str, int]]:
    return [{n: 1} for n in names]


def _square_terms(names: Sequence[str]) -> list[dict[str, int]]:
    return [{n: 2} for n in names]


def _cross_terms(names: Sequence[str]) -> list[dict[str, int]]:
    k = len(names)
    return [{names[i]: 1, names[j]: 1} for i in range(k) for j in range(i + 1, k)]


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


def normalize_criterion(criterion: str) -> str:
    """Map a criterion name or textbook letter (``"D"``, ``"I"``, ...) to its canonical name.

    Raises ``ValueError`` for an unknown name.
    """
    name = str(criterion)
    name = CRITERION_ALIASES.get(name, CRITERION_ALIASES.get(name.upper(), name))
    if name not in ALL_CRITERIA:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of {list(ALL_CRITERIA)} "
            f"or {sorted(CRITERION_ALIASES)}"
        )
    return name


def trace_inverse(fim: np.ndarray) -> float:
    """A-optimality ``trace(FIM^{-1})``, or ``inf`` unless the FIM is positive definite.

    ``np.linalg.inv`` raises only for an *exactly* singular matrix. For a
    numerically singular FIM (rank-deficient up to round-off, e.g. two sampling
    times at the same point) it returns garbage whose trace is frequently a huge
    *negative* number, and a minimizing search then prefers exactly the designs
    that cannot estimate the parameters. Going through a Cholesky factor refuses
    any matrix that is not positive definite, and computes the trace stably as
    ``||L^{-1}||_F^2`` when it is.
    """
    fim = np.asarray(fim, dtype=float)
    if not np.all(np.isfinite(fim)):
        return float("inf")
    try:
        L = np.linalg.cholesky(0.5 * (fim + fim.T))
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        return float("inf")
    value = float(np.sum(L_inv**2))
    return value if np.isfinite(value) else float("inf")


def evaluate_criterion(
    fim: np.ndarray, criterion: str, *, region: DesignRegion | None = None
) -> float:
    """Evaluate a design criterion on a FIM.

    Matches :func:`discopt.doe.design._evaluate_criterion` term for term:
    D is ``log det`` (via ``slogdet``, which stays finite for badly scaled
    FIMs where ``det`` would overflow), A is ``trace(FIM⁻¹)``, E is the
    minimum eigenvalue, and ME is the condition number.

    The prediction criteria need ``region``: I is the average of
    ``f(x)ᵀ FIM⁻¹ f(x)`` over the region, ``trace(FIM⁻¹ W)`` with ``W`` the
    region's moment matrix, and G is its maximum over the region's grid.
    Because the FIM carries ``1/σ²``, both are variances of the fitted
    response in the response's own units. A singular FIM scores ``inf``.
    """
    criterion = normalize_criterion(criterion)
    if criterion == D_OPTIMAL:
        sign, logdet = np.linalg.slogdet(fim)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        return float(logdet)
    if criterion == A_OPTIMAL:
        return trace_inverse(fim)
    if criterion == E_OPTIMAL:
        return float(np.min(np.linalg.eigvalsh(fim)))
    if criterion == ME_OPTIMAL:
        return float(np.linalg.cond(fim))
    if region is None:
        raise ValueError(
            f"criterion {criterion!r} is about prediction over a region; pass region= "
            "(see DesignRegion / design_region)"
        )
    if criterion == I_OPTIMAL:
        return region.average_variance(fim)
    return region.max_variance(fim)


def is_maximized(criterion: str) -> bool:
    """Return True when a larger criterion value means a better design."""
    return normalize_criterion(criterion) in _MAXIMIZED


# --------------------------------------------------------------------------
# Regions for the prediction criteria (I and G)
# --------------------------------------------------------------------------


@dataclass
class DesignRegion:
    """Where predictions matter, reduced to what the I and G criteria need.

    Attributes
    ----------
    moment_rows : numpy.ndarray
        ``(m, p)`` model rows ``f(x)`` at the integration points.
    weights : numpy.ndarray
        ``(m,)`` integration weights, summing to 1.
    grid_rows : numpy.ndarray
        ``(g, p)`` model rows on the grid over which G takes its maximum.
    points, grid_points : numpy.ndarray
        The corresponding input points, ``(m, k)`` and ``(g, k)``.
    method : str
        How the region was discretized, e.g. ``"gauss-legendre"``, ``"sobol"``,
        ``"points"``.
    """

    moment_rows: np.ndarray
    weights: np.ndarray
    grid_rows: np.ndarray
    points: np.ndarray
    grid_points: np.ndarray
    method: str

    @cached_property
    def moment_matrix(self) -> np.ndarray:
        """``W = ∫ f(x) f(x)ᵀ dx / vol``: the region's average outer product.

        Cached: a region is built once and never mutated, while the I criterion
        reads this on every candidate evaluation inside the design search, so
        recomputing the product each time is pure overhead.
        """
        F = self.moment_rows
        return F.T @ (F * self.weights[:, None])

    def average_variance(self, fim: np.ndarray) -> float:
        """I-criterion: average of ``f(x)ᵀ FIM⁻¹ f(x)`` over the region."""
        try:
            val = float(np.trace(np.linalg.solve(fim, self.moment_matrix)))
        except np.linalg.LinAlgError:
            return np.inf
        return val if np.isfinite(val) and val >= 0 else np.inf

    def soft_max_variance(self, fim: np.ndarray, q: float = 32.0) -> float:
        """Smooth upper-envelope of the variances, ``(mean v^q)^(1/q)``.

        A differentiable stand-in for :meth:`max_variance` that the local
        optimizer can follow (the exact maximum is piecewise, and a gradient
        search stalls on its kinks). It approaches the maximum as ``q`` grows.
        """
        try:
            sol = np.linalg.solve(fim, self.grid_rows.T)
        except np.linalg.LinAlgError:
            return np.inf
        v = np.einsum("ij,ji->i", self.grid_rows, sol)
        if not np.all(np.isfinite(v)) or np.min(v) < 0:
            return np.inf
        vmax = float(np.max(v))
        if vmax <= 0:
            return 0.0
        return vmax * float(np.mean((v / vmax) ** q)) ** (1.0 / q)

    def max_variance(self, fim: np.ndarray) -> float:
        """G-criterion: maximum of ``f(x)ᵀ FIM⁻¹ f(x)`` over the region's grid."""
        try:
            sol = np.linalg.solve(fim, self.grid_rows.T)
        except np.linalg.LinAlgError:
            return np.inf
        v = np.einsum("ij,ji->i", self.grid_rows, sol)
        val = float(np.max(v))
        return val if np.isfinite(val) and np.min(v) > -1e-12 * max(1.0, val) else np.inf


# Gauss-Legendre nodes per dimension (tensor rule) and uniform grid points per
# dimension (for G), chosen to keep the total near a few thousand points.
_GL_NODES = {1: 24, 2: 16, 3: 10, 4: 7}
_GRID_POINTS = {1: 201, 2: 41, 3: 13, 4: 7}


def design_region(
    basis: Callable[[np.ndarray], np.ndarray],
    input_names: Sequence[str],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    points: Sequence[Mapping[str, float]] | np.ndarray | None = None,
    weights: Sequence[float] | None = None,
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_points: int = 4096,
    seed: int = 0,
) -> DesignRegion:
    """Discretize a region for the I and G criteria.

    * **Explicit points** (``points``, e.g. from
      :func:`discopt.doe.prediction.region_points` for a mixture simplex): used
      as-is, for both the average (equal ``weights`` unless given) and the
      maximum.
    * **A box** (``bounds``, no constraints) with up to 4 factors: the average
      uses a tensor Gauss-Legendre rule, exact for polynomial models of degree
      up to 13 per factor or more; the maximum is taken on a uniform tensor grid
      that includes the faces and corners, where polynomial prediction variance
      peaks. With 5 or more factors both use ``n_points`` scrambled Sobol points
      plus the corners.
    * **A constrained box** (``inequality_constraints`` or a
      ``feasible_projection``): ``n_points`` Sobol points in the box, kept if
      they satisfy every ``h(x) >= 0`` and mapped by the projection. A
      projection does not generally give a *uniform* sample of the feasible set,
      so for mixtures prefer explicit ``points``.

    G is a maximum over finitely many points, so it can only under-estimate the
    true maximum; the dense grids above keep the error small for low-order
    polynomial models, and an explicit fine grid gives full control.

    Parameters
    ----------
    basis : callable
        ``f(x_vector) -> model row``, ``x_vector`` ordered by ``input_names``.
    input_names : sequence of str
        Factor names.
    bounds : mapping name -> (lb, ub), optional
        The box. Required unless ``points`` is given.
    points, weights : optional
        Explicit region points (mappings or an ``(m, k)`` array) and weights.
    inequality_constraints, feasible_projection : optional
        Restrict the box, as in :func:`linear_batch_design`.
    n_points : int, default 4096
        Sample size for sampled regions.
    seed : int, default 0
        Seed for the Sobol scrambling.
    """
    names = list(input_names)
    k = len(names)

    def rows_of(X: np.ndarray) -> np.ndarray:
        return np.array([np.asarray(basis(x), dtype=np.float64) for x in X], dtype=np.float64)

    if points is not None:
        if isinstance(points, np.ndarray):
            X = np.atleast_2d(np.asarray(points, dtype=np.float64))
        else:
            X = np.array([[float(pt[n]) for n in names] for pt in points], dtype=np.float64)
        if X.shape[0] < 1 or X.shape[1] != k:
            raise ValueError(f"points must have shape (m, {k})")
        w = (
            np.full(X.shape[0], 1.0 / X.shape[0])
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if w.shape != (X.shape[0],) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per point, not all zero")
        F = rows_of(X)
        return DesignRegion(F, w / w.sum(), F, X, X, "points")

    if bounds is None:
        raise ValueError("design_region needs bounds= (a box) or points=")
    missing = [n for n in names if n not in bounds]
    if missing:
        raise ValueError(f"bounds missing entries for {missing}")
    lo = np.array([float(bounds[n][0]) for n in names])
    hi = np.array([float(bounds[n][1]) for n in names])
    if np.any(hi <= lo):
        raise ValueError("bounds must satisfy ub > lb")

    from itertools import product

    corners = np.array(list(product(*zip(lo, hi))), dtype=np.float64)
    constrained = bool(inequality_constraints) or feasible_projection is not None
    if not constrained and k in _GL_NODES:
        nodes, gw = np.polynomial.legendre.leggauss(_GL_NODES[k])
        grids = [lo[j] + (nodes + 1.0) * (hi[j] - lo[j]) / 2.0 for j in range(k)]
        X = np.array(list(product(*grids)), dtype=np.float64)
        w = np.array([np.prod(c) for c in product(*([gw] * k))], dtype=np.float64)
        G = np.array(
            list(product(*[np.linspace(lo[j], hi[j], _GRID_POINTS[k]) for j in range(k)])),
            dtype=np.float64,
        )
        F, FG = rows_of(X), rows_of(G)
        return DesignRegion(F, w / w.sum(), FG, X, G, "gauss-legendre")

    from scipy.stats import qmc

    m = 1 << max(4, int(np.ceil(np.log2(max(16, int(n_points))))))
    U = qmc.Sobol(d=k, scramble=True, seed=seed).random(m)
    X = lo + U * (hi - lo)
    if not constrained:
        X = np.vstack([X, corners])
    else:
        keep = []
        for x in X:
            d = dict(zip(names, map(float, x)))
            if feasible_projection is not None:
                d = feasible_projection(d)
            if all(float(h(d)) >= -1e-12 for h in (inequality_constraints or ())):
                keep.append([d[n] for n in names])
        if len(keep) < 2 * k + 2:
            raise ValueError(
                "too few sampled points satisfy the constraints to represent the region; "
                "pass explicit points="
            )
        X = np.array(keep, dtype=np.float64)
    F = rows_of(X)
    w = np.full(X.shape[0], 1.0 / X.shape[0])
    return DesignRegion(F, w, F, X, X, "sobol")


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
    regularization: float | np.ndarray = 0.0,
    region: DesignRegion | None = None,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Multi-start search for the point that best improves ``accumulated``.

    ``region`` is required for the I and G criteria. ``start``, when given, is
    tried as the first starting point (the exchange pass seeds it with the run
    being replaced, so a run already at the optimum is not lost to the random
    restarts).

    ``regularization`` adds a diagonal ridge (a scalar ``ε·I`` or a per-parameter
    vector) to every trial FIM before it is scored. It is only used while
    ``accumulated`` is rank-deficient, where the unregularized criterion is
    degenerate (the same ``-inf`` / ``inf`` for every candidate) and the choice
    of point would otherwise be arbitrary.
    """
    maximize = is_maximized(criterion)
    inv_var = 1.0 / (float(sigma) ** 2)
    reg = np.asarray(regularization, dtype=np.float64)
    if reg.ndim == 0:
        ridge = float(reg) * np.eye(accumulated.shape[0]) if float(reg) else 0.0
    else:
        ridge = np.diag(reg)

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
            trial = accumulated + np.outer(f, f) * inv_var + ridge
            if criterion == G_OPTIMAL and region is not None:
                # Follow a smooth envelope of the maximum; callers re-score exactly.
                value = region.soft_max_variance(trial)
            else:
                value = evaluate_criterion(trial, criterion, region=region)
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

    starts = max(1, int(n_starts))
    for i_start in range(starts + (start is not None)):
        if start is not None and i_start == 0:
            x0 = project(np.clip(np.asarray(start, dtype=np.float64), lbs, ubs))
        else:
            x0 = project(rng.uniform(lbs, ubs))
        try:
            if constraints:
                res = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options=_SLSQP_OPTIONS,
                )
            else:
                res = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds, options=_LBFGSB_OPTIONS
                )
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


def _resolve_region(
    region: DesignRegion | Sequence[Mapping[str, float]] | np.ndarray | None,
    criterion: str,
    basis: Callable[[np.ndarray], np.ndarray],
    names: Sequence[str],
    region_bounds: Mapping[str, tuple[float, float]] | None,
    design_bounds: Mapping[str, tuple[float, float]],
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None,
    seed: int,
) -> DesignRegion | None:
    """Build the region the I/G criteria need; ``None`` for the FIM criteria."""
    if criterion not in REGION_CRITERIA:
        return None
    if isinstance(region, DesignRegion):
        return region
    if region is not None:
        return design_region(basis, names, points=region)
    return design_region(
        basis,
        names,
        bounds=region_bounds if region_bounds is not None else design_bounds,
        inequality_constraints=inequality_constraints,
        feasible_projection=feasible_projection,
        seed=seed,
    )


def _joint_polish(
    points: list[np.ndarray],
    accumulated: np.ndarray,
    basis: Callable[[np.ndarray], np.ndarray],
    prior_fim: np.ndarray | None,
    n_p: int,
    inv_var: float,
    criterion: str,
    region: DesignRegion | None,
    lbs: np.ndarray,
    ubs: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Optimize every run's coordinates at once; keep the result only if it is better."""
    k = len(lbs)
    n = len(points)
    base = (
        np.zeros((n_p, n_p), dtype=np.float64)
        if prior_fim is None
        else np.asarray(prior_fim, dtype=np.float64)
    )

    def fim_of(z: np.ndarray) -> np.ndarray:
        F = np.array([basis(z[i * k : (i + 1) * k]) for i in range(n)], dtype=np.float64)
        return base + F.T @ F * inv_var

    def exact(z: np.ndarray) -> float:
        return evaluate_criterion(fim_of(z), criterion, region=region)

    z = np.concatenate(points)
    best_val = exact(z)
    bounds = [(lo, hi) for _ in range(n) for lo, hi in zip(lbs, ubs)]
    lo_all, hi_all = np.tile(lbs, n), np.tile(ubs, n)
    qs = (64.0, 512.0) if criterion == G_OPTIMAL else (None,)
    # Symmetric designs are often saddle points of I and G (the gradient
    # vanishes by symmetry), so also start from a few small random jitters.
    jitter_rng = np.random.default_rng(12345)
    starts = [z] + [
        np.clip(z + 0.05 * (hi_all - lo_all) * jitter_rng.standard_normal(z.size), lo_all, hi_all)
        for _ in range(1 if criterion == G_OPTIMAL else 4)
    ]
    for z_start, q in [(zs, q) for zs in starts for q in qs]:

        def smooth(zz: np.ndarray, q=q) -> float:
            M = fim_of(zz)
            if q is None:
                v = evaluate_criterion(M, criterion, region=region)
            else:
                assert region is not None
                v = region.soft_max_variance(M, q=q)
            return v if np.isfinite(v) else _SINGULAR_SENTINEL

        try:
            res = minimize(
                smooth, z_start, method="L-BFGS-B", bounds=bounds, options=_LBFGSB_OPTIONS
            )
        except Exception:  # noqa: BLE001 - a failed polish keeps the exchange result
            continue
        z_new = np.clip(np.asarray(res.x, dtype=np.float64), lo_all, hi_all)
        val = exact(z_new)
        if np.isfinite(val) and val < best_val - 1e-12 * max(1.0, abs(best_val)):
            z, best_val = z_new, val
    new_points = [z[i * k : (i + 1) * k].copy() for i in range(n)]
    return new_points, fim_of(z)


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
    region: DesignRegion | Sequence[Mapping[str, float]] | np.ndarray | None = None,
    region_bounds: Mapping[str, tuple[float, float]] | None = None,
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
        One of :data:`ALL_CRITERIA`, or a textbook letter (``"D"``, ``"A"``,
        ``"E"``, ``"ME"``, ``"I"``, ``"G"``). I and G are about the variance of
        predictions over a region rather than about the coefficients.
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
    region : DesignRegion, points, or None
        For the I and G criteria: the region over which prediction variance is
        averaged / maximized. A :class:`DesignRegion`, explicit points
        (mappings or an array), or ``None`` to use ``region_bounds`` (default:
        ``design_bounds``, restricted by any constraints). See
        :func:`design_region`.
    region_bounds : mapping, optional
        The box for the default region, when it differs from ``design_bounds``.

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
    criterion = normalize_criterion(criterion)

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
    region_obj = _resolve_region(
        region,
        criterion,
        basis,
        names,
        region_bounds,
        design_bounds,
        inequality_constraints,
        feasible_projection,
        seed,
    )
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
        region=region_obj,
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
    exchange_passes: int = 2,
    region: DesignRegion | Sequence[Mapping[str, float]] | np.ndarray | None = None,
    region_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> LinearBatchDesignResult:
    """Design ``n_experiments`` runs greedily, accumulating information.

    Experiments are independent, so their information adds:
    ``FIM_total = Σᵢ f(xᵢ) f(xᵢ)ᵀ / σ² + prior``. Each round picks the point
    that most improves the criterion given everything chosen so far, which is
    what makes a batch escape the rank-1 degeneracy of a single run.

    Parameters are as :func:`linear_optimal_design` (including ``region`` and
    ``region_bounds`` for the I and G criteria), plus:

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
    The first picks of a from-scratch batch are degenerate: until the
    accumulated FIM is full rank, every candidate scores ``log det = -inf``
    (and ``λ_min = 0``), so the requested criterion cannot rank them. Those
    rounds maximize ``log det(FIM + εI)`` instead, with ``ε`` a tiny multiple of
    the typical single-run information. That picks the most informative point
    first and then fills the missing directions, rather than an arbitrary
    point. Once the FIM is full rank the requested criterion takes over.

    Greedy selection never revisits an early pick, so the batch is then
    polished by up to ``exchange_passes`` exchange sweeps: each run in turn is
    removed and replaced by the best point given all the others, if that
    improves the criterion. Set ``exchange_passes=0`` for pure greedy.
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")
    if template not in LINEAR_TEMPLATES:
        raise ValueError(
            f"{template!r} is not linear in its parameters; use "
            "discopt.doe.design.batch_optimal_experiment (which needs jax) instead. "
            f"Linear templates: {sorted(LINEAR_TEMPLATES)}"
        )
    criterion = normalize_criterion(criterion)

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
        exchange_passes=exchange_passes,
        region=region,
        region_bounds=region_bounds,
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
    exchange_passes: int = 2,
    region: DesignRegion | Sequence[Mapping[str, float]] | np.ndarray | None = None,
    region_bounds: Mapping[str, tuple[float, float]] | None = None,
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

    Other parameters are as :func:`linear_batch_design`; ``region`` and
    ``region_bounds`` (for the I and G criteria) as :func:`linear_optimal_design`.
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")
    criterion = normalize_criterion(criterion)

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

    region_obj = _resolve_region(
        region,
        criterion,
        basis,
        names,
        region_bounds,
        design_bounds,
        inequality_constraints,
        feasible_projection,
        seed,
    )
    rng = np.random.default_rng(seed)
    inv_var = 1.0 / (float(measurement_error) ** 2)
    eq_cons = tuple(equality_constraints or ())
    ineq_cons = tuple(inequality_constraints or ())

    # Scale for the rank-deficiency ridge: the typical information in one run.
    samples = rng.uniform(lbs, ubs, size=(32, len(names)))
    if feasible_projection is not None:
        samples = np.array(
            [
                [feasible_projection(dict(zip(names, map(float, u))))[n] for n in names]
                for u in samples
            ],
            dtype=np.float64,
        )
    # Per-parameter ridge: a tiny fraction of each parameter's own typical
    # single-run information. An isotropic ridge is swamped when parameters
    # differ in scale by many orders of magnitude (an Arrhenius k0 ~ 1e9 next
    # to Ea ~ 1e4 gives information 1e-18 vs 1e-7), and every run then
    # collapsed onto one point.
    rows_f = np.array([basis(u) for u in samples], dtype=np.float64)
    typical = np.mean(rows_f**2, axis=0) * inv_var
    fallback = float(np.max(typical)) if np.any(typical > 0) else 1.0
    typical = np.where(np.isfinite(typical) & (typical > 0), typical, fallback)
    eps = _RANK_RIDGE * typical

    def full_rank(m: np.ndarray) -> bool:
        # Test rank on the correlation-scaled matrix, so a full-rank FIM with a
        # huge scale spread is not mistaken for a singular one.
        if not m.any():
            return False
        d = np.sqrt(np.clip(np.diag(m), 0.0, None))
        if np.any(d == 0):
            return False
        return int(np.linalg.matrix_rank(m / np.outer(d, d))) >= n_p

    # G-optimal designs are built from the D-optimal one: in the approximate
    # (large-N) limit the two coincide (the Kiefer-Wolfowitz equivalence
    # theorem), and the maximum-variance surface is too kinked for a greedy build
    # from scratch. The G exchange passes that follow can then only improve on it.
    stages = [D_OPTIMAL, G_OPTIMAL] if criterion == G_OPTIMAL else [criterion]
    build = stages[0]

    def pick(
        base: np.ndarray,
        round_seed_rng: np.random.Generator,
        start: np.ndarray | None = None,
        target: str | None = None,
    ):
        # While ``base`` is rank-deficient every criterion is degenerate; rank
        # candidates by a lightly regularized log det, which is what fills in
        # the missing directions fastest.
        if full_rank(base):
            active, reg = (target or build), 0.0
        else:
            active, reg = D_OPTIMAL, eps
        return _search_one_point(
            basis,
            base,
            measurement_error,
            lbs,
            ubs,
            active,
            n_starts,
            round_seed_rng,
            names,
            eq_cons,
            ineq_cons,
            feasible_projection,
            regularization=reg,
            region=region_obj,
            start=start,
        )

    points: list[np.ndarray] = []
    for _ in range(int(n_experiments)):
        found = pick(accumulated, rng)
        if found is None:
            raise RuntimeError(
                "batch design search failed to find a finite criterion value "
                f"on round {len(points) + 1}; check the design bounds and constraints."
            )
        x, _ = found
        f = basis(x)
        accumulated = accumulated + np.outer(f, f) * inv_var
        points.append(x)

    # Exchange refinement: greedy never revisits an early pick, which leaves
    # runs stranded where they were only useful before the FIM was full rank.
    if full_rank(accumulated):
        for stage in stages:
            maximize = is_maximized(stage)
            current = evaluate_criterion(accumulated, stage, region=region_obj)
            passes = max(0, int(exchange_passes))
            if stage != stages[0]:
                passes = max(passes, 2)
            for _ in range(passes):
                improved = False
                for i in range(len(points)):
                    fi = basis(points[i])
                    others = accumulated - np.outer(fi, fi) * inv_var
                    found = pick(others, rng, start=points[i], target=stage)
                    if found is None:
                        continue
                    x_new, _ = found
                    fn = basis(x_new)
                    trial = others + np.outer(fn, fn) * inv_var
                    if not full_rank(trial):
                        continue
                    value = evaluate_criterion(trial, stage, region=region_obj)
                    gain = (value - current) if maximize else (current - value)
                    if np.isfinite(value) and gain > 1e-10 * max(1.0, abs(current)):
                        points[i] = x_new
                        accumulated = trial
                        current = value
                        improved = True
                if not improved:
                    break

    # Joint polish for the prediction criteria on a plain box. Exchange moves one
    # run at a time, which stalls on the I/G surfaces where runs must move
    # together (a symmetric pair drifting outward); a local search over every
    # coordinate at once, accepted only if the exact criterion improves, fixes it.
    if (
        criterion in REGION_CRITERIA
        and full_rank(accumulated)
        and not eq_cons
        and not ineq_cons
        and feasible_projection is None
    ):
        points, accumulated = _joint_polish(
            points, accumulated, basis, prior_fim, n_p, inv_var, criterion, region_obj, lbs, ubs
        )

    designs: list[dict[str, float]] = [{n: float(v) for n, v in zip(names, x)} for x in points]
    # Report the criterion as the batch accumulates, in run order. While the FIM
    # is still rank-deficient report the degenerate value: evaluate_criterion
    # would otherwise hand back a finite-looking log-det (the determinant
    # underflows rather than reaching exactly zero), which reads as a real
    # score for a design that identifies nothing yet.
    per_round: list[float] = []
    running = (
        np.zeros((n_p, n_p), dtype=np.float64)
        if prior_fim is None
        else np.asarray(prior_fim, dtype=np.float64).copy()
    )
    for x in points:
        f = basis(x)
        running = running + np.outer(f, f) * inv_var
        if full_rank(running):
            per_round.append(evaluate_criterion(running, criterion, region=region_obj))
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
    "ALL_CRITERIA",
    "A_OPTIMAL",
    "CRITERIA",
    "CRITERION_ALIASES",
    "DesignRegion",
    "D_OPTIMAL",
    "E_OPTIMAL",
    "G_OPTIMAL",
    "I_OPTIMAL",
    "REGION_CRITERIA",
    "LINEAR_TEMPLATES",
    "LinearBatchDesignResult",
    "LinearDesignResult",
    "ME_OPTIMAL",
    "basis_parameter_names",
    "basis_terms",
    "batch_design_from_basis",
    "design_matrix",
    "design_region",
    "design_row",
    "evaluate_criterion",
    "is_maximized",
    "linear_batch_design",
    "linear_fim",
    "linear_optimal_design",
    "normalize_criterion",
]
