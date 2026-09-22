"""Classical response-surface methodology on a fitted polynomial.

These are the analysis steps of an RSM campaign (Box & Wilson 1951), done on
the coefficients of a fitted model:

* :func:`steepest_ascent_path` -- from a first-order fit far from the optimum,
  the path along which the response rises fastest.
* :func:`canonical_analysis` -- for a second-order fit
  ``ŷ = b0 + xᵀb + xᵀBx``, the stationary point ``x_s = -½ B⁻¹ b`` and the
  eigen-decomposition of ``B``, which says whether it is a maximum, a
  minimum, a saddle, or a ridge.
* :func:`stationary_point_ci` -- a delta-method confidence interval on the
  stationary point's location (Box & Hunter 1954; del Castillo & Cahya 2001).
  An optimum's position is an estimate with an uncertainty like any other.
* :func:`ridge_analysis` -- the best response at each distance from the design
  centre, for when the stationary point lies outside the region or is a
  saddle (Hoerl 1959; Draper 1963).
* :func:`desirability` and :func:`overall_desirability` -- Derringer & Suich
  (1980) desirability functions for trading several responses off at once.

Coefficients can be passed as the ``estimates`` dict that
:func:`~discopt.doe.symbolic.fit_least_squares` returns for a quadratic
model, named as :func:`~discopt.doe.linear_design.basis_parameter_names`
does (``b0``, ``b1``…``bk``, ``b11``…``bkk``, ``b12``…), or as arrays ``b``
and ``B``. All coordinates are in whatever units the model was fitted in;
fitting in coded units (−1…+1) is strongly recommended, since the geometry of
steepest ascent and ridge analysis is only meaningful when the factors are on
comparable scales. Only numpy and scipy are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "CanonicalAnalysis",
    "RidgeAnalysis",
    "StationaryPointCI",
    "SteepestAscentPath",
    "canonical_analysis",
    "desirability",
    "overall_desirability",
    "quadratic_form",
    "ridge_analysis",
    "stationary_point_ci",
    "steepest_ascent_path",
]


# --------------------------------------------------------------------------
# Coefficient handling
# --------------------------------------------------------------------------


def _quadratic_names(k: int) -> list[str]:
    from discopt.doe.linear_design import basis_parameter_names

    return basis_parameter_names("quadratic", k)


def _k_from_quadratic_count(p: int) -> int:
    # p = 1 + 2k + k(k-1)/2  ->  k² + 3k + 2 - 2p = 0
    k = int(round((-3 + np.sqrt(9 - 4 * (2 - 2 * p))) / 2))
    if k < 1 or 1 + 2 * k + k * (k - 1) // 2 != p:
        raise ValueError(
            f"{p} coefficients is not a full quadratic in any number of factors "
            "(expected 1 + 2k + k(k-1)/2: 6 for k=2, 10 for k=3, 15 for k=4, ...)"
        )
    return k


def _unwrap_fit(coefficients: Any) -> Mapping[str, float]:
    """Accept either an estimates mapping or a whole fit_least_squares result."""
    if isinstance(coefficients, Mapping) and "estimates" in coefficients:
        return coefficients["estimates"]
    return coefficients


def quadratic_form(
    coefficients: Mapping[str, float] | Sequence[float],
    *,
    parameter_names: Sequence[str] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Split quadratic-model coefficients into ``(b0, b, B)``.

    ``ŷ = b0 + xᵀb + xᵀBx`` with ``B`` symmetric: ``B[i, i] = b_ii`` and
    ``B[i, j] = B[j, i] = b_ij / 2``.

    Parameters
    ----------
    coefficients : mapping or sequence
        A mapping keyed by the names of
        :func:`~discopt.doe.linear_design.basis_parameter_names` for the
        ``"quadratic"`` basis (a ``fit_least_squares`` result or its
        ``estimates`` are accepted), or values in that order.
    parameter_names : sequence of str, optional
        Names for a positional sequence, if they are not the standard order.
    """
    coefficients = _unwrap_fit(coefficients)
    if isinstance(coefficients, Mapping):
        k = _k_from_quadratic_count(len(coefficients))
        names = _quadratic_names(k)
        missing = [n for n in names if n not in coefficients]
        if missing:
            raise ValueError(
                f"coefficients are missing {missing}; expected the quadratic-basis names {names}"
            )
        values = np.array([float(coefficients[n]) for n in names])
    else:
        values = np.asarray(coefficients, dtype=np.float64).ravel()
        k = _k_from_quadratic_count(values.size)
        if parameter_names is not None:
            order = {n: i for i, n in enumerate(parameter_names)}
            values = np.array([values[order[n]] for n in _quadratic_names(k)])
    b0 = float(values[0])
    b = values[1 : 1 + k].copy()
    B = np.diag(values[1 + k : 1 + 2 * k])
    cross = values[1 + 2 * k :]
    idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            B[i, j] = B[j, i] = cross[idx] / 2.0
            idx += 1
    return b0, b, B


def _resolve_bB(coefficients, b, B, b0):
    if coefficients is not None:
        if b is not None or B is not None:
            raise ValueError("give either coefficients= or b= and B=, not both")
        return quadratic_form(coefficients)
    if b is None or B is None:
        raise ValueError("give coefficients=, or both b= and B=")
    b = np.asarray(b, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    if B.shape != (b.size, b.size):
        raise ValueError(f"B has shape {B.shape}; expected ({b.size}, {b.size})")
    if not np.allclose(B, B.T):
        raise ValueError("B must be symmetric (B[i, j] = b_ij / 2)")
    return float(b0), b, B


# --------------------------------------------------------------------------
# Steepest ascent
# --------------------------------------------------------------------------


@dataclass
class SteepestAscentPath:
    """Points along the path of steepest ascent (or descent).

    Attributes
    ----------
    direction : numpy.ndarray
        Unit vector of the path, in coded units.
    coded : numpy.ndarray
        ``(n_steps + 1, k)`` points in coded units; row 0 is the design centre.
    natural : numpy.ndarray or None
        The same points in natural units, when ``center`` and ``half_range``
        were given.
    input_names : list of str or None
        Factor names, for :meth:`rows`.
    """

    direction: np.ndarray
    coded: np.ndarray
    natural: np.ndarray | None
    input_names: list[str] | None

    def rows(self) -> list[dict[str, float]]:
        """The path as run dicts (natural units when available)."""
        pts = self.natural if self.natural is not None else self.coded
        names = self.input_names or [f"x{i + 1}" for i in range(pts.shape[1])]
        return [dict(zip(names, map(float, p))) for p in pts]


def steepest_ascent_path(
    coefficients: Mapping[str, float] | Sequence[float],
    *,
    n_steps: int = 5,
    step: float = 1.0,
    base_factor: str | int | None = None,
    center: Sequence[float] | Mapping[str, float] | None = None,
    half_range: Sequence[float] | Mapping[str, float] | None = None,
    input_names: Sequence[str] | None = None,
    direction: str = "ascent",
) -> SteepestAscentPath:
    """Path of steepest ascent from a first-order model (Box & Wilson 1951).

    For ``ŷ = b0 + Σ b_i x_i`` in coded units, the response rises fastest along
    ``b``. Two step conventions are supported:

    * default: each step moves a Euclidean distance ``step`` (coded units)
      along ``b / ||b||``;
    * ``base_factor``: the textbook recipe -- the named factor moves ``step``
      coded units per step and every other factor moves in proportion,
      ``Δx_j = step · b_j / |b_base|``.

    Parameters
    ----------
    coefficients : mapping or sequence
        A first-order fit, keyed ``b0, b1, ..., bk`` (the ``"linear"`` basis;
        a ``fit_least_squares`` result is accepted) or the slopes
        ``[b1, ..., bk]`` alone.
    n_steps : int, default 5
        Steps beyond the centre.
    step : float, default 1.0
        Step size in coded units, as described above.
    base_factor : str or int, optional
        Factor whose coded step is ``step``; by name (needs ``input_names`` or
        a mapping of coefficients) or 0-based index.
    center, half_range : sequence or mapping, optional
        Natural-unit centre and half-range of each factor (coded ``x = (ξ -
        center) / half_range``), to report the path in natural units too.
    input_names : sequence of str, optional
        Factor names, in coefficient order.
    direction : {"ascent", "descent"}
        ``"descent"`` follows ``-b`` (for a response to minimize).
    """
    coefficients = _unwrap_fit(coefficients)
    if isinstance(coefficients, Mapping):
        k = len(coefficients) - 1
        names_b = [f"b{i + 1}" for i in range(k)]
        missing = [n for n in ["b0", *names_b] if n not in coefficients]
        if missing:
            raise ValueError(f"first-order coefficients are missing {missing}")
        slopes = np.array([float(coefficients[n]) for n in names_b])
    else:
        slopes = np.asarray(coefficients, dtype=np.float64).ravel()
        k = slopes.size
    if direction not in ("ascent", "descent"):
        raise ValueError(f"direction must be 'ascent' or 'descent', got {direction!r}")
    if direction == "descent":
        slopes = -slopes
    norm = float(np.linalg.norm(slopes))
    if norm == 0.0:
        raise ValueError("all slopes are zero: there is no direction of steepest ascent")
    names = list(input_names) if input_names is not None else None
    if names is not None and len(names) != k:
        raise ValueError(f"{len(names)} input_names for {k} factors")

    if base_factor is None:
        per_step = float(step) * slopes / norm
    else:
        if isinstance(base_factor, str):
            if names is None:
                raise ValueError("base_factor by name needs input_names=")
            j = names.index(base_factor)
        else:
            j = int(base_factor)
        if slopes[j] == 0.0:
            raise ValueError("the base factor has a zero slope; choose another")
        per_step = float(step) * slopes / abs(slopes[j])

    coded = np.outer(np.arange(int(n_steps) + 1), per_step)

    def _vec(v):
        if v is None:
            return None
        if isinstance(v, Mapping):
            if names is None:
                raise ValueError("mapping center/half_range needs input_names=")
            return np.array([float(v[n]) for n in names])
        return np.asarray(v, dtype=np.float64).ravel()

    c, h = _vec(center), _vec(half_range)
    if (c is None) != (h is None):
        raise ValueError("give both center= and half_range=, or neither")
    natural = None if c is None else c + h * coded
    return SteepestAscentPath(
        direction=slopes / norm, coded=coded, natural=natural, input_names=names
    )


# --------------------------------------------------------------------------
# Canonical analysis
# --------------------------------------------------------------------------


@dataclass
class CanonicalAnalysis:
    """Canonical analysis of ``ŷ = b0 + xᵀb + xᵀBx``.

    Attributes
    ----------
    stationary_point : numpy.ndarray or None
        ``x_s = -½ B⁻¹ b``; ``None`` if ``B`` is singular.
    response_at_stationary : float or None
        ``ŷ(x_s) = b0 + ½ x_sᵀ b``.
    eigenvalues : numpy.ndarray
        Eigenvalues of ``B``, ascending. Each is the curvature along its
        canonical axis: ``ŷ = ŷ_s + Σ λ_i w_i²``.
    eigenvectors : numpy.ndarray
        Columns are the canonical axes.
    nature : str
        ``"maximum"`` (all λ < 0), ``"minimum"`` (all λ > 0), ``"saddle"``
        (mixed signs), or ``"ridge"`` (some |λ| negligible: the response is
        nearly flat along that axis, so the optimum is poorly located).
    distance : float or None
        ``||x_s||``: in coded units, > 1 means the stationary point lies
        outside the region the design explored.
    """

    stationary_point: np.ndarray | None
    response_at_stationary: float | None
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nature: str
    distance: float | None

    def summary(self) -> str:
        lam = ", ".join(f"{v:.4g}" for v in self.eigenvalues)
        if self.stationary_point is None:
            return f"B is singular (eigenvalues {lam}): {self.nature}, no unique stationary point"
        xs = ", ".join(f"{v:.4g}" for v in self.stationary_point)
        return (
            f"{self.nature} at x_s = ({xs}), ŷ = {self.response_at_stationary:.4g}, "
            f"|x_s| = {self.distance:.3g}; eigenvalues {lam}"
        )


def canonical_analysis(
    coefficients: Mapping[str, float] | Sequence[float] | None = None,
    *,
    b0: float = 0.0,
    b: Sequence[float] | None = None,
    B: np.ndarray | None = None,
    ridge_tol: float = 0.05,
) -> CanonicalAnalysis:
    """Locate and classify the stationary point of a second-order model.

    Parameters
    ----------
    coefficients : mapping or sequence, optional
        Quadratic-basis coefficients (see :func:`quadratic_form`).
    b0, b, B : optional
        The model directly, instead of ``coefficients``.
    ridge_tol : float, default 0.05
        An eigenvalue smaller than ``ridge_tol`` times the largest ``|λ|`` is
        treated as zero, and the surface is reported as a ``"ridge"``.
    """
    b0, b, B = _resolve_bB(coefficients, b, B, b0)
    lam, vec = np.linalg.eigh(B)
    scale = float(np.max(np.abs(lam))) if lam.size else 0.0
    small = np.abs(lam) <= ridge_tol * scale if scale > 0 else np.ones_like(lam, bool)
    if np.any(small):
        nature = "ridge"
    elif np.all(lam < 0):
        nature = "maximum"
    elif np.all(lam > 0):
        nature = "minimum"
    else:
        nature = "saddle"

    if scale > 0 and np.all(np.abs(lam) > 1e-12 * scale):
        xs = -0.5 * np.linalg.solve(B, b)
        ys = float(b0 + 0.5 * xs @ b)
        dist = float(np.linalg.norm(xs))
    else:
        xs, ys, dist = None, None, None
    return CanonicalAnalysis(
        stationary_point=xs,
        response_at_stationary=ys,
        eigenvalues=lam,
        eigenvectors=vec,
        nature=nature,
        distance=dist,
    )


# --------------------------------------------------------------------------
# Confidence interval on the stationary point
# --------------------------------------------------------------------------


@dataclass
class StationaryPointCI:
    """Delta-method confidence interval on the stationary point.

    Attributes
    ----------
    point : numpy.ndarray
        Estimated stationary point.
    std_errors : numpy.ndarray
        Standard error of each coordinate.
    lower, upper : numpy.ndarray
        Confidence limits at ``level``.
    covariance : numpy.ndarray
        Approximate covariance of the stationary point, ``J Σ Jᵀ``.
    level : float
        Confidence level.
    """

    point: np.ndarray
    std_errors: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    covariance: np.ndarray
    level: float


def stationary_point_ci(
    coefficients: Mapping[str, Any] | Sequence[float],
    covariance: np.ndarray | None = None,
    *,
    parameter_names: Sequence[str] | None = None,
    level: float = 0.95,
    dof: int | None = None,
) -> StationaryPointCI:
    """Delta-method confidence interval on the location of the stationary point.

    The stationary point ``x_s = -½ B⁻¹ b`` is a nonlinear function of the
    coefficients, so its covariance is approximated by ``J Σ Jᵀ`` with the
    Jacobian taken analytically: ``∂x_s/∂b = -½ B⁻¹``, and for each second-order
    coefficient ``∂x_s = -B⁻¹ (∂B) x_s``. The approximation is good when the
    surface is well curved; near a ridge, where ``B`` is nearly singular, the
    interval becomes very wide, which is the honest answer, but it also stops
    being accurate (Box & Hunter 1954; del Castillo & Cahya 2001).

    Parameters
    ----------
    coefficients : mapping or sequence
        Quadratic-basis coefficients, or a whole
        :func:`~discopt.doe.symbolic.fit_least_squares` result, in which case
        ``covariance``, ``parameter_names`` and ``dof`` are taken from it.
    covariance : numpy.ndarray, optional
        Coefficient covariance, ordered by ``parameter_names``.
    parameter_names : sequence of str, optional
        Order of the covariance rows. Defaults to the mapping's key order, or
        the standard quadratic-basis order.
    level : float, default 0.95
        Confidence level.
    dof : int, optional
        Residual degrees of freedom: use a ``t`` critical value. The normal
        value is used when omitted.
    """
    from scipy import stats

    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if isinstance(coefficients, Mapping) and "estimates" in coefficients:
        fit = coefficients
        estimates = fit["estimates"]
        if covariance is None:
            covariance = fit.get("covariance")
        if parameter_names is None:
            parameter_names = fit.get("parameter_names") or list(estimates)
        if dof is None:
            dof = fit.get("degrees_of_freedom")
        coefficients = estimates
    if covariance is None:
        raise ValueError("covariance is required (or pass a fit_least_squares result)")

    b0, b, B = quadratic_form(coefficients)
    k = b.size
    std_names = _quadratic_names(k)
    if parameter_names is None:
        parameter_names = list(coefficients) if isinstance(coefficients, Mapping) else std_names
    parameter_names = list(parameter_names)
    if sorted(parameter_names) != sorted(std_names):
        raise ValueError(f"parameter_names {parameter_names} do not match {std_names}")
    cov = np.asarray(covariance, dtype=np.float64)
    p = len(std_names)
    if cov.shape != (p, p):
        raise ValueError(f"covariance has shape {cov.shape}; expected ({p}, {p})")
    # Reorder the covariance into the standard order.
    perm = [parameter_names.index(n) for n in std_names]
    cov = cov[np.ix_(perm, perm)]

    B_inv = np.linalg.inv(B)
    xs = -0.5 * B_inv @ b
    J = np.zeros((k, p))
    # b0 does not move the stationary point (column 0 stays zero).
    J[:, 1 : 1 + k] = -0.5 * B_inv
    for i in range(k):  # pure quadratic b_ii: dB = E_ii
        e = np.zeros(k)
        e[i] = xs[i]
        J[:, 1 + k + i] = -B_inv @ e
    col = 1 + 2 * k
    for i in range(k):  # cross b_ij: dB = (E_ij + E_ji) / 2
        for j in range(i + 1, k):
            e = np.zeros(k)
            e[i] += xs[j] / 2.0
            e[j] += xs[i] / 2.0
            J[:, col] = -B_inv @ e
            col += 1

    cov_xs = J @ cov @ J.T
    se = np.sqrt(np.clip(np.diag(cov_xs), 0.0, None))
    q = 0.5 + float(level) / 2.0
    crit = float(stats.t.ppf(q, dof)) if dof else float(stats.norm.ppf(q))
    return StationaryPointCI(
        point=xs,
        std_errors=se,
        lower=xs - crit * se,
        upper=xs + crit * se,
        covariance=cov_xs,
        level=float(level),
    )


# --------------------------------------------------------------------------
# Ridge analysis
# --------------------------------------------------------------------------


@dataclass
class RidgeAnalysis:
    """Best response at each distance from the centre.

    Attributes
    ----------
    radii : numpy.ndarray
        Distances from the design centre (coded units).
    points : numpy.ndarray
        ``(len(radii), k)``: the optimum on each sphere.
    response : numpy.ndarray
        Predicted response at each point.
    multipliers : numpy.ndarray
        The Lagrange multiplier ``μ`` for each radius (NaN at radius 0).
    """

    radii: np.ndarray
    points: np.ndarray
    response: np.ndarray
    multipliers: np.ndarray


def ridge_analysis(
    coefficients: Mapping[str, float] | Sequence[float] | None = None,
    radii: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    *,
    b0: float = 0.0,
    b: Sequence[float] | None = None,
    B: np.ndarray | None = None,
    maximize: bool = True,
) -> RidgeAnalysis:
    """Constrained optimum of a second-order model on spheres of given radius.

    On the sphere ``||x|| = R`` the optimum satisfies ``(B - μI) x = -b/2``,
    with ``μ`` above the largest eigenvalue of ``B`` for a maximum (below the
    smallest for a minimum), where ``||x(μ)||`` decreases monotonically in
    ``μ``. Each radius is found by a 1-D root search. The path is the right
    tool when the stationary point is a saddle or lies outside the explored
    region: it shows how far, and in which direction, to move (Hoerl 1959;
    Draper 1963).
    """
    from scipy.optimize import brentq

    b0, b, B = _resolve_bB(coefficients, b, B, b0)
    k = b.size
    sign = 1.0 if maximize else -1.0
    Bs, bs = sign * B, sign * b  # maximize the (possibly negated) surface
    lam_max = float(np.linalg.eigvalsh(Bs).max())
    eye = np.eye(k)

    def x_of(mu: float) -> np.ndarray:
        return np.linalg.solve(Bs - mu * eye, -0.5 * bs)

    radii_arr = np.asarray(radii, dtype=np.float64)
    pts, mus = [], []
    for R in radii_arr:
        if R < 0:
            raise ValueError("radii must be non-negative")
        if R == 0:
            pts.append(np.zeros(k))
            mus.append(np.nan)
            continue
        if np.linalg.norm(bs) == 0:
            raise ValueError("b = 0: every point on a sphere is stationary; ridge undefined")
        # ||x(mu)|| -> inf as mu -> lam_max+, -> 0 as mu -> inf.
        span = max(1.0, abs(lam_max))
        lo = lam_max + 1e-12 * span
        hi = lam_max + span
        while np.linalg.norm(x_of(hi)) > R:
            hi = lam_max + 2 * (hi - lam_max)
        g = lambda mu: np.linalg.norm(x_of(mu)) - R  # noqa: E731
        if g(lo) < 0:
            # b has (numerically) no component along the top eigenvector (the
            # "hard case"); the root search cannot reach R from above.
            raise ValueError(
                f"radius {R} is not reachable along the ridge (degenerate case: b is "
                "orthogonal to the leading canonical axis)"
            )
        mu = brentq(g, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=500)
        pts.append(x_of(mu))
        mus.append(sign * mu)
    P = np.array(pts)
    resp = np.array([b0 + p @ b + p @ B @ p for p in P])
    return RidgeAnalysis(radii=radii_arr, points=P, response=resp, multipliers=np.array(mus))


# --------------------------------------------------------------------------
# Desirability
# --------------------------------------------------------------------------


def desirability(
    y: float | Sequence[float] | np.ndarray,
    kind: str,
    low: float,
    high: float,
    *,
    target: float | None = None,
    weight: float = 1.0,
    weight_upper: float | None = None,
) -> np.ndarray:
    """Derringer–Suich (1980) individual desirability, in ``[0, 1]``.

    * ``"maximize"``: 0 at or below ``low``, 1 at or above ``high``,
      ``((y - low)/(high - low))^weight`` between.
    * ``"minimize"``: 1 at or below ``low``, 0 at or above ``high``,
      ``((high - y)/(high - low))^weight`` between.
    * ``"target"``: 1 at ``target``, 0 outside ``[low, high]``, rising with
      exponent ``weight`` below the target and falling with ``weight_upper``
      (default ``weight``) above it.

    A weight above 1 makes the function demanding (desirability rises only
    near the goal); below 1, lenient.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    low, high = float(low), float(high)
    if not high > low:
        raise ValueError(f"need high > low, got low={low}, high={high}")
    if weight <= 0 or (weight_upper is not None and weight_upper <= 0):
        raise ValueError("weights must be positive")
    if kind == "maximize":
        d = np.clip((y_arr - low) / (high - low), 0.0, 1.0) ** weight
    elif kind == "minimize":
        d = np.clip((high - y_arr) / (high - low), 0.0, 1.0) ** weight
    elif kind == "target":
        if target is None or not low <= float(target) <= high:
            raise ValueError("kind='target' needs low <= target <= high")
        t = float(target)
        wu = weight if weight_upper is None else weight_upper
        below = np.clip((y_arr - low) / (t - low), 0.0, 1.0) ** weight if t > low else None
        above = np.clip((high - y_arr) / (high - t), 0.0, 1.0) ** wu if high > t else None
        d = np.where(
            y_arr <= t,
            below if below is not None else (y_arr == t).astype(float),
            above if above is not None else (y_arr == t).astype(float),
        )
        d = np.where((y_arr < low) | (y_arr > high), 0.0, d)
    else:
        raise ValueError(f"kind must be 'maximize', 'minimize' or 'target', got {kind!r}")
    return d


def overall_desirability(
    desirabilities: Sequence[float | Sequence[float] | np.ndarray],
    importance: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted geometric mean of individual desirabilities.

    ``D = (Π d_i^{r_i})^{1/Σ r_i}``: if any response is unacceptable
    (``d_i = 0``) the whole is, which is the point of a geometric rather than
    arithmetic mean. ``importance`` gives the relative weights ``r_i``
    (default all 1).
    """
    ds = [np.asarray(d, dtype=np.float64) for d in desirabilities]
    if not ds:
        raise ValueError("need at least one desirability")
    r = np.ones(len(ds)) if importance is None else np.asarray(importance, dtype=np.float64)
    if r.shape != (len(ds),) or np.any(r <= 0):
        raise ValueError("importance must be one positive weight per desirability")
    stacked = np.stack(np.broadcast_arrays(*ds))
    if np.any((stacked < 0) | (stacked > 1)):
        raise ValueError("desirabilities must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        logs = np.where(stacked > 0, np.log(np.where(stacked > 0, stacked, 1.0)), -np.inf)
    total = np.tensordot(r, logs, axes=1) / r.sum()
    return np.exp(total)
