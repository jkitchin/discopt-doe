"""Mixture-constraint geometry: the simplex ``sum(components) == total``.

Pure numpy — no model, no Fisher information, no autodiff. It lives apart from
:mod:`discopt.doe.design` for exactly that reason: a Scheffé design needs these
three helpers to constrain its search, and reaching for them through
``discopt.doe.design`` drags in ``discopt.doe.fim`` and ``discopt.estimate``,
neither of which exists in a WebAssembly build. That is not hypothetical — it
is what made every mixture template in the browser app fail on ``No module
named 'discopt.estimate'``.

``discopt.doe.design`` re-exports all three, so the public import paths that
predate this module keep working.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# A constraint is a callable that takes a design dict and returns a scalar.
# Equality constraints are enforced as ``fn(design) == 0`` and inequality
# constraints as ``fn(design) >= 0`` (scipy SLSQP convention).
DesignConstraint = Callable[[dict[str, float]], float]


def sum_constraint(variables: Sequence[str], total: float = 1.0) -> DesignConstraint:
    """Return an equality constraint that enforces ``sum(variables) == total``.

    Useful for mixture designs where component fractions or volumes must
    sum to a fixed amount.

    Parameters
    ----------
    variables : sequence of str
        Design variable names that participate in the sum.
    total : float, default ``1.0``
        Required sum value.

    Returns
    -------
    Callable[[dict[str, float]], float]
        Function ``g(design)`` returning ``sum(design[v] for v in variables) - total``.
        Pass to ``optimal_experiment(..., equality_constraints=[g])``.
    """
    names = tuple(variables)
    target = float(total)

    def _g(design: dict[str, float]) -> float:
        return float(sum(design[n] for n in names) - target)

    return _g


def sample_simplex(
    variables: Sequence[str],
    total: float,
    rng: np.random.Generator,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    max_tries: int = 200,
) -> dict[str, float]:
    """Draw a uniform sample on the simplex ``sum(variables) == total``.

    Without ``bounds`` this is a flat Dirichlet draw (unchanged behaviour).
    With ``bounds`` the sample is uniform over the constrained region: the
    lower bounds are absorbed exactly by drawing in L-pseudo-components
    ``x = L + (total - sum(L)) * w`` (Crosier 1984), and draws that break an
    upper bound are rejected. If ``max_tries`` draws are all rejected (a very
    thin region), the last one is projected onto the feasible region with
    :func:`project_to_simplex`, which is exact but no longer uniform. Either way
    the result satisfies the bounds and the sum exactly; the previous
    clip-and-rescale could leave a component outside its bounds.
    """
    names = list(variables)
    q = len(names)
    if bounds is None:
        weights = rng.dirichlet(np.ones(q))
        return {n: float(total * w) for n, w in zip(names, weights)}

    lo = np.array([float(bounds.get(n, (0.0, total))[0]) for n in names])
    hi = np.array([float(bounds.get(n, (0.0, total))[1]) for n in names])
    free = float(total) - float(lo.sum())
    if free < -1e-12 or float(hi.sum()) < float(total) - 1e-12 or np.any(hi < lo):
        raise ValueError(
            f"bounds are infeasible for total {total}: sum of lower bounds {lo.sum():g}, "
            f"sum of upper bounds {hi.sum():g}"
        )
    x = lo.copy()
    for _ in range(max(1, int(max_tries))):
        x = lo + max(free, 0.0) * rng.dirichlet(np.ones(q))
        if np.all(x <= hi + 1e-12):
            return {n: float(v) for n, v in zip(names, x)}
    return project_to_simplex({n: float(v) for n, v in zip(names, x)}, names, total, bounds=bounds)


def project_to_simplex(
    point: dict[str, float],
    variables: Sequence[str],
    total: float = 1.0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Euclidean projection of ``point`` onto the (bounded) mixture simplex.

    Without ``bounds`` this is the standard projection onto
    ``{w : w >= 0, sum(w) == total}`` (Wang & Carreira-Perpiñán, 2013). With
    ``bounds`` it projects onto ``{w : lo <= w <= hi, sum(w) == total}``, so the
    result satisfies the bounds *and* the sum. Earlier versions clipped to the
    bounds after projecting, which could leave the sum off ``total``. Lower
    bounds below zero are raised to zero, as mixture components are
    non-negative; a variable missing from ``bounds`` is bounded by
    ``[0, inf)``. Variables outside ``variables`` are returned unchanged.

    Raises
    ------
    ValueError
        If the bounds admit no blend summing to ``total`` (the lower bounds
        sum past it, or the upper bounds fall short of it).
    """
    names = list(variables)
    v = np.array([point[n] for n in names], dtype=float)
    if bounds is None:
        w = _project_unbounded(v, float(total))
    else:
        lo = np.array([max(float(bounds.get(n, (0.0, np.inf))[0]), 0.0) for n in names])
        hi = np.array([float(bounds.get(n, (0.0, np.inf))[1]) for n in names])
        w = _project_bounded(v, lo, hi, float(total))

    projected: dict[str, float] = dict(point)
    for n, val in zip(names, w):
        projected[n] = float(val)
    return projected


def _project_unbounded(v: np.ndarray, total: float) -> np.ndarray:
    """Projection onto ``{w >= 0, sum(w) = total}`` by the sort-and-threshold rule."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    rho = np.where(u - cssv / (np.arange(len(u)) + 1) > 0)[0]
    rho_idx = 0 if len(rho) == 0 else int(rho[-1])
    theta = cssv[rho_idx] / (rho_idx + 1)
    return np.maximum(v - theta, 0.0)


def _project_bounded(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float) -> np.ndarray:
    """Projection onto ``{lo <= w <= hi, sum(w) = total}``.

    The KKT conditions give ``w = clip(v - theta, lo, hi)`` for the scalar
    ``theta`` at which the sum equals ``total``. The sum is non-increasing in
    ``theta``, so bisection finds which components are free (strictly inside
    their bounds), and ``theta`` is then solved exactly on that active set.
    """
    if np.any(hi < lo):
        bad = int(np.argmax(hi < lo))
        raise ValueError(f"bounds for component {bad} have upper < lower ({lo[bad]}, {hi[bad]})")
    scale = max(1.0, abs(total), float(np.max(np.abs(v))))
    tol = 1e-12 * scale
    if lo.sum() > total + tol or hi.sum() < total - tol:
        raise ValueError(
            f"infeasible mixture bounds: the lower bounds sum to {lo.sum():g} and the upper "
            f"bounds to {hi.sum():g}, so no blend can sum to {total:g}"
        )

    def s(theta: float) -> float:
        return float(np.clip(v - theta, lo, hi).sum())

    a = float(np.min(v - lo)) - 1.0
    step = 1.0
    while s(a) < total:  # only reachable when some upper bound is infinite
        step *= 2.0
        a -= step * scale
    b = float(np.max(v - lo))  # every component at its lower bound: sum <= total
    for _ in range(200):
        mid = 0.5 * (a + b)
        if s(mid) >= total:
            a = mid
        else:
            b = mid
        if b - a <= 1e-15 * max(1.0, abs(a)):
            break
    theta = 0.5 * (a + b)

    shifted = v - theta
    free = (shifted > lo) & (shifted < hi)
    if np.any(free):
        fixed = np.where(shifted <= lo, lo, hi)
        theta = (float(v[free].sum()) + float(fixed[~free].sum()) - total) / int(free.sum())
    return np.clip(v - theta, lo, hi)


__all__ = [
    "DesignConstraint",
    "project_to_simplex",
    "sample_simplex",
    "sum_constraint",
]
