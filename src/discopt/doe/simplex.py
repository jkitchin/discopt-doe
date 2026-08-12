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
) -> dict[str, float]:
    """Draw a Dirichlet sample on the simplex ``sum(variables) == total``.

    When ``bounds`` is supplied, samples that fall outside the per-axis
    bounds are clipped and rescaled; for tight bounds this may bias the
    sample, but it always returns a feasible point on the equality
    surface (up to one rescaling pass).
    """
    names = list(variables)
    q = len(names)
    weights = rng.dirichlet(np.ones(q))
    point = {n: float(total * w) for n, w in zip(names, weights)}
    if bounds is None:
        return point
    # Clip to bounds, then rescale the free mass back to ``total``.
    clipped = {n: float(np.clip(point[n], bounds[n][0], bounds[n][1])) for n in names}
    s = sum(clipped.values())
    if s > 0 and abs(s - total) > 1e-12:
        scale = total / s
        clipped = {n: clipped[n] * scale for n in names}
    return clipped


def project_to_simplex(
    point: dict[str, float],
    variables: Sequence[str],
    total: float = 1.0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Project ``point`` onto ``sum(variables) == total`` with non-negativity.

    Uses the standard Euclidean projection onto the probability simplex
    (Wang & Carreira-Perpiñán, 2013), scaled by ``total``. Variables
    outside ``variables`` are returned unchanged. If ``bounds`` is given,
    each projected component is then clipped to its box; this may push
    the sum slightly off ``total`` if the bounds are tight, but provides
    a reasonable warm start for SLSQP.
    """
    names = list(variables)
    v = np.array([point[n] for n in names], dtype=float)
    # Project v onto {w : w >= 0, sum(w) = total}.
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    rho = np.where(u - cssv / (np.arange(len(u)) + 1) > 0)[0]
    if len(rho) == 0:
        rho_idx = 0
    else:
        rho_idx = int(rho[-1])
    theta = cssv[rho_idx] / (rho_idx + 1)
    w = np.maximum(v - theta, 0.0)

    projected: dict[str, float] = dict(point)
    for n, val in zip(names, w):
        projected[n] = float(val)
    if bounds is not None:
        for n in names:
            lo, hi = bounds[n]
            projected[n] = float(np.clip(projected[n], lo, hi))
    return projected


__all__ = [
    "DesignConstraint",
    "project_to_simplex",
    "sample_simplex",
    "sum_constraint",
]
