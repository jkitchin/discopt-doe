"""Constrained mixture regions: bounds, extreme vertices, pseudo-components.

Once component bounds are added, the mixture region stops being the whole
simplex. It becomes a smaller simplex (lower bounds only) or an irregular
polytope (upper bounds as well). This module holds the geometry for working
in such regions:

* :func:`check_mixture_bounds`: are the stated bounds consistent, and what
  are the bounds actually reachable once the sum constraint is applied?
  (Piepel 1983)
* :func:`extreme_vertices`: the corners of the constrained region
  (McLean & Anderson 1966).
* :func:`extreme_vertices_design`: the classical design on it: vertices,
  centroids of its faces of chosen dimensions, and the overall centroid.
* :func:`to_pseudo_components` and :func:`from_pseudo_components`: the
  L-pseudo-component rescaling that maps a lower-bounded region back onto a
  full simplex (Crosier 1984).
* :func:`cox_direction_trace`: points along Cox's direction from a reference
  blend (Cox 1971), for trace (effect) plots.

Optimal designs over the same region use
:func:`~discopt.doe.simplex.sum_constraint` and
:func:`~discopt.doe.simplex.project_to_simplex` with ``bounds=``.

numpy only, so it works in the browser build as well.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class MixtureBoundsCheck:
    """Result of :func:`check_mixture_bounds`.

    Attributes
    ----------
    feasible : bool
        Whether any blend satisfies every bound and the sum constraint.
    bounds : dict
        The implied (reachable) bounds per component. They equal the stated
        ones where those are consistent, and are tightened where part of a
        stated range cannot be reached. Empty when infeasible.
    adjusted : tuple of str
        Components whose stated bounds were tightened.
    message : str
        A plain-language summary.
    """

    feasible: bool
    bounds: dict[str, tuple[float, float]]
    adjusted: tuple[str, ...]
    message: str


def _split_bounds(
    bounds: Mapping[str, tuple[float, float]], total: float
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(bounds)
    if len(names) < 2:
        raise ValueError("a mixture needs at least 2 components")
    lo = np.array([float(bounds[n][0]) for n in names])
    hi = np.array([float(bounds[n][1]) for n in names])
    if np.any(lo < 0):
        raise ValueError("mixture lower bounds must be >= 0")
    if np.any(hi < lo):
        bad = names[int(np.argmax(hi < lo))]
        raise ValueError(f"component {bad!r}: upper bound below lower bound")
    if not total > 0:
        raise ValueError(f"total must be positive, got {total}")
    return names, lo, np.minimum(hi, total)


def check_mixture_bounds(
    bounds: Mapping[str, tuple[float, float]], total: float = 1.0
) -> MixtureBoundsCheck:
    """Check that component bounds are consistent, and tighten the ones that are not.

    With ``sum(x) == total``, a component cannot exceed ``total`` minus the
    other components' lower bounds, nor fall below ``total`` minus their upper
    bounds. Stated bounds outside those limits cannot be reached, and a
    design built on them wastes runs or fails. One pass of this tightening
    makes a feasible set of bounds consistent (Piepel 1983).

    Parameters
    ----------
    bounds : mapping name -> (lower, upper)
        Component bounds, in the same units as ``total``.
    total : float, default 1.0
        What the components sum to (1 for fractions, 100 for percentages).

    Examples
    --------
    >>> r = check_mixture_bounds({"A": (0.0, 1.0), "B": (0.0, 0.2), "C": (0.0, 0.2)})
    >>> r.bounds["A"]
    (0.6, 1.0)
    """
    names, lo, hi = _split_bounds(bounds, total)
    if lo.sum() > total + _TOL or hi.sum() < total - _TOL:
        return MixtureBoundsCheck(
            feasible=False,
            bounds={},
            adjusted=(),
            message=(
                f"infeasible: lower bounds sum to {lo.sum():g} and upper bounds to "
                f"{hi.sum():g}; no blend sums to {total:g}"
            ),
        )
    new_lo = np.maximum(lo, total - (hi.sum() - hi))
    new_hi = np.minimum(hi, total - (lo.sum() - lo))
    stated_hi = np.array([float(bounds[n][1]) for n in names])
    adjusted = tuple(
        n
        for n, a, b, c, d in zip(names, lo, stated_hi, new_lo, new_hi)
        if abs(a - c) > _TOL or abs(b - d) > _TOL
    )
    implied = {
        n: (float(np.round(a, 12)), float(np.round(b, 12)))
        for n, a, b in zip(names, new_lo, new_hi)
    }
    stated = {n: (float(bounds[n][0]), float(bounds[n][1])) for n in names}
    msg = (
        "consistent: every stated bound is reachable"
        if not adjusted
        else "tightened unreachable bounds: "
        + "; ".join(
            f"{n} [{stated[n][0]:g}, {stated[n][1]:g}] -> [{implied[n][0]:g}, {implied[n][1]:g}]"
            for n in adjusted
        )
    )
    return MixtureBoundsCheck(True, implied, adjusted, msg)


Ratios = Mapping[tuple[str, str], tuple[float | None, float | None]]


def _ratio_rows(names: Sequence[str], ratios: Ratios | None) -> list[tuple[np.ndarray, float]]:
    """Ratio bounds as rows ``a . x >= b``.

    ``low <= x_i / x_j <= high`` is not linear, but every blend has
    ``x_j >= 0``, so multiplying through gives ``x_i - low * x_j >= 0`` and
    ``high * x_j - x_i >= 0``, which are. That is what makes a ratio
    expressible as a face of the region rather than a curved boundary.
    """
    rows: list[tuple[np.ndarray, float]] = []
    index = {n: i for i, n in enumerate(names)}
    for pair, (low, high) in dict(ratios or {}).items():
        try:
            num, den = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ratio key {pair!r} must be a (numerator, denominator) pair of component names"
            ) from exc
        missing = [n for n in (num, den) if n not in index]
        if missing:
            raise ValueError(f"ratio {num}/{den}: unknown component(s) {missing}")
        if num == den:
            raise ValueError(f"ratio {num}/{den}: a component's ratio to itself is not a bound")
        if low is not None and high is not None and float(low) > float(high):
            raise ValueError(f"ratio {num}/{den}: lower bound {low} exceeds upper bound {high}")
        for value, sign in ((low, 1.0), (high, -1.0)):
            if value is None:
                continue
            value = float(value)
            if value < 0:
                raise ValueError(f"ratio {num}/{den}: bounds must be non-negative, got {value}")
            a = np.zeros(len(names))
            a[index[num]] = sign
            a[index[den]] = -sign * value
            rows.append((a, 0.0))
    return rows


def _bound_rows(lo: np.ndarray, hi: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Component bounds as rows ``a . x >= b``, so every constraint has one form."""
    rows = []
    q = len(lo)
    for i in range(q):
        e = np.zeros(q)
        e[i] = 1.0
        rows.append((e.copy(), float(lo[i])))
        rows.append((-e, -float(hi[i])))
    return rows


_MAX_COMBINATIONS = 250_000


def _vertices_from_rows(
    rows: Sequence[tuple[np.ndarray, float]], total: float, q: int
) -> np.ndarray:
    """Vertices of ``{x : a.x >= b for every row, sum(x) = total}``.

    A vertex of a polytope is where enough constraints meet: the region lives
    in the ``q - 1`` dimensional plane ``sum(x) = total``, so ``q - 1``
    independent constraints pin a point down. Every such combination is solved
    and kept if it satisfies the constraints it was not built from. With only
    component bounds the McLean-Anderson enumeration above is the same set and
    much cheaper, so this runs when a ratio makes the region something other
    than a box slice.
    """
    tol = 1e-9 * max(1.0, abs(total))
    n_rows = len(rows)
    combinations = math.comb(n_rows, q - 1) if n_rows >= q - 1 else 0
    if combinations > _MAX_COMBINATIONS:
        raise ValueError(
            f"this region needs {combinations:,} constraint combinations to enumerate "
            f"({n_rows} constraints, {q} components), which is more than this method is "
            "meant for. Drop a ratio bound, or sample the region with sample_simplex "
            "and use an optimal design over it instead."
        )
    ones = np.ones((1, q))
    found: list[np.ndarray] = []
    for combo in itertools.combinations(range(n_rows), q - 1):
        A = np.vstack([np.array([rows[i][0] for i in combo]).reshape(q - 1, q), ones])
        b = np.append([rows[i][1] for i in combo], total)
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue  # the chosen constraints are parallel: no single point
        if not np.all(np.isfinite(x)):
            continue
        if any(float(a @ x) < bb - tol for a, bb in rows):
            continue  # outside a constraint it was not built from
        x = np.round(x, 12)
        if not any(np.allclose(x, y, atol=1e-9 * max(1.0, total)) for y in found):
            found.append(x)
    return np.array(found) if found else np.zeros((0, q))


def _vertex_matrix(
    bounds: Mapping[str, tuple[float, float]], total: float, ratios: Ratios | None = None
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    check = check_mixture_bounds(bounds, total)
    if not check.feasible:
        raise ValueError(check.message)
    names = list(bounds)
    lo = np.array([check.bounds[n][0] for n in names])
    hi = np.array([check.bounds[n][1] for n in names])
    q = len(names)
    ratio_rows = _ratio_rows(names, ratios)
    if ratio_rows:
        verts = _vertices_from_rows(_bound_rows(lo, hi) + ratio_rows, total, q)
        if len(verts) == 0:
            raise ValueError(
                "no blend satisfies both the component bounds and the ratio bounds; "
                "check_mixture_bounds reports the bounds are feasible on their own, so "
                "the ratios are what closes the region"
            )
        return names, verts, lo, hi
    found: list[np.ndarray] = []
    # McLean & Anderson (1966): set all components but one at a bound, solve the
    # sum constraint for the remaining one, and keep it if within its bounds.
    for free in range(q):
        fixed = [k for k in range(q) if k != free]
        for choice in itertools.product(*[(lo[k], hi[k]) for k in fixed]):
            x = np.empty(q)
            x[fixed] = choice
            x[free] = total - float(np.sum(choice))
            if lo[free] - _TOL <= x[free] <= hi[free] + _TOL:
                x[free] = min(max(x[free], lo[free]), hi[free])
                x = np.round(x, 12)  # drop float noise such as 0.30000000000000004
                if not any(np.allclose(x, y, atol=1e-9 * max(1.0, total)) for y in found):
                    found.append(x)
    return names, np.array(found), lo, hi


def extreme_vertices(
    bounds: Mapping[str, tuple[float, float]],
    total: float = 1.0,
    *,
    ratios: Ratios | None = None,
) -> list[dict[str, float]]:
    """Corners of a constrained mixture region (McLean & Anderson 1966).

    Parameters
    ----------
    bounds : mapping name -> (lower, upper)
        Component bounds. Inconsistent bounds are tightened first, as in
        :func:`check_mixture_bounds`.
    total : float, default 1.0
    ratios : mapping (numerator, denominator) -> (low, high), optional
        Bounds on component ratios, which formulations state far more often
        than they state bounds on a component alone: an emulsifier that has to
        be two to three times the oil is ``{("emulsifier", "oil"): (2.0, 3.0)}``.
        Either end may be ``None`` for one-sided. A ratio is linear once
        multiplied out, so it cuts the region with a straight face like any
        other constraint -- and the region stops being a box slice, so the
        corners are found by general enumeration rather than by the
        McLean-Anderson rule.

    Returns
    -------
    list of dict
        One dict per vertex, in a deterministic order.

    Raises
    ------
    ValueError
        If the bounds are infeasible.
    """
    names, verts, _, _ = _vertex_matrix(bounds, total, ratios)
    order = np.lexsort(verts.T[::-1])
    return [{n: float(v) for n, v in zip(names, verts[i])} for i in order]


def extreme_vertices_design(
    bounds: Mapping[str, tuple[float, float]],
    total: float = 1.0,
    *,
    ratios: Ratios | None = None,
    face_dimensions: Sequence[int] = (1,),
    include_centroid: bool = True,
) -> list[dict[str, object]]:
    """The classical design for a constrained mixture region.

    Vertices of the region, plus the centroids of its faces of the requested
    dimensions (1 = edges, 2 = two-dimensional faces, ...), plus the overall
    centroid, as in McLean & Anderson (1966). A face's centroid is the average
    of the vertices on it.

    Parameters
    ----------
    bounds : mapping name -> (lower, upper)
    total : float, default 1.0
    ratios : mapping (numerator, denominator) -> (low, high), optional
        Ratio bounds, as in :func:`extreme_vertices`.
    face_dimensions : sequence of int, default (1,)
        Which face centroids to add. ``()`` gives the vertices (and centroid)
        only. For ``q`` components the region has dimension ``q - 1``;
        dimensions outside ``1 .. q - 2`` are ignored.
    include_centroid : bool, default True
        Add the overall centroid.

    Returns
    -------
    list of dict
        Each run maps components to values, plus ``point_type``: one of
        ``"vertex"``, ``"face-1"`` (edge), ``"face-2"``, ...,
        ``"centroid"``.
    """
    names, verts, lo, hi = _vertex_matrix(bounds, total, ratios)
    q = len(names)
    tol = 1e-9 * max(1.0, total)
    # A face is where a set of constraints is active, and a ratio bound is a
    # constraint like any other once written as a . x >= b, so the edges and
    # faces below come out right whether the region is a box slice or cut by a
    # ratio.
    rows = _bound_rows(lo, hi) + _ratio_rows(names, ratios)

    def active(x: np.ndarray) -> frozenset[int]:
        return frozenset(k for k, (a, b) in enumerate(rows) if abs(float(a @ x) - b) <= tol)

    act = [active(v) for v in verts]
    # Every face is the set of vertices on which some set of bound constraints
    # is active; those sets are closed under intersection, so generate them by
    # intersecting the vertices' active sets until nothing new appears.
    sets: set[frozenset[int]] = {a for a in act if a}
    frontier = set(sets)
    while frontier:
        new = set()
        for s1 in frontier:
            for s2 in sets:
                inter = s1 & s2
                if inter and inter not in sets:
                    new.add(inter)
        sets |= new
        frontier = new

    faces: dict[frozenset[int], int] = {}
    for s in sets:
        members = frozenset(i for i, a in enumerate(act) if s <= a)
        if len(members) < 2 or members in faces:
            continue
        pts = verts[sorted(members)]
        dim = int(np.linalg.matrix_rank(pts - pts[0], tol=tol)) if len(pts) > 1 else 0
        faces[members] = dim

    rows: list[dict[str, object]] = []
    order = np.lexsort(verts.T[::-1])
    for i in order:
        rows.append({**{n: float(x) for n, x in zip(names, verts[i])}, "point_type": "vertex"})
    wanted = sorted({int(d) for d in face_dimensions if 1 <= int(d) <= q - 2})
    for d in wanted:
        cents = [verts[sorted(m)].mean(axis=0) for m, dim in faces.items() if dim == d]
        cents.sort(key=lambda c: tuple(np.round(c, 12)))
        for c in cents:
            rows.append({**{n: float(x) for n, x in zip(names, c)}, "point_type": f"face-{d}"})
    if include_centroid:
        c = verts.mean(axis=0)
        rows.append({**{n: float(x) for n, x in zip(names, c)}, "point_type": "centroid"})
    return rows


def to_pseudo_components(
    x: Mapping[str, float], lower: Mapping[str, float], total: float = 1.0
) -> dict[str, float]:
    """L-pseudo-components: ``x'_i = (x_i - L_i) / (total - sum(L))`` (Crosier 1984).

    Maps the lower-bounded region ``x_i >= L_i`` onto the full simplex (with
    total 1), which makes lattice designs usable again and improves the
    conditioning of a Scheffé fit. Components missing from ``lower`` have
    ``L = 0``.
    """
    names = list(x)
    lo = {n: float(lower.get(n, 0.0)) for n in names}
    room = float(total) - sum(lo.values())
    if not room > 0:
        raise ValueError("lower bounds leave no room: sum(lower) must be below total")
    return {n: (float(x[n]) - lo[n]) / room for n in names}


def from_pseudo_components(
    xp: Mapping[str, float], lower: Mapping[str, float], total: float = 1.0
) -> dict[str, float]:
    """Inverse of :func:`to_pseudo_components`: ``x_i = L_i + (total - sum(L)) x'_i``."""
    names = list(xp)
    lo = {n: float(lower.get(n, 0.0)) for n in names}
    room = float(total) - sum(lo.values())
    if not room > 0:
        raise ValueError("lower bounds leave no room: sum(lower) must be below total")
    return {n: lo[n] + room * float(xp[n]) for n in names}


def cox_direction_trace(
    reference: Mapping[str, float],
    component: str,
    deltas: Sequence[float] | None = None,
    *,
    total: float = 1.0,
    n: int = 51,
) -> list[dict[str, float]]:
    """Blends along Cox's direction from a reference blend (Cox 1971).

    Changing ``component`` by ``delta`` and taking the others up or down *in
    their reference proportions* keeps the sum at ``total``:
    ``x_i = s_i + delta`` and ``x_j = s_j - delta * s_j / (total - s_i)``. Model
    predictions along these lines, one per component, form a trace (effect)
    plot.

    Parameters
    ----------
    reference : mapping name -> value
        The reference blend (it must sum to ``total``), often the centroid of
        the region.
    component : str
        The component to vary.
    deltas : sequence of float, optional
        Changes in ``component`` from its reference value. The default is ``n``
        evenly spaced values over the whole feasible line, from ``-s_i`` (the
        component absent) to ``total - s_i`` (the pure component).
    total : float, default 1.0
    n : int, default 51
        Number of points when ``deltas`` is omitted.

    Returns
    -------
    list of dict
        One blend per delta, each carrying a ``delta`` key as well.
    """
    names = list(reference)
    if component not in reference:
        raise ValueError(f"component {component!r} not in the reference blend")
    s = {k: float(v) for k, v in reference.items()}
    if abs(sum(s.values()) - total) > 1e-6 * max(1.0, total):
        raise ValueError(f"reference blend sums to {sum(s.values()):g}, not {total:g}")
    si = s[component]
    rest = total - si
    if not rest > 0:
        raise ValueError("the reference is the pure component; Cox's direction is undefined")
    if deltas is None:
        deltas = list(np.linspace(-si, rest, int(n)))
    out = []
    for d in deltas:
        d = float(d)
        if d < -si - 1e-12 or d > rest + 1e-12:
            raise ValueError(f"delta {d:g} leaves the simplex; it must lie in [{-si:g}, {rest:g}]")
        row = {k: (si + d if k == component else s[k] - d * s[k] / rest) for k in names}
        row["delta"] = d
        out.append(row)
    return out


__all__ = [
    "MixtureBoundsCheck",
    "check_mixture_bounds",
    "cox_direction_trace",
    "extreme_vertices",
    "extreme_vertices_design",
    "from_pseudo_components",
    "to_pseudo_components",
]
