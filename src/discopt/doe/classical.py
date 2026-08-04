"""Classical space-filling and response-surface designs.

These are the *closed-form* designs: the run list follows from the factor
count and a handful of options, with no model, no Fisher information, and no
optimizer in the loop. That makes them the counterpart to the FIM-optimal
designs in :mod:`discopt.doe.design` — you reach for these when you want a
standard, defensible design without committing to a parametric model first,
and for those you reach for :func:`~discopt.doe.design.optimal_experiment`.

Three families live here:

Latin hypercube
    :func:`latin_hypercube_design` — stratified random sampling that puts
    exactly one point in each of ``n`` equal-probability strata per factor.
    The general-purpose space-filling design for computer experiments,
    surrogate fitting, and any situation where you want good coverage of a
    continuous box without assuming a model form.

Central composite
    :func:`central_composite_design` — the workhorse for fitting a full
    quadratic response surface: a 2-level factorial core (estimates main
    effects and two-factor interactions), axial points (estimate pure
    quadratic curvature), and replicated centre points (estimate pure error
    and check lack of fit).

Box-Behnken
    :func:`box_behnken_design` — a quadratic design that never visits a corner
    of the design box. Every point sits at the midpoint of an edge, so no run
    combines the extreme level of *every* factor at once, which matters when
    corners are expensive, unsafe, or physically unreachable. All points lie
    on a common sphere, and the designs are rotatable or near-rotatable
    depending on the factor count.

Dependency note
---------------
This module is deliberately numpy-and-scipy only: no jax, no base ``discopt``.
It is one of the modules that must keep working in a WebAssembly build where
neither can be installed (see ``tests/test_import_hygiene.py``). The one scipy
call — ``scipy.stats.qmc`` for the Latin hypercube — degrades to a pure-numpy
implementation when unavailable, matching the fallback the ``optimize``
template already uses.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Latin hypercube stratification and the CCD/Box-Behnken coded levels are all
# exact in floating point except for the scaling multiply; points that should
# land exactly on a bound are snapped within this tolerance so a design never
# trips a downstream `lb <= x <= ub` check by one ulp.
_BOUND_TOL = 1e-12


@dataclass(frozen=True)
class ClassicalDesign:
    """A closed-form design over a continuous factor box.

    Attributes
    ----------
    kind : str
        Which generator produced this design: ``"latin-hypercube"``,
        ``"central-composite"``, or ``"box-behnken"``.
    factors : tuple of str
        Factor names, in input order.
    bounds : tuple of (float, float)
        ``(lb, ub)`` per factor, matching ``factors``.
    rows : list of dict
        One dict per run: every factor name mapped to its natural-unit
        value, plus ``run_order`` (0-based, randomized) and ``is_center``
        (True for centre-point runs). Central-composite rows additionally
        carry ``block``, one of ``"factorial"``, ``"axial"``, or
        ``"center"``.
    """

    kind: str
    factors: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)

    def to_matrix(self) -> np.ndarray:
        """Return the runs as an ``(n_runs, n_factors)`` float array."""
        return np.array(
            [[float(r[name]) for name in self.factors] for r in self.rows],
            dtype=np.float64,
        )

    def design_rows(self) -> list[dict[str, float]]:
        """Return factor-only dicts, ready for ``Workbook.append_runs``.

        Drops the bookkeeping keys (``run_order``, ``is_center``, ``block``)
        that describe the design rather than the operating conditions.
        """
        return [{name: float(r[name]) for name in self.factors} for r in self.rows]


def _validate_factors(
    factors: Mapping[str, tuple[float, float]],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Split a factor mapping into names, lower bounds, and upper bounds."""
    if not factors:
        raise ValueError("at least one factor required")
    names = tuple(factors.keys())
    lbs: list[float] = []
    ubs: list[float] = []
    for name in names:
        pair = factors[name]
        if len(pair) != 2:
            raise ValueError(f"factor {name!r}: expected (lb, ub), got {pair!r}")
        lb, ub = (float(pair[0]), float(pair[1]))
        if not math.isfinite(lb) or not math.isfinite(ub):
            raise ValueError(f"factor {name!r}: bounds must be finite, got ({lb}, {ub})")
        if not ub > lb:
            raise ValueError(f"factor {name!r}: upper bound must exceed lower bound ({lb}, {ub})")
        lbs.append(lb)
        ubs.append(ub)
    return names, np.array(lbs, dtype=np.float64), np.array(ubs, dtype=np.float64)


def _finalize(
    kind: str,
    names: tuple[str, ...],
    lbs: np.ndarray,
    ubs: np.ndarray,
    values: np.ndarray,
    *,
    is_center: Sequence[bool],
    blocks: Sequence[str] | None = None,
    seed: int | None,
) -> ClassicalDesign:
    """Snap to bounds, randomize run order, and assemble the design."""
    # Snap only what round-off pushed a hair past a bound. A hard clip here
    # would silently fold the deliberate excursions of a textbook-scaled CCD
    # (within_bounds=False) back onto the bounds, turning a rotatable design
    # into a face-centred one without saying so.
    values = np.where(np.abs(values - lbs) <= _BOUND_TOL, lbs, values)
    values = np.where(np.abs(values - ubs) <= _BOUND_TOL, ubs, values)

    rows: list[dict[str, object]] = []
    for i, point in enumerate(values):
        row: dict[str, object] = {names[j]: float(point[j]) for j in range(len(names))}
        row["is_center"] = bool(is_center[i])
        if blocks is not None:
            row["block"] = blocks[i]
        rows.append(row)

    # Randomize execution order so uncontrolled drift (ambient temperature,
    # catalyst ageing, operator fatigue) cannot alias onto a factor.
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: d["run_order"])  # type: ignore[arg-type,return-value]

    return ClassicalDesign(
        kind=kind,
        factors=names,
        bounds=tuple((float(lo), float(hi)) for lo, hi in zip(lbs, ubs)),
        rows=rows,
    )


def _lhs_unit_samples(k: int, n: int, seed: int | None, optimize: bool) -> np.ndarray:
    """Return ``(n, k)`` Latin hypercube samples on the unit cube.

    Prefers ``scipy.stats.qmc.LatinHypercube``; falls back to an explicit
    stratified permutation when scipy is built without ``qmc`` — the same
    degradation the ``optimize`` template applies to its Sobol seeding.
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        rng = np.random.default_rng(seed)
        # One point per stratum per dimension, independently permuted.
        strata = np.tile(np.arange(n, dtype=np.float64), (k, 1))
        for dim in range(k):
            rng.shuffle(strata[dim])
        return ((strata + rng.random((k, n))) / n).T

    sampler = qmc.LatinHypercube(
        d=k,
        seed=seed,
        # "random-cd" iteratively lowers centred discrepancy, giving noticeably
        # better coverage than a plain permutation at these sample sizes.
        **({"optimization": "random-cd"} if optimize else {}),
    )
    return sampler.random(n)


def latin_hypercube_design(
    factors: Mapping[str, tuple[float, float]],
    n_samples: int,
    *,
    optimize: bool = True,
    seed: int | None = None,
) -> ClassicalDesign:
    """Build a Latin hypercube design over a continuous factor box.

    Each factor's range is divided into ``n_samples`` equal-width strata and
    exactly one run is placed in each, so every factor is sampled evenly no
    matter how many factors there are. Unlike a full factorial, the run count
    is chosen by you rather than dictated by the factor count — which is what
    makes it practical past three or four factors.

    Parameters
    ----------
    factors : mapping name -> (lb, ub)
        Continuous range per factor. All factors must be numeric; use
        :func:`~discopt.doe.screening.factorial_2level_design` for
        categorical levels.
    n_samples : int
        Number of runs. Must be at least 2. A common rule of thumb for
        surrogate fitting is ``10 * n_factors``.
    optimize : bool, default True
        Improve space-filling by minimizing centred discrepancy. Costs a
        little sampling time and needs ``scipy.stats.qmc``; set False for a
        plain stratified permutation.
    seed : int, optional
        Reproducible sampling and run-order seed.

    Returns
    -------
    ClassicalDesign
        ``kind="latin-hypercube"``. No run is a centre point.

    Examples
    --------
    >>> d = latin_hypercube_design({"T": (300.0, 400.0), "P": (1.0, 5.0)}, 8, seed=0)
    >>> len(d)
    8
    """
    names, lbs, ubs = _validate_factors(factors)
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    unit = _lhs_unit_samples(len(names), int(n_samples), seed, optimize)
    values = lbs + unit * (ubs - lbs)
    return _finalize(
        "latin-hypercube",
        names,
        lbs,
        ubs,
        values,
        is_center=[False] * len(values),
        seed=seed,
    )


def _resolve_alpha(alpha: float | str, k: int) -> float:
    """Resolve the axial distance in coded units."""
    if isinstance(alpha, str):
        if alpha == "rotatable":
            # Constant prediction variance at equal distance from the centre.
            return float(2 ** (k / 4))
        if alpha == "face":
            return 1.0
        raise ValueError(f"alpha must be a float, 'rotatable', or 'face', got {alpha!r}")
    a = float(alpha)
    if not math.isfinite(a) or a <= 0:
        raise ValueError(f"alpha must be a positive finite number, got {alpha!r}")
    return a


def central_composite_design(
    factors: Mapping[str, tuple[float, float]],
    *,
    alpha: float | str = "rotatable",
    center_points: int = 4,
    within_bounds: bool = True,
    seed: int | None = None,
) -> ClassicalDesign:
    """Build a central composite design for a quadratic response surface.

    The design is three blocks: a ``2**k`` factorial core at the coded corners
    ``(±1, ..., ±1)``, ``2k`` axial ("star") points at ``±alpha`` along each
    axis with the others at centre, and ``center_points`` replicates of the
    centre. The factorial core estimates main effects and interactions, the
    axial points make the pure quadratic terms estimable, and the replicated
    centre gives a model-free estimate of pure error.

    Parameters
    ----------
    factors : mapping name -> (lb, ub)
        Continuous range per factor, 2 to 6 factors.
    alpha : float or {"rotatable", "face"}, default "rotatable"
        Axial distance in coded units. ``"rotatable"`` uses ``2**(k/4)``,
        giving constant prediction variance at equal distance from the
        centre. ``"face"`` uses 1.0, placing the axial points on the faces of
        the cube (a face-centred design, or CCF) — the right choice when a
        factor cannot be pushed past its ±1 level at all.
    center_points : int, default 4
        Replicates at the centre. Must be at least 1 for a quadratic model to
        have a pure-error estimate.
    within_bounds : bool, default True
        Keep every run inside ``(lb, ub)``. The coded design is scaled so the
        axial points land exactly on the bounds and the factorial corners sit
        inside them. Set False for the textbook convention, where ``lb``/``ub``
        are the ±1 factorial levels and the axial points fall *outside* the
        stated range — only do this when the factor really can be driven past
        the bounds you gave. Has no effect when ``alpha`` is ``"face"``,
        since the two conventions coincide at ``alpha=1``.
    seed : int, optional
        Run-order randomization seed.

    Returns
    -------
    ClassicalDesign
        ``kind="central-composite"``. Rows carry ``block`` set to
        ``"factorial"``, ``"axial"``, or ``"center"``.

    Notes
    -----
    Total runs are ``2**k + 2k + center_points``. Beyond six factors the
    factorial core dominates (2**7 = 128 runs); use a fractional core or an
    optimal design instead.
    """
    names, lbs, ubs = _validate_factors(factors)
    k = len(names)
    if not 2 <= k <= 6:
        raise ValueError(f"central composite designs support 2 to 6 factors, got {k}")
    if center_points < 1:
        raise ValueError(f"center_points must be >= 1, got {center_points}")

    a = _resolve_alpha(alpha, k)

    coded: list[list[float]] = []
    blocks: list[str] = []
    is_center: list[bool] = []

    for corner in itertools.product((-1.0, 1.0), repeat=k):
        coded.append(list(corner))
        blocks.append("factorial")
        is_center.append(False)

    for axis in range(k):
        for sign in (-a, a):
            point = [0.0] * k
            point[axis] = sign
            coded.append(point)
            blocks.append("axial")
            is_center.append(False)

    for _ in range(center_points):
        coded.append([0.0] * k)
        blocks.append("center")
        is_center.append(True)

    coded_arr = np.array(coded, dtype=np.float64)

    # Map coded units to natural units. The divisor decides which coded level
    # the user's bounds refer to: the axial extreme (default) or the factorial
    # corner (textbook).
    mid = (lbs + ubs) / 2.0
    half = (ubs - lbs) / 2.0
    values = mid + coded_arr * (half / a if within_bounds else half)

    return _finalize(
        "central-composite",
        names,
        lbs,
        ubs,
        values,
        is_center=is_center,
        blocks=blocks,
        seed=seed,
    )


def box_behnken_design(
    factors: Mapping[str, tuple[float, float]],
    *,
    center_points: int = 3,
    seed: int | None = None,
) -> ClassicalDesign:
    """Build a Box-Behnken design for a quadratic response surface.

    For every pair of factors, all four ``(±1, ±1)`` combinations are run with
    the remaining factors held at their centre. Every point therefore lies at
    the midpoint of an edge of the design cube and no run sits at a corner —
    the property that makes this the design of choice when combining the
    extreme level of every factor at once is expensive, unsafe, or impossible.

    Parameters
    ----------
    factors : mapping name -> (lb, ub)
        Continuous range per factor, 3 to 5 factors. A quadratic surface in
        fewer than 3 factors is better served by
        :func:`central_composite_design`.
    center_points : int, default 3
        Replicates at the centre, for the pure-error estimate.
    seed : int, optional
        Run-order randomization seed.

    Returns
    -------
    ClassicalDesign
        ``kind="box-behnken"``. Every run lies within ``(lb, ub)``.

    Notes
    -----
    Total runs are ``4 * C(k, 2) + center_points``: 15 for k=3, 27 for k=4,
    and 46 for k=5 at the default centre-point counts of 3, 3, and 6, matching
    the standard published designs. Six or more factors are rejected: the
    published designs there are built from a balanced incomplete block design
    rather than all pairs, so the all-pairs construction used here would
    silently produce a larger, non-standard design.
    """
    names, lbs, ubs = _validate_factors(factors)
    k = len(names)
    if not 3 <= k <= 5:
        raise ValueError(
            f"Box-Behnken designs support 3 to 5 factors, got {k}. Use "
            "central_composite_design for 2 factors, or an optimal design "
            "(discopt.doe.design.optimal_experiment) beyond 5."
        )
    if center_points < 1:
        raise ValueError(f"center_points must be >= 1, got {center_points}")

    coded: list[list[float]] = []
    is_center: list[bool] = []

    for i, j in itertools.combinations(range(k), 2):
        for si, sj in itertools.product((-1.0, 1.0), repeat=2):
            point = [0.0] * k
            point[i] = si
            point[j] = sj
            coded.append(point)
            is_center.append(False)

    for _ in range(center_points):
        coded.append([0.0] * k)
        is_center.append(True)

    coded_arr = np.array(coded, dtype=np.float64)
    mid = (lbs + ubs) / 2.0
    half = (ubs - lbs) / 2.0
    values = mid + coded_arr * half

    return _finalize(
        "box-behnken",
        names,
        lbs,
        ubs,
        values,
        is_center=is_center,
        seed=seed,
    )


__all__ = [
    "ClassicalDesign",
    "box_behnken_design",
    "central_composite_design",
    "latin_hypercube_design",
]
