"""Screening designs beyond the regular two-level factorial.

* :func:`plackett_burman_design` -- two-level designs in run counts that are
  multiples of four (12, 20, 24 ...), filling the gaps between the powers of
  two of regular fractions (Plackett & Burman 1946). Main effects are
  orthogonal; two-factor interactions are *partially* aliased with many main
  effects instead of completely aliased with one.
* :func:`definitive_screening_design` -- three-level designs of ``2m + 1``
  runs built from a conference matrix (Jones & Nachtsheim 2011; Xiao, Lin &
  Bai 2012). Main effects are orthogonal to each other *and* to every
  two-factor interaction and pure quadratic, and each factor has a centre
  level, so curvature is visible.
* :func:`fold_over` -- the mirror-image follow-up that breaks the aliasing of
  main effects with two-factor interactions in a resolution III design (or,
  folding one factor, de-aliases that factor and its interactions).
* :func:`full_factorial_design` -- every combination of any number of levels
  per factor (a general, mixed-level factorial).
* :func:`conference_matrix` -- the Paley construction used by the DSD.

Every design is randomized: rows carry ``run_order`` (0-based, shuffled) like
the other generators in this package, so :func:`~discopt.doe.effects_estimates`,
:func:`~discopt.doe.anova_report` and :func:`~discopt.doe.aliasing.alias_structure`
apply unchanged.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np

from discopt.doe.screening import FactorialDesign

# First rows of the cyclic Plackett-Burman designs (Plackett & Burman 1946).
# The design is the N - 1 cyclic shifts of the row plus a row of minus signs.
_PB_GENERATORS: dict[int, str] = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


@dataclass(frozen=True)
class GeneralFactorialDesign:
    """A randomized full factorial with any number of levels per factor.

    Attributes
    ----------
    factors : tuple of str
        Factor names in input order.
    levels : tuple of tuple
        The levels of each factor, in input order.
    rows : list of dict
        One dict per run: every factor's level plus ``replicate`` (0-based)
        and ``run_order`` (0-based, shuffled).
    """

    factors: tuple[str, ...]
    levels: tuple[tuple[object, ...], ...]
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)


def plackett_burman_design(
    factors: Mapping[str, tuple[object, object]],
    *,
    n_runs: int | None = None,
    center_points: int = 0,
    replicates: int = 1,
    seed: int | None = None,
) -> FactorialDesign:
    """Build a randomized Plackett-Burman screening design.

    Parameters
    ----------
    factors : mapping name -> (low, high)
        Two levels per factor, numeric or categorical.
    n_runs : int, optional
        Number of runs (excluding centre points): 4, 8, 12, 16, 20, 24 or 32.
        Must exceed the number of factors. Defaults to the smallest that does.
        Powers of two use a Sylvester Hadamard matrix (a regular fraction);
        12, 20 and 24 use the cyclic Plackett-Burman generators, which are
        nonregular.
    center_points : int, default 0
        Centre runs added per replicate (numeric factors only).
    replicates : int, default 1
        Number of replications of the whole design.
    seed : int, optional
        Reproducible randomization seed.

    Returns
    -------
    FactorialDesign
        The first ``k`` columns of the ``n_runs``-run design; unused columns
        are left out (they remain available as error or "dummy" contrasts).
    """
    names, lows, highs, numeric = _two_level_factors(factors, center_points)
    k = len(names)
    supported = sorted(set(_PB_GENERATORS) | {4, 8, 16, 32})
    if n_runs is None:
        fits = [n for n in supported if n > k]
        if not fits:
            raise ValueError(f"Plackett-Burman designs here support up to 31 factors, got {k}")
        n_runs = fits[0]
    if n_runs not in supported:
        raise ValueError(f"n_runs must be one of {supported}, got {n_runs}")
    if n_runs <= k:
        raise ValueError(f"n_runs ({n_runs}) must exceed the number of factors ({k})")
    H = _pb_matrix(n_runs)[:, :k]
    return _two_level_from_matrix(names, lows, highs, H, center_points, replicates, seed)


def conference_matrix(order: int) -> np.ndarray:
    """A conference matrix of the given even order, by the Paley construction.

    A conference matrix ``C`` has a zero diagonal, ``+-1`` elsewhere, and
    ``C.T @ C = (order - 1) I``. Paley's construction needs ``order - 1`` to be
    a prime power; primes and 9 are supported here (orders 4, 6, 8, 10, 12, 14,
    18, 20, 24, ...). For ``order - 1 = 3 (mod 4)`` the result is
    antisymmetric, otherwise symmetric.
    """
    q = order - 1
    if order < 2 or order % 2:
        raise ValueError(f"conference matrix order must be even and >= 2, got {order}")
    if order == 2:
        return np.array([[0, 1], [1, 0]])
    chi = _quadratic_character(q)
    Q = np.array([[chi(j, i) for j in range(q)] for i in range(q)], dtype=int)
    C = np.zeros((order, order), dtype=int)
    C[0, 1:] = 1
    C[1:, 0] = -1 if q % 4 == 3 else 1
    C[1:, 1:] = Q
    return C


def definitive_screening_design(
    factors: Mapping[str, tuple[float, float]],
    *,
    fake_factors: int = 0,
    center_points: int = 1,
    seed: int | None = None,
) -> FactorialDesign:
    """Build a randomized definitive screening design (DSD).

    The design is a conference matrix ``C`` of order ``m``, its fold-over
    ``-C``, and centre runs: ``2m + center_points`` runs in all, each factor at
    three levels (Jones & Nachtsheim 2011). ``m`` is the number of factors
    rounded up to an even order that has a conference matrix; surplus columns
    are dropped, which is also how an odd number of factors is handled.

    Parameters
    ----------
    factors : mapping name -> (low, high)
        Numeric factors; the middle level is the midpoint.
    fake_factors : int, default 0
        Extra (unused) columns to build in. Each pair adds four runs and more
        degrees of freedom for error, the recommended way to strengthen a
        DSD's ability to fit interactions and quadratics.
    center_points : int, default 1
        Number of all-centre runs (at least 1 for the standard design).
    seed : int, optional
        Reproducible randomization seed.

    Returns
    -------
    FactorialDesign
        Rows hold low, middle or high values; the all-middle runs are flagged
        ``is_center``. ``low``/``high`` give the extremes.
    """
    names, lows, highs, numeric = _two_level_factors(factors, 0)
    if not all(numeric):
        bad = [n for n, ok in zip(names, numeric) if not ok]
        raise ValueError(f"definitive screening designs need numeric factors; got {bad}")
    if fake_factors < 0:
        raise ValueError(f"fake_factors must be >= 0, got {fake_factors}")
    if center_points < 0:
        raise ValueError(f"center_points must be >= 0, got {center_points}")
    k = len(names)
    m = max(4, k + fake_factors)
    m += m % 2
    while not _has_paley(m):
        m += 2
    C = conference_matrix(m)[:, :k]
    D = np.vstack([C, -C])
    return _two_level_from_matrix(
        names, lows, highs, D.astype(float), center_points, 1, seed, three_level=True
    )


def fold_over(
    design: FactorialDesign,
    *,
    factor: str | None = None,
    seed: int | None = None,
) -> FactorialDesign:
    """Append the mirror image of a two-level (or three-level) design.

    With ``factor=None`` every factor's sign is reversed in the new runs (the
    full fold-over): in a resolution III design this de-aliases every main
    effect from every two-factor interaction, giving resolution IV. With a
    factor name only that factor is reversed, which clears that factor's main
    effect and all its two-factor interactions.

    The original runs are kept in their order; the new runs follow in a fresh
    random order. Every row gets a ``block`` column (0 original, 1 fold-over),
    since the two halves are usually run at different times and a block term
    absorbs any shift between them. Centre runs are not duplicated.
    """
    if factor is not None and factor not in design.factors:
        raise ValueError(f"unknown factor {factor!r}; known: {list(design.factors)}")
    swap = {
        f: (lo, hi)
        for f, lo, hi in zip(design.factors, design.low, design.high)
        if factor is None or f == factor
    }
    rows = [dict(r, block=0) for r in design.rows]
    new_rows: list[dict[str, object]] = []
    for r in design.rows:
        if r.get("is_center"):
            continue
        new = dict(r, block=1)
        for f, (lo, hi) in swap.items():
            if r[f] == lo:
                new[f] = hi
            elif r[f] == hi:
                new[f] = lo
        new_rows.append(new)
    start = len(rows)
    order = list(range(len(new_rows)))
    random.Random(seed).shuffle(order)
    for new_idx, original_idx in enumerate(order):
        new_rows[original_idx]["run_order"] = start + new_idx
    new_rows.sort(key=lambda d: cast(int, d["run_order"]))
    return FactorialDesign(
        factors=design.factors, low=design.low, high=design.high, rows=rows + new_rows
    )


def full_factorial_design(
    levels: Mapping[str, Sequence[object]],
    *,
    replicates: int = 1,
    seed: int | None = None,
) -> GeneralFactorialDesign:
    """Every combination of the given levels: a general (mixed-level) factorial.

    Parameters
    ----------
    levels : mapping name -> sequence of levels
        Two or more distinct levels per factor, numeric or categorical, e.g.
        ``{"catalyst": ["A", "B", "C"], "T": [300, 350]}`` (6 runs).
    replicates : int, default 1
        Number of replications of the whole design.
    seed : int, optional
        Reproducible randomization seed.
    """
    if not levels:
        raise ValueError("at least one factor required")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    names = tuple(levels.keys())
    lv: list[tuple[object, ...]] = []
    for n in names:
        vals = tuple(levels[n])
        if len(vals) < 2:
            raise ValueError(f"factor {n!r}: need at least 2 levels, got {len(vals)}")
        if len(set(vals)) != len(vals):
            raise ValueError(f"factor {n!r}: levels must be distinct")
        lv.append(vals)
    n_cells = int(np.prod([len(v) for v in lv]))
    if n_cells * replicates > 100_000:
        raise ValueError(f"design would have {n_cells * replicates} runs; too large")
    rows: list[dict[str, object]] = []
    for r in range(replicates):
        for combo in itertools.product(*lv):
            row: dict[str, object] = dict(zip(names, combo))
            row["replicate"] = r
            rows.append(row)
    _randomize(rows, seed)
    return GeneralFactorialDesign(factors=names, levels=tuple(lv), rows=rows)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _pb_matrix(n: int) -> np.ndarray:
    """``n x (n-1)`` two-level design with orthogonal, balanced columns."""
    if n in _PB_GENERATORS:
        g = np.array([1 if c == "+" else -1 for c in _PB_GENERATORS[n]])
        rows = [np.roll(g, i) for i in range(n - 1)] + [-np.ones(n - 1, dtype=int)]
        return np.array(rows)
    H = np.array([[1]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:, 1:]  # drop the all-ones column


def _is_prime(q: int) -> bool:
    return q >= 2 and all(q % d for d in range(2, int(q**0.5) + 1))


def _has_paley(order: int) -> bool:
    q = order - 1
    return order == 2 or _is_prime(q) or q == 9


def _quadratic_character(q: int):
    """``chi(a, b)`` = quadratic character of ``a - b`` in GF(q), q prime or 9."""
    if _is_prime(q):
        squares = {(i * i) % q for i in range(1, q)}

        def chi_p(a: int, b: int) -> int:
            d = (a - b) % q
            return 0 if d == 0 else (1 if d in squares else -1)

        return chi_p
    if q == 9:
        # GF(9) = GF(3)[i] / (i^2 + 1); element e <-> (e // 3) + (e % 3) i.
        def mul(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
            return ((x[0] * y[0] - x[1] * y[1]) % 3, (x[0] * y[1] + x[1] * y[0]) % 3)

        elems = [(e // 3, e % 3) for e in range(9)]
        squares9 = {mul(x, x) for x in elems if x != (0, 0)}

        def chi_9(a: int, b: int) -> int:
            d = ((a // 3 - b // 3) % 3, (a % 3 - b % 3) % 3)
            return 0 if d == (0, 0) else (1 if d in squares9 else -1)

        return chi_9
    raise ValueError(f"no Paley conference matrix for order {q + 1}")


def _two_level_factors(
    factors: Mapping[str, tuple[object, object]], center_points: int
) -> tuple[tuple[str, ...], list[object], list[object], list[bool]]:
    if not factors:
        raise ValueError("at least one factor required")
    if center_points < 0:
        raise ValueError(f"center_points must be >= 0, got {center_points}")
    names = tuple(factors.keys())
    lows: list[object] = []
    highs: list[object] = []
    numeric: list[bool] = []
    for n in names:
        lo, hi = factors[n]
        if lo == hi:
            raise ValueError(f"factor {n!r}: low and high levels must differ")
        lows.append(lo)
        highs.append(hi)
        numeric.append(_is_numeric(lo) and _is_numeric(hi))
    if center_points > 0 and not all(numeric):
        bad = [n for n, ok in zip(names, numeric) if not ok]
        raise ValueError(
            f"center_points > 0 requires all factors to be numeric; non-numeric factors: {bad}"
        )
    return names, lows, highs, numeric


def _two_level_from_matrix(
    names: tuple[str, ...],
    lows: list[object],
    highs: list[object],
    M: np.ndarray,
    center_points: int,
    replicates: int,
    seed: int | None,
    *,
    three_level: bool = False,
) -> FactorialDesign:
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    k = len(names)

    def mid(i: int) -> float:
        return (float(cast(float, lows[i])) + float(cast(float, highs[i]))) / 2.0

    rows: list[dict[str, object]] = []
    for r in range(replicates):
        for code in M:
            row: dict[str, object] = {}
            for i in range(k):
                c = code[i]
                row[names[i]] = highs[i] if c > 0 else (lows[i] if c < 0 else mid(i))
            row["replicate"] = r
            row["is_center"] = bool(three_level and np.all(code == 0))
            rows.append(row)
        for _ in range(center_points):
            cp: dict[str, object] = {names[i]: mid(i) for i in range(k)}
            cp["replicate"] = r
            cp["is_center"] = True
            rows.append(cp)
    _randomize(rows, seed)
    return FactorialDesign(factors=names, low=tuple(lows), high=tuple(highs), rows=rows)


def _randomize(rows: list[dict[str, object]], seed: int | None) -> None:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: cast(int, d["run_order"]))


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


__all__ = [
    "GeneralFactorialDesign",
    "conference_matrix",
    "definitive_screening_design",
    "fold_over",
    "full_factorial_design",
    "plackett_burman_design",
]
