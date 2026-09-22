"""Alias structure of two- and three-level screening designs.

A fractional design saves runs by letting some effects share a contrast: the
column that estimates one effect is identical (up to sign) to the column of
another, so the data cannot tell them apart. Knowing *which* effects share a
column is what lets you read a screening experiment correctly.

Two views are provided, and :func:`alias_structure` computes both:

* **Words and resolution** (regular 2-level fractions). A *word* is a set of
  factors whose column product is constant over the runs, e.g. ``I = ABD``.
  Multiplying an effect by a word gives an effect it is aliased with
  (``A = BD``), and the length of the shortest word is the design's
  **resolution** (Box & Hunter 1961).
* **Correlations** (any design). The correlation between every pair of effect
  columns -- main effects, interactions up to ``max_order``, and pure
  quadratics for three-level designs. ``+-1`` is complete aliasing, ``0``
  none, and anything in between is the *partial* (complex) aliasing of
  nonregular designs such as Plackett-Burman or definitive screening designs
  (Hamada & Wu 1992), which have no words at all.

The design is accepted as a :class:`~discopt.doe.screening.FactorialDesign`,
a list of row dicts, or a coded matrix with entries in ``[-1, 1]``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np

_BOOKKEEPING = {"replicate", "run_order", "is_center", "block"}
_TOL = 1e-9
# Enumerating every column subset is exact but exponential; beyond this many
# factors only words up to _MAX_WORD_LEN are searched.
_FULL_WORD_SEARCH = 15
_MAX_WORD_LEN = 6


@dataclass(frozen=True)
class AliasStructure:
    """Alias structure of a screening design.

    Attributes
    ----------
    factors : tuple of str
        Factor names, in column order.
    labels : tuple of str
        Effect columns in ``correlation``: main effects, then interactions
        (``"A:B"``, ``"A:B:C"``, ...) up to ``max_order``, then pure quadratics
        (``"A^2"``) when a factor has a middle level.
    correlation : numpy.ndarray
        Correlation between the centred effect columns, ``len(labels)``
        square. A column that is constant over the runs has correlation 0
        with everything (it estimates nothing).
    words : tuple of str
        Signed words of the defining relation (``"+ABD"``, ``"-ACF"``), in
        order of length. Empty for a full factorial or a nonregular design.
    resolution : int or None
        Length of the shortest word; ``None`` when there are no words.
    regular : bool
        True when every non-centre run is at ``+-1`` on every factor and every
        pair of effect columns is either orthogonal or completely aliased.
    n_runs : int
        Number of runs analysed (including centre runs).
    words_complete : bool
        False if the word search was truncated (more than 15 factors).
    """

    factors: tuple[str, ...]
    labels: tuple[str, ...]
    correlation: np.ndarray
    words: tuple[str, ...]
    resolution: int | None
    regular: bool
    n_runs: int
    words_complete: bool = True

    def aliases(self, effect: str, *, tol: float = 1e-6) -> list[tuple[str, float]]:
        """Effects correlated with ``effect`` (largest first), as ``(label, r)``."""
        if effect not in self.labels:
            raise ValueError(f"unknown effect {effect!r}; known: {list(self.labels)}")
        i = self.labels.index(effect)
        row = self.correlation[i]
        out = [
            (self.labels[j], float(row[j]))
            for j in range(len(self.labels))
            if j != i and abs(row[j]) > tol
        ]
        out.sort(key=lambda t: (-abs(t[1]), self.labels.index(t[0])))
        return out

    def alias_groups(self) -> list[tuple[str, ...]]:
        """Sets of effect columns that are completely aliased (``|r| = 1``)."""
        seen: set[int] = set()
        groups: list[tuple[str, ...]] = []
        for i in range(len(self.labels)):
            if i in seen or not np.any(self.correlation[i]):
                continue
            members = [
                j for j in range(len(self.labels)) if abs(abs(self.correlation[i, j]) - 1.0) < 1e-6
            ]
            if len(members) > 1:
                seen.update(members)
                groups.append(tuple(self.labels[j] for j in members))
        return groups

    def summary(self, *, tol: float = 1e-6) -> str:
        """Text report: resolution, defining relation, and each main effect's aliases."""
        kind = "regular 2-level fraction" if self.regular else "nonregular design"
        lines = [f"Alias structure: {self.n_runs} runs, {len(self.factors)} factors, {kind}"]
        if self.words:
            lines.append(f"Resolution: {_roman(self.resolution)}")
            shown = list(self.words[:12])
            more = f" ... ({len(self.words)} words)" if len(self.words) > 12 else ""
            lines.append("Defining relation: I = " + " = ".join(shown) + more)
            if not self.words_complete:
                lines.append(f"(words searched up to length {_MAX_WORD_LEN} only)")
        elif self.regular:
            lines.append("No words: every effect up to the analysed order is estimable.")
        else:
            lines.append("No complete words; partial aliasing is shown as correlations.")
        lines.append("")
        width = max([len("Effect")] + [len(f) for f in self.factors])
        lines.append(f"{'Effect':<{width}}  Aliased with (correlation)")
        for f in self.factors:
            hits = self.aliases(f, tol=tol)
            text = ", ".join(f"{r:+.2g}*{lab}" for lab, r in hits) or "-"
            lines.append(f"{f:<{width}}  {text}")
        return "\n".join(lines)


def alias_structure(
    design: object,
    factors: Sequence[str] | None = None,
    *,
    max_order: int = 2,
) -> AliasStructure:
    """Words, resolution and effect-column correlations of a design.

    Parameters
    ----------
    design : FactorialDesign, sequence of dict, or array-like
        The design. A :class:`~discopt.doe.screening.FactorialDesign` is coded
        with its own ``low``/``high``; row dicts are coded from each factor's
        observed range (numeric) or sorted levels (two categorical levels);
        a matrix is used as given and must hold coded values in ``[-1, 1]``.
    factors : sequence of str, optional
        Factor names. Defaults to the design's factors, every non-bookkeeping
        column of the rows, or ``A, B, C, ...`` for a matrix.
    max_order : int, default 2
        Highest interaction order included in ``labels``/``correlation``.

    Returns
    -------
    AliasStructure
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    X, names = _coded_matrix(design, factors)
    n, k = X.shape
    if n < 2:
        raise ValueError("need at least 2 runs")

    # Effect columns: mains, interactions, quadratics (three-level factors).
    labels: list[str] = list(names)
    cols: list[np.ndarray] = [X[:, i] for i in range(k)]
    for order in range(2, min(max_order, k) + 1):
        for S in itertools.combinations(range(k), order):
            labels.append(":".join(names[s] for s in S))
            cols.append(np.prod(X[:, list(S)], axis=1))
    for i in range(k):
        if np.any(np.abs(X[:, i]) < 1 - _TOL) and np.any(np.abs(X[:, i]) > _TOL):
            labels.append(f"{names[i]}^2")
            cols.append(X[:, i] ** 2)
    C = np.column_stack(cols) - np.column_stack(cols).mean(axis=0)
    norms = np.linalg.norm(C, axis=0)
    safe = np.where(norms > _TOL, norms, 1.0)
    R = (C.T @ C) / np.outer(safe, safe)
    R[norms <= _TOL, :] = 0.0
    R[:, norms <= _TOL] = 0.0
    R[np.abs(R) < 1e-12] = 0.0

    # Words: only defined on +-1 runs. Centre runs (all zero) are ignored.
    corner = X[~np.all(np.abs(X) < _TOL, axis=1)]
    two_level = corner.size > 0 and bool(np.all(np.abs(np.abs(corner) - 1.0) < _TOL))
    words: list[str] = []
    lengths: list[int] = []
    complete = True
    if two_level:
        signs = np.sign(corner)
        max_len = k if k <= _FULL_WORD_SEARCH else _MAX_WORD_LEN
        complete = max_len >= k
        compact = all(len(nm) == 1 for nm in names)
        for size in range(1, max_len + 1):
            for S in itertools.combinations(range(k), size):
                p = np.prod(signs[:, list(S)], axis=1)
                if np.all(p == p[0]):
                    body = ("" if compact else ":").join(names[s] for s in S)
                    words.append(("+" if p[0] > 0 else "-") + body)
                    lengths.append(size)
    main_block = R[:k, :]
    partial = bool(np.any((np.abs(main_block) > 1e-6) & (np.abs(np.abs(main_block) - 1) > 1e-6)))
    regular = two_level and not partial
    resolution = min(lengths) if lengths else None
    return AliasStructure(
        factors=tuple(names),
        labels=tuple(labels),
        correlation=R,
        words=tuple(words),
        resolution=resolution,
        regular=regular,
        n_runs=n,
        words_complete=complete,
    )


def _roman(r: int | None) -> str:
    table = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
    return "none" if r is None else table.get(r, str(r))


def coded_matrix(
    design: object, factors: Sequence[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Return a design's coded run matrix (entries in [-1, 1]) and its factor names.

    Low levels map to -1, high levels to +1, and numeric levels in between
    (centre points, the middle level of a definitive screening design) to their
    coded position, e.g. 0 for a midpoint. Two-level text factors
    (``"A"``/``"B"``) are coded by sorted order.

    Parameters
    ----------
    design : FactorialDesign, sequence of dict, or array_like
        A design object, its run rows, or an already-coded matrix (validated).
    factors : sequence of str, optional
        Which factors, in which column order. Defaults to every design factor,
        skipping bookkeeping columns such as ``run_order`` and ``replicate``.

    Returns
    -------
    (numpy.ndarray, list of str)
        The ``(n_runs, n_factors)`` coded matrix and the column names.

    Examples
    --------
    >>> X, names = coded_matrix([{"T": 80, "t": 10}, {"T": 120, "t": 30}, {"T": 100, "t": 20}])
    >>> X.tolist(), names
    ([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], ['T', 't'])
    """
    return _coded_matrix(design, factors)


def _coded_matrix(design: object, factors: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    """Coded design matrix in [-1, 1] and the factor names."""
    from discopt.doe.screening import FactorialDesign

    if isinstance(design, FactorialDesign):
        names = list(factors) if factors is not None else list(design.factors)
        lo_hi = dict(zip(design.factors, zip(design.low, design.high)))
        missing = [f for f in names if f not in lo_hi]
        if missing:
            raise ValueError(f"unknown factors {missing}")
        X = np.array([[_code(r[f], *lo_hi[f]) for f in names] for r in design.rows], dtype=float)
        return X, names

    if isinstance(design, np.ndarray) or (
        isinstance(design, Sequence) and design and not isinstance(design[0], Mapping)
    ):
        X = np.asarray(design, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"design matrix must be 2-D, got shape {X.shape}")
        if np.any(np.abs(X) > 1 + _TOL):
            raise ValueError("a coded design matrix must have entries in [-1, 1]")
        names = list(factors) if factors is not None else _letters(X.shape[1])
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} factor names for {X.shape[1]} columns")
        return X, names

    rows = [cast(Mapping[str, object], r) for r in cast(Sequence[object], design)]
    if not rows:
        raise ValueError("design has no runs")
    if factors is None:
        names = [c for c in rows[0] if c not in _BOOKKEEPING and not str(c).startswith("_")]
    else:
        names = list(factors)
    cols = []
    for f in names:
        values = [r[f] for r in rows]
        levels = sorted(set(values), key=lambda v: (not _is_number(v), v))
        if all(_is_number(v) for v in levels):
            lo, hi = float(cast(float, levels[0])), float(cast(float, levels[-1]))
            cols.append([_code(v, lo, hi) for v in values])
        elif len(levels) == 2:
            cols.append([_code(v, levels[0], levels[1]) for v in values])
        else:
            raise ValueError(f"factor {f!r}: cannot code {len(levels)} non-numeric levels")
    return np.array(cols, dtype=float).T, names


def _code(v: object, lo: object, hi: object) -> float:
    if v == hi:
        return 1.0
    if v == lo:
        return -1.0
    if _is_number(v) and _is_number(lo) and _is_number(hi):
        lo_f, hi_f = float(cast(float, lo)), float(cast(float, hi))
        if hi_f == lo_f:
            return 0.0
        return 2.0 * (float(cast(float, v)) - (lo_f + hi_f) / 2.0) / (hi_f - lo_f)
    raise ValueError(f"level {v!r} is neither {lo!r} nor {hi!r}")


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _letters(k: int) -> list[str]:
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # I is reserved for the identity word
    if k <= len(alphabet):
        return list(alphabet[:k])
    return [f"x{i + 1}" for i in range(k)]


__all__ = ["AliasStructure", "alias_structure"]
