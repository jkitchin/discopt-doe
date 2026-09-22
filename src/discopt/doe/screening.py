"""Two-level factorial designs for factor screening.

When to use this template
-------------------------

You have a list of candidate factors (process knobs, ingredients,
operating modes, ...) and the question is **"does each one matter?"** —
not "what is the optimum" and not "what is the full response surface".
Screening designs deliberately probe each factor at only **two levels**
(LOW and HIGH) so a small number of runs can resolve which factors
have detectable effects on the response.

Use a 2-level factorial design when:

* You have **2 -- 7 candidate factors** and want to filter the "vital
  few" before investing in a quadratic response-surface or mechanistic
  model. Beyond ~7 factors, a fractional or Plackett-Burman design
  scales better, but the same diagnostic logic applies.
* Each factor has a sensible LOW and HIGH setting. For continuous
  factors these are usually the extremes of the safe operating range;
  for categorical factors they are the two values you actually want to
  compare (e.g. catalyst A vs B, supplier 1 vs 2).
* You can afford ``2**k`` runs at minimum (plus optional center points
  + replicates). For ``k = 3`` factors that's 8 runs; for ``k = 5``
  it's 32. If that's too many, consider a half-fraction (resolution V
  for k = 5: 16 runs) -- not yet implemented here; use the ``linear``
  template + ``discopt doe fit`` instead, which produces an
  exact-D-optimal design at the same factor count.

Avoid this template when:

* You already know all factors matter and want to **optimize**: use
  ``response-surface-2d`` / ``response-surface-3d`` for a full quadratic.
* You only have **one** factor: a one-way ANOVA via ``latin-square`` with
  k = 1 treatment is overkill -- a paired t-test on completed runs is
  enough, or just use the ``polynomial-1d`` template.
* Your factors are **proportions of a blend** summing to a constant:
  use the Scheffé mixture templates instead.

Center points and curvature
---------------------------

Adding ``center_points = c > 0`` inserts ``c`` runs at the midpoint of
every continuous factor. Center points serve two purposes:

1. **Curvature test**: comparing the average corner response to the
   average center response detects whether a planar (additive) model is
   adequate. If center response is far from the predicted plane, you
   know to escalate to a response-surface design.
2. **Pure-error estimate**: replicated center points give an unbiased
   estimate of σ that doesn't assume the linear model is correct.

Center points require **all factors to be numeric**. If any factor is
categorical (e.g. ``catalyst: [A, B]``), the midpoint is undefined and
the design rejects ``center_points > 0``.

Replication
-----------

``replicates = r`` repeats the whole 2**k corner set ``r`` times
(plus ``center_points`` per replicate). The randomization is fresh for
each replicate. Replication is necessary when you need a residual
degrees-of-freedom > 0 for the F-tests:

* With ``r = 1`` and no interactions, residual df = ``2**k - 1 - k``.
* Adding 2-way interactions consumes ``k * (k - 1) / 2`` more df.
* For ``k = 3`` and a full interactions model, residual df is 0 unless
  you add center points or replicates.

Analysis
--------

After filling in the response column, run::

    discopt doe anova WORKBOOK [--interaction A:B ...]

which reports Type-I sums of squares + F + p-value for each factor and
each requested interaction. The signed **effect estimate** for a
factor is ``mean(y | factor=HIGH) - mean(y | factor=LOW)``; that and
its standard error are returned by :func:`effects_estimates`. A Pareto
chart of ``|effect|`` is the standard visualization.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence, cast

import numpy as np


@dataclass(frozen=True)
class FactorialDesign:
    """A randomized 2-level factorial design.

    Attributes
    ----------
    factors : tuple of str
        Factor names in input order.
    low : tuple
        LOW level of each factor (numeric or string).
    high : tuple
        HIGH level of each factor (numeric or string).
    rows : list of dict
        One dict per run with all factor values, plus ``replicate``
        (0-based), ``run_order`` (0-based, shuffled), and ``is_center``
        (True for center-point runs).
    """

    factors: tuple[str, ...]
    low: tuple[object, ...]
    high: tuple[object, ...]
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)


def factorial_2level_design(
    factors: Mapping[str, tuple[object, object]],
    *,
    center_points: int = 0,
    replicates: int = 1,
    seed: int | None = None,
) -> FactorialDesign:
    """Build a randomized 2-level full factorial design.

    Parameters
    ----------
    factors : mapping name -> (low, high)
        Two levels per factor. Levels may be numeric (e.g.
        ``("temp", (80.0, 120.0))``) or categorical (e.g.
        ``("catalyst", ("A", "B"))``).
    center_points : int, default 0
        Number of center-point runs added **per replicate**. Requires
        all factors to be numeric. The center value is the midpoint of
        the (low, high) pair.
    replicates : int, default 1
        Number of independent replications of the whole 2**k + cp set.
    seed : int, optional
        Reproducible randomization seed.

    Returns
    -------
    FactorialDesign
        Rows are tagged with ``replicate``, ``run_order``, and
        ``is_center``. Run order is shuffled across the whole experiment.
    """
    if not factors:
        raise ValueError("at least one factor required")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    if center_points < 0:
        raise ValueError(f"center_points must be >= 0, got {center_points}")

    names = tuple(factors.keys())
    lows: list[object] = []
    highs: list[object] = []
    numeric_flags: list[bool] = []
    for n in names:
        lo, hi = factors[n]
        if lo == hi:
            raise ValueError(f"factor {n!r}: low and high levels must differ")
        lows.append(lo)
        highs.append(hi)
        numeric_flags.append(_is_numeric(lo) and _is_numeric(hi))

    if center_points > 0 and not all(numeric_flags):
        non_numeric = [n for n, ok in zip(names, numeric_flags) if not ok]
        raise ValueError(
            f"center_points > 0 requires all factors to be numeric; "
            f"non-numeric factors: {non_numeric}"
        )

    k = len(names)
    if k > 8:
        raise ValueError(f"2-level factorial supports up to 8 factors, got {k}")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    corner_codes = list(itertools.product((0, 1), repeat=k))  # 2**k corners
    for r in range(replicates):
        rep_rows: list[dict[str, object]] = []
        for code in corner_codes:
            row: dict[str, object] = {names[i]: highs[i] if code[i] else lows[i] for i in range(k)}
            row["replicate"] = r
            row["is_center"] = False
            rep_rows.append(row)
        for _ in range(center_points):
            cp_row: dict[str, object] = {
                names[i]: (float(cast(float, lows[i])) + float(cast(float, highs[i]))) / 2.0
                for i in range(k)
            }
            cp_row["replicate"] = r
            cp_row["is_center"] = True
            rep_rows.append(cp_row)
        rows.extend(rep_rows)

    order = list(range(len(rows)))
    rng.shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: cast(int, d["run_order"]))

    return FactorialDesign(
        factors=names,
        low=tuple(lows),
        high=tuple(highs),
        rows=rows,
    )


def effects_estimates(
    rows: Sequence[Mapping[str, object]],
    response: str,
    factors: Sequence[str] | None = None,
    levels: Mapping[str, tuple[object, object]] | None = None,
    *,
    interactions: Sequence[tuple[str, ...]] | None = None,
) -> list[dict[str, object]]:
    """Signed effect estimates for a 2-level design.

    For each factor, returns ``mean(y | factor=HIGH) - mean(y | factor=LOW)``
    together with a standard error, t-statistic and two-sided p-value. With
    ``interactions`` (e.g. ``[("A", "B"), ("A", "B", "C")]``) the same contrast
    is computed for each product column: ``mean(y | +1) - mean(y | -1)``.

    The standard error is derived from the residual of the *joint* model (all
    requested effects fitted simultaneously by least squares; the columns of a
    2-level design are orthogonal), so one effect's real size does not inflate
    another's standard error. Center-point runs contribute to that residual.
    Effects left out of the model end up in the residual: if a real
    interaction is omitted, every standard error is inflated.

    When the model leaves **no residual degrees of freedom** (an unreplicated
    design with every effect estimated) there is nothing to estimate sigma
    from, so the standard error falls back to Lenth's pseudo standard error
    (:func:`lenth_pse`), which borrows it from the smaller effects under
    effect sparsity. The ``method`` key says which was used.

    The estimates are contrasts, exact for orthogonal designs. In a design with
    partial aliasing (Plackett-Burman, definitive screening) a contrast also
    picks up a fraction of the effects correlated with it; see
    :func:`discopt.doe.aliasing.alias_structure`.

    ``levels`` fixes each factor's ``(low, high)`` orientation (e.g. from a
    :class:`FactorialDesign`'s ``low``/``high``). Without it the low/high are
    taken as the sorted min/max of the observed values, which can flip an
    effect's sign for categorical (``("B", "A")``) or reversed-numeric
    (``(120, 80)``) levels -- pass ``levels`` to be safe.

    Returns
    -------
    list of dict
        One entry per effect with keys ``factor`` (``"A:B"`` for an
        interaction), ``effect``, ``se``, ``t``, ``p``, ``low``, ``high``
        (``-1``/``+1`` for an interaction), ``method`` (``"residual"`` or
        ``"lenth"``) and ``df``, the degrees of freedom behind ``se``: the
        residual df, or Lenth's ``m/3`` for ``"lenth"``. Use it for the ``t``
        critical value of a confidence interval. Sorted by ``|effect|``
        descending.
    """
    from scipy import stats

    rows = list(rows)
    if not rows:
        raise ValueError("need at least one row")
    if response not in rows[0]:
        raise ValueError(f"response column {response!r} missing from rows")

    if factors is None:
        factors = [
            k
            for k in rows[0].keys()
            if k != response
            and k not in {"replicate", "run_order", "is_center"}
            and not k.startswith("_")
        ]
    factors = list(factors)

    y = []
    for r in rows:
        try:
            y.append(float(cast(float, r[response])))
        except (TypeError, ValueError) as e:
            raise ValueError(f"response value {r[response]!r} is not numeric") from e

    # Guard against non-finite / extreme responses (bare float ** 2 overflows).
    y_arr = np.asarray(y, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.all(np.isfinite(y_arr)) or not np.isfinite(float(np.sum(y_arr**2))):
            raise ValueError(
                "response column contains non-finite or extreme values (inf/nan, "
                "or magnitudes too large to form sums of squares); check the data"
            )

    n = len(y)

    # Determine each factor's (low, high). Prefer the caller-supplied
    # orientation; otherwise fall back to the sorted min/max of observed
    # values (which can flip the sign for categorical / reversed levels).
    levels_per_factor: dict[str, tuple[object, object]] = {}
    for f in factors:
        vals: list[Any] = []
        for r in rows:
            if r[f] not in vals:
                vals.append(r[f])
        if len(vals) < 2:
            continue
        if levels and f in levels:
            lo, hi = levels[f]
        else:
            try:
                sorted_vals = sorted(vals)
            except TypeError:
                sorted_vals = vals
            lo, hi = sorted_vals[0], sorted_vals[-1]
        levels_per_factor[f] = (lo, hi)

    # Coded column per factor: +1 at HIGH, -1 at LOW, 0 anywhere else (centre
    # points, the middle level of a definitive screening design).
    coded: dict[str, np.ndarray] = {
        f: np.array([1.0 if r[f] == hi else (-1.0 if r[f] == lo else 0.0) for r in rows])
        for f, (lo, hi) in levels_per_factor.items()
    }

    # (label, column, low, high) for every effect to estimate.
    terms: list[tuple[str, np.ndarray, object, object]] = []
    for f, (lo, hi) in levels_per_factor.items():
        col = coded[f]
        if np.any(col > 0) and np.any(col < 0):
            terms.append((f, col, lo, hi))
    for inter in interactions or ():
        inter = tuple(inter)
        missing = [f for f in inter if f not in coded]
        if missing:
            raise ValueError(f"interaction {inter}: unknown or constant factor(s) {missing}")
        if len(set(inter)) != len(inter) or len(inter) < 2:
            raise ValueError(f"interaction {inter} needs at least two distinct factors")
        col = np.prod([coded[f] for f in inter], axis=0)
        if np.any(col > 0) and np.any(col < 0):
            terms.append((":".join(inter), col, -1, 1))

    # Signed contrast for each effect.
    contrasts: list[tuple[str, float, int, int, object, object]] = []
    for label, col, lo, hi in terms:
        y_hi = y_arr[col > 0]
        y_lo = y_arr[col < 0]
        contrasts.append((label, float(y_hi.mean() - y_lo.mean()), y_lo.size, y_hi.size, lo, hi))

    # Residual of the joint least-squares model on the same columns.
    X = np.column_stack([np.ones(n)] + [col for _, col, _, _ in terms])
    coef, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    resid_ss = float(np.sum((y_arr - X @ coef) ** 2))
    df_resid = n - int(np.linalg.matrix_rank(X))

    effects: list[dict[str, object]] = []
    if df_resid > 0:
        sigma2 = resid_ss / df_resid
        for label, effect, n_lo, n_hi, lo, hi in contrasts:
            se = math.sqrt(sigma2 * (1.0 / n_lo + 1.0 / n_hi))
            t = effect / se if se > 0 else float("nan")
            p = float(2.0 * stats.t.sf(abs(t), df_resid)) if math.isfinite(t) else float("nan")
            effects.append(
                {
                    "factor": label,
                    "effect": effect,
                    "se": se,
                    "t": t,
                    "p": p,
                    "low": lo,
                    "high": hi,
                    "method": "residual",
                    "df": df_resid,
                }
            )
    else:
        vals = [c[1] for c in contrasts]
        if len(vals) >= 3:
            pse = lenth_pse(vals).pse
            d = len(vals) / 3.0
        else:
            pse, d = float("nan"), float("nan")
        for label, effect, _n_lo, _n_hi, lo, hi in contrasts:
            if math.isfinite(pse) and pse > 0:
                t = effect / pse
                p = float(2.0 * stats.t.sf(abs(t), d))
            else:
                t = p = float("nan")
            effects.append(
                {
                    "factor": label,
                    "effect": effect,
                    "se": pse,
                    "t": t,
                    "p": p,
                    "low": lo,
                    "high": hi,
                    "method": "lenth",
                    "df": d,
                }
            )

    effects.sort(key=lambda d: abs(float(cast(float, d["effect"]))), reverse=True)
    return effects


class LenthResult(NamedTuple):
    """Lenth's pseudo standard error and the margins of error built on it."""

    pse: float
    margin_of_error: float
    simultaneous_margin: float


def lenth_pse(effects: Sequence[float], *, alpha: float = 0.05) -> LenthResult:
    """Lenth's pseudo standard error for an unreplicated 2-level design.

    With every effect estimated there are no residual degrees of freedom. Under
    effect sparsity most effects are pure noise, so their typical size
    estimates the standard error of one effect (Lenth 1989):

    ``s0 = 1.5 * median|c|``, then ``PSE = 1.5 * median{|c| : |c| < 2.5 s0}``.

    Parameters
    ----------
    effects : sequence of float
        All ``m`` effect estimates (at least 3).
    alpha : float, default 0.05
        Two-sided level of the margins.

    Returns
    -------
    LenthResult
        ``(pse, margin_of_error, simultaneous_margin)``. The margin of error
        is ``t_{1-alpha/2, m/3} * PSE``, an individual 95 % interval
        half-width; the simultaneous margin uses
        ``gamma = (1 + (1 - alpha)**(1/m)) / 2`` in place of ``1 - alpha/2`` and
        controls the chance of *any* false alarm among the ``m`` effects.
    """
    from scipy import stats

    c = np.abs(np.asarray(list(effects), dtype=float))
    m = c.size
    if m < 3:
        raise ValueError(f"Lenth's method needs at least 3 effects, got {m}")
    if not np.all(np.isfinite(c)):
        raise ValueError("effects must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    s0 = 1.5 * float(np.median(c))
    trimmed = c[c < 2.5 * s0]
    pse = 1.5 * float(np.median(trimmed)) if trimmed.size else 0.0
    d = m / 3.0
    me = float(stats.t.ppf(1.0 - alpha / 2.0, d)) * pse
    gamma = (1.0 + (1.0 - alpha) ** (1.0 / m)) / 2.0
    sme = float(stats.t.ppf(gamma, d)) * pse
    return LenthResult(pse, me, sme)


@dataclass(frozen=True)
class HalfNormalScores:
    """Coordinates for a half-normal plot of effects (Daniel 1959).

    ``abs_effects`` ascending, with the matching ``labels`` and the half-normal
    ``quantiles`` ``Phi^-1(0.5 + 0.5 (i - 0.5) / m)``. Inactive effects fall on
    a line through the origin; active ones stand off it to the upper right.
    """

    labels: tuple[str, ...]
    abs_effects: np.ndarray
    quantiles: np.ndarray


def half_normal_scores(
    effects: Sequence[float] | Mapping[str, float] | Sequence[Mapping[str, object]],
) -> HalfNormalScores:
    """Half-normal plotting positions for a set of effects.

    Accepts a plain sequence of numbers, a mapping ``label -> effect``, or the
    list returned by :func:`effects_estimates`. Returns data only; plot
    ``quantiles`` (x) against ``abs_effects`` (y) with any library.
    """
    from scipy import stats

    if isinstance(effects, Mapping):
        labels = [str(k) for k in effects]
        values = [float(v) for v in effects.values()]
    else:
        items = list(effects)
        if items and isinstance(items[0], Mapping):
            labels = [str(cast(Mapping[str, object], e)["factor"]) for e in items]
            values = [float(cast(float, cast(Mapping[str, object], e)["effect"])) for e in items]
        else:
            values = [float(cast(float, v)) for v in items]
            labels = [str(i) for i in range(len(values))]
    if not values:
        raise ValueError("need at least one effect")
    a = np.abs(np.asarray(values))
    order = np.argsort(a, kind="stable")
    m = a.size
    q = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, m + 1) - 0.5) / m)
    return HalfNormalScores(
        labels=tuple(labels[i] for i in order), abs_effects=a[order], quantiles=q
    )


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


__all__ = [
    "FactorialDesign",
    "HalfNormalScores",
    "LenthResult",
    "effects_estimates",
    "factorial_2level_design",
    "half_normal_scores",
    "lenth_pse",
]
