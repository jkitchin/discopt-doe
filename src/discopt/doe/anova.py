"""ANOVA analysis for orthogonal (balanced) experimental designs.

Computes the orthogonal sum-of-squares decomposition for main effects and
optional interactions. Each main effect's SS is the *marginal* (unadjusted)
SS of its level means about the grand mean; this decomposition is exact
only when the factor columns are mutually orthogonal, which holds for
Latin-square family designs, full and fractional factorials, randomized
complete blocks, and one-way layouts.

The standard decomposition for orthogonal balanced data is

    SS_total = sum_i SS_main_i + sum_(i,j) SS_interaction_ij + SS_residual

with degrees of freedom

    df_factor   = n_levels(factor) - 1
    df_interact = prod(n_levels(f) - 1 for f in factors)
    df_residual = N - 1 - sum(df_factors) - sum(df_interactions)

F-statistics and p-values are computed against the residual mean square
via the F-distribution survival function.

For non-orthogonal data (aliased or correlated columns) the marginal SS do
not add up: the function emits a warning, and if the implied residual SS is
negative it raises rather than print a nonsensical table. Marginal balance
of each factor alone does *not* guarantee orthogonality -- fully aliased
columns pass a per-factor balance check -- so pairwise cross-tabulation is
also verified. Fit a joint linear model for genuinely non-orthogonal data.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence, cast


@dataclass(frozen=True)
class AnovaEffect:
    """One row of the ANOVA table.

    Attributes
    ----------
    source : str
        Factor name, ``"A:B"`` for an interaction, or ``"Residual"`` /
        ``"Total"`` for the summary rows.
    ss : float
        Sum of squares.
    df : int
        Degrees of freedom.
    ms : float
        Mean square (``ss / df``). For ``Total`` this is undefined and
        returned as 0.0.
    f : float | None
        F-statistic against the residual MS (``None`` for residual/total).
    p : float | None
        Upper-tail (one-sided) p-value via ``scipy.stats.f.sf`` (``None``
        for residual/total).
    """

    source: str
    ss: float
    df: int
    ms: float
    f: float | None
    p: float | None


@dataclass(frozen=True)
class AnovaTable:
    """Result of :func:`anova_report`."""

    rows: list[AnovaEffect]
    response: str
    n_obs: int
    grand_mean: float
    balanced: bool

    def summary(self) -> str:
        """Return a formatted, fixed-width ANOVA table."""
        header = f"{'Source':<20s} {'SS':>12s} {'df':>5s} {'MS':>12s} {'F':>9s} {'p':>10s}"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            f_str = "      ---" if r.f is None else f"{r.f:9.3f}"
            p_str = "       ---" if r.p is None else f"{r.p:10.4g}"
            ms_str = "     ---   " if r.df == 0 else f"{r.ms:12.4f}"
            lines.append(f"{r.source:<20s} {r.ss:12.4f} {r.df:5d} {ms_str} {f_str} {p_str}")
        if not self.balanced:
            lines.append("")
            lines.append("Note: design is unbalanced; Type-I SS depend on factor order.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _is_bookkeeping_column(name: str) -> bool:
    """Columns excluded from automatic factor detection.

    ``is_center`` marks center-point rows in factorial/fractional designs; if
    treated as a factor it perfectly confounds with the mid-level of every
    real factor and produces a wrong table. ``replicate``/``run_order`` are
    likewise bookkeeping, not factors. ``effects_estimates`` excludes the same
    set, so the two modules agree.
    """
    return name in {"replicate", "run_order", "is_center"} or name.startswith("_")


def anova_report(
    rows: Sequence[Mapping[str, object]],
    response: str,
    *,
    factors: Sequence[str] | None = None,
    interactions: Sequence[tuple[str, ...]] | None = None,
    include_replicate: bool = False,
) -> AnovaTable:
    """Compute an ANOVA table for a balanced design.

    Parameters
    ----------
    rows : sequence of dict-like
        Each row maps factor names + ``response`` -> value.
    response : str
        Column to analyze as the dependent variable.
    factors : sequence of str, optional
        Which columns to treat as factors. Defaults to every column
        except ``response`` and the bookkeeping columns
        (``replicate``, ``run_order``, anything starting with ``_``).
    interactions : sequence of tuple of str, optional
        Interaction terms to include, e.g. ``[("A", "B"), ("A", "B", "C")]``.
        Each tuple lists the factor names involved.
    include_replicate : bool, default False
        If ``True`` and a ``replicate`` column is present, treat it as
        an additional blocking factor.

    Returns
    -------
    AnovaTable
        Rows for each main effect, each requested interaction, residual,
        and total.

    Raises
    ------
    ValueError
        If the response column is missing, contains non-numeric values,
        or if fewer than two rows are provided.
    """
    from scipy.stats import f as f_dist

    rows = list(rows)
    if len(rows) < 2:
        raise ValueError("ANOVA requires at least 2 observations")
    if response not in rows[0]:
        raise ValueError(f"response column {response!r} missing from rows")

    if factors is None:
        candidates = [k for k in rows[0].keys() if k != response and not _is_bookkeeping_column(k)]
        if include_replicate and "replicate" in rows[0]:
            candidates.append("replicate")
        factors = candidates
    factors = list(factors)
    if not factors:
        raise ValueError("no factor columns identified")

    y = []
    for r in rows:
        try:
            y.append(float(cast(float, r[response])))
        except (TypeError, ValueError) as e:
            raise ValueError(f"response value {r[response]!r} is not numeric") from e

    n = len(y)
    grand_mean = sum(y) / n
    ss_total = sum((yi - grand_mean) ** 2 for yi in y)

    level_lists: dict[str, list[object]] = {}
    for f in factors:
        seen: list[object] = []
        for r in rows:
            v = r[f]
            if v not in seen:
                seen.append(v)
        level_lists[f] = seen

    # Balance check.
    counts: dict[str, dict[object, int]] = {f: defaultdict(int) for f in factors}
    for r in rows:
        for f in factors:
            counts[f][r[f]] += 1
    balanced = True
    for f in factors:
        cs = set(counts[f].values())
        if len(cs) != 1:
            balanced = False
            break
    if not balanced:
        warnings.warn(
            "ANOVA design is unbalanced; marginal SS may be misleading",
            stacklevel=2,
        )

    # Orthogonality check. The marginal SS decomposition below is exact only
    # when the factor columns are mutually orthogonal, i.e. every pair of
    # factors has proportional cross-tabulation counts. Per-factor balance
    # alone does NOT guarantee this (fully aliased columns pass it), so verify
    # pairwise proportional frequencies. A single factor is trivially
    # orthogonal (no pairs), so this only fires for >= 2 factors.
    orthogonal = True
    factor_seq = list(factors)
    for i in range(len(factor_seq)):
        if not orthogonal:
            break
        for j in range(i + 1, len(factor_seq)):
            fi, fj = factor_seq[i], factor_seq[j]
            cross: dict[tuple[object, object], int] = defaultdict(int)
            for r in rows:
                cross[(r[fi], r[fj])] += 1
            for a in level_lists[fi]:
                for b in level_lists[fj]:
                    expected = counts[fi][a] * counts[fj][b] / n
                    if abs(cross.get((a, b), 0) - expected) > 1e-9:
                        orthogonal = False
                        break
                if not orthogonal:
                    break
            if not orthogonal:
                break
    if not orthogonal:
        warnings.warn(
            "ANOVA factors are not orthogonal (aliased or correlated "
            "columns); the reported main-effect sums of squares are marginal "
            "(unadjusted) and do not decompose additively -- interpret with "
            "caution or fit a joint linear model",
            stacklevel=2,
        )

    effect_rows: list[AnovaEffect] = []

    def _cell_mean(filters: dict[str, object]) -> tuple[float, int]:
        total = 0.0
        count = 0
        for yi, r in zip(y, rows):
            if all(r[k] == v for k, v in filters.items()):
                total += yi
                count += 1
        if count == 0:
            return 0.0, 0
        return total / count, count

    # Main effects.
    df_used = 0
    ss_explained = 0.0
    main_means: dict[str, dict[object, float]] = {}
    for f in factors:
        means_f: dict[object, float] = {}
        ss_f = 0.0
        for level in level_lists[f]:
            mean_lvl, cnt = _cell_mean({f: level})
            means_f[level] = mean_lvl
            ss_f += cnt * (mean_lvl - grand_mean) ** 2
        main_means[f] = means_f
        df_f = len(level_lists[f]) - 1
        df_used += df_f
        ss_explained += ss_f
        ms_f = ss_f / df_f if df_f > 0 else 0.0
        effect_rows.append(AnovaEffect(f, ss_f, df_f, ms_f, None, None))

    # Interactions (Type-I, sequential).
    if interactions:
        for inter in interactions:
            inter = tuple(inter)
            for fname in inter:
                if fname not in factors:
                    raise ValueError(f"interaction {inter}: factor {fname!r} not in factors")
            lvl_axes = [level_lists[f] for f in inter]
            ss_cells = 0.0
            for combo in product(*lvl_axes):
                filt = dict(zip(inter, combo))
                mean_cell, cnt = _cell_mean(filt)
                if cnt == 0:
                    continue
                ss_cells += cnt * (mean_cell - grand_mean) ** 2
            # Subtract lower-order main effects that already account for
            # within-this-set variation.
            ss_inter = ss_cells - sum(
                _ss_for_subset(rows, y, grand_mean, sub, level_lists, main_means)
                for sub in _proper_subsets(inter)
            )
            df_i = 1
            for f in inter:
                df_i *= len(level_lists[f]) - 1
            df_used += df_i
            ss_explained += ss_inter
            ms_i = ss_inter / df_i if df_i > 0 else 0.0
            effect_rows.append(AnovaEffect(":".join(inter), ss_inter, df_i, ms_i, None, None))

    df_residual = n - 1 - df_used
    ss_residual = ss_total - ss_explained
    if ss_residual < -1e-9 * max(1.0, ss_total):
        raise ValueError(
            f"negative residual sum of squares ({ss_residual:.6g}): the factor "
            "columns are not orthogonal, so the marginal SS decomposition is "
            "invalid (the explained SS exceed the total). This indicates an "
            "aliased or non-orthogonal design; fit a joint linear model instead."
        )
    ss_residual = max(ss_residual, 0.0)
    if df_residual < 1:
        raise ValueError(
            f"no residual degrees of freedom (n={n}, df_used={df_used}); "
            "remove an effect or add replicates"
        )
    ms_residual = ss_residual / df_residual

    # Compute F and p now that we know the residual MS.
    final_rows: list[AnovaEffect] = []
    for eff in effect_rows:
        if eff.df == 0 or ms_residual <= 0:
            final_rows.append(eff)
            continue
        f_stat = eff.ms / ms_residual
        p = float(f_dist.sf(f_stat, eff.df, df_residual)) if f_stat > 0 else 1.0
        final_rows.append(AnovaEffect(eff.source, eff.ss, eff.df, eff.ms, f_stat, p))
    final_rows.append(AnovaEffect("Residual", ss_residual, df_residual, ms_residual, None, None))
    final_rows.append(AnovaEffect("Total", ss_total, n - 1, 0.0, None, None))

    return AnovaTable(
        rows=final_rows,
        response=response,
        n_obs=n,
        grand_mean=grand_mean,
        balanced=balanced,
    )


def _proper_subsets(inter: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    n = len(inter)
    for mask in range(1, 2**n - 1):
        yield tuple(inter[i] for i in range(n) if mask & (1 << i))


def _ss_for_subset(
    rows: Sequence[Mapping[str, object]],
    y: list[float],
    grand_mean: float,
    subset: tuple[str, ...],
    level_lists: dict[str, list[object]],
    main_means: dict[str, dict[object, float]],
) -> float:
    """SS attributable to a subset of factors -- used to subtract lower-order
    terms from a higher-order interaction (Type-I sequential decomposition).

    For a single factor this returns the main-effect SS already computed.
    For higher orders we recursively re-decompose.
    """
    if len(subset) == 1:
        f = subset[0]
        ss = 0.0
        means = main_means[f]
        for level, mean_lvl in means.items():
            cnt = sum(1 for r in rows if r[f] == level)
            ss += cnt * (mean_lvl - grand_mean) ** 2
        return ss
    # Higher-order subsets: cell SS minus lower-order subsets.
    lvl_axes = [level_lists[f] for f in subset]
    ss_cells = 0.0
    for combo in product(*lvl_axes):
        filt = dict(zip(subset, combo))
        total = 0.0
        count = 0
        for yi, r in zip(y, rows):
            if all(r[k] == v for k, v in filt.items()):
                total += yi
                count += 1
        if count == 0:
            continue
        mean_cell = total / count
        ss_cells += count * (mean_cell - grand_mean) ** 2
    return ss_cells - sum(
        _ss_for_subset(rows, y, grand_mean, sub, level_lists, main_means)
        for sub in _proper_subsets(subset)
    )


__all__ = ["AnovaEffect", "AnovaTable", "anova_report"]
