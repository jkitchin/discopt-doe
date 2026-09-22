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

The sums of squares are computed by sequential least squares (Type I, in
the order: factors as listed, then interactions), which equals the classical
marginal decomposition above whenever the terms are orthogonal and stays
correct when they are not. Two failure modes are handled explicitly:

* **Aliasing.** If a term has fewer estimable degrees of freedom than its
  nominal count once the earlier terms are fitted (two fully confounded
  factors; an interaction in a Latin square, which is confounded with a
  block), no analysis can separate it from those terms, so the function
  raises rather than report an F-ratio for a term that is partly another.
* **Non-orthogonality.** If an interaction is only *partly* confounded with
  another term (e.g. day x catalyst in a Latin square replicated with a
  different randomization, with the operator block still in the model), the
  terms are estimable but not orthogonal. The table is then reported with a
  warning and ``balanced=False``: the SS are sequential and depend on term
  order. Per-factor balance and pairwise main-effect orthogonality do *not*
  detect this case, which is why the decomposition is not taken on trust.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, cast

import numpy as np


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
            undefined = r.df == 0 or r.source == "Total"
            ms_str = "         ---" if undefined else f"{r.ms:12.4f}"
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
        factors = [k for k in rows[0].keys() if k != response and not _is_bookkeeping_column(k)]
    factors = list(factors)
    # ``replicate`` is a bookkeeping column, so automatic detection skips it and
    # a caller naming its factors has no reason to list it -- which is exactly
    # why the flag has to be honoured in both cases. Gating it on
    # ``factors is None`` made it silently inert for every caller that passed
    # ``factors=``, the common case, so a blocked analysis quietly came back
    # unblocked with the replicate variance left in the residual.
    if include_replicate and "replicate" in rows[0] and "replicate" not in factors:
        factors.append("replicate")
    if not factors:
        raise ValueError("no factor columns identified")
    missing = sorted({c for c in [response, *factors] for r in rows if c not in r})
    if missing:
        raise ValueError(
            f"column(s) {missing} missing from some rows; every row must carry the "
            "response and every factor"
        )

    y = []
    for r in rows:
        try:
            y.append(float(cast(float, r[response])))
        except (TypeError, ValueError) as e:
            raise ValueError(f"response value {r[response]!r} is not numeric") from e

    # Reject non-finite or extreme response values up front: a bare Python-float
    # ``x ** 2`` raises OverflowError for |x| ~ 1e154+, so validate before the
    # sum-of-squares computations below rather than crashing on adversarial data.
    y_arr = np.asarray(y, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.all(np.isfinite(y_arr)) or not np.isfinite(float(np.sum(y_arr**2))):
            raise ValueError(
                "response column contains non-finite or extreme values (inf/nan, "
                "or magnitudes too large to form sums of squares); check the data"
            )

    n = len(y)
    grand_mean = sum(y) / n
    ss_total = sum((yi - grand_mean) ** 2 for yi in y)

    # Centre points of a 2-level design are not a third level of every factor:
    # they share one mid value on all numeric factors at once, so as a level
    # they would alias every factor's "centre" contrast with every other's.
    # The standard analysis codes each factor -1/+1 (centre = 0, one df) and
    # spends the centre runs on a one-df curvature term plus pure error.
    center = _center_rows(rows, factors)
    coded = _coded_columns(rows, factors, center) if center else {}
    check_rows = [r for i, r in enumerate(rows) if i not in center] if center else rows
    n_check = len(check_rows)

    level_lists: dict[str, list[object]] = {}
    for f in factors:
        seen: list[object] = []
        for r in check_rows:
            v = r[f]
            if v not in seen:
                seen.append(v)
        level_lists[f] = seen

    # Balance check.
    counts: dict[str, dict[object, int]] = {f: defaultdict(int) for f in factors}
    for r in check_rows:
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
            for r in check_rows:
                cross[(r[fi], r[fj])] += 1
            for a in level_lists[fi]:
                for b in level_lists[fj]:
                    expected = counts[fi][a] * counts[fj][b] / n_check
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

    # Sums of squares by sequential least squares (Type I, in the order
    # factors, then interactions). For an orthogonal design this equals the
    # classical marginal decomposition exactly. It stays correct when the terms
    # are not orthogonal -- e.g. a day x catalyst interaction in a replicated
    # Latin square is partly confounded with the operator block even though
    # every pair of *main* effects is orthogonal, which the pairwise check
    # above cannot see and which made the marginal decomposition silently wrong.
    terms: list[tuple[str, ...]] = [(f,) for f in factors]
    for inter in interactions or ():
        inter = tuple(inter)
        for fname in inter:
            if fname not in factors:
                raise ValueError(f"interaction {inter}: factor {fname!r} not in factors")
        terms.append(inter)
    if center:
        terms.append(("curvature",))

    y_vec = np.asarray(y, dtype=float)
    X = np.ones((n, 1))
    rss_prev = ss_total
    rank_prev = 1
    effect_rows: list[AnovaEffect] = []
    df_used = 0
    ss_explained = 0.0
    exact_orthogonal = True
    for term in terms:
        if term == ("curvature",) and center:
            nominal_df = 1
            block = np.array([[1.0 if i in center else 0.0] for i in range(n)])
        elif coded and all(f in coded for f in term):
            nominal_df = 1
            block = np.prod([coded[f] for f in term], axis=0).reshape(-1, 1)
        else:
            nominal_df = 1
            for f in term:
                nominal_df *= len(level_lists[f]) - 1
            cells: dict[tuple[object, ...], int] = {}
            keys = [tuple(r[f] for f in term) for r in rows]
            for k in keys:
                cells.setdefault(k, len(cells))
            block = np.zeros((n, len(cells)))
            for i, k in enumerate(keys):
                block[i, cells[k]] = 1.0
        X_new = np.hstack([X, block])
        rank_new = int(np.linalg.matrix_rank(X_new))
        df_term = rank_new - rank_prev
        if df_term < nominal_df:
            name = ":".join(term)
            raise ValueError(
                f"term {name!r} is aliased: it has {nominal_df} degrees of freedom but "
                f"only {df_term} are not already explained by the terms before it. If "
                "a requested term is aliased with others (for example an interaction "
                "in a Latin square, which is confounded with a block), no analysis can "
                "separate them: drop the term, or replicate the design and leave one "
                "block out of the model."
            )
        coef, *_ = np.linalg.lstsq(X_new, y_vec, rcond=None)
        rss_new = float(np.sum((y_vec - X_new @ coef) ** 2))
        ss_term = max(rss_prev - rss_new, 0.0)
        if len(term) > 1 and exact_orthogonal and not center:
            # Compare with the marginal (cell-mean) decomposition an orthogonal
            # design would give; a mismatch means the order of terms matters.
            marginal = _marginal_ss(rows, y, grand_mean, term, level_lists)
            if abs(marginal - ss_term) > 1e-9 * max(1.0, ss_total):
                exact_orthogonal = False
        # A factor that is constant in the collected data contributes no
        # degrees of freedom; report the 0-df row (which __str__ renders as
        # "---") rather than dividing by zero.
        ms_term = ss_term / df_term if df_term > 0 else 0.0
        effect_rows.append(AnovaEffect(":".join(term), ss_term, df_term, ms_term, None, None))
        df_used += df_term
        ss_explained += ss_term
        X, rss_prev, rank_prev = X_new, rss_new, rank_new

    if not exact_orthogonal:
        balanced = False
        warnings.warn(
            "ANOVA terms are not mutually orthogonal (an interaction is partly "
            "confounded with another term); sums of squares are sequential "
            "(Type I) and depend on the order of the terms",
            stacklevel=2,
        )

    df_residual = n - 1 - df_used
    ss_residual = ss_total - ss_explained
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


def _center_rows(rows: Sequence[Mapping[str, object]], factors: Sequence[str]) -> set[int]:
    """Indices of centre-point runs of a 2-level design, or an empty set.

    A row is a centre point when it is flagged ``is_center`` (as
    :func:`factorial_2level_design` does), or -- for data without the flag --
    when every numeric design factor sits at the midpoint of its only two other
    values. Only designs whose non-centre runs are 2-level in every numeric
    factor qualify; anything else is analyzed with ordinary levels.
    """
    design = [f for f in factors if f != "replicate"]
    if not design:
        return set()
    flagged = {i for i, r in enumerate(rows) if bool(r.get("is_center"))}
    if flagged:
        candidates = flagged
    else:
        mids: dict[str, float] = {}
        for f in design:
            try:
                values = sorted({float(cast(float, r[f])) for r in rows})
            except (TypeError, ValueError):
                return set()
            if len(values) != 3 or abs(values[1] - (values[0] + values[2]) / 2) > 1e-9 * max(
                1.0, abs(values[2] - values[0])
            ):
                return set()
            mids[f] = values[1]
        candidates = {
            i
            for i, r in enumerate(rows)
            if all(abs(float(cast(float, r[f])) - mids[f]) <= 1e-12 for f in design)
        }
    rest = [r for i, r in enumerate(rows) if i not in candidates]
    if not candidates or not rest:
        return set()
    for f in design:
        if len({r[f] for r in rest}) != 2:
            return set()
        try:
            [float(cast(float, r[f])) for r in rows]
        except (TypeError, ValueError):
            return set()
    return candidates


def _coded_columns(
    rows: Sequence[Mapping[str, object]], factors: Sequence[str], center: set[int]
) -> dict[str, np.ndarray]:
    """-1/+1 coding of each 2-level numeric factor, with centre runs at 0."""
    out: dict[str, np.ndarray] = {}
    for f in factors:
        if f == "replicate":
            continue
        lo, hi = sorted({float(cast(float, r[f])) for i, r in enumerate(rows) if i not in center})
        out[f] = np.array(
            [
                0.0 if i in center else (-1.0 if float(cast(float, r[f])) == lo else 1.0)
                for i, r in enumerate(rows)
            ]
        )
    return out


def _marginal_ss(
    rows: Sequence[Mapping[str, object]],
    y: list[float],
    grand_mean: float,
    term: tuple[str, ...],
    level_lists: dict[str, list[object]],
) -> float:
    """Classical (orthogonal-design) SS for ``term``: cell SS minus lower orders."""
    sums: dict[tuple[object, ...], list[float]] = defaultdict(lambda: [0.0, 0])
    for yi, r in zip(y, rows):
        acc = sums[tuple(r[f] for f in term)]
        acc[0] += yi
        acc[1] += 1
    ss_cells = sum(c * (t / c - grand_mean) ** 2 for t, c in sums.values())
    if len(term) == 1:
        return ss_cells
    return ss_cells - sum(
        _marginal_ss(rows, y, grand_mean, sub, level_lists) for sub in _proper_subsets(term)
    )


def _proper_subsets(inter: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    n = len(inter)
    for mask in range(1, 2**n - 1):
        yield tuple(inter[i] for i in range(n) if mask & (1 << i))


__all__ = ["AnovaEffect", "AnovaTable", "anova_report"]
