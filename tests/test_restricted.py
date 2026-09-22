"""Tests for discopt.doe.restricted: blocked factorials and split-plot designs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from discopt.doe.anova import anova_report
from discopt.doe.restricted import (
    blocked_factorial_design,
    split_plot_anova,
    split_plot_design,
)

ABC = {"A": (-1, 1), "B": (-1, 1), "C": (-1, 1)}


def _contrast(rows, word, factors):
    """±1 column of the interaction ``word`` ("A:B") for 2-level rows coded -1/+1."""
    col = np.ones(len(rows))
    for f in word.split(":"):
        col *= np.array([float(r[f]) for r in rows])
    return col


def _block_indicators(rows):
    blocks = sorted({r["block"] for r in rows})
    B = np.array([[1.0 if r["block"] == b else 0.0 for b in blocks] for r in rows])
    return B - B.mean(axis=0)  # centred: the block space orthogonal to the mean


def _all_effects(factors):
    names = list(factors)
    return [":".join(c) for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]


# ---------------------------------------------------------------------------
# Blocked factorials
# ---------------------------------------------------------------------------


def test_2cubed_in_two_blocks_confounds_abc_only():
    d = blocked_factorial_design(ABC, n_blocks=2, seed=1)
    assert d.generators == (("A:B:C",),)
    assert d.confounded_effects == (("A:B:C",),)
    assert len(d) == 8 and {r["block"] for r in d.rows} == {1, 2}
    B = _block_indicators(d.rows)
    for effect in _all_effects(ABC):
        overlap = np.abs(_contrast(d.rows, effect, ABC) @ B).max()
        if effect == "A:B:C":
            assert overlap > 0  # the block difference *is* the ABC contrast
        else:
            assert overlap == pytest.approx(0.0)  # everything else orthogonal to blocks


def test_runs_are_randomized_within_blocks_and_blocks_are_contiguous():
    d = blocked_factorial_design(ABC, n_blocks=2, replicates=2, seed=3)
    blocks_in_order = [r["block"] for r in sorted(d.rows, key=lambda r: r["run_order"])]
    # Each block occupies one contiguous stretch of the run sequence.
    changes = sum(1 for a, b in zip(blocks_in_order, blocks_in_order[1:]) if a != b)
    assert changes == 3
    assert {r["block"] for r in d.rows} == {1, 2, 3, 4}  # unique across replicates


def _wordlengths(confounded):
    return sorted(len(w.split(":")) for w in confounded)


@pytest.mark.parametrize(
    "k, n_blocks, expected",
    [
        # Minimum-aberration blocking of full factorials (e.g. Montgomery 2017,
        # Table 7.8; Sun, Wu & Chen 1997): word-length patterns of the confounded set.
        (4, 4, [2, 3, 3]),
        (5, 4, [3, 3, 4]),
        (6, 8, [3, 3, 3, 3, 4, 4, 4]),
        (7, 8, [4, 4, 4, 4, 4, 4, 4]),
    ],
)
def test_default_generators_are_minimum_aberration(k, n_blocks, expected):
    factors = {c: (-1, 1) for c in "ABCDEFG"[:k]}
    d = blocked_factorial_design(factors, n_blocks=n_blocks, seed=0)
    assert _wordlengths(d.confounded_effects[0]) == expected
    B = _block_indicators(d.rows)
    confounded = set(d.confounded_effects[0])
    for effect in _all_effects(factors):
        overlap = np.abs(_contrast(d.rows, effect, factors) @ B).max()
        assert (overlap > 1e-9) == (effect in confounded)


def test_explicit_generators_and_validation():
    factors = {c: (-1, 1) for c in "ABCD"}
    d = blocked_factorial_design(factors, n_blocks=4, block_generators=["ABC", "ACD"], seed=0)
    assert set(d.confounded_effects[0]) == {"A:B:C", "A:C:D", "B:D"}
    d2 = blocked_factorial_design(
        {"temp": (1, 2), "time": (1, 2), "cat": ("a", "b")},
        n_blocks=2,
        block_generators=[("temp", "time", "cat")],
        seed=0,
    )
    assert d2.confounded_effects == (("temp:time:cat",),)
    with pytest.raises(ValueError, match="main effect"):
        blocked_factorial_design(factors, n_blocks=4, block_generators=["AB", "ABC"])
    with pytest.raises(ValueError, match="independent"):
        blocked_factorial_design(factors, n_blocks=4, block_generators=["ABC", "ABC"])
    with pytest.raises(ValueError, match="power of two"):
        blocked_factorial_design(factors, n_blocks=3)


def test_partial_confounding_recovers_every_interaction_somewhere():
    d = blocked_factorial_design(ABC, n_blocks=2, replicates=4, partial_confounding=True, seed=0)
    per_rep = [set(c) for c in d.confounded_effects]
    assert len({frozenset(c) for c in per_rep}) == 4  # different in every replicate
    for effect in ["A:B", "A:C", "B:C", "A:B:C"]:
        clear = [
            r
            for r in range(4)
            if np.abs(
                _contrast([x for x in d.rows if x["replicate"] == r], effect, ABC)
                @ _block_indicators([x for x in d.rows if x["replicate"] == r])
            ).max()
            < 1e-9
        ]
        assert clear, f"{effect} is confounded in every replicate"


def test_center_points_per_block_and_anova_with_blocks():
    d = blocked_factorial_design(ABC, n_blocks=2, center_points_per_block=2, seed=2)
    assert sum(r["is_center"] for r in d.rows) == 4
    assert all(sum(r["is_center"] for r in d.rows if r["block"] == b) == 2 for b in (1, 2))
    rng = np.random.default_rng(0)
    rows = [dict(r, y=3.0 * float(r["A"]) + 2.0 * (r["block"] == 2) + rng.normal()) for r in d.rows]
    table = anova_report(rows, "y", factors=["A", "B", "C", "block"])
    assert {e.source for e in table.rows} >= {"A", "B", "C", "block"}


# ---------------------------------------------------------------------------
# Split-plot designs
# ---------------------------------------------------------------------------


def test_split_plot_structure():
    d = split_plot_design(
        {"T": (-1, 1)}, {"resin": ("x", "y"), "surface": (-1, 1)}, whole_plot_replicates=4, seed=5
    )
    assert d.n_whole_plots == 8 and len(d) == 32
    for w in range(d.n_whole_plots):
        members = [r for r in d.rows if r["whole_plot"] == w]
        assert len(members) == 4
        assert len({r["T"] for r in members}) == 1  # whole-plot factor constant
        assert {(r["resin"], r["surface"]) for r in members} == set(
            itertools.product(("x", "y"), (-1, 1))
        )
    order = [r["whole_plot"] for r in sorted(d.rows, key=lambda r: r["run_order"])]
    assert order == sorted(order)  # each whole plot is run as one contiguous batch
    assert sum(r["T"] == 1 for r in d.rows) == 16


def _simulate(d, rng, temp_effect=0.0, sd_wp=1.5, sd_sp=0.5):
    wp_err = rng.normal(0, sd_wp, d.n_whole_plots)
    return [
        dict(
            r,
            y=50
            + temp_effect / 2 * r["T"]
            + 1.0 * r["resin"]
            + 0.5 * r["surface"]
            + 0.4 * r["T"] * r["resin"]
            + wp_err[r["whole_plot"]]
            + rng.normal(0, sd_sp),
        )
        for r in d.rows
    ]


SP_KW = dict(
    whole_plot_factors=["T"],
    sub_plot_factors=["resin", "surface"],
    interactions=[
        ("resin", "surface"),
        ("T", "resin"),
        ("T", "surface"),
        ("T", "resin", "surface"),
    ],
)


def test_split_plot_anova_strata_and_df():
    d = split_plot_design(
        {"T": (-1, 1)}, {"resin": (-1, 1), "surface": (-1, 1)}, whole_plot_replicates=4, seed=81
    )
    rows = _simulate(d, np.random.default_rng(1))
    tab = split_plot_anova(rows, "y", **SP_KW)
    by = {e.source: e for e in tab.rows}
    assert tab.strata["T"] == "whole-plot"
    assert tab.strata["T:resin"] == "sub-plot" and tab.strata["resin"] == "sub-plot"
    assert by["Whole-plot error"].df == 8 - 1 - 1
    assert by["Sub-plot error"].df == 32 - 8 - 6
    # T is tested against whole-plot error, resin against sub-plot error.
    assert by["T"].f == pytest.approx(by["T"].ms / by["Whole-plot error"].ms)
    assert by["resin"].f == pytest.approx(by["resin"].ms / by["Sub-plot error"].ms)
    # Strata add up to the total.
    parts = sum(e.ss for e in tab.rows if e.source != "Total")
    assert parts == pytest.approx(by["Total"].ss)

    # Whole-plot stratum by hand: regress whole-plot means on T.
    plot_mean = {w: np.mean([r["y"] for r in rows if r["whole_plot"] == w]) for w in range(8)}
    t_level = {r["whole_plot"]: r["T"] for r in rows}
    ym = np.array([plot_mean[w] for w in range(8)])
    X = np.column_stack([np.ones(8), [t_level[w] for w in range(8)]])
    coef, *_ = np.linalg.lstsq(X, ym, rcond=None)
    ss_wp_err = 4 * float(np.sum((ym - X @ coef) ** 2))
    assert by["Whole-plot error"].ss == pytest.approx(ss_wp_err)

    # Effects: T's standard error comes from the whole-plot stratum.
    eff = {e["term"]: e for e in tab.effects}
    assert eff["T"]["df"] == 6 and eff["resin"]["df"] == 18
    assert eff["T"]["se"] == pytest.approx(np.sqrt(4 * by["Whole-plot error"].ms / 32))
    assert tab.variance_components["sub_plot"] == pytest.approx(by["Sub-plot error"].ms)


def test_split_plot_anova_holds_its_false_alarm_rate():
    """A null whole-plot factor is declared significant about 5% of the time by
    the split-plot analysis; the completely randomized analysis, which pools the
    two errors, does so far more often."""
    d = split_plot_design(
        {"T": (-1, 1)}, {"resin": (-1, 1), "surface": (-1, 1)}, whole_plot_replicates=4, seed=81
    )
    rng = np.random.default_rng(2026)
    right = wrong = 0
    n_rep = 300
    for _ in range(n_rep):
        rows = _simulate(d, rng, temp_effect=0.0)
        tab = split_plot_anova(rows, "y", **SP_KW)
        right += next(e for e in tab.rows if e.source == "T").p < 0.05
        crd = anova_report(
            rows, "y", factors=["T", "resin", "surface"], interactions=SP_KW["interactions"]
        )
        wrong += next(e for e in crd.rows if e.source == "T").p < 0.05
    assert abs(right / n_rep - 0.05) < 0.035
    assert wrong / n_rep > 0.2


def test_split_plot_variance_components_match_reml_when_balanced():
    sm = pytest.importorskip("statsmodels.formula.api")
    import pandas as pd

    d = split_plot_design(
        {"T": (-1, 1)}, {"resin": (-1, 1), "surface": (-1, 1)}, whole_plot_replicates=4, seed=81
    )
    rows = _simulate(d, np.random.default_rng(7))
    tab = split_plot_anova(rows, "y", **SP_KW)
    df = pd.DataFrame(rows)
    fit = sm.mixedlm("y ~ T * resin * surface", df, groups=df["whole_plot"]).fit(reml=True)
    assert tab.variance_components["whole_plot"] == pytest.approx(
        float(fit.cov_re.iloc[0, 0]), rel=1e-3
    )
    assert tab.variance_components["sub_plot"] == pytest.approx(float(fit.scale), rel=1e-3)


def test_split_plot_anova_validation():
    d = split_plot_design({"T": (-1, 1)}, {"resin": (-1, 1)}, whole_plot_replicates=2, seed=0)
    rows = [dict(r, y=1.0 + i) for i, r in enumerate(d.rows)]
    bad = [dict(r) for r in rows]
    bad[0]["T"] = -bad[0]["T"]
    with pytest.raises(ValueError, match="varies within whole plot"):
        split_plot_anova(bad, "y", whole_plot_factors=["T"], sub_plot_factors=["resin"])
    with pytest.raises(ValueError, match="missing"):
        split_plot_anova(rows, "yield", whole_plot_factors=["T"], sub_plot_factors=["resin"])
    with pytest.raises(ValueError, match="unequal sizes"):
        split_plot_anova(rows[:-1], "y", whole_plot_factors=["T"], sub_plot_factors=["resin"])


def test_anova_report_names_a_missing_column():
    rows = [{"A": 1, "y": 1.0}, {"A": 2, "y": 2.0}, {"A": 1, "y": 1.5}]
    with pytest.raises(ValueError, match=r"\['B'\] missing"):
        anova_report(rows, "y", factors=["A", "B"])
