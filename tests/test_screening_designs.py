"""Tests for Plackett-Burman, definitive screening, fold-over, general
factorials, explicit-generator fractions, Lenth's method and the upgraded
effects_estimates."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from discopt.doe.aliasing import alias_structure
from discopt.doe.fractional import fractional_factorial_design
from discopt.doe.screening import (
    effects_estimates,
    factorial_2level_design,
    half_normal_scores,
    lenth_pse,
)
from discopt.doe.screening_designs import (
    conference_matrix,
    definitive_screening_design,
    fold_over,
    full_factorial_design,
    plackett_burman_design,
)


def _coded(design):
    return np.array(
        [
            [
                1 if r[f] == hi else (-1 if r[f] == lo else 0)
                for f, lo, hi in zip(design.factors, design.low, design.high)
            ]
            for r in design.rows
        ]
    )


# --------------------------------------------------------------------------
# Plackett-Burman
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [4, 8, 12, 16, 20, 24, 32])
def test_plackett_burman_columns_orthogonal_and_balanced(n_runs):
    k = n_runs - 1
    d = plackett_burman_design({f"x{i}": (-1, 1) for i in range(k)}, n_runs=n_runs, seed=1)
    X = _coded(d)
    assert X.shape == (n_runs, k)
    assert np.array_equal(X.T @ X, n_runs * np.eye(k, dtype=int))
    assert np.all(X.sum(axis=0) == 0)


def test_plackett_burman_default_size_and_levels():
    d = plackett_burman_design({"A": (80.0, 120.0), "cat": ("x", "y"), "C": (1, 2)}, seed=0)
    assert len(d) == 4  # smallest supported run count > 3 factors
    d7 = plackett_burman_design({f"f{i}": (0, 1) for i in range(7)}, seed=0)
    assert len(d7) == 8
    d9 = plackett_burman_design({f"f{i}": (0, 1) for i in range(9)}, seed=0)
    assert len(d9) == 12
    assert sorted(r["run_order"] for r in d9.rows) == list(range(12))
    assert {r["cat"] for r in d.rows} == {"x", "y"}


def test_plackett_burman_center_points_and_errors():
    d = plackett_burman_design({f: (0.0, 1.0) for f in "ABCDE"}, center_points=2, seed=0)
    assert len(d) == 10
    assert sum(r["is_center"] for r in d.rows) == 2
    with pytest.raises(ValueError, match="must exceed"):
        plackett_burman_design({f: (0, 1) for f in "ABCD"}, n_runs=4)
    with pytest.raises(ValueError, match="one of"):
        plackett_burman_design({f: (0, 1) for f in "AB"}, n_runs=28)
    with pytest.raises(ValueError, match="numeric"):
        plackett_burman_design({"A": ("a", "b"), "B": (0, 1)}, center_points=1)


# --------------------------------------------------------------------------
# Conference matrices and definitive screening designs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6, 8, 10, 12, 14, 18, 20, 24])
def test_conference_matrix_property(order):
    C = conference_matrix(order)
    assert np.all(np.diag(C) == 0)
    off = C[~np.eye(order, dtype=bool)]
    assert set(np.unique(off)) <= {-1, 1}
    assert np.array_equal(C.T @ C, (order - 1) * np.eye(order, dtype=int))


def test_conference_matrix_rejects_unsupported_orders():
    with pytest.raises(ValueError):
        conference_matrix(16)  # 15 is not a prime power
    with pytest.raises(ValueError):
        conference_matrix(7)


@pytest.mark.parametrize(
    "k, runs", [(4, 9), (5, 13), (6, 13), (7, 17), (8, 17), (9, 21), (12, 25), (15, 37)]
)
def test_dsd_run_counts(k, runs):
    """2m + 1 runs, m the smallest supported even order >= k (J&N 2011).
    k = 15, 16 would use order 16, which has no Paley matrix, so order 18."""
    d = definitive_screening_design({f"x{i}": (0.0, 1.0) for i in range(k)}, seed=0)
    assert len(d) == runs


def test_dsd_structure():
    d = definitive_screening_design({f: (10.0, 30.0) for f in "ABCDEFG"}, seed=3)
    X = _coded(d)
    assert set(np.unique(X)) == {-1, 0, 1}
    assert sum(r["is_center"] for r in d.rows) == 1
    # Fold-over pairs: every run's mirror image is in the design.
    rows = {tuple(x) for x in X}
    assert all(tuple(-x) in rows for x in X)
    # Each factor sits at its middle level in exactly 3 runs (two fold-over
    # rows with a zero in that column, plus the centre).
    assert np.all((X == 0).sum(axis=0) == 3)
    assert {r["A"] for r in d.rows} == {10.0, 20.0, 30.0}


def test_dsd_fake_factors_and_errors():
    d = definitive_screening_design({f: (0.0, 1.0) for f in "ABCD"}, fake_factors=2, seed=0)
    assert len(d) == 13
    with pytest.raises(ValueError, match="numeric"):
        definitive_screening_design({"A": ("a", "b"), "B": (0.0, 1.0)})


# --------------------------------------------------------------------------
# Fold-over
# --------------------------------------------------------------------------


def test_full_fold_over_gives_resolution_iv():
    base = fractional_factorial_design(
        {f: (-1, 1) for f in "ABCDEFG"}, generators=["D=AB", "E=AC", "F=BC", "G=ABC"], seed=0
    )
    folded = fold_over(base, seed=1)
    assert len(folded) == 16
    assert {r["block"] for r in folded.rows} == {0, 1}
    assert [r["run_order"] for r in folded.rows] == list(range(16))
    assert alias_structure(folded).resolution == 4


def test_single_factor_fold_over_clears_that_factor():
    base = fractional_factorial_design(
        {f: (-1, 1) for f in "ABCDEFG"}, generators=["D=AB", "E=AC", "F=BC", "G=ABC"], seed=0
    )
    a = alias_structure(fold_over(base, factor="A"))
    assert a.aliases("A") == []
    for other in "BCDEFG":
        assert all("A" not in lab.split(":") for lab, _ in a.aliases(other))


def test_fold_over_skips_centers_and_checks_factor():
    d = factorial_2level_design({f: (0.0, 1.0) for f in "AB"}, center_points=2, seed=0)
    assert len(fold_over(d)) == 6 + 4
    with pytest.raises(ValueError, match="unknown factor"):
        fold_over(d, factor="Z")


# --------------------------------------------------------------------------
# General factorial
# --------------------------------------------------------------------------


def test_full_factorial_mixed_levels():
    d = full_factorial_design({"cat": ["A", "B", "C"], "T": [300, 350]}, replicates=2, seed=4)
    assert len(d) == 12
    combos = {(r["cat"], r["T"], r["replicate"]) for r in d.rows}
    assert len(combos) == 12
    assert sorted(r["run_order"] for r in d.rows) == list(range(12))
    assert d.levels == (("A", "B", "C"), (300, 350))
    with pytest.raises(ValueError, match="at least 2"):
        full_factorial_design({"A": [1]})
    with pytest.raises(ValueError, match="distinct"):
        full_factorial_design({"A": [1, 1]})


# --------------------------------------------------------------------------
# Explicit generators in fractional_factorial_design
# --------------------------------------------------------------------------


def test_generator_path_multichar_names_and_checks():
    d = fractional_factorial_design(
        {"temp": (80, 120), "time": (10, 30), "cat": ("A", "B")},
        generators=["cat=temp*time"],
        seed=0,
    )
    assert len(d) == 4
    for r in d.rows:
        s = (1 if r["temp"] == 120 else -1) * (1 if r["time"] == 30 else -1)
        assert r["cat"] == ("B" if s == 1 else "A")
    with pytest.raises(ValueError, match="resolution"):
        fractional_factorial_design(
            {f: (-1, 1) for f in "ABCD"}, generators=["C=AB", "D=AB"], resolution=3
        )
    with pytest.raises(ValueError, match="disagrees"):
        fractional_factorial_design({f: (-1, 1) for f in "ABCD"}, generators=["D=ABC"], n_runs=4)
    with pytest.raises(ValueError, match="base factors"):
        fractional_factorial_design({f: (-1, 1) for f in "ABCDE"}, generators=["D=AB", "E=AD"])
    with pytest.raises(ValueError, match="resolution 3"):
        fractional_factorial_design(
            {f: (-1, 1) for f in "ABCDE"}, generators=["D=AB", "E=AC"], resolution=4
        )


def test_generator_path_scales_past_milp_limit():
    """16 factors in 32 runs, no solver: the MILP path is capped at 12."""
    names = "ABCDEFGHJKLMNOPQ"
    gens = [
        f"{g}={w}"
        for g, w in zip(
            "FGHJKLMNOPQ",
            ["ABC", "ABD", "ABE", "ACD", "ACE", "ADE", "BCD", "BCE", "BDE", "CDE", "ABCDE"],
        )
    ]
    d = fractional_factorial_design({f: (-1, 1) for f in names}, generators=gens, seed=0)
    assert len(d) == 32
    X = _coded(d)
    assert np.array_equal(X.T @ X, 32 * np.eye(16, dtype=int))


# --------------------------------------------------------------------------
# Lenth, half-normal scores, effects_estimates
# --------------------------------------------------------------------------


def test_lenth_pse_hand_example():
    effects = [10.0, -8.0, 1.0, -0.5, 0.8, -1.2, 0.3]
    # median|c| = 1.0 -> s0 = 1.5; trim at 3.75 leaves {0.3,0.5,0.8,1.0,1.2}
    # -> PSE = 1.5 * 0.8 = 1.2 (Lenth 1989).
    res = lenth_pse(effects)
    assert res.pse == pytest.approx(1.2)
    d = 7 / 3
    assert res.margin_of_error == pytest.approx(stats.t.ppf(0.975, d) * 1.2)
    gamma = (1 + 0.95 ** (1 / 7)) / 2
    assert res.simultaneous_margin == pytest.approx(stats.t.ppf(gamma, d) * 1.2)
    pse, me, sme = res  # tuple unpacking
    assert sme > me > pse
    with pytest.raises(ValueError, match="at least 3"):
        lenth_pse([1.0, 2.0])


def test_lenth_null_false_alarm_rate_is_near_nominal():
    rng = np.random.default_rng(0)
    alarms = []
    for _ in range(2000):
        c = rng.normal(0.0, 1.0, 15)
        alarms.append(np.mean(np.abs(c) > lenth_pse(c).margin_of_error))
    # Lenth's ME is slightly conservative per effect (about 0.03-0.05).
    assert 0.02 < np.mean(alarms) < 0.06


def test_half_normal_scores():
    s = half_normal_scores({"A": -4.0, "B": 0.5, "C": 2.0})
    assert s.labels == ("B", "C", "A")
    assert list(s.abs_effects) == [0.5, 2.0, 4.0]
    assert np.all(np.diff(s.quantiles) > 0)
    assert s.quantiles[0] == pytest.approx(stats.norm.ppf(0.5 + 0.5 * 0.5 / 3))
    from_list = half_normal_scores(
        [{"factor": "A", "effect": 1.0}, {"factor": "B", "effect": -3.0}]
    )
    assert from_list.labels == ("A", "B")


def _lab_rows(design, rng, sigma=0.1):
    rows = []
    for r in design.rows:
        a = 1.0 if r["A"] == 1 else -1.0
        b = 1.0 if r["B"] == 1 else -1.0
        rows.append(dict(r, y=5.0 + 2.0 * a + 1.5 * a * b + rng.normal(0.0, sigma)))
    return rows


def test_effects_estimates_interactions_and_p_values():
    rng = np.random.default_rng(1)
    d = factorial_2level_design({f: (-1, 1) for f in "ABC"}, replicates=2, seed=0)
    rows = _lab_rows(d, rng)
    est = {
        e["factor"]: e
        for e in effects_estimates(
            rows, "y", factors=list("ABC"), interactions=[("A", "B"), ("A", "C")]
        )
    }
    assert est["A"]["effect"] == pytest.approx(4.0, abs=0.2)
    assert est["A:B"]["effect"] == pytest.approx(3.0, abs=0.2)
    assert est["A:C"]["effect"] == pytest.approx(0.0, abs=0.2)
    assert est["A:B"]["p"] < 1e-6 and est["A:C"]["p"] > 0.01
    assert est["A:B"]["low"] == -1 and est["A:B"]["high"] == 1
    assert all(e["method"] == "residual" for e in est.values())


def test_omitting_a_real_interaction_inflates_the_se():
    rng = np.random.default_rng(2)
    d = factorial_2level_design({f: (-1, 1) for f in "ABC"}, replicates=2, seed=0)
    rows = _lab_rows(d, rng)
    mains = {e["factor"]: e for e in effects_estimates(rows, "y", factors=list("ABC"))}
    full = {
        e["factor"]: e
        for e in effects_estimates(rows, "y", factors=list("ABC"), interactions=[("A", "B")])
    }
    assert mains["A"]["se"] > 5 * full["A"]["se"]
    # df is the residual df behind the SE: n - rank(model matrix).
    assert mains["A"]["df"] == len(rows) - 4 and full["A"]["df"] == len(rows) - 5


def test_effects_estimates_falls_back_to_lenth_without_residual_df():
    rng = np.random.default_rng(3)
    d = factorial_2level_design({f: (-1, 1) for f in "ABC"}, seed=0)
    rows = _lab_rows(d, rng)
    inter = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C")]
    est = effects_estimates(rows, "y", factors=list("ABC"), interactions=inter)
    assert len(est) == 7
    assert {e["method"] for e in est} == {"lenth"}
    assert all(e["df"] == pytest.approx(7 / 3) for e in est)  # Lenth's m/3
    pse = lenth_pse([e["effect"] for e in est]).pse
    assert all(e["se"] == pytest.approx(pse) for e in est)
    top = est[0]
    assert top["factor"] == "A" and np.isfinite(top["t"]) and top["p"] < 0.05


def test_effects_estimates_rejects_bad_interactions():
    rows = [{"A": a, "B": b, "y": 1.0} for a in (-1, 1) for b in (-1, 1)]
    with pytest.raises(ValueError, match="unknown"):
        effects_estimates(rows, "y", interactions=[("A", "Z")])
    with pytest.raises(ValueError, match="distinct"):
        effects_estimates(rows, "y", interactions=[("A", "A")])
