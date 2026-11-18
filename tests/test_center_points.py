"""Replicated center points through the analysis paths.

Center points are the usual way to replicate a design (pure error) and to check
for curvature, so every analysis must treat them correctly: the screening
effects table must not fold curvature into its error term, a response-surface
ANOVA must split the residual into lack of fit and pure error, and the
spreadsheet workflow must catch a center row replicated by copy-and-paste.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from discopt.doe import anova_report, effects_estimates
from discopt.doe.cli import DoEError, NewParams, do_anova, do_fit, do_new, do_status
from discopt.doe.workbook import Workbook


def _two_level_rows(n_center: int, seed: int = 3) -> list[dict]:
    """2^2 factorial plus center runs, with a strong center bump (curvature)."""
    rng = np.random.default_rng(seed)
    rows = [{"A": a, "B": b} for a in (0, 10) for b in (0, 10)]
    rows += [{"A": 5, "B": 5, "is_center": True} for _ in range(n_center)]
    for r in rows:
        xa, xb = (r["A"] - 5) / 5, (r["B"] - 5) / 5
        bump = 3.0 if r.get("is_center") else 0.0
        r["y"] = 10 + xa + 0.8 * xb + bump + rng.normal(0, 0.3)
    return rows


@pytest.mark.parametrize("n_center", [1, 2, 4, 8])
def test_effects_agree_with_anova_when_there_is_curvature(n_center) -> None:
    # Before the fix the center runs sat at 0 on every effect column with no
    # term of their own, so the curvature landed in the residual: with 8 center
    # points the effects table gave p = 0.15 for a factor the ANOVA table
    # (printed right above it) gave p = 1e-6.
    rows = _two_level_rows(n_center)
    eff = {
        e["factor"]: e
        for e in effects_estimates(
            rows, "y", factors=["A", "B"], levels={"A": (0, 10), "B": (0, 10)}
        )
    }
    table = anova_report(
        [{k: v for k, v in r.items() if k != "is_center"} for r in rows],
        response="y",
        factors=["A", "B"],
    )
    p_anova = {r.source: r.p for r in table.rows}
    for f in ("A", "B"):
        assert eff[f]["method"] == "residual"
        assert eff[f]["p"] == pytest.approx(p_anova[f], rel=1e-8)
        # one df spent on curvature
        assert eff[f]["df"] == len(rows) - 4


def test_more_center_points_do_not_inflate_effect_standard_errors() -> None:
    se = [
        effects_estimates(
            _two_level_rows(n), "y", factors=["A", "B"], levels={"A": (0, 10), "B": (0, 10)}
        )[0]["se"]
        for n in (2, 8)
    ]
    # true SE of a 2^2 effect at sigma = 0.3 is 0.3; the old code gave ~1.6-1.9
    assert max(se) < 0.6


def test_unreplicated_center_with_saturated_model_falls_back_to_lenth() -> None:
    # 2^3 with every interaction plus one center run: the single center run is
    # spent on curvature, so there is no pure error and Lenth's PSE applies.
    rng = np.random.default_rng(0)
    rows = [{"A": a, "B": b, "C": c} for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    rows.append({"A": 0, "B": 0, "C": 0})
    for r in rows:
        r["y"] = 2 * r["A"] + r["B"] + rng.normal(0, 0.1)
    eff = effects_estimates(
        rows,
        "y",
        factors=["A", "B", "C"],
        levels={f: (-1, 1) for f in "ABC"},
        interactions=[("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C")],
    )
    assert {e["method"] for e in eff} == {"lenth"}


def test_three_level_designs_get_no_curvature_term() -> None:
    # A definitive screening design's middle level is part of the design, not a
    # curvature check: its center run must stay in the residual as before.
    from discopt.doe.screening import _center_indicator

    coded = {
        "A": np.array([0.0, 1.0, -1.0, 1.0, 0.0]),
        "B": np.array([0.0, -1.0, 0.0, 1.0, 1.0]),
    }
    rows = [{} for _ in range(5)]
    assert _center_indicator(rows, coded) is None
    coded2 = {"A": np.array([0.0, 1.0, -1.0]), "B": np.array([0.0, -1.0, 1.0])}
    np.testing.assert_array_equal(_center_indicator(rows[:3], coded2), [1.0, 0.0, 0.0])


# ── response-surface workbooks ────────────────────────────────────────────


def _rsm_workbook(
    tmp_path, template, n_inputs, center_points, *, curved=0.0, seed=0, basis="quadratic"
):
    inputs = [(f"x{i + 1}", 0.0, 10.0) for i in range(n_inputs)]
    out = do_new(
        NewParams(
            output=tmp_path / f"{template}.xlsx",
            n=0,
            inputs=inputs,
            response_name="y",
            measurement_error=1.0,
            criterion="determinant",
            seed=0,
            n_starts=1,
            template=template,
            center_points=center_points,
            basis=basis,
        )
    )
    path = out["workbook_path"]
    rng = np.random.default_rng(seed)
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    headers = [c.value for c in sheet[1]]
    for row in sheet.iter_rows(min_row=2):
        vals = dict(zip(headers, [c.value for c in row]))
        if vals.get("run_id") is None:
            continue
        x = np.array([float(vals[f"x{i + 1}"]) for i in range(n_inputs)])
        y = (
            1
            + x.sum() * 0.3
            + 0.05 * (x**2).sum()
            + curved * np.prod(np.sin(x))
            + rng.normal(0, 0.3)
        )
        row[headers.index("y")].value = float(y)
    book.save(path)
    return path


@pytest.mark.parametrize(
    "template,k", [("central-composite", 2), ("box-behnken", 3), ("definitive-screening", 3)]
)
def test_anova_on_a_response_surface_design_is_a_regression_anova(tmp_path, template, k):
    # The level-means ANOVA used to fail on every CCD ("term 'x2' is aliased")
    # and warn "unbalanced" on Box-Behnken/DSD because of the center replicates.
    basis = "linear" if template == "definitive-screening" else "quadratic"
    path = _rsm_workbook(tmp_path, template, k, center_points=4, basis=basis)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = do_anova({"workbook": path})
    assert out["analysis"] == "regression"
    sources = [r["source"].strip() for r in out["rows"]]
    assert sources[0] == "Model"
    assert {"Error", "Lack of fit", "Pure error", "Total (corrected)"} <= set(sources)
    # pure error df = (center replicates - 1)
    pe = next(r for r in out["rows"] if r["source"].strip() == "Pure error")
    assert pe["df"] == 3
    # each term's F is the square of its coefficient t statistic
    t = {c["name"]: c["t_statistic"] for c in out["coefficients"]}
    for r in out["rows"]:
        if "term" in r:
            assert r["f"] == pytest.approx(t[r["term"]] ** 2, rel=1e-8)
    lof = next(r for r in out["rows"] if r["source"].strip() == "Lack of fit")
    err = next(r for r in out["rows"] if r["source"] == "Error")
    assert lof["ss"] + pe["ss"] == pytest.approx(err["ss"], rel=1e-10)
    assert "Lack of fit" in out["summary"]


def test_lack_of_fit_is_detected(tmp_path) -> None:
    path = _rsm_workbook(tmp_path, "central-composite", 2, center_points=5, curved=4.0, seed=1)
    out = do_fit({"workbook": path})
    assert out["summary"]["lack_of_fit_p_value"] < 0.05
    sources = [r["source"].strip() for r in out["regression_anova"]]
    assert sources == ["Regression", "Residual", "Lack of fit", "Pure error", "Total (corrected)"]
    # and it reaches the workbook's anova sheet
    sheet = openpyxl.load_workbook(path)["anova"]
    assert any(str(c.value).strip() == "Lack of fit" for c in sheet["A"] if c.value)


def test_no_lack_of_fit_split_without_replicates(tmp_path) -> None:
    path = _rsm_workbook(tmp_path, "central-composite", 2, center_points=1)
    out = do_fit({"workbook": path})
    assert "lack_of_fit_p_value" not in out["summary"]
    assert [r["source"] for r in out["regression_anova"]] == [
        "Regression",
        "Residual",
        "Total (corrected)",
    ]
    assert "No lack-of-fit test" in do_anova({"workbook": path})["summary"]


def test_level_means_options_are_rejected_for_regression_designs(tmp_path) -> None:
    path = _rsm_workbook(tmp_path, "central-composite", 2, center_points=3)
    with pytest.raises(DoEError, match="--interaction"):
        do_anova({"workbook": path, "interactions": [("x1", "x2")]})


# ── spreadsheet workflow ──────────────────────────────────────────────────


def test_copied_center_row_with_the_same_run_id_is_rejected(tmp_path) -> None:
    path = _rsm_workbook(tmp_path, "central-composite", 2, center_points=1)
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    last = [c.value for c in sheet[sheet.max_row]]
    sheet.append(last)  # replicate the last run by copy-paste, id and all
    book.save(path)
    with pytest.raises(ValueError, match=r"duplicate run_id.*next free"):
        Workbook.open(path).all_runs()
    with pytest.raises((DoEError, ValueError), match="duplicate run_id"):
        do_status({"workbook": path})


# ── GP surrogate ──────────────────────────────────────────────────────────


def test_gp_ignores_replicates_that_agree_exactly() -> None:
    pytest.importorskip("sklearn")
    from discopt.doe.surrogate import gp_surrogate

    rng = np.random.default_rng(1)
    X = np.vstack([rng.uniform(0, 1, (10, 2)), [[0.5, 0.5]] * 4])
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2  # identical at the replicated center
    with pytest.warns(UserWarning, match="identical responses"):
        g = gp_surrogate().fit(X, y)
    assert g.noise_source_ == "fitted"
    assert g.noise_sd_ > 0

    y_noisy = y + rng.normal(0, 0.05, len(y))
    g2 = gp_surrogate().fit(X, y_noisy)
    assert g2.noise_source_ == "replicates"


def test_inestimable_basis_gives_a_clear_error(tmp_path) -> None:
    # A full quadratic in 3 factors (10 terms) from a 3-factor DSD's 9 distinct
    # settings: the table used to fill with NaN rows.
    path = _rsm_workbook(tmp_path, "definitive-screening", 3, center_points=4)
    with pytest.raises(DoEError, match="not estimable"):
        do_anova({"workbook": path})


def test_fit_of_an_inestimable_basis_reports_no_standard_errors(tmp_path, capsys) -> None:
    # Inverting the singular XᵀX used to report SE = 0.0 for every coefficient
    # (false certainty) with the residual df counted from n_p, not the rank.
    path = _rsm_workbook(tmp_path, "definitive-screening", 3, center_points=4)
    out = do_fit({"workbook": path})
    assert out["estimable"] is False
    assert all(np.isnan(p["std_error"]) for p in out["parameters"])
    assert any("not estimable" in w for w in out["warnings"])
    # 12 runs, 9 distinct settings, but the 10-term quadratic has rank 8 there
    assert out["degrees_of_freedom"] == 12 - 8
    rows = {r["source"].strip(): r for r in out["regression_anova"]}
    assert rows["Regression"]["df"] == 7
    assert rows["Residual"]["df"] == 4
    assert rows["Pure error"]["df"] == 3
    assert "extend" in out["next_command"]

    import argparse

    from discopt.doe.cli import add_subparser

    top = argparse.ArgumentParser()
    add_subparser(top.add_subparsers(dest="cmd"))
    args = top.parse_args(["doe", "fit", str(path)])
    assert args.doe_func(args) == 0
    captured = capsys.readouterr()
    assert "not estimable" in captured.err
    assert "± n/a" in captured.out and "± 0\n" not in captured.out


def test_estimable_fit_is_unchanged(tmp_path) -> None:
    path = _rsm_workbook(tmp_path, "definitive-screening", 3, center_points=4, basis="linear")
    out = do_fit({"workbook": path})
    assert out["estimable"] is True and out["warnings"] == []
    assert out["degrees_of_freedom"] == 12 - 4
    assert all(np.isfinite(p["std_error"]) and p["std_error"] > 0 for p in out["parameters"])


# ── center points with a categorical factor ───────────────────────────────


def test_center_points_with_a_categorical_factor_run_at_each_level() -> None:
    # This raised "center_points > 0 requires all factors to be numeric" --
    # including for the GUI's own default factor table (temp, pressure,
    # catalyst A/B) as soon as center points were added.
    from discopt.doe import factorial_2level_design

    des = factorial_2level_design(
        {"temp": (80, 120), "pressure": (1, 5), "catalyst": ("A", "B")},
        center_points=3,
        replicates=2,
        seed=0,
    )
    centers = [r for r in des.rows if r["is_center"]]
    assert len(des.rows) == 2 * (8 + 3 * 2)
    assert len(centers) == 2 * 3 * 2
    assert all(r["temp"] == 100.0 and r["pressure"] == 3.0 for r in centers)
    for rep in (0, 1):
        cats = sorted(r["catalyst"] for r in centers if r["replicate"] == rep)
        assert cats == ["A"] * 3 + ["B"] * 3


def test_two_categorical_factors_cross_their_center_runs() -> None:
    from discopt.doe import factorial_2level_design

    des = factorial_2level_design(
        {"T": (0, 1), "c1": ("A", "B"), "c2": ("X", "Y")}, center_points=2, seed=0
    )
    centers = [(r["c1"], r["c2"]) for r in des.rows if r["is_center"]]
    assert sorted(centers) == sorted([(a, b) for a in "AB" for b in "XY"] * 2)


def test_all_categorical_factors_still_reject_center_points() -> None:
    from discopt.doe import factorial_2level_design

    with pytest.raises(ValueError, match="at least one numeric factor"):
        factorial_2level_design({"a": ("X", "Y"), "b": ("P", "Q")}, center_points=2)


@pytest.mark.parametrize("flagged", [True, False])
def test_anova_with_a_categorical_factor_and_center_points(flagged) -> None:
    from discopt.doe import factorial_2level_design

    des = factorial_2level_design(
        {"temp": (80, 120), "pressure": (1, 5), "catalyst": ("A", "B")},
        center_points=3,
        seed=0,
    )
    rng = np.random.default_rng(0)
    rows = []
    for r in des.rows:
        t, p = (r["temp"] - 100) / 20, (r["pressure"] - 3) / 2
        c = 1 if r["catalyst"] == "B" else -1
        y = 50 + 5 * t + 2 * p + 3 * c + 1.5 * t * c + 4 * r["is_center"] + rng.normal(0, 0.5)
        row = {k: r[k] for k in ("temp", "pressure", "catalyst")} | {"y": y}
        if flagged:
            row["is_center"] = r["is_center"]
        rows.append(row)
    factors = ["temp", "pressure", "catalyst"]
    inter = [("temp", "catalyst")]
    table = anova_report(
        [{k: v for k, v in r.items() if k != "is_center"} for r in rows] if not flagged else rows,
        response="y",
        factors=factors,
        interactions=inter,
    )
    got = {r.source: r for r in table.rows}
    assert "curvature" in got and got["curvature"].df == 1

    # Independent check: OLS on coded columns plus a center indicator.
    is_c = np.array([r["temp"] == 100.0 for r in rows], dtype=float)
    t = np.array([0.0 if c else (r["temp"] - 100) / 20 for r, c in zip(rows, is_c)])
    p = np.array([0.0 if c else (r["pressure"] - 3) / 2 for r, c in zip(rows, is_c)])
    cat = np.array([1.0 if r["catalyst"] == "B" else -1.0 for r in rows])
    X = np.column_stack([np.ones(len(rows)), t, p, cat, t * cat, is_c])
    y = np.array([r["y"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_res = len(y) - X.shape[1]
    assert got["Residual"].df == df_res
    assert got["Residual"].ss == pytest.approx(float(resid @ resid), rel=1e-9)

    eff = {
        e["factor"]: e
        for e in effects_estimates(
            rows,
            "y",
            factors=factors,
            levels={"temp": (80, 120), "pressure": (1, 5), "catalyst": ("A", "B")},
            interactions=inter,
        )
    }
    for f in ("temp", "pressure", "catalyst", "temp:catalyst"):
        assert eff[f]["p"] == pytest.approx(got[f].p, rel=1e-8)
    assert eff["catalyst"]["effect"] == pytest.approx(2 * beta[3], rel=1e-9)


@pytest.mark.parametrize("template", ["factorial-2level", "plackett-burman", "fractional"])
def test_cli_two_level_workbooks_accept_a_categorical_factor_with_centers(tmp_path, template):
    factor_pairs = {"A": (0.0, 10.0), "B": (0.0, 10.0), "cat": ("X", "Y")}
    extra = {}
    if template == "fractional":
        factor_pairs = {"A": (0.0, 10.0), "B": (0.0, 10.0), "C": (0.0, 10.0), "cat": ("X", "Y")}
        extra = {"generators": ["cat=A*B*C"]}
        template = "fractional-factorial"
    out = do_new(
        NewParams(
            output=tmp_path / "w.xlsx",
            n=0,
            inputs=[],
            response_name="y",
            measurement_error=1.0,
            criterion="anova",
            seed=0,
            n_starts=1,
            template=template,
            factor_pairs=factor_pairs,
            center_points=2,
            **extra,
        )
    )
    path = out["workbook_path"]
    rng = np.random.default_rng(1)
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    headers = [c.value for c in sheet[1]]
    n_center = 0
    for row in sheet.iter_rows(min_row=2):
        v = dict(zip(headers, [c.value for c in row]))
        n_center += v["A"] == 5
        y = v["A"] + 2 * v["B"] + (3 if v["cat"] == "Y" else 0) + rng.normal(0, 0.3)
        row[headers.index("y")].value = float(y)
    book.save(path)
    assert n_center == 4  # 2 center points at each of cat = X, Y
    res = do_anova({"workbook": path})
    sources = {r["source"] for r in res["rows"]}
    assert {"cat", "curvature"} <= sources
    p_anova = {r["source"]: r["p"] for r in res["rows"]}
    for e in res["effects"]:
        assert e["p"] == pytest.approx(p_anova[e["factor"]], rel=1e-8)


# ── blocks, replicates, fold-over and aliasing around center runs ─────────


def _blocked_rows(seed: int = 0) -> list[dict]:
    import discopt.doe as d

    b = d.blocked_factorial_design(
        {"A": (80, 120), "B": (1, 5), "C": (0, 1)}, n_blocks=2, center_points_per_block=2, seed=0
    )
    rng = np.random.default_rng(seed)
    rows = []
    for r in b.rows:
        a, bb = (r["A"] - 100) / 20, (r["B"] - 3) / 2
        y = 10 + 2 * a + bb + 3 * (r["block"] == 1) + rng.normal(0, 0.2)
        rows.append(
            {**{k: r[k] for k in ("A", "B", "C", "block")}, "y": y, "is_center": r["is_center"]}
        )
    return rows


@pytest.mark.parametrize("flagged", [True, False])
def test_block_is_not_coded_as_a_centred_factor(flagged) -> None:
    # The block index is numeric, so center runs had it coded 0 -- as if they ran
    # "between" blocks -- leaving the block shift in the residual (MS 1.6 for a
    # noise variance of 0.04). Unflagged, the design failed as "aliased".
    rows = _blocked_rows()
    if not flagged:
        rows = [{k: v for k, v in r.items() if k != "is_center"} for r in rows]
    t = anova_report(rows, response="y", factors=["A", "B", "C", "block"])
    got = {r.source: r for r in t.rows}
    # independent OLS: coded A, B, C (0 at centers), block dummy, center indicator
    is_c = np.array([r["A"] == 100 for r in rows], dtype=float)
    cols = [np.ones(len(rows))]
    for f, (lo, hi) in {"A": (80, 120), "B": (1, 5), "C": (0, 1)}.items():
        cols.append(np.array([(2 * r[f] - lo - hi) / (hi - lo) for r in rows]))
    cols += [np.array([float(r["block"] == 1) for r in rows]), is_c]
    X = np.column_stack(cols)
    y = np.array([r["y"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    assert got["Residual"].ss == pytest.approx(rss, rel=1e-9)
    assert got["Residual"].df == len(rows) - X.shape[1]
    assert got["block"].p < 1e-4


def test_effects_blocks_match_anova_with_a_block_term() -> None:
    rows = _blocked_rows()
    t = anova_report(rows, response="y", factors=["A", "B", "C", "block"])
    p_anova = {r.source: r.p for r in t.rows}
    eff = effects_estimates(
        rows,
        "y",
        factors=["A", "B", "C"],
        levels={"A": (80, 120), "B": (1, 5), "C": (0, 1)},
        blocks=["block"],
    )
    for e in eff:
        assert e["p"] == pytest.approx(p_anova[e["factor"]], rel=1e-8)
    with pytest.raises(ValueError, match="both as a factor and as a block"):
        effects_estimates(rows, "y", factors=["A", "block"], blocks=["block"])


def _two_level_workbook(tmp_path, template, factor_pairs, **kw):
    out = do_new(
        NewParams(
            output=tmp_path / "w.xlsx",
            n=0,
            inputs=[],
            response_name="y",
            measurement_error=1.0,
            criterion="anova",
            seed=0,
            n_starts=1,
            template=template,
            factor_pairs=factor_pairs,
            **kw,
        )
    )
    return out["workbook_path"]


def _fill_pending(path, seed=1):
    rng = np.random.default_rng(seed)
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    headers = [c.value for c in sheet[1]]
    for row in sheet.iter_rows(min_row=2):
        v = dict(zip(headers, [c.value for c in row]))
        if v["y"] is not None:
            continue
        y = v["A"] + 2 * v["B"] + (3 if v.get("cat") == "Y" else 0) + rng.normal(0, 0.3)
        y += 0.5 * (v.get("replicate") or 0)
        row[headers.index("y")].value = float(y)
    book.save(path)


def test_include_replicate_effects_agree_with_anova(tmp_path) -> None:
    path = _two_level_workbook(
        tmp_path,
        "factorial-2level",
        {"A": (0.0, 10.0), "B": (0.0, 10.0), "cat": ("X", "Y")},
        center_points=2,
        replicates=2,
    )
    _fill_pending(path)
    out = do_anova({"workbook": path, "include_replicate": True})
    p_anova = {r["source"]: r["p"] for r in out["rows"]}
    assert "replicate" in p_anova
    for e in out["effects"]:
        assert e["p"] == pytest.approx(p_anova[e["factor"]], rel=1e-8)


def test_foldover_keeps_the_replicate_index(tmp_path) -> None:
    # The mirrored runs had a blank replicate, so `anova --include-replicate`
    # refused the workbook ("column(s) ['replicate'] missing from some rows").
    from discopt.doe.cli import FoldoverParams, do_foldover

    path = _two_level_workbook(
        tmp_path,
        "fractional-factorial",
        {"A": (0.0, 10.0), "B": (0.0, 10.0), "C": (0.0, 10.0)},
        generators=["C=AB"],
        replicates=2,
    )
    _fill_pending(path)
    do_foldover(FoldoverParams(workbook=Path(path)))
    runs = Workbook.open(path).all_runs()
    assert all(r["replicate"] is not None for r in runs)
    _fill_pending(path)
    out = do_anova({"workbook": path, "include_replicate": True})
    assert out["n_observations"] == 16


def test_alias_structure_with_categorical_center_runs() -> None:
    from discopt.doe.aliasing import alias_structure
    from discopt.doe.fractional import fractional_factorial_design

    factors = {"A": (0, 10), "B": (0, 10), "C": (0, 10), "cat": ("X", "Y")}
    plain = fractional_factorial_design(factors, generators=["cat=A*B*C"], seed=0)
    with_c = fractional_factorial_design(factors, generators=["cat=A*B*C"], center_points=3, seed=0)
    a0 = alias_structure([dict(r) for r in plain.rows], list(plain.factors))
    a1 = alias_structure([dict(r) for r in with_c.rows], list(with_c.factors))
    assert a1.resolution == a0.resolution == 4
    assert a1.words == a0.words and a1.regular
    # numeric factors get a curvature column; the categorical one has no center
    assert "A^2" in a1.labels and "cat^2" not in a1.labels


def test_one_numeric_factor_and_a_categorical_one_via_the_workbook(tmp_path) -> None:
    # Unflagged, A in {0, 5, 10} x cat in {X, Y} is indistinguishable from a
    # 3 x 2 factorial, so anova_report alone treats it as one; the workbook knows
    # it is a 2-level design and flags its center runs.
    path = _two_level_workbook(
        tmp_path, "factorial-2level", {"A": (0.0, 10.0), "cat": ("X", "Y")}, center_points=3
    )
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    headers = [c.value for c in sheet[1]]
    rng = np.random.default_rng(0)
    for row in sheet.iter_rows(min_row=2):
        v = dict(zip(headers, [c.value for c in row]))
        y = v["A"] + (3 if v["cat"] == "Y" else 0) + 2 * (v["A"] == 5) + rng.normal(0, 0.3)
        row[headers.index("y")].value = float(y)
    book.save(path)
    out = do_anova({"workbook": path})
    rows = {r["source"]: r for r in out["rows"]}
    assert rows["A"]["df"] == 1 and rows["curvature"]["df"] == 1
    assert rows["curvature"]["p"] < 1e-3
