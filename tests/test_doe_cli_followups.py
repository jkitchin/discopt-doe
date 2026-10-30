"""Follow-up batches the CLI used to refuse.

``extend`` on a classical design (central composite, Box-Behnken, definitive
screening, Latin hypercube) augments it through its own regression basis, and
``foldover`` appends the mirror image of a two-level screening design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

openpyxl = pytest.importorskip("openpyxl")

from discopt.doe.aliasing import alias_structure  # noqa: E402
from discopt.doe.cli import (  # noqa: E402
    DoEError,
    ExtendParams,
    FoldoverParams,
    NewParams,
    do_anova,
    do_extend,
    do_fit,
    do_foldover,
    do_new,
    do_status,
)
from discopt.doe.linear_design import design_row  # noqa: E402
from discopt.doe.workbook import Workbook  # noqa: E402


def _fill(path: Path, predictor, response: str = "y") -> None:
    wb = openpyxl.load_workbook(path)
    runs = wb["runs"]
    headers = [c.value for c in runs[1]]
    col = headers.index(response)
    for row in runs.iter_rows(min_row=2):
        if row[0].value is None or row[col].value not in (None, ""):
            continue
        row[col].value = float(predictor(dict(zip(headers, [c.value for c in row]))))
    wb.save(path)


def _params(tmp_path, template, **kw) -> NewParams:
    base = dict(
        output=tmp_path / f"{template}.xlsx",
        n=8,
        inputs=[],
        response_name="y",
        measurement_error=1.0,
        criterion="determinant",
        seed=0,
        n_starts=1,
        template=template,
    )
    base.update(kw)
    return NewParams(**base)


def _design_matrix(path: Path) -> np.ndarray:
    wb = Workbook.open(path)
    names = wb.parameter_names()
    inputs = [s.name for s in wb.input_specs()]
    return np.array(
        [
            design_row(wb.template_name(), wb.template_args(), names, inputs, r)
            for r in wb.all_runs()
        ],
        dtype=float,
    )


BOX = [("T", 300.0, 400.0), ("P", 1.0, 5.0), ("F", 0.1, 2.0)]


@pytest.mark.parametrize(
    ("template", "extra"),
    [
        ("central-composite", {}),
        ("box-behnken", {}),
        ("latin-hypercube", {"n": 10}),
        ("definitive-screening", {"basis": "linear"}),
    ],
)
def test_extend_augments_a_classical_design(tmp_path, template, extra):
    out = do_new(_params(tmp_path, template, inputs=BOX, **extra))
    path = Path(out["workbook_path"])
    _fill(path, lambda r: 1 + r["T"] / 100 - (r["P"] - 3) ** 2)
    n_before = len(Workbook.open(path).all_runs())
    X_before = _design_matrix(path)

    ext = do_extend(ExtendParams(workbook=path, n=3, n_starts=2))

    assert ext["batch"] == 2
    assert len(ext["designs"]) == 3
    for d in ext["designs"]:
        for name, lo, hi in BOX:
            assert lo - 1e-9 <= d[name] <= hi + 1e-9
    status = do_status({"workbook": str(path)})
    assert status["n_total"] == n_before + 3
    assert status["n_pending"] == 3

    # The stated precision gain is exactly what the appended rows add.
    X_after = _design_matrix(path)
    p = X_after.shape[1]
    gain = np.exp(
        (np.linalg.slogdet(X_after.T @ X_after)[1] - np.linalg.slogdet(X_before.T @ X_before)[1])
        / (2 * p)
    )
    assert ext["precision_gain"] == pytest.approx(gain, rel=1e-6)
    assert gain > 1.0

    # The campaign still fits once the new runs are measured.
    _fill(path, lambda r: 1 + r["T"] / 100 - (r["P"] - 3) ** 2)
    fit = do_fit({"workbook": str(path)})
    assert fit["parameters"]


def test_classical_extend_counts_pending_runs_as_information(tmp_path):
    """A linear basis's information does not depend on the responses, so runs not
    yet measured still count: two extends in a row must not pile up on the same
    points the first one chose."""
    out = do_new(_params(tmp_path, "central-composite", inputs=BOX[:2]))
    path = Path(out["workbook_path"])
    first = do_extend(ExtendParams(workbook=path, n=4, n_starts=2))
    second = do_extend(ExtendParams(workbook=path, n=4, n_starts=2))
    assert second["batch"] == first["batch"] + 1
    assert second["precision_gain"] > 1.0


def test_extend_on_a_screening_design_points_to_foldover(tmp_path):
    pairs = {n: (-1.0, 1.0) for n in "abcde"}
    out = do_new(
        _params(
            tmp_path,
            "fractional-factorial",
            n=0,
            factor_pairs=pairs,
            generators=["d=ab", "e=ac"],
            criterion="anova",
        )
    )
    with pytest.raises(DoEError, match="foldover"):
        do_extend(ExtendParams(workbook=Path(out["workbook_path"]), n=2, n_starts=1))


def test_full_foldover_de_aliases_a_resolution_iii_fraction(tmp_path):
    pairs = {n: (-1.0, 1.0) for n in "abcde"}
    out = do_new(
        _params(
            tmp_path,
            "fractional-factorial",
            n=0,
            factor_pairs=pairs,
            generators=["d=ab", "e=ac"],
            criterion="anova",
        )
    )
    path = Path(out["workbook_path"])
    original = [dict(r) for r in Workbook.open(path).all_runs()]
    assert alias_structure(original, list("abcde")).resolution == 3

    fold = do_foldover(FoldoverParams(workbook=path))

    assert fold["batch"] == 2
    assert fold["resolution"] == 4
    mirrored = {tuple(-d[n] for n in "abcde") for d in fold["designs"]}
    assert mirrored == {tuple(r[n] for n in "abcde") for r in original}

    # Truth: a, c and the a*c interaction. In the 8-run fraction the interaction
    # sits on e (e = ac); after the fold-over, e is clear of it.
    _fill(path, lambda r: 10 + 4 * r["a"] + 3 * r["c"] + 3 * r["a"] * r["c"])
    effects = {row["factor"]: row["effect"] for row in do_anova({"workbook": str(path)})["effects"]}
    assert abs(effects["e"]) < 1e-9
    assert effects["a"] == pytest.approx(8.0)


def test_single_factor_foldover_reverses_only_that_factor(tmp_path):
    pairs = {n: (-1.0, 1.0) for n in "abcde"}
    out = do_new(
        _params(
            tmp_path,
            "fractional-factorial",
            n=0,
            factor_pairs=pairs,
            generators=["d=ab", "e=ac"],
            criterion="anova",
        )
    )
    path = Path(out["workbook_path"])
    original = {tuple(r[n] for n in "abcde") for r in Workbook.open(path).all_runs()}
    fold = do_foldover(FoldoverParams(workbook=path, factor="a"))
    assert {tuple(d[n] for n in "abcde") for d in fold["designs"]} == {
        (-a, b, c, d, e) for a, b, c, d, e in original
    }


def test_foldover_skips_centre_runs_and_refuses_twice(tmp_path):
    pairs = {n: (-1.0, 1.0) for n in "abc"}
    out = do_new(
        _params(
            tmp_path,
            "factorial-2level",
            n=0,
            factor_pairs=pairs,
            center_points=3,
            criterion="anova",
        )
    )
    path = Path(out["workbook_path"])
    corners = [r for r in Workbook.open(path).all_runs() if all(r[n] in (-1, 1) for n in "abc")]
    fold = do_foldover(FoldoverParams(workbook=path))
    assert len(fold["designs"]) == len(corners)
    with pytest.raises(DoEError, match="already been followed up"):
        do_foldover(FoldoverParams(workbook=path))


def test_foldover_refuses_other_templates_and_unknown_factors(tmp_path):
    out = do_new(_params(tmp_path, "central-composite", inputs=BOX[:2]))
    with pytest.raises(DoEError, match="two-level screening"):
        do_foldover(FoldoverParams(workbook=Path(out["workbook_path"])))
    pairs = {n: (-1.0, 1.0) for n in "abc"}
    out = do_new(_params(tmp_path, "factorial-2level", n=0, factor_pairs=pairs, criterion="anova"))
    with pytest.raises(DoEError, match="unknown factor"):
        do_foldover(FoldoverParams(workbook=Path(out["workbook_path"]), factor="zz"))
