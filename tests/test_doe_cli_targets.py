"""Design targets on the command line: --target, --target-expr, --target-response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from discopt.doe.cli import (  # noqa: E402
    DoEError,
    ExtendParams,
    NewParams,
    _parse_target_response,
    _target_spec_from_args,
    add_subparser,
    do_extend,
    do_fit,
    do_new,
    do_status,
)
from discopt.doe.workbook import Workbook  # noqa: E402

THETA = {"A": 2.0, "k": 0.5}
_SYMBOLIC_ARGS = [
    "--input",
    "t:0.05:10",
    "--expr",
    "A*exp(-k*t)",
    "--param",
    "A=2",
    "--param",
    "k=0.5",
]


def _parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    add_subparser(top.add_subparsers(dest="cmd"))
    return top


def _symbolic(tmp_path: Path, n: int, target: dict | None, name: str = "s.xlsx") -> dict:
    return do_new(
        NewParams(
            output=tmp_path / name,
            n=n,
            inputs=[("t", 0.05, 10.0)],
            response_name="y",
            measurement_error=0.05,
            criterion="trace",
            seed=0,
            n_starts=4,
            template="symbolic",
            expression="A*exp(-k*t)",
            param_initial_guess=dict(THETA),
            target=target,
        )
    )


def _fill(path: Path) -> None:
    wb = Workbook.open(path)
    rng = np.random.default_rng(0)
    wb.record_responses(
        {
            int(r["run_id"]): 2.0 * np.exp(-0.5 * float(r["t"])) + rng.normal(0, 0.05)
            for r in wb.pending_runs()
        }
    )
    wb.save()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_flags_build_a_spec():
    args = _parser().parse_args(
        [
            "doe",
            "new",
            "symbolic",
            "-o",
            "x.xlsx",
            *_SYMBOLIC_ARGS,
            "--target",
            "k",
            "--target",
            "A=2",
            "--target-expr",
            "r=A/k",
            "--target-response",
            "y@t=1",
            "--target-response",
            "y@t=2.5",
            "--tolerance",
            "k=0.1",
            "--relative-precision",
        ]
    )
    assert _target_spec_from_args(args) == {
        "relative": True,
        "parameters": {"k": 1.0, "A": 2.0},
        "expressions": {"r": "A/k"},
        "responses": {"y": [{"t": 1.0}, {"t": 2.5}]},
        "tolerances": {"k": 0.1},
    }


def test_no_target_flags_mean_no_target():
    base = ["doe", "new", "symbolic", "-o", "x.xlsx", *_SYMBOLIC_ARGS]
    assert _target_spec_from_args(_parser().parse_args(base)) is None
    args = _parser().parse_args([*base, "--tolerance", "k=1"])
    with pytest.raises(DoEError, match="qualify a target"):
        _target_spec_from_args(args)


def test_extend_accepts_target_flags():
    args = _parser().parse_args(["doe", "extend", "w.xlsx", "--n", "2", "--target", "k"])
    assert _target_spec_from_args(args) == {"relative": False, "parameters": {"k": 1.0}}
    args = _parser().parse_args(["doe", "extend", "w.xlsx", "--n", "2", "--clear-target"])
    assert args.clear_target


@pytest.mark.parametrize("bad", ["", "y@", "y@t", "y@t=abc", "@t=1"])
def test_bad_response_targets_are_refused(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_target_response(bad)


# ---------------------------------------------------------------------------
# new / fit / status / extend
# ---------------------------------------------------------------------------


def test_symbolic_campaign_with_a_tolerance(tmp_path):
    spec = {"parameters": {"k": 1.0}, "tolerances": {"k": 0.05}, "relative": True}
    out = _symbolic(tmp_path, 3, spec)
    path = Path(out["workbook_path"])
    assert out["target"] == spec
    assert out["target_precision"]["estimable"]
    assert Workbook.open(path).target_spec() == spec

    _fill(path)
    do_fit({"workbook": str(path)})
    status = do_status({"workbook": str(path)})
    assert status["target"] == spec
    prec = status["target_precision"]
    assert prec is not None and set(prec["std_errors"]) == {"k"}
    assert prec["ratios"]["k"] == pytest.approx(prec["std_errors"]["k"] / (0.05 * 0.485), rel=0.05)

    # extend reuses the stored target and predicts the precision after the runs.
    # A targeted symbolic campaign is designed through its Experiment, so every
    # batch strategy is available (an untargeted one is greedy only).
    ext = do_extend(ExtendParams(workbook=path, n=2, n_starts=4, batch_strategy="joint"))
    assert ext["target"] == spec
    assert ext["target_precision"]["std_errors"]["k"] < prec["std_errors"]["k"]
    json.dumps(ext)  # JSON-able


def test_fewer_runs_than_parameters_with_a_target(tmp_path):
    with pytest.raises(DoEError, match="name the parameters you need with --target"):
        _symbolic(tmp_path, 1, None)
    out = _symbolic(tmp_path, 1, {"parameters": {"k": 1.0}})
    # One sample cannot separate k from A: reported as such, not as a
    # ridge-sized standard error.
    assert out["target_precision"]["estimable"] is False


def test_extend_can_replace_and_clear_the_target(tmp_path):
    path = Path(_symbolic(tmp_path, 3, {"parameters": {"k": 1.0}})["workbook_path"])
    _fill(path)
    do_fit({"workbook": str(path)})
    ext = do_extend(ExtendParams(workbook=path, n=1, n_starts=3, target={"parameters": {"A": 1}}))
    assert ext["target"]["parameters"] == {"A": 1.0}
    assert Workbook.open(path).target_spec()["parameters"] == {"A": 1.0}
    ext = do_extend(ExtendParams(workbook=path, n=1, n_starts=3, clear_target=True))
    assert ext["target"] is None and ext["target_precision"] is None
    assert Workbook.open(path).target_spec() is None
    with pytest.raises(DoEError, match="not both"):
        do_extend(
            ExtendParams(
                workbook=path, n=1, n_starts=3, target={"parameters": {"A": 1}}, clear_target=True
            )
        )


def test_parametric_template_target_on_one_coefficient(tmp_path):
    """Two runs cannot fit a quadratic, but they pin down its slope: x = ±1."""
    out = do_new(
        NewParams(
            output=tmp_path / "q.xlsx",
            n=2,
            inputs=[("x", -1.0, 1.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="trace",
            seed=0,
            n_starts=4,
            template="polynomial-1d",
            degree=2,
            target={"parameters": {"b1": 1.0}},
        )
    )
    assert sorted(d["x"] for d in out["designs"]) == pytest.approx([-1.0, 1.0], abs=1e-3)
    prec = out["target_precision"]
    assert prec["estimable"]
    # b1 = (y(1) - y(-1)) / 2 from two runs at sigma 0.1: SE = 0.1 / sqrt(2).
    assert prec["std_errors"]["b1"] == pytest.approx(0.1 / np.sqrt(2), rel=1e-4)


def test_response_target_on_a_template(tmp_path):
    out = do_new(
        NewParams(
            output=tmp_path / "r.xlsx",
            n=3,
            inputs=[("x", -1.0, 1.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="trace",
            seed=0,
            n_starts=4,
            template="polynomial-1d",
            degree=2,
            target={"responses": {"y": [{"x": 0.5}]}},
        )
    )
    assert set(out["target_precision"]["std_errors"]) == {"y"}


_MODULE = """
import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel


class TargetDecay(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("decay")
        A = m.continuous("A", lb=0.01, ub=100.0)
        k = m.continuous("k", lb=1e-3, ub=10.0)
        t = m.continuous("t", lb=0.0, ub=20.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"A": A, "k": k},
            design_inputs={"t": t},
            responses={"y": A * dm.exp(-k * t)},
            measurement_error={"y": 0.05},
        )
"""


def test_module_campaign_with_an_expression_target(tmp_path, monkeypatch):
    (tmp_path / "tdecay.py").write_text(_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("tdecay", None)
    out = do_new(
        NewParams(
            output=tmp_path / "m.xlsx",
            n=2,
            inputs=[("t", 0.05, 10.0)],
            response_name="y",
            measurement_error=0.05,
            criterion="trace",
            seed=0,
            n_starts=4,
            module_callable="tdecay:TargetDecay",
            param_initial_guess=dict(THETA),
            target={"expressions": {"half_life": "log(2)/k"}},
        )
    )
    assert out["target"]["expressions"] == {"half_life": "log(2)/k"}
    assert out["target_precision"]["estimable"]
    sys.modules.pop("tdecay", None)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,match",
    [
        ({"parameters": {"nope": 1}}, "unknown parameter"),
        ({"responses": {"z": [{"t": 1.0}]}}, "unknown response"),
        ({"responses": {"y": []}}, "needs an operating point"),
        ({"responses": {"y": [{"T": 1.0}]}}, "exactly the design inputs"),
        ({"parameters": {"k": 1}, "tolerances": {"A": 0.1}}, "not targets"),
        ({"parameters": {"k": 1}, "expressions": {"r": "A/k"}}, "different units"),
        (
            {"parameters": {"k": 1, "A": 1}, "tolerances": {"k": 0.1}},
            "every target needs one",
        ),
    ],
)
def test_bad_targets_are_refused(tmp_path, spec, match):
    with pytest.raises(DoEError, match=match):
        _symbolic(tmp_path, 3, spec)


def test_fixed_designs_refuse_a_target(tmp_path):
    with pytest.raises(DoEError, match="fixed or model-free design"):
        do_new(
            NewParams(
                output=tmp_path / "c.xlsx",
                n=1,
                inputs=[("x", -1.0, 1.0), ("z", -1.0, 1.0)],
                response_name="y",
                measurement_error=0.1,
                criterion="determinant",
                seed=0,
                n_starts=2,
                template="central-composite",
                target={"parameters": {"b1": 1.0}},
            )
        )


# ---------------------------------------------------------------------------
# jax-free basis path: classical augmentation and --use-linear-design
# ---------------------------------------------------------------------------


def test_classical_design_extended_for_an_interaction(tmp_path):
    out = do_new(
        NewParams(
            output=tmp_path / "ccd.xlsx",
            n=1,
            inputs=[("a", -1.0, 1.0), ("b", -1.0, 1.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="determinant",
            seed=0,
            n_starts=2,
            template="central-composite",
        )
    )
    path = Path(out["workbook_path"])
    wb = Workbook.open(path)
    names = list(wb.parameter_names())
    assert "b12" in names, names  # the a*b interaction coefficient
    inter = ["b12"]
    ext = do_extend(
        ExtendParams(workbook=path, n=2, n_starts=3, target={"parameters": {inter[0]: 1.0}})
    )
    assert ext["target"]["parameters"] == {inter[0]: 1.0}
    prec = ext["target_precision"]
    assert prec["estimable"] and prec["std_errors"][inter[0]] > 0
    # Runs that measure an interaction sit at the corners of the square.
    for d in ext["designs"]:
        assert abs(abs(d["a"]) - 1) < 1e-3 and abs(abs(d["b"]) - 1) < 1e-3
    assert Workbook.open(path).target_spec()["parameters"] == {inter[0]: 1.0}


def test_linear_design_path_takes_a_target(tmp_path):
    out = do_new(
        NewParams(
            output=tmp_path / "lin.xlsx",
            n=2,
            inputs=[("x", -1.0, 1.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="trace",
            seed=0,
            n_starts=4,
            template="polynomial-1d",
            degree=2,
            use_linear_design=True,
            target={"parameters": {"b1": 1.0}},
        )
    )
    assert sorted(d["x"] for d in out["designs"]) == pytest.approx([-1.0, 1.0], abs=1e-3)
    assert out["target_precision"]["std_errors"]["b1"] == pytest.approx(0.1 / np.sqrt(2), rel=1e-3)


def test_symbolic_target_without_jax_matches(tmp_path):
    """The browser has no jax: the basis fallback gives the same design quality."""
    from discopt.doe.cli import _symbolic_target_design
    from discopt.doe.symbolic import SymbolicModel

    model = SymbolicModel(
        source="A*exp(-k*t)",
        parameter_names=("A", "k"),
        input_names=("t",),
        measurement_error=0.05,
    )
    spec = {"parameters": {"k": 1.0}, "relative": False}
    ridge = 1e-8 * np.eye(2)
    kw = dict(criterion="trace", prior_fim=ridge, ridge=ridge, n_starts=4, seed=0)
    bounds = {"t": (0.05, 10.0)}
    jx = _symbolic_target_design(model, THETA, bounds, 3, spec, use_jax=True, **kw)
    nj = _symbolic_target_design(model, THETA, bounds, 3, spec, use_jax=False, **kw)
    assert nj[2].std["k"] == pytest.approx(jx[2].std["k"], rel=0.02)
