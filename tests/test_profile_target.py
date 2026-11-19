"""profile_target: a design target's linearized precision vs its profile likelihood."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from discopt.doe import SymbolicModel, Target, profile_target  # noqa: E402
from discopt.doe.runs import campaign_experiment, symbolic_experiment  # noqa: E402

MODEL = SymbolicModel(
    source="a*exp(-b*x)", parameter_names=("a", "b"), input_names=("x",), measurement_error=0.05
)
TRUTH = {"a": 2.0, "b": 0.7}


def _campaign(xs):
    rng = np.random.default_rng(0)
    runs = [{"x": x, "y": MODEL.predict(TRUTH, {"x": x}) + rng.normal(0, 0.05)} for x in xs]
    camp = campaign_experiment(MODEL, runs)
    return camp, camp.data_from_runs(runs)


@pytest.fixture(scope="module")
def early():
    """Three runs close together near the start: b is measured, far predictions are not."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _campaign([0.1, 0.2, 0.3])


def test_a_parameter_the_data_measure_is_adequate(early):
    camp, data = early
    # Two-point check (the default): two constrained fits settle it.
    chk = profile_target(camp, data, Target.parameters(["b"]))
    p = chk.profiles["b"]
    assert chk.adequate and p.adequate and p.method == "two-point" and p.profile is None
    assert all(abs(r - 1) < 0.1 for r in p.half_width_ratios)
    # The full profile agrees with the linear interval.
    full = profile_target(camp, data, Target.parameters(["b"]), mode="full").profiles["b"]
    assert full.method == "profile"
    lo, hi = full.linear_ci
    assert full.profile.ci_lower == pytest.approx(lo, rel=0.05)
    assert full.profile.ci_upper == pytest.approx(hi, rel=0.05)


def test_a_nonlinear_target_is_flagged(early):
    """Half-life = log(2)/b: the profile is lopsided where the SE is symmetric."""
    camp, data = early
    chk = profile_target(camp, data, Target.expression({"half_life": "log(2)/b"}))
    p = chk.profiles["half_life"]
    assert not chk.adequate
    assert p.method == "profile"  # flagged by the two-point check, then profiled
    below, above = p.half_width_ratios
    assert above > 1.5 > below  # much wider above the estimate than the SE says
    assert "optimistic" in p.note
    assert "CHECK" in chk.summary()
    # The two-point check alone already sees it.
    quick = profile_target(
        camp, data, Target.expression({"half_life": "log(2)/b"}), mode="quick"
    ).profiles["half_life"]
    assert not quick.adequate and quick.half_width_ratios[1] > 1.5


def test_a_prediction_far_from_the_data(early):
    camp, data = early
    per_run = symbolic_experiment(MODEL)
    target = Target.responses(per_run, at=[{"x": 8.0}], responses=["y"])
    assert target.values(per_run, TRUTH)["y"] == pytest.approx(MODEL.predict(TRUTH, {"x": 8.0}))
    chk = profile_target(camp, data, target, model=per_run)
    p = chk.profiles["y"]
    # Linearized, the prediction's interval runs below zero; the profile's does not.
    assert p.linear_ci[0] < 0 < p.profile.ci_lower
    assert not p.adequate


def test_profile_target_needs_a_target(early):
    camp, data = early
    resolved = Target.parameters(["b"]).resolve(camp, TRUTH)
    with pytest.raises(TypeError, match="needs a Target"):
        profile_target(camp, data, resolved)


def test_status_profile_from_the_cli(tmp_path):
    from discopt.doe.cli import NewParams, do_fit, do_new, do_status
    from discopt.doe.workbook import Workbook

    out = do_new(
        NewParams(
            output=tmp_path / "p.xlsx",
            n=3,
            inputs=[("x", 0.05, 0.4)],
            response_name="y",
            measurement_error=0.05,
            criterion="trace",
            seed=0,
            n_starts=3,
            template="symbolic",
            expression="a*exp(-b*x)",
            param_initial_guess={"a": 2.0, "b": 0.7},
            target={"expressions": {"half_life": "log(2)/b"}},
        )
    )
    path = out["workbook_path"]
    wb = Workbook.open(path)
    rng = np.random.default_rng(1)
    wb.record_responses(
        {
            int(r["run_id"]): MODEL.predict(TRUTH, {"x": float(r["x"])}) + rng.normal(0, 0.05)
            for r in wb.pending_runs()
        }
    )
    wb.save()
    do_fit({"workbook": path})
    st = do_status({"workbook": path, "profile": True})
    prof = st["target_profile"]
    assert prof is not None, st.get("target_precision_error")
    t = prof["targets"]["half_life"]
    assert t["note"] and t["method"] in ("two-point", "profile")
    assert prof["cached"] is False
    assert do_status({"workbook": path})["target_profile"] is None  # only on request
    # Same data, same target: the stored result, no refit.
    again = do_status({"workbook": path, "profile": True})["target_profile"]
    assert again["cached"] is True and again["targets"] == prof["targets"]
