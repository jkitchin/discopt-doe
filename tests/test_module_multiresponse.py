"""Multi-response ``--module`` campaigns driven end to end through the CLI layer.

Issue #46: a user-defined Experiment with several responses per run, some of
them left blank, has to go new -> fit -> extend -> fit from the workbook and
give the same answer as ``sequential_doe(..., initial_runs=...)`` in Python.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("openpyxl")
pytest.importorskip("jax")

from discopt.doe.cli import (  # noqa: E402
    DoEError,
    ExtendParams,
    NewParams,
    do_extend,
    do_fit,
    do_new,
    do_status,
)
from discopt.doe.workbook import Workbook  # noqa: E402

_PARTITION_SOURCE = '''
import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel


class Partition(Experiment):
    """Two species split between two phases; log distribution ratio quadratic in pH."""

    def create_model(self, **kwargs):
        m = dm.Model("partition")
        p = {n: m.continuous(n, lb=-10.0, ub=10.0)
             for n in ("aA", "bA", "cA", "aB", "bB", "cB")}
        x = m.continuous("pH", lb=2.0, ub=8.0)
        u = x - 5.0
        out = {}
        for s in ("A", "B"):
            D = dm.exp(p["a" + s] + p["b" + s] * u + p["c" + s] * u * u)
            out[s + "_org"] = D / (1 + D)
            out[s + "_aq"] = 1 / (1 + D)
        return ExperimentModel(
            model=m,
            unknown_parameters=p,
            design_inputs={"pH": x},
            responses=out,
            measurement_error={k: 0.02 for k in out},
        )
'''

GUESS = {"aA": 0.5, "bA": 0.8, "cA": -0.1, "aB": -0.3, "bB": -0.6, "cB": 0.05}
TRUTH = {"aA": 0.7, "bA": 0.6, "cA": -0.05, "aB": -0.1, "bB": -0.8, "cB": 0.08}
RESPONSES = ["A_org", "A_aq", "B_org", "B_aq"]
# sequential_doe searches with batch_optimal_experiment's default seed (42);
# `extend` seeds with wb.seed() + batch index, and the first extend is batch 2.
SEED = 40


@pytest.fixture
def partition_module(tmp_path, monkeypatch):
    pkg = tmp_path / "mod_pkg"
    pkg.mkdir()
    (pkg / "partition46.py").write_text(_PARTITION_SOURCE)
    monkeypatch.syspath_prepend(str(pkg))
    sys.modules.pop("partition46", None)
    yield "partition46:Partition"
    sys.modules.pop("partition46", None)


def _truth(pH: float) -> dict[str, float]:
    u = pH - 5.0
    out = {}
    for s in ("A", "B"):
        D = np.exp(TRUTH["a" + s] + TRUTH["b" + s] * u + TRUTH["c" + s] * u * u)
        out[s + "_org"] = float(D / (1 + D))
        out[s + "_aq"] = float(1 / (1 + D))
    return out


def _measure(wb: Workbook, run_ids, rng, blank=()) -> None:
    """Record noisy truth for ``run_ids``; ``blank`` lists (run_id, response) left empty."""
    rows = {int(r["run_id"]): r for r in wb.all_runs()}
    values = {}
    for rid in run_ids:
        y = _truth(float(rows[rid]["pH"]))
        values[rid] = {
            k: (None if (rid, k) in blank else v + 0.01 * rng.standard_normal())
            for k, v in y.items()
        }
    wb.record_responses(values)
    wb.save()


def _new(path, module, **kw) -> dict:
    params = dict(
        output=path,
        n=8,
        inputs=[("pH", 2.0, 8.0)],
        response_name="y",
        measurement_error=1.0,
        criterion="determinant",
        seed=SEED,
        n_starts=10,
        module_callable=module,
        param_initial_guess=dict(GUESS),
    )
    params.update(kw)
    return do_new(NewParams(**params))


def _python_runs(wb: Workbook) -> list[dict]:
    """The completed rows as sequential_doe ``initial_runs`` (blank -> None)."""
    out = []
    for r in wb.completed_runs():
        run = {"run_id": r["run_id"], "pH": float(r["pH"])}
        for k in RESPONSES:
            run[k] = None if r.get(k) in (None, "") else float(r[k])
        out.append(run)
    return out


def test_new_module_makes_one_column_per_response(tmp_path, partition_module):
    path = tmp_path / "p.xlsx"
    out = _new(path, partition_module, sigma_overrides={"B_aq": 0.03})
    assert out["response_names"] == RESPONSES
    wb = Workbook.open(path)
    assert wb.response_names() == RESPONSES
    assert wb.response_sigmas() == {"A_org": 0.02, "A_aq": 0.02, "B_org": 0.02, "B_aq": 0.03}
    headers = wb._runs_headers()
    assert headers[: 3 + len(RESPONSES)] == ["run_id", "batch", "pH", *RESPONSES]
    # Legacy keys mirror the first response for older readers.
    assert wb.response_name() == "A_org"
    assert wb.measurement_error() == 0.02
    status = do_status({"workbook": str(path)})
    assert status["response_names"] == RESPONSES
    assert status["n_pending"] == 8


def test_sigma_override_must_name_a_response(tmp_path, partition_module):
    with pytest.raises(DoEError, match="not responses of the model"):
        _new(tmp_path / "p.xlsx", partition_module, sigma_overrides={"C_org": 0.1})


def test_partial_rows_count_as_completed(tmp_path, partition_module):
    path = tmp_path / "p.xlsx"
    _new(path, partition_module)
    wb = Workbook.open(path)
    wb.record_responses({1: {"A_org": 0.4}, 2: {"B_aq": 0.7, "A_aq": None}})
    wb.save()
    wb = Workbook.open(path)
    assert [int(r["run_id"]) for r in wb.completed_runs()] == [1, 2]
    assert len(wb.pending_runs()) == 6
    # Clearing the only value returns the run to pending.
    wb.record_responses({1: {"A_org": None}})
    assert [int(r["run_id"]) for r in wb.completed_runs()] == [2]
    with pytest.raises(ValueError, match="not response columns"):
        wb.record_responses({3: {"nope": 1.0}})


@pytest.mark.slow
def test_cli_loop_matches_sequential_doe(tmp_path, partition_module):
    """new -> fit -> extend -> append -> fit equals sequential_doe on the same rows."""
    from discopt.doe import sequential_doe
    from partition46 import Partition

    path = tmp_path / "p.xlsx"
    _new(path, partition_module)
    rng = np.random.default_rng(0)
    wb = Workbook.open(path)
    _measure(wb, range(1, 9), rng, blank={(3, "B_aq"), (5, "A_org")})

    fit1 = do_fit({"workbook": str(path)})
    assert fit1["n_missing"] == 2
    assert fit1["n_observations"] == 8 * 4 - 2

    n_new = 4
    ext = do_extend(ExtendParams(workbook=path, n=n_new, n_starts=10))
    assert ext["batch"] == 2

    runs = _python_runs(Workbook.open(path))
    history = sequential_doe(
        Partition(),
        None,
        dict(GUESS),
        {"pH": (2.0, 8.0)},
        initial_runs=runs,
        n_rounds=1,
        experiments_per_round=n_new,
    )
    est = history[0].estimation
    for p in fit1["parameters"]:
        n = p["name"]
        assert p["estimate"] == pytest.approx(est.parameters[n], rel=1e-6, abs=1e-9)
        assert p["std_error"] == pytest.approx(est.standard_errors[n], rel=1e-6)
        lo, hi = est.confidence_intervals[n]
        assert p["ci_lower_95"] == pytest.approx(lo, rel=1e-6, abs=1e-9)
        assert p["ci_upper_95"] == pytest.approx(hi, rel=1e-6, abs=1e-9)
    np.testing.assert_allclose(fit1["covariance"], est.covariance, rtol=1e-6)

    # The workbook stores the FIM through openpyxl, which writes "%.16g" (one
    # digit short of a lossless float64), so the prior `extend` reads differs from
    # est.fim by ~1 ulp. The design is the same optimum -- the criterion agrees
    # to ~1e-11 -- but the local refinement stops within its own tolerance of it.
    assert ext["criterion_value"] == pytest.approx(history[0].design.criterion_value, rel=1e-9)
    cli_batch = sorted(d["pH"] for d in ext["designs"])
    py_batch = sorted(float(d["pH"]) for d in history[0].design.designs)
    np.testing.assert_allclose(cli_batch, py_batch, atol=1e-4)

    # Run the new batch, refit, and check round 2 against Python too.
    wb = Workbook.open(path)
    _measure(wb, ext["new_run_ids"], rng)
    fit2 = do_fit({"workbook": str(path)})
    runs2 = _python_runs(Workbook.open(path))
    est2 = sequential_doe(
        Partition(),
        None,
        dict(GUESS),
        {"pH": (2.0, 8.0)},
        initial_runs=runs2,
        n_rounds=1,
    )[0].estimation
    for p in fit2["parameters"]:
        assert p["estimate"] == pytest.approx(est2.parameters[p["name"]], rel=1e-6, abs=1e-9)
        assert p["std_error"] == pytest.approx(est2.standard_errors[p["name"]], rel=1e-6)
    # More data, tighter parameters.
    se1 = {p["name"]: p["std_error"] for p in fit1["parameters"]}
    assert all(p["std_error"] < se1[p["name"]] for p in fit2["parameters"])


def test_unfitted_prior_counts_only_measured_responses(tmp_path, partition_module, monkeypatch):
    """Before any fit, extend's prior FIM must drop blank cells, not count them."""
    import discopt.doe.cli as cli
    from discopt.doe.runs import campaign_experiment
    from partition46 import Partition

    path = tmp_path / "p.xlsx"
    _new(path, partition_module, n=6)
    wb = Workbook.open(path)
    _measure(wb, range(1, 7), np.random.default_rng(1), blank={(2, "A_aq"), (2, "B_org")})

    seen = {}
    real = cli._measured_fim

    def spy(experiment, runs, theta):
        seen["fim"] = real(experiment, runs, theta)
        return seen["fim"]

    monkeypatch.setattr(cli, "_measured_fim", spy)
    do_extend(ExtendParams(workbook=path, n=1, n_starts=2))

    runs = _python_runs(Workbook.open(path))
    camp = campaign_experiment(Partition(), runs)
    from discopt.doe import compute_fim

    full = np.asarray(compute_fim(camp, GUESS).fim)
    # The full campaign FIM counts the two blank cells; the measured one does not.
    assert np.trace(seen["fim"]) < np.trace(full)
    blank_only = [
        {
            "run_id": 2,
            "pH": runs[1]["pH"],
            **{k: (1.0 if k in ("A_aq", "B_org") else None) for k in RESPONSES},
        }
    ]
    missing = real(Partition(), blank_only, GUESS)
    np.testing.assert_allclose(seen["fim"] + missing, full, rtol=1e-8, atol=1e-10)


def test_batch_strategy_is_passed_through(tmp_path, partition_module):
    path = tmp_path / "p.xlsx"
    _new(path, partition_module, n=6)
    out = do_extend(ExtendParams(workbook=path, n=2, n_starts=2, batch_strategy="joint"))
    assert out["batch_strategy"] == "joint"
    with pytest.raises(DoEError, match="unknown batch strategy"):
        do_extend(ExtendParams(workbook=path, n=2, n_starts=2, batch_strategy="random"))


def test_batch_strategy_rejected_for_basis_templates(tmp_path):
    path = tmp_path / "s.xlsx"
    do_new(
        NewParams(
            output=path,
            n=3,
            inputs=[("x", 0.0, 5.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="determinant",
            seed=0,
            n_starts=2,
            template="symbolic",
            expression="a * exp(-b * x)",
            param_initial_guess={"a": 1.0, "b": 0.5},
        )
    )
    with pytest.raises(DoEError, match="greedily"):
        do_extend(ExtendParams(workbook=path, n=2, n_starts=2, batch_strategy="joint"))


def test_old_single_response_workbook_still_reads(tmp_path):
    """A workbook without the ``responses`` key reads back as one response."""
    from discopt.doe.workbook import InputSpec

    path = tmp_path / "old.xlsx"
    wb = Workbook.create(
        path,
        template="linear",
        template_args={},
        input_specs=[InputSpec("x", 0.0, 1.0)],
        criterion="determinant",
        measurement_error=0.3,
        seed=1,
        response_name="yield",
    )
    sheet = wb._wb["metadata"]
    for row in list(sheet.iter_rows(min_row=2)):
        if row[0].value == "responses":
            sheet.delete_rows(row[0].row, 1)
    wb.save()
    wb = Workbook.open(path)
    assert wb.metadata().get("responses") is None
    assert wb.response_names() == ["yield"]
    assert wb.response_sigmas() == {"yield": 0.3}
