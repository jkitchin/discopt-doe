"""Several objectives in the model-based optimization loop.

``model_based_optimize_round(objectives=...)`` fits one mechanistic model to
every objective response at once and runs ParEGO on the model's scalarized
predictions. The failures worth guarding are silent ones: a joint fit that
ignores one response's noise level, an uncertainty that is not the gradient of
the scalarization, and a weight that buys the objective it should not.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe.model_based import (
    MultiResponseParametricSurrogate,
    _ChebyshevScalarizedModel,
    model_based_optimize_round,
)
from discopt.doe.workbook import InputSpec, Workbook
from discopt.estimate import Experiment, ExperimentModel

openpyxl = pytest.importorskip("openpyxl")

TRUE = {"a": 2.0, "b": 3.0}


class TwoResponses(Experiment):
    """``gain = a x`` (more is better) and ``cost = b x^2`` (less is better)."""

    def create_model(self, a=1.0, b=1.0):
        m = dm.Model("two")
        av = m.continuous("a", lb=0.01, ub=10.0)
        bv = m.continuous("b", lb=0.01, ub=10.0)
        x = m.continuous("x", lb=0.0, ub=1.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"a": av, "b": bv},
            design_inputs={"x": x},
            responses={"gain": av * x, "cost": bv * x * x},
            measurement_error={"gain": 0.01, "cost": 0.02},
        )


def _truth(x: float) -> dict[str, float]:
    return {"gain": TRUE["a"] * x, "cost": TRUE["b"] * x * x}


def _workbook(tmp_path, xs=(0.1, 0.4, 0.7, 1.0)):
    path = tmp_path / "mo.xlsx"
    Workbook.create(
        path,
        template=None,
        template_args={},
        input_specs=[InputSpec("x", 0.0, 1.0)],
        criterion="model-based",
        measurement_error=0.01,
        seed=0,
        response_name="gain",
        module_callable=None,
        param_initial_guess={"a": 1.0, "b": 1.0},
    )
    wb = Workbook.open(path)
    wb.append_runs(0, [{"x": float(x)} for x in xs])
    wb.save()
    _fill(path)
    return path


def _fill(path):
    """Write noiseless responses into every pending run, adding the cost column."""
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    header = [c.value for c in sheet[1]]
    if "cost" not in header:
        header.append("cost")
        sheet.cell(row=1, column=len(header), value="cost")
    for row in sheet.iter_rows(min_row=2):
        if row[0].value is None or row[header.index("gain")].value is not None:
            continue
        values = _truth(float(row[header.index("x")].value))
        for name, value in values.items():
            sheet.cell(row=row[0].row, column=header.index(name) + 1, value=value)
    book.save(path)


OBJECTIVES = {"gain": "maximize", "cost": "minimize"}


def test_joint_fit_recovers_both_parameters_and_weights_by_noise() -> None:
    xs = np.linspace(0.1, 1.0, 6)[:, None]
    Y = np.array([[_truth(x)["gain"], _truth(x)["cost"]] for x in xs.ravel()])
    s = MultiResponseParametricSurrogate(
        TwoResponses(),
        input_names=["x"],
        response_names=["gain", "cost"],
        measurement_noise_var=[0.01**2, 0.02**2],
    )
    s.fit(xs, Y)
    assert s.parameters_["a"] == pytest.approx(2.0, abs=1e-6)
    assert s.parameters_["b"] == pytest.approx(3.0, abs=1e-6)

    # The FIM is J^T Σ^-1 J with each response weighted by its own σ.
    fim_a = np.sum(xs.ravel() ** 2) / 0.01**2
    fim_b = np.sum(xs.ravel() ** 4) / 0.02**2
    np.testing.assert_allclose(np.diag(s.fim_), [fim_a, fim_b], rtol=1e-6)

    mu, se, _ = s.predict_all(np.array([[0.5]]))
    np.testing.assert_allclose(mu, [[1.0, 0.75]], atol=1e-6)
    np.testing.assert_allclose(se[0, 0], 0.5 / np.sqrt(fim_a), rtol=1e-6)


def test_too_few_measurements_raise() -> None:
    s = MultiResponseParametricSurrogate(
        TwoResponses(),
        input_names=["x"],
        response_names=["gain"],
        measurement_noise_var=[1.0],
    )
    with pytest.raises(ValueError, match="at least 2"):
        s.fit(np.array([[0.5]]), np.array([[1.0]]))


def test_scalarized_uncertainty_is_the_gradient_of_the_scalarization() -> None:
    """σ_g² = ∇θg Σ ∇θgᵀ, checked against a finite-difference gradient."""
    xs = np.linspace(0.1, 1.0, 6)[:, None]
    Y = np.array([[_truth(x)["gain"], _truth(x)["cost"]] for x in xs.ravel()])
    s = MultiResponseParametricSurrogate(
        TwoResponses(),
        input_names=["x"],
        response_names=["gain", "cost"],
        measurement_noise_var=[0.01**2, 0.02**2],
    ).fit(xs, Y)
    signs = np.array([1.0, -1.0])
    lo, span = np.array([0.0, -3.0]), np.array([2.0, 3.0])
    w = np.array([0.3, 0.7])
    g_model = _ChebyshevScalarizedModel(s, w, signs, lo, span)
    X = np.array([[0.25], [0.6], [0.9]])
    g0, sd = g_model.predict_latent(X)

    theta0 = dict(s.parameters_)
    grads = []
    for name in s.parameter_names_:
        h = 1e-6
        s.parameters_ = {**theta0, name: theta0[name] + h}
        g_plus, _ = g_model.predict_latent(X)
        s.parameters_ = dict(theta0)
        grads.append((g_plus - g0) / h)
    G = np.column_stack(grads)
    expected = np.sqrt(np.einsum("ki,ij,kj->k", G, s.covariance_, G))
    np.testing.assert_allclose(sd, expected, rtol=1e-4)


def test_a_round_returns_both_fronts_and_spreads_its_weights(tmp_path) -> None:
    path = _workbook(tmp_path)
    result = model_based_optimize_round(
        path,
        experiment=TwoResponses(),
        initial_guess={"a": 1.0, "b": 1.0},
        objectives=OBJECTIVES,
        batch_size=4,
        n_candidates=256,
        seed=0,
    )
    assert result.incumbent_x is None and result.incumbent_y is None
    assert result.parameters["a"] == pytest.approx(2.0, abs=1e-4)
    assert result.parameters["b"] == pytest.approx(3.0, abs=1e-4)
    # Every measured run is non-dominated here: more gain always costs more.
    assert len(result.pareto_front) == 4
    front = result.predicted_front
    assert len(front) > 50
    assert all(front[i]["cost"] <= front[i + 1]["cost"] for i in range(len(front) - 1))
    assert {"gain_se", "cost_se"} <= set(front[0])
    assert len(result.scalarization_weights) == 4
    assert len({tuple(np.round(w, 3)) for w in result.scalarization_weights}) == 4
    assert len({d["x"] for d in result.next_designs}) == 4
    assert len(Workbook.open(path).pending_runs()) == 4


def test_a_bigger_weight_buys_more_of_that_objective(tmp_path) -> None:
    """The inversion this guards against still returns a front -- the wrong end of it."""
    xs = np.linspace(0.1, 1.0, 6)[:, None]
    Y = np.array([[_truth(x)["gain"], _truth(x)["cost"]] for x in xs.ravel()])
    s = MultiResponseParametricSurrogate(
        TwoResponses(),
        input_names=["x"],
        response_names=["gain", "cost"],
        measurement_noise_var=[0.01**2, 0.02**2],
    ).fit(xs, Y)
    signs = np.array([1.0, -1.0])
    signed = Y * signs
    lo, span = signed.min(axis=0), np.ptp(signed, axis=0)
    X = np.linspace(0.0, 1.0, 101)[:, None]
    favour_gain, _ = _ChebyshevScalarizedModel(s, [0.9, 0.1], signs, lo, span).predict_latent(X)
    favour_cost, _ = _ChebyshevScalarizedModel(s, [0.1, 0.9], signs, lo, span).predict_latent(X)
    assert X[np.argmax(favour_gain), 0] > X[np.argmax(favour_cost), 0] + 0.3


def test_objectives_are_validated(tmp_path) -> None:
    path = _workbook(tmp_path)
    kw = {"experiment": TwoResponses(), "initial_guess": {"a": 1.0, "b": 1.0}, "seed": 0}
    with pytest.raises(ValueError, match="at least two"):
        model_based_optimize_round(path, objectives=["gain"], **kw)
    with pytest.raises(ValueError, match="not responses of the experiment"):
        model_based_optimize_round(path, objectives=["gain", "nope"], **kw)
    with pytest.raises(ValueError, match="maximize"):
        model_based_optimize_round(path, objectives={"gain": "up", "cost": "minimize"}, **kw)
    with pytest.raises(ValueError, match="needs objectives="):
        model_based_optimize_round(path, measurement_noise_override={"gain": 0.1}, **kw)


def test_per_response_noise_override_changes_the_fit_weights(tmp_path) -> None:
    path = _workbook(tmp_path)
    kw = {
        "experiment": TwoResponses(),
        "initial_guess": {"a": 1.0, "b": 1.0},
        "objectives": OBJECTIVES,
        "n_candidates": 64,
        "seed": 0,
    }
    base = model_based_optimize_round(path, **kw)
    noisy = model_based_optimize_round(path, measurement_noise_override={"cost": 0.2}, **kw)
    # Ten times the σ on cost: ten times the standard error of b; a unchanged.
    assert noisy.parameter_se["b"] == pytest.approx(10 * base.parameter_se["b"], rel=1e-3)
    assert noisy.parameter_se["a"] == pytest.approx(base.parameter_se["a"], rel=1e-3)
