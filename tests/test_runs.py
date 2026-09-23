"""Campaigns: a per-run model stacked over run conditions (discopt.doe.runs).

The load-bearing claims:

* a campaign's FIM is the sum of the per-run FIMs, for every kind of per-run
  model (SymbolicModel, ODEExperiment, generic Experiment);
* the whole-campaign Experiment drives the existing tools (identifiability,
  estimability, profile likelihood, sequential design) with no hand-encoding;
* ``fit_campaign`` recovers the truth, and its standard errors match the
  Monte Carlo spread of the estimates.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    ParametricSurrogate,
    SymbolicModel,
    compute_fim,
    diagnose_identifiability,
    estimability_rank,
    fit_least_squares,
    ode_experiment,
    profile_likelihood,
    sequential_doe,
)
from discopt.doe.runs import (
    CampaignExperiment,
    campaign_experiment,
    fit_campaign,
    symbolic_experiment,
)
from discopt.estimate import Experiment, ExperimentModel

THETA = {"a": 2.0, "b": 0.7}
RUNS = [{"x": x} for x in (0.5, 1.0, 2.0, 4.0)]


def decay(sigma: float = 0.1) -> SymbolicModel:
    return SymbolicModel(
        source="a*exp(-b*x)",
        parameter_names=("a", "b"),
        input_names=("x",),
        measurement_error=sigma,
    )


class LineExperiment(Experiment):
    """y = a*x + b at design input x (a plain discopt.modeling Experiment)."""

    def create_model(self, **kwargs):
        m = dm.Model("line")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)
        x = m.continuous("x", lb=0.0, ub=10.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b},
            design_inputs={"x": x},
            responses={"y": a * x + b},
            measurement_error={"y": 0.2},
        )


# ---------------------------------------------------------------------------
# The campaign FIM is the sum of the per-run FIMs
# ---------------------------------------------------------------------------


def test_symbolic_campaign_fim_is_the_sum_of_per_run_fims():
    m = decay()
    camp = campaign_experiment(m, RUNS)
    assert isinstance(camp, CampaignExperiment)
    assert camp.response_names == ["y[0]", "y[1]", "y[2]", "y[3]"]
    np.testing.assert_allclose(compute_fim(camp, THETA).fim, m.fim(THETA, RUNS), rtol=1e-10)


def test_generic_experiment_campaign_fim_is_the_sum():
    exp = LineExperiment()
    runs = [{"x": x} for x in (0.0, 3.0, 10.0)]
    camp = campaign_experiment(exp, runs)
    theta = {"a": 1.5, "b": -0.5}
    per_run = sum(compute_fim(exp, theta, r).fim for r in runs)
    np.testing.assert_allclose(compute_fim(camp, theta).fim, per_run, rtol=1e-10)


def test_ode_campaign_fim_matches_the_analytic_sensitivities():
    """A -> B, first order, measured at t = 1 and 3 from two initial loads:
    dA/dk = -t A0 exp(-k t)."""
    ode = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": "A0"},
        parameters={"k": (0.5, 1e-3, 10.0)},
        measured=["A"],
        sample_times=[1.0, 3.0],
        design_inputs={"A0": (0.5, 2.0)},
        measurement_error=0.02,
    )
    camp = campaign_experiment(ode, [{"A0": 1.0}, {"A0": 2.0}])
    analytic = (
        sum((t * a0 * np.exp(-0.5 * t)) ** 2 for a0 in (1.0, 2.0) for t in (1.0, 3.0)) / 0.02**2
    )
    assert compute_fim(camp, {"k": 0.5}).fim[0, 0] == pytest.approx(analytic, rel=1e-6)
    assert camp.response_names[0] == "A@1[0]"


def test_run_labels_and_measurement_error_override():
    camp = campaign_experiment(
        decay(), [dict(r, run_id=f"r{i}") for i, r in enumerate(RUNS)], measurement_error=0.5
    )
    assert camp.response_names[0] == "y[r0]"
    em = camp.create_model()
    assert all(v == 0.5 for v in em.measurement_error.values())
    with pytest.raises(ValueError, match="missing condition"):
        campaign_experiment(decay(), [{"z": 1.0}])


# ---------------------------------------------------------------------------
# Existing tools work on whole campaigns
# ---------------------------------------------------------------------------


def test_identifiability_flags_a_product_only_pair_on_a_campaign():
    """y = a*b*x: only the product a*b enters, so no campaign can separate them."""
    m = SymbolicModel(source="a*b*x", parameter_names=("a", "b"), input_names=("x",))
    camp = campaign_experiment(m, RUNS)
    diag = diagnose_identifiability(camp, THETA)
    assert not diag.is_identifiable
    rank = estimability_rank(camp, THETA)
    assert len(rank.recommended_subset) == 1


def test_profile_likelihood_on_a_campaign_is_bounded():
    m = decay(0.05)
    rng = np.random.default_rng(3)
    runs = RUNS * 3
    camp = campaign_experiment(m, runs)
    data = camp.data_from_runs([{"y": m.predict(THETA, r) + rng.normal(0, 0.05)} for r in runs])
    prof = profile_likelihood(camp, data, "b")
    assert prof.ci_lower < THETA["b"] < prof.ci_upper


# ---------------------------------------------------------------------------
# fit_campaign
# ---------------------------------------------------------------------------


def test_fit_campaign_recovers_truth_and_its_ses_match_the_spread():
    m = decay(0.1)
    runs = RUNS * 3
    rng = np.random.default_rng(0)
    est, se = [], []
    for _ in range(150):
        rows = [dict(r, y=m.predict(THETA, r) + rng.normal(0, 0.1)) for r in runs]
        out = fit_campaign(m, rows, {"a": 1.0, "b": 1.0}, sigma="declared")
        est.append([out["estimates"]["a"], out["estimates"]["b"]])
        se.append([out["std_errors"]["a"], out["std_errors"]["b"]])
    est, se = np.array(est), np.array(se)
    assert est.mean(axis=0) == pytest.approx([2.0, 0.7], abs=0.02)
    # Declared-sigma SEs describe the sampling spread of the estimates.
    assert se.mean(axis=0) == pytest.approx(est.std(axis=0, ddof=1), rel=0.15)
    assert out["sigma_source"] == "declared"
    np.testing.assert_allclose(np.linalg.inv(out["fim"]), out["covariance"], rtol=1e-8)


def test_fit_campaign_ode_with_run_conditions():
    ode = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": "A0"},
        parameters={"k": (0.5, 1e-3, 10.0)},
        measured=["A"],
        sample_times=[1.0, 3.0],
        design_inputs={"A0": (0.5, 2.0)},
        measurement_error=0.01,
    )
    rng = np.random.default_rng(1)
    runs = [{"A0": a0} for a0 in (0.6, 1.0, 1.5, 2.0)]
    truth = {"k": 0.8}
    rows = [
        dict(r, **{k: v + rng.normal(0, 0.01) for k, v in ode.predict(truth, r).items()})
        for r in runs
    ]
    out = fit_campaign(ode, rows)
    assert out["estimates"]["k"] == pytest.approx(0.8, abs=4 * out["std_errors"]["k"])
    assert out["n_observations"] == 8


def test_fit_campaign_warns_when_parameters_cannot_be_separated():
    m = SymbolicModel(source="a*b*x", parameter_names=("a", "b"), input_names=("x",))
    rows = [dict(r, y=1.4 * r["x"]) for r in RUNS]
    with pytest.warns(UserWarning, match="cannot identify"):
        fit_campaign(m, rows, {"a": 1.0, "b": 1.0})


# ---------------------------------------------------------------------------
# sequential_doe on runs with conditions
# ---------------------------------------------------------------------------


def test_sequential_doe_with_design_inputs_shrinks_the_uncertainty():
    m = decay(0.05)
    rng = np.random.default_rng(4)

    def run(design):
        return {"y": m.predict(THETA, design) + rng.normal(0, 0.05)}

    initial = [dict(r, **run(r)) for r in ({"x": 0.2}, {"x": 0.5})]
    history = sequential_doe(
        m,
        None,
        {"a": 1.5, "b": 1.0},
        {"x": (0.0, 5.0)},
        initial_runs=initial,
        n_rounds=3,
        run_experiment=run,
    )
    assert len(history) == 3
    assert [len(h.runs) for h in history] == [2, 3, 4]
    # The designed runs are real conditions, not replicates of one key.
    assert len({round(r["x"], 6) for r in history[-1].runs}) >= 3
    se = [np.sqrt(np.diag(h.estimation.covariance)) for h in history]
    assert se[-1][1] < se[0][1]


def test_sequential_doe_rejects_both_data_forms():
    with pytest.raises(ValueError, match="either"):
        sequential_doe(decay(), {"y": 1.0}, THETA, {"x": (0, 1)}, initial_runs=[{"x": 0.1, "y": 1}])


# ---------------------------------------------------------------------------
# fit_least_squares: known sigma and rank warning
# ---------------------------------------------------------------------------


def test_fit_least_squares_known_sigma():
    m = decay(0.1)
    rng = np.random.default_rng(5)
    rows = [dict(r, y=m.predict(THETA, r) + rng.normal(0, 0.1)) for r in RUNS * 2]
    declared = fit_least_squares(m, rows, {"a": 1, "b": 1}, sigma="declared")
    given = fit_least_squares(m, rows, {"a": 1, "b": 1}, sigma=0.2)
    assert declared["sigma"] == pytest.approx(0.1) and declared["sigma_source"] == "declared"
    assert given["sigma_source"] == "given"
    assert given["std_errors"]["b"] == pytest.approx(2 * declared["std_errors"]["b"])
    np.testing.assert_allclose(np.linalg.inv(given["fim"]), given["covariance"], rtol=1e-8)
    with pytest.raises(ValueError, match="sigma"):
        fit_least_squares(m, rows, sigma="bogus")


def test_fit_least_squares_warns_on_rank_deficiency():
    m = SymbolicModel(source="a*b*x", parameter_names=("a", "b"), input_names=("x",))
    rows = [dict(r, y=1.4 * r["x"]) for r in RUNS]
    with pytest.warns(UserWarning, match=r"cannot identify every parameter.*\['a', 'b'\]"):
        fit_least_squares(m, rows, {"a": 1.0, "b": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_least_squares(decay(), [dict(r, y=decay().predict(THETA, r)) for r in RUNS])


# ---------------------------------------------------------------------------
# One equation, everywhere
# ---------------------------------------------------------------------------


def test_symbolic_experiment_fim_matches_symbolic_model():
    m = decay()
    exp = symbolic_experiment(m, input_bounds={"x": (0.0, 5.0)})
    np.testing.assert_allclose(
        compute_fim(exp, THETA, {"x": 2.0}).fim, m.fim(THETA, [{"x": 2.0}]), rtol=1e-10
    )


def test_parametric_surrogate_from_symbolic():
    m = decay(0.05)
    s = ParametricSurrogate.from_symbolic(
        m, {"a": 1.0, "b": 1.0}, parameter_bounds={"a": (0, 10), "b": (0, 5)}
    )
    X = np.linspace(0.0, 4.0, 8).reshape(-1, 1)
    y = np.array([m.predict(THETA, {"x": x}) for x in X[:, 0]])
    s.fit(X, y)
    mu, sd = s.predict(np.array([[1.5], [3.0]]))
    np.testing.assert_allclose(
        mu, [m.predict(THETA, {"x": 1.5}), m.predict(THETA, {"x": 3.0})], rtol=1e-6
    )
    assert s.parameters_["b"] == pytest.approx(0.7, rel=1e-6)
    assert np.all(sd > 0)


# ---------------------------------------------------------------------------
# Deviance-based tools on a campaign
# ---------------------------------------------------------------------------


def test_deviance_of_a_campaign_uses_its_own_sigma():
    """A campaign spells sigma nowhere an attribute lookup can see it.

    It keeps the errors on the per-run model and states them only when it
    builds the campaign model, so a deviance that reads attributes alone fell
    back to sigma = 1 -- a factor of 1/0.1**2 = 100 on this campaign, and
    10,000 on one measured to 0.01. Every deviance-based answer moves with it.
    """
    from discopt.doe._estimation import DevianceFunction

    sigma = 0.1
    camp = campaign_experiment(decay(sigma), RUNS)
    clean = camp.predict(THETA)
    rng = np.random.default_rng(0)
    data = {k: float(v) + rng.normal(0, sigma) for k, v in clean.items()}

    dev = DevianceFunction(camp, data, THETA)
    by_hand = sum(((data[k] - clean[k]) / sigma) ** 2 for k in clean)
    assert dev(dev.vector(THETA)) == pytest.approx(by_hand, rel=1e-9)


def test_profile_of_a_combination_works_on_a_campaign():
    """Only this path builds a DevianceFunction, and it called predict(theta, design).

    A campaign's conditions live in its runs, so its predict takes theta alone
    and the two-argument call raised TypeError. Profiling a *parameter* never
    reaches that code, which is why the existing campaign profile test passed
    while a combination could not be profiled at all.
    """
    sigma = 0.05
    m = decay(sigma)
    runs = RUNS * 3
    camp = campaign_experiment(m, runs)
    rng = np.random.default_rng(3)
    data = camp.data_from_runs([{"y": m.predict(THETA, r) + rng.normal(0, sigma)} for r in runs])

    combination = profile_likelihood(camp, data, expression="a*b", name="a*b")
    assert combination.shape == "bounded"
    assert combination.ci_lower < THETA["a"] * THETA["b"] < combination.ci_upper
