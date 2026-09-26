"""I- and G-optimality on the jax (nonlinear / FIM) design path.

The prediction criteria already existed for models linear in their parameters.
The check that matters is that the two agree: for a model that is linear in its
parameters, the sensitivity rows *are* the basis rows, so the jax path and the
linear path must produce the same number from the same FIM over the same
points. Everything else here pins the properties that a sign error or an
inverted ratio would break.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import compute_fim, ode_experiment
from discopt.doe.design import (
    BatchStrategy,
    DesignCriterion,
    batch_optimal_experiment,
    experiment_region,
    optimal_experiment,
)
from discopt.doe.linear_design import design_region, evaluate_criterion
from discopt.estimate import Experiment, ExperimentModel

SIGMA = 0.05


class Line(Experiment):
    """``y = b0 + b1 x``: linear in its parameters, so both paths apply."""

    def create_model(self, **kwargs):
        m = dm.Model("line")
        b0 = m.continuous("b0", lb=-10.0, ub=10.0)
        b1 = m.continuous("b1", lb=-10.0, ub=10.0)
        x = m.continuous("x", lb=-1.0, ub=1.0)
        return ExperimentModel(m, {"b0": b0, "b1": b1}, {"x": x}, {"y": b0 + b1 * x}, {"y": SIGMA})


def _decay(**kw):
    return ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": 1.0},
        parameters={"k": (0.5, 1e-3, 10.0)},
        measured=["A"],
        sample_times=["t1"],
        design_inputs={"t1": (0.05, 10.0)},
        measurement_error=SIGMA,
        **kw,
    )


# ---------------------------------------------------------------------------
# The two paths agree
# ---------------------------------------------------------------------------


def test_jax_region_matches_the_linear_region_on_a_linear_model() -> None:
    """Same model, same points, same FIM: the I and G values must be identical."""
    points = [{"x": float(v)} for v in np.linspace(-1.0, 1.0, 21)]
    theta = {"b0": 1.0, "b1": 2.0}

    jax_region = experiment_region(Line(), theta, points=points)

    def basis(x):  # one point in, one row out. Plain model row: the FIM carries
        return np.array([1.0, float(np.ravel(x)[0])])  # the 1/sigma^2, not this

    linear_region = design_region(basis, ["x"], points=points)

    # A FIM from an actual design, so the comparison is on a realistic matrix.
    design = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    fim = sum(compute_fim(Line(), theta, d).fim for d in design)

    assert jax_region.average_variance(fim) == pytest.approx(
        evaluate_criterion(fim, "I", region=linear_region), rel=1e-10
    )
    assert jax_region.max_variance(fim) == pytest.approx(
        evaluate_criterion(fim, "G", region=linear_region), rel=1e-10
    )


def test_prediction_variance_is_in_response_units() -> None:
    """``v(x)`` carries 1/sigma^2 through the FIM, so it is a variance of y-hat.

    A three-point design of a straight line predicts its centre with variance
    sigma^2 / n, which is a closed form worth pinning.
    """
    theta = {"b0": 1.0, "b1": 2.0}
    design = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    fim = sum(compute_fim(Line(), theta, d).fim for d in design)
    centre = experiment_region(Line(), theta, points=[{"x": 0.0}])
    assert centre.average_variance(fim) == pytest.approx(SIGMA**2 / 3, rel=1e-8)


# ---------------------------------------------------------------------------
# Properties of the criteria
# ---------------------------------------------------------------------------


def test_average_never_exceeds_the_maximum() -> None:
    theta = {"k": 0.5}
    exp = _decay()
    region = experiment_region(exp, theta, bounds={"t1": (0.05, 10.0)}, n_points=64)
    fim = compute_fim(exp, theta, {"t1": 2.0}).fim
    assert region.average_variance(fim) <= region.max_variance(fim) + 1e-12


def test_one_parameter_makes_d_i_and_g_agree() -> None:
    """With p = 1 the FIM is a scalar, so every criterion ranks designs alike.

    The D-optimal sampling time for exponential decay is the closed form
    t* = 1/k, which is what all three must find.
    """
    exp = _decay()
    theta = {"k": 0.5}
    bounds = {"t1": (0.05, 10.0)}
    designs = {
        name: optimal_experiment(exp, theta, bounds, criterion=name, n_starts=5, seed=0).design[
            "t1"
        ]
        for name in (
            DesignCriterion.D_OPTIMAL,
            DesignCriterion.I_OPTIMAL,
            DesignCriterion.G_OPTIMAL,
        )
    }
    for name, t in designs.items():
        assert t == pytest.approx(1.0 / theta["k"], rel=2e-2), f"{name} picked t = {t}"


def test_an_i_optimal_design_beats_a_d_optimal_one_at_its_own_criterion() -> None:
    """Otherwise the search is not optimizing what it claims to.

    A second, fixed sampling time keeps the FIM full rank: with B measured at
    t1 alone, two rate constants give a rank-1 FIM everywhere, both criteria
    are round-off (~1e13), and the comparison is between noise.
    """
    exp = ode_experiment(
        lambda t, x, p, u: {"A": -p["k1"] * x["A"], "B": p["k1"] * x["A"] - p["k2"] * x["B"]},
        states={"A": 1.0, "B": 0.0},
        parameters={"k1": (0.5, 1e-3, 10.0), "k2": (0.2, 1e-3, 10.0)},
        measured=["B"],
        sample_times=["t1", 5.0],
        design_inputs={"t1": (0.05, 20.0)},
        measurement_error=0.02,
    )
    theta = {"k1": 0.5, "k2": 0.2}
    bounds = {"t1": (0.05, 20.0)}
    region = experiment_region(exp, theta, bounds=bounds, n_points=128)

    d_design = optimal_experiment(exp, theta, bounds, n_starts=4, seed=0).design
    i_design = optimal_experiment(
        exp,
        theta,
        bounds,
        criterion=DesignCriterion.I_OPTIMAL,
        prediction_region=region,
        n_starts=4,
        seed=0,
    ).design

    i_of = lambda d: region.average_variance(compute_fim(exp, theta, d).fim)  # noqa: E731
    assert i_of(i_design) <= i_of(d_design) + 1e-12


@pytest.mark.parametrize("strategy", [BatchStrategy.GREEDY, BatchStrategy.JOINT])
@pytest.mark.parametrize("criterion", [DesignCriterion.I_OPTIMAL, DesignCriterion.G_OPTIMAL])
def test_batch_designs_accept_the_prediction_criteria(strategy, criterion) -> None:
    exp = _decay()
    theta = {"k": 0.5}
    res = batch_optimal_experiment(
        exp,
        theta,
        {"t1": (0.05, 10.0)},
        n_experiments=3,
        criterion=criterion,
        strategy=strategy,
        n_starts=3,
        seed=0,
    )
    assert len(res.designs) == 3
    assert np.isfinite(res.criterion_value) and res.criterion_value > 0
    # Three runs of the same experiment cut the variance, so the batch must
    # score better (lower) than any one of its runs alone.
    region = experiment_region(exp, theta, bounds={"t1": (0.05, 10.0)}, n_points=64)
    single = compute_fim(exp, theta, res.designs[0]).fim
    alone = (
        region.average_variance(single)
        if criterion == DesignCriterion.I_OPTIMAL
        else region.max_variance(single)
    )
    assert res.criterion_value < alone


# ---------------------------------------------------------------------------
# The region itself
# ---------------------------------------------------------------------------


def test_every_response_contributes_a_row() -> None:
    """A dynamic experiment measuring four times predicts four things per point."""
    exp = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": 1.0},
        parameters={"k": (0.5, 1e-3, 10.0)},
        measured=["A"],
        sample_times=[1.0, 2.0, 4.0, 8.0],
        design_inputs={"C0": (0.5, 2.0)},
        measurement_error=SIGMA,
    )
    points = [{"C0": 0.5}, {"C0": 1.0}, {"C0": 2.0}]
    region = experiment_region(exp, {"k": 0.5}, points=points)
    assert region.moment_rows.shape == (len(points) * len(exp.response_names), 1)
    assert region.weights.sum() == pytest.approx(1.0)


def test_a_prediction_criterion_without_a_region_says_what_to_pass() -> None:
    from discopt.doe.design import _criterion_from_fim

    with pytest.raises(ValueError, match="needs one"):
        _criterion_from_fim(np.eye(2), DesignCriterion.I_OPTIMAL, None)


def test_region_needs_points_or_bounds() -> None:
    with pytest.raises(ValueError, match="bounds .*or points"):
        experiment_region(_decay(), {"k": 0.5})
