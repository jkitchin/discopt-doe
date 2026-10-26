"""The information-vs-second-objective front from ``design_tradeoff``.

Every point is an ε-constrained optimal design, so the checks are the ones a
planner relies on when reading the curve: each design honours its level, the
information never rises as the level tightens, and the reference point is the
unconstrained optimum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import DesignCriterion, design_tradeoff, optimal_experiment
from discopt.doe.tradeoff import _efficiency
from discopt.estimate import Experiment, ExperimentModel

THETA = {"k1": 0.3, "k2": 0.05}
TIMES = {f"t{i}": (0.1, 40.0) for i in (1, 2, 3)}


class SamplingTimes(Experiment):
    """A -> B -> C; B and C measured at three sampling times (the design)."""

    def create_model(self, k1=0.3, k2=0.05):
        m = dm.Model("abc")
        k1v = m.continuous("k1", lb=1e-3, ub=10.0)
        k2v = m.continuous("k2", lb=1e-3, ub=10.0)
        times = {f"t{i}": m.continuous(f"t{i}", lb=0.1, ub=40.0) for i in (1, 2, 3)}
        responses = {}
        for name, t in times.items():
            b = k1v / (k2v - k1v) * (dm.exp(-k1v * t) - dm.exp(-k2v * t))
            responses[f"B_{name}"] = b
            responses[f"C_{name}"] = 1.0 - dm.exp(-k1v * t) - b
        return ExperimentModel(
            model=m,
            unknown_parameters={"k1": k1v, "k2": k2v},
            design_inputs=times,
            responses=responses,
            measurement_error=dict.fromkeys(responses, 0.01),
        )


def duration(d):
    return max(d.values())


@pytest.fixture(scope="module")
def front():
    return design_tradeoff(SamplingTimes(), THETA, TIMES, tradeoff=duration, n_levels=5, n_starts=6)


def test_reference_is_the_unconstrained_optimum(front) -> None:
    ref = optimal_experiment(SamplingTimes(), THETA, TIMES, n_starts=6)
    assert front.reference.criterion_value == pytest.approx(ref.criterion_value, rel=1e-6)
    assert front.points[0].efficiency == pytest.approx(1.0)
    assert front.tradeoff_name == "duration"


def test_every_design_honours_its_level_and_information_only_falls(front) -> None:
    assert len(front.points) == 6
    for p in front.points[1:]:
        assert p.tradeoff_value <= p.level + 1e-4
        assert all(0.1 - 1e-9 <= v <= 40.0 + 1e-9 for v in p.design.values())
    eff = [p.efficiency for p in front.points]
    assert all(eff[i] >= eff[i + 1] - 1e-6 for i in range(len(eff) - 1))
    assert 0.0 < eff[-1] < 0.5
    assert all(p.on_front for p in front.points)


def test_knee_summary_rows_and_plot(front) -> None:
    knee = front.knee
    assert knee is not front.points[0] and knee is not front.points[-1]
    text = front.summary()
    assert "<- knee" in text and "duration" in text
    rows = front.rows()
    assert len(rows) == len(front.points) and {"t1", "efficiency", "duration"} <= set(rows[0])
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = front.plot()
    assert ax.get_xlabel() == "duration"
    assert ax.get_ylabel() == "D-efficiency"


class Yield(Experiment):
    """y = 1 - exp(-k x) measured at two settings; y is also what the run sells."""

    def create_model(self, k=0.5):
        m = dm.Model("y")
        kv = m.continuous("k", lb=1e-3, ub=10.0)
        x = m.continuous("x", lb=0.1, ub=10.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": kv},
            design_inputs={"x": x},
            responses={"y": 1.0 - dm.exp(-kv * x)},
            measurement_error={"y": 0.01},
        )


def test_a_response_to_maximize(tmp_path) -> None:
    """Information about k peaks at x = 1/k = 2 (y = 0.63); more yield costs information."""
    result = design_tradeoff(
        Yield(),
        {"k": 0.5},
        {"x": (0.1, 10.0)},
        tradeoff="y",
        sense="maximize",
        levels=[0.8, 0.95, 0.7],
        n_starts=4,
    )
    assert result.reference.design["x"] == pytest.approx(2.0, rel=1e-2)
    assert [p.level for p in result.points[1:]] == [0.7, 0.8, 0.95]
    for p in result.points[1:]:
        assert p.tradeoff_value >= p.level - 1e-4
    # With one parameter the efficiency is the ratio of the sensitivities squared.
    p = result.points[-1]
    x = p.design["x"]
    expected = (x * np.exp(-0.5 * x)) ** 2 / (2.0 * np.exp(-1.0)) ** 2
    assert p.efficiency == pytest.approx(expected, rel=1e-3)


def test_validation() -> None:
    kw = {"tradeoff": "y", "n_starts": 2}
    with pytest.raises(ValueError, match="sense"):
        design_tradeoff(Yield(), {"k": 0.5}, {"x": (0.1, 10.0)}, sense="less", **kw)
    with pytest.raises(ValueError, match="not a response"):
        design_tradeoff(Yield(), {"k": 0.5}, {"x": (0.1, 10.0)}, tradeoff="z", n_starts=2)
    with pytest.raises(TypeError, match="initial_designs"):
        design_tradeoff(Yield(), {"k": 0.5}, {"x": (0.1, 10.0)}, initial_designs=[{"x": 1.0}], **kw)
    # x is already at its cheapest at the information optimum: nothing to trade.
    with pytest.raises(ValueError, match="no trade-off"):
        design_tradeoff(
            Yield(), {"k": 0.5}, {"x": (0.1, 2.0)}, tradeoff=lambda d: -d["x"], n_starts=2
        )


def test_efficiency_definitions() -> None:
    D, A, E = DesignCriterion.D_OPTIMAL, DesignCriterion.A_OPTIMAL, DesignCriterion.E_OPTIMAL
    # D: (det F / det F*)^(1/p); halving det F with p = 1 halves the efficiency.
    assert _efficiency(np.log(0.5), 0.0, D, 1) == pytest.approx(0.5)
    assert _efficiency(np.log(0.25), 0.0, D, 2) == pytest.approx(0.5)
    assert _efficiency(4.0, 2.0, A, 2) == pytest.approx(0.5)  # trace(F^-1) doubled
    assert _efficiency(1.0, 2.0, E, 2) == pytest.approx(0.5)
    assert _efficiency(-np.inf, 0.0, D, 2) == 0.0


def test_response_bounds_hold_at_every_level_and_bound_the_automatic_range() -> None:
    """A cap of y <= 0.9 caps the sweep too: no level asks for more than 0.9."""
    result = design_tradeoff(
        Yield(),
        {"k": 0.5},
        {"x": (0.1, 10.0)},
        tradeoff="y",
        sense="maximize",
        n_levels=3,
        response_bounds={"y": (None, 0.9)},
        n_starts=4,
    )
    assert not result.failed_levels
    assert all(p.tradeoff_value <= 0.9 + 1e-4 for p in result.points)
    assert max(p.level for p in result.points) <= 0.9
