"""
Tests for discopt.doe.sequential -- sequential DoE loop.

Test classes:
  - TestSequentialDoE: the main sequential_doe() function
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import DoERound, sequential_doe
from discopt.estimate import Experiment, ExperimentModel

# ──────────────────────────────────────────────────────────
# Helper experiment
# ──────────────────────────────────────────────────────────


class SimpleExperiment(Experiment):
    """y_i = k * x_i for multiple x_i values.

    Design input: x (measurement location).
    Unknown parameter: k (slope).
    """

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("simple")
        k = m.continuous("k", lb=0.01, ub=20)

        responses = {}
        errors = {}
        for i, xi in enumerate(self.x_data):
            responses[f"y_{i}"] = k * xi
            errors[f"y_{i}"] = 0.1

        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────


class TestSequentialDoE:
    def test_no_runner_returns_one_round(self):
        """Without run_experiment, returns after first recommendation."""
        x_data = np.array([1.0, 2.0, 3.0])
        exp = SimpleExperiment(x_data)
        k_true = 3.0
        data = {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        history = sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
            n_rounds=5,
        )
        assert len(history) == 1
        assert isinstance(history[0], DoERound)
        assert history[0].round == 0
        assert history[0].data_collected is None

    def test_estimation_in_first_round(self):
        """First round estimates parameters correctly."""
        x_data = np.array([1.0, 2.0, 3.0, 5.0])
        exp = SimpleExperiment(x_data)
        k_true = 2.5
        data = {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        history = sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
        )
        est = history[0].estimation
        assert est.parameters["k"] == pytest.approx(k_true, abs=0.1)

    def test_callback_called(self):
        """Callback receives DoERound."""
        x_data = np.array([1.0, 2.0])
        exp = SimpleExperiment(x_data)
        data = {f"y_{i}": 2.0 * x_data[i] for i in range(len(x_data))}

        received = []
        sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
            callback=lambda r: received.append(r),
        )
        assert len(received) == 1
        assert isinstance(received[0], DoERound)

    def test_with_experiment_runner(self):
        """Multiple rounds with synthetic experiment runner."""
        x_data = np.array([1.0, 2.0])
        exp = SimpleExperiment(x_data)
        k_true = 3.0
        data = {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        # Synthetic runner just returns data at k_true
        def runner(design):
            return {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        history = sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
            n_rounds=3,
            run_experiment=runner,
        )
        assert len(history) == 3
        # Each round should have data
        for r in history:
            assert r.data_collected is not None

    def test_prior_fim_not_double_counted(self, monkeypatch):
        """The prior FIM fed to the design step is the cumulative FIM, once.

        Regression: est.fim is already fit on *all* accumulated data, so
        summing it across rounds (the old behaviour) counted early data
        multiple times. Each round's prior_fim must equal that round's
        estimation FIM.
        """
        import discopt.doe.sequential as seq

        x_data = np.array([1.0, 2.0, 3.0])
        exp = SimpleExperiment(x_data)
        k_true = 2.0
        data = {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        def runner(design):
            return {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        captured = []
        real = seq.optimal_experiment

        def spy(experiment, params, bounds, **kw):
            captured.append(np.asarray(kw.get("prior_fim")))
            return real(experiment, params, bounds, **kw)

        monkeypatch.setattr(seq, "optimal_experiment", spy)

        history = sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
            n_rounds=3,
            run_experiment=runner,
        )
        assert len(captured) == 3
        for rnd, cap in zip(history, captured):
            np.testing.assert_allclose(cap, np.asarray(rnd.estimation.fim), rtol=1e-6)

    def test_fim_accumulates(self):
        """FIM det should increase over rounds (more data)."""
        x_data = np.array([1.0, 2.0, 3.0])
        exp = SimpleExperiment(x_data)
        k_true = 2.0
        data = {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        def runner(design):
            return {f"y_{i}": k_true * x_data[i] for i in range(len(x_data))}

        history = sequential_doe(
            experiment=exp,
            initial_data=data,
            initial_guess={"k": 1.0},
            design_bounds={},
            n_rounds=3,
            run_experiment=runner,
        )
        # FIM det should increase (more data = more information)
        dets = [np.linalg.det(r.estimation.fim) for r in history]
        # The FIM from estimation on growing data should be non-decreasing
        # (each round adds the same data, so FIM should grow)
        assert dets[-1] >= dets[0] - 1e-6
