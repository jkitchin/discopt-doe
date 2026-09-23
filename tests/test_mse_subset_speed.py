"""``mse_subset_selection`` on the compiled deviance instead of p estimator solves.

The p fits minimize the *same* deviance over different subsets of its
coordinates, so the function is built once and reused. What has to be proved is
that this changes only the cost: the deviances, the critical ratios and the
recommended subset must be what the estimator path gives, and an experiment
with no compiled path must still go through the estimator.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe.estimability import _subset_deviances, estimability_rank, mse_subset_selection
from discopt.estimate import Experiment, ExperimentModel

XS = np.linspace(0.5, 5.0, 24)
SIGMA = 0.1


class Poly(Experiment):
    """``y = sum_j a_j x^j`` at fixed x: linear in the parameters, p of them."""

    def __init__(self, p: int = 5) -> None:
        self.p = p

    def create_model(self, **kwargs):
        m = dm.Model("poly")
        params = {f"a{j}": m.continuous(f"a{j}", lb=-50.0, ub=50.0) for j in range(self.p)}
        responses = {
            f"y{i}": sum(params[f"a{j}"] * float(x) ** j for j in range(self.p))
            for i, x in enumerate(XS)
        }
        return ExperimentModel(m, params, {}, responses, {k: SIGMA for k in responses})


class Arrhenius(Experiment):
    """Nonlinear in its parameters, so the two optimizers work harder to agree."""

    TEMPS = np.linspace(300.0, 400.0, 12)

    def create_model(self, **kwargs):
        m = dm.Model("arr")
        k0 = m.continuous("k0", lb=1e-3, ub=1e3)
        ea = m.continuous("ea", lb=0.0, ub=60.0)
        c = m.continuous("c", lb=-5.0, ub=5.0)
        responses = {
            f"r{i}": k0 * dm.exp(-ea * (1000.0 / float(t) - 1000.0 / 350.0)) + c
            for i, t in enumerate(self.TEMPS)
        }
        return ExperimentModel(
            m, {"k0": k0, "ea": ea, "c": c}, {}, responses, {k: 0.05 for k in responses}
        )


def _poly_data(p: int, seed: int = 0):
    truth = {f"a{j}": 0.7**j for j in range(p)}
    rng = np.random.default_rng(seed)
    data = {
        f"y{i}": float(sum(truth[f"a{j}"] * x**j for j in range(p)) + rng.normal(0, SIGMA))
        for i, x in enumerate(XS)
    }
    return truth, data


def _arrhenius_data(seed: int = 0):
    truth = {"k0": 2.0, "ea": 8.0, "c": 0.5}
    rng = np.random.default_rng(seed)
    exp = Arrhenius()
    clean = exp.create_model(**truth)  # noqa: F841 - the closed form is used below
    data = {
        f"r{i}": float(
            truth["k0"] * np.exp(-truth["ea"] * (1000.0 / t - 1000.0 / 350.0))
            + truth["c"]
            + rng.normal(0, 0.05)
        )
        for i, t in enumerate(Arrhenius.TEMPS)
    }
    return truth, data


def _estimator_deviances(experiment, data, nominal, ranking):
    """The deviances the old implementation computed, one estimator fit per k."""
    from discopt.doe._estimation import estimate_parameters

    out = np.zeros(len(ranking))
    for k in range(1, len(ranking) + 1):
        fixed = {name: nominal[name] for name in ranking[k:]}
        res = estimate_parameters(
            experiment, data, initial_guess=nominal, fixed_parameters=fixed or None, n_starts=1
        )
        out[k - 1] = float(res.objective)
    return out


def test_compiled_deviances_match_the_estimator_on_a_linear_model() -> None:
    exp = Poly(5)
    truth, data = _poly_data(5)
    ranking = estimability_rank(exp, truth).ranking

    fast = _subset_deviances(
        exp, data, dict(truth), ranking, design_values=None, n_starts=1, extra={}
    )
    slow = _estimator_deviances(exp, data, dict(truth), ranking)
    np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-8)


def test_compiled_deviances_match_the_estimator_on_a_nonlinear_model() -> None:
    exp = Arrhenius()
    truth, data = _arrhenius_data()
    ranking = estimability_rank(exp, truth).ranking

    fast = _subset_deviances(
        exp, data, dict(truth), ranking, design_values=None, n_starts=1, extra={}
    )
    slow = _estimator_deviances(exp, data, dict(truth), ranking)
    # Two different optimizers on the same objective: the minima agree, and
    # neither is allowed to be the worse one by any margin that matters.
    np.testing.assert_allclose(fast, slow, rtol=1e-4, atol=1e-6)


def test_the_recommendation_is_unchanged() -> None:
    exp = Poly(5)
    truth, data = _poly_data(5)
    result = mse_subset_selection(exp, data, truth)

    ranking = estimability_rank(exp, truth).ranking
    slow = _estimator_deviances(exp, data, dict(truth), ranking)
    np.testing.assert_allclose(result.objectives, slow, rtol=1e-6, atol=1e-8)
    assert list(result.recommended_subset) == ranking[: result.recommended_k]
    assert result.recommended_k == int(np.nanargmin(result.corrected_ratios)) + 1


def test_the_estimator_is_not_called_on_a_compiled_experiment(monkeypatch) -> None:
    """The whole point: p estimator solves become p minimizations of one function."""
    import discopt.doe._estimation as est

    calls = {"n": 0}
    real = est.estimate_parameters

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(est, "estimate_parameters", counting)

    exp = Poly(5)
    truth, data = _poly_data(5)
    mse_subset_selection(exp, data, truth)
    assert calls["n"] == 0


def test_an_experiment_without_a_compiled_path_still_works(monkeypatch) -> None:
    """A model that needs a solve falls back to the estimator, and still answers."""
    import discopt.doe._estimation as est

    class NoCompile(est.DevianceFunction):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.path = "fallback"  # pretend there is no compiled path

    monkeypatch.setattr("discopt.doe._estimation.DevianceFunction", NoCompile)

    exp = Poly(4)
    truth, data = _poly_data(4)
    ranking = estimability_rank(exp, truth).ranking
    fallback = _subset_deviances(
        exp, data, dict(truth), ranking, design_values=None, n_starts=1, extra={}
    )
    expected = _estimator_deviances(exp, data, dict(truth), ranking)
    np.testing.assert_allclose(fallback, expected, rtol=1e-8)


@pytest.mark.slow
def test_it_stays_quick_as_p_grows() -> None:
    """p = 7 took over seven seconds through the estimator."""
    import time

    exp = Poly(7)
    truth, data = _poly_data(7)
    t0 = time.perf_counter()
    mse_subset_selection(exp, data, truth)
    assert time.perf_counter() - t0 < 4.0
