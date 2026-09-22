"""Tests for the model-based follow-ups: combination profiles, quiet solving,
multi-start estimation, estimability extensions (raw norms, MSE subset size),
and robust (pseudo-Bayesian / max-min) designs."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import logging
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe._estimation import DevianceFunction, estimate_parameters
from discopt.doe._logging import quiet_solver
from discopt.doe.estimability import estimability_rank, mse_subset_selection
from discopt.doe.profile import profile_likelihood
from discopt.doe.robust import design_efficiencies, robust_optimal_experiment
from discopt.estimate import Experiment, ExperimentModel

XS = [1.0, 2.0, 3.0, 4.0, 5.0]


class ProductExperiment(Experiment):
    """y_i = a*b*x_i: only the product a*b is identifiable."""

    def create_model(self, **kwargs):
        m = dm.Model("ab")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.1, ub=10)
        responses = {f"y{i}": a * b * x for i, x in enumerate(XS)}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, {k: 0.1 for k in responses})


PRESSURES = [0.002, 0.005, 0.01, 0.02]


class LowPressureLH(Experiment):
    """r = k K P / (1 + K P) at low pressure: nearly only k*K is identifiable."""

    def create_model(self, **kwargs):
        m = dm.Model("lh")
        k = m.continuous("k", lb=0.01, ub=1000)
        K = m.continuous("K", lb=0.01, ub=1000)
        responses = {f"r{i}": k * K * p / (1 + K * p) for i, p in enumerate(PRESSURES)}
        return ExperimentModel(m, {"k": k, "K": K}, {}, responses, {n: 0.002 for n in responses})


TIMES = np.linspace(0.2, 6.0, 15)


class TwoExponentials(Experiment):
    """a exp(-b t) + c exp(-d t): the two terms can swap roles (mirror optima)."""

    def create_model(self, **kwargs):
        m = dm.Model("te")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.05, ub=5)
        c = m.continuous("c", lb=0.1, ub=10)
        d = m.continuous("d", lb=0.05, ub=5)
        responses = {f"y{i}": a * dm.exp(-b * t) + c * dm.exp(-d * t) for i, t in enumerate(TIMES)}
        return ExperimentModel(
            m, {"a": a, "b": b, "c": c, "d": d}, {}, responses, {n: 0.01 for n in responses}
        )


QX = np.linspace(0.0, 1.0, 12)


class Quadratic(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("q")
        a = m.continuous("a", lb=-50, ub=50)
        b = m.continuous("b", lb=-50, ub=50)
        c = m.continuous("c", lb=-50, ub=50)
        responses = {f"y{i}": a + b * x + c * x * x for i, x in enumerate(QX)}
        return ExperimentModel(
            m, {"a": a, "b": b, "c": c}, {}, responses, {n: 0.5 for n in responses}
        )


class ExpDecay(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("ed")
        k = m.continuous("k", lb=0.01, ub=5)
        t = m.continuous("t", lb=0.1, ub=20)
        return ExperimentModel(m, {"k": k}, {"t": t}, {"y": 5.0 * dm.exp(-k * t)}, {"y": 0.05})


# --------------------------------------------------------------------------
# Deviance and quiet solving
# --------------------------------------------------------------------------


def test_deviance_function_matches_the_estimator_objective():
    rng = np.random.default_rng(0)
    data = {f"y{i}": 6.0 * x + rng.normal(0, 0.1) for i, x in enumerate(XS)}
    fit = estimate_parameters(ProductExperiment(), data)
    dev = DevianceFunction(ProductExperiment(), data, fit.parameters)
    assert dev.path == "compiled"
    assert dev(fit.parameters) == pytest.approx(fit.objective, rel=1e-9)
    from discopt.estimate import estimate_parameters as base

    fixed = {"a": 2.0, "b": 3.5}
    assert dev(fixed) == pytest.approx(
        base(ProductExperiment(), data, fixed_parameters=fixed).objective
    )
    assert dev.gradient(dev.vector(fit.parameters)) is not None


def test_quiet_solver_raises_and_restores_the_logger_level():
    logger = logging.getLogger("discopt")
    before = logger.level
    with quiet_solver():
        assert logger.getEffectiveLevel() >= logging.ERROR
    assert logger.level == before
    with quiet_solver(False):
        assert logger.level == before


# --------------------------------------------------------------------------
# Profile of a combination
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_profile_of_a_product_matches_the_exact_interval():
    """y = a*b*x is linear in the product, so the profile interval for a*b
    must equal the exact (known-sigma) interval for the slope."""
    rng = np.random.default_rng(0)
    data = {f"y{i}": 6.0 * x + rng.normal(0, 0.1) for i, x in enumerate(XS)}
    prof = profile_likelihood(ProductExperiment(), data, expression="a*b")
    x = np.asarray(XS)
    y = np.array([data[f"y{i}"] for i in range(len(XS))])
    slope = (x @ y) / (x @ x)
    half = 1.959964 * 0.1 / np.sqrt(x @ x)
    assert prof.shape == "bounded"
    assert prof.parameter == "a*b"
    assert prof.ci_lower == pytest.approx(slope - half, abs=1e-4)
    assert prof.ci_upper == pytest.approx(slope + half, abs=1e-4)


@pytest.mark.slow
def test_product_is_identifiable_when_its_factors_are_not():
    rng = np.random.default_rng(1)
    data = {
        f"r{i}": 3.0 * p / (1 + 1.5 * p) + rng.normal(0, 0.002) for i, p in enumerate(PRESSURES)
    }
    prod = profile_likelihood(
        LowPressureLH(), data, function=lambda th: th["k"] * th["K"], name="kK"
    )
    single = profile_likelihood(LowPressureLH(), data, "k", max_steps=20)
    assert prod.shape == "bounded" and prod.parameter == "kK"
    assert prod.ci_lower < 3.0 < prod.ci_upper
    assert prod.ci_upper / prod.ci_lower < 2.0
    # k alone is barely determined: its interval is open or spans decades.
    assert single.shape != "bounded" or single.ci_upper / single.ci_lower > 100.0


def test_profile_needs_exactly_one_target():
    with pytest.raises(ValueError, match="exactly one"):
        profile_likelihood(ProductExperiment(), {"y0": 1.0}, "a", expression="a*b")


# --------------------------------------------------------------------------
# Multi-start estimation
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_multistart_finds_and_reports_mirror_optima():
    rng = np.random.default_rng(0)
    data = {
        f"y{i}": 2 * np.exp(-0.3 * t) + np.exp(-2 * t) + rng.normal(0, 0.01)
        for i, t in enumerate(TIMES)
    }
    with pytest.warns(UserWarning, match="equally well"):
        fit = estimate_parameters(
            TwoExponentials(),
            data,
            initial_guess={"a": 1, "b": 0.5, "c": 1, "d": 1.5},
            n_starts=12,
            seed=1,
        )
    assert len(fit.alternative_optima) == 1
    alt = fit.alternative_optima[0]
    # The alternative is the same fit with the two terms swapped.
    assert alt["a"] == pytest.approx(fit.parameters["c"], rel=1e-2)
    assert alt["b"] == pytest.approx(fit.parameters["d"], rel=1e-2)


def test_single_start_behaves_as_before():
    rng = np.random.default_rng(0)
    data = {f"y{i}": 6.0 * x + rng.normal(0, 0.1) for i, x in enumerate(XS)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit = estimate_parameters(ProductExperiment(), data)
    assert fit.alternative_optima == []


# --------------------------------------------------------------------------
# Estimability: raw norms and the MSE criterion
# --------------------------------------------------------------------------


def test_estimability_exposes_raw_norms():
    res = estimability_rank(Quadratic(), {"a": 1.0, "b": 2.0, "c": 1.0})
    assert res.raw_norms.shape == (3,)
    # Projection can only shrink a column: raw >= projected, equal for the first pivot.
    assert np.all(res.raw_norms >= res.projected_norms - 1e-9)
    assert res.raw_norms[0] == pytest.approx(res.projected_norms[0])
    assert res.method == "cutoff" and res.mse is None


@pytest.mark.slow
def test_mse_criterion_fixes_a_right_value_and_estimates_a_wrong_one():
    """Wu et al. (2011): with c's fixed value right, estimating it only adds
    variance, so k = 2; with it wrong, the bias dominates, so k = 3."""
    rng = np.random.default_rng(3)
    counts = {}
    for c_true in (0.0, 4.0):
        ks = []
        for _ in range(12):
            data = {
                f"y{i}": 1 + 2 * x + c_true * x * x + rng.normal(0, 0.5) for i, x in enumerate(QX)
            }
            r = mse_subset_selection(
                Quadratic(), data, {"a": 0.0, "b": 0.0, "c": 0.0}, ranking=["a", "b", "c"]
            )
            ks.append(r.recommended_k)
        counts[c_true] = np.bincount(ks, minlength=4)
    assert counts[0.0][2] > 6  # mostly k = 2
    assert counts[4.0][3] > 6  # mostly k = 3
    # Table invariants on the last fit.
    assert r.corrected_ratios[-1] == 0.0 and np.isnan(r.critical_ratios[-1])
    assert r.recommended_subset == r.ranking[: r.recommended_k]
    assert "recommended" in r.summary()


@pytest.mark.slow
def test_estimability_rank_mse_method():
    rng = np.random.default_rng(5)
    data = {f"y{i}": 1 + 2 * x + 4 * x * x + rng.normal(0, 0.5) for i, x in enumerate(QX)}
    res = estimability_rank(Quadratic(), {"a": 1.0, "b": 2.0, "c": 1.0}, method="mse", data=data)
    assert res.method == "mse" and res.mse is not None
    assert res.recommended_subset == res.mse.recommended_subset
    with pytest.raises(ValueError, match="data"):
        estimability_rank(Quadratic(), {"a": 1.0, "b": 2.0, "c": 1.0}, method="mse")


# --------------------------------------------------------------------------
# Robust designs
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def k_samples():
    rng = np.random.default_rng(0)
    return [{"k": float(k)} for k in np.exp(rng.normal(np.log(0.5), 0.6, 16))]


@pytest.mark.slow
def test_pseudo_bayesian_time_is_one_over_the_mean_rate(k_samples):
    """E[log FIM] = const + 2 log t - 2 E[k] t, so the pseudo-Bayesian sampling
    time is exactly 1 / E[k], and it beats the local design (t = 1/k at the
    median, 2) on average over the samples."""
    res = robust_optimal_experiment(
        ExpDecay(), k_samples, {"t": (0.1, 20.0)}, robust="expected", n_starts=4
    )
    mean_k = np.mean([s["k"] for s in k_samples])
    assert res.designs[0]["t"] == pytest.approx(1.0 / mean_k, rel=1e-3)
    ks = np.array([s["k"] for s in k_samples])
    local_mean = np.mean(np.log((5.0 * 2.0 * np.exp(-ks * 2.0)) ** 2 / 0.05**2))
    assert res.log_dets.mean() >= local_mean - 1e-9


@pytest.mark.slow
def test_maximin_improves_the_worst_case(k_samples):
    res = robust_optimal_experiment(
        ExpDecay(), k_samples, {"t": (0.1, 20.0)}, robust="maximin", n_starts=4
    )
    local = design_efficiencies(
        ExpDecay(), [{"t": 2.0}], k_samples, reference_log_dets=res.reference_log_dets
    )
    assert res.efficiencies is not None
    assert res.efficiencies.min() > local.min() + 0.05
    # Each sample's reference is its local optimum at t = 1/k.
    ks = np.array([s["k"] for s in k_samples])
    analytic = np.log((5.0 / ks * np.exp(-1.0)) ** 2 / 0.05**2)
    assert res.reference_log_dets == pytest.approx(analytic, rel=1e-3)


def test_robust_rejects_unknown_options(k_samples):
    with pytest.raises(ValueError, match="'D'"):
        robust_optimal_experiment(ExpDecay(), k_samples, {"t": (0.1, 20.0)}, criterion="A")
    with pytest.raises(ValueError, match="robust"):
        robust_optimal_experiment(ExpDecay(), k_samples, {"t": (0.1, 20.0)}, robust="bayes")
