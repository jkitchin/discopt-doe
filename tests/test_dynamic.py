"""Tests for discopt.doe.dynamic: ODE experiments through the whole MBDoE stack.

Every check is against a closed-form result: the analytic sensitivity FIM of
first-order and consecutive reactions, the known D-optimal sampling time
t* = 1/k for exponential decay, the exact C0**2 scaling of information with a
designed initial concentration, and a parameter pair that enters only as a
product.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt.doe import (
    batch_optimal_experiment,
    compute_fim,
    diagnose_identifiability,
    discriminate_design,
    estimability_rank,
    optimal_experiment,
    profile_likelihood,
)
from discopt.doe.dynamic import ODEExperiment, ode_experiment

SIGMA = 0.01


def _decay(**kw):
    base = dict(
        states={"A": 1.0},
        parameters={"k": (0.5, 1e-6, 10.0)},
        measured=["A"],
        sample_times=["t1"],
        design_inputs={"t1": (0.05, 10.0)},
        measurement_error=SIGMA,
    )
    base.update(kw)
    return ode_experiment(lambda t, x, p, u: {"A": -p["k"] * x["A"]}, **base)


def _consecutive(times=(0.5, 1.0, 2.0, 4.0, 8.0), n_steps=80, method="rk4"):
    return ode_experiment(
        lambda t, x, p, u: {
            "A": -p["k1"] * x["A"],
            "B": p["k1"] * x["A"] - p["k2"] * x["B"],
            "C": p["k2"] * x["B"],
        },
        states={"A": 1.0, "B": 0.0, "C": 0.0},
        parameters={"k1": 0.8, "k2": 0.3},
        measured=["A", "B"],
        sample_times=list(times),
        measurement_error=SIGMA,
        n_steps=n_steps,
        method=method,
    )


def test_first_order_fim_matches_analytic_sensitivity() -> None:
    """C = C0 exp(-k t): dC/dk = -t C0 exp(-k t), so FIM = (t e^{-kt})^2 / sigma^2."""
    exp = _decay()
    assert isinstance(exp, ODEExperiment)
    for t in (0.5, 2.0, 5.0):
        fim = compute_fim(exp, {"k": 0.5}, {"t1": t}).fim[0, 0]
        assert fim == pytest.approx((t * np.exp(-0.5 * t)) ** 2 / SIGMA**2, rel=1e-6)


def test_d_optimal_sampling_time_is_one_over_k() -> None:
    """The single most informative time to estimate a decay rate is t* = 1/k."""
    for k in (0.5, 2.0):
        design = optimal_experiment(_decay(), {"k": k}, {"t1": (0.05, 10.0)}, n_starts=4, seed=0)
        assert design.design["t1"] == pytest.approx(1.0 / k, rel=1e-4)


def test_consecutive_reaction_fim_matches_analytic_solution() -> None:
    """A -> B -> C has a closed form for A(t) and B(t); the FIMs must agree."""
    times = (0.5, 1.0, 2.0, 4.0, 8.0)
    k1, k2 = 0.8, 0.3
    fim = compute_fim(_consecutive(times), {"k1": k1, "k2": k2}).fim

    def analytic(theta):
        a, b = theta
        t = jnp.asarray(times)
        A = jnp.exp(-a * t)
        B = a / (b - a) * (jnp.exp(-a * t) - jnp.exp(-b * t))
        return jnp.stack([A, B], axis=1).reshape(-1)

    J = np.asarray(jax.jacfwd(analytic)(jnp.array([k1, k2])))
    np.testing.assert_allclose(fim, J.T @ J / SIGMA**2, rtol=1e-6)


@pytest.mark.slow
def test_trapezoid_is_accurate_and_stable_on_a_stiff_decay() -> None:
    exp = _consecutive(method="trapezoid", n_steps=400)
    ref = compute_fim(_consecutive(), {"k1": 0.8, "k2": 0.3}).fim
    np.testing.assert_allclose(compute_fim(exp, {"k1": 0.8, "k2": 0.3}).fim, ref, rtol=1e-3)
    # k h = 25 per step: far outside RK4's stability region. The trapezoidal
    # rule is A-stable, so it stays bounded (each step multiplies by
    # (1 - kh/2) / (1 + kh/2) = -0.85); it is not L-stable, so it damps slowly.
    stiff = dict(parameters={"k": (500.0, 1.0, 1e4)}, sample_times=[1.0], design_inputs={})
    rk4 = _decay(n_steps=20, **stiff).predict({"k": 500.0})["A@1"]
    trap = _decay(n_steps=20, method="trapezoid", **stiff).predict({"k": 500.0})["A@1"]
    assert not abs(rk4) < 1.0  # RK4 blows up
    assert abs(trap) == pytest.approx(abs((1 - 12.5) / (1 + 12.5)) ** 20, rel=1e-6)


def test_designed_initial_concentration_scales_information_as_c0_squared() -> None:
    exp = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": "C0"},
        parameters={"k": 0.5},
        measured=["A"],
        sample_times=[2.0],
        design_inputs={"C0": (0.1, 5.0)},
        measurement_error=SIGMA,
    )
    f1 = compute_fim(exp, {"k": 0.5}, {"C0": 1.0}).fim[0, 0]
    f3 = compute_fim(exp, {"k": 0.5}, {"C0": 3.0}).fim[0, 0]
    assert f3 == pytest.approx(9.0 * f1, rel=1e-9)


def test_product_only_parameters_are_flagged_unidentifiable() -> None:
    """dA/dt = -(a b) A: only the product a*b is estimable, from any design."""
    exp = ode_experiment(
        lambda t, x, p, u: {"A": -p["a"] * p["b"] * x["A"]},
        states={"A": 1.0},
        parameters={"a": 1.0, "b": 0.5},
        measured=["A"],
        sample_times=[0.5, 1.0, 2.0, 4.0],
        measurement_error=SIGMA,
    )
    diag = diagnose_identifiability(exp, {"a": 1.0, "b": 0.5})
    assert not diag.is_identifiable and diag.fim_rank == 1
    direction = diag.null_space[0]
    # Null direction: d(ab) = b da + a db = 0, i.e. (da, db) proportional to (a, -b).
    assert direction["a"] / direction["b"] == pytest.approx(-1.0 / 0.5 * 0.5, rel=1e-6)
    est = estimability_rank(exp, {"a": 1.0, "b": 0.5})
    assert len(est.recommended_subset) == 1


@pytest.mark.slow
def test_estimate_recovers_parameters_with_honest_standard_errors() -> None:
    exp = _consecutive()
    truth = {"k1": 0.8, "k2": 0.3}
    clean = exp.predict(truth)
    rng = np.random.default_rng(0)
    z = []
    for _ in range(40):
        data = {rn: v + rng.normal(0.0, SIGMA) for rn, v in clean.items()}
        fit = exp.estimate(data, initial_guess={"k1": 0.5, "k2": 0.5})
        se = fit.standard_errors
        z.append([(fit.parameters[n] - truth[n]) / se[n] for n in truth])
    z = np.array(z)
    # Standardized errors: mean ~0 and SD ~1 when the covariance is honest.
    assert np.all(np.abs(z.mean(axis=0)) < 0.5)
    assert np.all((0.6 < z.std(axis=0)) & (z.std(axis=0) < 1.5))
    # The fit's FIM equals compute_fim at the estimate.
    np.testing.assert_allclose(fit.fim, compute_fim(exp, fit.parameters).fim, rtol=1e-6)


def test_estimate_requires_design_values_for_designed_experiments() -> None:
    exp = _decay()
    with pytest.raises(ValueError, match="design"):
        exp.estimate({"A@t1": 0.4})
    fit = exp.estimate({"A@t1": [0.37, 0.36, 0.38]}, design={"t1": 2.0})
    assert fit.parameters["k"] == pytest.approx(-np.log(0.37) / 2.0, rel=1e-3)


def test_validation_errors() -> None:
    rhs = lambda t, x, p, u: {"A": -p["k"] * x["A"]}  # noqa: E731
    base = dict(states={"A": 1.0}, parameters={"k": 0.5}, measured=["A"])
    with pytest.raises(ValueError, match="design input"):
        ode_experiment(rhs, sample_times=["t9"], **base)
    with pytest.raises(ValueError, match="after t0"):
        ode_experiment(rhs, sample_times=[0.0], **base)
    with pytest.raises(ValueError, match="method"):
        ode_experiment(rhs, sample_times=[1.0], method="euler", **base)
    with pytest.raises(ValueError, match="unknown state"):
        ode_experiment(
            rhs, sample_times=[1.0], states={"A": 1.0}, parameters={"k": 0.5}, measured={"y": "B"}
        )


@pytest.mark.slow
def test_arrhenius_temperature_batch_spreads_over_two_temperatures() -> None:
    R = 8.314
    exp = ode_experiment(
        lambda t, x, p, u: {
            "A": -p["kr"] * jnp.exp(-p["E"] * 1e4 / R * (1.0 / u["T"] - 1.0 / 350.0)) * x["A"]
        },
        states={"A": 1.0},
        parameters={"kr": (0.5, 1e-6, 50.0), "E": (6.0, 0.1, 30.0)},
        measured=["A"],
        sample_times=[1.0, 3.0],
        design_inputs={"T": (300.0, 400.0)},
        measurement_error=SIGMA,
    )
    theta = {"kr": 0.5, "E": 6.0}
    batch = batch_optimal_experiment(exp, theta, {"T": (300.0, 400.0)}, n_experiments=4, seed=0)
    temps = sorted(d["T"] for d in batch.designs)
    assert len({round(t) for t in temps}) == 2  # two support temperatures
    single = sum(compute_fim(exp, theta, {"T": temps[0]}).fim for _ in range(4))
    assert np.linalg.slogdet(batch.joint_fim)[1] > np.linalg.slogdet(single)[1]


@pytest.mark.slow
def test_profile_likelihood_on_an_ode_model() -> None:
    exp = _consecutive()
    rng = np.random.default_rng(1)
    data = {rn: v + rng.normal(0.0, SIGMA) for rn, v in exp.predict({"k1": 0.8, "k2": 0.3}).items()}
    prof = profile_likelihood(exp, data, "k2")
    assert prof.shape == "bounded"
    assert prof.ci_lower < 0.3 < prof.ci_upper


def test_deviance_uses_the_experiments_measurement_error() -> None:
    """A dynamic experiment spells its errors ``sigma``, not ``measurement_error``.

    Reading the missing attribute defaulted to sigma = 1, which scaled the
    deviance by 1/SIGMA**2 = 10,000 and made every profile built from it look
    flat. The deviance must match the estimator's own objective.
    """
    from discopt.doe._estimation import DevianceFunction

    exp = _consecutive()
    theta = {"k1": 0.8, "k2": 0.3}
    rng = np.random.default_rng(3)
    data = {rn: v + rng.normal(0.0, SIGMA) for rn, v in exp.predict(theta).items()}
    fit = exp.estimate(data)
    dev = DevianceFunction(exp, data, fit.parameters)
    assert dev.path == "predict"
    theta_hat = [float(fit.parameters[n]) for n in dev.names]
    assert dev(np.asarray(theta_hat)) == pytest.approx(float(fit.objective), rel=1e-6)
    # ... and the analytic gradient agrees with finite differences.
    grad = dev.gradient(np.asarray(theta_hat))
    assert grad is not None
    for i in range(len(theta_hat)):
        step = 1e-6 * abs(theta_hat[i])
        up, down = list(theta_hat), list(theta_hat)
        up[i] += step
        down[i] -= step
        fd = (dev(np.asarray(up)) - dev(np.asarray(down))) / (2 * step)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


@pytest.mark.slow
def test_profile_of_a_combination_matches_the_parameter_profile() -> None:
    """``expression="k2"`` profiles the same thing as ``parameter_name="k2"``."""
    exp = _consecutive()
    rng = np.random.default_rng(1)
    data = {rn: v + rng.normal(0.0, SIGMA) for rn, v in exp.predict({"k1": 0.8, "k2": 0.3}).items()}
    direct = profile_likelihood(exp, data, "k2")
    combo = profile_likelihood(exp, data, expression="k2")
    assert combo.shape == "bounded"
    assert combo.ci_lower == pytest.approx(direct.ci_lower, rel=1e-3)
    assert combo.ci_upper == pytest.approx(direct.ci_upper, rel=1e-3)

    ratio = profile_likelihood(exp, data, expression="k2/k1")
    assert ratio.shape == "bounded"
    assert ratio.ci_lower < 0.3 / 0.8 < ratio.ci_upper


@pytest.mark.slow
def test_discrimination_between_first_and_second_order_odes() -> None:
    kw = dict(
        states={"A": 1.0},
        measured=["A"],
        sample_times=["t1"],
        design_inputs={"t1": (0.05, 10.0)},
        measurement_error=SIGMA,
    )
    first = ode_experiment(lambda t, x, p, u: {"A": -p["k"] * x["A"]}, parameters={"k": 0.5}, **kw)
    second = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"] ** 2}, parameters={"k": 0.7}, **kw
    )
    res = discriminate_design(
        {"first": first, "second": second},
        {"first": {"k": 0.5}, "second": {"k": 0.7}},
        {"t1": (0.05, 10.0)},
        n_starts=4,
        seed=0,
    )
    t = res.design["t1"]
    gap = abs(
        first.predict({"k": 0.5}, {"t1": t})["A@t1"] - second.predict({"k": 0.7}, {"t1": t})["A@t1"]
    )
    early = abs(
        first.predict({"k": 0.5}, {"t1": 0.05})["A@t1"]
        - second.predict({"k": 0.7}, {"t1": 0.05})["A@t1"]
    )
    assert gap > 5 * early  # it samples where the rival models disagree
