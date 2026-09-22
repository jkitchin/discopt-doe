"""Tests for the compiled-Jacobian FIM cache and the public ODE sensitivity API.

The cache must be invisible except for speed: every cached FIM equals the one
the uncached autodiff path computes, a batch equals its per-point FIMs, and an
experiment edited after a first call is recompiled rather than served a stale
function. The ODE experiment's public ``jacobian``/``fim`` must agree with
``compute_fim``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from discopt.doe import compute_fim, diagnose_identifiability, explore_design_space
from discopt.doe.dynamic import ode_experiment
from discopt.doe.fim import (
    _compile_response,
    _compute_jacobian_autodiff,
    _get_param_indices,
    clear_fim_cache,
    compute_fim_batch,
)

SIGMA = 0.02
TREF = 350.0


def _reactor(n_steps: int = 30):
    def rhs(t, x, p, u):
        k1 = p["k1"] * jnp.exp(-p["E1"] * (1000.0 / u["T"] - 1000.0 / TREF))
        k2 = p["k2"] * jnp.exp(-p["E2"] * (1000.0 / u["T"] - 1000.0 / TREF))
        return {"A": -k1 * x["A"], "B": k1 * x["A"] - k2 * x["B"], "C": k2 * x["B"]}

    return ode_experiment(
        rhs,
        states={"A": 2.0, "B": 0.0, "C": 0.0},
        parameters={
            "k1": (0.5, 1e-4, 50.0),
            "E1": (8.0, 0.1, 40.0),
            "k2": (0.2, 1e-4, 50.0),
            "E2": (12.0, 0.1, 40.0),
        },
        measured=["A", "B"],
        sample_times=[0.5, 1.5, 4.0, 10.0],
        design_inputs={"T": (320.0, 380.0)},
        measurement_error=SIGMA,
        n_steps=n_steps,
    )


TRUTH = {"k1": 0.5, "E1": 8.0, "k2": 0.2, "E2": 12.0}


def _uncached_fim(experiment, theta, design):
    """The FIM by the original, un-jitted autodiff path, for comparison."""
    from discopt.doe.fim import _assemble_x_flat_direct, _measurement_sigma
    from discopt.parametric import flatten_params

    em = experiment.create_model(**theta)
    x = _assemble_x_flat_direct(em, theta, design)
    fns = [_compile_response(em.responses[n], em.model) for n in em.response_names]
    J = np.asarray(
        _compute_jacobian_autodiff(fns, x, flatten_params(em.model), _get_param_indices(em))
    )
    sigma = _measurement_sigma(em)
    return J.T @ np.diag(1.0 / sigma**2) @ J


def test_cached_fim_equals_uncached_autodiff() -> None:
    exp = _reactor()
    for T in (325.0, 350.0, 377.0):
        cached = compute_fim(exp, TRUTH, {"T": T}).fim
        np.testing.assert_allclose(cached, _uncached_fim(exp, TRUTH, {"T": T}), rtol=1e-10)


def test_repeat_calls_reuse_the_compiled_jacobian() -> None:
    exp = _reactor()
    compute_fim(exp, TRUTH, {"T": 330.0})  # builds and compiles
    t0 = time.perf_counter()
    for T in np.linspace(321.0, 379.0, 20):
        compute_fim(exp, TRUTH, {"T": float(T)})
    per_call = (time.perf_counter() - t0) / 20
    assert per_call < 0.05  # un-jitted: ~1 s per call for this model


def test_batch_equals_per_point() -> None:
    exp = _reactor()
    points = [{"T": 322.0}, {"T": 351.0}, {"T": 379.0}]
    batch = compute_fim_batch(exp, TRUTH, points)
    for res, dp in zip(batch, points):
        np.testing.assert_allclose(res.fim, compute_fim(exp, TRUTH, dp).fim, rtol=1e-10)


def test_editing_the_experiment_recompiles() -> None:
    """A stale compiled function after an edit would silently give the old FIM."""
    exp = _reactor(n_steps=3)
    coarse = compute_fim(exp, TRUTH, {"T": 360.0}).fim
    exp.n_steps = 60
    edited = compute_fim(exp, TRUTH, {"T": 360.0}).fim
    fresh = compute_fim(_reactor(n_steps=60), TRUTH, {"T": 360.0}).fim
    assert not np.allclose(coarse, edited, rtol=1e-6)
    np.testing.assert_allclose(edited, fresh, rtol=1e-10)


def test_new_nominal_values_get_their_own_entry() -> None:
    exp = _reactor()
    a = compute_fim(exp, TRUTH, {"T": 340.0}).fim
    other = dict(TRUTH, k1=0.9)
    b = compute_fim(exp, other, {"T": 340.0}).fim
    np.testing.assert_allclose(b, _uncached_fim(exp, other, {"T": 340.0}), rtol=1e-10)
    assert not np.allclose(a, b)
    clear_fim_cache(exp)
    np.testing.assert_allclose(compute_fim(exp, TRUTH, {"T": 340.0}).fim, a, rtol=1e-12)


def test_unknown_design_name_still_rejected() -> None:
    exp = _reactor()
    compute_fim(exp, TRUTH, {"T": 340.0})
    with pytest.raises(ValueError, match="unknown design input"):
        compute_fim(exp, TRUTH, {"Temp": 340.0})


# ---------------------------------------------------------------------------
# Public ODE API
# ---------------------------------------------------------------------------


def test_ode_jacobian_and_fim_match_compute_fim() -> None:
    exp = _reactor()
    design = {"T": 345.0}
    res = compute_fim(exp, TRUTH, design)
    np.testing.assert_allclose(exp.jacobian(TRUTH, design), res.jacobian, rtol=1e-10)
    np.testing.assert_allclose(exp.fim(TRUTH, design), res.fim, rtol=1e-10)
    prior = np.eye(4)
    np.testing.assert_allclose(exp.fim(TRUTH, design, prior_fim=prior), res.fim + prior)
    y = exp.predict(TRUTH, design)
    assert list(y) == exp.response_names


def _timed_decay():
    return ode_experiment(
        lambda t, x, p, u: {
            "A": -p["k"] * jnp.exp(-p["E"] * (1.0 / u["T"] - 1.0 / 350.0)) * x["A"]
        },
        states={"A": 1.0},
        parameters={"k": 0.5, "E": 1000.0},
        measured=["A"],
        sample_times=["t1"],
        design_inputs={"t1": (0.1, 10.0), "T": (300.0, 400.0)},
        measurement_error=0.01,
    )


def test_simulate_needs_only_the_inputs_it_uses() -> None:
    exp = _timed_decay()
    traj = exp.simulate({"k": 0.5, "E": 1000.0}, {"T": 350.0}, times=[0.0, 1.0, 2.0])
    np.testing.assert_allclose(traj["A"], np.exp(-0.5 * np.array([0.0, 1.0, 2.0])), rtol=1e-6)
    with pytest.raises(ValueError, match="'T'"):
        exp.simulate({"k": 0.5, "E": 1000.0}, {}, times=[1.0])
    with pytest.raises(ValueError, match="t1"):
        exp.simulate({"k": 0.5, "E": 1000.0}, {"T": 350.0})
    with pytest.raises(ValueError, match="unknown design"):
        exp.simulate({"k": 0.5, "E": 1000.0}, {"T": 350.0, "Tx": 1.0}, times=[1.0])


def test_check_accuracy_flags_too_few_steps() -> None:
    fast = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": 1.0},
        parameters={"k": 8.0},
        measured=["A"],
        sample_times=[0.2, 0.5],
        measurement_error=0.01,
        n_steps=3,
    )
    with pytest.warns(UserWarning, match="n_steps=3 is not converged"):
        change = fast.check_accuracy({"k": 8.0})
    assert change > 1e-3
    fine = ode_experiment(
        lambda t, x, p, u: {"A": -p["k"] * x["A"]},
        states={"A": 1.0},
        parameters={"k": 8.0},
        measured=["A"],
        sample_times=[0.2, 0.5],
        measurement_error=0.01,
        n_steps=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fine.check_accuracy({"k": 8.0}) < 1e-3


# ---------------------------------------------------------------------------
# Named correlations and exploration failures
# ---------------------------------------------------------------------------


def test_identifiability_correlations_carry_names() -> None:
    exp = _reactor()
    diag = diagnose_identifiability(exp, TRUTH, {"T": 350.0})
    assert diag.parameter_names == ["k1", "E1", "k2", "E2"]
    frame = diag.correlation_frame()
    assert set(frame) == {"k1", "E1", "k2", "E2"}
    C = np.asarray(diag.correlation_matrix)
    assert diag.correlation("k2", "E1") == pytest.approx(C[2, 1], nan_ok=True)
    assert frame["k1"]["E2"] == pytest.approx(C[0, 3], nan_ok=True)
    with pytest.raises(KeyError):
        diag.correlation("k1", "nope")


def test_explore_design_space_reports_failed_points(monkeypatch) -> None:
    import discopt.doe.exploration as exploration

    exp = _reactor()
    real = exploration.compute_fim

    def flaky(experiment, param_values, design, prior_fim=None):
        if design["T"] > 360.0:  # 367.5 and 380 of 330..380
            raise RuntimeError("integrator blew up")
        return real(experiment, param_values, design, prior_fim=prior_fim)

    monkeypatch.setattr(exploration, "compute_fim", flaky)
    with pytest.warns(UserWarning, match="2 of 5 grid points"):
        res = explore_design_space(exp, TRUTH, {"T": np.linspace(330.0, 380.0, 5)})
    assert res.n_failed == 2
    assert res.failures[0]["error"].startswith("RuntimeError")
    assert np.isnan(res.metrics["log_det_fim"][-1])


def test_explore_design_space_raises_when_nothing_works() -> None:
    with pytest.raises(ValueError, match="any of the 3 grid points"):
        explore_design_space(_reactor(), TRUTH, {"Temp": np.array([330.0, 340.0, 350.0])})
