"""Discrimination follow-ups: prior-aware compound designs, Box-Hill, the
boundary-corrected likelihood-ratio test, and the compiled-once predictor."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from types import SimpleNamespace

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    DiscriminationCriterion,
    discriminate_compound,
    discriminate_design,
    evaluate_discrimination_criterion,
    likelihood_ratio_test,
)
from discopt.estimate import Experiment, ExperimentModel

SIGMA = 0.5


class LineExp(Experiment):
    """``y = a + b x``."""

    def create_model(self, **kw):
        m = dm.Model("line")
        a = m.continuous("a", lb=-10.0, ub=10.0)
        b = m.continuous("b", lb=-10.0, ub=10.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        return ExperimentModel(m, {"a": a, "b": b}, {"x": x}, {"y": a + b * x}, {"y": SIGMA})


class SquareExp(Experiment):
    """``y = a + b x^2``."""

    def create_model(self, **kw):
        m = dm.Model("square")
        a = m.continuous("a", lb=-10.0, ub=10.0)
        b = m.continuous("b", lb=-10.0, ub=10.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        return ExperimentModel(m, {"a": a, "b": b}, {"x": x}, {"y": a + b * x * x}, {"y": SIGMA})


EXPS = {"line": LineExp(), "square": SquareExp()}
PE = {"line": {"a": 1.0, "b": 1.0}, "square": {"a": 1.0, "b": 0.5}}
BOUNDS = {"x": (0.0, 2.0)}
X_DONE = [0.0, 0.5, 1.0]  # runs already made, near the low end


def _fim(rows):
    F = np.array(rows, dtype=float)
    return F.T @ F / SIGMA**2


PRIOR = {
    "line": _fim([[1.0, x] for x in X_DONE]),
    "square": _fim([[1.0, x * x] for x in X_DONE]),
}


# ─────────────────────────────────────────────────────────────────────
# prior_fims in the compound design
# ─────────────────────────────────────────────────────────────────────


def test_compound_at_zero_weight_is_d_optimal_given_the_data():
    """lambda = 0 must be the D-optimal next run *given the data already
    collected*: without the prior a single run cannot identify two parameters
    at all, so the old compound had nothing meaningful to optimise."""
    res = discriminate_compound(
        EXPS,
        PE,
        BOUNDS,
        discrimination_weight=0.0,
        precision_model="line",
        prior_fims=PRIOR,
        seed=0,
    )
    grid = np.linspace(0.0, 2.0, 2001)
    logdet = [np.linalg.slogdet(PRIOR["line"] + _fim([[1.0, x]]))[1] for x in grid]
    x_best = grid[int(np.argmax(logdet))]
    assert res.design["x"] == pytest.approx(x_best, abs=1e-3)
    assert res.criterion_value == pytest.approx(max(logdet), rel=1e-6)


def test_compound_at_unit_weight_reproduces_the_bf_design():
    comp = discriminate_compound(
        EXPS,
        PE,
        BOUNDS,
        discrimination_weight=1.0,
        precision_model="line",
        prior_fims=PRIOR,
        seed=3,
    )
    bf = discriminate_design(
        EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BF, prior_fims=PRIOR, seed=3
    )
    assert comp.design["x"] == pytest.approx(bf.design["x"], abs=1e-9)
    assert comp.criterion_value == pytest.approx(bf.criterion_value, rel=1e-9)


def test_evaluate_criterion_accepts_prior_fims_and_matches_the_design():
    bf = discriminate_design(
        EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BF, prior_fims=PRIOR, seed=1
    )
    value = evaluate_discrimination_criterion(
        EXPS, PE, bf.design, criterion=DiscriminationCriterion.BF, prior_fims=PRIOR
    )
    assert value == pytest.approx(bf.criterion_value, rel=1e-9)
    # ... and the prior matters: without it the value differs.
    value_no_prior = evaluate_discrimination_criterion(
        EXPS, PE, bf.design, criterion=DiscriminationCriterion.BF
    )
    assert abs(value - value_no_prior) > 1e-3


def test_prior_fims_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        discriminate_design(EXPS, PE, BOUNDS, prior_fims={"line": np.eye(3)})


# ─────────────────────────────────────────────────────────────────────
# Box-Hill
# ─────────────────────────────────────────────────────────────────────


def _box_hill_scalar(x, prior):
    """Box & Hill (1967) for one response, written out directly."""
    f_line = np.array([1.0, x])
    f_sq = np.array([1.0, x * x])
    y_line = PE["line"]["a"] + PE["line"]["b"] * x
    y_sq = PE["square"]["a"] + PE["square"]["b"] * x * x
    v_line = f_line @ np.linalg.inv(prior["line"]) @ f_line
    v_sq = f_sq @ np.linalg.inv(prior["square"]) @ f_sq
    s1, s2 = SIGMA**2 + v_line, SIGMA**2 + v_sq
    pi1 = pi2 = 0.5
    return (
        0.5
        * pi1
        * pi2
        * ((v_line - v_sq) ** 2 / (s1 * s2) + (y_line - y_sq) ** 2 * (1 / s1 + 1 / s2))
    )


@pytest.mark.parametrize("x", [0.3, 1.2, 1.9])
def test_box_hill_matches_the_original_formula(x):
    value = evaluate_discrimination_criterion(
        EXPS, PE, {"x": x}, criterion=DiscriminationCriterion.BH, prior_fims=PRIOR
    )
    assert value == pytest.approx(_box_hill_scalar(x, PRIOR), rel=1e-9)


def test_box_hill_design_maximises_the_criterion():
    res = discriminate_design(
        EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BH, prior_fims=PRIOR, seed=0
    )
    grid = np.linspace(0.0, 2.0, 801)
    best = max(_box_hill_scalar(x, PRIOR) for x in grid)
    assert res.criterion_value == pytest.approx(best, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────
# The compiled-once predictor agrees with the reference path
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("use_prior", [False, True])
def test_predictor_matches_reference_path(use_prior):
    from discopt.doe.discrimination import _Predictor, _predict_with_covariance

    priors = PRIOR if use_prior else None
    predict = _Predictor(EXPS, PE, priors)
    for x in (0.0, 0.7, 2.0):
        fast = predict({"x": x})
        for name in EXPS:
            ref = _predict_with_covariance(
                EXPS[name], PE[name], {"x": x}, None if priors is None else priors[name]
            )
            np.testing.assert_allclose(fast[name].y_hat, ref.y_hat, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(fast[name].V, ref.V, rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(
                fast[name].fim_result.fim, ref.fim_result.fim, rtol=1e-12, atol=1e-14
            )


def test_predictor_falls_back_when_x_needs_a_solve(monkeypatch):
    """Models whose x* cannot be assembled directly use the reference path."""
    import discopt.doe.fim as fimmod
    from discopt.doe.discrimination import _Predictor

    calls = {"n": 0}
    real = fimmod._assemble_x_flat_direct

    def always_none(*a, **k):
        calls["n"] += 1
        return None

    predict = _Predictor(EXPS, PE)
    monkeypatch.setattr(fimmod, "_assemble_x_flat_direct", always_none)
    preds = predict({"x": 1.0})
    monkeypatch.setattr(fimmod, "_assemble_x_flat_direct", real)
    assert calls["n"] >= 2
    assert preds["line"].y_hat[0] == pytest.approx(2.0, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────
# Likelihood-ratio test on the boundary
# ─────────────────────────────────────────────────────────────────────


def _fit(objective, names, n):
    return SimpleNamespace(
        objective=float(objective), parameter_names=list(names), n_observations=n
    )


def _boundary_lrt_sizes(n_rep=6000, seed=0):
    """y = c + d x + e, d >= 0 with true d = 0 (on the boundary)."""
    rng = np.random.default_rng(seed)
    n = 20
    x = np.linspace(-1.0, 1.0, n)  # centred: slope estimate independent of the intercept
    sxx = float(x @ x)
    rej = {0: 0, 1: 0}
    zeros = 0
    for _ in range(n_rep):
        y = 1.0 + rng.normal(0.0, 1.0, n)
        c = y.mean()
        d_ols = float(x @ (y - c)) / sxx
        d_hat = max(d_ols, 0.0)  # constrained fit
        rss_nested = float(np.sum((y - c) ** 2))
        rss_full = float(np.sum((y - c - d_hat * x) ** 2))
        zeros += d_hat == 0.0
        nested, full = _fit(rss_nested, ["c"], n), _fit(rss_full, ["c", "d"], n)
        for b in (0, 1):
            rej[b] += likelihood_ratio_test(nested, full, boundary=b).p_value < 0.05
    return {b: rej[b] / n_rep for b in rej}, zeros / n_rep


def test_boundary_correction_restores_the_test_size():
    sizes, frac_zero = _boundary_lrt_sizes()
    assert frac_zero == pytest.approx(0.5, abs=0.03)  # half the fits sit on the boundary
    assert sizes[1] == pytest.approx(0.05, abs=0.01)  # chi-bar-squared: nominal
    assert sizes[0] == pytest.approx(0.025, abs=0.008)  # naive chi2_1: half the size


def test_boundary_argument_is_checked():
    nested, full = _fit(10.0, ["c"], 20), _fit(8.0, ["c", "d"], 20)
    with pytest.raises(ValueError, match="boundary"):
        likelihood_ratio_test(nested, full, boundary=2)
    # boundary = df with G2 = 0 gives p = 1 (the point mass is at the observed value).
    same = _fit(10.0, ["c", "d"], 20)
    assert likelihood_ratio_test(nested, same, boundary=1).p_value == pytest.approx(1.0)
