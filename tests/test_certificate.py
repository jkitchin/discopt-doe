"""Approximate targeted designs and certified efficiency bounds."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    Target,
    approximate_targeted_design,
    batch_optimal_experiment,
    certify_design,
    compute_fim,
    optimal_experiment,
)
from discopt.estimate import Experiment, ExperimentModel

SIGMA = 0.1


class Quadratic(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("quad")
        b0 = m.continuous("b0", lb=-10, ub=10)
        b1 = m.continuous("b1", lb=-10, ub=10)
        b2 = m.continuous("b2", lb=-10, ub=10)
        x = m.continuous("x", lb=-1, ub=1)
        return ExperimentModel(
            m,
            {"b0": b0, "b1": b1, "b2": b2},
            {"x": x},
            {"y": b0 + b1 * x + b2 * x * x},
            {"y": SIGMA},
        )


class Decay(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("decay")
        A = m.continuous("A", lb=0.01, ub=100)
        k = m.continuous("k", lb=1e-3, ub=10)
        t = m.continuous("t", lb=0.0, ub=20)
        return ExperimentModel(
            m, {"A": A, "k": k}, {"t": t}, {"y": A * dm.exp(-k * t)}, {"y": 0.05}
        )


Q = {"b0": 1.0, "b1": 1.0, "b2": 1.0}
QB = {"x": (-1.0, 1.0)}
DT = {"A": 2.0, "k": 0.5}
DB = {"t": (0.05, 10.0)}


def _points(approx):
    return sorted(round(p["x"], 3) for p, _ in approx.support(1e-2))


def test_slope_is_c_optimal_at_the_ends():
    a = approximate_targeted_design(Quadratic(), Q, Target.parameters(["b1"]), design_bounds=QB)
    assert _points(a) == [-1.0, 1.0]
    np.testing.assert_allclose([w for _, w in a.support(1e-2)], [0.5, 0.5], atol=1e-3)
    assert a.value == pytest.approx(SIGMA**2, rel=1e-4)  # Var(b1) = sigma^2 / n, n = 1
    assert a.gap <= 1e-6 * a.value


def test_full_quadratic_is_the_d_optimal_three_point_design():
    a = approximate_targeted_design(
        Quadratic(),
        Q,
        Target.parameters(["b0", "b1", "b2"]),
        design_bounds=QB,
        criterion="determinant",
    )
    assert _points(a) == [-1.0, 0.0, 1.0]
    np.testing.assert_allclose([w for _, w in a.support(1e-2)], [1 / 3] * 3, atol=2e-3)
    # Equivalence theorem: max sensitivity = number of targets at the optimum.
    assert a.sensitivity.max() == pytest.approx(3.0, rel=1e-3)


def test_curvature_target_d_s():
    # D_s for b2: weights 1/4, 1/2, 1/4 at -1, 0, 1 (Atkinson et al. 2007, ch. 10).
    a = approximate_targeted_design(
        Quadratic(), Q, Target.parameters(["b2"]), design_bounds=QB, criterion="determinant"
    )
    assert _points(a) == [-1.0, 0.0, 1.0]
    np.testing.assert_allclose([w for _, w in a.support(1e-2)], [0.25, 0.5, 0.25], atol=2e-3)


@pytest.mark.parametrize("criterion", ["trace", "determinant"])
def test_certificates_bound_the_true_efficiency(criterion):
    target = Target.parameters(["b2"])
    good = batch_optimal_experiment(
        Quadratic(), Q, QB, 4, criterion=criterion, target=target, strategy="joint", n_starts=6
    ).designs
    bad = [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}, {"x": 0.9}]
    cg = certify_design(Quadratic(), Q, good, target, criterion=criterion, design_bounds=QB)
    cb = certify_design(Quadratic(), Q, bad, target, criterion=criterion, design_bounds=QB)
    assert 0.9 <= cg.efficiency_lower_bound <= 1.0 + 1e-9
    assert cb.efficiency_lower_bound < 0.5
    # The bound never exceeds the efficiency relative to the approximate optimum.
    for c in (cg, cb):
        a = c.approximate
        if criterion == "trace":
            true = a.value / c.value
        else:
            true = np.exp(-(c.value - a.value) / a.target.n_targets)
        assert c.efficiency_lower_bound <= true + 1e-9
    assert "certified efficiency" in cg.summary()


def test_with_a_prior_and_a_rounded_design():
    prior = compute_fim(Decay(), DT, {"t": 0.3}).fim
    target = Target.parameters(["k"])
    a = approximate_targeted_design(
        Decay(), DT, target, design_bounds=DB, n_runs=1, prior_fim=prior
    )
    best = optimal_experiment(Decay(), DT, DB, criterion="trace", target=target, prior_fim=prior)
    # One run on top of the prior: the approximate optimum is a single point,
    # the same one the exact search finds.
    (point, weight), *_ = sorted(a.support(1e-2), key=lambda t: -t[1])
    assert point["t"] == pytest.approx(best.design["t"], rel=0.02)
    cert = certify_design(Decay(), DT, [best.design], target, design_bounds=DB, prior_fim=prior)
    assert cert.efficiency_lower_bound > 0.99
    assert len(a.rounded(3)) == 3


def test_refusals():
    with pytest.raises(ValueError, match="certificates are available"):
        approximate_targeted_design(
            Quadratic(),
            Q,
            Target.parameters(["b1"]),
            design_bounds=QB,
            criterion="condition_number",
        )
    with pytest.raises(ValueError, match="candidates or design_bounds"):
        approximate_targeted_design(Quadratic(), Q, Target.parameters(["b1"]))


def test_a_target_estimable_on_a_singular_prior_is_found():
    """One run at x = 0.5 already estimates y(0.5) while the FIM is rank 1; the
    next run helps only at exactly x = 0.5, which a search on the exact
    (discontinuous) criterion cannot find."""
    prior = compute_fim(Quadratic(), Q, {"x": 0.5}).fim
    target = Target.responses(at=[{"x": 0.5}])
    res = optimal_experiment(
        Quadratic(),
        Q,
        QB,
        criterion="trace",
        target=target,
        prior_fim=prior,
        n_starts=4,
        initial_designs=[{"x": 0.5}],  # the run behind the prior
    )
    assert res.design["x"] == pytest.approx(0.5, abs=1e-4)
    assert res.criterion_value == pytest.approx(SIGMA**2 / 2, rel=1e-3)
    # A batch offers its own earlier picks as candidates: three replicates.
    batch = batch_optimal_experiment(
        Quadratic(), Q, QB, 3, criterion="trace", target=target, n_starts=4
    )
    assert batch.criterion_value == pytest.approx(SIGMA**2 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# Minimax criteria (certified through their dual)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["max_variance", "min_eigenvalue"])
def test_minimax_bound_is_below_every_exact_design(criterion):
    import itertools

    target = Target.parameters({"b0": 1.0, "b1": 4.0, "b2": 1.0})
    grid = np.linspace(-1.0, 1.0, 21)
    cands = [{"x": float(x)} for x in grid]
    a = approximate_targeted_design(
        Quadratic(), Q, target, candidates=cands, criterion=criterion, n_runs=3
    )
    assert a.gap <= 1e-3 * a.value
    infos = {x: compute_fim(Quadratic(), Q, {"x": float(x)}).fim for x in grid}
    exact = min(
        a.phi(sum(infos[x] for x in runs))
        for runs in itertools.combinations_with_replacement(grid, 3)
    )
    # Approximate optimum <= every exact design; the certificate is below it.
    assert a.lower_bound <= a.value <= exact * (1 + 1e-9)
    assert a.dual is not None and a.dual.shape == (3, 3)
    assert np.trace(a.dual) == pytest.approx(1.0)


def test_minimax_analytic_optimum():
    """max(Var b1, Var b2): the curvature binds; the optimum is the D_s(b2) design
    (1/4, 1/2, 1/4), where Var b2 = 4 sigma^2 / n and Var b1 = 2 sigma^2 / n."""
    a = approximate_targeted_design(
        Quadratic(),
        Q,
        Target.parameters(["b1", "b2"]),
        candidates=[{"x": float(x)} for x in np.linspace(-1, 1, 21)],
        criterion="max_variance",
        n_runs=3,
    )
    assert a.value == pytest.approx(4 * SIGMA**2 / 3, rel=1e-4)
    assert np.diag(a.dual) == pytest.approx([0.0, 1.0], abs=1e-3)  # only b2 binds
    assert _points(a) == [-1.0, 0.0, 1.0]


def test_certify_a_minimax_design():
    target = Target.parameters({"b1": 1.0, "b2": 1.0})
    cands = [{"x": float(x)} for x in np.linspace(-1, 1, 41)]
    good = [{"x": -1.0}, {"x": 0.0}, {"x": 0.0}, {"x": 1.0}]
    bad = [{"x": 0.2}, {"x": 0.4}, {"x": 0.6}, {"x": 0.8}]
    for crit in ("max_variance", "min_eigenvalue"):
        cg = certify_design(Quadratic(), Q, good, target, criterion=crit, candidates=cands)
        cb = certify_design(Quadratic(), Q, bad, target, criterion=crit, candidates=cands)
        assert cg.efficiency_lower_bound > 0.99
        assert cb.efficiency_lower_bound < 0.3
