"""Targeted (goal-oriented) design: the K-matrix criteria in discopt.doe.targets."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    ResolvedTarget,
    SymbolicModel,
    Target,
    batch_optimal_experiment,
    compute_fim,
    experiment_region,
    optimal_experiment,
    sequential_doe,
    target_precision,
)
from discopt.estimate import Experiment, ExperimentModel

THETA = {"A": 2.0, "k": 0.5}
BOUNDS = {"t": (0.05, 10.0)}


class Decay(Experiment):
    """y = A exp(-k t), sampled at a designed time t."""

    def create_model(self, **kwargs):
        m = dm.Model("decay")
        A = m.continuous("A", lb=0.01, ub=100)
        k = m.continuous("k", lb=0.001, ub=10)
        t = m.continuous("t", lb=0.0, ub=20)
        return ExperimentModel(
            model=m,
            unknown_parameters={"A": A, "k": k},
            design_inputs={"t": t},
            responses={"y": A * dm.exp(-k * t)},
            measurement_error={"y": 0.05},
        )


class TwoChannel(Experiment):
    """Three parameters, two responses: one run cannot identify all three.

    y1 = a x, y2 = (b + c) x -- only a and b + c are identified by any design.
    """

    def create_model(self, **kwargs):
        m = dm.Model("two_channel")
        a = m.continuous("a", lb=-10, ub=10)
        b = m.continuous("b", lb=-10, ub=10)
        c = m.continuous("c", lb=-10, ub=10)
        x = m.continuous("x", lb=0.0, ub=10)
        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b, "c": c},
            design_inputs={"x": x},
            responses={"y1": a * x, "y2": (b + c) * x},
            measurement_error={"y1": 0.1, "y2": 0.1},
        )


TWO = {"a": 1.0, "b": 2.0, "c": -0.5}


def _fim(design, prior=None):
    return compute_fim(Decay(), THETA, design, prior_fim=prior).fim


def _grid_best(target, criterion, prior, n=400):
    """Brute-force optimum over a log grid of t, for checking the search."""
    ts = np.geomspace(*BOUNDS["t"], n)
    vals = [target.criterion(_fim({"t": float(t)}, prior), criterion) for t in ts]
    i = int(np.nanargmax(vals) if criterion == "determinant" else np.nanargmin(vals))
    return ts[i], vals[i]


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def test_subset_information_is_the_schur_complement():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(6, 4))
    M = G.T @ G + 0.1 * np.eye(4)
    S, N = [0, 2], [1, 3]
    t = ResolvedTarget.from_matrix(np.eye(4)[:, S], ["p0", "p1", "p2", "p3"])
    C = t.information(M)
    schur = M[np.ix_(S, S)] - M[np.ix_(S, N)] @ np.linalg.solve(M[np.ix_(N, N)], M[np.ix_(N, S)])
    np.testing.assert_allclose(C, schur, rtol=1e-10)
    # Dropping the nuisance rows instead overstates the information.
    assert np.linalg.det(M[np.ix_(S, S)]) > np.linalg.det(schur)
    np.testing.assert_allclose(
        t.criterion(M, "determinant"), np.log(np.linalg.det(schur)), rtol=1e-10
    )
    np.testing.assert_allclose(
        t.criterion(M, "trace"), np.trace(np.linalg.inv(M)[np.ix_(S, S)]), rtol=1e-10
    )


def test_criteria_match_their_definitions():
    rng = np.random.default_rng(1)
    G = rng.normal(size=(5, 3))
    M = G.T @ G
    K = rng.normal(size=(3, 2))
    w = np.array([1.0, 4.0])
    t = ResolvedTarget.from_matrix(K, ["a", "b", "c"], weights=w)
    sigma = K.T @ np.linalg.inv(M) @ K
    sw = np.diag(np.sqrt(w)) @ sigma @ np.diag(np.sqrt(w))
    ev = np.linalg.eigvalsh(sw)
    assert t.criterion(M, "trace") == pytest.approx(np.sum(w * np.diag(sigma)))
    assert t.criterion(M, "average_variance") == pytest.approx(np.sum(w * np.diag(sigma)) / 5)
    assert t.criterion(M, "max_variance") == pytest.approx(np.max(w * np.diag(sigma)))
    assert t.criterion(M, "determinant") == pytest.approx(-np.log(np.linalg.det(sw)))
    assert t.criterion(M, "min_eigenvalue") == pytest.approx(1 / ev[-1])
    assert t.criterion(M, "condition_number") == pytest.approx(ev[-1] / ev[0])


def test_singular_fim_still_estimates_an_identified_combination():
    em = TwoChannel()
    M = compute_fim(em, TWO, {"x": 1.0}).fim
    assert np.linalg.matrix_rank(M) == 2
    sum_bc = Target.linear({"b+c": {"b": 1, "c": 1}}).resolve(em, TWO)
    prec = sum_bc.precision(M)
    assert prec.estimable
    # y2 = (b+c) x with x = 1 and sigma 0.1: the standard error of b+c is 0.1.
    assert prec.std["b+c"] == pytest.approx(0.1, rel=1e-8)
    only_b = Target.parameters(["b"]).resolve(em, TWO)
    assert not only_b.precision(M).estimable
    assert only_b.criterion(M, "trace") == np.inf


def test_parameter_scaling_does_not_change_a_targeted_criterion():
    M = _fim({"t": 1.0}, _fim({"t": 4.0}))
    t = Target.parameters({"A": 1.0, "k": 3.0}).resolve(Decay(), THETA)
    s = np.array([THETA["A"], THETA["k"]])
    scaled = ResolvedTarget(
        K=s[:, None] * t.K, weights=t.weights, names=t.names, parameter_names=t.parameter_names
    )
    Ms = s[:, None] * M * s[None, :]
    for crit in ("trace", "determinant", "min_eigenvalue", "max_variance"):
        assert scaled.criterion(Ms, crit) == pytest.approx(t.criterion(M, crit), rel=1e-9)


# ---------------------------------------------------------------------------
# Target specifications
# ---------------------------------------------------------------------------


def test_parameter_weights_relative_and_tolerances():
    em = Decay()
    t = Target.parameters({"A": 1.0, "k": 0.0}).resolve(em, THETA)
    assert t.names == ["A"]  # zero weight: a nuisance parameter
    rel = Target.parameters(["A", "k"], relative=True).resolve(em, THETA)
    np.testing.assert_allclose(rel.weights, [1 / 2.0**2, 1 / 0.5**2])
    tol = Target.parameters(tolerances={"k": 0.1}, relative=True).resolve(em, THETA)
    np.testing.assert_allclose(tol.tolerances, [0.05])  # 10 % of k = 0.5
    np.testing.assert_allclose(tol.weights, [1 / 0.05**2])
    with pytest.raises(KeyError, match="not a parameter"):
        Target.parameters(["nope"]).resolve(em, THETA)
    with pytest.raises(ValueError, match="weights or tolerances"):
        Target.parameters({"A": 1.0}, tolerances={"A": 0.1})


def test_response_target_is_the_prediction_sensitivity():
    em = Decay()
    t = Target.responses(at=[{"t": 3.0}]).resolve(em, THETA)
    J = compute_fim(em, THETA, {"t": 3.0}).jacobian
    np.testing.assert_allclose(t.K[:, 0], np.ravel(J))
    assert t.names == ["y"]
    two = Target.responses(at=[{"t": 1.0}, {"t": 3.0}]).resolve(em, THETA)
    assert two.names == ["y@t=1", "y@t=3"]


def test_function_target_uses_the_jacobian():
    t = Target.function(lambda th: th["A"] / th["k"], ["ratio"]).resolve(Decay(), THETA)
    np.testing.assert_allclose(t.K[:, 0], [1 / 0.5, -2.0 / 0.5**2], rtol=1e-12)


def test_mixing_kinds_needs_tolerances():
    with pytest.raises(ValueError, match="different units"):
        Target.parameters(["k"]) + Target.responses(at=[{"t": 1.0}])
    both = Target.parameters(tolerances={"k": 0.05}) + Target.responses(
        at=[{"t": 1.0}], tolerances={"y": 0.01}
    )
    r = both.resolve(Decay(), THETA)
    assert r.names == ["k", "y"]
    # Same kind needs no tolerances.
    Target.parameters(["k"]) + Target.linear({"A+k": {"A": 1, "k": 1}})


def test_region_target_reproduces_the_i_criterion():
    pts = [{"t": float(t)} for t in np.linspace(0.1, 8, 25)]
    em = Decay()
    M = _fim({"t": 1.0}, _fim({"t": 5.0}))
    region = experiment_region(em, THETA, points=pts)
    t = Target.region(points=pts).resolve(em, THETA)
    assert t.criterion(M, "average_variance") == pytest.approx(region.average_variance(M))
    assert t.criterion(M, "max_variance") == pytest.approx(region.max_variance(M))


def test_information_criteria_need_independent_targets():
    t = Target.responses(at=[{"t": float(t)} for t in (0.5, 1, 2, 4)])
    with pytest.raises(ValueError, match="linearly independent"):
        optimal_experiment(
            Decay(), THETA, BOUNDS, criterion="determinant", target=t, prior_fim=_fim({"t": 1})
        )
    with pytest.raises(ValueError, match="not available with a target"):
        optimal_experiment(Decay(), THETA, BOUNDS, criterion="bogus", target=t)


def test_weights_under_determinant_warn():
    with pytest.warns(UserWarning, match="do not change a 'determinant' design"):
        optimal_experiment(
            Decay(),
            THETA,
            BOUNDS,
            criterion="determinant",
            target=Target.parameters({"A": 1.0, "k": 5.0}),
            prior_fim=_fim({"t": 1.0}),
            n_starts=4,
        )


# ---------------------------------------------------------------------------
# Design searches
# ---------------------------------------------------------------------------

PRIOR = None


def _prior():
    global PRIOR
    if PRIOR is None:
        PRIOR = _fim({"t": 1.0})
    return PRIOR


@pytest.mark.parametrize("which", ["A", "k"])
def test_subset_design_finds_the_global_optimum(which):
    target = Target.parameters([which])
    res = optimal_experiment(
        Decay(), THETA, BOUNDS, criterion="trace", target=target, prior_fim=_prior()
    )
    resolved = target.resolve(Decay(), THETA)
    t_best, v_best = _grid_best(resolved, "trace", _prior())
    assert res.criterion_value <= v_best * (1 + 1e-6)
    assert res.target is not None and res.target.names == [which]
    assert res.target_precision().std[which] == pytest.approx(np.sqrt(res.criterion_value))


def test_targets_move_the_design():
    # After one run at t = 0.3, the next run that best pins down the amplitude
    # is at the earliest time (y ≈ A there), the one for the rate near
    # t ≈ 1/k -- and neither is the D-optimal compromise (brute force: t ≈ 2.29).
    prior = _fim({"t": 0.3})
    designs = {
        w: optimal_experiment(
            Decay(),
            THETA,
            BOUNDS,
            criterion="trace",
            target=Target.parameters([w]),
            prior_fim=prior,
        ).design["t"]
        for w in ("A", "k")
    }
    assert designs["A"] == pytest.approx(0.05, abs=1e-3)
    assert designs["k"] == pytest.approx(2.51, rel=0.02)


def test_prediction_target_finds_the_global_optimum():
    target = Target.responses(at=[{"t": 6.0}])
    res = optimal_experiment(
        Decay(), THETA, BOUNDS, criterion="trace", target=target, prior_fim=_prior()
    )
    _, v_best = _grid_best(target.resolve(Decay(), THETA), "trace", _prior())
    assert res.criterion_value <= v_best * (1 + 1e-6)


def test_scale_parameters_is_ignored_with_a_target():
    kw = dict(criterion="trace", target=Target.parameters(["k"]), prior_fim=_prior())
    a = optimal_experiment(Decay(), THETA, BOUNDS, **kw)
    b = optimal_experiment(Decay(), THETA, BOUNDS, scale_parameters=True, **kw)
    assert a.design["t"] == pytest.approx(b.design["t"], rel=1e-6)


def test_single_run_identifies_a_target_in_a_singular_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no singular-FIM warning: the target is fine
        res = optimal_experiment(
            TwoChannel(),
            TWO,
            {"x": (0.5, 3.0)},
            criterion="trace",
            target=Target.parameters(["a"]),
        )
    assert res.design["x"] == pytest.approx(3.0)
    assert res.target_precision().estimable
    with pytest.raises(RuntimeError, match="estimable"):
        optimal_experiment(
            TwoChannel(), TWO, {"x": (0.5, 3.0)}, criterion="trace", target=Target.parameters(["b"])
        )


def test_batch_design_with_a_target():
    res = batch_optimal_experiment(
        Decay(),
        THETA,
        BOUNDS,
        2,
        criterion="trace",
        target=Target.parameters(["k"]),
    )
    prec = res.target_precision()
    assert prec.estimable and np.isfinite(prec.std["k"])
    assert res.target.names == ["k"]
    assert res.criterion_value == pytest.approx(prec.std["k"] ** 2, rel=1e-8)


@pytest.mark.parametrize("criterion", ["min_eigenvalue", "max_variance"])
def test_other_criteria_run(criterion):
    target = Target.parameters({"A": 1.0, "k": 4.0})
    res = optimal_experiment(
        Decay(), THETA, BOUNDS, criterion=criterion, target=target, prior_fim=_prior(), n_starts=6
    )
    _, v_best = _grid_best(target.resolve(Decay(), THETA), criterion, _prior())
    if criterion == "min_eigenvalue":
        assert res.criterion_value >= v_best * (1 - 1e-4)
    else:
        assert res.criterion_value <= v_best * (1 + 1e-4)


def test_target_precision_function_and_tolerance_check():
    M = _fim({"t": 1.0}, _fim({"t": 4.0}))
    tight = Target.parameters(tolerances={"k": 1e-6})
    loose = Target.parameters(tolerances={"k": 10.0})
    assert target_precision(M, tight, Decay(), THETA).met is False
    p = target_precision(M, loose, Decay(), THETA)
    assert p.met is True and p.ratios["k"] < 1
    assert "within tolerance" in p.summary()
    assert target_precision(M, Target.parameters(["k"]), Decay(), THETA).met is None


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


def _decay_model():
    return SymbolicModel(
        source="a*exp(-b*x)",
        parameter_names=("a", "b"),
        input_names=("x",),
        measurement_error=0.05,
    )


@pytest.mark.parametrize("tol,rounds", [(10.0, 1), (1e-9, 3)])
def test_sequential_stops_when_the_target_is_precise(tol, rounds):
    m = _decay_model()
    truth = {"a": 2.0, "b": 0.7}
    rng = np.random.default_rng(3)

    def run(design):
        return {"y": m.predict(truth, design) + rng.normal(0, 0.05)}

    initial = [dict(r, **run(r)) for r in ({"x": 0.2}, {"x": 2.0})]
    history = sequential_doe(
        m,
        None,
        {"a": 1.5, "b": 1.0},
        {"x": (0.0, 5.0)},
        initial_runs=initial,
        n_rounds=3,
        run_experiment=run,
        target=Target.parameters(tolerances={"b": tol}),
        stop_at_tolerance=True,
    )
    assert len(history) == rounds
    assert all(h.target_precision is not None for h in history)
    assert history[-1].stopped is (tol > 1)
    if tol > 1:
        assert history[-1].data_collected is None


def test_stop_at_tolerance_needs_a_target():
    with pytest.raises(ValueError, match="needs a target"):
        sequential_doe(Decay(), {"y": 1.0}, THETA, BOUNDS, stop_at_tolerance=True)


def test_a_scaled_experiment_gets_the_same_targeted_design():
    from discopt.doe import ParameterScaledExperiment
    from discopt.doe.fim import parameter_scale

    wrapped = ParameterScaledExperiment(Decay(), parameter_scale(Decay(), THETA))
    target = Target.responses(at=[{"t": 6.0}])
    kw = {"criterion": "trace", "target": target, "prior_fim": _prior()}
    a = optimal_experiment(Decay(), THETA, BOUNDS, **kw)
    b = optimal_experiment(wrapped, THETA, BOUNDS, **kw)
    assert a.design["t"] == pytest.approx(b.design["t"], rel=1e-5)
    assert a.criterion_value == pytest.approx(b.criterion_value, rel=1e-6)
