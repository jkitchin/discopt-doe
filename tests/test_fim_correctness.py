"""Regressions found by cross-checking discopt.doe against pyomo.doe.

See benchmarks/pyomo_doe_comparison/REPORT.md for the full comparison.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import DesignCriterion, compute_fim, optimal_experiment
from discopt.doe.linear_design import trace_inverse
from discopt.estimate import Experiment, ExperimentModel
from scipy.optimize import brentq

THETA = {"k": 0.8, "a": 1.5}
SIGMA = 0.05


def _exact_implicit(x):
    """z solves z + k z^3 = x;  y1 = a z,  y2 = z + k."""
    k, a = THETA["k"], THETA["a"]
    z = brentq(lambda z: z + k * z**3 - x, -10, 10)
    dz = -(z**3) / (1 + 3 * k * z**2)
    J = np.array([[a * dz, z], [dz + 1.0, 0.0]])
    return J, J.T @ J / SIGMA**2


class ImplicitStateExperiment(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("implicit")
        k = m.continuous("k", lb=0, ub=10)
        a = m.continuous("a", lb=0, ub=10)
        x = m.continuous("x", lb=0.1, ub=5)
        z = m.continuous("z", lb=-10, ub=10)
        m.subject_to(z + k * z**3 == x)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k, "a": a},
            design_inputs={"x": x},
            responses={"y1": a * z, "y2": z + k},
            measurement_error={"y1": SIGMA, "y2": SIGMA},
        )


class PureStateExperiment(Experiment):
    """Responses are the states themselves: the old code returned J == 0."""

    def create_model(self, **kwargs):
        m = dm.Model("pure_state")
        k = m.continuous("k", lb=0, ub=10)
        a = m.continuous("a", lb=0, ub=10)
        x = m.continuous("x", lb=0.1, ub=5)
        z = m.continuous("z", lb=-10, ub=10)
        w = m.continuous("w", lb=-100, ub=100)
        m.subject_to(z + k * z**3 == x)
        m.subject_to(w == a * z)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k, "a": a},
            design_inputs={"x": x},
            responses={"z": z, "w": w},
            measurement_error={"z": SIGMA, "w": SIGMA},
        )


class TestImplicitStateSensitivity:
    @pytest.mark.parametrize("x", [0.5, 2.0, 4.0])
    def test_mixed_response_matches_implicit_function_theorem(self, x):
        r = compute_fim(ImplicitStateExperiment(), THETA, {"x": x})
        J, F = _exact_implicit(x)
        np.testing.assert_allclose(r.jacobian, J, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(r.fim, F, rtol=1e-6)

    def test_pure_state_responses_are_not_zero(self):
        k, a, x = THETA["k"], THETA["a"], 2.0
        z = brentq(lambda z: z + k * z**3 - x, -10, 10)
        dz = -(z**3) / (1 + 3 * k * z**2)
        r = compute_fim(PureStateExperiment(), THETA, {"x": x})
        np.testing.assert_allclose(r.jacobian, [[dz, 0.0], [a * dz, z]], rtol=1e-6, atol=1e-7)

    def test_finite_difference_method_agrees(self):
        ad = compute_fim(ImplicitStateExperiment(), THETA, {"x": 2.0})
        fd = compute_fim(ImplicitStateExperiment(), THETA, {"x": 2.0}, method="finite_difference")
        np.testing.assert_allclose(fd.fim, ad.fim, rtol=1e-5)

    def test_optimal_design_uses_total_sensitivity(self):
        res = optimal_experiment(ImplicitStateExperiment(), THETA, {"x": (0.1, 5.0)}, n_starts=8)
        xs = np.linspace(0.1, 5.0, 491)
        best = xs[int(np.argmax([np.linalg.slogdet(_exact_implicit(x)[1])[1] for x in xs]))]
        assert abs(res.design["x"] - best) < 0.02

    def test_undetermined_state_raises(self):
        class Underdetermined(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model("under")
                k = m.continuous("k", lb=0, ub=10)
                x = m.continuous("x", lb=0.1, ub=5)
                z = m.continuous("z", lb=-10, ub=10)
                u = m.continuous("u", lb=-10, ub=10)
                m.subject_to(z + u == k * x)  # one equation, two states
                return ExperimentModel(m, {"k": k}, {"x": x}, {"y": z}, {"y": 0.1})

        with pytest.raises(ValueError, match="determine only"):
            compute_fim(Underdetermined(), {"k": 1.0}, {"x": 1.0})


class TestNominalOutsideBounds:
    def test_explicit_model_refuses_instead_of_clipping(self):
        class RB(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model("rb")
                A = m.continuous("A", lb=0, ub=100)
                k = m.continuous("k", lb=0, ub=0.2)
                t = m.continuous("t", lb=0, ub=10)
                return ExperimentModel(
                    m, {"A": A, "k": k}, {"t": t}, {"y": A * (1 - dm.exp(-k * t))}, {"y": 0.1}
                )

        with pytest.raises(ValueError, match="outside its variable bounds"):
            compute_fim(RB(), {"A": 15.0, "k": 0.5}, {"t": 3.0})
        # On the bound is fine.
        compute_fim(RB(), {"A": 15.0, "k": 0.2}, {"t": 3.0})

    def test_constrained_model_refuses_too(self):
        with pytest.raises(ValueError, match="outside its variable bounds"):
            compute_fim(ImplicitStateExperiment(), {"k": 20.0, "a": 1.5}, {"x": 2.0})


class TestAOptimalitySingular:
    def test_numerically_singular_fim_scores_inf(self):
        # Rank-2 4x4 FIM: np.linalg.inv does not raise, and its trace can be
        # a huge negative number that a minimizer would prefer.
        rng = np.random.default_rng(0)
        J = rng.normal(size=(2, 4)) * np.array([1.0, 1e-3, 10.0, 0.1])
        F = J.T @ J
        assert trace_inverse(F) > 1e12  # inf, or huge -- never negative
        assert trace_inverse(F - 1e-3 * np.eye(4)) == np.inf  # indefinite
        assert trace_inverse(np.eye(3) * 2.0) == pytest.approx(1.5)

    def test_a_optimal_design_is_not_singular(self):
        class BiExp(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model("biexp")
                p = {n: m.continuous(n, lb=0, ub=10) for n in ("a1", "k1", "a2", "k2")}
                d = {f"t{i}": m.continuous(f"t{i}", lb=0.05, ub=30) for i in range(4)}
                ys = {
                    f"y{i}": p["a1"] * dm.exp(-p["k1"] * d[f"t{i}"])
                    + p["a2"] * dm.exp(-p["k2"] * d[f"t{i}"])
                    for i in range(4)
                }
                return ExperimentModel(m, p, d, ys, {n: 0.01 for n in ys})

        theta = {"a1": 1.0, "k1": 2.0, "a2": 0.5, "k2": 0.1}
        bounds = {f"t{i}": (0.05, 30.0) for i in range(4)}
        res = optimal_experiment(BiExp(), theta, bounds, criterion=DesignCriterion.A_OPTIMAL)
        assert np.isfinite(res.criterion_value) and res.criterion_value > 0
        assert res.criterion_value < 0.0095  # brute-force optimum is 0.009066


class TestMultimodalDesign:
    def test_refines_several_starts_to_find_global_optimum(self):
        # y = amp*sin(w x): the D-criterion has a local maximum near every
        # peak of the sensitivity; refining only the best scan point misses
        # the global one (x ~ 9.44, log det 13.698; the next best is 13.463).
        class Osc(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model("osc")
                amp = m.continuous("amp", lb=0, ub=10)
                w = m.continuous("w", lb=0, ub=10)
                x = m.continuous("x", lb=0, ub=10)
                return ExperimentModel(
                    m, {"amp": amp, "w": w}, {"x": x}, {"y": amp * dm.sin(w * x)}, {"y": 0.1}
                )

        res = optimal_experiment(
            Osc(), {"amp": 1.0, "w": 3.0}, {"x": (0.0, 10.0)}, prior_fim=np.diag([100.0, 1.0])
        )
        assert res.criterion_value == pytest.approx(13.6984, abs=1e-3)


class TestSingularOptimumWarning:
    def _experiment(self):
        class Product(Experiment):
            # Only the product a*b is identifiable: the FIM is singular everywhere.
            def create_model(self, **kwargs):
                m = dm.Model("product")
                a = m.continuous("a", lb=0, ub=10)
                b = m.continuous("b", lb=0, ub=10)
                x = m.continuous("x", lb=0, ub=2)
                ys = {"y1": a * b * x, "y2": a * b * x**2}
                return ExperimentModel(m, {"a": a, "b": b}, {"x": x}, ys, {"y1": 0.1, "y2": 0.1})

        return Product()

    def test_warns_when_optimal_fim_is_singular(self):
        with pytest.warns(UserWarning, match="numerically singular"):
            optimal_experiment(self._experiment(), {"a": 1.0, "b": 2.0}, {"x": (0.1, 2.0)})

    def test_prior_restores_identifiability_and_silences_it(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            optimal_experiment(
                self._experiment(),
                {"a": 1.0, "b": 2.0},
                {"x": (0.1, 2.0)},
                prior_fim=np.diag([1.0, 1.0]),
            )
