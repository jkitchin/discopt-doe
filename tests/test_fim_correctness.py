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


class TestOtherJacobianPathsWithStates:
    """Every module that differentiates responses must see the implicit states."""

    def test_discrimination_prediction_uses_total_sensitivity(self):
        from discopt.doe.discrimination import _predict_with_covariance

        pred = _predict_with_covariance(ImplicitStateExperiment(), THETA, {"x": 2.0}, None)
        J, F = _exact_implicit(2.0)
        np.testing.assert_allclose(pred.fim_result.jacobian, J, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(pred.fim_result.fim, F, rtol=1e-6)

    def test_parametric_surrogate_refuses_implicit_states(self):
        from discopt.doe.model_based import ParametricSurrogate

        with pytest.raises(ValueError, match="explicit"):
            ParametricSurrogate(
                ImplicitStateExperiment(),
                input_names=["x"],
                response_name="y1",
                initial_guess=THETA,
            )


def _biexp_experiment():
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

    return BiExp()


BIEXP_THETA = {"a1": 1.0, "k1": 2.0, "a2": 0.5, "k2": 0.1}
BIEXP_BOUNDS = {f"t{i}": (0.05, 30.0) for i in range(4)}


class TestConstrainedSeeding:
    def test_no_feasible_random_candidate_still_finds_optimum(self):
        # No random point in [0.05, 30]^4 has sum <= 5; seeding from a single
        # infeasible point used to return a singular design (log det -39.7).
        def budget(d):
            return 5.0 - sum(d[f"t{i}"] for i in range(4))

        res = optimal_experiment(
            _biexp_experiment(), BIEXP_THETA, BIEXP_BOUNDS, inequality_constraints=[budget]
        )
        assert budget(res.design) >= -1e-6
        assert res.criterion_value == pytest.approx(30.4335, abs=1e-3)


class TestJointBatch:
    def test_joint_is_never_worse_than_greedy(self):
        from discopt.doe import BatchStrategy, batch_optimal_experiment

        class MM(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model("mm")
                V = m.continuous("V", lb=0, ub=10)
                K = m.continuous("K", lb=0, ub=10)
                S = m.continuous("S", lb=0.01, ub=5)
                return ExperimentModel(
                    m, {"V": V, "K": K}, {"S": S}, {"y": V * S / (K + S)}, {"y": 0.05}
                )

        args = (MM(), {"V": 2.0, "K": 0.3}, {"S": (0.01, 5.0)}, 4)
        greedy = batch_optimal_experiment(*args, strategy=BatchStrategy.GREEDY)
        joint = batch_optimal_experiment(*args, strategy=BatchStrategy.JOINT)
        # optimum: two runs at K*Smax/(2K+Smax) = 0.2679 and two at Smax = 5
        assert joint.criterion_value >= greedy.criterion_value - 1e-9
        assert joint.criterion_value == pytest.approx(14.0413, abs=1e-3)


class TestInitialDesigns:
    def test_initial_design_is_used_and_validated(self):
        ex = _biexp_experiment()
        good = {"t0": 0.361, "t1": 3.349, "t2": 0.05, "t3": 1.24}
        budget = [lambda d: 5.0 - sum(d.values())]
        res = optimal_experiment(
            ex,
            BIEXP_THETA,
            BIEXP_BOUNDS,
            inequality_constraints=budget,
            initial_designs=[good],
            n_starts=1,
            local_refine=False,
        )
        assert res.criterion_value == pytest.approx(30.4335, abs=1e-2)
        with pytest.raises(ValueError, match="missing design inputs"):
            optimal_experiment(ex, BIEXP_THETA, BIEXP_BOUNDS, initial_designs=[{"t0": 1.0}])


class TestODEBreakpoints:
    def test_piecewise_input_keeps_rk4_accuracy(self):
        import jax
        import jax.numpy as jnp
        from discopt.doe import ode_experiment

        # dx/dt = -k(T) x with T switching at t = 0.3 and 0.7; exact solution
        # is a product of exponentials.
        edges = jnp.array([0.3, 0.7])

        def rhs(t, x, p, u):
            T = jnp.stack([u["T0"], u["T1"], u["T2"]])[jnp.searchsorted(edges, t, side="left")]
            return {"x": -p["k"] * jnp.exp(-p["E"] / T) * x["x"]}

        design = {"T0": 1.0, "T1": 3.0, "T2": 0.5}
        theta = {"k": 2.0, "E": 1.0}

        def exact(th):
            k = lambda T: th[0] * jnp.exp(-th[1] / T)  # noqa: E731
            return jnp.exp(-(0.3 * k(1.0) + 0.4 * k(3.0) + 0.3 * k(0.5)))

        J_exact = np.asarray(jax.jacfwd(exact)(jnp.array([2.0, 1.0])))
        kw = dict(
            states={"x": 1.0},
            parameters=theta,
            measured=["x"],
            sample_times=[1.0],
            design_inputs={k: (0.1, 5.0) for k in design},
            measurement_error=0.01,
            n_steps=20,
        )
        plain = compute_fim(ode_experiment(rhs, **kw), theta, design).jacobian.ravel()
        split = compute_fim(
            ode_experiment(rhs, breakpoints=[0.3, 0.7], **kw), theta, design
        ).jacobian.ravel()
        assert np.max(np.abs(split - J_exact)) < 1e-7
        assert np.max(np.abs(plain - J_exact)) > 100 * np.max(np.abs(split - J_exact))

    def test_breakpoints_validated(self):
        from discopt.doe import ode_experiment

        with pytest.raises(ValueError, match="after t0"):
            ode_experiment(
                lambda t, x, p, u: {"x": -x["x"]},
                states={"x": 1.0},
                parameters={"k": 1.0},
                measured=["x"],
                sample_times=[1.0],
                breakpoints=[0.0],
            )


class TestVectorParameter:
    def _experiment(self):
        class Quadratic(Experiment):
            # One vector-valued unknown parameter k = (k0, k1, k2).
            def create_model(self, **kwargs):
                m = dm.Model("vec")
                k = m.continuous("k", shape=(3,), lb=-10, ub=10)
                x = m.continuous("x", lb=0, ub=2)
                ys = {
                    f"y{j}": k[0] + k[1] * (x * c) + k[2] * (x * c) ** 2
                    for j, c in enumerate([0.5, 1.0, 1.5])
                }
                return ExperimentModel(m, {"k": k}, {"x": x}, ys, {n: 0.1 for n in ys})

        return Quadratic()

    def test_one_name_per_fim_row(self):
        r = compute_fim(self._experiment(), {"k": np.array([1.0, -0.5, 0.3])}, {"x": 1.3})
        assert r.fim.shape == (3, 3)
        assert r.parameter_names == ["k[0]", "k[1]", "k[2]"]

    def test_identifiability_diagnostics_run(self):
        from discopt.doe import diagnose_identifiability

        # used to fail with "negative dimensions are not allowed"
        diagnose_identifiability(self._experiment(), {"k": np.array([1.0, -0.5, 0.3])}, {"x": 1.3})


class TestEigenvalueCriteria:
    @pytest.mark.parametrize(
        ("criterion", "optimum"),
        [(DesignCriterion.E_OPTIMAL, 119.667), (DesignCriterion.ME_OPTIMAL, 335.309)],
    )
    def test_biexp_reaches_brute_force_optimum(self, criterion, optimum):
        # Random candidates are nearly singular and E/ME are nonsmooth, so the
        # refiner never moved from them (E = 1e-7); the D-optimal seed fixes it.
        res = optimal_experiment(
            _biexp_experiment(), BIEXP_THETA, BIEXP_BOUNDS, criterion=criterion
        )
        assert res.criterion_value == pytest.approx(optimum, rel=1e-3)


class TestODERound3:
    @staticmethod
    def _abc(theta, b0=0.0, **kw):
        from discopt.doe import ode_experiment

        return ode_experiment(
            lambda t, x, p, u: {"A": -p["k1"] * x["A"], "B": p["k1"] * x["A"] - p["k2"] * x["B"]},
            states={"A": "CA0", "B": b0},
            parameters=theta,
            measured=["A", "B"],
            sample_times=[0.05, 0.2, 1.0, 3.0],
            design_inputs={"CA0": (0.5, 2.0)},
            measurement_error=0.01,
            **kw,
        )

    def test_blown_up_integration_is_flagged(self):
        # RK4 with h = 0.06 on k1 = 500 overflows; the FIM is inf. It used to be
        # returned silently, and check_accuracy compared nans and passed.
        theta = {"k1": 500.0, "k2": 1.0}
        ex = self._abc(theta, n_steps=50)
        with pytest.warns(UserWarning, match="non-finite"):
            compute_fim(ex, theta, {"CA0": 1.0})
        with pytest.warns(UserWarning, match="not converged"):
            assert ex.check_accuracy(theta, {"CA0": 1.0}) == np.inf

    def test_unknown_initial_condition_as_parameter(self):
        import jax
        import jax.numpy as jnp

        theta = {"k1": 2.0, "k2": 0.5, "B0": 0.3}
        times = [0.05, 0.2, 1.0, 3.0]

        def exact(v):
            k1, k2, b0 = v
            ca = [jnp.exp(-k1 * t) for t in times]
            cb = [
                b0 * jnp.exp(-k2 * t) + k1 / (k2 - k1) * (jnp.exp(-k1 * t) - jnp.exp(-k2 * t))
                for t in times
            ]
            return jnp.stack(ca + cb)

        J = np.asarray(jax.jacfwd(exact)(jnp.array([2.0, 0.5, 0.3])))
        r = compute_fim(self._abc(theta, b0="B0"), theta, {"CA0": 1.0})
        assert r.parameter_names == ["k1", "k2", "B0"]
        # responses are ordered A@t..., B@t... per time; compare as FIMs
        np.testing.assert_allclose(r.fim, J.T @ J / 0.01**2, rtol=1e-6)
