"""R3-4..7: model Parameters, vector design inputs, a 6-parameter design,
malformed prior FIMs."""

import time
import warnings

warnings.filterwarnings("ignore")
import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
from common import logdet, pyomo_fim, pyomo_run_doe, reference_fim, rel_err, to_discopt, to_pyomo
from discopt.doe import compute_fim, optimal_experiment
from discopt.estimate import Experiment, ExperimentModel
from scipy.optimize import minimize

if __name__ == "__main__":
    print("== R3-4 fixed model Parameter inside the response ==")

    class WithParam(Experiment):
        def create_model(self, **kw):
            m = dm.Model("param")
            c = m.parameter("c", value=2.5)  # a known constant carried as a Parameter
            k = m.continuous("k", lb=0, ub=10)
            a = m.continuous("a", lb=0, ub=10)
            t = m.continuous("t", lb=0, ub=5)
            ys = {"y1": a * dm.exp(-k * t) + c, "y2": a * c * dm.exp(-k * 2 * t)}
            return ExperimentModel(m, {"k": k, "a": a}, {"t": t}, ys, {"y1": 0.05, "y2": 0.05})

    resp = lambda th, d, o: [
        th["a"] * o.exp(-th["k"] * d["t"]) + 2.5,
        th["a"] * 2.5 * o.exp(-th["k"] * 2 * d["t"]),
    ]
    th, dsg = {"k": 0.7, "a": 1.3}, {"t": 1.2}
    ref = reference_fim(resp, th, dsg, 0.05)
    try:
        r = compute_fim(WithParam(), th, dsg)
        print(f"  discopt: {rel_err(r.fim, ref):.1e}")
    except Exception as e:
        print(f"  discopt ERROR {type(e).__name__}: {str(e)[:120]}")
    F, _ = pyomo_fim(to_pyomo(resp, th, dsg, {"t": (0, 5)}, 0.05, n_resp=2))
    print(f"  pyomo:   {rel_err(F, ref):.1e}")

    print("\n== R3-5 vector-valued design input (one Variable of size 3) ==")

    class VecDesign(Experiment):
        def create_model(self, **kw):
            m = dm.Model("vd")
            A = m.continuous("A", lb=0, ub=100)
            k = m.continuous("k", lb=0, ub=10)
            t = m.continuous("t", shape=(3,), lb=0.1, ub=10)
            ys = {f"y{i}": A * (1 - dm.exp(-k * t[i])) for i in range(3)}
            return ExperimentModel(m, {"A": A, "k": k}, {"t": t}, ys, {n: 0.1 for n in ys})

    resp3 = lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d[f"t{i}"])) for i in range(3)]
    th = {"A": 15.0, "k": 0.5}
    tv = np.array([0.5, 2.0, 6.0])
    ref = reference_fim(resp3, th, {f"t{i}": tv[i] for i in range(3)}, 0.1)
    try:
        r = compute_fim(VecDesign(), th, {"t": tv})
        print(f"  discopt compute_fim: {rel_err(r.fim, ref):.1e}")
    except Exception as e:
        print(f"  discopt compute_fim ERROR {type(e).__name__}: {str(e)[:120]}")
    try:
        o = optimal_experiment(VecDesign(), th, {"t": (0.1, 10.0)})
        print(f"  discopt optimal_experiment: {o.design}")
    except Exception as e:
        print(f"  discopt optimal_experiment ERROR {type(e).__name__}: {str(e)[:140]}")
    # the scalar version for the optimum
    b3 = {f"t{i}": (0.1, 10.0) for i in range(3)}
    o = optimal_experiment(to_discopt(resp3, th, b3, 0.1), th, b3)
    print(f"  scalar-inputs optimum: {o.design} logdet={o.criterion_value:.5f}")

    print("\n== R3-6 six parameters, six sampling times (tri-exponential) ==")
    th6 = {"a1": 1.0, "k1": 5.0, "a2": 0.5, "k2": 0.8, "a3": 0.2, "k3": 0.05}
    resp6 = lambda th, d, o: [
        sum(th[f"a{j}"] * o.exp(-th[f"k{j}"] * d[f"t{i}"]) for j in (1, 2, 3)) for i in range(6)
    ]
    b6 = {f"t{i}": (0.01, 60.0) for i in range(6)}
    names = list(b6)

    @jax.jit
    def ld(z):
        from common import JNP_OPS

        def f(v):
            thv = dict(zip(th6, v))
            return jnp.stack(resp6(thv, dict(zip(names, z)), JNP_OPS))

        J = jax.jacfwd(f)(jnp.array(list(th6.values())))
        return jnp.linalg.slogdet(J.T @ J / 1e-4)[1]

    g = jax.jit(jax.grad(ld))
    rng = np.random.default_rng(0)
    best = -np.inf
    t = time.time()
    for _ in range(200):
        z0 = rng.uniform(0.01, 60.0, 6)
        r = minimize(
            lambda z: -float(ld(z)),
            z0,
            jac=lambda z: -np.asarray(g(z)),
            method="L-BFGS-B",
            bounds=[(0.01, 60.0)] * 6,
        )
        best = max(best, -r.fun)
    print(f"  reference (200-start L-BFGS-B, exact gradients): {best:.5f} ({time.time() - t:.0f}s)")
    t = time.time()
    o = optimal_experiment(to_discopt(resp6, th6, b6, 0.01), th6, b6)
    dv = logdet(reference_fim(resp6, th6, o.design, 0.01))
    print(
        f"  discopt: {dv:.5f} ({time.time() - t:.0f}s)  t={sorted(round(v, 3) for v in o.design.values())}"
    )
    rng = np.random.default_rng(5)
    for j in range(3):
        st = {n: float(rng.uniform(0.01, 60.0)) for n in names}
        t = time.time()
        try:
            res, _ = pyomo_run_doe(to_pyomo(resp6, th6, st, b6, 0.01, n_resp=6))
            d = dict(zip(names, map(float, res["Experiment Design"])))
            print(
                f"  pyomo start {j} [{res['Termination Condition']}]: "
                f"{logdet(reference_fim(resp6, th6, d, 0.01)):.5f} ({time.time() - t:.0f}s)"
            )
        except Exception as e:
            print(f"  pyomo start {j}: ERROR {type(e).__name__}: {str(e)[:80]}")

    print("\n== R3-7 malformed prior FIMs ==")
    rb = lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d["t"]))]
    th = {"A": 15.0, "k": 0.5}
    ex = to_discopt(rb, th, {"t": (0, 10)}, 0.1)
    for label, P in [
        ("3x3 for 2 parameters", np.eye(3)),
        ("not symmetric", np.array([[1.0, 5.0], [0.0, 1.0]])),
        ("indefinite", np.array([[1.0, 0.0], [0.0, -5.0]])),
        ("contains nan", np.array([[np.nan, 0], [0, 1.0]])),
    ]:
        try:
            o = optimal_experiment(ex, th, {"t": (0.0, 10.0)}, prior_fim=P)
            dres = f"accepted -> t={o.design['t']:.3f}, crit={o.criterion_value:.4g}"
        except Exception as e:
            dres = f"{type(e).__name__}: {str(e)[:70]}"
        try:
            res, _ = pyomo_run_doe(
                to_pyomo(rb, th, {"t": 5.0}, {"t": (0, 10)}, 0.1, n_resp=1), prior_FIM=P
            )
            pres = f"accepted -> t={float(res['Experiment Design'][0]):.3f}"
        except Exception as e:
            pres = f"{type(e).__name__}: {str(e)[:70]}"
        print(f"  {label:22s} discopt: {dres}\n  {'':22s} pyomo:   {pres}")
