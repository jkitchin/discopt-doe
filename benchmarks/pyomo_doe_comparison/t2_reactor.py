"""R2-4: pyomo.doe's own reactor example (pyomo/contrib/doe/examples).

A -> B -> C with Arrhenius rates and a piecewise-constant temperature profile
(9 control points) plus CA0 as design; CA, CB, CC measured at the 9 control
times. pyomo uses its shipped ReactorExperiment (collocation, nfe=10, ncp=3,
the example's settings). discopt uses ode_experiment (RK4) with the same
profile. The reference is exact: on each interval T is constant, the ODE is
linear, and the state propagates by a matrix exponential.
"""

import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import numpy as np
from common import ipopt, logdet, rel_err

DATA = json.loads((pathlib.Path(__file__).parent / "reactor_result.json").read_text())
R = 8.314  # the example's value
TC = sorted(float(k) for k in DATA["control_points"])  # 0, 0.125, ..., 1
PNAMES = ["A1", "A2", "E1", "E2"]  # pyomo's unknown_parameters order
THETA = {k: DATA[k] for k in PNAMES}
SIG = 1e-2
T_BOUNDS, CA_BOUNDS = tuple(DATA["T_bounds"]), tuple(DATA["CA_bounds"])
DNAMES = [f"T{i}" for i in range(len(TC))] + ["CA0"]


# Best design found in any run (a pyomo run from a random start; Ipopt is not
# reproducible run to run here, so it is pinned as a reference seed).
BEST_SEEN = {
    "T0": 300.0,
    "T1": 490.37,
    "T2": 300.0,
    "T3": 300.0,
    "T4": 302.24,
    "T5": 304.4,
    "T6": 302.64,
    "T7": 300.0,
    "T8": 300.01,
    "CA0": 5.0,
}


def example_design():
    d = {
        f"T{i}": float(DATA["control_points"][k])
        for i, k in enumerate(sorted(DATA["control_points"], key=float))
    }
    d["CA0"] = DATA["CA0"]
    return d


# ---------------------------------------------------------------- reference
def _outputs(theta_vec, design_vec):
    A1, A2, E1, E2 = theta_vec
    Ts, ca0 = design_vec[:-1], design_vec[-1]
    x = jnp.array([ca0, 0.0, 0.0])
    outs = [x]
    for i in range(len(TC) - 1):  # interval (t_i, t_i+1] runs at T_i (pyomo's T_control)
        k1 = A1 * jnp.exp(-E1 * 1000 / (R * Ts[i]))
        k2 = A2 * jnp.exp(-E2 * 1000 / (R * Ts[i]))
        K = jnp.array([[-k1, 0.0, 0.0], [k1, -k2, 0.0], [0.0, k2, 0.0]])
        x = jax.scipy.linalg.expm(K * (TC[i + 1] - TC[i])) @ x
        outs.append(x)
    return jnp.concatenate([jnp.stack(outs)[:, j] for j in range(3)])  # CA@t.., CB@t.., CC@t..


def ref_fim(design, scale=False):
    th = jnp.array([THETA[n] for n in PNAMES])
    J = jax.jacfwd(_outputs)(th, jnp.array([design[n] for n in DNAMES]))
    if scale:
        J = J * th
    return np.asarray(J.T @ J / SIG**2)


@jax.jit
def _ref_logdet(dvec):
    th = jnp.array([THETA[n] for n in PNAMES])
    J = jax.jacfwd(_outputs)(th, dvec)
    return jnp.linalg.slogdet(J.T @ J / SIG**2)[1]


def best_known(n_starts=30, seed=0, seeds=()):
    """Multistart L-BFGS-B on the exact criterion with exact gradients.

    Random starts alone do badly here (30 of them stay below the example's own
    design), so every design either package returned is also used as a start.
    """
    from scipy.optimize import minimize

    g = jax.jit(jax.grad(_ref_logdet))
    bounds = [T_BOUNDS] * len(TC) + [CA_BOUNDS]
    rng = np.random.default_rng(seed)
    best = (None, -np.inf)
    starts = [np.array([rng.uniform(*b) for b in bounds]) for _ in range(n_starts)]
    starts += [np.array([d[n] for n in DNAMES]) for d in seeds]
    for z0 in starts:
        r = minimize(
            lambda z: -float(_ref_logdet(z)),
            z0,
            jac=lambda z: -np.asarray(g(z)),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if -r.fun > best[1]:
            best = (dict(zip(DNAMES, map(float, r.x))), -r.fun)
    return best


# ---------------------------------------------------------------- discopt
def discopt_reactor(n_steps=50, breakpoints=False):
    from discopt.doe import ode_experiment

    edges = jnp.array(TC[1:-1])

    def rhs(t, x, p, u):
        Ts = jnp.stack([u[f"T{i}"] for i in range(len(TC) - 1)])
        T = Ts[jnp.searchsorted(edges, t, side="left")]  # (t_i, t_i+1] -> T_i
        k1 = p["A1"] * jnp.exp(-p["E1"] * 1000 / (R * T))
        k2 = p["A2"] * jnp.exp(-p["E2"] * 1000 / (R * T))
        return {"CA": -k1 * x["CA"], "CB": k1 * x["CA"] - k2 * x["CB"], "CC": k2 * x["CB"]}

    return ode_experiment(
        rhs,
        states={"CA": "CA0", "CB": 0.0, "CC": 0.0},
        parameters={n: (THETA[n], 0.0, 1e4) for n in PNAMES},
        measured=["CA", "CB", "CC"],
        sample_times=TC[1:],  # t=0 rows carry no information
        design_inputs={**{f"T{i}": T_BOUNDS for i in range(len(TC))}, "CA0": CA_BOUNDS},
        measurement_error=SIG,
        n_steps=n_steps,
        **({"breakpoints": TC[1:-1]} if breakpoints else {}),
    )


# ---------------------------------------------------------------- pyomo
def pyomo_reactor(design=None, nfe=10, ncp=3):
    from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment

    data = dict(DATA)
    design = design or example_design()
    data["control_points"] = {t: design[f"T{i}"] for i, t in enumerate(TC)}
    data["CA0"] = design["CA0"]
    return ReactorExperiment(data=data, nfe=nfe, ncp=ncp)


def pyomo_doe(exp, **kw):
    import logging

    from pyomo.contrib.doe import DesignOfExperiments

    return DesignOfExperiments(
        exp, fd_formula="central", step=1e-3, solver=ipopt(), logger_level=logging.ERROR, **kw
    )


def pyomo_design(res):
    vals = dict(zip(res["Experiment Design Names"], map(float, res["Experiment Design"])))
    d = {
        f"T{i}": vals[f"T[{t:g}]"] if f"T[{t:g}]" in vals else vals[f"T[{t}]"]
        for i, t in enumerate(TC)
    }
    d["CA0"] = vals["CA[0]"]
    return d


if __name__ == "__main__":
    from discopt.doe import compute_fim, optimal_experiment

    d0 = example_design()
    F0 = ref_fim(d0)
    print("example design:", d0)
    print(f"reference: logdet={logdet(F0):.6f}  cond={np.linalg.cond(F0):.2e}")
    print("\n== FIM at the example design, rel. error vs exact ==")
    for bp in [False, True]:
        for n in [5, 10, 50] if bp else [50, 200, 1000]:
            t = time.time()
            F = compute_fim(discopt_reactor(n, bp), THETA, d0).fim
            tag = " per segment (breakpoints)" if bp else ""
            print(
                f"  discopt rk4 n_steps={n:4d}{tag}: {rel_err(F, F0):.2e}  "
                f"logdet err={logdet(F) - logdet(F0):+.2e} ({time.time() - t:.1f}s)"
            )
    for nfe, ncp in [(10, 3), (20, 3), (40, 3)]:
        t = time.time()
        F = pyomo_doe(pyomo_reactor(d0, nfe, ncp)).compute_FIM()
        print(
            f"  pyomo collocation nfe={nfe},ncp={ncp}: {rel_err(F, F0):.2e}  logdet err={logdet(F) - logdet(F0):+.2e} ({time.time() - t:.1f}s)"
        )

    print("\n== D-optimal design (10 design variables) ==")
    found = {"example design": d0}
    bounds = {**{f"T{i}": T_BOUNDS for i in range(len(TC))}, "CA0": CA_BOUNDS}
    for label, kw in [
        ("random starts", {}),
        ("+ example design", {"initial_designs": [d0]}),
        ("+ best design seen", {"initial_designs": [BEST_SEEN]}),
    ]:
        t = time.time()
        r = optimal_experiment(discopt_reactor(10, True), THETA, bounds, **kw)
        found[f"discopt {label}"] = r.design
        print(
            f"discopt ({label}): exact logdet={logdet(ref_fim(r.design)):.6f} ({time.time() - t:.0f}s)"
        )
        print("   ", {k: round(v, 2) for k, v in r.design.items()})
    rng = np.random.default_rng(11)
    for j in range(3):
        st = {n: float(rng.uniform(*bounds[n])) for n in DNAMES}
        t = time.time()
        doe = pyomo_doe(pyomo_reactor(st), objective_option="determinant")
        try:
            doe.run_doe()
            d = pyomo_design(doe.results)
            found[f"pyomo random {j}"] = d
            print(
                f"pyomo from random start {j} [{doe.results['Termination Condition']}]: "
                f"exact logdet={logdet(ref_fim(d)):.6f} ({time.time() - t:.0f}s)"
            )
        except Exception as e:
            print(f"pyomo from random start {j}: ERROR {type(e).__name__}: {str(e)[:100]}")
    # pyomo's example settings (scale_nominal_param_value=True), from the example's design
    for scale in [True, False]:
        t = time.time()
        doe = pyomo_doe(
            pyomo_reactor(d0), objective_option="determinant", scale_nominal_param_value=scale
        )
        try:
            doe.run_doe()
            res = doe.results
            d = pyomo_design(res)
            found[f"pyomo scale={scale}"] = d
            print(
                f"pyomo (scale_nominal={scale}) [{res['Termination Condition']}]: exact logdet="
                f"{logdet(ref_fim(d)):.6f} ({time.time() - t:.0f}s)"
            )
            print("   ", {k: round(v, 2) for k, v in d.items()})
        except Exception as e:
            print(f"pyomo (scale_nominal={scale}) ERROR {type(e).__name__}: {str(e)[:100]}")
    t = time.time()
    bk, bv = best_known(seeds=list(found.values()) + [BEST_SEEN])
    print(
        f"best known (30 random + {len(found)} seeded L-BFGS-B starts, exact gradients): "
        f"logdet={bv:.6f} ({time.time() - t:.0f}s)"
    )
    print("   ", {k: round(v, 2) for k, v in bk.items()})
