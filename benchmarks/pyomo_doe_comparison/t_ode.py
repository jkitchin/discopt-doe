"""T8: dynamic experiment A -> B -> C (pyomo.doe's reactor kinetics).

k_i = A_i exp(-E_i*1000/(R T)); design: temperature T and initial CA0; measured
CA, CB, CC at five times. The constant-T system has a closed form, which is the
exact reference. pyomo.doe differentiates a pyomo.dae collocation model by
finite differences; discopt differentiates an RK4 integration by autodiff.
"""

import time
import warnings

warnings.filterwarnings("ignore")
from common import (
    logdet,
    pyomo_fim,
    pyomo_run_doe,
    reference_fim,
    rel_err,
)
import jax.numpy as jnp
import numpy as np

R = 8.31446261815324
THETA = {"A1": 84.79, "E1": 7.78, "A2": 371.72, "E2": 15.05}
TIMES = [0.125, 0.25, 0.5, 0.75, 1.0]
SIG = 0.02
BOUNDS = {"T": (300.0, 700.0), "CA0": (1.0, 5.0)}


def analytic(th, d, o):
    k1 = th["A1"] * o.exp(-th["E1"] * 1000 / (R * d["T"]))
    k2 = th["A2"] * o.exp(-th["E2"] * 1000 / (R * d["T"]))
    out = []
    for t in TIMES:
        ca = d["CA0"] * o.exp(-k1 * t)
        cb = d["CA0"] * k1 / (k2 - k1) * (o.exp(-k1 * t) - o.exp(-k2 * t))
        out += [ca, cb, d["CA0"] - ca - cb]
    return out


def discopt_ode(n_steps=50):
    from discopt.doe import ode_experiment

    def rhs(t, x, p, u):
        k1 = p["A1"] * jnp.exp(-p["E1"] * 1000 / (R * u["T"]))
        k2 = p["A2"] * jnp.exp(-p["E2"] * 1000 / (R * u["T"]))
        return {"A": -k1 * x["A"], "B": k1 * x["A"] - k2 * x["B"], "C": k2 * x["B"]}

    return ode_experiment(
        rhs,
        states={"A": "CA0", "B": 0.0, "C": 0.0},
        parameters=THETA,
        measured=["A", "B", "C"],
        sample_times=TIMES,
        design_inputs=BOUNDS,
        measurement_error=SIG,
        n_steps=n_steps,
    )


def pyomo_ode(design, nfe=10, ncp=3, scheme="LAGRANGE-RADAU"):
    import pyomo.environ as pyo
    import pyomo.dae as dae
    from pyomo.contrib.parmest.experiment import Experiment as PE

    class E(PE):
        def __init__(self):
            self.model = None

        def get_labeled_model(self):
            if self.model is not None:
                return self.model
            m = pyo.ConcreteModel()
            m.t = dae.ContinuousSet(bounds=(0, 1), initialize=TIMES)
            m.th = pyo.Var(list(THETA), initialize=THETA)
            m.th.fix()
            m.T = pyo.Var(initialize=design["T"], bounds=BOUNDS["T"])
            m.T.fix()
            m.CA0 = pyo.Var(initialize=design["CA0"], bounds=BOUNDS["CA0"])
            m.CA0.fix()
            m.C = pyo.Var(["A", "B", "C"], m.t, initialize=0.5, bounds=(0, None))
            m.dC = dae.DerivativeVar(m.C, wrt=m.t)
            k1 = lambda m: m.th["A1"] * pyo.exp(-m.th["E1"] * 1000 / (R * m.T))
            k2 = lambda m: m.th["A2"] * pyo.exp(-m.th["E2"] * 1000 / (R * m.T))

            @m.Constraint(m.t)
            def ra(m, t):
                return m.dC["A", t] == -k1(m) * m.C["A", t]

            @m.Constraint(m.t)
            def rb(m, t):
                return m.dC["B", t] == k1(m) * m.C["A", t] - k2(m) * m.C["B", t]

            @m.Constraint(m.t)
            def rc(m, t):
                return m.dC["C", t] == k2(m) * m.C["B", t]

            for c in (m.ra, m.rb, m.rc):
                c[0].deactivate()
            m.ic = pyo.ConstraintList()
            m.ic.add(m.C["A", 0] == m.CA0)
            m.ic.add(m.C["B", 0] == 0)
            m.ic.add(m.C["C", 0] == 0)
            if scheme == "BACKWARD":
                pyo.TransformationFactory("dae.finite_difference").apply_to(
                    m, nfe=nfe, scheme="BACKWARD"
                )
            else:
                pyo.TransformationFactory("dae.collocation").apply_to(
                    m, nfe=nfe, ncp=ncp, scheme=scheme
                )
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            for t in TIMES:
                for s in "ABC":
                    m.experiment_outputs[m.C[s, t]] = None
                    m.measurement_error[m.C[s, t]] = SIG
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs.update([(m.T, design["T"]), (m.CA0, design["CA0"])])
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update((m.th[n], THETA[n]) for n in THETA)
            self.model = m
            return m

    return E()


def reorder_pyomo(F):
    # pyomo outputs are ordered A@t1..A@t5, B@..., C@... ; FIM is output-order independent.
    return F


if __name__ == "__main__":
    from discopt.doe import compute_fim, optimal_experiment

    designs = [{"T": 300.0, "CA0": 1.0}, {"T": 450.0, "CA0": 3.0}, {"T": 700.0, "CA0": 5.0}]
    print("== FIM rel. error vs closed form ==")
    for dsg in designs:
        ref = reference_fim(analytic, THETA, dsg, SIG)
        line = f"{dsg}: cond(ref)={np.linalg.cond(ref):.1e}"
        for n in [10, 50, 200]:
            line += f" | discopt rk4 n={n}: {rel_err(compute_fim(discopt_ode(n), THETA, dsg).fim, ref):.1e}"
        print(line)
        line = "   "
        for nfe, ncp, sch in [
            (4, 3, "LAGRANGE-RADAU"),
            (10, 3, "LAGRANGE-RADAU"),
            (20, 3, "LAGRANGE-RADAU"),
            (20, 1, "BACKWARD"),
            (100, 1, "BACKWARD"),
        ]:
            try:
                F, _ = pyomo_fim(pyomo_ode(dsg, nfe, ncp, sch))
                line += f" | pyomo {sch[:5]} nfe={nfe},ncp={ncp}: {rel_err(F, ref):.1e}"
            except Exception as e:
                line += f" | pyomo {sch[:5]} nfe={nfe}: ERR {str(e)[:50]}"
        print(line)

    print("\n== D-optimal design (T, CA0) ==")
    import itertools

    grid = [
        dict(T=float(T), CA0=float(c))
        for T, c in itertools.product(np.linspace(300, 700, 161), np.linspace(1, 5, 9))
    ]
    vals = [logdet(reference_fim(analytic, THETA, g, SIG)) for g in grid]
    bi = int(np.argmax(vals))
    print("grid best:", grid[bi], f"logdet={vals[bi]:.5f}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = time.time()
        r = optimal_experiment(discopt_ode(50), THETA, BOUNDS)
    print("discopt warnings:", [str(x.message)[:110] for x in w if "singular" in str(x.message)])
    print(
        f"discopt: {r.design} true logdet={logdet(reference_fim(analytic, THETA, r.design, SIG)):.5f} ({time.time() - t:.1f}s)"
    )
    for st in [{"T": 500.0, "CA0": 3.0}, {"T": 350.0, "CA0": 2.0}, {"T": 650.0, "CA0": 4.5}]:
        t = time.time()
        try:
            res, _ = pyomo_run_doe(pyomo_ode(st, 10, 3))
            dsg = dict(zip(["T", "CA0"], map(float, res["Experiment Design"])))
            print(
                f"pyomo from {st}: {dsg} [{res['Termination Condition']}] true logdet="
                f"{logdet(reference_fim(analytic, THETA, dsg, SIG)):.5f} reported ln det={res['log10 D-opt'] * np.log(10):.5f} ({time.time() - t:.1f}s)"
            )
        except Exception as e:
            print(f"pyomo from {st}: ERROR {str(e)[:100]}")

    print("\n== Well-posed: prior FIM from runs at T=350 and T=600 (CA0=3) ==")
    PRIOR = sum(reference_fim(analytic, THETA, {"T": T, "CA0": 3.0}, SIG) for T in (350.0, 600.0))
    vals = [logdet(reference_fim(analytic, THETA, g, SIG, PRIOR)) for g in grid]
    bi = int(np.argmax(vals))
    print("grid best:", grid[bi], f"logdet={vals[bi]:.5f}")
    t = time.time()
    r = optimal_experiment(discopt_ode(50), THETA, BOUNDS, prior_fim=PRIOR)
    print(
        f"discopt: {r.design} true logdet={logdet(reference_fim(analytic, THETA, r.design, SIG, PRIOR)):.5f} "
        f"reported={r.criterion_value:.5f} ({time.time() - t:.1f}s)"
    )
    for st in [{"T": 500.0, "CA0": 3.0}, {"T": 350.0, "CA0": 2.0}, {"T": 650.0, "CA0": 4.5}]:
        t = time.time()
        try:
            res, _ = pyomo_run_doe(pyomo_ode(st, 10, 3), prior_FIM=PRIOR)
            dsg = dict(zip(["T", "CA0"], map(float, res["Experiment Design"])))
            print(
                f"pyomo from {st}: {dsg} [{res['Termination Condition']}] true logdet="
                f"{logdet(reference_fim(analytic, THETA, dsg, SIG, PRIOR)):.5f} reported={res['log10 D-opt'] * np.log(10):.5f} ({time.time() - t:.1f}s)"
            )
        except Exception as e:
            print(f"pyomo from {st}: ERROR {str(e)[:100]}")
