"""R3-1..3: stiff kinetics, an unknown initial condition, a parameter in the
measurement function. All against closed forms.

A -> B -> C, first order, k1 >> k2 (stiff). Measured at fixed times; design is
the initial amount CA0 (plus, for R3-2, an unknown initial B0 as a parameter;
for R3-3, measured signal = f * CB with an unknown response factor f).
"""

import time
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import numpy as np
from common import logdet, pyomo_fim, rel_err

TIMES = [0.05, 0.2, 1.0, 3.0]
SIG = 0.01


def closed_form(k1, k2, a0, b0, t):
    ea, eb = jnp.exp(-k1 * t), jnp.exp(-k2 * t)
    ca = a0 * ea
    cb = b0 * eb + a0 * k1 / (k2 - k1) * (ea - eb)
    return ca, cb


def ref_jac(fun, theta):
    names = list(theta)
    J = jax.jacfwd(lambda v: fun(dict(zip(names, v))))(jnp.array([theta[n] for n in names]))
    return np.asarray(J)


def fim(J):
    return J.T @ J / SIG**2


# --------------------------------------------------------------- builders
def discopt_exp(theta, *, b0="zero", factor=False, method="rk4", n_steps=50):
    from discopt.doe import ode_experiment

    def rhs(t, x, p, u):
        return {"A": -p["k1"] * x["A"], "B": p["k1"] * x["A"] - p["k2"] * x["B"]}

    measured = {"S": lambda x, p, u: p["f"] * x["B"]} if factor else ["A", "B"]
    return ode_experiment(
        rhs,
        states={"A": "CA0", "B": ("B0" if b0 == "param" else 0.0)},
        parameters=theta,
        measured=measured,
        sample_times=TIMES,
        design_inputs={"CA0": (0.5, 2.0)},
        measurement_error=SIG,
        n_steps=n_steps,
        method=method,
    )


def pyomo_exp(theta, design, *, b0="zero", factor=False, nfe=10, ncp=3):
    import pyomo.dae as dae
    import pyomo.environ as pyo
    from pyomo.contrib.parmest.experiment import Experiment as PE

    class E(PE):
        def __init__(self):
            self.model = None

        def get_labeled_model(self):
            if self.model is not None:
                return self.model
            m = pyo.ConcreteModel()
            m.t = dae.ContinuousSet(bounds=(0, TIMES[-1]), initialize=TIMES)
            m.th = pyo.Var(list(theta), initialize=theta)
            m.th.fix()
            m.CA0 = pyo.Var(initialize=design["CA0"], bounds=(0.5, 2.0))
            m.CA0.fix()
            m.C = pyo.Var(["A", "B"], m.t, initialize=0.5)
            m.dC = dae.DerivativeVar(m.C, wrt=m.t)
            m.ra = pyo.Constraint(m.t, rule=lambda m, t: m.dC["A", t] == -m.th["k1"] * m.C["A", t])
            m.rb = pyo.Constraint(
                m.t,
                rule=lambda m, t: (
                    m.dC["B", t] == m.th["k1"] * m.C["A", t] - m.th["k2"] * m.C["B", t]
                ),
            )
            m.ra[0].deactivate()
            m.rb[0].deactivate()
            m.ic_a = pyo.Constraint(expr=m.C["A", 0] == m.CA0)
            m.ic_b = pyo.Constraint(expr=m.C["B", 0] == (m.th["B0"] if b0 == "param" else 0.0))
            pyo.TransformationFactory("dae.collocation").apply_to(
                m, nfe=nfe, ncp=ncp, scheme="LAGRANGE-RADAU"
            )
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            if factor:
                m.S = pyo.Var(TIMES, initialize=0.5)
                m.sdef = pyo.Constraint(TIMES, rule=lambda m, t: m.S[t] == m.th["f"] * m.C["B", t])
                for t in TIMES:
                    m.experiment_outputs[m.S[t]] = None
                    m.measurement_error[m.S[t]] = SIG
            else:
                for s in "AB":
                    for t in TIMES:
                        m.experiment_outputs[m.C[s, t]] = None
                        m.measurement_error[m.C[s, t]] = SIG
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs[m.CA0] = design["CA0"]
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update((m.th[n], theta[n]) for n in theta)
            self.model = m
            return m

    return E()


def reference(theta, design, *, b0="zero", factor=False):
    def f(th):
        outs = []
        for t in TIMES:
            ca, cb = closed_form(
                th["k1"], th["k2"], design["CA0"], th["B0"] if b0 == "param" else 0.0, t
            )
            outs.append(th["f"] * cb if factor else jnp.stack([ca, cb]))
        if factor:
            return jnp.stack(outs)
        return jnp.concatenate([jnp.stack([o[0] for o in outs]), jnp.stack([o[1] for o in outs])])

    return fim(ref_jac(f, theta))


if __name__ == "__main__":
    from discopt.doe import compute_fim

    d = {"CA0": 1.0}
    print("== R3-1 stiff kinetics (k1=500, k2=1), FIM rel. error vs closed form ==")
    th = {"k1": 500.0, "k2": 1.0}
    F0 = reference(th, d)
    print(f"  cond(FIM)={np.linalg.cond(F0):.1e}")
    for method, n in [
        ("rk4", 50),
        ("rk4", 400),
        ("rk4", 2000),
        ("trapezoid", 50),
        ("trapezoid", 400),
    ]:
        ex = discopt_exp(th, method=method, n_steps=n)
        t = time.time()
        F = compute_fim(ex, th, d).fim
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ex.check_accuracy(th, d)
        flag = " [check_accuracy warns]" if w else ""
        print(
            f"  discopt {method:9s} n={n:5d}: {rel_err(F, F0):.2e}{flag} ({time.time() - t:.1f}s)"
        )
    for nfe in [10, 40]:
        F, _ = pyomo_fim(pyomo_exp(th, d, nfe=nfe))
        print(f"  pyomo collocation nfe={nfe}: {rel_err(F, F0):.2e}")

    print("\n== R3-2 unknown initial condition B0 (a parameter) ==")
    th = {"k1": 2.0, "k2": 0.5, "B0": 0.3}
    F0 = reference(th, d, b0="param")
    try:
        F = compute_fim(discopt_exp(th, b0="param"), th, d).fim
        print(
            f"  discopt rk4 n=50: {rel_err(F, F0):.2e}  names="
            f"{compute_fim(discopt_exp(th, b0='param'), th, d).parameter_names}"
        )
    except Exception as e:
        print(f"  discopt: ERROR {type(e).__name__}: {str(e)[:100]}")
    F, _ = pyomo_fim(pyomo_exp(th, d, b0="param"))
    print(f"  pyomo collocation nfe=10: {rel_err(F, F0):.2e}")

    print("\n== R3-3 unknown response factor in the measurement function ==")
    th = {"k1": 2.0, "k2": 0.5, "f": 3.0}
    F0 = reference(th, d, factor=True)
    F = compute_fim(discopt_exp(th, factor=True), th, d).fim
    print(
        f"  discopt rk4 n=50: {rel_err(F, F0):.2e}  logdet={logdet(F):.4f} (exact {logdet(F0):.4f})"
    )
    F, _ = pyomo_fim(pyomo_exp(th, d, factor=True))
    print(f"  pyomo collocation nfe=10: {rel_err(F, F0):.2e}  logdet={logdet(F):.4f}")
