"""T6: responses defined through equality constraints (implicit / algebraic states).

Model: a state z solves  z + k*z**3 = x  (a cubic equilibrium), and a second
state w = a*z enters through another constraint. Responses: z and w, i.e. the
measured quantities are *algebraic states*, the normal situation in a
flowsheet or a DAE. Exact sensitivities by the implicit function theorem:
    dz/dk = -z^3 / (1 + 3 k z^2),  dz/da = 0,  dw/dk = a dz/dk,  dw/da = z.
"""

import warnings

warnings.filterwarnings("ignore")
from common import (
    logdet,
    pyomo_fim,
    rel_err,
)
import numpy as np
import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel
from discopt.doe import compute_fim
from scipy.optimize import brentq

theta = {"k": 0.8, "a": 1.5}
sigma = 0.05


def exact(x):
    k, a = theta["k"], theta["a"]
    z = brentq(lambda z: z + k * z**3 - x, -10, 10)
    dzdk = -(z**3) / (1 + 3 * k * z**2)
    J = np.array([[dzdk, 0.0], [a * dzdk, z]])
    return J, J.T @ J / sigma**2


class DiscoptImplicit(Experiment):
    def create_model(self, **kw):
        m = dm.Model("implicit")
        k = m.continuous("k", lb=0, ub=10)
        a = m.continuous("a", lb=0, ub=10)
        x = m.continuous("x", lb=0, ub=5)
        z = m.continuous("z", lb=-10, ub=10)
        w = m.continuous("w", lb=-100, ub=100)
        m.subject_to(z + k * z**3 == x)
        m.subject_to(w == a * z)
        return ExperimentModel(
            m, {"k": k, "a": a}, {"x": x}, {"z": z, "w": w}, {"z": sigma, "w": sigma}
        )


def pyomo_exp(x0):
    import pyomo.environ as pyo
    from pyomo.contrib.parmest.experiment import Experiment as PE

    class E(PE):
        def get_labeled_model(self):
            m = pyo.ConcreteModel()
            m.k = pyo.Var(initialize=theta["k"])
            m.a = pyo.Var(initialize=theta["a"])
            m.k.fix()
            m.a.fix()
            m.x = pyo.Var(initialize=x0, bounds=(0, 5))
            m.x.fix()
            m.z = pyo.Var(initialize=1.0, bounds=(-10, 10))
            m.w = pyo.Var(initialize=1.0)
            m.c1 = pyo.Constraint(expr=m.z + m.k * m.z**3 == m.x)
            m.c2 = pyo.Constraint(expr=m.w == m.a * m.z)
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update([(m.z, None), (m.w, None)])
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update([(m.z, sigma), (m.w, sigma)])
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs.update([(m.x, x0)])
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update([(m.k, theta["k"]), (m.a, theta["a"])])
            return m

    return E()


if __name__ == "__main__":
    for x0 in [0.5, 2.0, 4.0]:
        Jx, Fx = exact(x0)
        print(f"\nx = {x0}\n exact J =\n{Jx}")
        try:
            r = compute_fim(DiscoptImplicit(), theta, {"x": x0})
            print(
                " discopt J =\n",
                r.jacobian,
                "\n discopt FIM rel err:",
                f"{rel_err(r.fim, Fx):.2e}",
                " logdet exact/discopt:",
                logdet(Fx),
                logdet(r.fim),
            )
        except Exception as e:
            print(" discopt ERROR", type(e).__name__, str(e)[:300])
        F, doe = pyomo_fim(pyomo_exp(x0))
        print(" pyomo J =\n", doe.seq_jac, "\n pyomo FIM rel err:", f"{rel_err(F, Fx):.2e}")
