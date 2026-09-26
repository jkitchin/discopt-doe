"""Shared adapters for the discopt.doe vs pyomo.doe comparison.

A problem is written once as ``resp(theta, design, ops) -> list of responses``
using only arithmetic and ``ops.exp`` / ``ops.log`` / ``ops.sin``. The same
callable is then

* built as a discopt ``Experiment`` (``to_discopt``),
* built as a Pyomo.DoE experiment with one output Var + defining constraint per
  response (``to_pyomo``), exactly the pattern of Pyomo's own examples, and
* differentiated directly with ``jax.jacfwd`` (``reference_jacobian``). This is
  the exact sensitivity, independent of both packages' machinery (no discopt
  model compilation, no finite differences), and is the yardstick both are
  judged against.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

IPOPT = os.environ.get("IPOPT_EXE", "ipopt")


# ─────────────────────────────── reference ────────────────────────────────


JNP_OPS = types.SimpleNamespace(exp=jnp.exp, log=jnp.log, sin=jnp.sin, cos=jnp.cos)


def reference_jacobian(resp, theta, design):
    names = list(theta)

    def f(vec):
        th = dict(zip(names, vec))
        return jnp.stack([jnp.asarray(v, dtype=float) for v in resp(th, design, JNP_OPS)])

    return np.asarray(jax.jacfwd(f)(jnp.array([theta[n] for n in names], dtype=float)))


def reference_fim(resp, theta, design, sigma, prior=None):
    J = reference_jacobian(resp, theta, design)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    F = J.T @ np.diag(1.0 / s**2) @ J
    return F if prior is None else F + prior


# ─────────────────────────────── discopt ──────────────────────────────────


def to_discopt(resp, theta, design_bounds, sigma, *, param_bounds=None, n_resp=None):
    import discopt.modeling as dm
    from discopt.estimate import Experiment, ExperimentModel

    ops = types.SimpleNamespace(exp=dm.exp, log=dm.log, sin=dm.sin, cos=dm.cos)
    param_bounds = param_bounds or {}

    class _Exp(Experiment):
        def create_model(self, **kw):
            m = dm.Model("cmp")
            th = {}
            for n, v in theta.items():
                lo, hi = param_bounds.get(n, (-1e6, 1e6))
                th[n] = m.continuous(n, lb=lo, ub=hi)
            d = {n: m.continuous(n, lb=lo, ub=hi) for n, (lo, hi) in design_bounds.items()}
            ys = resp(th, d, ops)
            s = np.broadcast_to(np.asarray(sigma, dtype=float), (len(ys),))
            return ExperimentModel(
                model=m,
                unknown_parameters=th,
                design_inputs=d,
                responses={f"y{i}": y for i, y in enumerate(ys)},
                measurement_error={f"y{i}": float(s[i]) for i in range(len(ys))},
            )

    return _Exp()


# ──────────────────────────────── pyomo ───────────────────────────────────


def to_pyomo(resp, theta, design, design_bounds, sigma, *, n_resp):
    import pyomo.environ as pyo
    from pyomo.contrib.parmest.experiment import Experiment as PyoExperiment

    ops = types.SimpleNamespace(exp=pyo.exp, log=pyo.log, sin=pyo.sin, cos=pyo.cos)

    class _Exp(PyoExperiment):
        def __init__(self):
            self.model = None

        def get_labeled_model(self):
            if self.model is not None:
                return self.model
            m = pyo.ConcreteModel()
            m.th = pyo.Var(list(theta), initialize=theta)
            m.th.fix()
            m.d = pyo.Var(list(design_bounds), initialize=design)
            for n, (lo, hi) in design_bounds.items():
                m.d[n].setlb(lo)
                m.d[n].setub(hi)
            m.d.fix()
            th = {n: m.th[n] for n in theta}
            dd = {n: m.d[n] for n in design_bounds}
            ys = resp(th, dd, ops)
            m.y = pyo.Var(range(n_resp), initialize=0.0)
            m.con = pyo.Constraint(range(n_resp), rule=lambda m, i: m.y[i] == ys[i])
            s = np.broadcast_to(np.asarray(sigma, dtype=float), (n_resp,))
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update((m.y[i], None) for i in range(n_resp))
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update((m.y[i], float(s[i])) for i in range(n_resp))
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs.update((m.d[n], design[n]) for n in design_bounds)
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update((m.th[n], theta[n]) for n in theta)
            self.model = m
            return m

    return _Exp()


def ipopt(**options):
    import pyomo.environ as pyo

    s = pyo.SolverFactory("ipopt", executable=IPOPT)
    s.options["linear_solver"] = "mumps"
    s.options.update(options)
    return s


def pyomo_fim(pexp, **kw):
    import logging

    from pyomo.contrib.doe import DesignOfExperiments

    kw.setdefault("solver", ipopt())
    doe = DesignOfExperiments(experiment=pexp, logger_level=logging.ERROR, **kw)
    return np.asarray(doe.compute_FIM(method="sequential")), doe


def pyomo_run_doe(pexp, **kw):
    import logging

    from pyomo.contrib.doe import DesignOfExperiments

    kw.setdefault("solver", ipopt())
    doe = DesignOfExperiments(experiment=pexp, logger_level=logging.ERROR, **kw)
    doe.run_doe()
    return doe.results, doe


def rel_err(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300))


def logdet(F):
    s, v = np.linalg.slogdet(F)
    return v if s > 0 else -np.inf
