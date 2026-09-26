"""Round 5: the two gaps closed in discopt.doe, compared with pyomo.doe.

R5-1 scale_parameters=True vs pyomo's scale_nominal_param_value=True: A-, E-
and D-optimal designs, judged by brute force on the scaled criterion.
R5-2 response_bounds (constraints on predicted outputs) vs a constraint on the
output variable in the pyomo model, judged by brute force on the feasible set.
"""

import itertools
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from common import logdet, pyomo_run_doe, reference_fim, to_discopt, to_pyomo
from discopt.doe import compute_fim, optimal_experiment
from t2_criteria import greybox_solver
from t_design import PROBLEMS
from t_ode import BOUNDS as ODE_BOUNDS
from t_ode import SIG as ODE_SIG
from t_ode import THETA as ODE_THETA
from t_ode import analytic


def crit(F, c):
    if c == "determinant":
        return logdet(F)
    ev = np.linalg.eigvalsh(F)
    if ev[0] <= ev[-1] * 1e-14:
        return np.inf if c == "trace" else 0.0
    return float(np.sum(1 / ev)) if c == "trace" else float(ev[0])


def grid(bounds, n):
    names = list(bounds)
    axes = [np.linspace(lo, hi, n) for lo, hi in bounds.values()]
    return [dict(zip(names, map(float, c))) for c in itertools.product(*axes)]


if __name__ == "__main__":
    print("== R5-1 relative-parameter (scaled) FIM ==")
    for name in ["michaelis_menten_2pt", "arrhenius_2T", "rooney_biegler+prior"]:
        p = PROBLEMS[name]
        th = p["theta"]
        S = np.diag([abs(th[n]) for n in th])
        prior = p["prior"]
        ex = to_discopt(p["resp"], th, p["bounds"], p["sigma"])
        pts = grid(p["bounds"], 201 if len(p["bounds"]) == 1 else 161)
        for c, pc in [
            ("trace", "trace"),
            ("min_eigenvalue", "minimum_eigenvalue"),
            ("determinant", "determinant"),
        ]:

            def scaled(d):
                F = reference_fim(p["resp"], th, d, p["sigma"], prior)
                return crit(S @ F @ S, c)

            vals = [scaled(d) for d in pts]
            ok = [v for v in vals if np.isfinite(v)]
            best = min(ok) if c == "trace" else max(ok)
            bd = pts[vals.index(best)]
            plain = optimal_experiment(ex, th, p["bounds"], criterion=c, prior_fim=prior)
            r = optimal_experiment(
                ex, th, p["bounds"], criterion=c, prior_fim=prior, scale_parameters=True
            )
            unscaled_ok = np.allclose(
                r.fim, compute_fim(ex, th, r.design, prior_fim=prior).fim, rtol=1e-8
            )
            line = (
                f"  {name:21s} {c:15s} brute(scaled) { ({k: round(v, 2) for k, v in bd.items()}) }"
                f" = {best:.6g}\n      discopt scaled   { ({k: round(v, 2) for k, v in r.design.items()}) }"
                f" = {scaled(r.design):.6g} (reported {r.criterion_value:.6g}; fim_result unscaled: {unscaled_ok})"
                f"\n      discopt unscaled { ({k: round(v, 2) for k, v in plain.design.items()}) }"
                f" -> scaled criterion {scaled(plain.design):.6g}"
            )
            st = {k: 0.37 * lo + 0.63 * hi for k, (lo, hi) in p["bounds"].items()}
            kw = dict(
                objective_option=pc,
                scale_nominal_param_value=True,
                prior_FIM=None if prior is None else S @ prior @ S,
            )
            if c == "min_eigenvalue":
                kw.update(use_grey_box_objective=True, grey_box_solver=greybox_solver(1e-8))
            try:
                res, _ = pyomo_run_doe(
                    to_pyomo(p["resp"], th, st, p["bounds"], p["sigma"], n_resp=p["n"]), **kw
                )
                d = dict(zip(p["bounds"], map(float, res["Experiment Design"])))
                line += (
                    f"\n      pyomo scaled     { ({k: round(v, 2) for k, v in d.items()}) } = "
                    f"{scaled(d):.6g} [{res['Termination Condition']}]"
                )
            except Exception as e:
                line += f"\n      pyomo scaled     ERROR {type(e).__name__}: {str(e)[:70]}"
            print(line)

    print("\n== R5-2 constraints on predicted outputs ==")
    import pyomo.environ as pyo

    # (a) Rooney-Biegler with a prior; unconstrained D-optimum t = 10 gives y = 14.9.
    p = PROBLEMS["rooney_biegler+prior"]
    th = p["theta"]
    ex = to_discopt(p["resp"], th, p["bounds"], p["sigma"])
    for ub in [12.0, 8.0]:
        pts = [{"t": float(t)} for t in np.linspace(0, 10, 4001)]
        feas = [d for d in pts if th["A"] * (1 - np.exp(-th["k"] * d["t"])) <= ub]
        vals = [logdet(reference_fim(p["resp"], th, d, p["sigma"], p["prior"])) for d in feas]
        bd = feas[int(np.argmax(vals))]
        r = optimal_experiment(
            ex, th, p["bounds"], prior_fim=p["prior"], response_bounds={"y0": (None, ub)}
        )
        yv = th["A"] * (1 - np.exp(-th["k"] * r.design["t"]))
        pe = to_pyomo(p["resp"], th, {"t": 1.0}, p["bounds"], p["sigma"], n_resp=1)
        m = pe.get_labeled_model()
        m.output_limit = pyo.Constraint(expr=m.y[0] <= ub)
        res, _ = pyomo_run_doe(pe, prior_FIM=p["prior"])
        tp = float(res["Experiment Design"][0])
        yp = th["A"] * (1 - np.exp(-th["k"] * tp))
        print(
            f"  RB, y <= {ub}: brute t={bd['t']:.4f} (logdet {max(vals):.5f}) | discopt t={r.design['t']:.4f} "
            f"y={yv:.4f} logdet {logdet(reference_fim(p['resp'], th, r.design, p['sigma'], p['prior'])):.5f}"
            f" | pyomo t={tp:.4f} y={yp:.4f} [{res['Termination Condition']}]"
        )

    # (b) A->B->C (closed form), design T and CA0, prior from two runs; limit the
    # amount of C formed at t = 1 (a by-product cap): response index 14 = CC@1.
    prior = sum(
        reference_fim(analytic, ODE_THETA, {"T": T, "CA0": 3.0}, ODE_SIG) for T in (350.0, 600.0)
    )
    exo = to_discopt(analytic, ODE_THETA, ODE_BOUNDS, ODE_SIG)
    cc = lambda d: float(analytic(ODE_THETA, d, np)[14])
    pts = grid(ODE_BOUNDS, 201)
    for cap in [1.0, 0.6]:
        feas = [d for d in pts if cc(d) <= cap]
        vals = [logdet(reference_fim(analytic, ODE_THETA, d, ODE_SIG, prior)) for d in feas]
        bd = feas[int(np.argmax(vals))]
        r = optimal_experiment(
            exo, ODE_THETA, ODE_BOUNDS, prior_fim=prior, response_bounds={"y14": (None, cap)}
        )
        lr = logdet(reference_fim(analytic, ODE_THETA, r.design, ODE_SIG, prior))
        ptxt = []
        for st, label in [
            ({"T": 450.0, "CA0": 2.0}, "start violating the cap"),
            ({"T": 300.0, "CA0": 1.0}, "feasible start"),
        ]:
            pe = to_pyomo(analytic, ODE_THETA, st, ODE_BOUNDS, ODE_SIG, n_resp=15)
            m = pe.get_labeled_model()
            m.byproduct_cap = pyo.Constraint(expr=m.y[14] <= cap)
            try:
                res, _ = pyomo_run_doe(pe, prior_FIM=prior)
                dp = dict(zip(ODE_BOUNDS, map(float, res["Experiment Design"])))
                ld = logdet(reference_fim(analytic, ODE_THETA, dp, ODE_SIG, prior))
                ptxt.append(
                    f"pyomo ({label}) { ({k: round(v, 2) for k, v in dp.items()}) } CC={cc(dp):.4f} "
                    f"logdet {ld:.5f} [{res['Termination Condition']}]"
                )
            except Exception as e:
                ptxt.append(f"pyomo ({label}) ERROR {type(e).__name__}: {str(e)[:60]}")
        ptxt = "\n      ".join(ptxt)
        print(
            f"  A->B->C, CC@1 <= {cap}: brute { ({k: round(v, 2) for k, v in bd.items()}) } logdet {max(vals):.5f}"
            f"\n      discopt { ({k: round(v, 2) for k, v in r.design.items()}) } CC={cc(r.design):.4f} logdet {lr:.5f}"
            f"\n      {ptxt}"
        )
