"""R2-1: E-, ME- (and grey-box D/A) optimal designs.

pyomo.doe reaches E and ME only through its grey-box objective (cyipopt). Its
grey-box A objective is trace(pinv(FIM)) and its D objective log|det(FIM)|; for
a singular FIM, pinv sums only the nonzero directions.
"""

import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from common import logdet, pyomo_run_doe, reference_fim, to_discopt, to_pyomo
from discopt.doe import optimal_experiment
from scipy.optimize import minimize
from t_design import PROBLEMS

CRITS = {  # discopt name -> (pyomo name, maximize?)
    "min_eigenvalue": ("minimum_eigenvalue", True),
    "condition_number": ("condition_number", False),
    "determinant": ("determinant", True),
    "trace": ("trace", False),
}


def crit_value(F, crit):
    if crit == "determinant":
        return logdet(F)
    ev = np.linalg.eigvalsh(F)
    if crit == "min_eigenvalue":
        return float(ev[0])
    if ev[0] <= ev[-1] * 1e-14:
        return np.inf
    return float(ev[-1] / ev[0]) if crit == "condition_number" else float(np.sum(1 / ev))


def ref_crit(p, d, crit):
    return crit_value(reference_fim(p["resp"], p["theta"], d, p["sigma"], p["prior"]), crit)


def brute(p, crit, n=3000, seed=0):
    names = list(p["bounds"])
    maximize = CRITS[crit][1]
    rng = np.random.default_rng(seed)
    if len(names) == 1:
        pts = [{names[0]: float(v)} for v in np.linspace(*p["bounds"][names[0]], 2001)]
    else:
        pts = [{k: float(rng.uniform(*p["bounds"][k])) for k in names} for _ in range(n)]
    vals = np.array([ref_crit(p, d, crit) for d in pts])
    ok = np.isfinite(vals)
    order = np.argsort(-vals if maximize else vals)
    order = [i for i in order if ok[i]][:20]
    best, bv = None, None
    for i in order:  # polish the 20 best with Nelder-Mead (criteria may be nonsmooth)
        f = lambda z: (
            (-1 if maximize else 1)
            * (
                ref_crit(
                    p, dict(zip(names, np.clip(z, *np.array(list(p["bounds"].values())).T))), crit
                )
            )
        )
        r = minimize(
            f,
            [pts[i][k] for k in names],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000},
        )
        lo, hi = np.array(list(p["bounds"].values())).T
        d = dict(zip(names, map(float, np.clip(r.x, lo, hi))))
        v = ref_crit(p, d, crit)
        if np.isfinite(v) and (bv is None or (v > bv if maximize else v < bv)):
            best, bv = d, v
    return best, bv


def _shim_numpy_eig():
    """numpy >= 2.5 returns complex arrays from np.linalg.eig even for real
    eigenvalues, which crashes pyomo.doe's grey-box E/ME objectives (pynumero
    refuses a complex Jacobian). Restore the numpy <= 2.4 behaviour inside that
    module only, so the objectives can be compared at all."""
    import types

    from pyomo.contrib.doe import grey_box_utilities as G

    def eig(a):
        w, v = np.linalg.eig(a)
        return np.real_if_close(w), np.real_if_close(v)

    G.np = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    G.np.linalg = types.SimpleNamespace(**{**vars(np.linalg), "eig": eig})


_shim_numpy_eig()


def greybox_solver(tol=1e-4):
    import pyomo.environ as pyo

    s = pyo.SolverFactory("cyipopt")
    s.config.options["linear_solver"] = "mumps"
    s.config.options["tol"] = tol  # 1e-4 is pyomo.doe's grey-box default
    s.config.options["mu_strategy"] = "monotone"
    # One grey-box solve from a bad start ran for over 20 minutes; cap it.
    s.config.options["max_cpu_time"] = 120.0
    return s


def fmtd(d):
    return "{" + ", ".join(f"{k}={v:.4g}" for k, v in d.items()) + "}" if d else str(d)


if __name__ == "__main__":
    import sys

    probs = sys.argv[1:] or [
        "michaelis_menten_2pt",
        "arrhenius_2T",
        "oscillator+prior",
        "biexp_4times",
        "rooney_biegler+prior",
    ]
    rng = np.random.default_rng(3)
    for name in probs:
        p = PROBLEMS[name]
        for crit, (pcrit, _) in CRITS.items():
            bd, bv = brute(p, crit)
            t = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = optimal_experiment(
                    to_discopt(p["resp"], p["theta"], p["bounds"], p["sigma"]),
                    p["theta"],
                    p["bounds"],
                    criterion=crit,
                    prior_fim=p["prior"],
                )
            sing = any("singular" in str(x.message) for x in w)
            print(f"\n### {name} [{crit}] brute {fmtd(bd)} = {bv:.6g}")
            print(
                f"  discopt {fmtd(r.design)} = {ref_crit(p, r.design, crit):.6g} "
                f"({time.time() - t:.1f}s){' [singular warning]' if sing else ''}"
            )
            starts = [{k: 0.5 * (lo + hi) for k, (lo, hi) in p["bounds"].items()}]
            starts += [
                {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in p["bounds"].items()}
                for _ in range(3)
            ]
            runs = [(st, 1e-4) for st in starts] + [(starts[1], 1e-8)]
            for st, tol in runs:
                t = time.time()
                try:
                    res, _ = pyomo_run_doe(
                        to_pyomo(p["resp"], p["theta"], st, p["bounds"], p["sigma"], n_resp=p["n"]),
                        objective_option=pcrit,
                        prior_FIM=p["prior"],
                        use_grey_box_objective=True,
                        grey_box_solver=greybox_solver(tol),
                    )
                    d = dict(zip(p["bounds"], map(float, res["Experiment Design"])))
                    rank = np.linalg.matrix_rank(
                        reference_fim(p["resp"], p["theta"], d, p["sigma"], p["prior"])
                    )
                    print(
                        f"  pyomo-gb tol={tol:.0e} from {fmtd(st)} -> {fmtd(d)} = {ref_crit(p, d, crit):.6g} "
                        f"[{res['Termination Condition']}] rank={rank} ({time.time() - t:.1f}s)"
                    )
                except Exception as e:
                    print(
                        f"  pyomo-gb tol={tol:.0e} from {fmtd(st)} -> ERROR {type(e).__name__}: {str(e)[:90]}"
                    )
