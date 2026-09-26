"""T7: optimal design, discopt.optimal_experiment vs pyomo run_doe vs brute force."""

import itertools
import time
import warnings

warnings.filterwarnings("ignore")
from common import (
    logdet,
    pyomo_run_doe,
    reference_fim,
    to_discopt,
    to_pyomo,
)
import numpy as np
from discopt.doe import optimal_experiment
from scipy.optimize import minimize

rb_data = [1, 2, 3, 4, 5, 7]
rb = lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d["t"]))]
rb_th = {"A": 15.0, "k": 0.5}
rb_prior = sum(reference_fim(rb, rb_th, {"t": t}, 0.1) for t in rb_data)

PROBLEMS = {
    # pyomo's own Rooney-Biegler DoE example (prior from its 6 data points)
    "rooney_biegler+prior": dict(
        resp=rb, theta=rb_th, bounds={"t": (0.0, 10.0)}, n=1, sigma=0.1, prior=rb_prior
    ),
    # classic: D-optimal two-point MM design is S = {K*Smax/(2K+Smax), Smax}
    "michaelis_menten_2pt": dict(
        resp=lambda th, d, o: [
            th["V"] * d["S1"] / (th["K"] + d["S1"]),
            th["V"] * d["S2"] / (th["K"] + d["S2"]),
        ],
        theta={"V": 2.0, "K": 0.3},
        bounds={"S1": (0.01, 5.0), "S2": (0.01, 5.0)},
        n=2,
        sigma=0.05,
        prior=None,
    ),
    "arrhenius_2T": dict(
        resp=lambda th, d, o: [
            o.exp(th["lnA"] - th["E"] / (8.314 * d["T1"])),
            o.exp(th["lnA"] - th["E"] / (8.314 * d["T2"])),
        ],
        theta={"lnA": 20.0, "E": 7.0e4},
        bounds={"T1": (300.0, 400.0), "T2": (300.0, 400.0)},
        n=2,
        sigma=1e-3,
        prior=None,
    ),
    # many local optima in x
    "oscillator+prior": dict(
        resp=lambda th, d, o: [th["amp"] * o.sin(th["w"] * d["x"])],
        theta={"amp": 1.0, "w": 3.0},
        bounds={"x": (0.0, 10.0)},
        n=1,
        sigma=0.1,
        prior=np.array([[100.0, 0.0], [0.0, 1.0]]),
    ),
    "biexp_4times": dict(
        resp=lambda th, d, o: [
            th["a1"] * o.exp(-th["k1"] * d[f"t{i}"]) + th["a2"] * o.exp(-th["k2"] * d[f"t{i}"])
            for i in range(4)
        ],
        theta={"a1": 1.0, "k1": 2.0, "a2": 0.5, "k2": 0.1},
        bounds={f"t{i}": (0.05, 30.0) for i in range(4)},
        n=4,
        sigma=0.01,
        prior=None,
    ),
}


def ref_crit(p, dsg, crit):
    F = reference_fim(p["resp"], p["theta"], dsg, p["sigma"], p["prior"])
    if crit == "determinant":
        return logdet(F)
    try:
        return float(np.trace(np.linalg.inv(F))) if np.linalg.cond(F) < 1e14 else np.inf
    except np.linalg.LinAlgError:
        return np.inf


def brute_force(p, crit, seed=0):
    names = list(p["bounds"])
    better = (lambda a, b: a > b) if crit == "determinant" else (lambda a, b: a < b)
    best, best_v = None, None
    if len(names) <= 2:
        grids = [
            np.linspace(lo, hi, 401 if len(names) == 1 else 161) for lo, hi in p["bounds"].values()
        ]
        pts = [dict(zip(names, map(float, c))) for c in itertools.product(*grids)]
    else:
        rng = np.random.default_rng(seed)
        pts = [{n: float(rng.uniform(*p["bounds"][n])) for n in names} for _ in range(3000)]
    for d in pts:
        v = ref_crit(p, d, crit)
        if np.isfinite(v) and (best is None or better(v, best_v)):
            best, best_v = d, v
    # polish
    sgn = -1 if crit == "determinant" else 1
    f = lambda z: (
        sgn * ref_crit(p, dict(zip(names, z)), crit)
        if np.isfinite(ref_crit(p, dict(zip(names, z)), crit))
        else 1e30
    )
    r = minimize(f, [best[n] for n in names], bounds=list(p["bounds"].values()), method="L-BFGS-B")
    if sgn * r.fun < sgn * best_v or not np.isfinite(best_v):
        pass
    cand = dict(zip(names, map(float, r.x)))
    if better(ref_crit(p, cand, crit), best_v):
        best, best_v = cand, ref_crit(p, cand, crit)
    return best, best_v


def run_discopt(p, crit):
    ex = to_discopt(p["resp"], p["theta"], p["bounds"], p["sigma"])
    t = time.time()
    r = optimal_experiment(ex, p["theta"], p["bounds"], criterion=crit, prior_fim=p["prior"])
    return r.design, ref_crit(p, r.design, crit), time.time() - t


def run_pyomo(p, crit, start):
    pe = to_pyomo(p["resp"], p["theta"], start, p["bounds"], p["sigma"], n_resp=p["n"])
    t = time.time()
    try:
        res, doe = pyomo_run_doe(pe, objective_option=crit, prior_FIM=p["prior"])
        names = list(p["bounds"])
        dsg = dict(zip(names, map(float, res["Experiment Design"])))
        return dsg, ref_crit(p, dsg, crit), str(res["Termination Condition"]), time.time() - t, res
    except Exception as e:
        return None, None, f"ERROR {type(e).__name__}: {str(e)[:80]}", time.time() - t, None


def fmtd(d):
    return "{" + ", ".join(f"{k}={v:.4g}" for k, v in d.items()) + "}" if d else str(d)


if __name__ == "__main__":
    import sys

    only = sys.argv[1:] or list(PROBLEMS)
    rng = np.random.default_rng(1)
    for name in only:
        p = PROBLEMS[name]
        for crit in ["determinant", "trace"]:
            bd, bv = brute_force(p, crit)
            dd, dv, dt = run_discopt(p, crit)
            print(f"\n### {name} [{crit}]  brute-force best {fmtd(bd)} crit={bv:.6g}")
            print(f"  discopt: {fmtd(dd)} crit={dv:.6g} ({dt:.1f}s)")
            starts = [{k: 0.5 * (lo + hi) for k, (lo, hi) in p["bounds"].items()}]
            starts += [
                {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in p["bounds"].items()}
                for _ in range(4)
            ]
            for st in starts:
                pd_, pv, term, pt, res = run_pyomo(p, crit, st)
                extra = ""
                if res is not None and crit == "determinant":
                    extra = f" reported log10D={res['log10 D-opt']:.6g} (ln={res['log10 D-opt'] * np.log(10):.6g})"
                pvs = f"{pv:.6g}" if pv is not None else "-"
                print(
                    f"  pyomo from {fmtd(st)} -> {fmtd(pd_)} crit={pvs} [{term}] ({pt:.1f}s){extra}"
                )
