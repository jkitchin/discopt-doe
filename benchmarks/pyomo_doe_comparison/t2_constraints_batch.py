"""R2-2: constrained designs; R2-3: multi-experiment (batch) designs."""

import itertools
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from common import logdet, pyomo_run_doe, reference_fim, to_discopt, to_pyomo
from discopt.doe import BatchStrategy, batch_optimal_experiment, optimal_experiment
from t_design import PROBLEMS


def fmtd(d):
    return "{" + ", ".join(f"{k}={v:.4g}" for k, v in d.items()) + "}"


def constrained():
    print("== R2-2 constrained designs (D-optimal) ==")
    cases = [
        # (problem, constraint g(d) >= 0 in discopt form, pyomo builder, label)
        (
            "michaelis_menten_2pt",
            lambda d: 3.0 - d["S1"] - d["S2"],
            lambda m: m.d["S1"] + m.d["S2"] <= 3.0,
            "S1 + S2 <= 3",
        ),
        (
            "arrhenius_2T",
            lambda d: 720.0 - d["T1"] - d["T2"],
            lambda m: m.d["T1"] + m.d["T2"] <= 720.0,
            "T1 + T2 <= 720",
        ),
        (
            "arrhenius_2T",
            lambda d: d["T2"] - d["T1"] - 50.0,
            lambda m: m.d["T2"] >= m.d["T1"] + 50.0,
            "T2 >= T1 + 50",
        ),
        (
            "biexp_4times",
            lambda d: 5.0 - (d["t0"] + d["t1"] + d["t2"] + d["t3"]),
            lambda m: sum(m.d[f"t{i}"] for i in range(4)) <= 5.0,
            "sum t <= 5",
        ),
    ]
    rng = np.random.default_rng(7)
    for name, g, pcon, label in cases:
        p = PROBLEMS[name]
        names = list(p["bounds"])
        crit = lambda d: logdet(reference_fim(p["resp"], p["theta"], d, p["sigma"], p["prior"]))
        # brute force: dense random feasible sample + SLSQP polish of the best 10
        from scipy.optimize import minimize

        pts = [{k: float(rng.uniform(*p["bounds"][k])) for k in names} for _ in range(20000)]
        pts = [d for d in pts if g(d) >= 0]
        pts.sort(key=lambda d: -crit(d) if np.isfinite(crit(d)) else np.inf)
        best, bv = None, -np.inf
        for d0 in pts[:10]:
            r = minimize(
                lambda z: -crit(dict(zip(names, z))),
                [d0[k] for k in names],
                method="SLSQP",
                bounds=list(p["bounds"].values()),
                constraints=[{"type": "ineq", "fun": lambda z: g(dict(zip(names, z)))}],
            )
            d = dict(zip(names, map(float, r.x)))
            if g(d) >= -1e-7 and crit(d) > bv:
                best, bv = d, crit(d)
        r = optimal_experiment(
            to_discopt(p["resp"], p["theta"], p["bounds"], p["sigma"]),
            p["theta"],
            p["bounds"],
            prior_fim=p["prior"],
            inequality_constraints=[g],
        )
        print(f"\n### {name}  [{label}]  brute {fmtd(best)} = {bv:.6g}")
        print(f"  discopt {fmtd(r.design)} = {crit(r.design):.6g}  g={g(r.design):.2e}")
        # pyomo: feasible start (scale the midpoint into the feasible set)
        starts = [d for d in pts[len(pts) // 2 :: max(1, len(pts) // 8)]][:3]
        for st in starts:
            pe = to_pyomo(p["resp"], p["theta"], st, p["bounds"], p["sigma"], n_resp=p["n"])
            import pyomo.environ as pyo

            m = pe.get_labeled_model()
            m.design_con = pyo.Constraint(expr=pcon(m))
            try:
                res, doe = pyomo_run_doe(pe, prior_FIM=p["prior"])
                d = dict(zip(names, map(float, res["Experiment Design"])))
                print(
                    f"  pyomo from {fmtd(st)} -> {fmtd(d)} = {crit(d):.6g}  g={g(d):.2e} "
                    f"[{res['Termination Condition']}]"
                )
            except Exception as e:
                print(f"  pyomo from {fmtd(st)} -> ERROR {type(e).__name__}: {str(e)[:80]}")


def batch():
    print("\n== R2-3 multi-experiment designs (N single-measurement experiments, D-optimal) ==")
    mm = lambda th, d, o: [th["V"] * d["S"] / (th["K"] + d["S"])]
    rb = lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d["t"]))]
    exp2 = lambda th, d, o: [th["a"] * o.exp(-th["b"] * d["t"]) + th["c"]]
    cases = [
        ("michaelis_menten", mm, {"V": 2.0, "K": 0.3}, {"S": (0.01, 5.0)}, 0.05, None, 2),
        ("michaelis_menten", mm, {"V": 2.0, "K": 0.3}, {"S": (0.01, 5.0)}, 0.05, None, 4),
        ("rooney_biegler", rb, {"A": 15.0, "k": 0.5}, {"t": (0.5, 10.0)}, 0.1, None, 2),
        ("exp+offset", exp2, {"a": 1.0, "b": 0.7, "c": 0.2}, {"t": (0.0, 10.0)}, 0.02, None, 3),
    ]
    for name, resp, th, bounds, sig, prior, N in cases:
        k = list(bounds)[0]
        F1 = lambda v: reference_fim(resp, th, {k: v}, sig)
        grid = np.linspace(*bounds[k], 121 if N <= 3 else 41)
        Fg = [F1(v) for v in grid]
        best, bv = None, -np.inf
        for combo in itertools.combinations_with_replacement(range(len(grid)), N):
            v = logdet(sum(Fg[i] for i in combo) + (prior if prior is not None else 0))
            if v > bv:
                best, bv = [grid[i] for i in combo], v
        from scipy.optimize import minimize

        r = minimize(
            lambda z: -logdet(sum(F1(v) for v in z)),
            best,
            method="L-BFGS-B",
            bounds=[bounds[k]] * N,
        )
        if -r.fun > bv:
            best, bv = sorted(r.x), -r.fun
        print(f"\n### {name} N={N}: brute joint {np.round(sorted(best), 4).tolist()} = {bv:.6g}")
        ex = to_discopt(resp, th, bounds, sig)
        for strat in [BatchStrategy.GREEDY, BatchStrategy.JOINT]:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                br = batch_optimal_experiment(ex, th, bounds, N, strategy=strat)
            vals = sorted(d[k] for d in br.designs)
            tv = logdet(sum(F1(v) for v in vals))
            print(
                f"  discopt {strat:6s} {np.round(vals, 4).tolist()} = {tv:.6g} (reported {br.criterion_value:.6g})"
            )
        # pyomo: run_multi_doe_* raise NotImplementedError; sequential via prior_FIM
        acc, picks = np.zeros((len(th), len(th))), []
        try:
            for i in range(N):
                mid = {k: 0.37 * bounds[k][0] + 0.63 * bounds[k][1]}
                res, _ = pyomo_run_doe(
                    to_pyomo(resp, th, mid, bounds, sig, n_resp=1),
                    prior_FIM=acc.copy() if acc.any() else None,
                )
                v = float(res["Experiment Design"][0])
                picks.append(v)
                acc = acc + F1(v)
            print(
                f"  pyomo sequential(prior) {np.round(sorted(picks), 4).tolist()} = {logdet(acc):.6g}"
            )
        except Exception as e:
            print(
                f"  pyomo sequential(prior) picks={np.round(picks, 4).tolist()} -> ERROR "
                f"{type(e).__name__}: {str(e)[:90]}"
            )


if __name__ == "__main__":
    constrained()
    batch()
