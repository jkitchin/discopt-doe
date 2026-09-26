"""Round 4: discopt.doe features pyomo.doe does not have, against exact
references / brute force: model discrimination, I-optimality, robust designs,
the symbolic (sympy) FIM path, sequential DoE bookkeeping."""

import itertools
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from common import logdet, reference_fim, reference_jacobian, rel_err, to_discopt


def grid1(lo, hi, n=2001):
    return np.linspace(lo, hi, n)


if __name__ == "__main__":
    from discopt.doe import optimal_experiment
    from discopt.doe.discrimination import (
        DiscriminationCriterion as DC,
    )
    from discopt.doe.discrimination import (
        discriminate_design,
        evaluate_discrimination_criterion,
    )

    print("== R4-1 model discrimination: first- vs second-order decay, design t ==")
    m1 = lambda th, d, o: [th["c0"] * o.exp(-th["k"] * d["t"])]
    m2 = lambda th, d, o: [th["c0"] / (1 + th["c0"] * th["k"] * d["t"])]
    th1, th2 = {"c0": 1.0, "k": 0.5}, {"c0": 1.0, "k": 0.6}
    b = {"t": (0.0, 10.0)}
    exps = {"first": to_discopt(m1, th1, b, 0.02), "second": to_discopt(m2, th2, b, 0.02)}
    ests = {"first": th1, "second": th2}
    ts = grid1(0.0, 10.0, 401)
    for crit in [DC.HR, DC.BF, DC.BH, DC.JR]:
        vals = [
            evaluate_discrimination_criterion(exps, ests, {"t": float(t)}, criterion=crit)
            for t in ts
        ]
        tb = ts[int(np.argmax(vals))]
        r = discriminate_design(exps, ests, b, criterion=crit, seed=0)
        rv = evaluate_discrimination_criterion(exps, ests, r.design, criterion=crit)
        print(
            f"  {crit.value:14s} grid best t={tb:.3f} ({max(vals):.5g}) | discopt t={r.design['t']:.3f} ({rv:.5g})"
        )
    # Hunter-Reiner has a closed form: (y1 - y2)^2
    hr_exact = [(float(np.exp(-0.5 * t)) - 1.0 / (1 + 0.6 * t)) ** 2 for t in ts]
    hr = [
        evaluate_discrimination_criterion(exps, ests, {"t": float(t)}, criterion=DC.HR) for t in ts
    ]
    w = 0.25  # equal priors -> w_i w_j = 1/4
    print(
        f"  HR vs closed form (w1 w2 (y1-y2)^2): max rel err "
        f"{max(abs(a - w * e) / max(w * e, 1e-300) for a, e in zip(hr, hr_exact) if e > 1e-12):.1e}"
    )

    print("\n== R4-2 I-optimal design vs brute force (Michaelis-Menten, 2 points) ==")
    from discopt.doe import experiment_region

    mm = lambda th, d, o: [
        th["V"] * d["S1"] / (th["K"] + d["S1"]),
        th["V"] * d["S2"] / (th["K"] + d["S2"]),
    ]
    th = {"V": 2.0, "K": 0.3}
    b2 = {"S1": (0.01, 5.0), "S2": (0.01, 5.0)}
    ex = to_discopt(mm, th, b2, 0.05)
    pts = [{"S1": float(s), "S2": float(s)} for s in np.linspace(0.01, 5.0, 64)]
    region = experiment_region(ex, th, points=pts)
    # reference I: mean over region rows f of f^T F^-1 f with f = d y / d theta
    rows = np.vstack([reference_jacobian(mm, th, p) for p in pts])

    def I_ref(d):
        F = reference_fim(mm, th, d, 0.05)
        try:
            Fi = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return np.inf
        return float(np.mean(np.einsum("ij,jk,ik->i", rows, Fi, rows)))

    g = np.linspace(0.01, 5.0, 150)
    vals = {(a, c): I_ref({"S1": a, "S2": c}) for a, c in itertools.product(g, g) if a < c}
    (a, c), bv = min(vals.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else np.inf)
    r = optimal_experiment(ex, th, b2, criterion="average_variance", prediction_region=region)
    print(
        f"  brute S=({a:.3f}, {c:.3f}) I={bv:.6g} | discopt {r.design} I={I_ref(r.design):.6g} "
        f"(reported {r.criterion_value:.6g})"
    )

    print("\n== R4-3 robust (maximin-efficiency and expected log det) designs vs brute force ==")
    from discopt.doe import robust_optimal_experiment

    rb = lambda th, d, o: [
        th["A"] * (1 - o.exp(-th["k"] * d["t1"])),
        th["A"] * (1 - o.exp(-th["k"] * d["t2"])),
    ]
    samples = [{"A": 15.0, "k": k} for k in (0.1, 0.3, 1.0, 3.0)]
    b3 = {"t1": (0.05, 20.0), "t2": (0.05, 20.0)}
    ex = to_discopt(rb, samples[0], b3, 0.1)
    g = np.exp(np.linspace(np.log(0.05), np.log(20.0), 160))
    best_single = []
    for s in samples:  # per-sample optimum log det, for efficiencies
        best_single.append(
            max(
                logdet(reference_fim(rb, s, {"t1": a, "t2": c}, 0.1))
                for a, c in itertools.product(g, g)
                if a < c
            )
        )

    def eff(d):
        return [
            np.exp((logdet(reference_fim(rb, s, d, 0.1)) - bs) / 2)
            for s, bs in zip(samples, best_single)
        ]

    exp_ld = {
        (a, c): np.mean([logdet(reference_fim(rb, s, {"t1": a, "t2": c}, 0.1)) for s in samples])
        for a, c in itertools.product(g, g)
        if a < c
    }
    mm_eff = {(a, c): min(eff({"t1": a, "t2": c})) for a, c in itertools.product(g, g) if a < c}
    for mode, table, better in [("expected", exp_ld, max), ("maximin", mm_eff, max)]:
        (a, c), bv = better(table.items(), key=lambda kv: kv[1])
        try:
            r = robust_optimal_experiment(ex, samples, b3, robust=mode, seed=0)
            d = r.designs[0] if hasattr(r, "designs") else r.design
            val = (
                np.mean([logdet(reference_fim(rb, s, d, 0.1)) for s in samples])
                if mode == "expected"
                else min(eff(d))
            )
            print(
                f"  {mode:8s} brute ({a:.3f}, {c:.3f}) = {bv:.5f} | discopt { ({k: round(v, 3) for k, v in d.items()}) } = {val:.5f}"
            )
        except Exception as e:
            print(
                f"  {mode:8s} brute ({a:.3f}, {c:.3f}) = {bv:.5f} | discopt ERROR {type(e).__name__}: {str(e)[:100]}"
            )

    print("\n== R4-4 symbolic (sympy) FIM vs exact ==")
    from discopt.doe.symbolic import SymbolicModel

    cases = [
        ("A*(1-exp(-k*t))", ("A", "k"), ("t",), {"A": 15.0, "k": 0.5}),
        ("V*S/(K+S)", ("V", "K"), ("S",), {"V": 2.0, "K": 0.3}),
        ("a*exp(-E/(8.314*T))", ("a", "E"), ("T",), {"a": 1e6, "E": 5e4}),
        (
            "b0 + b1*x + b2*x**2 + b12*x*z + b3*log(z)",
            ("b0", "b1", "b2", "b12", "b3"),
            ("x", "z"),
            {"b0": 1.0, "b1": -2.0, "b2": 0.5, "b12": 0.3, "b3": 1.1},
        ),
        ("A*sin(w*t + phi)", ("A", "w", "phi"), ("t",), {"A": 1.0, "w": 3.0, "phi": 0.2}),
    ]
    import jax.numpy as jnp

    for src, pn, inn, thv in cases:
        sm = SymbolicModel(src, pn, inn, measurement_error=0.1)
        rng = np.random.default_rng(1)
        designs = [{n: float(rng.uniform(0.5, 3.0)) for n in inn} for _ in range(max(len(pn), 3))]
        if "T" in inn:
            designs = [{"T": float(T)} for T in (300.0, 350.0, 400.0)]
        Fs = sm.fim(thv, designs)
        env = {"exp": jnp.exp, "log": jnp.log, "sin": jnp.sin}

        def resp(th, d, o, src=src):
            return [eval(src, {**env}, {**th, **d})]

        Fr = sum(reference_fim(resp, thv, d, 0.1) for d in designs)
        print(f"  {src:42s} rel err {rel_err(Fs, Fr):.1e}")

    print("\n== R4-5 sequential_doe bookkeeping (noise-free simulator at the true parameters) ==")
    from discopt.doe import sequential_doe

    rbm = lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d["t"]))]
    true, guess, bt = {"A": 15.0, "k": 0.5}, {"A": 10.0, "k": 1.0}, {"t": (0.1, 10.0)}
    ex = to_discopt(rbm, guess, bt, 0.1)
    sim = lambda d: {"y0": float(true["A"] * (1 - np.exp(-true["k"] * d["t"])))}
    runs0 = [{"t": 1.0, **sim({"t": 1.0})}, {"t": 3.0, **sim({"t": 3.0})}]
    for r in sequential_doe(
        ex, None, guess, bt, initial_runs=runs0, n_rounds=4, run_experiment=sim
    ):
        est = r.estimation.parameters
        F_exp = sum(reference_fim(rbm, est, {"t": run["t"]}, 0.1) for run in r.runs)
        direct = optimal_experiment(ex, est, bt, prior_fim=r.estimation.fim).design
        print(
            f"  round {r.round}: estimate { ({k: round(v, 6) for k, v in est.items()}) }, "
            f"FIM vs sum of run FIMs {rel_err(r.estimation.fim, F_exp):.1e}, "
            f"design t={r.design.design['t']:.4f} (direct {direct['t']:.4f})"
        )

    print("\n== R4-6 identifiability / estimability on a known structure (a*b, inert c) ==")
    import discopt.modeling as dm
    from discopt.doe import diagnose_identifiability, estimability_rank
    from discopt.estimate import Experiment, ExperimentModel

    class Known(Experiment):
        def create_model(self, **kw):
            m = dm.Model("id")
            p = {n: m.continuous(n, lb=0, ub=10) for n in ("a", "b", "c", "k")}
            x = m.continuous("x", lb=0, ub=5)
            ys = {
                f"y{i}": p["a"] * p["b"] * dm.exp(-p["k"] * x * s) + 0 * p["c"]
                for i, s in enumerate([0.5, 1, 2, 4])
            }
            return ExperimentModel(m, p, {"x": x}, ys, {n: 0.01 for n in ys})

    thk = {"a": 2.0, "b": 1.5, "c": 0.7, "k": 0.4}
    dg = diagnose_identifiability(Known(), thk, {"x": 1.0})
    print(
        f"  rank {dg.fim_rank} of {dg.n_parameters}; null space "
        f"{[{k: round(v, 4) for k, v in ns.items()} for ns in dg.null_space]} (expected (a,-b)/sqrt2 and c)"
    )
    er = estimability_rank(Known(), thk, {"x": 1.0})
    print(
        f"  estimability ranking {er.ranking}, projected norms {np.round(er.projected_norms, 6).tolist()} "
        "(expected: a, k estimable; b collinear with a; c inert)"
    )
