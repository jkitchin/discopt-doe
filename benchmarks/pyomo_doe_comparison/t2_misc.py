"""R2-5..7: prior FIM with pyomo's parameter scaling; full-factorial FIM grids;
extreme measurement-error heterogeneity."""

import logging
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from common import ipopt, logdet, pyomo_run_doe, reference_fim, rel_err, to_discopt, to_pyomo
from discopt.doe import compute_fim, optimal_experiment
from discopt.doe.fim import compute_fim_batch
from t_design import PROBLEMS

if __name__ == "__main__":
    print("== R2-5 prior FIM x scale_nominal_param_value (Rooney-Biegler + prior, D-opt) ==")
    p = PROBLEMS["rooney_biegler+prior"]
    th = p["theta"]
    S = np.diag([th[n] for n in th])
    # A prior whose balance between A and k decides the design, so treating it
    # as scaled or unscaled changes the answer.
    prior = np.diag([1.0, 30.0])
    bounds = {"t": (0.0, 10.0)}
    grid = np.linspace(0, 10, 2001)
    true = [logdet(reference_fim(p["resp"], th, {"t": t}, p["sigma"], prior)) for t in grid]
    print(f"  exact optimum (unscaled prior): t={grid[int(np.argmax(true))]:.4f}")
    r = optimal_experiment(
        to_discopt(p["resp"], th, bounds, p["sigma"]), th, bounds, prior_fim=prior
    )
    print(f"  discopt: t={r.design['t']:.4f}")
    for scale, pr, label in [
        (False, prior, "unscaled prior"),
        (True, prior, "SAME unscaled prior"),
        (True, S @ prior @ S, "prior scaled as S P S"),
    ]:
        res, _ = pyomo_run_doe(
            to_pyomo(p["resp"], th, {"t": 5.0}, bounds, p["sigma"], n_resp=1),
            prior_FIM=pr,
            scale_nominal_param_value=scale,
        )
        t = float(res["Experiment Design"][0])
        tl = true[int(np.argmin(np.abs(grid - t)))]
        print(
            f"  pyomo scale_nominal={scale!s:5s} + {label:22s}: t={t:.4f} (exact logdet there {tl:.4f}, "
            f"max {max(true):.4f})"
        )

    print("\n== R2-6 compute_FIM_full_factorial vs discopt compute_fim_batch (Rooney-Biegler) ==")
    from pyomo.contrib.doe import DesignOfExperiments

    ts = [1.0, 3.0, 5.0, 7.0, 9.0]
    ref = [reference_fim(p["resp"], th, {"t": t}, p["sigma"], p["prior"]) for t in ts]
    dres = compute_fim_batch(
        to_discopt(p["resp"], th, bounds, p["sigma"]),
        th,
        [{"t": t} for t in ts],
        prior_fim=p["prior"],
    )
    print(
        "  discopt batch max rel err:", f"{max(rel_err(a.fim, b) for a, b in zip(dres, ref)):.1e}"
    )
    for fd in ["central", "forward"]:
        doe = DesignOfExperiments(
            to_pyomo(p["resp"], th, {"t": 5.0}, bounds, p["sigma"], n_resp=1),
            fd_formula=fd,
            solver=ipopt(),
            logger_level=logging.ERROR,
            prior_FIM=p["prior"],
        )
        out = doe.compute_FIM_full_factorial(
            design_ranges={"d[t]": [1.0, 9.0, 5]}, method="sequential"
        )
        # log10 D-opt per grid point is reported; compare with the exact value
        got = np.asarray(out["log10 D-opt"], dtype=float)
        exact = np.array([logdet(F) / np.log(10) for F in ref])
        print(
            f"  pyomo full_factorial [{fd:7s}] log10 D-opt: max abs err {np.max(np.abs(got - exact)):.2e}"
            f"   (got {np.round(got, 3).tolist()}, exact {np.round(exact, 3).tolist()})"
        )

    print("\n== R2-7 heterogeneous measurement errors (sigma = 1e-4 and 10) ==")
    resp = lambda th, d, o: [th["a"] * o.exp(-th["b"] * d["t"]), th["a"] + th["b"] * d["t"]]
    th2, dsg = {"a": 2.0, "b": 0.4}, {"t": 1.5}
    for sig in [(1e-4, 10.0), (10.0, 1e-4)]:
        ref = reference_fim(resp, th2, dsg, np.array(sig))
        dx = compute_fim(to_discopt(resp, th2, {"t": (0, 5)}, np.array(sig)), th2, dsg).fim
        from common import pyomo_fim

        px, _ = pyomo_fim(to_pyomo(resp, th2, dsg, {"t": (0, 5)}, np.array(sig), n_resp=2))
        print(f"  sigma={sig}: discopt {rel_err(dx, ref):.1e} | pyomo {rel_err(px, ref):.1e}")
