"""T1-T4: FIM at fixed designs, including adversarial parameter values."""

import warnings

warnings.filterwarnings("ignore")
from common import (
    pyomo_fim,
    reference_fim,
    rel_err,
    to_discopt,
    to_pyomo,
)
import numpy as np
from discopt.doe import compute_fim

MODELS = {
    "linear": (
        lambda th, d, o: [th["a"] + th["b"] * d["x"], th["a"] + th["b"] * (-d["x"])],
        {"a": 1.0, "b": 2.0},
        {"x": (-1, 1)},
        2,
    ),
    "rooney_biegler": (
        lambda th, d, o: [th["A"] * (1 - o.exp(-th["k"] * d["t"]))],
        {"A": 15.0, "k": 0.5},
        {"t": (0.1, 10)},
        1,
    ),
    "michaelis_menten": (
        lambda th, d, o: [
            th["V"] * d["S"] / (th["K"] + d["S"]),
            th["V"] * 2 * d["S"] / (th["K"] + 2 * d["S"]),
        ],
        {"V": 2.0, "K": 0.3},
        {"S": (0.01, 5)},
        2,
    ),
    "arrhenius": (
        lambda th, d, o: [
            o.exp(th["lnA"] - th["E"] / (8.314 * d["T"])) * d["t"],
            o.exp(th["lnA"] - th["E"] / (8.314 * (d["T"] + 20))) * d["t"],
        ],
        {"lnA": 20.0, "E": 7.0e4},
        {"T": (300, 400), "t": (1, 10)},
        2,
    ),
    "bi_exponential": (
        lambda th, d, o: [
            th["a1"] * o.exp(-th["k1"] * d["t"]) + th["a2"] * o.exp(-th["k2"] * d["t"]),
            th["a1"] * o.exp(-th["k1"] * 2 * d["t"]) + th["a2"] * o.exp(-th["k2"] * 2 * d["t"]),
            th["a1"] * o.exp(-th["k1"] * 4 * d["t"]) + th["a2"] * o.exp(-th["k2"] * 4 * d["t"]),
            th["a1"] * o.exp(-th["k1"] * 8 * d["t"]) + th["a2"] * o.exp(-th["k2"] * 8 * d["t"]),
        ],
        {"a1": 1.0, "k1": 2.0, "a2": 0.5, "k2": 0.1},
        {"t": (0.05, 2)},
        4,
    ),
    "oscillator": (
        lambda th, d, o: [th["amp"] * o.sin(th["w"] * d["x"]), th["amp"] * o.cos(th["w"] * d["x"])],
        {"amp": 1.0, "w": 40.0},
        {"x": (0, 10)},
        2,
    ),
}


def run_model(name, resp, theta, bounds, n, designs, pyomo_kw=None, sigma=0.1):
    rows = []
    for dsg in designs:
        ref = reference_fim(resp, theta, dsg, sigma)
        try:
            dx = compute_fim(to_discopt(resp, theta, bounds, sigma), theta, dsg).fim
            e_d = rel_err(dx, ref)
        except Exception as e:
            e_d = f"ERROR {type(e).__name__}: {e}"[:120]
        try:
            px, _ = pyomo_fim(
                to_pyomo(resp, theta, dsg, bounds, sigma, n_resp=n), **(pyomo_kw or {})
            )
            e_p = rel_err(px, ref)
            if not np.all(np.isfinite(px)):
                e_p = f"non-finite FIM {px.tolist()}"
        except Exception as e:
            e_p = f"ERROR {type(e).__name__}: {e}"[:120]
        rows.append((name, dsg, e_d, e_p))
    return rows


def fmt(v):
    return f"{v:.2e}" if isinstance(v, float) else v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("== T1 battery: relative Frobenius error of FIM vs exact reference ==")
    for name, (resp, theta, bounds, n) in MODELS.items():
        designs = [
            {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()} for _ in range(4)
        ]
        for fd in ["central", "forward"]:
            rows = run_model(name, resp, theta, bounds, n, designs, {"fd_formula": fd})
            worst_d = max((r[2] for r in rows if isinstance(r[2], float)), default=None)
            worst_p = max((r[3] for r in rows if isinstance(r[3], float)), default=None)
            errs = [r for r in rows if not isinstance(r[3], float) or not isinstance(r[2], float)]
            print(
                f"{name:18s} pyomo[{fd:7s}] worst discopt={fmt(worst_d)} worst pyomo={fmt(worst_p)} {errs[:1]}"
            )

    print("\n== T2 zero nominal parameter (b=0) ==")
    resp, _, bounds, n = MODELS["linear"]
    for r in run_model("linear b=0", resp, {"a": 1.0, "b": 0.0}, bounds, n, [{"x": 0.7}]):
        print(r[0], "discopt:", fmt(r[2]), "| pyomo:", fmt(r[3]))
    resp, _, bounds, n = MODELS["rooney_biegler"]
    for r in run_model("RB k=1e-9", resp, {"A": 15.0, "k": 1e-9}, bounds, n, [{"t": 3.0}]):
        print(r[0], "discopt:", fmt(r[2]), "| pyomo:", fmt(r[3]))

    print("\n== T3 negative nominal parameter ==")
    resp, _, bounds, n = MODELS["rooney_biegler"]
    for r in run_model("RB k=-0.3", resp, {"A": 15.0, "k": -0.3}, bounds, n, [{"t": 3.0}]):
        print(r[0], "discopt:", fmt(r[2]), "| pyomo:", fmt(r[3]))

    print("\n== T4 nominal outside the discopt Variable bounds ==")
    resp, theta, bounds, n = MODELS["rooney_biegler"]
    th = {"A": 15.0, "k": 0.5}
    dsg = {"t": 3.0}
    ref = reference_fim(resp, th, dsg, 0.1)
    ex = to_discopt(resp, th, bounds, 0.1, param_bounds={"k": (0.0, 0.2)})
    try:
        dx = compute_fim(ex, th, dsg).fim
        print(
            "k=0.5 but Variable ub=0.2 -> discopt rel err:",
            fmt(rel_err(dx, ref)),
            "| equals FIM at k=0.2:",
            fmt(rel_err(dx, reference_fim(resp, {"A": 15.0, "k": 0.2}, dsg, 0.1))),
        )
    except ValueError as e:
        print("k=0.5 but Variable ub=0.2 -> discopt refuses:", str(e)[:90])

    print("\n== T5 pyomo FD step sensitivity (oscillator w=40, arrhenius) ==")
    for name in ["oscillator", "arrhenius", "bi_exponential"]:
        resp, theta, bounds, n = MODELS[name]
        dsg = {k: 0.5 * (lo + hi) for k, (lo, hi) in bounds.items()}
        for step in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
            r = run_model(name, resp, theta, bounds, n, [dsg], {"step": step})[0]
            print(f"{name:15s} step={step:.0e}  discopt={fmt(r[2])}  pyomo={fmt(r[3])}")
