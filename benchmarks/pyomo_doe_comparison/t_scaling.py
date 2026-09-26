"""T9: parameter scaling. D-optimal designs are invariant to rescaling the
parameters (log det changes by a constant); A-optimal designs are not."""

import warnings

warnings.filterwarnings("ignore")
from common import (
    pyomo_run_doe,
    reference_fim,
    to_discopt,
    to_pyomo,
)
import numpy as np
from t_design import PROBLEMS
from discopt.doe import optimal_experiment

p = PROBLEMS["arrhenius_2T"]
th = p["theta"]
S = np.diag([th[n] for n in th])  # scaled FIM = S F S
for crit in ["determinant", "trace"]:
    d = optimal_experiment(
        to_discopt(p["resp"], th, p["bounds"], p["sigma"]), th, p["bounds"], criterion=crit
    )
    print(f"[{crit}] discopt (raw units): {d.design}")
    for scale in [False, True]:
        res, _ = pyomo_run_doe(
            to_pyomo(p["resp"], th, {"T1": 390.0, "T2": 360.0}, p["bounds"], p["sigma"], n_resp=2),
            objective_option=crit,
            scale_nominal_param_value=scale,
        )
        dsg = dict(zip(p["bounds"], map(float, res["Experiment Design"])))
        F = reference_fim(p["resp"], th, dsg, p["sigma"])
        print(
            f"   pyomo scale_nominal={scale}: {dsg} [{res['Termination Condition']}] "
            f"reported FIM == S F S: {np.allclose(res['FIM'], S @ F @ S, rtol=1e-3)}"
        )
    # discopt equivalent of scaling: optimize the criterion of S F S
