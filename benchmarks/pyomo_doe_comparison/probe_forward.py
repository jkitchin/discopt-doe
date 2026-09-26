import warnings

warnings.filterwarnings("ignore")
from common import (
    ipopt,
    pyomo_fim,
    reference_jacobian,
    to_pyomo,
)
import numpy as np

resp = lambda th, d, o: [th["a"] + th["b"] * d["x"], th["a"] - th["b"] * d["x"]]
theta = {"a": 1.0, "b": 2.0}
dsg = {"x": 0.7}
print("reference J:\n", reference_jacobian(resp, theta, dsg))
for fd in ["central", "forward", "backward"]:
    F, doe = pyomo_fim(to_pyomo(resp, theta, dsg, {"x": (-1, 1)}, 0.1, n_resp=2), fd_formula=fd)
    print(fd, "sequential J:\n", doe.seq_jac)
# Is the simultaneous (run_doe / create_doe_model) path also affected?  Solve square with design fixed.
from pyomo.contrib.doe import DesignOfExperiments
import pyomo.environ as pyo
import logging

for fd in ["forward", "backward"]:
    doe = DesignOfExperiments(
        experiment=to_pyomo(resp, theta, dsg, {"x": (-1, 1)}, 0.1, n_resp=2),
        fd_formula=fd,
        objective_option="zero",
        solver=ipopt(),
        logger_level=logging.ERROR,
    )
    doe.create_doe_model()
    m = doe.model
    for c in m.scenario_blocks[0].experiment_inputs:
        c.fix()
    m.o = pyo.Objective(expr=0)
    ipopt().solve(m)
    print(fd, "simultaneous-model J:\n", np.array(doe.get_sensitivity_matrix()))
