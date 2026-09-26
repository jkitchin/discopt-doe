#!/usr/bin/env bash
# Re-run the discopt.doe vs pyomo.doe comparison and save the logs to results/.
#
# Needs Ipopt on PATH or in IPOPT_EXE (e.g. `conda install -c conda-forge ipopt`).
# pyomo is not a dependency of this package; it is added for the run only.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
for t in probe_forward t_fim t_implicit t_design t_ode t_scaling; do
    echo "== $t"
    uv run --with "pyomo>=6.10" --with pandas python "$t.py" 2>&1 \
        | grep -v -E '^WARNING|^\s*$|model.name=|termination condition:|message from solver:|Exceeded\.|Problem may be infeasible|finite elements specified|will be used\.' \
        | tee "results/$t.txt"
done
