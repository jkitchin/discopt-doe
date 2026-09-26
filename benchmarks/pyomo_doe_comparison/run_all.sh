#!/usr/bin/env bash
# Re-run the discopt.doe vs pyomo.doe comparison and save the logs to results/.
#
# Needs Ipopt on PATH or in IPOPT_EXE (e.g. `conda install -c conda-forge ipopt`).
# pyomo is not a dependency of this package; it is added for the run only.
#
# Round 2's E/ME test (t2_criteria) also needs pyomo's grey-box stack:
#   * cyipopt built against that Ipopt
#     (PKG_CONFIG_PATH=<ipopt prefix>/lib/pkgconfig uv pip install cyipopt,
#      with LD_LIBRARY_PATH=<ipopt prefix>/lib at run time), and
#   * libpynumero_ASL.so in ~/.pyomo/lib (e.g. from the IDAES idaes-ext
#     solver bundle). The conda-forge `pynumero_libraries` package is a stale
#     py37 build that downgrades Ipopt -- avoid it.
# Usage: ./run_all.sh [script ...]   (default: every script)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
scripts=("$@")
if [ ${#scripts[@]} -eq 0 ]; then
    scripts=(probe_forward t_fim t_implicit t_design t_ode t_scaling
             t2_criteria t2_constraints_batch t2_reactor t2_misc t3_ode t3_structure)
fi
for t in "${scripts[@]}"; do
    echo "== $t"
    PYTHONUNBUFFERED=1 uv run --with "pyomo>=6.10" --with pandas python "$t.py" 2>&1 \
        | grep -v -E '^WARNING|^\s*$|model.name=|termination condition:|message from solver:|Exceeded\.|Problem may be infeasible|finite elements specified|will be used\.|slow_operation_alarm|Very slow compile|^\*+$|^E0[0-9]{3} |See also https|not in domain Reals|outside the bounds' \
        | tee "results/$t.txt"
done
