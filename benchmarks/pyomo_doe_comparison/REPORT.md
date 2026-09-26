# discopt.doe vs pyomo.doe — adversarial comparison

**Versions:** discopt-doe 0.4.0 (+ the fixes below), discopt 0.9, Pyomo 6.10.1
(`pyomo.contrib.doe`), Ipopt 3.14.20 (MUMPS). Re-run with `./run_all.sh`; the logs
are in `results/`.

## Method

Each test problem is written **once**, as `resp(theta, design, ops)`, and built
three ways by `common.py`:

1. a discopt `Experiment` → `discopt.doe.compute_fim` / `optimal_experiment`;
2. a Pyomo.DoE experiment, following the pattern of Pyomo's own examples (fixed
   parameter Vars, one output Var and defining constraint per response, and the
   four labelling Suffixes) → `DesignOfExperiments.compute_FIM` / `run_doe`;
3. an **independent reference**: `jax.jacfwd` of the raw function, or a closed
   form (implicit-function theorem for the algebraic model, the analytic
   solution for A→B→C). Designs are judged by the reference criterion at the
   returned design, and against brute-force grids or dense random search.

So "which one is right" is never decided by one package agreeing with the other.
Both are compared to ground truth.

How the two packages work:

| | discopt.doe | pyomo.doe |
|---|---|---|
| Sensitivities | exact (JAX autodiff) | finite differences, relative step 1e-3 × θ |
| FIM | `Jᵀ Σ⁻¹ J + prior` | same |
| Design search | multistart scan + L-BFGS-B / SLSQP | one simultaneous NLP (all FD scenarios as blocks), Ipopt, single start |
| D criterion reported | ln det | log10 det (`results["log10 D-opt"]`) |
| A criterion | trace(F⁻¹), unscaled | same, optionally on the θ-scaled FIM |
| Dynamic models | RK4 / trapezoid integration, autodiff | pyomo.dae discretization + FD |

## Results at a glance

| # | Test | Agree? | Which is right / root cause |
|---|---|---|---|
| T1 | FIM, 6 explicit models, central FD | ✅ to FD accuracy | discopt exact (≤1e-15); pyomo 1e-13…1e-4 (FD truncation) |
| T1 | same, `fd_formula="forward"`/`"backward"` | ❌ | **pyomo bug**: `compute_FIM` returns a wrong Jacobian, even for a linear model |
| T5 | high-curvature model (sin(40x)) | ❌ 1–5 % | pyomo FD truncation at the default step; discopt exact |
| T2 | parameter with nominal value 0 | ❌ | **pyomo**: `ZeroDivisionError`; nominal 1e-9 → FIM silently 100 % wrong |
| T4 | nominal outside discopt Variable bounds | ❌ | **discopt bug (fixed)**: silently clipped |
| T6 | responses that are constraint-defined states | ❌ | **discopt bug (fixed)**: all-zero or wrong FIM; pyomo right |
| T7 | optimal design, 5 problems × {D, A} | mostly ✅ | two **discopt bugs (fixed)**; pyomo single-start and non-convergence issues |
| T8 | ODE A→B→C (pyomo's reactor kinetics) | ✅ when well posed | discopt RK4 more accurate; both silent on an ill-posed design (discopt now warns) |
| T9 | `scale_nominal_param_value` | differs for A only | a difference in definition, not a bug |

## Findings in discopt.doe (all fixed, with regression tests in `tests/test_fim_correctness.py`)

### D1. Constraint-defined states got an all-zero or wrong FIM (serious)

Take a response that depends on a variable fixed by an equality constraint:
`z + k z³ = x`, measure `z`. This is the normal case for equilibria, mass
balances and discretized ODEs. `compute_fim` differentiated the response with
respect to θ **holding z fixed**, so it dropped `dz/dθ`.

* A response that *is* a state: J ≡ 0, and the FIM is all zeros (log det = −∞).
* A mixed response (`a·z`, `z + k`): a plausible, **non-singular but wrong** FIM
  (13–57 % error). The D-optimal design moved from x = 2.84 to the bound x = 5.

pyomo.doe re-solves the model at each perturbed θ, so it captures the total
derivative (error 2e-7 to 2e-5).
**Fix:** add the implicit-function-theorem term
`dy/dθ = ∂y/∂θ + ∂y/∂s·(−(∂g/∂s)⁺ ∂g/∂θ)` using the equality constraints. It
raises if the constraints don't determine a state the responses depend on, and
warns if an inequality is active. discopt now matches the closed form to about
1e-9 (the solve tolerance) and finds x = 2.836.
(`discopt.modeling.implicit` blocks were already correct.)

### D2. Nominal values outside the Variable bounds were silently clipped

For `k = 0.5` with `ub = 0.2`, the FIM was computed at k = 0.2 (a 5× error),
without any message. **Fix:** `compute_fim` now raises `ValueError`.

### D3. The A criterion could prefer singular designs

`trace(np.linalg.inv(F))` doesn't raise for a numerically singular F. It returns
garbage, often a large **negative** number, and since A is minimized the
multistart picked exactly those designs. On a 4-parameter bi-exponential with
4 sampling times, discopt returned t = (15, 15, 15, 30) with criterion = ∞.
**Fix:** `trace_inverse()` goes through a Cholesky factor and scores any
non-positive-definite FIM ∞.

### D4. Refining only the best scan point missed global optima

`optimal_experiment` refined only the single best of its random candidates:

* multimodal criterion (y = amp·sin(w·x)): D = 13.46 against a global 13.70;
* bi-exponential, A criterion: after D3, it stalled at A = 1e7 against an
  optimum of 0.0091. The first L-BFGS-B step hit a singular region, and
  round-off in the seed's last digits decided whether it recovered.

**Fix:** refine the best `n_refine` (default 4) candidates, and refine the
A and ME criteria on a log scale. discopt now reaches the brute-force optimum on
all 10 problem/criterion pairs in T7.

### D5. No warning for an unidentifiable optimal design (added)

At a single constant temperature, Arrhenius A and E can't be separated, so the
FIM is exactly rank 2 of 4 (cond ≈ 1e20). Both packages returned a "D-optimal"
design that was pure round-off. `optimal_experiment` now warns when the
diagonal-normalized FIM at the optimum has condition number above 1e12.

## Findings in pyomo.doe (not fixed here — upstream issues)

### P1. `compute_FIM(method="sequential")` is wrong for forward/backward differences

For the linear model y = a ± b·x (exact J = [[1, 0.7], [1, −0.7]]), pyomo returns
`[[0, 0.2], [0, −1.2]]`. Two problems in `_sequential_FIM` cause this:

* for scenario 0 (the unperturbed base case) it `continue`s **before** solving
  and appending the outputs, so the base case is never stored;
* the column indices are off by one (`col_1 = i` should be `i + 1`).

So column 0 is always zero, and column i differences perturbation i against
perturbation 0. Errors on the other models ran from 100 % to 27 000 %.
`compute_FIM_full_factorial` calls this code path, so it's affected too. The
simultaneous path used by `run_doe` is correct. Reproducer: `probe_forward.py`.

### P2. Relative FD step fails for zero or tiny nominal values

The step is `step × θ`. For θ = 0 this raises `ZeroDivisionError`. For θ = 1e-9 the
perturbation (1e-12) is below Ipopt's tolerance: Ipopt accepts the warm start
unchanged, the sensitivity comes out as 0, and the FIM is **100 % wrong without
any error**. The same thing happens at user-chosen steps ≤ 1e-8 (bi-exponential:
100 % error at step 1e-8).

### P3. FD truncation at the default step

The error is 1e-7 to 1e-4 on smooth models, but 1.3 % (up to 5 %) for sin(40·x).
The optimal step depends on the problem: 1e-2 gives 79 % error there, and
1e-8 is ruined by P2. discopt's autodiff has no step to tune.

### P4. `run_doe` is a single local solve, and reports stale numbers when it fails

* Multimodal oscillator: the result depends on the start point, landing on
  x = 5.26, 6.30, 8.39 or 2.15 (D = 10.7–13.5 against a global 13.70).
* Arrhenius (T in K, E ≈ 7e4) and the 4-sampling-time bi-exponential: many
  starts end in `maxIterations` (e.g. A-opt from the midpoint: A = 3.8e11
  against 3.3e5), and one start ended in a solver error.
* On non-convergence, `results` still carries a FIM and "log10 D-opt" **that
  don't correspond to the returned design**. Bi-exponential: it reported
  ln det = 44.0 where the true value at the returned design is −38.8. ODE: it
  reported ln det = 8.8, 0.15 and 4.9 (nan, 8.6 and −5.0 in another run) where
  the truth is −54 to −∞. `results` is filled
  regardless of termination status.

### P5. Ill-posed designs pass silently

In the constant-T reactor problem (rank-deficient FIM), each start returns a
different design, with no warning.

## Where they agree (and are both right)

* **Rooney–Biegler** (pyomo's own DoE example, same prior): both give t = 10, with
  identical D and A values; pyomo's log10 D × ln 10 equals discopt's ln D.
* **Michaelis–Menten, two points:** both find S = {0.268, 5}. That is the textbook
  D-optimal design, S₁ = K·S_max/(2K + S_max). The A-optimal design is {0.196, 5}.
* **Arrhenius, two temperatures,** D and A (from starts where Ipopt converges).
* **Bi-exponential, D and A** (from pyomo starts that converge).
* **A→B→C with a prior** from two earlier runs (well-posed sequential DoE): both
  give T = 300 K, CA0 = 5, the grid optimum, from all 3 pyomo starts. pyomo was
  faster here (≈1 s vs ≈10 s, mostly JAX compilation).
* **Negative nominal values:** pyomo is fine (2.7e-7 error).
* **FIM accuracy for ODEs** (against the closed form): discopt RK4 with 50 steps
  gives 1e-7; pyomo collocation (Radau, 10 elements × 3 points) gives 1e-4 to
  2e-3 (discretization error dominates); backward Euler gives 2–16 %. RK4 with
  10 steps is unstable at 700 K (discopt: use `ODEExperiment.check_accuracy`).
* **Parameter scaling (T9):** with the same definition the designs agree. pyomo's
  `scale_nominal_param_value=True` leaves the D-optimal design unchanged, but
  moves the A-optimal design from (400, 380) K to (390, 356) K. A-optimality
  depends on parameter units; that's a matter of definition, not a bug. discopt
  has no scaling switch (reparameterize instead).

## Bottom line

On well-posed explicit problems where Ipopt converges, the two packages give the
same designs, and both agree with brute force. The differences all trace back to
four mechanisms:

1. **Exact vs finite-difference sensitivities.** discopt's autodiff is exact.
   pyomo's FD adds 1e-7 to a few % error, fails for zero or tiny parameters, and
   its sequential forward/backward FIM is outright wrong (P1).
2. **Which variables are differentiated.** pyomo re-solves the whole model and
   gets implicit states right. discopt dropped them (D1, fixed).
3. **Global vs local search.** discopt's multistart (after D4) finds the global
   optimum on every test problem. pyomo's single Ipopt solve depends on the start
   point and often hits max iterations on badly scaled problems.
4. **Guarding degenerate cases.** discopt had two silent failures (D2, D3) and
   now refuses or warns. pyomo reports values that don't match the returned
   design when Ipopt fails, and gives no identifiability warning.
