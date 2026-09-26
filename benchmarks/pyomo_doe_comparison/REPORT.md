# discopt.doe vs pyomo.doe — adversarial comparison

**Versions:** discopt-doe 0.4.0 (+ the fixes below, from four rounds, and the two features of round 5), discopt 0.9, Pyomo 6.10.1
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
| T9 | `scale_nominal_param_value` | differs for A only | **corrected in round 5:** pyomo's scaled A-optimal design was a solver failure, not a scaling effect |

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
* **Parameter scaling (T9), corrected in round 5:** with the same definition
  the designs agree. pyomo's `scale_nominal_param_value=True` leaves the
  D-optimal design unchanged and moved the A-optimal Arrhenius design from
  (400, 380) K to (390, 356) K. I originally called that a difference in
  definition. Round 5 checked it by brute force on the *scaled* criterion: the
  scaled A-optimum is still ≈ (380, 400) K, and pyomo's scaled run ends in
  `maxIterations` at a criterion 88× worse. So it was a solver failure, not a
  scaling effect. (Scaling does move A-optimal designs in general; for
  Michaelis–Menten S₁ goes from 0.196 to 0.223. discopt now has
  `scale_parameters=True`.)

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
   optimum on every round-1 test problem. pyomo's single Ipopt solve depends on
   the start point and often hits max iterations on badly scaled problems.
   (Round 2 found the opposite on pyomo's 10-variable reactor example; see R6.)
4. **Guarding degenerate cases.** discopt had two silent failures (D2, D3) and
   now refuses or warns. pyomo reports values that don't match the returned
   design when Ipopt fails, and gives no identifiability warning.

---

# Round 2

Round 2 covers what round 1 didn't: E- and ME-optimality (pyomo's grey-box
path), design constraints, multi-experiment designs, pyomo's own reactor example
(a 10-variable piecewise temperature profile), vector-valued parameters, prior
FIMs combined with pyomo's parameter scaling, `compute_FIM_full_factorial`, and
the other discopt code paths that differentiate responses. Scripts: `t2_*.py`;
logs: `results/t2_*.txt`.

**Extra setup:** pyomo's grey box needs `cyipopt` (built against the same Ipopt)
and `libpynumero_ASL` (from the IDAES solver bundle), described in `run_all.sh`.
Caveat: pyomo defaults to the MA57 linear solver (HSL licence), which isn't
available here, so every pyomo run uses MUMPS.

## Round-2 results at a glance

| # | Test | Outcome |
|---|---|---|
| R1 | E/ME/D/A via pyomo's grey box, 5 problems | discopt = brute force on all 20 at the default seed (after fixes D8, D9; one ME case is seed-sensitive); pyomo grey box: crashes on numpy ≥ 2.5, fails from singular starts, loose default tol, wrong "optimal" answers |
| R2 | Design constraints, 4 cases | both right on 3; **discopt wrong** on Σt ≤ 5 until fixed (D6); pyomo right |
| R3 | Multi-experiment (batch) designs | discopt greedy right; **discopt joint < greedy** until fixed (D7); pyomo has no multi-experiment design |
| R4 | Other discopt paths through constraint-defined states | **discrimination and `ParametricSurrogate` wrong** (fixed, D10) |
| R5 | Vector-valued parameter | **discopt mislabels FIM rows, crashes diagnostics** (fixed, D11) |
| R6 | pyomo's reactor example (10 design variables) | FIM: discopt more accurate (with the new `breakpoints`); **search: pyomo finds better optima** |
| R7 | `prior_FIM` + `scale_nominal_param_value` | pyomo: unscaled prior + scaling silently gives a wrong design |
| R8 | `compute_FIM_full_factorial` | round-1 bug P1 carries over with forward differences |
| R9 | Measurement errors spanning 1e-4 to 10 | both right |

## New findings in discopt.doe (all fixed; tests in `tests/test_fim_correctness.py`)

### D6. Constrained designs from infeasible seeds returned singular designs
With four sampling times in [0.05, 30] and the budget Σt ≤ 5, none of the 18
multistart candidates is feasible. SLSQP then ran once, from the best
*infeasible* candidate, and the feasible corner it reached was accepted:
t = (4.85, 0.05, 0.05, 0.05), log det = −38.9 (singular), against an optimum of
30.43. pyomo got 30.43.
**Fix:** each infeasible candidate is pulled toward an interior max-slack point
(bisection along the segment), or projected when there are equality
constraints, and the best `n_refine` feasible seeds are refined. A plain
nearest-point projection wasn't enough: it pushes seeds onto the bounds, giving
duplicate sampling times (log det −1.3). discopt now reaches 30.4335.

### D7. The "joint" batch strategy could lose to "greedy"
Michaelis–Menten with 4 runs: joint 13.75 < greedy 14.04 (= the joint optimum:
two runs at 0.268, two at 5). The joint search refined only its best random
start. **Fix:** it's seeded with the greedy batch and refines its best few
starts, so it's never worse than greedy.

### D8. E-optimal refinement stalled on small eigenvalues
For Arrhenius (E in J/mol), λ_min ≈ 3e-6, so L-BFGS-B's absolute gradient
tolerance declared convergence at the first step: E = 6.3e-7 against 3.0e-6.
**Fix:** E is refined on a log scale, like A and ME.

### D9. E/ME searches didn't leave nearly singular starts
Bi-exponential sampling times: E = 1e-7 against 120, ME = 8e11 against 335.
These criteria are nonsmooth, and the random candidates were all nearly
singular. **Fix:** the D-optimal design is added to the E/ME candidates.
Checking several seeds then exposed a second problem: from the D-optimal seed,
L-BFGS-B's finite-difference line search stopped after one iteration for some
seeds and not others (round-off in the start point decides which of two nearly
equal eigenvalues is the minimum). E/ME are now refined on a smooth
soft-min/soft-max of the log-eigenvalues at shrinking temperatures, followed by
a Nelder–Mead polish. Across 6 seeds, 9 of the 10 E/ME problem/criterion pairs
reach the brute-force optimum every time. The exception is ME on the
multimodal sine model, where 10 random candidates sometimes miss a narrow basin:
5 of 10 seeds at the default `n_starts=10`, 9 of 10 at 40. The default seed
finds it.

### D10. The constraint-state bug (round-1 D1) also lived in two other modules
* **Discrimination** (`_predict_with_covariance`) used the partial Jacobian for
  constrained models. It now shares `compute_fim`'s total-sensitivity helper.
* **`ParametricSurrogate`** built x* with every non-parameter, non-design
  variable **set to 0**, so predictions and Jacobians of a constrained model
  were silently wrong. It now refuses such models, as campaigns already did.

### D11. Vector-valued parameters got one label for several FIM rows
A parameter declared as a single size-3 `Variable` gave a correct 3×3 FIM but
`parameter_names == ['k']`. `diagnose_identifiability` crashed ("negative
dimensions are not allowed"), and anything that pairs names with rows
mislabelled them. Now `k[0], k[1], k[2]`.

### Added (from R6)
* `ode_experiment(breakpoints=[...])`: integrate piecewise across input jumps.
  Without it, RK4 converges only at first order across a temperature step (see
  R6). All segments run in one `lax.scan`. A first version with one scan per
  segment made each FIM ~17× slower (13 s → 210 s); the rewrite restores the
  original time.
* `optimal_experiment(initial_designs=[...])`: add known designs (the current
  operating point, a previous round's design) to the multistart. pyomo always
  starts from the experiment's design; discopt previously couldn't.

## New findings in pyomo.doe

### P6. The grey-box E and ME objectives crash on numpy ≥ 2.5
From numpy 2.5 on, `np.linalg.eig` returns complex arrays even when every
eigenvalue is real (2.2 and 2.4 return float64). The grey box uses `eig`, its
Jacobian becomes complex, and pynumero refuses it ("Multiple dtypes found:
complex128, float64"). `results["log10 E-opt"]` also becomes complex. The fix
is `eigh` for the symmetric FIM. discopt-doe uses only `eigh`. (For the
comparison below, `t2_criteria.py` shims `eig` inside that one module.)

### P7. Grey box: singular starting designs crash with an opaque error
The natural start of a symmetric midpoint (S₁ = S₂) gives a rank-1 FIM. The
grey box then initializes log det = log 0 and fails with "NaN values found in
initialization of primals" (D objective) or `LinAlgError: Singular matrix`
(E objective). The non-grey-box path regularizes this case; the grey-box path
doesn't.

### P8. Grey box: the default tolerance is too loose, and "optimal" answers can be wrong
* The default `tol=1e-4` stops the A-optimal Michaelis–Menten design at
  S = 4.90 instead of the bound 5 (0.7% worse). With `tol=1e-8` it's exact.
* E-optimal Arrhenius: **every start returns the rank-1 design T₁ = T₂ = 350
  (λ_min = 0, the worst possible) as `optimal`.** E values (~1e-6) are below the
  tolerance.
* ME-optimal Arrhenius: `maxIterations` or `solverFailure` from every start,
  with results up to 1,000× worse than the optimum.
* D-optimal Michaelis–Menten: one start reports `optimal` at log det −0.70
  (true optimum 12.655). One E start reports `optimal` at 0.128 (optimum 240).
* The A objective is `trace(pinv(FIM))`. For a singular FIM, pinv sums only the
  nonzero directions, so singular designs can outscore full-rank ones (the same
  flaw as discopt's round-1 D3). The D objective uses log|det| and ignores
  the sign.

**E/ME/D/A head to head** (`t2_criteria.py`). discopt is one call at the default
settings. For pyomo's grey box, the count is runs (4 starts, default `tol=1e-4`)
that end within 0.1% of the brute-force optimum. Failures include crashes,
`maxIterations`/`solverFailure`, local optima, and the loose tolerance. The
Michaelis–Menten and Rooney–Biegler A misses are tolerance effects (0.7% and
0.6% short, exact at `tol=1e-8`). The other A misses are local optima (sine
model) or `maxIterations` (Arrhenius, bi-exponential).

| problem | E: discopt / pyomo-gb | ME | D | A |
|---|---|---|---|---|
| Michaelis–Menten, 2 points | ✓ / 2 of 4 | ✓ / 3 of 4 | ✓ / 2 of 4 | ✓ / 0 of 4 |
| Arrhenius, 2 temperatures | ✓ / 0 of 4 (rank-1 design reported optimal) | ✓ / 1 of 4 (via solverFailure) | ✓ / 3 of 4 | ✓ / 0 of 4 (maxIterations) |
| sine model + prior (multimodal) | ✓ / 3 of 4 | ✓ / 1 of 4 | ✓ / 0 of 4 | ✓ / 0 of 4 |
| bi-exponential, 4 times | ✓ / 1 of 4 | ✓ / 1 of 4 | ✓ / 0 of 4 | ✓ / 0 of 4 |
| Rooney–Biegler + prior | ✓ / 4 of 4 | ✓ / 4 of 4 | ✓ / 3 of 4 | ✓ / 0 of 4 |
| **total** | **discopt 20 of 20** | | | **pyomo grey box 28 of 80 runs** |

For comparison, pyomo's standard (non-grey-box) D and A path does much better
on the same problems (round 1, T7). The grey box is what makes E/ME available,
and it's the least robust part.

### P9. No multi-experiment design
`run_multi_doe_sequential` and `run_multi_doe_simultaneous` raise
`NotImplementedError`. The obvious workaround, repeated `run_doe` with the
accumulated `prior_FIM`, breaks down whenever one experiment alone can't
identify every parameter (one measurement, two parameters). MM: first pick
S = 2.70, joint log det 9.18 against 12.655. RB and exp+offset: solver error
at the first pick. discopt's greedy and joint strategies reach the brute-force
joint optimum on all 4 cases.

### P10. `prior_FIM` and `scale_nominal_param_value` interact silently
With scaling on, pyomo assumes the prior is **already scaled** (documented).
Passing the same unscaled prior you'd pass without scaling moves the
Rooney–Biegler design from t = 2.12 to t = 10 (log det 8.04 against 9.51),
with no warning. With the prior scaled as S·P·S, the result is correct again.

### P11. `compute_FIM_full_factorial` inherits P1
With `fd_formula="forward"`, the factorial grid of log10 D-opt comes out
monotone (6.61 → 7.52), where the truth dips to 6.68 at t = 3 and rises to 6.85.
A map like that points you to the wrong designs. With central differences it's
correct (error 2e-7).

## R6 in detail: pyomo's reactor example (`t2_reactor.py`)

pyomo's shipped `ReactorExperiment` (data file `reactor_result.json` fetched from
Pyomo 6.10.1; the wheel doesn't include it), with 10 design variables:
T at 9 control times, plus CA0. The reference is exact (matrix exponential per
constant-T interval).

**FIM accuracy at the example's design** (relative error; exact log det 14.0614):

| | error |
|---|---|
| discopt RK4, 50 / 200 / 1000 steps over the whole run | 3.2e-3 / 1.6e-3 / 2.8e-4 (first order: steps straddle the jumps) |
| discopt RK4 with `breakpoints`, 5 / 10 / 50 steps per segment | 3.4e-4 / 1.7e-5 / 2.2e-8 |
| pyomo collocation, nfe = 10 (the example's setting) / 20 / 40 | 2.2e-2 / 1.1e-2 / 5.4e-3 (also first order) |

pyomo's example FIM is 2% off at its shipped settings. That shifts its optimum
by ~9 K (T₀ = 481.9 vs 472.9 exact in that basin).

**D-optimal design** (exact log det of each returned design; best known
**14.4999**, T₁ ≈ 490 K, others ≈ 300 K, CA0 = 5):

| run | exact log det | time |
|---|---|---|
| example's starting design (T₀ = 500, rest 300, CA0 = 5) | 14.0614 | — |
| discopt, random starts (`n_starts` 10 / 40) | 13.900 / 13.969 | ~2 min |
| discopt, `initial_designs=[example design]` | 14.177 (T₀ = 472.9) | ~1.5 min |
| discopt, `initial_designs=[best design seen]` | 14.4999 (keeps it) | ~1.5 min |
| pyomo from the example's design, `scale_nominal` on / off | 14.164 (T₀ = 481.9) | 1–5 s |
| pyomo from random starts, 9 runs over 3 executions | 14.50 ×2, 14.16 ×5, 14.09 ×2 | 1–20 s |

* **pyomo searches this problem better.** From random starts it lands at
  14.09–14.50 in seconds (reaching the best-known 14.50 in 2 of 9 runs); discopt
  lands at 13.90–13.97 in about 2 minutes (10 or 40 starts). This isn't the gradient: from pyomo's *exact* starting points,
  L-BFGS-B with exact gradients on the exact criterion reaches 13.57 / 13.57 /
  14.10. Across 20 random starts, neither L-BFGS-B, SLSQP nor scipy's interior
  point `trust-constr` reached 14.49 (best 13.68 / 13.76 / 14.08). Ipopt on
  pyomo's lifted Cholesky formulation takes a better path on this landscape.
  discopt only reaches the good basins through `initial_designs`.
* **Ipopt isn't reproducible here.** The identical random-start runs (same
  seeds) gave 14.498 / 14.498 / 14.164, 14.088 / 14.164 / 14.164, and
  14.164 / 14.096 / 14.164 in three executions. The example's own `scale_nominal_param_value=True` run hit
  `maxIterations` after 117 s once, and converged in 2–5 s in other runs.

## Round-2 bottom line

* **Correctness:** round 2 found **six more discopt bugs** (D6, D7, D8, D9, D10,
  D11): wrong answers in the constrained, batch-joint, E/ME, discrimination,
  surrogate and vector-parameter paths. All are fixed and all are covered by
  regression tests. After the fixes, discopt matches brute force on every
  small-to-medium problem tested: 30 cases (10 D/A and 10 E/ME
  problem/criterion pairs, 4 constrained designs, 4 batch designs, the
  implicit-state design and the well-posed ODE design).
* **pyomo's grey-box path (E/ME) is the weakest part of pyomo.doe** in this
  environment. It crashes on current numpy, crashes on singular starts, has a
  loose default tolerance, and returns wrong designs labelled "optimal". Its
  `prior_FIM`-with-scaling convention is a silent trap, and multi-experiment
  design is unimplemented.
* **Where pyomo is better:** global search on its large dynamic flagship
  problem, and speed on dynamic models (seconds vs minutes, because discopt pays
  JAX compile time). A lifted or interior-point refinement in discopt, instead of
  L-BFGS-B on the reduced criterion, is the natural follow-up.

---

# Round 3

Round 3 covers ODE features (stiff kinetics, unknown initial conditions,
parameters in the measurement function) and structural variants (model
`Parameter`s, vector-valued design inputs, a 6-parameter design, malformed prior
FIMs). Scripts: `t3_ode.py`, `t3_structure.py`; logs: `results/t3_*.txt`.

| # | Test | Outcome |
|---|---|---|
| R3-1 | Stiff A→B→C (k₁ = 500) | **discopt: RK4 overflow gave an infinite FIM silently, and `check_accuracy` passed it** (fixed, D12). Trapezoid: 1.5e-6. pyomo: stuck at 6e-5 whatever the mesh (FD truncation) |
| R3-2 | Unknown initial condition as a parameter | **discopt couldn't express it** (added); now 8.6e-9. pyomo 1.1e-5 |
| R3-3 | Response factor in the measurement function | both right (3.2e-8 / 2.2e-5) |
| R3-4 | Fixed model `Parameter` in a response | both right (exact / 7.3e-7) |
| R3-5 | Vector-valued design input | **discopt set all entries to one value** (now refused, D14) |
| R3-6 | 6 parameters, 6 sampling times | **discopt −198 vs 42.87** (fixed, D13); pyomo reached 42.87 in 1 of 9 runs over 3 executions (others `maxIterations`, a solver error, the `_parent` crash) |
| R3-7 | Malformed `prior_fim` | **discopt accepted or misreported them** (now validated, D15); pyomo validates all four |

## New findings in discopt.doe (fixed; tests in `tests/test_fim_correctness.py`)

### D12. A blown-up ODE integration passed silently
Explicit RK4 on k₁ = 500 with 50 or 400 steps overflows (k₁h beyond RK4's
stability limit ≈ 2.8). `compute_fim` returned an infinite FIM without comment,
and `ODEExperiment.check_accuracy`, the tool meant to catch exactly this,
compared nans (every comparison false) and reported success. It now returns
`inf` and warns; `compute_fim` warns about any non-finite FIM.

### D13. Uniform multistart candidates missed fast modes
Tri-exponential with rate constants 5, 0.8, 0.05 and six sampling times in
[0.01, 60]: every candidate had all its times above 7, where the fast modes
have decayed to e⁻³⁵. Every FIM was numerically singular and flat, the
refinement couldn't move, and the result was log det −198 (reference 42.87 from
200 exact-gradient starts). Inputs spanning ≥ 2 decades now also get
log-uniform candidates. discopt now reaches 42.87 for every seed tried, and all
round-1/2 optima still hold (re-checked under three seeds).

### D14. A vector-valued design input was optimized as one scalar
`compute_fim` handles a size-3 design `Variable` correctly. The searches,
though, treat every design name as one scalar, so all three sampling times came
back equal (`{'t': 7.0}`), a degenerate design. Now refused with guidance to
declare scalar inputs.

### D15. Malformed prior FIMs
A 3×3 prior for 2 parameters or a NaN entry: "No feasible design point found".
A non-symmetric or indefinite prior: silently used. Now `ValueError`s naming
the problem, in both `compute_fim` and the design searches, matching pyomo's
checks.

### Added
* `ode_experiment` initial values may name an unknown parameter (R3-2).

## New findings in pyomo.doe

* **P12.** On stiff kinetics the FIM error plateaus at 6e-5 (10 or 40 finite
  elements alike). The floor is the central-difference truncation at step 1e-3
  where k₁t is large, not the discretization; refining the mesh can't fix it.
* **P13.** On the 6-parameter design, `run_doe` reached the optimum in 1 of 9
  random-start runs over three executions. The others hit `maxIterations`
  (results as low as log det −238), a solver error, or the intermittent
  `'NoneType' object has no attribute '_parent'` crash while building the
  Jacobian constraints, which reappeared here after round 2.

---

# Round 4 (no significant new differences — the loop stopped here)

Areas with a direct pyomo.doe counterpart were covered by rounds 1–3 (FIM
computation in every form pyomo offers, all six objectives including the grey
box, priors, scaling, full-factorial grids, constraints, dynamic models, and
pyomo's own examples). Round 4 therefore tested discopt.doe features pyomo
lacks, against exact or brute-force references (`t4_discopt_only.py`,
`results/t4_discopt_only.txt`):

| # | Test | Result |
|---|---|---|
| R4-1 | Model discrimination (first- vs second-order decay), Hunter–Reiner / Buzzi-Ferraris / Box–Hill / Jensen–Rényi | optima match a 401-point grid; HR matches its closed form to 4e-14. JR saturates near ln 2, so its maximum is a plateau (discopt's t = 5.84 scores 0.6930 against the grid's 0.69301) |
| R4-2 | I-optimal design (Michaelis–Menten, 2 points) | matches brute force (I = 0.00193697) |
| R4-3 | Robust designs over 4 parameter samples, expected log det and maximin efficiency | expected: matches brute force; maximin: discopt 0.3424 vs 0.3355 on a 160×160 grid (discopt finer) |
| R4-4 | Symbolic (sympy) FIM path, 5 models incl. 5 parameters and log/sin terms | exact to ≤ 8e-14 |
| R4-5 | `sequential_doe` with a noise-free simulator | estimates recover the truth; each round's FIM = sum of its runs' FIMs (1e-16); each design = a direct `optimal_experiment` with that prior |
| R4-6 | Identifiability / estimability on a known structure (only a·b identifiable, c inert) | rank 2 of 4; null space exactly (a, −b)/√2 and c; estimability ranks a, k estimable, b collinear, c inert |

No bugs found, so the loop's stopping condition (a round without significant
new differences) was met.

## Overall summary (rounds 1–4)

* **discopt.doe: 15 bugs found and fixed** (D1–D4 and D6–D15; D10 covers two
  modules), all with regression tests in `tests/test_fim_correctness.py`, plus
  a singular-design warning (D5) and three features the comparison showed were
  missing: `initial_designs`, ODE `breakpoints`, and a parameter as an ODE
  initial condition. The
  serious ones were wrong answers without warning: FIMs of constraint-defined
  states (all zero or wrong), the same bug in discrimination and the surrogate,
  negative A-criteria for singular FIMs, singular designs from constrained or
  multi-scale problems, collapsed vector design inputs, silently clipped nominal
  values, silent ODE blow-ups, and search failures on multimodal or
  ill-scaled criteria.
* **pyomo.doe: 13 findings** (P1–P13): wrong forward/backward-difference FIMs
  (also in the full-factorial grid), failures for zero or tiny parameters, FD
  truncation (1–5% on oscillatory models, a 6e-5 floor on stiff ones), results
  that don't match the returned design when Ipopt fails, a grey-box path that
  crashes on numpy ≥ 2.5 or from singular starts and mislabels bad answers as
  optimal, no multi-experiment design, and a silent prior-scaling trap.
* **Where pyomo remains better:** global search on its large dynamic example
  (R6) and speed on dynamic models. Improving discopt's search there is the
  open follow-up.

---

# Round 5: closing the two feature gaps

The comparison left discopt.doe without two things pyomo.doe has: **constraints
on what the experiment does** (bounds on predicted outputs, not just on its
settings) and a **relative-parameter (nominal-value-scaled) FIM** for the A-,
E- and ME-criteria. Both are now in discopt.doe and were checked against brute
force and pyomo (`t5_new_features.py`, `results/t5_new_features.txt`).

**New API**
* `optimal_experiment(..., response_bounds={"y@5": (lo, hi)})` and the same on
  `batch_optimal_experiment` (applied to every experiment of the batch): bounds on
  predicted responses at the nominal parameters. `discopt.doe.predict_responses`
  exposes the predictions for general output constraints, including quantities
  that aren't measured (predict with a second experiment that returns them).
* `optimal_experiment(..., scale_parameters=True)` (and on the batch): the
  criterion is evaluated on S·F·S, S = diag(|θ_nominal|). Unlike pyomo, the
  `prior_fim` stays in unscaled units and is scaled with the FIM, which removes
  the trap of P10. The returned FIM is unscaled; `criterion_value` is the scaled
  criterion. A zero nominal value is refused. `ParameterScaledExperiment` is the
  underlying wrapper.

## R5-1 scaled criteria (brute force on the scaled criterion; pyomo with its prior pre-scaled as S·P·S)

| problem | criterion | brute force | discopt `scale_parameters` | pyomo `scale_nominal_param_value` |
|---|---|---|---|---|
| Michaelis–Menten, 2 points | A | 0.0163983 (S₁ ≈ 0.23) | **0.0163939** (unscaled design: 0.0165505) | 0.0163939 |
| | E | 63.1699 | **63.1997** (unscaled design: 58.94) | crash (singular start) |
| | D | 11.6328 | 11.6333 (same design as unscaled) | 11.6333 |
| Arrhenius, 2 temperatures | A | 1.43049e-4 | 1.43047e-4 | 0.0126 (`maxIterations`, 88× worse) |
| | E | 6991.04 | 6991.12 | 0 (rank-deficient design reported optimal) |
| | D | 27.4272 | 27.4272 | 27.4272 |
| Rooney–Biegler + prior | A / E / D | t = 10 | t = 10 (all three) | t = 10 (all three) |

discopt matches brute force everywhere (slightly better where the grid is
coarse). Scaling changes the A- and E-optimal Michaelis–Menten designs and
leaves the D-optimal ones alone, as theory says it should.

## R5-2 constraints on predicted outputs

| case | brute force (feasible set) | discopt `response_bounds` | pyomo (constraint on the output variable) |
|---|---|---|---|
| Rooney–Biegler + prior, y ≤ 12 | t = 1.5625, 15.49202 (bound inactive) | t = 1.5617, 15.49202 | t = 1.5617 |
| Rooney–Biegler + prior, y ≤ 8 | t = 1.5225, 15.49183 | t = 1.5243, **y = 8.0000**, 15.49185 | t = 1.5220, y = 7.9920 |
| A→B→C (T, CA0), by-product cap CC@1 ≤ 1.0 | 7.86286 | CC = 1.0000, **7.87498** | infeasible start: **fails**; feasible start: CC = 0.9959, 7.86587 |
| same, CC@1 ≤ 0.6 | 6.97199 | CC = 0.6000, **6.99083** | infeasible start: **fails**; feasible start: CC = 0.5975, 6.98560 |

* discopt meets each bound exactly and matches or beats the brute-force grid.
* **P14. pyomo can't start from a design that violates an output constraint.**
  Its initial square solve fixes the design, so the model is infeasible
  ("Model from experiment did not solve appropriately"). discopt's constrained
  seeding (round 2, D6) handles this.
* **P15. pyomo's output constraints are slightly conservative.** The constraint
  is cloned into every finite-difference scenario block, so the *perturbed*
  parameter scenarios must satisfy it too. The design stops short of the bound
  (y = 7.992 for a bound of 8; CC = 0.9959 for a cap of 1.0), losing a little
  information.

## Remaining differences

Of the gaps listed after round 4, the two significant ones are closed. What's
left: DAEs with algebraic states inside `ode_experiment` (a DAE written as
constraints works but needs a solve per FIM), the pseudo-trace objective, JSON
result files, and pyomo's stronger global search on its large dynamic example
(R6; follow-up task).
