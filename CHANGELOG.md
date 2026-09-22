# Changelog

All notable changes to `discopt-doe` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Model-based design is fast.** `compute_fim` rebuilt and recompiled the model on
  every call, about a second for an ODE model. It now caches a compiled Jacobian per
  experiment, and `clear_fim_cache` resets it. A three-batch joint design for a
  four-parameter reactor dropped from 358 s to 0.2 s, and a constrained greedy
  batch that did not finish in ten minutes now takes 1.4 s, with the same designs.
- **The GP preset no longer reads a small design as pure noise.** With about 8
  runs, "everything is noise" was the exact maximum-likelihood fit, so batches
  degenerated into exploration.
  - Hyperparameters are now fit by MAP with weak log-normal priors on the
    length-scales and noise (after Hvarfner et al. 2024).
  - `ard="auto"` compares log posteriors, which no longer under-selects per-factor
    length-scales in high dimension. BO regret in 12-D fell from 0.74 to 0.18.
  - Unreplicated coverage is 0.97, where it was 0.92.
  - The cost: when the response is mostly noise, intervals run
    overconfident. A few replicated runs, which give a pure-error noise
    estimate, fix it.
  - `priors=False` restores plain maximum likelihood.
- `discriminate_design` builds and jits each model once per call, and sequential
  discrimination is about 3× faster.
- `explore_design_space` reports and warns about points it could not evaluate,
  where it used to leave NaN silently.
- **`anova_report` no longer reports wrong sums of squares when an
  interaction is partly confounded with a block.** The decomposition used
  marginal (cell-mean) sums of squares, which are exact only for orthogonal
  terms, and the orthogonality check compared main effects pairwise. A day ×
  catalyst interaction in a Latin square replicated with a fresh randomization
  passes that check but is partly confounded with the operator block, so the
  table was silently wrong (SS 8.26 where a joint least-squares fit gives
  6.68), still marked `balanced`. Sums of squares are now computed by
  sequential least squares (Type I, factors then interactions in the order
  given). That is identical for orthogonal designs and correct for the rest.
  A term with fewer estimable degrees of freedom than its nominal count is
  refused as aliased, with a message that says what to do. A term that is
  estimable but not orthogonal is reported with a warning and `balanced=False`.
  The data-dependent "negative residual" guard is gone: it caught this case only
  when the data happened to push the residual below zero.
- **Batch designs no longer waste their first runs.** `linear_batch_design`,
  `batch_design_from_basis`, and the greedy strategy of `batch_optimal_experiment`
  picked the first run arbitrarily: until the accumulated FIM is full rank every
  candidate scores `log det = -inf`, and the E-criterion fallback is flat too,
  since `λ_min = 0`. Greedy never revisited those picks, so a 6-run straight-line
  design put one run at x = 6.97 instead of at an end. Rank-deficient rounds now
  maximize `log det(FIM + εI)`, with ε a tiny multiple of the typical single-run
  information, and the batch is polished by exchange sweeps (new
  `exchange_passes`, default 2; 0 restores pure greedy). The results match the
  known D-optimal designs: 3 + 3 at the ends for a line, 2 + 2 + 2 at the ends
  and midpoint for a quadratic, and the 3² factorial for a 9-run two-factor
  quadratic.
  The ridge is scaled per parameter, and the rank test runs on the
  correlation-scaled FIM, so parameters on wildly different scales (an
  Arrhenius k0 ~ 1e9 next to Ea ~ 6e4) get the same design as a well-scaled
  model instead of collapsing every run onto one point.
- **Flat optimal-design criteria converge.** The linear-design search now uses
  tight L-BFGS-B/SLSQP tolerances. The default ones stopped a quadratic's center
  points at 4.86 and 5.05 instead of 5.0.
- **`fit_least_squares` reports a FIM consistent with its standard errors.**
  `fim` was `JᵀJ/σ²` with the *declared* measurement error, while the standard
  errors used the residual estimate, so `inv(fim)` disagreed with the reported
  covariance by `σ̂/σ` (0.092 vs 0.020 in a simple line fit). `fim` now uses
  the same σ. The result also carries `covariance`, `sigma` and `sigma_source`
  (`"residual"` or `"declared"`). The CLI rescales before writing the
  workbook's FIM, so `discopt doe extend` still adds new runs on the declared-σ
  scale and behaves as before.
- **`anova_report` analyzes 2-level factorials with centre points
  correctly.** Centre runs were treated as a third level of every factor. That
  gave each factor two df, aliased every factor's "centre" contrast with every
  other's, and made `discopt doe anova` fail on any `factorial-2level` workbook
  with `--center-points` once an interaction was requested. Now each factor is
  coded -1/+1 with the centres at 0 (one df each), and the centre runs add a
  one-df `curvature` term, the classical test for curvature, plus pure error.
  The results match an equivalent least-squares fit exactly. Centre rows are
  recognized by the `is_center` flag or, for data read back from a workbook, as
  the runs where every numeric factor sits at the midpoint of its two levels.
- **`project_to_simplex(..., bounds=)`** is now the exact Euclidean projection
  onto the bounded simplex. It used to clip after projecting, which could leave
  the sum off the total. **`sample_simplex(..., bounds=)`** now samples the
  bounded region uniformly, where clip-and-rescale could push a component back
  outside its bounds. Infeasible bounds now raise instead of producing a design
  that violates them. The Scheffé templates check the bounds up front with a
  clear message.
- `template_parameter_names` accepts `n_inputs` positionally.
- `anova_report` raises a `ValueError` naming a missing column instead of a
  bare `KeyError`.
- **`AnovaTable.summary()`** prints `---` for the Total row's mean square
  instead of a misleading `0.0000`.

### Added
- **Campaigns: runs with conditions.**
  - `campaign_experiment(model, runs)` turns a per-run model (a `SymbolicModel`,
    an `ODEExperiment` or an `Experiment` with design inputs) plus a list of run
    conditions into one experiment, so the FIM, identifiability,
    estimability and profile-likelihood tools take a whole campaign.
  - `fit_campaign` fits one with exact Jacobians and calibrated SEs.
  - `symbolic_experiment` wraps a `SymbolicModel` as an `Experiment`.
  - `sequential_doe` accepts runs that carry their conditions, so models
    with design inputs, including ODE models, work round by round.
  - `ParametricSurrogate.from_symbolic` builds the mechanistic BO surrogate
    from the same equation.
- **Robust designs.** `robust_optimal_experiment(..., robust="expected"|"maximin")`
  gives pseudo-Bayesian and max-min D-optimal designs over a sample of
  parameter values. `design_efficiencies` scores designs over that sample.
- **Identifiability and estimability.**
  - `profile_likelihood(..., expression="k*K")` or `function=` profiles a
    derived quantity, which can be identifiable when its parts are not.
  - Multi-start estimation (`n_starts=`) warns when distinct optima tie,
    as mirror solutions do.
  - `EstimabilityResult.raw_norms`, and `mse_subset_selection` /
    `estimability_rank(method="mse")`, choose how many parameters to
    estimate by the mean-squared-error criterion of Wu, McAuley & Harris
    (2011).
  - Correlation matrices carry parameter names: `correlation_frame()` and
    `correlation(a, b)`.
  - `quiet_solver()` silences base-discopt solver logging, and profiles and
    estimation use it by default.
- **Discrimination.**
  - `discriminate_compound` and `evaluate_discrimination_criterion` accept
    `prior_fims`, so designs account for data already collected. With data
    in hand, the compound design at weight 0 is now the D-optimal design
    given the data, and at weight 1 the Buzzi-Ferraris design.
  - A Box–Hill (1967) criterion (`DiscriminationCriterion.BH`).
  - `likelihood_ratio_test(..., boundary=)` applies the chi-bar-squared
    correction of Self & Liang (1987) for parameters on a bound.
- **ODE experiments.**
  - Public `jacobian`, `fim` and `response_function`.
  - `check_accuracy` compares against twice the steps.
  - `simulate` asks only for the inputs it uses.
- **`fit_least_squares`.** `sigma=` gives known-noise standard errors, and it
  warns, naming the parameters involved, when the Jacobian is rank-deficient.
  So does `check_jacobian_rank`.
- **Dynamic (ODE) experiments.** `ode_experiment(rhs, states=, parameters=,
  measured=, sample_times=, design_inputs=)` builds an `ODEExperiment`. Its
  sampling times, initial conditions and inputs (such as temperature) can be
  design variables. `compute_fim`, `optimal_experiment`,
  `batch_optimal_experiment`, the identifiability and estimability
  diagnostics, `profile_likelihood` and model discrimination all work on it.
  The ODE is integrated in JAX (RK4, or implicit trapezoid for stiff systems)
  inside one differentiable node, so the sensitivities are exact. The FIMs
  match the analytic results for A→B and A→B→C, and the optimal single
  sampling time for a first-order rate constant comes out at t = 1/k.
- **I- and G-optimal design, and candidate-set exchange.** `linear_optimal_design`,
  `linear_batch_design` and `batch_design_from_basis` accept `criterion="I"`
  (average prediction variance over a region) and `"G"` (maximum prediction
  variance) with a `region=` or `region_bounds=`. `design_region` builds the
  region's moment matrix, exactly by Gauss–Legendre quadrature on boxes of up
  to 4 factors. `candidate_exchange_design` runs a Fedorov exchange over a
  finite candidate list, for irregular regions, categorical factors, or
  choosing from the runs that are actually available. `relative_efficiency`,
  `d_efficiency` and `i_efficiency` compare designs. The textbook letters
  ("D", "A", "I", "G", ...) are accepted as criterion names.
- **Bayesian-optimization surrogates you can trust.** `gp_surrogate(noise=,
  ard=, ...)` is the new `"gp"` preset:
  - It estimates noise from exact replicates when there are any. Otherwise it
    fits the noise with a floor of 10% of the response SD and warns when the
    fit sits on that floor. `noise=` fixes a known noise, and `noise=0` suits a
    deterministic simulator.
  - It chooses one length-scale per input (ARD) or a shared one by BIC.
  - Its length-scale floor is sensible, and `seed` makes it reproducible.
  - `predict_latent` gives the uncertainty of the mean. EI and UCB use it by
    default, so EI decays as the optimum is pinned down instead of staying
    inflated by measurement noise.

  The old preset let the noise collapse to ~0 on unreplicated data. Its 95%
  intervals then covered ~79% of held-out points; they now cover ~91%.
- `optimize_round` also gains:
  - `candidates=` or a `candidate_sampler` callable, for known constraints
    and mixtures;
  - `infeasible_runs=`/`feasibility_column=`, where a classifier's
    P(feasible) steers away from runs that failed;
  - a `max_variance` (pure exploration) acquisition for active learning.
- The bootstrap surrogate adapter can include observation noise
  (`include_noise=True`).
- `Workbook.record_responses` records measured responses without touching
  openpyxl.
- **Screening designs and their diagnostics.**
  - `plackett_burman_design` (4 to 32 runs) and `definitive_screening_design`
    (Jones & Nachtsheim 2011, built from conference matrices).
  - `fold_over`, and `full_factorial_design` for general mixed-level factorials.
  - `fractional_factorial_design(..., generators=["D=ABC", ...])` builds a
    fraction directly from its generators. It needs no solver and never
    enumerates all 2^k runs.
  - `alias_structure` gives the defining relation, resolution, alias groups,
    and effect-correlation map of any two-level or three-level design,
    including nonregular ones.
  - `lenth_pse` and `half_normal_scores` handle unreplicated designs.
    `effects_estimates` now also estimates interactions and reports p-values.
    When no residual df are left it uses Lenth's PSE instead of returning NaN.
- **Response-surface analysis.**
  - `canonical_analysis` finds the stationary point and classifies it as a
    maximum, minimum, saddle or ridge.
  - `stationary_point_ci` gives a delta-method confidence interval on the
    optimum's location.
  - `steepest_ascent_path` and `ridge_analysis`.
  - `desirability` and `overall_desirability` (Derringer & Suich).
  - These accept a `fit_least_squares` result directly.
- **Prediction variance.** `prediction_variance`,
  `scaled_prediction_variance`, `fds_curve` (fraction-of-design-space plots),
  and `i_criterion`/`g_criterion`. They work over box and mixture regions, for
  the linear templates, any basis, or a `SymbolicModel`.
- **Restricted randomization.**
  - `blocked_factorial_design` uses minimum-aberration block generators and
    supports partial confounding.
  - `split_plot_design` builds split-plot designs, and `split_plot_anova`
    analyzes them with two error strata and variance components.
- **Space filling.**
  - `latin_hypercube_design(optimize="maximin")` uses Morris–Mitchell
    annealing.
  - `space_filling_metrics` reports minimum distance, φp, centered
    discrepancy, and projection gaps.
  - `quasi_random_design` gives Sobol and Halton designs, and
    `ClassicalDesign.to_unit_matrix()` rescales a design to the unit cube.
- **Constrained mixtures.**
  - `check_mixture_bounds` tests bound consistency and tightens implied bounds.
  - `extreme_vertices` and `extreme_vertices_design` (McLean & Anderson 1966).
  - Pseudo-component transforms, and `cox_direction_trace` for trace plots.
- **CLI and browser app templates.** `fractional-factorial` (generators; the
  `--runs` MILP search is CLI-only), `plackett-burman` and
  `definitive-screening` are available as `discopt doe new` templates and in
  the browser app. `latin-hypercube` takes `--optimize
  {discrepancy,maximin,none}`. `discopt doe anova` on a two-level design now
  also reports signed effects. On a saturated design it reports them against
  Lenth's PSE instead of failing.
- Follow-ups from using the new API in worked examples:
  - `coded_matrix`: a design's coded ±1 run matrix, with centre and middle levels
    at their coded positions.
  - `stationary_point_ci(..., center=, half_range=)`: natural units.
  - `effects_estimates` reports the `df` behind each standard error (residual
    df, or Lenth's m/3).
  - `split_plot_design(..., sub_plot_replicates=)`: several copies of the
    sub-plot factorial per whole plot.
  - `ridge_analysis` solves the degenerate "hard case" (`b` orthogonal to the
    leading canonical axis, including `b = 0`) in closed form instead of
    raising.
  - `check_mixture_bounds` states the tightened ranges.
  - `space_filling_metrics` warns when handed a natural-unit matrix without
    `bounds`.
- `fit_least_squares(..., level=0.95)`: the confidence level of the reported
  intervals. `initial` is now optional; a model linear in its parameters
  converges from the default start.

### Changed
- **I-optimal design search is faster.** `DesignRegion.moment_matrix` was a
  plain property, so the region's moment matrix `W` was rebuilt on every
  candidate evaluation inside the search even though a region is constructed
  once and never mutated. It is now cached per region. Measured on quadratic
  regions over a box: 1.8x fewer seconds per criterion evaluation with 2
  factors, 4x with 3, and 10x with 4, since the discarded work grows with the
  number of integration points. Results are unchanged.

### Docs
- `choosing-a-design.md`: replicating a Latin square is not enough to test an
  interaction. You must also drop a block, and the text now says so.
- `references.bib`: corrected the DOI of `atkinson1998-dt`
  (`10.1016/S0169-7439(98)00046-X`).

## [0.3.0] - 2026-08-16

### Added
- **The browser app is counted in Google Analytics.** A GA4 tag on
  `web/index.html` only — the docs are not instrumented — reporting to the same
  property as `kitchingroup.cheme.cmu.edu`, so the app's traffic lands beside
  the group site's and the two separate on the `hostname` dimension. Beyond the
  page view it records five events, because page views answer nothing about an
  app whose whole point is what people build with it: `design` and
  `design_failed` (which design types get generated, and which ones fail in the
  wild — otherwise visible only to the person who hit the error), `analyze` and
  `analyze_rejected` (the funnel question: do people run the experiments and
  come back with data?), and `boot_failed` (a Pyodide or wheel-install failure
  leaves the page dead on arrival and otherwise looks like an ordinary visit).
  The design type rides along as a `template` parameter, since GA4 event names
  take no hyphens. The snippet is wrapped rather than pasted verbatim: gtag.js
  has no local opt-out of its own, so a plain one would count every load from
  `build_web_app.py --serve` as a real visit. Every call is best-effort — read
  off `globalThis` so the Node test harness does not trip over a missing
  `window`, and an absent, still-loading or unconfigured `gtag` is a no-op.
  The masthead's "nothing is uploaded anywhere" is now "your data never leaves
  it", which is the claim that was ever the point and is still true.

### Changed
- **The `discopt` floor moved from 0.6 to 0.8.** 0.6 is still the release that
  introduced the public `discopt.parametric` API and the `"discopt.cli"` plugin
  hook this package is built on; 0.8 is what it is now pinned to, because that
  release fixes solver answers the constrained and model-based design paths
  depend on — a GDP disjunction wrongly reported `infeasible`, a false
  `Unbounded` on a bounded LP, vectorized models silently receiving no
  relaxation, and a non-zero `Constraint.rhs` ignored through the public API.
  The import-time guard (`_MIN_DISCOPT`) and its test moved with it. The browser
  app is unaffected: it never installs the base package, so the guard's
  "no dist metadata" branch is what it takes, as `web/test-wasm.mjs` confirms.

### Fixed
- **`anova_report(include_replicate=True)` was inert for most callers.** The flag
  was only honoured inside the `factors is None` branch, so it did nothing for
  anyone passing an explicit `factors=` list — the common case, and the one the
  CLI takes. `replicate` is a bookkeeping column that automatic detection skips
  and a caller naming its factors has no reason to list, which is exactly why
  the flag has to work in both cases: a request to block on replicate came back
  silently unblocked, with the replicate variance left in the residual and every
  F-ratio smaller than it should be. `discopt doe anova` carried a local
  workaround for this, now removed.

- **Model discrimination broke on any model with a large parameter.** Predicting
  a candidate design went through a dummy QP — `min Σ(θ - θ_nom)²` with the
  design pinned — purely to read back a point every term of which was already
  known. That objective is badly scaled when a parameter is large: an activation
  energy of ~8e4 leaves a KKT residual of 4e-6 against the solver's absolute
  1e-6 stationarity tolerance, and from discopt 0.8 on the stationarity guard
  correctly refuses to certify the point and returns `status="error"`, so
  `discriminate_design` died with *"No solution available in result"* on the
  canonical Arrhenius-vs-Eyring case. `_predict_with_covariance` now assembles
  `x*` directly via `fim._assemble_x_flat_direct` — the fast path `compute_fim`
  has taken since it was added — and falls back to the solve only for a
  constrained or implicit-state model that genuinely needs one. Guarded by a
  test that makes `Model.solve` raise.
- **Every mixture design was broken in the browser.** `scheffe-linear`,
  `scheffe-quadratic` and `scheffe-special-cubic` failed at design time with
  `No module named 'discopt.estimate'`: the mixture branch reached for
  `project_to_simplex` and `sum_constraint` through `discopt.doe`, whose
  package-level names resolved via `discopt.doe.design` — which imports the FIM
  machinery and the base package, neither of which exists in a WebAssembly
  build. The three helpers are pure simplex geometry and now live in a new
  `discopt.doe.simplex`, re-exported from `discopt.doe.design` so existing
  imports keep working. Found by `web/test-fuzz.mjs`; guarded by a hygiene test
  that *builds* a mixture design without the base package rather than only
  importing the module.
- **Browser app — a hidden element was not hidden.** The `hidden` attribute is
  honoured by a UA-stylesheet rule, which any author rule setting `display`
  outranks; `.boot { display: flex }` was enough to leave the analysis status
  line on screen permanently, so its spinner kept spinning after the work had
  finished — and was there before any workbook was dropped. A
  `[hidden] { display: none !important }` guard settles it for every element,
  and `test-dom.mjs` fails if the guard is removed while any element carries
  both `hidden` and a class.
- **Browser app — the analysis could stall in a backgrounded tab.** Each step
  yielded through `requestAnimationFrame` so the progress line would paint
  before Pyodide blocked the thread. Browsers stop firing frame callbacks for a
  tab that is hidden, backgrounded or fully occluded, so switching back to
  Excel after dropping a workbook left the analysis waiting for a frame that
  would not arrive until you returned — and a drop landing while it was stalled
  was silently ignored, which made it look permanent. The yield is now raced
  against a timer, and a drop during an analysis says so.
- **The regularizing ridge could decide the design.** `_RIDGE` is meant to keep
  `log det FIM` finite before a design reaches full rank, but as an absolute
  `1e-6` it outvoted the data whenever a parameter's information was smaller
  than that in its own units — which is ordinary, not exotic: an Arrhenius
  activation energy in J/mol has `∂y/∂Ea ~ 1e-4`, so its information is ~1e-8.
  For `k0·exp(-Ea/RT)` over T ∈ [300, 500] K at σ = 1 (the browser app's
  default, and its own worked example) a six-run design collapsed onto a single
  temperature and scored *higher* than the correct two-point design; the
  returned design identified neither parameter, silently. The ridge is now a
  fraction of the information in each direction, measured over the factor box,
  so it keeps the guard and loses the vote. Every design search and the FIM
  `fit` writes back are covered. A design is now independent of σ, as
  D-optimality requires, and `test_symbolic.py` pins that.

### Changed
- **Browser app — the analyze step runs on drop.** Dropping (or choosing) a
  filled-in workbook now reads it, fits the model, and runs the ANOVA on its
  own; the *Fit model* and *ANOVA* buttons are gone. Progress is reported while
  Pyodide works, and the fit and the ANOVA are independent, so whichever does
  not apply to a design explains itself instead of hiding the other's results.
- **Browser app — the right ANOVA for the design.** The factor-level ANOVA now
  runs only for designs built out of levels (the Latin-square family, 2-level
  factorials), where comparing level means *is* the analysis. On a continuous
  design every distinct factor value was becoming its own "level", which
  decomposes nothing and reports F-ratios against an aliased residual; those
  designs get the fit's regression ANOVA and coefficient tests instead. It
  still appears as a fallback when the fit could not run, labelled as one.

### Added
- **`discopt.doe.linear_design.basis_terms`**: the basis function multiplying
  each parameter, as factor powers (`{}` for the intercept, `{"T": 2}` for a
  pure square). It is `design_row`'s ordering knowledge in a form a caller can
  print, which is what lets the model be written out as an equation from
  workbook metadata alone; the test suite evaluates every term against
  `design_row` so the two cannot drift.
- **Browser app — the model, written out.** Step 1 shows the equation the
  design is built to estimate (`yield = b0 + b1·T + b11·T² + b12·T·P`), and the
  fit shows it again with the fitted numbers substituted — by sympy for a
  user-defined model, so the expression stays exact rather than being
  string-pasted. Designs with no model (the Latin-square family) show none.
- **Browser app — uploading a workbook fills step 1 in.** A campaign carries its
  whole design, so dropping one now shows the design that produced it —
  template, factors, and the options it was built with, including a
  user-defined model's expression and nominal parameters — instead of leaving
  the form on the page defaults. Factors come back in whichever editor style
  the template uses (continuous bounds, a low/high pair, a level list). A
  campaign the page cannot rebuild, such as a `--module` experiment, says so
  and leaves the form alone. New `design_spec_from_workbook` in
  `web/bootstrap.py`.
- **`web/test-fuzz.mjs`**: every design type, end to end, through the real page
  against real WebAssembly — fill the form, generate, write responses into the
  workbook, drop it back on step 2, check what the page shows. Randomization is
  seeded and the seed is printed, so a failure replays. It is the only test
  where the JavaScript and Python halves have to agree; it found the mixture
  bug above on its first run. Runs in CI, and `web/fake-dom.mjs` now holds the
  DOM stand-in that it shares with `test-dom.mjs`.
- **A user-defined model's fit reports the same statistics as a template's.**
  `_do_fit_symbolic` now returns coefficient t-tests, the
  Regression/Residual/Total decomposition and the summary (R², RMSE), and
  writes the ANOVA sheet into the workbook. For a nonlinear model these are the
  asymptotic forms — the footing its confidence intervals already stood on.
- **`do_fit` returns its regression statistics.** The per-coefficient
  t-statistics and p-values, the Regression/Residual/Total decomposition, and
  the summary (R², adjusted R², RMSE, F) were already computed and written to
  the workbook's ANOVA sheet, but were dropped from the returned dict, so every
  caller had to recompute the t-tests to answer the question a fit is usually
  asked. Now returned as `coefficients`, `regression_anova`, and `summary`.
- **Browser app — significance where it belongs.** Each fitted coefficient is
  marked ✓ or ✗ on whether its 95% interval excludes zero (equivalent to the
  t-test at α = 0.05, and the one rule that works for the nonlinear fit too,
  which reports intervals but no p-values), and the model's own
  Regression/Residual/Total ANOVA is shown with R². The glyph carries the
  verdict; colour only reinforces it. A fit with no residual left says so
  rather than presenting degenerate p-values as a result.
- **Browser app — actionable diagnostics for a workbook that cannot be
  analyzed.** A new `diagnose_workbook` entry point in `web/bootstrap.py`
  classifies every response and factor cell against both the formula-preserving
  and the `data_only` views of the sheet, so the page can name the runs at fault
  and say what is wrong with each: empty, text rather than a number, or — the
  case that looks like a perfectly good number in Excel — a formula whose
  computed result was never written to the file. Fewer completed runs than
  parameters is likewise called out rather than silently returning blank
  standard errors. `web/test-dom.mjs` covers the flow with no browser and no
  network.
- **Browser app** (`web/`, published at `/app/` alongside the docs): the design →
  download → fill in → upload → fit workflow, running entirely client-side on
  Pyodide with no server and no install. Workbooks round-trip with the CLI.
  Build it locally with `python scripts/build_web_app.py --serve`.
- **Classical designs** (`discopt.doe.classical`): `latin_hypercube_design`,
  `central_composite_design`, and `box_behnken_design`, plus matching
  `latin-hypercube`, `central-composite`, and `box-behnken` CLI templates. These
  record a regression basis (`--basis linear|quadratic`) rather than a model, so
  `discopt doe fit` works on them directly — including at factor counts the
  `response-surface-2d`/`-3d` templates do not cover.
- **`discopt.doe.linear_design`**: closed-form optimal design for models linear
  in their parameters. Every built-in template is of that form, so the Fisher
  information is exactly `XᵀX/σ²` and needs no autodiff; the test suite pins the
  result against the jax path to machine precision. `NewParams.use_linear_design`
  opts `do_new` into it. This is what lets model-based optimal design run where
  jax cannot be installed.
- `Workbook.parameter_names()`, which derives the parameter ordering from
  template metadata without constructing an `Experiment`.
- **User-defined models** (`discopt.doe.symbolic`) and a matching `symbolic` CLI
  template: write the response expression, name its parameters, and it is
  differentiated with sympy to design for it. Works for models nonlinear in
  their parameters, where the closed-form linear FIM does not apply — the tests
  pin the symbolic Jacobian against jax autodiff on Arrhenius,
  Michaelis-Menten, exponential-decay, and two-factor models. `fit` uses
  nonlinear least squares with that analytic Jacobian, and `extend` re-centres
  the next batch on the fitted values. The browser app gains a model editor
  that shows the parsed model and its derivatives as you type.
- `linear_design.batch_design_from_basis`, the greedy batch search factored out
  so it can be driven by any Jacobian-row provider — a basis function or a
  sympy derivative.
- `sympy` is now a runtime dependency (it has a WebAssembly build; jax does not).

### Changed
- `discopt.doe` now resolves its re-exports lazily (PEP 562). `from discopt.doe
  import X` is unchanged, but importing the package no longer pulls all 18
  submodules — and with them `discopt.estimate`, scipy, and the jax entry
  points. Submodules that do not need the base package or jax (`anova`,
  `classical`, `cli`, `latin`, `linear_design`, `screening`, `templates`,
  `workbook`) now import without either present, which `tests/test_import_hygiene.py`
  enforces and a new `wasm` CI job proves end to end.
- `discopt doe fit` derives its parameter ordering from metadata instead of
  rebuilding the `Experiment`, so it no longer requires `discopt.modeling`.
- Missing jax now raises an actionable error naming the alternatives, rather
  than a bare `ModuleNotFoundError` from inside a Jacobian call.
- Verified against `discopt` 0.7.0 and refreshed `uv.lock` to it (pulls `pounce-solver`
  0.9.0). The 0.7 release changes solver internals only — the `discopt.estimate`,
  `discopt.parametric`, `discopt.modeling` and `discopt.cli` surfaces this package
  consumes are unchanged — so the `discopt>=0.6` floor still holds and no code changes
  were needed.

### Security
- A campaign workbook stores a user-defined model as an *expression*, never as
  executable source, and reading one back never calls `eval`:
  `discopt.doe.symbolic.parse_expression` walks Python's AST and rebuilds the
  sympy tree node by node, rejecting attribute access, subscripts, lambdas,
  comprehensions, and any call outside a fixed function list. Opening a
  workbook someone sent you therefore executes none of their code.

### Documentation
- **The committed notebook outputs are now re-generated and gated.** Jupyter Book
  runs with `execute_notebooks: "off"`, so the committed outputs *are* the
  published book — and nothing re-ran them after the initial commit. Eight of the
  nine had drifted, and `latin-designs.ipynb` no longer ran at all. New
  `scripts/execute_notebooks.py` executes them and writes the outputs back
  (`make notebooks`); a `notebooks` CI job runs it with `--check` so this cannot
  recur silently. All notebooks re-executed, plus these content fixes:
  - `latin-designs.ipynb` asked for a row×treatment interaction on a single
    Latin square. Row and treatment determine column there, so the interaction
    subspace contains the column main effect and the terms are aliased — the
    non-orthogonality guard now (correctly) refuses it, and the notebook errored
    out on that cell. The section is rewritten to explain the aliasing, show the
    refusal deliberately, and demonstrate the two ways out: replicate and block
    on `replicate`, or drop `column` and pay for the interaction with a block.
  - `tutorial_batch_doe.ipynb` demonstrated `sequential_doe` with the exact
    replicate-confusion antipattern its docstring warns about: a runner always
    returning `{"y": ...}`, so observations taken at different designed times
    were fitted as replicates at one condition and `k_hat` wandered (0.219,
    0.543, 0.586 against `k_true = 0.7`). It now uses the prescribed stateful
    experiment, converging to 0.693, and explains why.
  - `active-learning.ipynb` drew its simulated measurement noise from
    `np.random.default_rng()` with no seed — fresh entropy on every call, so the
    notebook could not reproduce its own numbers. It now uses a seeded
    generator. Its `ConvergenceWarning` filter matched on the message text,
    which a `ConvergenceWarning` does not contain, so the filter never fired;
    it now filters on the category.
  - `identifiability-estimability.ipynb` published ~55 KB of solver log lines,
    ANSI escape codes and absolute `site-packages` paths from the deliberately
    singular fit it is built around. The solver's tracing is quieted via
    `RUST_LOG`, the `nan`-standard-error `RuntimeWarning` is filtered (the
    summary tables already report it), and the repeated `discopt.solver`
    "duals withheld" warning is scoped away from the profile loop — leaving the
    one occurrence where it is the point.
- **New chapter: "Choosing a design"** (`docs/choosing-a-design.md`), a router
  from the question you are trying to answer to the design that answers it —
  screening, blocking, response surfaces, space filling, optimal design,
  mixtures and your own model — with the run-count and criterion (D/A/E/ME)
  guidance, the pitfalls that cost a batch of experiments, and links into the
  notebook that works each one through. The browser app links to it from the
  design-type selector and its footer.

## [0.2.0] - 2026-07-12

First public release on PyPI.

### Added
- Design-of-experiments plugin for the `discopt` modeling language, packaged as the
  `discopt.doe` PEP 420 namespace subpackage.
- `discopt doe` CLI verbs (via the `discopt.cli` entry-point group) and the
  `discopt-doe-install-skill` console script.
- Optional `gui` (Streamlit app) and `ml` (scikit-learn surrogates) extras.
- Pre-commit hooks (ruff lint/format + hygiene) enforced identically locally and in CI
  (`lint` job), plus a release `Makefile` and an OIDC trusted-publishing workflow
  (`publish.yml`).

### Fixed
- `optimal_experiment`: constrained designs no longer spuriously raise
  "No constraint-feasible design point found" when the single SLSQP refinement seed
  diverges (observed on macOS/Accelerate). It now falls back to a multi-start SLSQP
  refine and keeps the best feasible design.

### Changed
- Depend on `discopt>=0.6` from PyPI (removed the temporary `[tool.uv.sources]` git pin).

[Unreleased]: https://github.com/jkitchin/discopt-doe/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jkitchin/discopt-doe/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jkitchin/discopt-doe/releases/tag/v0.2.0
