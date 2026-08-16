# Changelog

All notable changes to `discopt-doe` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/jkitchin/discopt-doe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jkitchin/discopt-doe/releases/tag/v0.2.0
