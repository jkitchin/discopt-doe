# Repository review — prioritized issues

> **Resolution status (2026-07-10).** All P0, all P1, and essentially all P2/P3
> issues below were fixed on this branch — one commit per issue, each with a
> regression test that fails on the old code. The fast test suite went from 219
> to 277 passing (~58 new regression tests). Deliberately **not** changed, with
> rationale:
>
> - **#45** (duplicate `doe` entry-point warning) — the warning comes from the
>   base `discopt` extraction branch still declaring the `doe` entry point;
>   nothing in *this* repo can suppress it. It disappears when discopt 0.6 ships
>   with `discopt.doe` removed. Upstream fix only.
> - **#53** (notebook build scripts drifted from the committed notebooks) —
>   regenerating would blank the published cell outputs and the scripts use a
>   private API; making them canonical (regenerate + execute + CI diff) is a
>   workflow decision left to the maintainer.
> - **#72** (flag-naming inconsistencies: `--input` vs `--bounds`, `extend --n`
>   vs `optimize --batch-size`, `new --seed` default 42 vs `optimize --seed`
>   default None) — these are breaking renames best decided before the first
>   release; documented here rather than changed unilaterally.
> - **#79/#80 remainder** — the load-bearing GUI bugs (folder browser, stale
>   ANOVA, seed default, flash messages, file-open errors) are fixed and a
>   short GUI doc section was added; the smaller polish items (mixture-editor
>   `ub` default, always-GP diagnostic plots, native-picker threading,
>   `_free_port` race, history logging for in-GUI edits) and broader `AppTest`
>   coverage remain as follow-ups.

Date: 2026-07-10. Full-repo review of `discopt-doe` for correctness, usability,
documentation, and practitioner foot-guns. Method: five independent review
passes (core numerics; sequential/model-based/discrimination statistics;
CLI/workbook/skill; Streamlit GUI; docs/packaging/CI), plus empirical checks:
the full test suite was run with and without extras, all ten notebooks' code
cells and the README/intro examples were executed against the pinned
`discopt` branch, and openpyxl round-trip behavior was tested directly.

**Baseline:** with `uv sync --all-extras --group dev`, all 219 tests pass
(65 slow tests deselected). Without extras, 13 tests fail on missing sklearn
(issue #46). All notebook code cells run cleanly against current source; all
README/intro Python examples and the README CLI sequence work end-to-end
(given the working uv install); `_toc.yml` and `references.bib` are fully
consistent; namespace packaging and package-data are correct.

**Priority definitions**

- **P0** — silently corrupts the core practitioner workflow, or blocks
  onboarding entirely. Fix before anything else.
- **P1** — wrong results or crashes in common, documented usage.
- **P2** — foot-guns, silent misbehavior, doc/API mismatches.
- **P3** — polish.

Each issue has **Confirm** (reproduce/validate the problem before touching
code), **Fix** (recommended strategy), and **Verify** (how to prove the fix,
usually a regression test to add). File:line references are against commit
`0d57b80`.

---

## P0 — fix first

### 1. `discopt doe optimize` ignores the workbook's stored direction/surrogate/acquisition — a minimize campaign gets maximized [correctness]

**Where:** `src/discopt/doe/cli.py:634-639` (settings written at `new` time),
`cli.py:681-736` (`do_optimize` reads only `params.*`), `cli.py:1830-1858`
(argparse defaults `maximize`/`gp`/`expected_improvement`).

**Problem:** `_do_new_optimize` persists
`{"criterion": ..., "surrogate": ..., "acquisition": ...}` in
`template_args`, but `do_optimize` never reads `wb.template_args()` — it uses
the argparse defaults unless flags are repeated on every invocation. The
documented driving command (`discopt doe optimize wb.xlsx`, e.g.
`skill/SKILL.md:29`) therefore **maximizes** a workbook created with
`--optimize-criterion minimize`, silently: recommendations chase the worst
observed region and the reported "incumbent" is the worst run.

**Confirm:** `discopt doe new optimize -o wb.xlsx --input x:0:1 --n 3
--optimize-criterion minimize`, fill in responses, run
`discopt doe optimize wb.xlsx --json` and observe `"criterion": "maximize"`
in the output and an incumbent at the max.

**Fix:** In `do_optimize`, default criterion/surrogate/acquisition from
`wb.template_args()` and let explicit CLI flags override (argparse: use
`default=None` sentinels so "explicitly given" is detectable); warn on
explicit-flag/workbook mismatch. Audit `do_extend`/`do_fit` for the same
pattern. The GUI optimize panel reads its own defaults too — align it
(see #48).

**Verify:** Regression test: create a minimize workbook, complete a few runs
with a known quadratic, call `do_optimize` with default params, assert the
recommendation direction and reported criterion are `minimize`. Add a
mismatch-warning test for explicit flags.

### 2. Excel formula cells in the response column are silently treated as *pending* — runs vanish from fit/ANOVA/optimize [correctness / data loss]

**Where:** `src/discopt/doe/workbook.py:286` (`load_workbook(path)` without
`data_only=True`), `workbook.py:175-181` (`_coerce_float` returns `None` for
`"=..."` strings), `workbook.py:404-410` (`completed_runs` filter).

**Problem:** openpyxl returns the formula *text* for formula cells unless
opened with `data_only=True`. A practitioner who fills the response with
`=AVERAGE(...)` (the natural way to enter a mean of replicate readings)
gets those rows silently classified as pending: dropped from the fit, the
ANOVA, and the optimizer, with `status` just showing them as not yet run.
If only some rows are formulas, the fit is silently wrong.

**Confirm:** Write `=2*3` into a response cell of a generated workbook, save
with Excel/openpyxl, run `discopt doe status` — the row counts as pending.
(Empirically confirmed during review.)

**Fix:** Read via a second `load_workbook(path, data_only=True)` handle for
run values (keep the formula-preserving handle for writes so user formulas
survive), and raise a clear error when a response cell holds a formula
string but the cached value is unavailable (file never opened/saved by
Excel: openpyxl's `data_only` returns `None` in that case — tell the user
"open and re-save in Excel, or enter values directly").

**Verify:** Round-trip test using openpyxl to inject a formula with a cached
value and one without; assert the former is read as its value and the latter
produces the actionable error, not "pending".

### 3. The documented install cannot work: package not on PyPI, `discopt>=0.6` unresolvable by pip, git pin is uv-only [usability / onboarding]

**Where:** `README.md:28-35`, `docs/intro.md:38-42`, `pyproject.toml:13-15,58-64`.

**Problem:** README/intro lead with `pip install discopt-doe`. Verified:
PyPI 404s for `discopt-doe`, latest `discopt` on PyPI is 0.5.0, and the
`[tool.uv.sources]` git pin that makes `discopt>=0.6` resolvable is ignored
by pip. Every pip path fails; the only working install (clone + `uv sync`)
is hidden in the Development section.

**Confirm:** `pip index versions discopt-doe` / `pip download 'discopt>=0.6'`
in a clean venv (both fail).

**Fix:** Until the 0.6.0 release: rewrite the Install sections around clone +
`uv sync` (or a two-step
`pip install git+https://github.com/jkitchin/discopt@refactor/389-extract-doe`
then `pip install git+https://github.com/jkitchin/discopt-doe`), with a loud
"not yet on PyPI" note. Keep the pip commands as the *post-release* form,
clearly labeled.

**Verify:** Follow the rewritten instructions verbatim in a clean container;
`discopt doe templates` runs.

---

## P1 — wrong results or crashes in common usage

### 4. `sequential_doe` double-counts prior information in the accumulated FIM [correctness]

**Where:** `src/discopt/doe/sequential.py:121`.

**Problem:** `prior_fim = est.fim if prior_fim is None else prior_fim + est.fim`.
`est.fim` from `estimate_parameters` is already computed on **all**
accumulated data each round, so summing across rounds counts round-0 data
r+1 times by round r. The inflated prior is fed to
`optimal_experiment(prior_fim=...)`, so every design after the first round
optimizes against a wrong information state (new-point information
under-weighted). The existing test (`test_fim_accumulates`, monotone
det only) cannot catch this.

**Confirm:** Two-round run with fixed data; assert
`history[1]` used `prior_fim == history[1].estimation.fim` — currently it is
`fim_round0 + fim_round1_cumulative`.

**Fix:** `prior_fim = est.fim` (it already accumulates), or compute per-round
incremental FIMs and sum those. Check `discrimination_sequential.py` for the
same pattern.

**Verify:** Regression test asserting the prior passed to the design step
equals the current cumulative FIM (monkeypatch `optimal_experiment` to
capture the kwarg).

### 5. Constrained `optimal_experiment` typically returns a constraint-violating design, silently [correctness]

**Where:** `src/discopt/doe/design.py:385-415` (candidate scan + acceptance),
`design.py:449-484` (`_scan_candidates` never evaluates constraints).

**Problem:** The multistart scan ranks candidates purely by criterion,
ignoring `equality_constraints`/`inequality_constraints`. The
SLSQP-refined feasible design is then accepted only if its criterion beats
the unconstrained incumbent — which it usually can't (the feasible optimum
is by definition no better). Result: the infeasible multistart winner is
returned with no warning. With `local_refine=False`, constraints are ignored
entirely. This is the mixture-design path (`sum_constraint`,
`project_to_simplex`) — and grep shows **zero tests** exercise it.

**Confirm:** `optimal_experiment` with a simple 2-factor model and
`sum_constraint(["x1","x2"], 1.0)` but no `feasible_projection`; check
`x1 + x2 != 1` in the returned design.

**Fix:** When constraints are supplied: filter or penalize infeasible
candidates in `_scan_candidates` (evaluate constraint residuals), and accept
the SLSQP result by comparing only among feasible points. Warn or raise if
no feasible candidate is found.

**Verify:** New test module for the constrained path: returned design
satisfies constraints to tolerance for equality and inequality cases, with
and without `feasible_projection`, plus a `local_refine=False` behavior test
(document/raise).

### 6. `anova_report` computes marginal SS with only a marginal balance check — non-orthogonal designs give silently wrong tables, including negative residual SS [correctness]

**Where:** `src/discopt/doe/anova.py:173-187` (balance check),
`anova.py:206-218` (per-factor SS vs grand mean), `anova.py:249-256`
(residual), `anova.py:3` (docstring claims "Type-I (sequential) SS").

**Problem:** Each main effect's SS is computed independently against the
grand mean — valid only for mutually orthogonal factor columns — but the
balance check only verifies per-factor level counts. Hand-trace: rows
(A0,B0)×2, (A1,B1)×2 with y=[0,0,2,2] is marginally balanced (no warning),
yet SS_A = SS_B = 4 with SS_total = 4 → SS_residual = −4, and F/p are
silently omitted because MS_residual ≤ 0. Aliased factors are exactly what a
practitioner feeds this after a botched fraction. The docstring/warning text
describe Type-I sequential SS, which this is not (the SS here are
order-independent marginal SS).

**Confirm:** Run the 4-row hand-trace above through `anova_report`; observe
negative residual SS and no warning.

**Fix:** Check pairwise cross-tabulations for proportional/orthogonal
balance, not just marginals; raise (or warn loudly and mark the table) when
`ss_residual < 0`; fix the docstring to describe what is actually computed,
or implement true sequential SS.

**Verify:** Tests: the aliased 4-row case raises/warns; an orthogonal design
still matches scipy/statsmodels reference values; assert
`ss_residual >= 0` invariant across the test corpus.

### 7. `ExplorationResult.best_point` returns the first *failed* (NaN) grid point when any grid evaluation failed [correctness]

**Where:** `src/discopt/doe/exploration.py:51-57` (argmax/argmin),
`exploration.py:186` (NaN init), `exploration.py:193-198`
(`except Exception: continue`).

**Problem:** Infeasible points are deliberately left NaN, and
`np.argmax`/`np.argmin` return the index of the first NaN when present. One
failed grid point makes `best_point()` return that failed point, silently —
in the exact "scan the landscape, take the best point" workflow the module
exists for.

**Confirm:** Grid where one point raises inside `compute_fim` (e.g. bound
that makes the model infeasible); `best_point("log_det_fim")` returns it.

**Fix:** Use `np.nanargmax`/`np.nanargmin`; raise a clear error when all
entries are NaN. Consider logging how many grid points failed (a fully-NaN
metric with a silent `except Exception` also masks systematic errors — see
#31 for the same pattern elsewhere).

**Verify:** Test with an injected failing grid point asserting the returned
best point is the true finite optimum; test all-fail raises.

### 8. `model_based_optimize_round` reports parameters, standard errors, and FIM log-det from a fantasy-contaminated fit [correctness]

**Where:** `src/discopt/doe/model_based.py:436-467`.

**Problem:** The batch loop refits the surrogate on mean-imputed fantasy
points *including after the last pick*, then reports
`result.parameters`, `result.parameter_se` (from `s.covariance_`), and
`result.fim_log_det` from that contaminated fit. Even at `batch_size=1`, the
FIM includes rows for not-yet-run experiments whose "observations" are the
model's own predictions: `parameter_se` is systematically too small and
`fim_log_det` inflated — precisely the convergence diagnostics a
practitioner watches. The contaminated log-det is also written to the
workbook log.

**Confirm:** Call with `batch_size=1` vs `batch_size=5` on the same data;
reported `parameter_se` shrinks with batch size despite identical real data.

**Fix:** Snapshot the real-data fit (θ̂, covariance, FIM) before the fantasy
loop and report from the snapshot; do fantasy refits on a copy of the
surrogate.

**Verify:** Regression test: `parameter_se` and `fim_log_det` identical
across `batch_size` values for fixed data.

### 9. `ParametricSurrogate.fit` crashes when observations < parameters and silently ignores parameter bounds [correctness / usability]

**Where:** `src/discopt/doe/model_based.py:290`
(`least_squares(..., method="lm")`).

**Problem:** SciPy's `method="lm"` raises `ValueError` when residuals <
variables — but the module's selling point is sample efficiency ("~1–2·d
points") and `model_based_optimize_round` requires only ≥ 1 completed run,
so a response-surface template (6 parameters) with < 6 runs dies with an
obscure scipy error. LM also supports no bounds, so declared parameter
bounds (e.g. `K ∈ [1e-4, 1e4]`) are silently ignored and θ can wander into
nonphysical values that `estimate_parameters` would forbid.

**Confirm:** 2-parameter model, 1 completed run → scipy `ValueError`.
Bounded parameter with data pulling it out of bounds → out-of-bounds θ̂.

**Fix:** Use `method="trf"` with bounds extracted from the experiment;
raise a clear "need at least p completed runs to fit p parameters (have n)"
error for the under-determined case.

**Verify:** Tests for both: friendly error message when n < p; fitted θ
respects bounds.

### 10. Every `fit`/`extend`/`optimize` re-saves the workbook via openpyxl, destroying user-added charts and images with no warning or backup [data loss]

**Where:** `src/discopt/doe/workbook.py:278-300` (`open` + `save` to same path).

**Problem:** openpyxl documents that charts/images (and some other content)
in a loaded file are lost on save. A practitioner who plots their results in
the workbook and then runs `discopt doe extend` loses the charts.

**Confirm:** Add a chart via openpyxl to a generated workbook, run
`discopt doe status && discopt doe extend`, reopen: chart gone.

**Fix:** Layered: (a) warn at open time when the workbook contains
charts/images; (b) write a one-generation `.bak` sibling before the first
in-place save of a session; (c) document prominently in the instructions
sheet ("keep plots in a separate file").

**Verify:** Test asserting the `.bak` exists and contains the pre-save bytes,
and that a warning fires when charts are present.

### 11. Falsy workbook metadata silently replaced by defaults: `seed 0` becomes 42, `measurement_error 0` becomes 1.0 [correctness]

**Where:** `src/discopt/doe/workbook.py:346-350` (`... or 1.0`, `... or 42`).

**Problem:** `0 or 42 == 42`. A campaign created with `--seed 0` is not
reproducible across `extend` (the CLI's own tests use `seed=0`). `--error 0`
is also accepted unvalidated (∞ weights in the design FIM at `new` time,
then silently read back as 1.0 at `fit` time).

**Confirm:** `wb.seed()` on a `--seed 0` workbook returns 42.

**Fix:** `v = self.metadata().get(k); return default if v in (None, "") else
cast(v)`. Validate `--error > 0` at argparse time (see also #P3 item on
`measurement_error=0` in `compute_fim`).

**Verify:** Unit tests for seed 0 and error 0.5/0 round-trips; argparse
rejection test for `--error 0`.

### 12. `discopt doe anova --include-replicate` is a dead flag: the replicate column is never written to the workbook [correctness]

**Where:** `src/discopt/doe/cli.py:569,846` (row projection drops
`replicate`), `workbook.py:256-260` (runs header), `cli.py:1388-1389`
(`if "replicate" in r` — never true); generators do tag rows
(`latin.py:266`, `screening.py:188`).

**Problem:** `_do_new_latin`/`_do_new_factorial` project design rows onto
factor names only, so the replicate tag never reaches the runs sheet, and
the flag's promise ("treat the replicate column as a blocking factor")
silently does nothing — different SS/df/F than the user asked for.

**Confirm:** Replicated latin-square workbook; `--include-replicate` output
identical to without.

**Fix:** Persist `replicate` (and consider `run_order`/`is_center`) as runs
columns; error if `--include-replicate` is passed but no replicate column
exists.

**Verify:** CLI round-trip test asserting the ANOVA table gains the
replicate blocking row and df change.

### 13. The sequential loops' data↔design contract is undocumented — returning the same response keys each round fits different conditions as replicates [footgun / docs]

**Where:** `src/discopt/doe/sequential.py:92-95,154-174`;
`discrimination_sequential.py`; the only correct usage pattern lives in
`tests/test_discrimination_sequential.py:36-136`.

**Problem:** `run_experiment(design) -> data_dict` results are merged by key
and passed to `estimate_parameters`, which treats repeated values under one
key as replicates at the model's *fixed* conditions — the design values are
never associated with the data. The natural implementation (same keys each
round) silently produces wrong parameter estimates that corrupt every
subsequent design. The required pattern (stateful experiment minting a fresh
response key per observation) is documented nowhere.

**Confirm:** Toy linear model, two rounds at different designed x returning
`{"y": value}` each round; fitted slope is garbage with no warning.

**Fix:** Document the stateful-experiment/fresh-key contract prominently in
both loops' docstrings and the docs; detect and refuse (or warn) when
`run_experiment` returns keys that already exist in `all_data` while the
experiment has design inputs.

**Verify:** Test that the misuse pattern raises/warns; a documented example
in `sequential.py`'s docstring that mirrors the discrimination test's
pattern.

### 14. matplotlib is required by every tutorial notebook, `ExplorationResult.plot_*`, and the GUI plots — but declared nowhere [packaging]

**Where:** `src/discopt/doe/exploration.py:92,133`,
`src/discopt/doe/gui/app.py:1898,1923,1958`, `pyproject.toml:22-24,36-47`,
all tutorial notebooks' first cells.

**Problem:** `uv sync --all-extras` yields no matplotlib (verified). The
first tutorial notebook fails at `import matplotlib.pyplot`; the GUI's
history/surrogate plots and `plot_sensitivity()` (advertised in
`model_based_doe.md:103`) crash. The `gui` extra also omits scikit-learn, so
a `[gui]`-only install crashes on its optimize round.

**Confirm:** `uv sync --all-extras && uv run python -c "import matplotlib"`.

**Fix:** Add matplotlib to the `gui` extra and the dev group; add
`scikit-learn` to `gui` (or document that GUI optimization needs `[gui,ml]`);
give `exploration.plot_*` a friendly ImportError.

**Verify:** Fresh `--all-extras` sync imports matplotlib; notebook execution
in CI (see #54) passes.

### 15. `surrogate="gp"` (README quickstart #2, CLI `optimize` default) crashes a core install with a bare `ModuleNotFoundError: sklearn` [usability]

**Where:** `src/discopt/doe/surrogate.py:195-206,223-226`,
`cli.py:1835-1838`, `README.md:56-67`.

**Problem:** The GP preset — the *default* surrogate for the CLI optimize
verb and the README quickstart — imports sklearn with no guard, and sklearn
lives in the optional `ml` extra. New users on `pip install discopt-doe`
(once that works) hit a raw traceback with no mention of
`discopt-doe[ml]`. (Also the cause of the 13 no-extras test failures, #46.)

**Confirm:** venv without sklearn: `coerce_surrogate("gp")` → bare
ModuleNotFoundError (verified).

**Fix:** Catch ImportError in `_gp_preset`/`_response_surface_preset` and
re-raise as a `DoEError`-family message: `pip install "discopt-doe[ml]"`.
Annotate README quickstart #2 with "requires the [ml] extra".

**Verify:** Test (with sklearn import blocked via `sys.modules` patch) that
the error message names the extra.

### 16. GUI folder browser navigation is instantly reverted by stale text-input state [usability]

**Where:** `src/discopt/doe/gui/app.py:1297-1336`.

**Problem:** The output-folder text input is keyed
(`key="output_dir_input"`), so Streamlit ignores its `value=` default once
state exists. After clicking "📁 subdir" or "⬆ Parent", the rerun re-reads
the *old* text state, sees it differs from `output_dir`, and writes the old
directory back — undoing the navigation. The folder browser is effectively
non-functional; only typing a path works.

**Confirm:** `AppTest`-driven test: click a browse-dir button, assert
`session_state["output_dir"]` changed (currently reverts).

**Fix:** When a browse button fires, also set
`st.session_state["output_dir_input"]` before the widget is instantiated on
the next run; or make the text input authoritative only via `on_change`.

**Verify:** The `AppTest` click test above passes.

### 17. `streamlit>=1.30` pin is below the APIs the app uses (`st.logo`, `width="stretch"`) — crash on permitted versions [correctness / packaging]

**Where:** `src/discopt/doe/gui/app.py:2268` (`st.logo`), many call sites
using `width="stretch"`; `pyproject.toml` `gui` extra.

**Problem:** `st.logo` landed ~1.35 and string `width` literals later still;
on streamlit 1.30–1.34 (allowed by the pin) the app crashes at startup with
`AttributeError`. The app also mixes deprecated `use_container_width=True`
(`app.py:599-624,2123`) with `width="stretch"`, so *some* streamlit version
always complains (see P3).

**Confirm:** `pip install streamlit==1.30` in a scratch venv; launch the app.

**Fix:** Determine the true minimum version by testing (likely ≥ 1.45 for
string widths), raise the pin, standardize on one width API.

**Verify:** CI or a smoke test (`AppTest`) against the pinned minimum
version.

### 18. Everything resolves `discopt` from a temporary git branch scheduled for deletion; no SHA pin [packaging / footgun]

**Where:** `pyproject.toml:58-64`, `.github/workflows/ci.yml:14-17`,
`uv.lock:405`.

**Problem:** The `[tool.uv.sources]` pin uses `branch =
"refactor/389-extract-doe"` — the pyproject comment itself says the branch
is temporary. Any re-lock after the branch merges/deletes fails to resolve;
if the commit becomes unreachable, even locked syncs break. The CI comment
("Requires discopt>=0.6 on PyPI") is wrong about where discopt comes from.

**Confirm:** `grep -A2 tool.uv.sources pyproject.toml`; `grep discopt uv.lock`.

**Fix:** Pin `rev = "<sha>"` alongside (or instead of) the branch; fix the CI
comment; file a tracking issue to switch to PyPI at discopt 0.6.0 (the
pyproject comment already says REMOVE — make it findable).

**Verify:** `uv lock --check` succeeds; simulate branch deletion by
re-locking against the SHA.

---

## P2 — foot-guns, silent misbehavior, doc/API mismatches

### Statistics and numerics

### 19. "BF" discrimination criterion is not the Buzzi-Ferraris–Forzatti criterion it claims to be [correctness / docs]

**Where:** `src/discopt/doe/discrimination.py:6-9,491-514`.

**Problem:** The 1984 BF statistic uses `S = 2Σ + V_i + V_j` (two
measurement-noise covariances) plus a trace term `tr(2Σ S⁻¹)`; the code uses
a single `Σ_y` (taken from model *i* only — asymmetric if models declare
different `measurement_error`) and drops the trace term, yet the docstring
sells it as "Buzzi-Ferraris-Forzatti (1984) … **Default**". Anyone applying
the classical BF adequacy stopping rule (`max T > n_y`) misjudges. Tests
check argmax only, never values.

**Confirm:** Hand-compute the true BF value for a 2-model 1-response toy
case; compare with `evaluate_discrimination_criterion`.

**Fix:** Implement `2Σ + V_i + V_j` + trace term with a symmetric Σ
convention (or rename the criterion and document the simplification
precisely).

**Verify:** Quantitative value test against the hand-computed reference,
plus a symmetric-in-(i,j) property test with differing model σ.

### 20. Discrimination prediction covariance `V` reflects the candidate design's own FIM, not parameter uncertainty from existing data; no `prior_fim` argument exists [correctness / usability]

**Where:** `src/discopt/doe/discrimination.py:424-425`.

**Problem:** `V = J pinv(JᵀΣ⁻¹J) Jᵀ` is built from the single candidate
design being scored: it is independent of how well parameters are actually
known, and when that one-point FIM is singular (p > n_responses — the common
case), `pinv` zeroes non-identifiable directions, i.e. treats
infinite-variance directions as zero variance. `discriminate_design` offers
no way to supply the parameter covariance from data collected so far, so
the documented stateless usage cannot be made statistically right.

**Confirm:** 1-parameter model: `V == σ²` identically for every candidate.

**Fix:** Add optional per-model `prior_fim` (or `parameter_covariance`)
kwargs and use `V = J Σ_prior Jᵀ`; document the stateful-experiment
alternative; wire `sequential_discrimination` to pass its accumulated FIMs.

**Verify:** Test that `V` shrinks as prior information grows; test the
p > n_responses case no longer silently zeroes variance.

### 21. HR/JR/MI criteria never check response-name alignment across models (only BF does) [correctness]

**Where:** `discrimination.py:474-488,517-554,557-596` vs the check at
`:502-506`.

**Problem:** `preds[i].y_hat - preds[j].y_hat` subtracts vectors ordered by
each model's own `response_names`; same responses in different dict orders
silently misalign.

**Confirm:** Two identical models with reversed response-dict order; HR value
changes.

**Fix:** Hoist BF's alignment check into `_predict_all_models` (align by
name, error on set mismatch).

**Verify:** The reversed-order test yields identical criterion values.

### 22. `model_selection` AIC/BIC/LRT comparisons silently invalid when candidate models declare different `measurement_error` [correctness / footgun]

**Where:** `src/discopt/doe/selection.py:104-132`.

**Problem:** The deviance `Σ((y−ŷ)/σ)²` drops the σ-dependent constant
`Σ log(2πσ_i²)`; with per-model σ differing, a model declaring σ=0.5 vs a
rival's σ=0.05 gets ~100× smaller deviance for identical misfit and wins
AIC/BIC regardless of fit. Only `n_observations` equality is validated.

**Confirm:** Two identical models differing only in declared σ;
`model_selection` prefers the large-σ one.

**Fix:** Reconstruct full log-likelihoods (the machinery exists in
`_per_obs_loglik`) or raise/warn when candidates' `measurement_error` maps
differ.

**Verify:** Test: identical models with different σ either tie (full
likelihood) or raise.

### 23. `vuong_test` silently uses only the first observation of each response [correctness]

**Where:** `src/discopt/doe/selection.py:354-366` (`data[name]).flat[0]`).

**Problem:** Array-valued data (replicates — explicitly supported by
`estimate_parameters` and produced by `sequential_doe`) is truncated to one
value per response name; N is the number of *names*; the z-statistic is
computed on truncated data with no warning.

**Confirm:** Vuong test with 10-replicate arrays vs the same data as 10
scalar keys — different results.

**Fix:** Iterate `np.atleast_1d(data[name])` (replicates share ŷ and σ),
matching the estimation objective.

**Verify:** Equivalence test: array form == exploded-scalar form.

### 24. `sequential_discrimination` proposes no design on the final round; `n_rounds=1` in caller-driven mode yields no design at all [correctness / docs]

**Where:** `src/discopt/doe/discrimination_sequential.py:96-98,152-169`.

**Problem:** Budget exhaustion is folded into the early-stop branch
(`design=None`), contradicting the docstring ("the final round carries the
recommended design"); with a runner, `n_rounds=k` runs only k−1 experiments.

**Confirm:** `sequential_discrimination(..., n_rounds=1, run_experiment=None)`
→ `rounds[-1].design is None`.

**Fix:** Design first, then decide whether to stop; or document k−1
semantics and reject `n_rounds=1` in caller-driven mode.

**Verify:** Test the n_rounds=1 caller-driven case returns a design; runner
case performs n_rounds experiments (or docs updated + guard test).

### 25. Estimability: a parameter with nominal value 0 is scaled by machine-eps and ranked dead-last regardless of true sensitivity [footgun]

**Where:** `src/discopt/doe/estimability.py:103-110`.

**Problem:** The scale fallback `abs(param_values[name]) or eps` with
eps=2.2e-16 effectively zeroes the sensitivity column for zero-valued
nominals (common: offsets; the discrimination docstring example itself uses
`"dS": 0.0`), so the parameter is declared unestimable and dropped from
`recommended_subset` no matter what.

**Confirm:** Linear model `y = a + b·x` with nominal `a=0`; `a` ranks last
with near-zero magnitude.

**Fix:** Fall back to 1.0 (with a warning) or require `parameter_scales` for
zero nominals.

**Verify:** The `a=0` test ranks `a` by actual sensitivity.

### 26. Yao cutoff implemented as *relative to the top pivot* but documented and defaulted as Yao's absolute 0.04 rule [correctness / docs]

**Where:** `estimability.py:140-143,172-173`.

**Problem:** Yao et al. (2003) apply the 0.04 cutoff to the scaled residual
column magnitude itself; the code applies it to the ratio against the
largest pivot. With well-excited data the relative rule keeps far fewer
parameters; results labeled "Yao et al. (2003)" can differ materially.

**Confirm:** Case with |R_11| ≫ 1; compare subsets under both rules.

**Fix:** Implement the absolute rule, or document the deviation and stop
attributing the default to Yao.

**Verify:** Reference test against a published/hand-computed Yao example.

### 27. `effects_estimates` SE/t pool all other factors' effects into the "residual" — t grossly deflated in multi-factor screens [correctness]

**Where:** `src/discopt/doe/screening.py:289-301`.

**Problem:** Within-level scatter for factor A contains the full variation
caused by B, C, … (varied within each A-level in a factorial). With several
large effects, every factor's SE is inflated and t deflated — the "does this
factor matter?" answer is biased toward "no". Tests check effect magnitudes,
never SE/t.

**Confirm:** 2³ full factorial with two large effects and tiny noise;
compare t values against OLS on the coded model matrix.

**Fix:** Compute SE from the residual of the joint main-effects OLS fit
(columns are orthogonal anyway) or from replicate/center-point pure error.

**Verify:** t-value test against the OLS reference.

### 28. `effects_estimates` derives low/high by sorting observed values, not the design's declared `(low, high)` — effect signs can flip [footgun]

**Where:** `screening.py:266-274`.

**Problem:** For `{"catalyst": ("B", "A")}` or reversed numeric levels
`(120, 80)`, the estimate reports `mean(sorted-last) − mean(sorted-first)`,
not `mean(HIGH) − mean(LOW)` as the docstring promises — a sign flip that
sends a practitioner the wrong direction.

**Confirm:** Reversed-levels design; effect sign differs from hand
calculation.

**Fix:** Accept an optional `FactorialDesign`/levels mapping to fix
orientation; document the sorting fallback.

**Verify:** Orientation tests for categorical and reversed-numeric levels.

### 29. `anova_report` default factor detection includes the `is_center` bookkeeping column [footgun]

**Where:** `anova.py:92-94,144`; rows carry `is_center` from
`screening.py:189` / `fractional.py:257`, whose docs promise `anova_report`
"applies unchanged".

**Problem:** `is_center` is treated as a factor; with center points, every
real factor also gains a mid level and `is_center` is perfectly confounded
with them. `effects_estimates` excludes it; the two modules disagree.

**Confirm:** `anova_report(design.rows, "y")` on a center-point design; table
contains an `is_center` row.

**Fix:** Add `is_center` (and `run_order`) to the exclusion set; rename the
misnamed `_is_response_column`.

**Verify:** Center-point ANOVA test: no `is_center` factor, SS as expected.

### 30. `compute_fim` and `discriminate_design` silently ignore unknown design-value keys — a typo yields results at an arbitrary design point [footgun]

**Where:** `fim.py:262-267` (`if name in em.design_inputs`),
`fim.py:150-152`; `discrimination.py:52,209-218` (docstring even says "must
be a subset").

**Problem:** `compute_fim(exp, pv, {"temperture": 300})` raises nothing; the
real design input floats free between bounds and the FIM is computed at an
arbitrary point. Propagates to `optimal_experiment` (identical FIM for every
candidate → meaningless "optimum") and the discrimination optimizer (scans a
variable no model consumes).

**Confirm:** Misspelled key: result identical to omitting the design value
entirely; no exception.

**Fix:** Raise on unknown keys in both modules; warn when design inputs are
left unfixed.

**Verify:** Typo tests raise `ValueError` naming the bad key and valid names.

### 31. Blanket exception swallowing converts real errors into "No feasible design point found" / arbitrary results [usability]

**Where:** `design.py:464-477` (`except Exception → None`s),
`discrimination.py:395-396` (`except Exception: return _SINGULAR_SENTINEL`
where the sentinel is *finite* 1e30, so the "all failed" guard never
triggers), `discrimination.py:671-695` (L-BFGS-B refine `except Exception:
pass`; `DiscriminationDesignResult.warnings` always `[]`), `cli.py:296-307`
(per-row `except Exception: continue`).

**Problem:** A misspelled parameter name, a shape bug, or a missing
dependency becomes `RuntimeError("No feasible design point found")` with the
root cause discarded — or worse, in discrimination the first random
candidate is returned as "best".

**Confirm:** Raise a deliberate `TypeError` inside a model callable; observe
the misleading error/result.

**Fix:** Chain the last exception (`raise ... from last_exc`); use `np.inf`
as the discrimination sentinel and count failures into `warnings`; log
skipped rows in the CLI helper.

**Verify:** Tests asserting the root cause appears in the raised chain and
`warnings` is populated on refine failure.

### 32. `fd_step` documented as "relative perturbation" but used as an absolute step [docs / correctness]

**Where:** `fim.py:236-238` vs `fim.py:908-911`.

**Problem:** For parameters of magnitude 1e6 (or 1e-8) the absolute 1e-5
step gives catastrophic cancellation (or ~100% perturbation); the FD path is
advertised for *validating* autodiff, so users cross-checking badly-scaled
models get misleading disagreement.

**Confirm:** FD-vs-autodiff comparison with a 1e6-scaled parameter.

**Fix:** `step * max(|x|, 1)` per component; fix the docstring.

**Verify:** FD≈autodiff test at extreme parameter scales.

### 33. Graeco/hyper-Graeco Latin squares fail for k = 8, 9, 10, 12, … with a misleading error [usability / docs]

**Where:** `latin.py:64-88,124-137,174-175`.

**Problem:** `_mols_prime_power` handles only primes plus hardcoded k=4
despite its name; composite k falls back to one cyclic square and the error
("k = 9 provides 1 MOLS, need 2") implies nonexistence — but MOLS(8)=7,
MOLS(9)=8, and Graeco-Latin squares exist for all k except 2 and 6 (the
docstring even says only 2 and 6 raise). Tests only cover k ∈ {3,4,5,7}.

**Confirm:** `graeco_latin_square(8)` raises.

**Fix:** Implement GF(p^m) MOLS (or hardcode 8, 9); make the error honest
("not supported by this construction").

**Verify:** Property tests (orthogonality) for k=8, 9; honest-error test for
k=10 if not implemented.

### 34. Acquisition dispatch swallows `TypeError` and silently drops `acquisition_kwargs`; built-in acquisitions absorb typos via `**_` [footgun]

**Where:** `optimize.py:279-282`, `model_based.py:440-443`,
`acquisition.py:97,109,122,137-138,154-161`.

**Problem:** `except TypeError: acq_fn(s, candidates, direction=direction)`
retries with the user's kwargs (`xi`, `kappa`) silently discarded — and
masks genuine `TypeError`s raised *inside* custom acquisitions. Separately,
`confidence_bound(..., **_)` silently absorbs `kapa=2.5`. Tuning knobs that
silently no-op are worse than crashes.

**Confirm:** `acquisition_kwargs={"kapa": 2.5}` with `"ucb"`: no error,
kappa stays 2.0.

**Fix:** Validate kwargs against the acquisition's signature
(`inspect.signature`) instead of try/except; drop `**_` from built-ins.

**Verify:** Typo'd kwarg raises with the valid names listed; custom-callable
path still works.

### 35. `extend`'s fallback cumulative FIM is evaluated at θ=0 instead of fitted/guess values [correctness]

**Where:** `cli.py:296-307` (`param_values = {name: 0.0 ...}` although
`_param_values_for_design` exists).

**Problem:** When the cached FIM is missing/stale for a nonlinear `--module`
experiment, the prior FIM is computed at θ=0 — meaningless for nonlinear
models — with silent per-row skips.

**Confirm:** Delete the FIM sheet from a module workbook; `extend`; inspect
prior.

**Fix:** Pass `_param_values_for_design(parameter_names, wb)`; log skipped
rows.

**Verify:** Unit test on the helper with fitted parameters present.

### CLI and workbook

### 36. `--json` output can be invalid JSON (`NaN`, `-Infinity`) [usability]

**Where:** `cli.py:594,674,871` (criterion_value NaN for factorial/latin/
optimize `new`), `cli.py:1143` (log_det can be −inf), dumps at
`cli.py:927,1012,1341,1541`.

**Problem:** `json.dumps` emits literal `NaN` — rejected by `jq`,
`JSON.parse`, and strict parsers. `discopt doe new factorial-2level --json`
always produces this.

**Confirm:** `discopt doe new factorial-2level ... --json | jq .` fails.

**Fix:** Sanitize NaN/±inf → `null` in one `_dump_json` helper used by all
verbs.

**Verify:** Test piping every verb's `--json` through `json.loads` with
`parse_constant` set to raise.

### 37. `status` and the embedded instructions sheet recommend `fit`/`extend` for workbooks where those verbs are guaranteed to fail — and the failure message says "unknown template 'optimize'" [usability]

**Where:** `cli.py:972-986` (template-blind `next_command`),
`workbook.py:75-111` (one-size instructions sheet), `workbook.py:604-628` +
`templates.py:317` (optimize template hits the generic "unknown template"
error).

**Problem:** Latin/factorial workbooks are told to `fit` (which raises "use
anova instead"); optimize workbooks are told to `fit`, which fails with
"unknown template 'optimize'" — confusing since the template is perfectly
known.

**Confirm:** `discopt doe status` on latin and optimize workbooks; run the
suggested commands.

**Fix:** Branch `next_command` on template family; template-specialize the
instructions blocks; add an explicit optimize guard in `rebuild_experiment`
directing to `discopt doe optimize`.

**Verify:** Status tests per template family asserting the suggested command
actually succeeds.

### 38. `extend`/`optimize` are blind to pending runs and reuse a fixed seed — repeated calls append duplicate design points silently [footgun]

**Where:** `cli.py:1459-1511` (completed-only prior + `seed=wb.seed()`
every call); analogous in `do_optimize`.

**Problem:** Two consecutive `extend --n 3` with no new data append the same
three conditions twice; `extend` right after `new` re-recommends points
overlapping the initial pending batch. No "N runs still pending" warning.

**Confirm:** `new` + `extend` + `extend`; diff the appended rows.

**Fix:** Include pending runs' FIM contribution in the prior (they will be
executed); warn when pending runs exist; vary seed per batch (seed + batch
index).

**Verify:** Test that consecutive extends produce distinct, non-overlapping
recommendations and emit the warning.

### 39. Inapplicable common flags (`--n`, `--criterion`, `--n-starts`, `--error`) are silently accepted and ignored for factorial/latin templates [footgun]

**Where:** `cli.py:1756` (`_add_common_new_options` on every subparser),
`cli.py:514-598,778-875`.

**Problem:** `discopt doe new latin-square --levels ... --n 100 --criterion
trace` ignores both; run count is `k²·replicates` regardless, and the help
text is wrong for these templates.

**Confirm:** Run the command above; row count unchanged.

**Fix:** Register only applicable options per subparser (or error when an
inapplicable flag is explicitly set).

**Verify:** Parser tests: inapplicable flag → argparse error.

### 40. No column-name collision validation: duplicate inputs, reserved names, or input == response silently corrupt the runs sheet (CLI, GUI, and Workbook.create) [correctness / footgun]

**Where:** `workbook.py:256-260,398-401` (`dict(zip(headers, row))` — last
duplicate wins), `cli.py:887-889`, GUI `app.py:1362-1383,1073-1089` (blank
factorial names silently skipped, duplicates silently collapse; a factor
named `y` collides with the default response and `_save_responses`
(`app.py:155`) then overwrites the factor column).

**Confirm:** `discopt doe new linear --input y:0:1` → header
`[run_id, batch, y, y, measured_at]`; fits read the wrong column.

**Fix:** Validate once in `Workbook.create`/`do_new` (uniqueness, disjoint
from `{run_id, batch, <response>, measured_at, replicate}`); mirror the
rename panel's rules in the GUI creation flows; error on blank names.

**Verify:** Creation tests for each collision class raising clear errors, at
both CLI and GUI (`_validate_factors`) layers.

### 41. User-edited workbooks crash verbs with raw tracebacks (`TypeError` on blanked cells, `PermissionError` on Excel-locked files) [usability]

**Where:** `cli.py:1305,1076-1083` (`float(row[nm])` → TypeError), catch
lists at `cli.py:1339,1597`; `wb.save()` PermissionError caught nowhere.

**Problem:** Blanking an input cell or leaving the file open in Excel — the
two most common practitioner mistakes — produce stack traces instead of
`error: ...` with exit 1.

**Confirm:** Blank an input cell → `fit`; open the file exclusively → any
writing verb.

**Fix:** Add `TypeError`/`PermissionError`(`OSError`) to the wrapper catches
with actionable messages ("run N has a blank value for input 'x'";
"close the workbook in Excel and retry").

**Verify:** Tests for both paths asserting message + exit code 1.

### 42. Install hints in code and skill docs point at extras that don't exist: `discopt[doe-gui]`, `discopt[doe]` [docs / usability]

**Where:** `cli.py:1906`, `gui/launcher.py:47-52`, `workbook.py:167`,
`skill/SKILL.md:30` — vs actual extras `discopt-doe[gui]` / `[ml]` and
openpyxl being a hard dep.

**Confirm:** grep `doe-gui\|'discopt\[` across `src/`.

**Fix:** Replace all with `pip install "discopt-doe[gui]"`; drop or correct
the openpyxl hint.

**Verify:** grep returns nothing; a test asserting `_missing_dep_message()`
contains `discopt-doe[gui]`.

### 43. Bundled skill/agent docs document APIs that don't exist — an agent following them emits broken code [docs]

**Where:** `skill/agents/identifiability-expert.md:38-97` (wrong
`profile_likelihood` kwargs `n_steps/step_factor/alpha`; wrong result fields
`theta_vals/deviance_vals/ci_lo/ci_hi`; wrong shape vocabulary; nonexistent
`diag.summary()`, `eigenvalue_range_log10`);
`model-discrimination-expert.md:43-87` (wrong `model_weights`,
`stop_when_concentrated`, per-model `initial_data`, `vuong_test` signature,
`lrt.g2_statistic/df`); `estimability-expert.md:61` (`rank.summary()`
doesn't exist); all four agents cite pre-extraction paths
(`python/discopt/doe/...`), `.crucible/wiki` articles, and expert agents
that don't ship.

**Confirm:** Run any documented snippet from those files → `TypeError`/
`AttributeError`.

**Fix:** Regenerate the API sections from current signatures; strip or
update stale paths/cross-references; consider a CI check that executes the
snippets in the agent docs.

**Verify:** Doctest-style test executing each documented snippet against a
toy experiment.

### 44. Version guard rejects valid discopt pre-releases like `0.6rc1` [correctness]

**Where:** `src/discopt/doe/__init__.py:67-75`.

**Problem:** `tuple(int(p) for p in _found.split(".")[:2] if p.isdigit())`
silently drops non-digit components: `"0.6rc1"` → `(0,)` < `(0, 6)` →
spurious ImportError "requires discopt>=0.6 (found 0.6rc1)".

**Confirm:** `python -c` reproducing the parse on `"0.6rc1"`, `"0.7b2"`.

**Fix:** `re.match(r"\d+", p)` per component, or `packaging.version` with a
lenient fallback.

**Verify:** Unit-test the parser helper on `0.5`, `0.6`, `0.6rc1`,
`0.6.0.dev0`, `0.10.1`.

### 45. Every `discopt doe` invocation prints "warning: ignoring discopt.cli plugin 'doe' … name already taken" [usability — partly upstream]

**Where:** `pyproject.toml:33-34` here; the pinned base-discopt branch still
declares `[discopt.cli] doe=discopt.doe.cli` in its own entry points.

**Problem:** The entry point is registered twice (both dists). It works only
because both point at the same module — on a base-only install of that
branch the plugin load would hard-fail (the module was removed there).
Alarming stderr noise on a new user's first command.

**Confirm:** Run any `discopt doe` command; check stderr.
`importlib.metadata.entry_points(group="discopt.cli")` shows the duplicate.

**Fix:** Upstream: remove the leftover entry point from the discopt
extraction branch. Defensive here/upstream: dedupe identical-value entry
points in the base CLI loader.

**Verify:** Clean stderr on `discopt doe templates` after the upstream fix.

### 46. Test suite fails on a no-extras install: 13 tests import sklearn without skip guards [testing]

**Where:** `tests/test_doe_optimize.py` (13 failures verified with
`uv sync --group dev` then `uv run pytest`).

**Confirm:** `uv sync --group dev && uv run pytest -q` (reproduced: 13
failed, 197 passed).

**Fix:** `sklearn = pytest.importorskip("sklearn")` (module-level or
per-test marker) for GP/adapter/EI tests; keep a no-extras CI leg honest
about what runs.

**Verify:** No-extras pytest run: 0 failures, 13+ skips. Consider adding
that leg to CI.

### GUI

### 47. Stale ANOVA results shown for the wrong workbook / stale data [footgun]

**Where:** `gui/app.py:1667-1671` (no workbook guard on
`last_anova_result`, unlike the optimize panel's guard at `:1855`).

**Confirm:** `AppTest`: run ANOVA on workbook A, open workbook B — A's table
renders.

**Fix:** Guard on `out.get("workbook_path") == str(Path(path))` (payload
already carries it); invalidate on response save.

**Verify:** The `AppTest` scenario shows no table for B until ANOVA is rerun.

### 48. GUI optimize seed defaults from a `status` key that doesn't exist — always 1, identical every round, diverging from the CLI [correctness / footgun]

**Where:** `gui/app.py:1809-1816` (`status.get("seed", 0) + 1`);
`cli.py:988-1003` (`do_status` returns no `"seed"`), CLI default is
`seed=None` (`cli.py:1880-1883`).

**Problem:** The Sobol candidate pool is identical round after round (the
recommender can only ever pick from one finite candidate set), and GUI
results silently diverge from CLI results.

**Confirm:** Read `do_status` return keys; the widget default is always 1.

**Fix:** Expose `wb.seed()` via `do_status` and use a round-dependent
default (e.g. seed + n_completed), or default to `None` like the CLI.

**Verify:** Unit test on the seed-default helper; two consecutive GUI rounds
propose different candidate pools.

### 49. Success/info messages are erased by immediate `st.rerun()` — saves, renames, and optimize rounds give no visible confirmation [usability]

**Where:** `gui/app.py:1597-1603,1554-1562,1850-1852`.

**Problem:** `st.rerun()` raises before the message renders; notably the
"fit artifacts cleared — run Fit again" notice after a rename is never seen.

**Confirm:** `AppTest`: trigger save; assert no success element in the next
render.

**Fix:** Stash a flash message in `session_state`, render+clear at the top
of the next run (or `st.toast`).

**Verify:** `AppTest` asserts the flash renders exactly once after the
action.

### 50. Opening a non-workbook or corrupt `.xlsx` bypasses the friendly-error/recovery path and dumps a raw traceback on every rerun [usability]

**Where:** `gui/app.py:65-71` (`_safe_status` catches only
`FileNotFoundError, ValueError, DoEError`); openpyxl raises
`InvalidFileException` / `zipfile.BadZipFile`.

**Confirm:** Point the GUI at a CSV renamed `.xlsx`.

**Fix:** Add those exception types (or broad `Exception` with surfaced
message) to `_safe_status` and mirror in `_runs_df`/`_read_anova_sheet`/
`_history_rows`, keeping the "Forget this workbook" recovery UI reachable.

**Verify:** Unit test `_safe_status` on a junk file returns the error tuple
rather than raising.

### Docs and packaging

### 51. README/intro advertise Plackett–Burman designs and Latin *hypercubes*; neither is implemented [docs]

**Where:** `README.md:15-16`, `docs/intro.md:17-22`; grep of `src/` finds PB
only in a comment and no LHS anywhere.

**Confirm:** grep.

**Fix:** Drop both from the feature lists (or implement them — PB is a small,
high-value addition for a screening package).

**Verify:** Docs no longer promise them; if implemented, standard PB/LHS
property tests.

### 52. `--surrogate` help advertises presets `'rf'` and `'linear'` that don't exist [docs]

**Where:** `cli.py:1836-1838` vs `surrogate.py:223-226`
(`PRESETS = {"gp", "response-surface"}`).

**Confirm:** `discopt doe optimize --surrogate rf` errors.

**Fix:** Correct the help string (or add the presets).

**Verify:** Help-text test against `PRESETS.keys()`.

### 53. Notebook build scripts have drifted from the committed notebooks — regenerating silently rewrites/deletes doc content [docs / footgun]

**Where:** `scripts/build_factor_screening_notebook.py:273` vs
`docs/notebooks/factor-screening.ipynb` (committed has an extra
fractional-factorial section the script would delete);
`build_latin_designs_notebook.py:110` (injects private `_build_squares`);
scripts emit output-free notebooks while `_config.yml` has
`execute_notebooks: "off"`. Only 4 of 10 notebooks have scripts; nothing
checks consistency.

**Confirm:** Run the build scripts and `git diff` the notebooks.

**Fix:** Pick one source of truth: either delete the stale scripts, or make
them canonical (regenerate + execute with nbclient in CI). Stop using
private APIs in generated docs.

**Verify:** CI step: regenerate + execute + `git diff --exit-code` (if
scripts are canonical).

### 54. The CLI-workflow chapter ships with zero outputs and the book never executes notebooks [docs]

**Where:** `docs/notebooks/doe_cli.ipynb` (all code cells `outputs: []`),
`docs/_config.yml:6-7` (`execute_notebooks: "off"`).

**Confirm:** Inspect the ipynb JSON; build the book.

**Fix:** Execute and commit `doe_cli.ipynb` with outputs, or switch the book
to `execute_notebooks: cache`. Add `jupyter-book build docs/ -W` to CI
(see #57).

**Verify:** Built page shows command outputs.

---

## P3 — polish

Compact format: **Confirm / Fix / Verify** in one line each.

55. **Use `slogdet`, not `det`+`log`** — `fim.py:63-68`, `design.py:294-297`.
    D-criterion overflows/underflows for badly scaled FIMs. Confirm: FIM with
    parameters spanning many decades → `det` inf/0. Fix: `np.linalg.slogdet`.
    Verify: extreme-scale test matches slogdet reference.
56. **`fractional_factorial_design` doc vs code on `n_runs`** —
    `fractional.py:106-110` says "power of 2 in [k+1, 2^k]"; code rejects
    `2^k` and enforces resolution-dependent minima (`:181-190`). Fix
    docstring. Verify: doctest.
57. **CI: one Python (3.12), one OS, no docs build** — `ci.yml:8-17` vs
    `requires-python>=3.10` (uv.lock even resolves different jax/numpy for
    <3.11). Fix: add a 3.10 leg + `jupyter-book build docs/ -W` + a
    no-extras leg (#46). Verify: green matrix.
58. **`_solve_row_milp` zero objective contradicts its comment** —
    `fractional.py:333-336`: `m.minimize(... * 0.0)`; uniqueness claim is
    false. Fix: drop the `* 0.0` or the comment. Verify: solver-portability
    smoke test.
59. **`AnovaEffect.p` documented "two-sided"** — `anova.py:52-53`; F-test
    upper-tail is one-sided. Fix docstring.
60. **`include_replicate=True` silently ignored when `factors` is passed
    explicitly** — `anova.py:143-147`. Fix: honor it or raise.
61. **`effects_estimates` docstring: center points "kept in the residual
    variance estimate"** — `screening.py:225-228`; they're excluded
    entirely; also dead `_ = grand_mean` (`:304`). Fix docstring, remove
    dead code.
62. **`optimize_round` mutates a user-supplied sklearn estimator in place;
    it ends trained on fantasy data** — `optimize.py:293-299`,
    `surrogate.py:121-127`. Fix: `sklearn.base.clone` in the adapter.
    Verify: user estimator unchanged after the call.
63. **`measurement_error=0` yields inf FIM unvalidated** — `fim.py:300-301`.
    Fix: raise on σ ≤ 0 (pairs with #11).
64. **`latin_square_design` supports 1 factor; docstring says 2–5** —
    `latin.py:206-215,247-248`. Fix docstring.
65. **`"ucb"`/`"lcb"` are direction-aware aliases of the same function,
    undocumented** — `acquisition.py:154-161`; `lcb` under maximization
    computes UCB. Also `lower/upper_confidence_bound` in module `__all__`
    but not re-exported at package level. Fix: document aliasing where
    `acquisition=` is described; align exports.
66. **Profile likelihood comment/doc nits** — `profile.py:70-72` (curvature
    derivation off by 2 — harmless, the adaptive scheme corrects it),
    `profile.py:145` ("chi2.ppf(alpha,1)" should say confidence level). Fix
    comments/docstrings.
67. **`sequential_doe` never re-estimates after the final round's data** —
    `sequential.py:112-186`; `history[-1].estimation` is one batch stale.
    Fix: document, or append a final estimation-only record.
68. **`ModelBasedRoundResult.log` is declared but never populated** —
    `model_based.py:135,481-493`. Fix: populate or remove.
69. **`do_new` silently overwrites existing workbooks** — guard lives only
    in `_cmd_new` (`cli.py:878-885`); `Workbook.create` truncates
    (`workbook.py:274-276`). GUI calls `do_new` directly. Fix: move the
    guard into `do_new`/`Workbook.create` with `overwrite=` flag. Verify:
    GUI create-over-existing test.
70. **`discopt doe new optimize` fails with the default `--n 1`** —
    `cli.py:1930` vs `cli.py:611-612` (requires n ≥ 2): the minimal
    documented invocation always errors. Fix: per-template default or help
    note. Verify: default invocation succeeds.
71. **`cli.__all__` omits `do_optimize`/`OptimizeParams`** —
    `cli.py:1976-1988`, despite the "do_<verb> functions are the GUI
    contract" docstring. Fix: add them.
72. **Flag-naming inconsistencies** — `--input NAME:LB:UB` vs the module
    subcommand's `--bounds` for the same concept (`cli.py:1769-1775`);
    `extend --n` vs `optimize --batch-size`; `new --seed` defaults 42 but
    `optimize --seed` defaults None. Fix: align names/defaults (breaking —
    do before any release), document.
73. **`status` fabricates numeric bounds for categorical factors** —
    `cli.py:822-825` prints `treatment in [0.0, 3.0]` for levels A–D. Fix:
    display actual levels for combinatorial templates.
74. **`workbook.py` module docstring: "parameters — written by fit and
    extend"** — `workbook.py:26-28`; extend never writes it. Fix docstring.
75. **`install.py` gaps** — no uninstall/manifest; `--project` resolves
    `./.claude` against CWD silently (`install.py:63`); `read_text/
    write_text` without `encoding="utf-8"` can mojibake on Windows locales
    (`install.py:30,43`). Fix: add encoding now; consider `--claude-dir` and
    an uninstall command.
76. **Formula injection on workbook write** — string levels/names beginning
    with `=` are stored as live formulas (`workbook.py:385`). Low risk
    (self-supplied values) but one-line fix: quote-prefix strings starting
    with `=`. Verify: `--factor 'cat:=A:=B'` round-trips as text.
77. **Stale "Five verbs" cli.py module docstring; README CLI section omits
    `optimize`/`anova`** — `cli.py:1-14`, `README.md:83-90` (parser
    registers eight verbs). Fix both lists.
78. **`test_doe_cli.py` docstring claims a parser test that doesn't exist;
    CLI verb coverage gaps** — no tests for `_cmd_*` wrappers, `--force`,
    `--json` validity (would have caught #36), exit codes; `do_new` untested
    for factorial-2level/optimize/scheffe templates. Fix: parser-level smoke
    test per verb + JSON validity + `--force` tests.
79. **GUI assorted** — mixture editor `ub` default doesn't follow "Sum to"
    (stale keyed state, `app.py:680-745` — include total in the editor key);
    `n_starts` default 5 vs CLI 10 (`app.py:803-811` vs `cli.py:1895-1898`
    — share a constant); surrogate diagnostic plots always fit a fresh GP
    regardless of configured surrogate (`app.py:1917-1987` — label or use
    the configured one); native file picker blocks the server thread up to
    1h and shows "(no file selected)" when zenity/kdialog are absent
    (`app.py:515-561` — detect availability up front); dead identical-
    branches conditional (`app.py:278`); mixed `use_container_width` /
    `width="stretch"` APIs (pairs with #17); `_free_port` TOCTOU race
    (`launcher.py:20-23`); renames and response saves bypass the workbook
    history sheet (`app.py:88-173` — log them); the GUI is essentially
    undocumented (two one-liners across README/intro — add a short docs
    page: launch flags, panels, rename-clears-fit behavior,
    `DISCOPT_DOE_WORKBOOK`).
80. **GUI test coverage** — `tests/test_doe_gui.py` covers launcher wiring
    and one static render; untested: `_save_responses` (highest-risk write
    path), `_rename_columns` (destructive), the browse buttons (#16),
    `_validate_factors` (#40), ANOVA/optimize panel flows. Fix: `AppTest`
    interaction tests + unit tests for the pure helpers.

---

## Suggested execution order for Opus

1. **Batch A — silent-corruption fixes (each small, independently
   testable):** #1, #2, #4, #7, #11, #35. Pure logic changes with clear
   regression tests.
2. **Batch B — string/doc one-liners (one commit):** #42, #44, #51, #52,
   #59, #61, #64, #66, #74, #77.
3. **Batch C — statistics correctness (needs care + references):** #5, #6,
   #8, #9, #19–#28. Write the reference-value tests *first*; several
   findings note the current tests only check argmax/monotonicity and can't
   detect wrong values.
4. **Batch D — input validation & error quality:** #30, #31, #34, #36,
   #39–#41, #37, #38.
5. **Batch E — packaging/CI/docs infrastructure:** #3, #14, #15, #17, #18,
   #46, #53, #54, #57.
6. **Batch F — GUI:** #16, #47–#50, then the #79/#80 batch.
7. **Batch G — skill docs regeneration:** #43, #45 (needs an upstream
   discopt change), #75.

Cross-cutting notes for the implementer:

- The workbook round-trip (generate → edit in Excel → import) is the
  package's core promise; #1, #2, #10, #11, #40, #41 all live on that path
  and deserve a shared integration test that simulates a user editing the
  file with openpyxl (formulas, blanks, locked file, collisions).
- Several statistical fixes (#19, #22, #26, #27) change numeric outputs.
  Land the reference tests first, then fix, so the diffs document the
  before/after values.
- Anything touching CLI flag semantics (#1, #38, #39, #72) is a behavior
  change — do it before the first release while there are no users to
  break.
