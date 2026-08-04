# Changelog

All notable changes to `discopt-doe` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
