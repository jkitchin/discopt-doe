# Changelog

All notable changes to `discopt-doe` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
