# discopt-doe

[![ci](https://github.com/jkitchin/discopt-doe/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jkitchin/discopt-doe/actions/workflows/ci.yml)

Design-of-experiments plugin for the [discopt](https://github.com/jkitchin/discopt)
modeling language. Installs as the `discopt.doe` namespace package — code written
against `from discopt.doe import ...` works unchanged.

Three complementary entry points:

1. **"What is the best operating condition?"** — active-learning optimization:
   fit a surrogate to completed experiments, recommend the next batch via an
   acquisition function (EI, UCB, steepest ascent), track everything in an
   Excel workbook.
2. **"Does this factor matter?"** — classical designs: 2-level full/fractional
   factorials, mixture designs, Latin/Graeco-Latin squares, main-effect
   estimates, ANOVA.
3. **"How precisely can I estimate the model parameters?"** — model-based DoE:
   exact D/A/E-optimal design from the Fisher Information Matrix (JAX
   autodiff), identifiability/estimability diagnostics, profile likelihood,
   model discrimination, sequential estimate–design loops.

Parameter estimation (`discopt.estimate`) lives in the base package; both
share the same `Experiment` interface.

## Install

> **Pre-release:** neither `discopt-doe` nor its `discopt>=0.6` dependency is
> on PyPI yet, so the plain `pip install discopt-doe` shown below does **not
> work today** — it can't resolve `discopt>=0.6`. Until the 0.6 release, use
> the uv or git install below.

Recommended (uv, resolves the pinned `discopt` automatically):

```bash
git clone https://github.com/jkitchin/discopt-doe
cd discopt-doe
uv sync --all-extras        # core + gui + ml + dev
```

With pip, install the pre-release `discopt` first, then this package:

```bash
pip install "git+https://github.com/jkitchin/discopt@refactor/389-extract-doe"
pip install "git+https://github.com/jkitchin/discopt-doe"          # core
pip install "git+https://github.com/jkitchin/discopt-doe#egg=discopt-doe[gui]"  # + GUI
```

Once `discopt>=0.6` and `discopt-doe` are published, the usual form applies:

```bash
pip install discopt-doe            # core
pip install "discopt-doe[gui]"     # + Streamlit workbook GUI
pip install "discopt-doe[ml]"      # + scikit-learn surrogates
```

Requires `discopt>=0.6` (the first release with the public
`discopt.parametric` API and the CLI plugin hook).

## Quick start

FIM-based optimal design:

```python
from discopt.doe import compute_fim, optimal_experiment, DesignCriterion

fim = compute_fim(experiment, param_values={"k": 2.0}, design_values={"T": 350.0})
design = optimal_experiment(
    experiment,
    param_values={"k": 2.0},
    design_bounds={"T": (300, 400)},
    criterion=DesignCriterion.D_OPTIMAL,
)
print(design.summary())
```

Active-learning round against a workbook:

```python
from discopt.doe import optimize_round, OptimizationCriterion

result = optimize_round(
    workbook="opt.xlsx",
    criterion=OptimizationCriterion.MAXIMIZE,
    surrogate="gp",
    acquisition="expected_improvement",
    batch_size=4,
)
print(result.next_designs)
```

Identifiability diagnostics before you spend beam time:

```python
from discopt.doe import diagnose_identifiability, estimability_rank

diag = diagnose_identifiability(experiment, param_values)
est = estimability_rank(experiment, param_values)
```

## Command line

Installing this package adds the `doe` subcommand to the base discopt CLI
(via the `"discopt.cli"` entry-point group):

```bash
discopt doe templates                 # list workbook templates
discopt doe new linear -o run.xlsx --input T:300:400 --n 8
discopt doe status run.xlsx
discopt doe fit run.xlsx              # fit parameters to completed rows
discopt doe anova run.xlsx           # ANOVA F-table (latin/factorial designs)
discopt doe extend run.xlsx --n 4     # design the next batch
discopt doe optimize run.xlsx        # one active-learning round (optimize template)
discopt doe gui run.xlsx              # Streamlit GUI (needs [gui] extra)
```

## Development

The project is uv-managed:

```bash
uv sync --all-extras   # create .venv with everything
uv run pytest          # fast test set (slow tests excluded by default)
uv run pytest -m ''    # everything
```

To develop against a local discopt checkout instead of the PyPI release, add
a `[tool.uv.sources]` override pointing `discopt` at your editable clone:

```toml
[tool.uv.sources]
discopt = { path = "../../projects/discopt", editable = true }
```

Docs are a Jupyter Book:

```bash
uv run jupyter-book build docs/
```

## Claude Code skill

The package bundles a `/discopt-doe` skill and four expert agents (DoE,
estimability, identifiability, model discrimination) for Claude Code:

```bash
discopt-doe-install-skill            # install to ~/.claude
discopt-doe-install-skill --project  # install to ./.claude (versioned)
```

## Literature

The methods implemented here are referenced throughout the docs; the full
bibliography is in [docs/references.bib](docs/references.bib) (Atkinson &
Donev optimal design, Franceschini & Macchietto MBDoE, Yao estimability,
Raue profile likelihood, Hunter–Reiner/Buzzi-Ferraris discrimination, and
more).
