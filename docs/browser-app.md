# The browser app

There is a
**[browser version of the design workflow](https://jkitchin.github.io/discopt-doe/app/)**
that needs no installation at all — no Python, no uv, no compiler. It runs in your
tab: pick a design, download the spreadsheet, run your experiments, fill in the
response column, and upload it back for fitting and ANOVA. Nothing is sent to a
server; there is no server.

It is the right starting point when you want to hand a colleague a design
without first walking them through a Python install, or when you are on a
machine you cannot install software on.

## What it does

**Design.** Choose a design type, name your factors and their ranges, and it
generates the run list as a `.xlsx` campaign workbook — the same format
{doc}`the CLI <notebooks/doe_cli>` reads and writes.

- **Space-filling and response surface** — Latin hypercube, central composite,
  Box-Behnken.
- **Screening and blocking** — 2-level factorial, Latin / Graeco-Latin /
  hyper-Graeco-Latin squares.
- **Model-based optimal** — linear, 1-D polynomial, 2- and 3-factor response
  surface, and the three Scheffé mixture models, each designed by maximizing
  the chosen information criterion (D, A, E, or modified-E).

**Analyze.** Upload the filled-in workbook to fit the model by least squares —
coefficients with standard errors and 95% confidence intervals — or run ANOVA
over the completed runs.

Workbooks move freely in both directions: a design generated in the browser
opens with `discopt doe status`, and a campaign started at the command line can
be analysed in the browser.

## What it does not do

`extend` (design the next batch given what you have measured so far) and
`optimize` (active-learning rounds against a surrogate) are command-line only
for now, as is the `--module` escape hatch for user-defined models. See
{doc}`the CLI workflow <notebooks/doe_cli>` for those.

## How it works, and why the limits are where they are

The app is [Pyodide](https://pyodide.org) — CPython compiled to WebAssembly —
running the real `discopt.doe` package, not a reimplementation. The design you
download is produced by the same code path the CLI uses.

The one thing it cannot do is load the base `discopt` package: that ships only
platform wheels and depends on jax, jaxlib, and a native solver, none of which
have WebAssembly builds. Since `discopt.doe` installs as a namespace
subpackage, it works there on its own — provided nothing it needs reaches into
the base package at import time, which `tests/test_import_hygiene.py` enforces.

That would ordinarily rule out the model-based optimal designs, since those are
built from a Fisher information matrix computed by jax autodiff. They work
anyway because of a property of the templates themselves: every one is *linear
in its parameters*, of the form

$$y(x, \theta) = f(x)^\mathsf{T}\theta$$

so the sensitivity Jacobian $\partial y/\partial\theta$ is just the
basis-function row $f(x)$, independent of $\theta$, and the information matrix
is exactly

$$\mathrm{FIM} = X^\mathsf{T}X/\sigma^2, \qquad X = [f(x_1); \ldots; f(x_n)].$$

No differentiation is involved. `discopt.doe.linear_design` computes this
directly in numpy, and the test suite pins it against the autodiff result to
machine precision. Designing still means optimizing the criterion over the
design box — that part is `scipy.optimize`, which Pyodide does have.

Nonlinear models are where this stops: for those the Jacobian genuinely depends
on the parameter values and you need autodiff, so they remain a desktop
feature.

## Running it locally

```bash
uv run python scripts/build_web_app.py --serve
```

That builds the wheel, assembles the app around it, and serves the result.
Pyodide needs real HTTP requests, so opening `index.html` from the filesystem
will not work.

The published app is rebuilt from the repository on every push to `main`, so it
always matches the current source rather than the latest PyPI release.
