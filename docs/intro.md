# discopt-doe

**discopt-doe** is the design-of-experiments plugin for the
[discopt](https://github.com/jkitchin/discopt) modeling language. It installs
as the `discopt.doe` namespace package, so `from discopt.doe import ...` works
exactly as it did when the module lived in the base package.

The package has three complementary entry points, each tailored to a
different question:

1. **"What is the best operating condition?"** — `optimize_round` runs one
   active-learning round: fit a surrogate to the experiments completed so
   far, recommend the next batch via an acquisition function (expected
   improvement, UCB, steepest ascent) {cite:p}`Jones1998,Snoek2012`, and
   append the batch to an Excel workbook for execution.

2. **"Does this factor matter?"** — 2-level full and fractional factorial
   screening designs {cite:p}`BoxHunter2005,Montgomery2017`, Plackett–Burman
   designs {cite:p}`PlackettBurman1946`, signed main-effect estimates, and
   ANOVA F-tables {cite:p}`Fisher1935`. Mixture designs
   {cite:p}`Scheffe1958,Cornell2002` and Latin squares/hypercubes
   {cite:p}`Cochran1957` round out the classical toolbox.

3. **"How precisely can I estimate the model parameters?"** — exact
   D/A/E-optimal design {cite:p}`Atkinson2007,Franceschini2008` using the
   Fisher Information Matrix computed with JAX autodiff, plus
   identifiability {cite:p}`miao2011-identifiability,raue2009-profile` and
   estimability {cite:p}`yao2003-estimability,brun2001-collinearity`
   diagnostics, model discrimination
   {cite:p}`hunter1965,buzzi-ferraris-1984`, and sequential model-based DoE
   {cite:p}`galvanin2009-online,franceschini-macchietto-2008`.

Parameter estimation itself (`discopt.estimate`) lives in the base discopt
package; both share the same `Experiment` interface.

## Install

```{warning}
Pre-release: neither `discopt-doe` nor its `discopt>=0.6` dependency is on
PyPI yet, so `pip install discopt-doe` cannot resolve `discopt>=0.6` today.
Until the 0.6 release, use the uv or git install below.
```

Recommended (uv resolves the pinned `discopt` automatically):

```bash
git clone https://github.com/jkitchin/discopt-doe
cd discopt-doe
uv sync --all-extras        # core + gui + ml + dev
```

With pip, install the pre-release `discopt` first, then this package:

```bash
pip install "git+https://github.com/jkitchin/discopt@refactor/389-extract-doe"
pip install "git+https://github.com/jkitchin/discopt-doe"
```

Once both are published, the usual form applies:

```bash
pip install discopt-doe            # core: FIM design, screening, workbooks
pip install "discopt-doe[gui]"     # + Streamlit workbook GUI
pip install "discopt-doe[ml]"      # + scikit-learn surrogates for active learning
```

The `discopt doe ...` command-line workflow (workbook campaigns:
`templates`, `new`, `status`, `fit`, `extend`, `optimize`, `anova`, `gui`)
registers itself with the base discopt CLI automatically when this package
is installed.

## Where to start

- New to DoE with discopt? Start with the {doc}`tutorial <notebooks/tutorial_doe>`.
- Running lab campaigns from a spreadsheet? See the {doc}`CLI workflow <notebooks/doe_cli>`.
- Optimizing a process with unknown model structure? See
  {doc}`active learning <notebooks/active-learning>`.
- Estimating parameters precisely? See
  {doc}`model-based DoE <notebooks/model_based_doe>`.
