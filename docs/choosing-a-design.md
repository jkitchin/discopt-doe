# Choosing a design

Every design in this package answers a different question, and picking the
wrong one is the most expensive mistake in the whole workflow — you find out
only after the experiments are run. This page is the router: start from what
you want to know, and it points at the design, the option settings that matter,
and the notebook that works it through.

The design names below are exactly the ones in the **Design type** menu of the
[browser app](https://jkitchin.github.io/discopt-doe/app/), and the same names
the CLI takes as `discopt doe new --template NAME`.

## Start here

| What you want to know | Design | In the app |
| --- | --- | --- |
| Which of these factors actually matter? | 2-level factorial | `factorial-2level` |
| …and something I cannot hold constant is also varying | Latin square family | `latin-square`, `graeco-latin`, `hyper-graeco-latin` |
| Where is the optimum, and what does the surface look like near it? | Central composite, Box-Behnken | `central-composite`, `box-behnken` |
| Nothing yet — I want coverage before I commit to a model | Latin hypercube | `latin-hypercube` |
| How precisely can I pin down *this* model's coefficients? | Model-based optimal | `linear`, `polynomial-1d`, `response-surface-2d`, `response-surface-3d` |
| Same, but my factors are proportions of a blend | Scheffé mixture | `scheffe-linear`, `scheffe-quadratic`, `scheffe-special-cubic` |
| Same, but my model is not in that list | Symbolic | `symbolic` |
| What is the best operating point? (not: what is the model?) | Active learning | CLI: `discopt doe optimize` |
| Which of two rival models is right? | Discrimination design | Python: `discriminate_design` |

Two rows are deliberately not in the browser app. Active learning needs a
surrogate refit between rounds, and discrimination needs the base `discopt`
solver; both are one command or one function call away at the keyboard. See
{doc}`the CLI workflow <notebooks/doe_cli>`.

## Which factors matter? — screening

**`factorial-2level`.** Run every combination of each factor at a low and a
high level. With $k$ factors that is $2^k$ runs, and from them you get a signed
estimate of every main effect and every interaction, plus an ANOVA table
{cite:p}`BoxHunter2005,Montgomery2017`. Levels may be numbers or strings, so
"catalyst A vs catalyst B" is a factor like any other.

Use it *first*, before fitting any response surface. A quadratic model in five
factors costs 21 coefficients; a screening run that shows three of the five do
nothing turns that into a 10-coefficient model over a smaller box.

- **Centre points** (numeric factors only) buy you two things for a handful of
  extra runs: a pure-error estimate that does not assume the model is right,
  and a curvature check — if the centre response sits well off the plane
  through the corners, a 2-level design cannot describe your system and you
  want a response surface.
- **Replicates** repeat the whole design. Replicate when your measurement noise
  is the thing you are unsure about; add centre points when the *model form* is
  what you are unsure about.
- $k \gtrsim 7$ makes the full factorial unaffordable (128 runs). Drop to a
  fractional design — `discopt.doe.fractional_factorial_design` selects the
  fraction by MILP at a resolution you request {cite:p}`PlackettBurman1946`.
  That one is Python-only: it needs the base `discopt` solver, which has no
  WebAssembly build.

Worked through in {doc}`notebooks/factor-screening`.

## Something else is varying — blocking

**`latin-square` (3 factors), `graeco-latin` (4), `hyper-graeco-latin` (5).**
When a nuisance variable — day, operator, batch of feedstock, position in the
furnace — will vary whether you like it or not, a square design arranges the
runs so that its effect is *estimated and removed* rather than left in the
residual {cite:p}`Fisher1935,Cochran1957`. One factor is the treatment you care
about; the rest are blocks.

The cost is severe constraints on shape. Every factor must have the same number
of levels $k$; a Latin square is then $k^2$ runs instead of $k^3$, which is the
whole point. Graeco-Latin squares need $k \ge 3$ and $k \ne 6$;
hyper-Graeco-Latin needs three mutually orthogonal Latin squares, so
$k \in \{4, 5, 7, 8, \ldots\}$.

```{warning}
A single Latin square cannot test an interaction. Row and treatment together
determine column, so the row×treatment subspace already contains the column
main effect — the terms are aliased, and `anova_report` refuses the fit rather
than reporting an F-ratio for a term that is partly some other term. Getting an
interaction means giving something up: replicate the square and block on
`replicate`, or drop one block from the model and spend its degrees of freedom
on the interaction. Both are shown in {doc}`notebooks/latin-designs`.
```

## Where is the optimum? — response surfaces

Both designs here fit a full quadratic — intercept, main, square and cross
terms — which is the cheapest model that has a stationary point to find
{cite:p}`Box1951,Montgomery2017`.

**`central-composite`** is the classic: a $2^k$ factorial core, $2k$ axial
points, and centre replicates.

- **Axial distance** `rotatable` makes prediction variance depend only on
  distance from the centre, which is what you want when you do not know which
  direction matters. `face` puts the axial points on the face centres, giving a
  3-level design.
- By default every run stays inside the bounds you typed. Tick **axial points
  outside bounds** for the textbook scaling, where the axial runs sit beyond
  them — only if those settings are actually reachable.

**`box-behnken`** never visits a corner of the design box: every run sits at an
edge midpoint, so no single run combines the extreme level of *every* factor
{cite:p}`BoxBehnken1960`. Choose it when the corners are dangerous, infeasible,
or simply outside what the equipment will do. It needs 3–5 factors (a central
composite handles 2–6).

## No model yet — space filling

**`latin-hypercube`.** A stratified sample over the continuous box: each factor
is split into $n$ equal-probability strata and each stratum is used exactly
once, so the sample covers every factor's range no matter how many factors
there are {cite:p}`McKay1979`. Unlike every other design here, *you* set the
run count — it is not dictated by the factor count.

Reach for it when you want coverage without committing to a model form: fitting
a surrogate, mapping a region before deciding where to look closely, or feeding
an active-learning loop. The **fit model** option (`linear` or `quadratic`)
only tells the later `fit` step which basis to estimate; it does not change the
sample.

## Estimate a known model precisely — optimal design

`linear`, `polynomial-1d`, `response-surface-2d`, `response-surface-3d`.

These do not spread runs out for coverage — they place them where they carry
the most information about the coefficients of a model you have already
committed to, by maximizing a scalar function of the Fisher information matrix
{cite:p}`Atkinson2007,Franceschini2008,KieferWolfowitz1959`. Expect the answer
to look sparse and repetitive; a D-optimal design for a straight line puts half
its runs at each end of the range, because that is genuinely the best way to
estimate a slope.

| Template | Model | Factors | Coefficients |
| --- | --- | --- | --- |
| `linear` | $b_0 + \sum_i b_i x_i$ | 1–8 | $1 + k$ |
| `polynomial-1d` | $\sum_{j=0}^{d} b_j x^j$ | exactly 1 | $d + 1$ |
| `response-surface-2d` | full quadratic | exactly 2 | 6 |
| `response-surface-3d` | full quadratic | exactly 3 | 10 |

Set the run count to at least the coefficient count — with fewer runs than
coefficients the information matrix is singular and standard errors are
meaningless. A little above it is where these designs pay off.

The theory is in {doc}`notebooks/model_based_doe`.

## Blends — mixture designs

`scheffe-linear`, `scheffe-quadratic`, `scheffe-special-cubic`.

When the factors are proportions of a formulation they are not free: they sum
to a fixed total, so the design space is a simplex rather than a box, and the
usual polynomial has no intercept {cite:p}`Scheffe1958,Scheffe1963,Cornell2002`.
Set **mixture total** to whatever your components sum to (1.0 for fractions,
100 for percentages).

With $q$ components:

| Template | Terms | Coefficients |
| --- | --- | --- |
| `scheffe-linear` | pure blends only | $q$ |
| `scheffe-quadratic` | + pairwise blending | $q + \binom{q}{2}$ |
| `scheffe-special-cubic` | + three-way blending | $q + \binom{q}{2} + \binom{q}{3}$ |

Start linear unless you have reason to expect synergy between components; the
quadratic terms are what a *blend* being better than either ingredient alone
looks like. See {doc}`notebooks/mixture-designs`.

## Your own model

**`symbolic`.** Type the response as an expression — `k0 * exp(-Ea / (8.314 *
T))` — list its parameters with nominal values, and it is differentiated
symbolically to build a design that estimates *those* parameters as precisely
as possible.

```{important}
A nonlinear model's information depends on the parameter values, so the design
is only optimal *around* the numbers you supply. This is not a flaw to work
around, it is the loop: design around a guess, run it, fit, then re-centre the
next batch on the fitted values with `discopt doe extend`
{cite:p}`franceschini-macchietto-2008,galvanin2009-online`. Several turns of
that loop are worked through in {doc}`notebooks/model-based-active-learning`.
```

Two practical notes. Parameter *scale* matters — an activation energy in J/mol
is ~$10^5$ while a pre-exponential may be ~1, and criteria computed from a
matrix spanning ten orders of magnitude are numerically fragile; prefer
parameters of comparable size, e.g. fitting $E_a/R$ in kelvin. And check
{doc}`notebooks/identifiability-estimability` before designing for a model with
many parameters: if two of them only ever appear as a product, no design will
separate them, and the diagnostics say so before you spend the runs.

## Beyond a single design

**Optimize, don't model.** If you want the best operating point and do not care
about the model that gets you there, run active-learning rounds instead: fit a
surrogate to what you have, propose the next batch by an acquisition function,
repeat {cite:p}`Jones1998,Snoek2012,Rasmussen2006`. That is `discopt doe
optimize`, worked through in {doc}`notebooks/active-learning`.

**Several runs at once.** With parallel capacity, design the batch jointly
rather than one point at a time — the information matrix is additive over
independent runs, so a joint design beats picking the best point $N$ times
{cite:p}`Galvanin2007,Sandrin2025`. See {doc}`notebooks/tutorial_batch_doe`.

**Two rival models.** When the question is *which* model rather than *what
parameters*, design for the condition where the models disagree most relative
to their prediction uncertainty {cite:p}`hunter1965,buzzi-ferraris-1984`. See
{doc}`notebooks/model-discrimination`.

## Which criterion?

The optimal-design templates take a **criterion** — a single number summarising
"how much information", since a matrix has no natural ordering
{cite:p}`Atkinson2007`.

| Option | Optimizes | Choose it when |
| --- | --- | --- |
| `determinant` (D) | maximize $\log\det\mathrm{FIM}$ | Default. Minimizes the volume of the joint confidence ellipsoid — the best all-round answer, and invariant to rescaling the parameters. |
| `trace` (A) | minimize $\operatorname{tr}\mathrm{FIM}^{-1}$ | You care about the average of the individual parameter variances rather than their joint volume. |
| `min_eigenvalue` (E) | maximize $\lambda_{\min}$ | One combination of parameters is much worse determined than the rest and you want to fix the worst case. |
| `condition_number` (ME) | minimize $\lambda_{\max}/\lambda_{\min}$ | The parameters are badly correlated and you want a *balanced* design more than a maximally informative one. |

Start with `determinant`. Switch to `min_eigenvalue` or `condition_number` when
a fit comes back with one enormous standard error, or with two parameters whose
errors are nearly perfectly correlated — that is the geometry those criteria
are for.

## How many runs?

- **Screening**: $2^k$, times replicates, plus centre points. Non-negotiable —
  the structure is the design.
- **Latin square family**: $k^2$ per replicate, where $k$ is the shared level
  count.
- **Response surface**: set by the template ($2^k + 2k +$ centres for a central
  composite).
- **Latin hypercube**: your call. A common starting point for surrogate work is
  10 runs per factor.
- **Optimal designs**: at least the coefficient count from the tables above, and
  more if you want to test lack of fit rather than only estimate coefficients.

Everywhere: below the parameter count you get no standard errors, and at
exactly the parameter count you get a perfect fit with zero residual degrees of
freedom, which tells you nothing about whether the model is right.

## After the experiments

Fill in the response column and drop the workbook back on the app — the fit and
the ANOVA run on their own. Both are reported where they apply: a model-based
design gets a coefficient table with confidence intervals, a Latin square gets
the factor-level F-table, and whichever does not apply says why. At the command
line the same steps are `discopt doe fit` and `discopt doe anova`, plus
`discopt doe extend` to design the next batch from what you now know
({doc}`notebooks/doe_cli`).

## Four ways this goes wrong

1. **Reading an interaction out of a single Latin square.** It is aliased with a
   block; the guard refuses it. Replicate, or give up a block.
2. **Trusting a locally-optimal design far from the truth.** For a nonlinear
   model the design is optimal around your nominal parameters and nowhere else.
   Design, fit, re-centre — do not run a hundred points off one guess.
3. **Fitting different conditions as replicates.** In a sequential loop, a runner
   that always returns the response under the same key hands the estimator a
   pile of values at one nominal condition, throwing away the very conditions
   the design chose. Give each observation its own response key
   ({doc}`notebooks/tutorial_batch_doe`).
4. **Skipping the identifiability check.** Diagnostics on the model you already
   have cost nothing; a batch of experiments that cannot separate two parameters
   costs a week ({doc}`notebooks/identifiability-estimability`).

## Further reading

- {cite:t}`BoxHunter2005` and {cite:t}`Montgomery2017` — the two standard texts
  for the classical designs; either covers screening, blocking and response
  surfaces end to end.
- {cite:t}`Atkinson2007` — optimal design: the criteria, the equivalence
  theorem, and what D-optimality actually buys.
- {cite:t}`Cornell2002` — mixtures, in far more depth than the Scheffé papers.
- {cite:t}`Franceschini2008` and {cite:t}`geremia2026-review` — model-based
  design for parameter precision, state of the art and where it is going.
- {cite:t}`raue2009-profile` and {cite:t}`yao2003-estimability` — what to do
  when a model has more parameters than the data can support.

Full bibliography in the {doc}`references`.
