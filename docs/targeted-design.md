# Targeted design: precision where you need it

A model often has many parameters, the experimental budget is small, and the
question is about only a few of them, or about quantities computed from
them, such as a yield at the plant's operating point. A D-optimal design spreads
its information over *every* parameter, so part of the budget goes to
parameters nobody asked about. A **targeted** design puts the budget on
weighted precision of the quantities you name.

The idea has a long history in optimal-design theory. Its parts are known as
$D_s$-optimality (a parameter subset), L- or weighted A-optimality (weights over
parameters), c-optimality (one linear combination), $D_A$-optimality (several
combinations) {cite:p}`Atkinson2007`, and, for model outputs, goal-oriented
optimal design {cite:p}`attia2018-goal-oriented`. `discopt.doe` implements all of
them through a single object, following {cite:t}`pukelsheim2006`.

The {doc}`worked example <notebooks/targeted-design-example>` is a four-parameter
kinetic model with a question about the plant's operating point. It shows
everything on this page at work: four targeted designs against D-optimal,
certified efficiencies, a sequential campaign that stops at a precision goal in
half the runs D-optimal needs, robust design, and the command line.

## One matrix for every kind of target

Let $M$ be the Fisher information matrix of the $p$ parameters $\theta$, and
$\psi = g(\theta)$ the $s$ quantities you care about. Let

$$K = \frac{\partial g}{\partial \theta} \in \mathbb{R}^{p \times s}$$

be taken at the current estimates. The delta method gives the predicted covariance of the targets:

$$\Sigma_\psi = K^\top M^{-1} K .$$

$K$ depends on the parameter estimates but **not on the design**, so it is
computed once per round, and each candidate design costs one factorization of $M$.

| You want | Target | $K$ |
|---|---|---|
| some parameters; the rest are nuisance | `Target.parameters(["k1", "Ea1"])` | columns of $I$ |
| graded importance over parameters | `Target.parameters({"k1": 1, "k2": 0.1})` | $I$, with weights |
| linear combinations | `Target.linear({"k1-k2": {"k1": 1, "k2": -1}})` | the coefficients |
| any function of the parameters | `Target.function(lambda th: th["k1"] / th["k2"], ["ratio"])` | its JAX Jacobian |
| the same, written as a formula | `Target.expression({"ratio": "k1/k2"})` | its sympy gradient |
| model outputs at operating points | `Target.responses(qoi_model, at=[{"T": 350}])` | $(\partial q / \partial \theta)^\top$ |
| predictions over a region (I, G) | `Target.region(bounds=...)` | sensitivity rows |

Targets add together: `Target.parameters(...) + Target.responses(...)` stacks the
columns of $K$.

## Criteria on the targets

The `criterion` argument keeps its meaning but is applied to the targets. With
weights $W = \operatorname{diag}(w)$ and
$\tilde\Sigma = W^{1/2} \Sigma_\psi W^{1/2}$:

| `criterion` | value | classical name |
|---|---|---|
| `"trace"` | $\operatorname{tr}(W \Sigma_\psi) = \sum_j w_j \operatorname{Var}(\hat\psi_j)$ (min) | L, weighted A, $A_s$, c |
| `"average_variance"` | $\operatorname{tr}(W \Sigma_\psi) / \sum_j w_j$ (min) | I on a region |
| `"determinant"` | $-\log\det \tilde\Sigma$ (max) | $D_s$, $D_A$ |
| `"min_eigenvalue"` | $1/\lambda_{\max}(\tilde\Sigma)$ (max) | $E_s$ |
| `"max_variance"` | $\max_j w_j \operatorname{Var}(\hat\psi_j)$ (min) | minimax; G on a region |

Some properties of these criteria:

- **Nuisance parameters are marginalized, not dropped.** For a parameter subset
  $S$, $(K^\top M^{-1} K)^{-1} = M_{SS} - M_{SN} M_{NN}^{-1} M_{NS}$, the Schur
  complement. A run that only decorrelates a target from a nuisance parameter
  still counts. Deleting the nuisance rows of $M$ instead overstates what is
  known.
- **Weights.** A weight of 0 makes a parameter a nuisance. Under `"determinant"`
  diagonal weights only shift $\log\det$ by a constant, so only *which* targets
  are included matters, and graded weights trigger a warning. Weights act
  through `"trace"`, `"min_eigenvalue"` and `"max_variance"`.
- **Tolerances.** `tolerances={"k1": 0.05}` sets $w_j = 1/\tau_j^2$. `"trace"` then
  becomes $\sum_j (\sigma_j/\tau_j)^2$ and `"max_variance"` becomes
  $\max_j (\sigma_j/\tau_j)^2$. Both are dimensionless, and "every ratio
  $\le 1$" is a natural stopping rule. With `relative=True` a tolerance is a
  fraction of the nominal value (0.05 = 5 %).
- **Units.** Targets of different kinds (parameters, outputs, functions) are in
  different units, so combining them requires tolerances or `relative=True`.
  Targeted criteria do not change under parameter rescaling ($K$ and $M$
  transform together), so `scale_parameters` is ignored when a target is given.
- **Singular $M$.** A target is estimable when
  $\operatorname{range}(K) \subseteq \operatorname{range}(M)$. The covariance
  uses the pseudo-inverse on the identified subspace, so a design can be
  optimal for the targets while $M$ is singular overall. This is the common
  case with many parameters and few experiments. If a target is not estimable,
  its criterion is infinite. If no single run can make the targets estimable,
  `optimal_experiment` stops and suggests a `prior_fim` or a batch.

## Using it

```python
from discopt.doe import Target, optimal_experiment, batch_optimal_experiment

# The rate constant only; the amplitude is a nuisance parameter.
res = optimal_experiment(
    experiment, theta, {"t": (0.05, 10.0)},
    criterion="trace", target=Target.parameters(["k"]), prior_fim=prior,
)
print(res.target_precision().summary())

# A prediction at t = 6 (a quantity of interest), to 1 % of its value.
t = Target.responses(at=[{"t": 6.0}], tolerances={"y": 0.01}, relative=True)
batch = batch_optimal_experiment(
    experiment, theta, {"t": (0.05, 10.0)}, 3, criterion="trace", target=t,
)
```

For the decay model $y = A e^{-kt}$ with $A = 2$, $k = 0.5$ and one run already at
$t = 0.3$, the best next sampling time depends on the question (brute force over
$t \in [0.05, 10]$):

| target | criterion | next $t$ |
|---|---|---|
| both parameters | D | 2.29 |
| amplitude $A$ | `"trace"` on `["A"]` | 0.05 |
| rate $k$ | `"trace"` on `["k"]` | 2.51 |
| prediction $y(6)$ | `"trace"` on `responses(at=[{"t": 6}])` | 2.45 |

Moving the prior run changes this in ways that are hard to guess. With the first
run at $t = 0.05$, the best next run for the *amplitude* is at $t \approx 4.4$:
an early sample already measures $A$, and the remaining uncertainty in $A$ comes
from its correlation with $k$, which a later sample removes. This is the Schur
complement at work.

### Sequential campaigns with a precision goal

`sequential_doe` re-resolves the target at every round's estimate, because output
and function targets move with the parameters. It records each round's
`target_precision` and can stop as soon as the goal is met:

```python
history = sequential_doe(
    model, None, guess, bounds, initial_runs=runs, n_rounds=10,
    run_experiment=run, criterion="trace",
    target=Target.parameters(tolerances={"Ea": 0.02}, relative=True),
    stop_at_tolerance=True,           # stop once SE(Ea) <= 2 % of Ea
)
```

`target_precision(fim, target, experiment, theta)` reports the same check for any
FIM, for example the FIM of data you already have.

## How good is a design? Certificates

The design searches are local and multi-start, so on their own they cannot show
how far a design is from the best possible. `certify_design` can. It solves the
*approximate* design problem: weights $w_i$ on candidate conditions $x_i$, with
information $M(w) = P + n\sum_i w_i I(x_i)$. That problem is convex for the
`"trace"` and `"determinant"` criteria. The *sensitivity* of a candidate is
$d(x) = -\langle \nabla\varphi(M), n\,I(x)\rangle$, for example
$n\operatorname{tr}(I(x) M^{-1} L M^{-1})$ for the trace. Convexity gives, for
every other design,

$$\varphi^\star \;\ge\; \varphi(w) - \Big(\max_x d(x) - \sum_i w_i\, d(x_i)\Big),$$

so the gap in the general equivalence theorem {cite:p}`KieferWolfowitz1960` is a
certified bound on the optimum. Every exact $n$-run design on the candidates is
also an approximate one, so this turns into a lower bound on any design's
efficiency:

```python
from discopt.doe import certify_design, approximate_targeted_design

cert = certify_design(experiment, theta, designs, target, design_bounds=bounds,
                      prior_fim=prior)
print(cert.summary())   # "... certified efficiency >= 97.3%"

approx = approximate_targeted_design(experiment, theta, target, design_bounds=bounds,
                                     n_runs=4, prior_fim=prior)
print(approx.summary())    # the optimal support and weights
approx.sensitivity         # d(x) on the candidates: where the information is
approx.rounded(4)          # an exact 4-run design (efficient rounding)
```

The weights are found by alternating multiplicative updates
{cite:p}`silvey1978-algorithm,yu2010-monotonic` with Frank-Wolfe vertex-direction
and away steps, in the spirit of the cocktail algorithm {cite:p}`yu2011-cocktail`.
Exact designs come from the weights by efficient rounding
{cite:p}`pukelsheim1992-rounding`. The bound is relative to designs on the
candidate grid. The default grid is fine in one or two inputs and Sobol in
more, and the runs being certified are always added to it.

## When the parameters are uncertain: robust targeted design

Targeted designs for nonlinear models are *local*: $K$ and $M$ are evaluated at
the current estimate. `robust_optimal_experiment(..., target=t)` averages the
targeted criterion over parameter samples (`robust="expected"`, pseudo-Bayesian)
or protects the worst sample's efficiency (`robust="maximin"`). It re-resolves
$K$ at every sample, because the sensitivity of an output or function target
moves with the parameters. With a target, `criterion="A"` means the weighted
target variance, `"D"` means $D_s$ on the targets, and `"E"` means the worst
weighted target direction. The result reports each target's standard error at
every sample (`target_std_errors`).

## Without JAX: classical designs and the browser app

The jax-free design path (`discopt.doe.linear_design`, which the browser app
uses) takes targets too. Pass `target=` to `linear_batch_design` or
`batch_design_from_basis`, built with
`basis_target(spec, parameter_names, input_names, basis)` from a target spec.
On the command line this is how
`discopt doe extend --target b12` augments a central composite or Box-Behnken
design for one interaction coefficient, and how `new ... --use-linear-design`
handles targets.

## Edge cases worth knowing

- **Replication is sometimes the only way.** On a singular FIM a target can be
  estimable yet improvable only by repeating a run already made: the criterion
  then jumps at exactly that point, and no local search finds it. The batch
  searches offer their own earlier picks as candidates, and `discopt doe
  extend` offers the completed runs. In Python, pass the runs behind a singular
  `prior_fim` as `initial_designs`.
- **From scratch, the best way to know a prediction is often to measure it.**
  With no data, a targeted design for the response at one operating point is to
  replicate that point. The model pays off once there are data: then it can
  combine runs elsewhere into a better prediction than replication gives.
- **First-order precision.** $\Sigma_\psi = K^\top M^{-1}K$ is a linearization at
  the current estimate. With few data and a strongly nonlinear target, check the
  result with a profile likelihood.

## From the command line

`discopt doe new` and `discopt doe extend` take the same targets as flags, for the
parametric templates, `--module` experiments and the `symbolic` template:

| Flag | Target |
|---|---|
| `--target PARAM[=WEIGHT]` | a parameter, optionally weighted (repeatable) |
| `--target-expr NAME=FORMULA` | a function of the parameters, e.g. `ratio=k1/k2`, `half_life=log(2)/k` |
| `--target-response RESP@INPUT=V[,...]` | a prediction at an operating point; repeat it for more points |
| `--tolerance NAME=TAU` | the standard error wanted on a target; if one target has a tolerance, all must |
| `--relative-precision` | tolerances become fractions of each target's value |

`--criterion` then applies to the targets (`trace` is usually what you want). For
example, to pin down a rate constant to 5 % with as few runs as possible:

```bash
discopt doe new symbolic --expr "A*exp(-k*t)" --param A=2 --param k=0.5 \
    --input t:0.05:10 --error 0.05 --n 3 --criterion trace \
    --target k --tolerance k=0.05 --relative-precision -o decay.xlsx
```

The target is stored in the workbook. After you record the responses and run
`discopt doe fit`, `discopt doe status` reports each target's standard error and
its ratio to the tolerance. It uses the fitted FIM at the *declared* measurement
error, the same scale the design predictions use. When every target is within its
tolerance, `status` says so.

`discopt doe extend` reuses the stored target. `--target ...` on `extend` replaces
it, and `--clear-target` goes back to designing for all parameters. Both `new` and
`extend` report the precision the new runs are predicted to give, counting only
the information the runs carry. That is why one run of a two-parameter model
reports "not estimable" rather than a small, misleading number.

A target can make fewer runs than parameters useful. For example, `new symbolic`
accepts `--n` below the parameter count when a target is named. With a target,
`new` searches the batch jointly. A greedy search picks the first runs before
the target is identifiable, so it can leave the batch in a poor arrangement:
two runs that should sit symmetrically to measure a slope are one example. Fixed
and model-free designs (factorial, classical, Latin, `optimize`) do not accept
targets.
