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
