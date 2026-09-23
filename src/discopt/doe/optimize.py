"""Active-learning loop for response optimization.

When to use this module
-----------------------

Use :func:`optimize_round` when the question is **"what input gives the
best response?"** -- minimize cost, maximize yield, find the operating
point that lands on a target -- and experiments are expensive enough
that you cannot afford a full factorial or response-surface design.

This is *active learning*: every new run is chosen to be the most
informative for the optimization objective, not to fill out a fixed
matrix.

Contrast with the rest of :mod:`discopt.doe`:

* :func:`~discopt.doe.optimal_experiment` -- design for **parameter
  estimation** (maximize Fisher information). Different objective:
  precise model parameters, not best response.
* :func:`~discopt.doe.factorial_2level_design` -- design for
  **factor screening** ("does this factor matter at all?"). Use that
  first if you don't yet know which factors to optimize over.
* :func:`~discopt.doe.batch_optimal_experiment` -- batch FIM design.
  Same parameter-estimation objective, batch interface.

A good full sequence is **screen → optimize**: run a 2-level factorial
to drop irrelevant factors, then call :func:`optimize_round` on the
surviving ones.

How one round works
-------------------

A single call to :func:`optimize_round`:

1. opens the workbook and reads every row that has a non-empty
   response value (the *completed* runs);
2. (optionally) standardizes inputs to zero mean + unit variance;
3. fits the chosen :class:`~discopt.doe.surrogate.Surrogate` to the
   completed runs;
4. draws a pool of ``n_candidates`` Sobol points inside the input
   bounds;
5. scores each candidate with the acquisition function;
6. greedily picks the top-``batch_size`` -- after each pick, the
   chosen point is added back into the surrogate with its predicted
   mean as a "fantasy" response, so the next pick sees lower
   uncertainty there and the batch diversifies;
7. appends the batch to the workbook as new pending runs and returns
   an :class:`OptimizationRoundResult`.

You then run the experiments, fill in the response column of the
workbook, and call :func:`optimize_round` again. There is no fixed
budget -- stop when the incumbent stops improving, or when you've
spent the runs you can afford.

Surrogate choice (convenience vs. control)
------------------------------------------

The ``surrogate`` argument accepts three escalating levels of
specificity. They all funnel through
:func:`~discopt.doe.surrogate.coerce_surrogate`:

* **String preset** -- ``"gp"`` (default-ish; Matern(5/2) + white
  noise) or ``"response-surface"`` (degree-2 polynomial + Bayesian
  ridge). Zero knobs. Good for first contact.
* **Any scikit-learn-compatible estimator** -- wrapped automatically
  in :class:`~discopt.doe.surrogate._SklearnUQAdapter`, which probes
  for ``predict(X, return_std=True)`` (sklearn GP / BayesianRidge /
  ARDRegression), then ``predict(X, return_interval=True)``
  (`pycse.sklearn.lpr.LinearLPR`), then falls back to a residual
  bootstrap. This is the right path when you want a specific GP
  kernel or LPR-style local prediction without writing any adapter
  code.
* **Custom object** implementing the
  :class:`~discopt.doe.surrogate.Surrogate` protocol (``fit`` +
  ``predict`` returning ``(mean, std)``). Opt in by setting the class
  attribute ``_is_discopt_surrogate = True`` so the router does not
  re-wrap it. Full escape hatch for bespoke Bayesian models.

The acquisition function only sees ``(mean, std)`` arrays. It never
looks inside the surrogate. That is the abstraction boundary --
swap kernels, swap libraries, none of the optimization code cares.

Acquisition choice
------------------

* ``"expected_improvement"`` (alias ``"ei"``) -- the textbook
  Bayesian-optimization choice. Balances exploitation and exploration
  through the surrogate's uncertainty. Default.
* ``"ucb"`` / ``"lcb"`` / ``"confidence_bound"`` -- ``μ ± κ σ`` with
  the sign tied to direction. Tune the explore/exploit balance via
  ``acquisition_kwargs={"kappa": ...}``. Larger κ explores more.
* ``"steepest_ascent"`` -- ignores σ entirely; picks the point with
  the best predicted mean in the optimization direction. Use with a
  response-surface surrogate to reproduce classical Box-Wilson
  behavior. Not recommended with a GP -- without uncertainty you give
  up the main reason to fit a GP in the first place.
* ``"max_variance"`` (alias ``"uncertainty"``) -- pure exploration for
  active learning: the run where the surrogate is least certain. Use it to
  make a surrogate accurate everywhere, not to find an optimum.

Constraints and failed runs
---------------------------

**Known** constraints (a feasible region you can write down, a mixture
that must sum to 1) are handled by the candidate pool: pass
``candidates=`` (an array or run dicts) or a ``candidate_sampler``
callable, and only those points are scored. **Unknown** constraints --
runs that simply fail, with no response -- are learned: mark them with
``infeasible_runs=[run_id, ...]`` or a workbook ``feasibility_column``,
and a Gaussian-process classifier estimates each candidate's probability
of feasibility, which weights the acquisition (expected improvement
times P(feasible); Schonlau, Welch & Jones 1998; Gramacy et al. 2016).

Categorical and mixed-input factors
-----------------------------------

The driver currently expects every input column in the workbook to be
numeric (it builds an X matrix). If you have categorical factors,
encode them upstream as 0/1 indicator columns or a small integer
code, and add the encoded columns to the workbook's ``input_specs``.
Future versions may auto-encode at the adapter boundary.

Quick start
-----------

>>> from discopt.doe import optimize_round, OptimizationCriterion
>>> result = optimize_round(
...     workbook="opt.xlsx",
...     criterion=OptimizationCriterion.MAXIMIZE,
...     surrogate="gp",                 # or a sklearn estimator, or a Surrogate
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.next_designs)
>>> print(f"incumbent so far: y={result.incumbent_y:.3f} at {result.incumbent_x}")

References
----------

* Jones, Schonlau, Welch (1998). *Efficient Global Optimization of
  Expensive Black-Box Functions.* JoGO 13:455-492. -- EI + Kriging.
* Box, Wilson (1951). *On the Experimental Attainment of Optimum
  Conditions.* JRSS-B 13:1-45. -- steepest-ascent / response-surface
  methodology.
* Snoek, Larochelle, Adams (2012). *Practical Bayesian Optimization
  of Machine Learning Algorithms.* NeurIPS. -- modern BO recipes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from discopt.doe.acquisition import call_acquisition, resolve_acquisition
from discopt.doe.surrogate import Surrogate, coerce_surrogate
from discopt.doe.workbook import Workbook


class OptimizationCriterion(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    @property
    def direction(self) -> int:
        return 1 if self is OptimizationCriterion.MAXIMIZE else -1


@dataclass
class OptimizationRoundResult:
    """Outcome of one active-learning round."""

    next_designs: list[dict[str, float]]
    new_run_ids: list[int]
    incumbent_x: dict[str, float] | None
    incumbent_y: float | None
    acquisition_scores: list[float]
    surrogate_mode: str | None
    n_completed: int
    workbook_path: str
    log: list[str] = field(default_factory=list)
    feasibility: list[float] | None = None
    """Estimated probability of feasibility of each recommended run, when
    failed runs were marked; otherwise None."""
    pareto_front: list[dict[str, Any]] | None = None
    """With several objectives, the non-dominated completed runs: the set where
    no other run is at least as good on every objective and better on one.
    Choosing a point on it is a judgement about value, not about data."""
    scalarization_weights: list[list[float]] | None = None
    """The objective weights drawn for each recommended run (ParEGO)."""


def _candidate_rows(
    *,
    candidates: Any,
    candidate_sampler: Any,
    n_candidates: int,
    names: Sequence[str],
    bounds_arr: np.ndarray,
    encoder: "_InputEncoder",
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """The candidate pool as run dicts, in natural units and with level labels.

    With categorical factors the pool is the product of their levels with the
    continuous box: the same continuous points are offered at every level, so a
    level is never passed over because it happened to be sampled at worse
    conditions than another.
    """
    if candidates is not None:
        rows = _rows_from_candidates(candidates, names)
        missing = [n for n in names if any(n not in r for r in rows)]
        if missing:
            raise ValueError(f"candidate rows are missing input(s) {missing}")
        return rows

    if not encoder.has_categorical:
        if callable(candidate_sampler):
            matrix = _as_matrix(candidate_sampler(n_candidates, rng), names)
        else:
            matrix = _sample_candidates(bounds_arr, n_candidates, candidate_sampler, rng)
        return [{n: float(row[j]) for j, n in enumerate(names)} for row in matrix]

    combos: list[dict[str, Any]] = [{}]
    for name, levels in encoder.categorical.items():
        combos = [{**combo, name: level} for combo in combos for level in levels]
    per_combo = max(1, n_candidates // max(1, len(combos)))

    continuous = encoder.continuous
    if continuous:
        idx = [names.index(n) for n in continuous]
        box = bounds_arr[idx]
        if callable(candidate_sampler):
            matrix = _as_matrix(candidate_sampler(per_combo, rng), names)[:, idx]
        else:
            matrix = _sample_candidates(box, per_combo, candidate_sampler, rng)
    else:
        matrix = np.zeros((1, 0))

    rows: list[dict[str, Any]] = []
    for combo in combos:
        for row in matrix:
            rows.append({**{n: float(row[j]) for j, n in enumerate(continuous)}, **combo})
    return rows


def _rows_from_candidates(candidates: Any, names: Sequence[str]) -> list[dict[str, Any]]:
    """Explicit candidates as run dicts, keeping any non-numeric level labels."""
    if isinstance(candidates, np.ndarray):
        return [{n: float(v) for n, v in zip(names, row)} for row in np.atleast_2d(candidates)]
    rows = []
    for item in candidates:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            rows.append({n: v for n, v in zip(names, item)})
    return rows


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    """``True`` where a row is not dominated, with every column "larger is better".

    A row is dominated when another is at least as good everywhere and strictly
    better somewhere.
    """
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        better_or_equal = np.all(values >= values[i], axis=1)
        strictly_better = np.any(values > values[i], axis=1)
        dominated_by = better_or_equal & strictly_better
        if np.any(dominated_by):
            keep[i] = False
    return keep


def _chebyshev_scalarization(
    values: np.ndarray, weights: np.ndarray, rho: float = 0.05
) -> np.ndarray:
    """ParEGO's augmented Chebyshev scalarization, "larger is better".

    Each objective is first put on a 0-1 scale from what has been observed, so
    a yield in percent and an impurity in ppm can be weighed against each other
    at all. The Chebyshev form ``min_j w_j y_j`` is what lets a scalarized
    search reach the *concave* parts of a Pareto front, which a weighted sum
    cannot; the small augmentation term keeps it from stopping on a weakly
    dominated point (Knowles 2006).
    """
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    # ParEGO is written for minimization, and the direction matters: the
    # weighted *cost* that is largest is the one the max picks up, so a bigger
    # weight means that objective is emphasized. Scalarizing the rewards
    # instead inverts it -- a big weight on yield would then buy low yield.
    cost = 1.0 - (values - lo) / span
    weighted = cost * weights
    return -(np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1))


def _objective_matrix(
    rows: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, str],
) -> np.ndarray:
    """``(n, m)`` objective values, signed so that larger is always better."""
    columns = []
    for name, sense in objectives.items():
        sign = 1.0 if str(sense).lower().startswith("max") else -1.0
        try:
            columns.append(sign * np.array([float(r[name]) for r in rows], dtype=float))
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"objective column {name!r} is missing or not numeric in the completed runs"
            ) from exc
    return np.column_stack(columns)


class _InputEncoder:
    """Turns run dicts into the numeric matrix a surrogate sees, and back.

    A GP measures distance between runs, so a categorical factor has to be
    encoded before it can be modelled at all. Coding levels as 0, 1, 2 invents
    an ordering and a spacing that the chemistry does not have -- it asserts
    that level 1 lies between the other two and is equally far from each. The
    honest encoding is **one-hot**: one indicator column per level, so every
    pair of levels is the same distance apart and none is between any others.

    Continuous inputs are standardized (mean 0, sd 1) when asked; the indicator
    columns are left alone, since their scale already is the distance between
    two levels.
    """

    def __init__(
        self,
        names: Sequence[str],
        categorical: Mapping[str, Sequence[Any]] | None,
        standardize: bool,
    ) -> None:
        self.names = list(names)
        self.categorical = {str(k): list(v) for k, v in (categorical or {}).items()}
        unknown = [k for k in self.categorical if k not in self.names]
        if unknown:
            raise ValueError(f"categorical factor(s) {unknown} are not inputs {self.names}")
        for name, levels in self.categorical.items():
            if len(levels) < 2:
                raise ValueError(f"categorical factor {name!r} needs at least two levels")
            if len(set(map(str, levels))) != len(levels):
                raise ValueError(f"categorical factor {name!r} has repeated levels")
        self.continuous = [n for n in self.names if n not in self.categorical]
        self.standardize = bool(standardize)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    @property
    def has_categorical(self) -> bool:
        return bool(self.categorical)

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "_InputEncoder":
        """Learn the standardization from the completed runs."""
        if self.continuous and self.standardize:
            values = np.array([[float(r[n]) for n in self.continuous] for r in rows], dtype=float)
            mu = values.mean(axis=0)
            sd = values.std(axis=0, ddof=0)
            self._mu, self._sd = mu, np.where(sd > 0.0, sd, 1.0)
        else:
            self._mu = np.zeros(len(self.continuous))
            self._sd = np.ones(len(self.continuous))
        return self

    def encode(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """``(n, d)`` features: standardized continuous columns, then indicators."""
        if self._mu is None:
            raise RuntimeError("encoder used before fit")
        blocks = []
        if self.continuous:
            values = np.array([[float(r[n]) for n in self.continuous] for r in rows], dtype=float)
            blocks.append((values - self._mu) / self._sd)
        for name, levels in self.categorical.items():
            keys = [str(level) for level in levels]
            indicator = np.zeros((len(rows), len(levels)), dtype=float)
            for i, row in enumerate(rows):
                value = str(row[name])
                if value not in keys:
                    raise ValueError(
                        f"run has {name}={row[name]!r}, which is not one of its levels {levels}"
                    )
                indicator[i, keys.index(value)] = 1.0
            blocks.append(indicator)
        return np.hstack(blocks) if blocks else np.zeros((len(rows), 0))


def optimize_round(
    workbook: str | Path | Workbook,
    *,
    criterion: OptimizationCriterion | str = OptimizationCriterion.MAXIMIZE,
    surrogate: object = "gp",
    acquisition: str | Callable = "expected_improvement",
    batch_size: int = 1,
    n_candidates: int = 2048,
    candidate_sampler: str | Callable[..., Any] = "sobol",
    candidates: np.ndarray | Sequence[Sequence[float]] | Sequence[dict[str, float]] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    input_names: Sequence[str] | None = None,
    standardize_inputs: bool = True,
    seed: int | None = None,
    acquisition_kwargs: dict[str, Any] | None = None,
    infeasible_runs: Sequence[int] | None = None,
    feasibility_column: str | None = None,
    categorical: Mapping[str, Sequence[Any]] | None = None,
    objectives: Sequence[str] | Mapping[str, str] | None = None,
) -> OptimizationRoundResult:
    """Propose the next batch of experiments using an active-learning surrogate.

    Parameters
    ----------
    workbook : path or Workbook
        Workbook containing the completed runs. The next batch is appended.
    criterion : OptimizationCriterion or str
        ``"maximize"`` or ``"minimize"``.
    surrogate : str, sklearn estimator, or Surrogate
        Passed through :func:`~discopt.doe.surrogate.coerce_surrogate`.
    acquisition : str or callable
        Looked up via :func:`~discopt.doe.acquisition.resolve_acquisition`.
    batch_size : int, default 1
        Number of new experiments to recommend.
    n_candidates : int, default 2048
        Size of the candidate pool sampled inside the bounding box.
    candidate_sampler : ``"sobol"``, ``"uniform"``, or callable
        Sobol gives better space-filling for the same budget. A callable
        ``f(n, rng)`` returns ``n`` candidates (an array ordered like
        ``input_names``, or run dicts) in natural units -- for a
        constrained region, e.g. mixtures drawn with
        :func:`~discopt.doe.sample_simplex`.
    candidates : array or sequence of run dicts, optional
        An explicit candidate pool in natural units; replaces sampling
        (``n_candidates`` and ``candidate_sampler`` are then ignored). Use it
        for known constraints, discrete settings, or a fixed grid.
    bounds : sequence of (lo, hi), optional
        Per-input box. Defaults to the input_specs stored in the workbook.
    input_names : sequence of str, optional
        Names of the input columns. Defaults to the workbook's input_specs
        in the original order.
    standardize_inputs : bool, default True
        If True, the surrogate sees inputs standardized to zero mean +
        unit variance (computed on the completed runs). Acquisition is
        evaluated in the standardized space; recommended points are
        unstandardized before being written back.
    seed : int, optional
        Reproducible candidate sampling.
    acquisition_kwargs : dict, optional
        Extra kwargs forwarded to the acquisition function (e.g.
        ``{"xi": 0.01}`` for EI, ``{"kappa": 2.5}`` for UCB).
    infeasible_runs : sequence of int, optional
        run_ids of runs that failed (no usable response). With
        ``feasibility_column`` they train a classifier whose probability of
        feasibility weights the acquisition; they never enter the surrogate.
    objectives : sequence of column names, or mapping name -> sense, optional
        Optimize several responses at once (ParEGO; Knowles 2006). Each round
        draws a random weight vector, scalarizes the objectives with an
        augmented Chebyshev function and runs the ordinary single-objective
        acquisition on it -- one weight vector per requested run, which is what
        makes a batch spread along the trade-off rather than crowd one end of
        it. A sequence uses ``criterion`` for every objective; a mapping gives
        each its own ``"maximize"``/``"minimize"``.

        The result then carries ``pareto_front`` (the non-dominated completed
        runs) instead of a single incumbent, because with several objectives
        there is no single best run. Which point on the front to take is a
        judgement about value, not about data.
    categorical : mapping name -> levels, optional
        Factors that take one of a fixed set of values (a catalyst, a solvent).
        They are one-hot encoded for the surrogate, and the candidate pool is
        the product of the levels with the continuous box, so every level is
        proposed on its merits. Coding them as 0, 1, 2 instead would assert an
        ordering and a spacing the chemistry does not have. The levels are
        written back to the workbook as given, not as numbers.
    feasibility_column : str, optional
        A workbook column marking each finished run as feasible (1, True,
        "yes") or not (0, False, "no", "fail", "infeasible").
    """
    crit = OptimizationCriterion(criterion) if isinstance(criterion, str) else criterion
    direction = crit.direction
    acq_fn = resolve_acquisition(acquisition)
    acq_kwargs = dict(acquisition_kwargs or {})

    if objectives is None:
        objective_senses: dict[str, str] = {}
    elif isinstance(objectives, Mapping):
        objective_senses = {str(k): str(v) for k, v in objectives.items()}
    else:
        objective_senses = {str(name): crit.value for name in objectives}
    if len(objective_senses) == 1:
        raise ValueError("objectives needs at least two columns; for one response use criterion=")

    wb = workbook if isinstance(workbook, Workbook) else Workbook.open(Path(workbook))

    specs = wb.input_specs()
    names = list(input_names) if input_names is not None else [s.name for s in specs]
    categorical_names = set(categorical or {})
    if bounds is None:
        # A categorical factor has levels, not a range, so it needs no bounds --
        # and it may be a plain column rather than a declared input. Its row is
        # a placeholder the encoder never reads.
        spec_by_name = {s.name: s for s in specs}
        rows = []
        for name in names:
            if name in categorical_names:
                rows.append((0.0, 1.0))
            elif name in spec_by_name:
                rows.append((spec_by_name[name].lb, spec_by_name[name].ub))
            else:
                raise ValueError(
                    f"no bounds for input {name!r}: it is not in the workbook's input specs "
                    f"{sorted(spec_by_name)}, so pass bounds= or declare it categorical"
                )
        bounds_arr = np.array(rows, dtype=float)
    else:
        bounds_arr = np.asarray(list(bounds), dtype=float)
    if bounds_arr.shape != (len(names), 2):
        raise ValueError(f"bounds shape {bounds_arr.shape} does not match {len(names)} input(s)")

    response = wb.response_name()
    failed_ids = {int(r) for r in (infeasible_runs or [])}
    all_rows = wb.all_runs()
    if feasibility_column is not None:
        if all_rows and feasibility_column not in all_rows[0]:
            raise ValueError(f"feasibility column {feasibility_column!r} is not in the workbook")
        for r in all_rows:
            if _is_infeasible_mark(r.get(feasibility_column)):
                failed_ids.add(int(r["run_id"]))
    failed = [r for r in all_rows if int(r["run_id"]) in failed_ids]
    completed = [r for r in wb.completed_runs() if int(r["run_id"]) not in failed_ids]
    if not completed:
        raise ValueError(
            "no completed runs in workbook -- fill in at least one response "
            "column before calling optimize_round"
        )

    encoder = _InputEncoder(names, categorical, standardize_inputs).fit(completed)
    X = encoder.encode(completed)
    if objective_senses:
        Y = _objective_matrix(completed, objective_senses)
        y = _chebyshev_scalarization(Y, np.full(Y.shape[1], 1.0 / Y.shape[1]))
    else:
        Y = None
        y = np.array([float(r[response]) for r in completed], dtype=float)

    s = coerce_surrogate(surrogate, random_state=seed)
    s.fit(X, y)

    rng = np.random.default_rng(seed)
    candidate_rows = _candidate_rows(
        candidates=candidates,
        candidate_sampler=candidate_sampler,
        n_candidates=int(n_candidates),
        names=names,
        bounds_arr=bounds_arr,
        encoder=encoder,
        rng=rng,
    )
    if not candidate_rows:
        raise ValueError("the candidate pool is empty")
    candidates_std = encoder.encode(candidate_rows)

    # Unknown constraints: P(feasible) from a classifier on feasible vs failed runs.
    p_feasible: np.ndarray | None = None
    if failed:
        X_fail = encoder.encode(failed)
        p_feasible = _feasibility_probability(X, X_fail, candidates_std, seed)

    def _natural(row: Mapping[str, Any]) -> dict[str, Any]:
        return {n: (row[n] if n in encoder.categorical else float(row[n])) for n in names}

    pareto_front: list[dict[str, Any]] | None = None
    weights_used: list[list[float]] | None = None
    if Y is not None:
        # With several objectives there is no single best run, so the incumbent
        # is a *set*: the runs nothing else beats on every objective at once.
        mask = _pareto_mask(Y)
        pareto_front = [
            {**_natural(completed[i]), **{n: float(completed[i][n]) for n in objective_senses}}
            for i in np.flatnonzero(mask)
        ]
        weights_used = []
        incumbent_x = None
        incumbent_y = None
    else:
        incumbent_idx = int(np.argmax(direction * y))
        incumbent_y = float(y[incumbent_idx])
        incumbent_x = _natural(completed[incumbent_idx])

    chosen_idx: list[int] = []
    chosen_scores: list[float] = []
    X_fantasy = X.copy()
    y_fantasy = y.copy()
    incumbent_for_acq = incumbent_y if incumbent_y is not None else float(np.max(y))

    for _ in range(int(batch_size)):
        if Y is not None:
            # ParEGO: a fresh weight vector per requested run. The spread of a
            # batch comes from the spread of the weights, so the fantasy
            # refitting the single-objective path uses is neither needed nor
            # meaningful here -- a mean imputed under one scalarization says
            # nothing about the next one.
            w = rng.dirichlet(np.ones(Y.shape[1]))
            weights_used.append([float(v) for v in w])
            y_round = _chebyshev_scalarization(Y, w)
            s = coerce_surrogate(surrogate, random_state=seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="the fitted GP noise sits on its floor")
                s.fit(X, y_round)
            incumbent_for_acq = float(np.max(y_round))

        scores = call_acquisition(
            acq_fn,
            s,
            candidates_std,
            direction=1.0 if Y is not None else direction,
            y_best=incumbent_for_acq,
            acq_kwargs=acq_kwargs,
        )
        scores = np.asarray(scores, dtype=float).ravel()
        if p_feasible is not None:
            if np.all(scores >= 0.0):  # EI, max_variance: weight by P(feasible)
                scores = scores * p_feasible
            else:  # signed scores (UCB, steepest ascent): drop likely failures
                scores = np.where(p_feasible >= 0.5, scores, -np.inf)
        if chosen_idx:
            scores[chosen_idx] = -np.inf
        pick = int(np.argmax(scores))
        chosen_idx.append(pick)
        chosen_scores.append(float(scores[pick]))

        if Y is not None:
            continue  # the next weight vector, not a fantasy, diversifies here

        # Mean-imputation: pretend the chosen point's response is the
        # surrogate's mean. Re-fit so the next pick sees lower
        # uncertainty there and diversifies.
        mu_pick, _ = s.predict(candidates_std[pick : pick + 1])
        X_fantasy = np.vstack([X_fantasy, candidates_std[pick : pick + 1]])
        y_fantasy = np.concatenate([y_fantasy, mu_pick])
        if direction * float(mu_pick[0]) > direction * incumbent_for_acq:
            incumbent_for_acq = float(mu_pick[0])
        s = coerce_surrogate(surrogate, random_state=seed)
        # The fantasy points are noiseless by construction, so a noise estimate
        # on its floor says nothing here; the real fit above already warned.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="the fitted GP noise sits on its floor")
            s.fit(X_fantasy, y_fantasy)

    next_designs = [dict(candidate_rows[i]) for i in chosen_idx]
    batch_idx = wb.next_batch_index()
    new_run_ids = wb.append_runs(batch_idx, next_designs)
    wb.log(
        "optimize",
        {
            "criterion": crit.value,
            "acquisition": acquisition if isinstance(acquisition, str) else acq_fn.__name__,
            "batch_size": int(batch_size),
            "n_completed": len(completed),
            "surrogate_mode": getattr(s, "mode", None),
        },
    )
    wb.save()

    return OptimizationRoundResult(
        next_designs=next_designs,
        new_run_ids=new_run_ids,
        incumbent_x=incumbent_x,
        incumbent_y=incumbent_y,
        acquisition_scores=chosen_scores,
        surrogate_mode=getattr(s, "mode", None),
        n_completed=len(completed),
        workbook_path=str(wb.path),
        feasibility=None if p_feasible is None else [float(p_feasible[i]) for i in chosen_idx],
        pareto_front=pareto_front,
        scalarization_weights=weights_used,
    )


# ──────────────────────────────────────────────────────────────────
# Candidate sampling
# ──────────────────────────────────────────────────────────────────


def _sample_candidates(
    bounds: np.ndarray, n: int, sampler: str, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` points uniformly in the box defined by ``bounds``."""
    d = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    if sampler == "uniform":
        u = rng.uniform(size=(n, d))
    elif sampler == "sobol":
        try:
            from scipy.stats import qmc

            m = int(math.ceil(math.log2(max(n, 2))))
            engine = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
            u = engine.random_base2(m=m)[:n]
        except ImportError:
            u = rng.uniform(size=(n, d))
    else:
        raise ValueError(f"unknown candidate_sampler {sampler!r}")
    return np.asarray(lo + (hi - lo) * u, dtype=float)


def _as_matrix(pool: Any, names: Sequence[str]) -> np.ndarray:
    """Candidates as an ``(n, d)`` array ordered like ``names``."""
    items = list(pool) if not isinstance(pool, np.ndarray) else pool
    if len(items) and isinstance(items[0], dict):
        return np.array([[float(r[n]) for n in names] for r in items], dtype=float)
    arr = np.atleast_2d(np.asarray(items, dtype=float))
    if arr.size and arr.shape[1] != len(names):
        raise ValueError(f"candidates have {arr.shape[1]} columns; expected {len(names)}")
    return arr


_INFEASIBLE_MARKS = {"0", "false", "no", "n", "fail", "failed", "infeasible"}


def _is_infeasible_mark(value: Any) -> bool:
    if value is None or value == "":
        return False  # not yet marked
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    return str(value).strip().lower() in _INFEASIBLE_MARKS


def _feasibility_probability(
    X_ok: np.ndarray, X_fail: np.ndarray, candidates: np.ndarray, seed: int | None
) -> np.ndarray:
    """P(feasible) at the candidates from a GP classifier on feasible vs failed runs."""
    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError as e:  # pragma: no cover - the GP preset already needs sklearn
        raise ImportError(
            'learning unknown constraints needs scikit-learn: pip install "discopt-doe[ml]"'
        ) from e
    X_all = np.vstack([X_ok, X_fail])
    labels = np.concatenate([np.ones(len(X_ok)), np.zeros(len(X_fail))])
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=1.0, length_scale_bounds=(0.05, 1e2), nu=2.5
    )
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0 if seed is None else int(seed))
    clf.fit(X_all, labels)
    return np.asarray(clf.predict_proba(candidates)[:, 1], dtype=float)


_ = Surrogate  # ensure protocol import is exported for downstream users


__all__ = [
    "OptimizationCriterion",
    "OptimizationRoundResult",
    "optimize_round",
]
