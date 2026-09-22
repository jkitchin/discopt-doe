"""Sequential model-based design of experiments.

Alternates between parameter estimation and optimal design in a loop:
1. Estimate parameters from all collected data
2. Compute FIM and optimize next experiment design
3. (Optionally) run experiment and collect new data
4. Repeat
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np

from discopt.doe.design import (
    BatchDesignResult,
    BatchStrategy,
    DesignCriterion,
    DesignResult,
    batch_optimal_experiment,
    optimal_experiment,
)
from discopt.doe._estimation import estimate_parameters
from discopt.estimate import (
    EstimationResult,
    Experiment,
)


@dataclass
class DoERound:
    """Record of one round in the sequential DoE loop.

    Attributes
    ----------
    round : int
        Zero-indexed round number.
    estimation : EstimationResult
        Parameter estimation result for this round.
    design : DesignResult or BatchDesignResult
        Recommended experiment(s) for the next round. A plain
        ``DesignResult`` for single-experiment rounds; a
        ``BatchDesignResult`` when ``experiments_per_round > 1``.
    data_collected : dict[str, float or numpy.ndarray] or None
        New data collected in this round (if experiment runner provided).
    runs : list of dict or None
        With ``initial_runs``: every run (conditions and responses) that this
        round's estimate was fitted to.
    """

    round: int
    estimation: EstimationResult
    design: DesignResult | BatchDesignResult
    data_collected: dict[str, Union[float, np.ndarray]] | None = None
    runs: list[dict[str, Any]] | None = None


def sequential_doe(
    experiment: Experiment,
    initial_data: dict[str, Union[float, np.ndarray]] | None,
    initial_guess: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    initial_runs: Sequence[Mapping[str, Any]] | None = None,
    n_rounds: int = 5,
    criterion: str = DesignCriterion.D_OPTIMAL,
    experiments_per_round: int = 1,
    batch_strategy: str = BatchStrategy.GREEDY,
    run_experiment: Callable[[dict[str, float]], dict[str, float]] | None = None,
    callback: Callable[[DoERound], None] | None = None,
) -> list[DoERound]:
    """Run the full sequential MBDoE loop.

    There are two ways to hand over the data. **Runs with conditions**
    (``initial_runs``, recommended whenever the model has design inputs): each
    run is a row holding its design-input values *and* its measured responses,
    every round refits one campaign (:func:`~discopt.doe.campaign_experiment`)
    over all runs so far, and ``run_experiment`` returns the responses of the
    run it was given, under the model's own response names (e.g. ``{"y": 3.2}``
    or ``{"A@2": 0.41, "A@5": 0.12}``). Nothing needs a fresh response key. The
    older **response-keyed** form (``initial_data``) is kept for models without
    design inputs; see the caveat under ``run_experiment``.

    Parameters
    ----------
    experiment : Experiment, ODEExperiment, or SymbolicModel
        The model of one run. A :class:`~discopt.doe.SymbolicModel` needs
        ``initial_runs``.
    initial_data : dict or None
        Response-keyed data for the first estimation. Pass ``None`` when
        giving ``initial_runs``.
    initial_guess : dict[str, float]
        Starting parameter estimates.
    design_bounds : dict[str, tuple[float, float]]
        Bounds on design variables.
    initial_runs : sequence of dict, optional
        Runs with conditions: each row gives every design input and the
        measured responses (missing or NaN responses are skipped).
    n_rounds : int, default 5
        Number of DoE rounds.
    criterion : str, default DesignCriterion.D_OPTIMAL
        Design criterion for optimization.
    experiments_per_round : int, default 1
        Number of experiments to design per round. When ``> 1``, the
        round calls :func:`batch_optimal_experiment` with ``batch_strategy``
        and ``run_experiment`` is invoked once per batch member.
    batch_strategy : str, default ``BatchStrategy.GREEDY``
        Batch strategy to use when ``experiments_per_round > 1``.
        Ignored otherwise.
    run_experiment : callable, optional
        Function ``f(design_dict) -> data_dict`` that runs an experiment
        at the given design conditions and returns observed data.
        If None, the loop returns after the first recommendation.

        .. important::
           The returned ``data_dict`` is merged into the cumulative data by
           response key and handed to :func:`~discopt.estimate.estimate_parameters`,
           which treats repeated values under one key as **replicate
           observations at the model's fixed conditions** -- the design values
           are *not* attached to the data. So if your model has design inputs
           and ``run_experiment`` returns the *same* response keys each round
           (e.g. always ``{"y": value}``), observations taken at *different*
           designed conditions are silently fitted as replicates at one
           condition, corrupting the estimates and every subsequent design.

           The correct pattern is a *stateful* ``Experiment`` whose
           ``create_model`` exposes a fresh response (and its design condition)
           per accumulated observation, with ``run_experiment`` appending a new
           observation under a new response key each round. See
           ``tests/test_discrimination_sequential.py`` for a worked example.
           When ``design_bounds`` is non-empty and a returned key already
           exists, this function warns that replicate-confusion may be
           occurring.
    callback : callable, optional
        Called with each ``DoERound`` after it completes.

    Returns
    -------
    list[DoERound]
        History of all rounds. Each round estimates parameters *before*
        collecting that round's data, so ``history[-1].estimation`` does not
        include the final round's ``data_collected``. Re-run
        :func:`~discopt.estimate.estimate_parameters` on the accumulated data
        (or start another loop) if you need the fully-updated estimate.
    """
    if experiments_per_round < 1:
        raise ValueError(f"experiments_per_round must be >= 1, got {experiments_per_round}")
    if initial_runs is not None:
        if initial_data:
            raise ValueError("pass either initial_data or initial_runs, not both")
        return _sequential_runs(
            experiment,
            list(initial_runs),
            initial_guess,
            design_bounds,
            n_rounds=n_rounds,
            criterion=criterion,
            experiments_per_round=experiments_per_round,
            batch_strategy=batch_strategy,
            run_experiment=run_experiment,
            callback=callback,
        )
    if initial_data is None:
        raise ValueError("initial_data is required unless initial_runs is given")

    history = []
    current_guess = dict(initial_guess)
    all_data = dict(initial_data)

    for round_idx in range(n_rounds):
        # Step 1: Estimate parameters from all data
        est = estimate_parameters(
            experiment,
            all_data,
            initial_guess=current_guess,
        )

        # Information already collected: est.fim is fit on *all* accumulated
        # data, so it is the cumulative prior for designing the next point.
        # (Summing est.fim across rounds would double-count earlier data,
        # since each round's est.fim already includes it.)
        prior_fim = est.fim

        # Step 2: Design next experiment(s)
        design: DesignResult | BatchDesignResult
        if experiments_per_round == 1:
            design = optimal_experiment(
                experiment,
                est.parameters,
                design_bounds,
                criterion=criterion,
                prior_fim=prior_fim,
            )
            round_designs = [design.design]
        else:
            design = batch_optimal_experiment(
                experiment,
                est.parameters,
                design_bounds,
                n_experiments=experiments_per_round,
                criterion=criterion,
                strategy=batch_strategy,
                prior_fim=prior_fim,
            )
            round_designs = list(design.designs)

        # Step 3: Record round
        round_result = DoERound(
            round=round_idx,
            estimation=est,
            design=design,
            data_collected=None,
        )

        # Step 4: Run experiment(s) if callback provided
        if run_experiment is not None:
            collected: dict[str, np.ndarray] = {}
            for d in round_designs:
                new_data = run_experiment(d)
                for key, val in new_data.items():
                    arr = np.atleast_1d(val)
                    existing = collected.get(key)
                    if existing is None:
                        collected[key] = arr
                    else:
                        collected[key] = np.concatenate([existing, arr])

            round_result.data_collected = dict(collected)

            # Merge round data into cumulative data
            for key, round_arr in collected.items():
                if key in all_data:
                    if design_bounds:
                        warnings.warn(
                            f"run_experiment returned response key {key!r} that "
                            "already has data, while design_bounds is non-empty. "
                            "estimate_parameters will fit these as replicates at "
                            "one fixed condition, ignoring the differing designed "
                            "conditions. Use a stateful Experiment that exposes a "
                            "fresh response key per observation (see the "
                            "sequential_doe docstring).",
                            stacklevel=2,
                        )
                    all_data[key] = np.concatenate([np.atleast_1d(all_data[key]), round_arr])
                else:
                    all_data[key] = round_arr

            current_guess = est.parameters

        history.append(round_result)

        if callback is not None:
            callback(round_result)

        # If no experiment runner, stop after first recommendation
        if run_experiment is None:
            break

    return history


def _sequential_runs(
    model: Any,
    runs: list[Mapping[str, Any]],
    initial_guess: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_rounds: int,
    criterion: str,
    experiments_per_round: int,
    batch_strategy: str,
    run_experiment: Callable[[dict[str, float]], dict[str, float]] | None,
    callback: Callable[[DoERound], None] | None,
) -> list[DoERound]:
    """The loop on runs-with-conditions: refit the campaign, design, run, repeat."""
    from discopt.doe.runs import campaign_experiment, symbolic_experiment
    from discopt.doe.symbolic import SymbolicModel

    design_model = (
        symbolic_experiment(model, input_bounds=design_bounds)
        if isinstance(model, SymbolicModel)
        else model
    )
    history: list[DoERound] = []
    current_guess = dict(initial_guess)
    all_runs = [dict(r) for r in runs]
    if not all_runs:
        raise ValueError("initial_runs is empty")

    for round_idx in range(n_rounds):
        campaign = campaign_experiment(model, all_runs)
        data = campaign.data_from_runs(all_runs)
        est = campaign.estimate(data, initial_guess=current_guess)
        prior_fim = est.fim

        design: DesignResult | BatchDesignResult
        if experiments_per_round == 1:
            design = optimal_experiment(
                design_model,
                est.parameters,
                design_bounds,
                criterion=criterion,
                prior_fim=prior_fim,
            )
            round_designs = [design.design]
        else:
            design = batch_optimal_experiment(
                design_model,
                est.parameters,
                design_bounds,
                n_experiments=experiments_per_round,
                criterion=criterion,
                strategy=batch_strategy,
                prior_fim=prior_fim,
            )
            round_designs = list(design.designs)

        round_result = DoERound(
            round=round_idx,
            estimation=est,
            design=design,
            data_collected=None,
            runs=[dict(r) for r in all_runs],
        )
        if run_experiment is not None:
            collected: dict[str, np.ndarray] = {}
            for d in round_designs:
                measured = dict(run_experiment(dict(d)))
                all_runs.append({**d, **measured})
                for key, val in measured.items():
                    arr = np.atleast_1d(val)
                    collected[key] = (
                        arr if key not in collected else np.concatenate([collected[key], arr])
                    )
            round_result.data_collected = collected
            current_guess = dict(est.parameters)
        history.append(round_result)
        if callback is not None:
            callback(round_result)
        if run_experiment is None:
            break
    return history
