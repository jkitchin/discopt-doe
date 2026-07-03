"""Sequential model-based design of experiments.

Alternates between parameter estimation and optimal design in a loop:
1. Estimate parameters from all collected data
2. Compute FIM and optimize next experiment design
3. (Optionally) run experiment and collect new data
4. Repeat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from discopt.doe.design import (
    BatchDesignResult,
    BatchStrategy,
    DesignCriterion,
    DesignResult,
    batch_optimal_experiment,
    optimal_experiment,
)
from discopt.estimate import (
    EstimationResult,
    Experiment,
    estimate_parameters,
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
    """

    round: int
    estimation: EstimationResult
    design: DesignResult | BatchDesignResult
    data_collected: dict[str, Union[float, np.ndarray]] | None = None


def sequential_doe(
    experiment: Experiment,
    initial_data: dict[str, Union[float, np.ndarray]],
    initial_guess: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_rounds: int = 5,
    criterion: str = DesignCriterion.D_OPTIMAL,
    experiments_per_round: int = 1,
    batch_strategy: str = BatchStrategy.GREEDY,
    run_experiment: Callable[[dict[str, float]], dict[str, float]] | None = None,
    callback: Callable[[DoERound], None] | None = None,
) -> list[DoERound]:
    """Run the full sequential MBDoE loop.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    initial_data : dict
        Initial experimental data for first estimation.
    initial_guess : dict[str, float]
        Starting parameter estimates.
    design_bounds : dict[str, tuple[float, float]]
        Bounds on design variables.
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
    callback : callable, optional
        Called with each ``DoERound`` after it completes.

    Returns
    -------
    list[DoERound]
        History of all rounds.
    """
    if experiments_per_round < 1:
        raise ValueError(f"experiments_per_round must be >= 1, got {experiments_per_round}")

    history = []
    current_guess = dict(initial_guess)
    all_data = dict(initial_data)
    prior_fim = None

    for round_idx in range(n_rounds):
        # Step 1: Estimate parameters from all data
        est = estimate_parameters(
            experiment,
            all_data,
            initial_guess=current_guess,
        )

        # Accumulate FIM as prior
        prior_fim = est.fim if prior_fim is None else prior_fim + est.fim

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
