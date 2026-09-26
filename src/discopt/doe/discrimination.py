"""Pre-experiment design criteria for **model discrimination**.

Given several candidate ``Experiment`` instances (one per hypothesised
model structure), find the design that best separates them. Five
criteria are exposed via the :class:`DiscriminationCriterion` enum:

- ``HR`` — Hunter-Reiner (1965), squared difference of point predictions.
- ``BF`` — Buzzi-Ferraris-Forzatti (1984) multiresponse, normalized
  by measurement and prediction-variance covariances. **Default.**
- ``BH`` — Box-Hill (1967) expected entropy decrease (its upper bound), the
  prior-weighted symmetric Kullback-Leibler divergence between the models'
  Gaussian predictives.
- ``JR`` — Jensen-Rényi divergence on per-model Gaussian predictives
  (Olofsson, Deisenroth & Misener 2019). Symmetric, M-model-friendly.
- ``MI`` — Mutual information :math:`I(M; y \\mid d)` between model
  index and future data, estimated by nested Monte Carlo on the
  Gaussian predictives (Lindley 1956; Foster et al. 2019).
- ``DT`` — DT-compound (Atkinson, Bogacka & Bogacki 1998), a weighted
  blend of D-optimal precision and discrimination.

All criteria operate in **prediction space** -- only :math:`\\hat y_i(d)`
and the prediction covariance :math:`V_i = J_i \\, \\mathrm{FIM}_i^{-1}
J_i^\\top` cross model boundaries. Candidate models therefore may have
**different parameter sets** without any name alignment.

Usage
-----

>>> from discopt.doe import discriminate_design, DiscriminationCriterion
>>> result = discriminate_design(
...     experiments={"arrhenius": ArrheniusExp(), "eyring": EyringExp()},
...     param_estimates={"arrhenius": {"A": 1e3, "Ea": 50e3},
...                      "eyring":    {"dH": 50e3, "dS": 0.0}},
...     design_bounds={"T": (300.0, 700.0)},
...     criterion=DiscriminationCriterion.BF,
... )
>>> result.design                                    # {"T": 480.0}
>>> result.criterion_value                           # scalar
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from discopt.doe.design import DesignCriterion
from discopt.doe.fim import FIMResult
from discopt.estimate import Experiment

_SINGULAR_SENTINEL = 1e30
_LOG_2PI = float(np.log(2 * np.pi))


class DiscriminationCriterion(str, Enum):
    """Criteria for model-discrimination experimental design."""

    HR = "hunter_reiner"
    BF = "buzzi_ferraris"
    BH = "box_hill"
    JR = "jensen_renyi"
    MI = "mutual_information"
    DT = "dt_compound"


@dataclass
class DiscriminationDesignResult:
    """Result of :func:`discriminate_design` /
    :func:`discriminate_compound`.

    Attributes
    ----------
    design : dict[str, float]
        Optimal design point.
    criterion : DiscriminationCriterion
        Criterion that was optimised.
    criterion_value : float
        Criterion value at the optimal design (after sign normalisation
        for maximisation).
    fim_results : dict[str, FIMResult]
        Per-model FIMResult evaluated at the optimal design.
    predicted_responses : dict[str, dict[str, float]]
        Per-model predicted response means at the optimal design.
    prediction_covariances : dict[str, numpy.ndarray]
        Per-model ``V_i = J_i FIM_i^{-1} J_i^T`` at the optimal design,
        indexed by model name; arrays of shape ``(n_responses, n_responses)``.
    pairwise_divergence : numpy.ndarray or None
        ``(M, M)`` array of pairwise contributions to the criterion
        (e.g. squared differences for HR, weighted norms for BF).
        ``None`` for criteria where this concept does not apply
        (currently MI).
    model_names : list[str]
        Ordered names of candidate models.
    warnings : list[str]
        Human-readable flags surfaced during optimisation.
    """

    design: dict[str, float]
    criterion: DiscriminationCriterion
    criterion_value: float
    fim_results: dict[str, FIMResult]
    predicted_responses: dict[str, dict[str, float]]
    prediction_covariances: dict[str, np.ndarray]
    pairwise_divergence: np.ndarray | None
    model_names: list[str]
    warnings: list[str]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def evaluate_discrimination_criterion(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design: dict[str, float],
    *,
    criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    model_priors: dict[str, float] | None = None,
    prior_fims: dict[str, np.ndarray] | None = None,
    mi_samples: int = 2000,
    seed: int | None = None,
) -> float:
    """Evaluate a discrimination criterion at a single design point.

    Useful for plotting the criterion surface or for diagnosing why
    :func:`discriminate_design` picked a particular point. Uses the
    same predictive-covariance machinery as the optimiser, so the
    returned value matches ``DiscriminationDesignResult.criterion_value``
    when ``design`` equals ``result.design``.

    Parameters
    ----------
    experiments, param_estimates : as in :func:`discriminate_design`.
    design : dict[str, float]
        Design point at which to evaluate the criterion.
    criterion : DiscriminationCriterion, default BF
        Which criterion to evaluate. ``DT`` is not supported here;
        evaluate its components (D-optimal + a discrimination criterion)
        separately.
    model_priors, prior_fims, mi_samples, seed : as in :func:`discriminate_design`.
        Pass the same ``prior_fims`` the design was built with, or the value
        will not match ``DiscriminationDesignResult.criterion_value``.

    Returns
    -------
    float
        The (positive, to-be-maximised) criterion value.
    """
    _validate_inputs(experiments, param_estimates, {k: (v, v) for k, v in design.items()})
    model_names = list(experiments.keys())
    weights = _normalise_priors(model_priors, model_names)
    _validate_prior_fims(prior_fims, experiments, param_estimates)
    rng = np.random.default_rng(seed)
    rng_seed = int(rng.integers(0, 2**31 - 1))
    preds = _predict_all_models(experiments, param_estimates, design, prior_fims)
    value, _ = _evaluate_criterion(criterion, preds, weights, mi_samples, rng_seed)
    return float(value)


def discriminate_design(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    model_priors: dict[str, float] | None = None,
    prior_fims: dict[str, np.ndarray] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    mi_samples: int = 2000,
    seed: int | None = None,
) -> DiscriminationDesignResult:
    """Find the design that best separates the candidate models.

    Parameters
    ----------
    experiments : dict[str, Experiment]
        Candidate models keyed by user-chosen names.
    param_estimates : dict[str, dict[str, float]]
        Nominal parameter values per model. Keys must match
        ``experiments`` exactly. Each model may have its own parameter
        names; names need not be aligned across models.
    design_bounds : dict[str, tuple[float, float]]
        Lower / upper bounds for each design input variable. Keys must
        be a subset of the design inputs of every candidate model.
    criterion : DiscriminationCriterion, default BF
        Which discrimination criterion to optimise.
    model_priors : dict[str, float], optional
        Prior probability per model (used by HR, BF, JR, MI). Defaults
        to a uniform prior.
    prior_fims : dict[str, numpy.ndarray], optional
        Per-model FIM accumulated from data collected so far (ordered by that
        model's parameter names). When given, each model's prediction
        covariance ``V`` reflects real parameter uncertainty instead of the
        candidate design's own FIM. Recommended when driving discrimination
        from a growing dataset; without it the stateless approximation applies.
    n_starts : int, default 10
        Number of random starting points. Bound and centre points are added to
        these, and with ``local_refine`` **every** feasible start is descended
        separately, so raising this genuinely widens the search rather than
        only sampling more densely. Cost grows with it accordingly.
    local_refine : bool, default True
        If True, descend from each start with L-BFGS-B and keep the best
        result. If False, the best sampled point is returned as-is.
    mi_samples : int, default 2000
        Outer sample count for the MI nested Monte Carlo estimator.
    seed : int, optional
        RNG seed for reproducibility (used by multi-start sampling and
        the MI estimator).

    Returns
    -------
    DiscriminationDesignResult
    """
    _validate_inputs(experiments, param_estimates, design_bounds)
    if DiscriminationCriterion(criterion) is DiscriminationCriterion.DT:
        raise ValueError(
            "DT-compound is not a standalone criterion; call "
            "discriminate_compound() instead of discriminate_design(..., criterion=DT)."
        )
    model_names = list(experiments.keys())
    weights = _normalise_priors(model_priors, model_names)
    _validate_prior_fims(prior_fims, experiments, param_estimates)

    rng = np.random.default_rng(seed)
    rng_seed = int(rng.integers(0, 2**31 - 1))

    # Build and compile every model once; the optimiser then only swaps design
    # values (see _Predictor). Rebuilding per design point dominated the cost.
    predict = _Predictor(experiments, param_estimates, prior_fims)
    last_exc: list[BaseException] = []

    def objective(design: dict[str, float]) -> float:
        """Return *negative* criterion value for minimisation."""
        try:
            preds = predict(design)
            value, _ = _evaluate_criterion(criterion, preds, weights, mi_samples, rng_seed)
        except Exception as e:  # noqa: BLE001 -- root cause surfaced below
            last_exc.clear()
            last_exc.append(e)
            return _SINGULAR_SENTINEL
        if not np.isfinite(value):
            return _SINGULAR_SENTINEL
        return -value  # all discrimination criteria are maximised

    best_design = _optimize_over_design(
        objective, design_bounds, n_starts=n_starts, local_refine=local_refine, seed=rng_seed
    )
    if best_design is None:
        msg = "No feasible design found for discrimination"
        if last_exc:
            raise RuntimeError(msg) from last_exc[-1]
        raise RuntimeError(msg)

    # Final evaluation at the optimum to populate the result.
    preds = predict(best_design)
    crit_value, pairwise = _evaluate_criterion(criterion, preds, weights, mi_samples, rng_seed)

    return DiscriminationDesignResult(
        design=best_design,
        criterion=DiscriminationCriterion(criterion),
        criterion_value=float(crit_value),
        fim_results={name: preds[name].fim_result for name in model_names},
        predicted_responses={
            name: dict(zip(preds[name].response_names, preds[name].y_hat)) for name in model_names
        },
        prediction_covariances={name: preds[name].V for name in model_names},
        pairwise_divergence=pairwise,
        model_names=model_names,
        warnings=[],
    )


def discriminate_compound(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    *,
    discrimination_weight: float = 0.5,
    precision_criterion: str = DesignCriterion.D_OPTIMAL,
    discrimination_criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    normalize: bool = False,
    precision_model: str | None = None,
    model_priors: dict[str, float] | None = None,
    prior_fims: dict[str, np.ndarray] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    mi_samples: int = 2000,
    seed: int | None = None,
) -> DiscriminationDesignResult:
    r"""DT-compound design balancing precision and discrimination.

    Optimises ``(1 - λ) * Φ_precision(M_p) + λ * Φ_discrimination``
    where ``Φ_precision`` is one of the standard FIM criteria
    evaluated against the ``precision_model`` and ``Φ_discrimination``
    is the requested discrimination criterion.

    Parameters
    ----------
    discrimination_weight : float, default 0.5
        ``λ ∈ [0, 1]``. ``λ = 0`` collapses to pure D-optimal for
        ``precision_model``; ``λ = 1`` collapses to the requested
        ``discrimination_criterion``.
    precision_criterion : DesignCriterion, default D_OPTIMAL
        FIM criterion used for the precision term.
    discrimination_criterion : DiscriminationCriterion, default BF
        Discrimination objective `Φ_discrimination`.
    normalize : bool, default False
        Combine **log-efficiencies** instead of raw criterion values.

        The default sums the two criteria as they come, and they are not on a
        common scale: ``Φ_D`` is a log-determinant, whose *differences* between
        designs do not depend on the parameterisation or on σ, while a
        discrimination criterion such as BF scales as ``1/σ²``. So ``λ`` is not
        dimensionless, and the weight at which the design switches from
        precision-led to discrimination-led moves when you restate the noise --
        on the worked example in the book, halving σ moves that crossover from
        λ ≈ 0.49 to λ ≈ 0.20. ``λ = 0.5`` does not mean "balanced".

        With ``normalize=True`` each term is divided by its own optimum first,
        which is the compound criterion as Atkinson, Bogacka & Bogacki (1998)
        and Cook & Wong (1994) define it:

        ``(1-λ)/p * log(|M(d)| / |M(d*_D)|) + λ * log(Φ_T(d) / Φ_T(d*_T))``

        Both terms are then ≤ 0, both are 0 at their own optimum, and λ is a
        dimensionless trade-off that means the same thing across problems. It
        costs two extra optimisations (one per reference optimum) and requires
        ``precision_criterion="determinant"`` and a positive discrimination
        value. The endpoints still collapse to the pure designs either way.
    precision_model : str, optional
        Which model anchors the precision objective. Defaults to the
        lexicographically first key in ``experiments`` and a warning
        is added to ``result.warnings``.
    prior_fims : dict[str, numpy.ndarray], optional
        Per-model FIM of the data already collected (ordered by that model's
        parameter names). Both terms then account for it: the precision term
        is evaluated on ``prior_fims[precision_model] + FIM(d)`` -- so
        ``λ = 0`` is the D-optimal *next* run given the data -- and the
        discrimination term uses the prediction covariances of
        :func:`discriminate_design`, so ``λ = 1`` reproduces its design.
        Without it both terms treat the candidate run as the only data.
    \*\*kwargs : dict
        Additional parameters; see :func:`discriminate_design`.

    Returns
    -------
    DiscriminationDesignResult
        ``criterion`` is set to :attr:`DiscriminationCriterion.DT` and
        ``criterion_value`` is the compound objective at the optimum.
    """
    if not 0.0 <= discrimination_weight <= 1.0:
        raise ValueError("discrimination_weight must be in [0, 1]")
    _validate_inputs(experiments, param_estimates, design_bounds)
    model_names = list(experiments.keys())
    weights = _normalise_priors(model_priors, model_names)

    warnings_out: list[str] = []
    if precision_model is None:
        precision_model = sorted(experiments.keys())[0]
        warnings_out.append(f"precision_model not specified; defaulting to {precision_model!r}")
    if precision_model not in experiments:
        raise KeyError(f"precision_model {precision_model!r} not in experiments")
    _validate_prior_fims(prior_fims, experiments, param_estimates)
    prior_prec = None if prior_fims is None else prior_fims.get(precision_model)

    rng = np.random.default_rng(seed)
    rng_seed = int(rng.integers(0, 2**31 - 1))
    lam = float(discrimination_weight)
    predict = _Predictor(experiments, param_estimates, prior_fims)

    def precision(pred: _ModelPrediction) -> float:
        return _precision_value(_with_prior(pred.fim_result, prior_prec), precision_criterion)

    def _terms(design: dict[str, float]) -> tuple[float, float] | None:
        try:
            preds = predict(design)
            disc_value, _ = _evaluate_criterion(
                discrimination_criterion, preds, weights, mi_samples, rng_seed
            )
            prec_value = precision(preds[precision_model])
        except Exception:
            return None
        if not (np.isfinite(disc_value) and np.isfinite(prec_value)):
            return None
        return prec_value, disc_value

    n_params = len(param_estimates[precision_model]) or 1
    ref_prec = ref_disc = None
    if normalize:
        if precision_criterion != DesignCriterion.D_OPTIMAL:
            raise ValueError(
                "normalize=True defines efficiency against the D-optimum, so it needs "
                f"precision_criterion='determinant', got {precision_criterion!r}"
            )
        # Each term is measured against its own best achievable value, so both
        # are log-efficiencies: zero at their own optimum, negative elsewhere.
        ref_prec = _reference_optimum(
            lambda d: (lambda t: -t[0] if t else _SINGULAR_SENTINEL)(_terms(d)),
            _terms,
            0,
            design_bounds,
            n_starts,
            local_refine,
            rng_seed,
        )
        ref_disc = _reference_optimum(
            lambda d: (lambda t: -t[1] if t else _SINGULAR_SENTINEL)(_terms(d)),
            _terms,
            1,
            design_bounds,
            n_starts,
            local_refine,
            rng_seed,
        )
        if ref_disc is None or ref_disc <= 0.0:
            raise ValueError(
                "normalize=True needs a positive discrimination optimum to take a ratio "
                f"against, got {ref_disc!r}; use normalize=False for this criterion"
            )

    def _combine(prec_value: float, disc_value: float) -> float:
        if not normalize:
            return (1.0 - lam) * prec_value + lam * disc_value
        prec_eff = (prec_value - ref_prec) / n_params
        disc_eff = np.log(disc_value / ref_disc) if disc_value > 0 else -np.inf
        return (1.0 - lam) * prec_eff + lam * disc_eff

    def objective(design: dict[str, float]) -> float:
        terms = _terms(design)
        if terms is None:
            return _SINGULAR_SENTINEL
        value = _combine(*terms)
        return _SINGULAR_SENTINEL if not np.isfinite(value) else -value

    best_design = _optimize_over_design(
        objective, design_bounds, n_starts=n_starts, local_refine=local_refine, seed=rng_seed
    )
    if best_design is None:
        raise RuntimeError("No feasible design found for compound discrimination")

    preds = predict(best_design)
    disc_value, pairwise = _evaluate_criterion(
        discrimination_criterion, preds, weights, mi_samples, rng_seed
    )
    prec_value = precision(preds[precision_model])
    compound_value = _combine(prec_value, disc_value)

    return DiscriminationDesignResult(
        design=best_design,
        criterion=DiscriminationCriterion.DT,
        criterion_value=float(compound_value),
        fim_results={name: preds[name].fim_result for name in model_names},
        predicted_responses={
            name: dict(zip(preds[name].response_names, preds[name].y_hat)) for name in model_names
        },
        prediction_covariances={name: preds[name].V for name in model_names},
        pairwise_divergence=pairwise,
        model_names=model_names,
        warnings=warnings_out,
    )


# ─────────────────────────────────────────────────────────────────────
# Internal: prediction + covariance per model
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _ModelPrediction:
    """Bundled per-model evaluation at a single design point."""

    y_hat: np.ndarray  # shape (R,)
    V: np.ndarray  # prediction covariance, shape (R, R)
    Sigma_y: np.ndarray  # measurement noise diag, shape (R, R)
    response_names: list[str]
    fim_result: FIMResult


def _predict_all_models(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_values: dict[str, float],
    prior_fims: dict[str, np.ndarray] | None = None,
) -> dict[str, _ModelPrediction]:
    """Compute (y_hat, V, Sigma_y, FIM, J) for every model at one design.

    All criteria index ``y_hat``/``V``/``Sigma_y`` positionally, so every
    model must expose the same responses in the same order; otherwise the
    pairwise differences would silently misalign. This is checked once here
    (previously only the BF criterion validated it).

    ``prior_fims`` optionally maps a model name to its accumulated FIM so the
    prediction covariance reflects real parameter uncertainty (see
    :func:`_predict_with_covariance`).
    """
    prior_fims = prior_fims or {}
    preds = {
        name: _predict_with_covariance(
            experiments[name], param_estimates[name], design_values, prior_fims.get(name)
        )
        for name in experiments
    }
    _check_response_alignment(preds)
    return preds


def _predict_with_covariance(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float],
    prior_fim: np.ndarray | None = None,
) -> _ModelPrediction:
    """Evaluate y_hat, the FIM, and the prediction covariance V at one design.

    Mirrors the pipeline used inside :func:`discopt.doe.fim.compute_fim`
    -- including its solve-free assembly of ``x*`` -- with the addition
    that we also read the predicted response values from the same point,
    avoiding a second model build per design point.

    ``prior_fim`` (if given, ordered by the model's parameter names) is the FIM
    accumulated from the data collected so far; the prediction covariance is
    then ``V = J Cov(theta) J^T`` with ``Cov(theta) = pinv(prior_fim)``. Without
    it, ``V`` falls back to the single candidate design's own FIM -- a stateless
    approximation that is independent of how well the parameters are actually
    known and treats non-identifiable directions as zero-variance.
    """
    from discopt.doe.fim import (
        _assemble_x_flat_direct,
        _compile_response,
        _total_jacobian,
        fim_parameter_names,
    )
    from discopt.parametric import extract_x_flat, flatten_params

    em = experiment.create_model(**param_values)

    # Fast path, same as compute_fim: for a pure explicit response model x* is
    # fully determined by the nominal parameters and the fixed design, so
    # assemble it directly. The QP this replaces was not merely wasted work --
    # its objective Sigma(theta - theta_nom)^2 is badly scaled whenever a
    # parameter is large (an activation energy of ~1e5 leaves a KKT residual
    # above the solver's absolute 1e-6 stationarity tolerance), and from
    # discopt 0.8 on that guard rejects the point and returns status="error"
    # instead of a solution, which took down every discrimination design over
    # such a model.
    x_flat = _assemble_x_flat_direct(em, param_values, design_values)

    if x_flat is None:
        # General path: a constrained / implicit-state model genuinely needs a
        # solve. Fix the design variables first (same recipe as compute_fim).
        for name, val in design_values.items():
            if name not in em.design_inputs:
                continue
            var = em.design_inputs[name]
            val_arr = np.asarray(float(val), dtype=np.float64)
            if var.shape:
                val_arr = np.full(var.shape, val_arr)
            var.lb = val_arr
            var.ub = val_arr

        # Trivial dummy objective to pin parameters at their nominal values.
        em.model.minimize(
            sum((em.unknown_parameters[n] - param_values[n]) ** 2 for n in em.parameter_names)
        )
        result = em.model.solve()

        x_flat = extract_x_flat(result, em.model)

    # Compile response functions and compute predicted means.
    response_fns = [_compile_response(em.responses[n], em.model) for n in em.response_names]
    p_flat = flatten_params(em.model)
    y_hat = np.array([float(np.asarray(fn(x_flat, p_flat)).flat[0]) for fn in response_fns])

    # Jacobian of responses w.r.t. unknown parameters, through any implicit
    # states (a constrained model's states move with θ).
    J = _total_jacobian(em, response_fns, x_flat, p_flat)

    sigma = np.array([em.measurement_error[n] for n in em.response_names], dtype=np.float64)
    Sigma_y = np.diag(sigma**2)
    Sigma_inv = np.diag(1.0 / sigma**2)
    fim = J.T @ Sigma_inv @ J
    # Propagate parameter uncertainty into the prediction covariance. Prefer the
    # prior FIM (parameter knowledge from data collected so far); fall back to
    # the candidate design's own FIM when none is supplied.
    cov_theta = np.linalg.pinv(np.asarray(prior_fim) if prior_fim is not None else fim)
    V = J @ cov_theta @ J.T

    fim_result = FIMResult(
        fim=np.asarray(fim),
        jacobian=J,
        parameter_names=fim_parameter_names(em),
        response_names=em.response_names,
    )

    return _ModelPrediction(
        y_hat=y_hat,
        V=np.asarray(V),
        Sigma_y=Sigma_y,
        response_names=em.response_names,
        fim_result=fim_result,
    )


def _with_prior(fim_result: FIMResult, prior: np.ndarray | None) -> FIMResult:
    """``fim_result`` with ``prior`` added to its information (or unchanged)."""
    if prior is None:
        return fim_result
    return FIMResult(
        fim=np.asarray(fim_result.fim) + np.asarray(prior, dtype=np.float64),
        jacobian=fim_result.jacobian,
        parameter_names=fim_result.parameter_names,
        response_names=fim_result.response_names,
    )


def _validate_prior_fims(
    prior_fims: dict[str, np.ndarray] | None,
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
) -> None:
    """Check keys and shapes of ``prior_fims`` against each model's parameters."""
    if not prior_fims:
        return
    unknown = set(prior_fims) - set(experiments)
    if unknown:
        raise ValueError(f"prior_fims has keys {sorted(unknown)} that are not candidate models")
    for name, fim in prior_fims.items():
        p = len(experiments[name].create_model(**param_estimates[name]).parameter_names)
        shape = np.asarray(fim).shape
        if shape != (p, p):
            raise ValueError(
                f"prior_fims[{name!r}] has shape {shape}; model {name!r} has {p} "
                "parameters, so it must be ({p}, {p}) ordered by its parameter names".format(p=p)
            )


class _ModelEvaluator:
    """One model at fixed parameter values, compiled once for many designs.

    :func:`_predict_with_covariance` rebuilds the model, recompiles every
    response and retraces the Jacobian on each call; inside a design
    optimisation that is dozens of rebuilds per model for what is, apart from
    the design values, the same computation. Here the model is built once, the
    response vector and its Jacobian are JIT-compiled once, and each design only
    reassembles the flat solution vector. Models whose ``x*`` needs a solve
    (constraints, implicit states) fall back to the uncached path per design.
    """

    def __init__(self, experiment: Experiment, param_values: dict[str, float]):
        from discopt.doe import fim as _fim
        from discopt.parametric import flatten_params

        self.experiment = experiment
        self.param_values = dict(param_values)
        self.em = experiment.create_model(**param_values)
        em = self.em
        self.response_names = list(em.response_names)
        fns = [_fim._compile_response(em.responses[n], em.model) for n in self.response_names]
        self.p_flat = flatten_params(em.model)
        self.param_indices = _fim._get_param_indices(em)
        sigma = np.array([em.measurement_error[n] for n in self.response_names], dtype=np.float64)
        self.Sigma_y = np.diag(sigma**2)
        self.Sigma_inv = np.diag(1.0 / sigma**2)

        jax, jnp = _fim._require_jax()
        p_flat = self.p_flat

        def vec(x_flat):
            return jnp.stack([jnp.reshape(fn(x_flat, p_flat), ()) for fn in fns])

        self._vec_raw = vec
        self._jac_raw = jax.jacobian(vec)
        self._vec = jax.jit(vec)
        self._jac = jax.jit(self._jac_raw)
        self._jit_ok = True

    def __call__(self, design: dict[str, float], prior_fim: np.ndarray | None) -> _ModelPrediction:
        from discopt.doe import fim as _fim

        x_flat = _fim._assemble_x_flat_direct(self.em, self.param_values, design)
        if x_flat is None:
            return _predict_with_covariance(self.experiment, self.param_values, design, prior_fim)
        if self._jit_ok:
            try:
                y_hat = np.asarray(self._vec(x_flat), dtype=np.float64).ravel()
                J_full = np.asarray(self._jac(x_flat), dtype=np.float64)
            except Exception:  # noqa: BLE001 - an untraceable node: run eagerly
                self._jit_ok = False
        if not self._jit_ok:
            y_hat = np.asarray(self._vec_raw(x_flat), dtype=np.float64).ravel()
            J_full = np.asarray(self._jac_raw(x_flat), dtype=np.float64)
        J = J_full[:, self.param_indices]
        fim = J.T @ self.Sigma_inv @ J
        cov_theta = np.linalg.pinv(np.asarray(prior_fim) if prior_fim is not None else fim)
        return _ModelPrediction(
            y_hat=y_hat,
            V=np.asarray(J @ cov_theta @ J.T),
            Sigma_y=self.Sigma_y,
            response_names=self.response_names,
            fim_result=FIMResult(
                fim=np.asarray(fim),
                jacobian=J,
                parameter_names=_fim.fim_parameter_names(self.em),
                response_names=self.response_names,
            ),
        )


class _Predictor:
    """``design -> {model: _ModelPrediction}`` with every model compiled once."""

    def __init__(
        self,
        experiments: dict[str, Experiment],
        param_estimates: dict[str, dict[str, float]],
        prior_fims: dict[str, np.ndarray] | None = None,
    ):
        self.prior_fims = prior_fims or {}
        self.evaluators = {
            name: _ModelEvaluator(experiments[name], param_estimates[name]) for name in experiments
        }

    def __call__(self, design: dict[str, float]) -> dict[str, _ModelPrediction]:
        preds = {
            name: ev(design, self.prior_fims.get(name)) for name, ev in self.evaluators.items()
        }
        _check_response_alignment(preds)
        return preds


def _check_response_alignment(preds: dict[str, _ModelPrediction]) -> None:
    names = list(preds)
    ref = preds[names[0]].response_names
    for name in names[1:]:
        if preds[name].response_names != ref:
            raise ValueError(
                f"models {names[0]!r} and {name!r} expose different response "
                f"namespaces ({ref} vs {preds[name].response_names}); model "
                "discrimination requires identical, identically-ordered responses."
            )


# ─────────────────────────────────────────────────────────────────────
# Internal: criterion dispatch
# ─────────────────────────────────────────────────────────────────────


def _evaluate_criterion(
    criterion: DiscriminationCriterion,
    preds: dict[str, _ModelPrediction],
    weights: dict[str, float],
    mi_samples: int,
    seed: int,
) -> tuple[float, np.ndarray | None]:
    """Dispatch to the requested criterion and return (value, pairwise)."""
    crit = DiscriminationCriterion(criterion)
    if crit is DiscriminationCriterion.HR:
        return _criterion_hunter_reiner(preds, weights)
    if crit is DiscriminationCriterion.BF:
        return _criterion_buzzi_ferraris(preds, weights)
    if crit is DiscriminationCriterion.BH:
        return _criterion_box_hill(preds, weights)
    if crit is DiscriminationCriterion.JR:
        return _criterion_jensen_renyi(preds, weights)
    if crit is DiscriminationCriterion.MI:
        value = _criterion_mutual_information(preds, weights, mi_samples, seed)
        return value, None
    if crit is DiscriminationCriterion.DT:
        raise ValueError(
            "DT-compound is not a standalone criterion; call "
            "discriminate_compound() instead of discriminate_design(..., criterion=DT)."
        )
    raise ValueError(f"Unknown discrimination criterion: {criterion!r}")


def _criterion_hunter_reiner(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray]:
    """``Σ w_i w_j ||ŷ_i − ŷ_j||²``."""
    names = list(preds.keys())
    M = len(names)
    pw = np.zeros((M, M))
    for i, j in combinations(range(M), 2):
        diff = preds[names[i]].y_hat - preds[names[j]].y_hat
        contrib = float(diff @ diff)
        pw[i, j] = pw[j, i] = contrib
    total = sum(
        weights[names[i]] * weights[names[j]] * pw[i, j] for i, j in combinations(range(M), 2)
    )
    return float(total), pw


def _criterion_buzzi_ferraris(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray]:
    """Buzzi-Ferraris–Forzatti (1984) pairwise statistic.

    ``T_ij = Δᵀ S⁻¹ Δ + tr(2Σ S⁻¹)`` with ``Δ = ŷ_i − ŷ_j`` and
    ``S = 2Σ + V_i + V_j`` (Olofsson et al. 2019). The ``2Σ`` reflects that
    ``Δ`` is a difference of two future *noisy* observations, and the trace
    term is the criterion's expected-value offset. Σ is symmetrized across the
    pair so the statistic is order-independent when the models declare
    different measurement errors.
    """
    names = list(preds.keys())
    M = len(names)
    pw = np.zeros((M, M))
    for i, j in combinations(range(M), 2):
        ni, nj = names[i], names[j]
        if preds[ni].response_names != preds[nj].response_names:
            raise ValueError(
                f"Models {ni!r} and {nj!r} have different response names; "
                "discrimination requires the same response namespace."
            )
        diff = preds[ni].y_hat - preds[nj].y_hat
        sigma = 0.5 * (preds[ni].Sigma_y + preds[nj].Sigma_y)
        S = 2.0 * sigma + preds[ni].V + preds[nj].V
        S_inv_diff = np.linalg.solve(S, diff)
        trace_term = float(np.trace(np.linalg.solve(S, 2.0 * sigma)))
        contrib = float(diff @ S_inv_diff) + trace_term
        pw[i, j] = pw[j, i] = contrib
    total = sum(
        weights[names[i]] * weights[names[j]] * pw[i, j] for i, j in combinations(range(M), 2)
    )
    return float(total), pw


def _criterion_box_hill(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray]:
    r"""Box & Hill (1967) discrimination criterion.

    Box and Hill choose the run that maximises the expected decrease in the
    entropy of the model probabilities, and maximise instead its upper bound

    .. math::
        D = \sum_{i<j} \pi_i \pi_j J_{ij},

    where, for Gaussian predictives :math:`p_i = N(\hat y_i, S_i)` with
    :math:`S_i = \Sigma_y + V_i` (measurement plus prediction covariance),
    :math:`J_{ij}` is the symmetric Kullback-Leibler (Jeffreys) divergence

    .. math::
        J_{ij} = \tfrac12\,\mathrm{tr}\big(S_i S_j^{-1} + S_j S_i^{-1} - 2I\big)
               + \tfrac12\,\Delta^\top (S_i^{-1} + S_j^{-1})\,\Delta,
        \qquad \Delta = \hat y_i - \hat y_j .

    For one response this is exactly Box and Hill's

    .. math::
        \tfrac12 \sum_{i<j} \pi_i\pi_j \Big[
        \frac{(\sigma_i^2-\sigma_j^2)^2}{(\sigma^2+\sigma_i^2)(\sigma^2+\sigma_j^2)}
        + (\hat y_i-\hat y_j)^2\Big(\frac1{\sigma^2+\sigma_i^2}
        + \frac1{\sigma^2+\sigma_j^2}\Big)\Big],

    with :math:`\sigma_i^2` model *i*'s prediction variance; the matrix form
    is its multiresponse generalisation. Unlike Hunter-Reiner it rewards runs
    where the models disagree *relative to* their predictive spread, and it
    also rewards runs where the models disagree about that spread. Box, G. E.
    P. and Hill, W. J. (1967) Technometrics 9, 57-71.
    """
    names = list(preds.keys())
    M = len(names)
    pw = np.zeros((M, M))
    for i, j in combinations(range(M), 2):
        pi, pj = preds[names[i]], preds[names[j]]
        S_i = _pd(pi.Sigma_y + pi.V)
        S_j = _pd(pj.Sigma_y + pj.V)
        n = S_i.shape[0]
        Si_inv_Sj = np.linalg.solve(S_i, S_j)
        Sj_inv_Si = np.linalg.solve(S_j, S_i)
        diff = pi.y_hat - pj.y_hat
        quad = float(diff @ np.linalg.solve(S_i, diff) + diff @ np.linalg.solve(S_j, diff))
        contrib = 0.5 * (float(np.trace(Si_inv_Sj) + np.trace(Sj_inv_Si)) - 2.0 * n + quad)
        pw[i, j] = pw[j, i] = contrib
    total = sum(
        weights[names[i]] * weights[names[j]] * pw[i, j] for i, j in combinations(range(M), 2)
    )
    return float(total), pw


def _criterion_jensen_renyi(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray | None]:
    """Jensen-Rényi divergence with α=2 on the Gaussian predictives.

    For Gaussian components ``p_i = N(μ_i, S_i)`` with ``S_i = Σ_y + V_i``
    and prior weights ``w_i``, the α=2 Rényi entropy is

        H_2(p_i) = -log ∫ p_i^2 = (n/2) log(4π) + (1/2) log det S_i

    (uses the identity ``∫ N(x;μ,S)² dx = (4π)^(-n/2) (det S)^(-1/2)``)
    and the mixture α=2 entropy is

        H_2(Σ w_i p_i) = -log Σ_{i,j} w_i w_j N(μ_i; μ_j, S_i + S_j).

    JR returns ``H_2(mixture) − Σ w_i H_2(p_i)``; equals 0 when all
    components are identical.
    """
    names = list(preds.keys())
    M = len(names)
    n = preds[names[0]].y_hat.shape[0]
    means = np.stack([preds[name].y_hat for name in names])
    covs = [preds[name].Sigma_y + preds[name].V for name in names]
    log_weights = np.array([np.log(weights[name]) for name in names])

    # Per-component α=2 entropy.
    H_components = np.array([0.5 * (n * np.log(4 * np.pi) + _logdet(covs[i])) for i in range(M)])

    # Mixture α=2 entropy via the closed-form ∫ p_i p_j = N(μ_i; μ_j, S_i + S_j).
    log_int = np.full((M, M), -np.inf)
    for i in range(M):
        for j in range(M):
            log_int[i, j] = _log_gaussian(means[i], means[j], covs[i] + covs[j])
    log_w = log_weights[:, None] + log_weights[None, :]
    H_mix = -_logsumexp((log_w + log_int).ravel())

    jr = H_mix - float(np.sum(np.exp(log_weights) * H_components))
    return float(jr), None


def _criterion_mutual_information(
    preds: dict[str, _ModelPrediction],
    weights: dict[str, float],
    mi_samples: int,
    seed: int,
) -> float:
    """``I(M; y | d)`` via nested Monte Carlo on Gaussian predictives."""
    names = list(preds.keys())
    M = len(names)
    n = preds[names[0]].y_hat.shape[0]
    means = [preds[name].y_hat for name in names]
    covs = [preds[name].Sigma_y + preds[name].V for name in names]
    w = np.array([weights[name] for name in names])
    log_w = np.log(w)

    rng = np.random.default_rng(seed)
    chol = [np.linalg.cholesky(_pd(C)) for C in covs]

    # Sample (M, y) ~ Σ w_i p_i. Vectorized: pick component indices, then draw.
    component = rng.choice(M, size=mi_samples, p=w)
    z = rng.standard_normal((mi_samples, n))
    samples = np.empty((mi_samples, n))
    for k in range(M):
        idx = np.where(component == k)[0]
        if idx.size:
            samples[idx] = means[k] + z[idx] @ chol[k].T

    # log p(y_n | M=k) for each (n, k).
    log_p = np.empty((mi_samples, M))
    for k in range(M):
        log_p[:, k] = _log_gaussian_batch(samples, means[k], covs[k])

    # H(y) ≈ -mean_n log Σ_k w_k p(y_n | M=k).
    log_marginal = _logsumexp_axis(log_p + log_w, axis=1)
    H_y = -float(np.mean(log_marginal))

    # H(y | M) = Σ w_k H(y | M=k); closed-form Gaussian entropy.
    H_yM = float(sum(w[k] * (0.5 * (n * (1.0 + _LOG_2PI) + _logdet(covs[k]))) for k in range(M)))

    return H_y - H_yM


def _precision_value(fim_result: FIMResult, criterion: str) -> float:
    """Return the precision-criterion value (with maximisation sign)."""
    if criterion == DesignCriterion.D_OPTIMAL:
        return fim_result.d_optimal  # already maximisation (log det)
    if criterion == DesignCriterion.A_OPTIMAL:
        return -fim_result.a_optimal  # trace of FIM^-1; minimise
    if criterion == DesignCriterion.E_OPTIMAL:
        return fim_result.e_optimal  # max min-eig
    if criterion == DesignCriterion.ME_OPTIMAL:
        return -fim_result.me_optimal  # condition number; minimise
    raise ValueError(f"Unknown precision_criterion: {criterion!r}")


# ─────────────────────────────────────────────────────────────────────
# Internal: validation, optimisation, math utilities
# ─────────────────────────────────────────────────────────────────────


def _validate_inputs(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
) -> None:
    if len(experiments) < 2:
        raise ValueError(
            f"Need at least 2 candidate models, got {len(experiments)}: {list(experiments)}"
        )
    if set(experiments.keys()) != set(param_estimates.keys()):
        raise ValueError(
            "experiments and param_estimates must share keys; "
            f"experiments={set(experiments)}, param_estimates={set(param_estimates)}"
        )
    if not design_bounds:
        raise ValueError("design_bounds must be non-empty")
    # Every design_bounds key must be a design input of every candidate model;
    # otherwise the optimizer scans a variable no model consumes and returns an
    # arbitrary "optimal" design (the docstring says keys must be a subset of
    # every model's design inputs).
    for name, exp in experiments.items():
        inputs = set(exp.create_model(**param_estimates[name]).design_inputs)
        unknown = [k for k in design_bounds if k not in inputs]
        if unknown:
            raise ValueError(
                f"design_bounds key(s) {unknown} are not design inputs of model "
                f"{name!r} (its design inputs are {sorted(inputs)})."
            )


def _normalise_priors(priors: dict[str, float] | None, model_names: list[str]) -> dict[str, float]:
    if priors is None:
        w = 1.0 / len(model_names)
        return {name: w for name in model_names}
    if set(priors) != set(model_names):
        raise ValueError(f"model_priors keys {set(priors)} != experiments keys {set(model_names)}")
    total = sum(priors.values())
    if total <= 0:
        raise ValueError("model_priors must sum to a positive value")
    return {name: priors[name] / total for name in model_names}


def _reference_optimum(
    pure_objective,
    terms,
    which: int,
    design_bounds: dict[str, tuple[float, float]],
    n_starts: int,
    local_refine: bool,
    seed: int,
) -> float | None:
    """Best achievable value of one term on its own, for log-efficiencies.

    ``normalize=True`` measures each criterion against its own optimum, so the
    optimum has to be found first -- one extra search per term.
    """
    best = _optimize_over_design(
        pure_objective, design_bounds, n_starts=n_starts, local_refine=local_refine, seed=seed
    )
    if best is None:
        return None
    pair = terms(best)
    return None if pair is None else float(pair[which])


def _optimize_over_design(
    objective: Callable[[dict[str, float]], float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_starts: int,
    local_refine: bool,
    seed: int,
) -> dict[str, float] | None:
    """Multi-start + optional L-BFGS-B over a scalar minimisation objective."""
    design_names = list(design_bounds.keys())
    rng = np.random.default_rng(seed)

    candidates: list[dict[str, float]] = []
    for _ in range(n_starts):
        candidates.append({n: float(rng.uniform(*design_bounds[n])) for n in design_names})
    for n in design_names:
        lo, hi = design_bounds[n]
        for val in (lo, hi):
            point = {nn: 0.5 * (design_bounds[nn][0] + design_bounds[nn][1]) for nn in design_names}
            point[n] = float(val)
            candidates.append(point)

    # Score every candidate and keep the feasible ones, best first. A value at
    # (or above) the sentinel means the objective could not be evaluated there.
    scored: list[tuple[float, dict[str, float]]] = []
    for cand in candidates:
        val = objective(cand)
        if np.isfinite(val) and val < _SINGULAR_SENTINEL:
            scored.append((float(val), cand))

    if not scored:
        return None

    scored.sort(key=lambda t: t[0])
    best_value, best_design = scored[0]

    if local_refine:
        bounds = [design_bounds[n] for n in design_names]

        # The scan already evaluated every candidate, and each descent re-asks
        # for its own starting point first, so seed a cache with what is known.
        # Keying on the exact bytes only ever merges genuinely identical points,
        # which leaves finite-difference steps untouched.
        cache: dict[bytes, float] = {}
        for val, cand in scored:
            cache[np.array([cand[n] for n in design_names], dtype=float).tobytes()] = val

        def _wrapped(x: np.ndarray) -> float:
            key = np.asarray(x, dtype=float).tobytes()
            hit = cache.get(key)
            if hit is not None:
                return hit
            val = objective({n: float(v) for n, v in zip(design_names, x)})
            cache[key] = val
            return val

        # Descend from *every* feasible start, not only from the best sample.
        # On a rugged objective the best sample often sits in a different basin
        # from the best optimum, so refining it alone makes the other starts
        # worthless: they densify the scan without ever being followed downhill.
        # Refining only a prefix of `scored` is not enough either -- the bound
        # and centre points are here for their exploration value, not their
        # score, so a fixed budget drops exactly the starts worth keeping.
        failures: list[str] = []
        for _, cand in scored:
            x0 = np.array([cand[n] for n in design_names], dtype=float)
            try:
                res = minimize(_wrapped, x0, method="L-BFGS-B", bounds=bounds)
            except Exception as e:  # noqa: BLE001
                failures.append(str(e))
                continue
            if np.isfinite(res.fun) and res.fun < best_value:
                best_value = float(res.fun)
                best_design = {n: float(v) for n, v in zip(design_names, res.x)}
        if failures:
            fell_back = "the best sampled point" if len(failures) == len(scored) else "the best"
            warnings.warn(
                f"{len(failures)} of {len(scored)} discrimination local refinements failed "
                f"(first: {failures[0]}); using {fell_back} result available.",
                stacklevel=2,
            )

    return best_design


def _logdet(M: np.ndarray) -> float:
    """``log det`` via SVD; safe for symmetric PD matrices."""
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        # Regularise.
        eps = max(1e-12, 1e-10 * float(np.trace(M)) / max(M.shape[0], 1))
        sign, val = np.linalg.slogdet(M + eps * np.eye(M.shape[0]))
    return float(val)


def _log_gaussian(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """``log N(x; μ, Σ)`` for a single sample (regularises Σ if needed)."""
    n = x.size
    diff = x - mu
    cov_safe = _pd(cov)
    L = np.linalg.cholesky(cov_safe)
    sol = np.linalg.solve(L, diff)
    quad = float(sol @ sol)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (n * _LOG_2PI + log_det + quad)


def _log_gaussian_batch(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``log N(X; μ, Σ)`` for a batch ``X`` of shape ``(N, n)``."""
    n = X.shape[1]
    cov_safe = _pd(cov)
    L = np.linalg.cholesky(cov_safe)
    diff = (X - mu).T  # (n, N)
    sol = np.linalg.solve(L, diff)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return np.asarray(-0.5 * (n * _LOG_2PI + log_det + quad))


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _logsumexp_axis(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.asarray(
        (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)
    )


def _pd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``M`` regularised to be symmetric positive-definite."""
    M = 0.5 * (M + M.T)
    diag_floor = max(eps, eps * float(np.trace(np.abs(M))) / max(M.shape[0], 1))
    return np.asarray(M + diag_floor * np.eye(M.shape[0]))


__all__ = [
    "DiscriminationCriterion",
    "DiscriminationDesignResult",
    "discriminate_compound",
    "discriminate_design",
    "evaluate_discrimination_criterion",
]
