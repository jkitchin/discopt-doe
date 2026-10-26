"""Model-based active-learning optimization.

When to use this module
-----------------------

Use :func:`model_based_optimize_round` when:

* you already have a **mechanistic / parametric model** of the response
  -- a kinetic rate law, a heat-transfer correlation, a competitive
  isotherm -- expressed as a :class:`discopt.estimate.Experiment`;
* you trust the functional form enough that you want to extrapolate, or
* runs are expensive enough that you cannot afford the data a GP needs
  to learn the shape on its own.

Contrast with :func:`discopt.doe.optimize_round` -- which uses an
*empirical* surrogate (GP, response-surface polynomial, LPR) that knows
nothing about the underlying physics. The empirical version is the right
default when you have no model in hand; the model-based version is the
right tool when you do.

Contrast with :func:`discopt.doe.optimal_experiment` -- which uses a
parametric model too, but optimizes the experimental design for
**parameter precision** (D-/A-/E-optimality on the FIM), not for the
best *response*. This module is the response-optimization counterpart
that reuses the same FIM machinery for uncertainty quantification.

How one round works
-------------------

A single call to :func:`model_based_optimize_round`:

1. opens the workbook and reads the completed runs;
2. refits the experiment's unknown parameters by maximum likelihood
   (Gauss-Newton via :func:`scipy.optimize.least_squares` with a
   JAX-computed Jacobian);
3. computes the parameter covariance ``Σ_θ ≈ FIM⁻¹`` at the fitted
   point;
4. draws a pool of ``n_candidates`` Sobol points inside the input box;
5. for each candidate ``d``, predicts the response mean
   ``μ(d) = f(d; θ̂)`` and the linearized predictive variance

       σ²(d) = ∇θ f(d; θ̂) · Σ_θ · ∇θ f(d; θ̂)ᵀ + σ²_meas

   -- the first term is *parameter uncertainty* (shrinks as more data
   comes in), the second is *measurement noise* (constant, set by the
   workbook's declared σ);
6. scores candidates with the chosen acquisition function (same EI/UCB
   used by :func:`optimize_round`);
7. greedily picks the top ``batch_size``, using mean-imputation fantasy
   refits to diversify within a batch;
8. appends the batch to the workbook as pending runs.

The acquisition functions consume only ``(mean, std)`` and do not care
how those came to be -- so EI on a GP and EI on a mechanistic model
look identical at the boundary.

Tradeoffs vs. an empirical surrogate
------------------------------------

============================  =======================  =======================
Property                      Empirical (GP / RSM)     Model-based (this)
============================  =======================  =======================
Functional form               learned from data        supplied by user
Extrapolation                 unsafe                   safe *iff* model right
Sample efficiency             ~ 5-20 d points          ~ 1-2 d points
Per-round cost                surrogate fit            param-estimation NLP
Categorical inputs            tricky                   natural (model decides)
Wrong-model risk              none                     systematic bias
============================  =======================  =======================

If you cannot decide -- run both in parallel; the agreement (or lack
of it) is itself a diagnostic.

Quick start
-----------

>>> from discopt.doe import model_based_optimize_round, OptimizationCriterion
>>> from my_kinetics import MichaelisMenten   # an Experiment subclass
>>> result = model_based_optimize_round(
...     workbook="reactor.xlsx",
...     experiment=MichaelisMenten(),
...     initial_guess={"Vmax": 1.0, "Km": 0.5},
...     criterion=OptimizationCriterion.MAXIMIZE,
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.parameters)
>>> print(result.next_designs)

References
----------

* Asprey, Macchietto (2000). *Statistical tools for optimal dynamic
  model building.* C&CE 24:1261-1267. -- model-based DoE foundations.
* Franceschini, Macchietto (2008). *Model-based design of experiments
  for parameter precision: state of the art.* Chem.Eng.Sci. 63:4846. --
  survey including the response-optimization variant used here.
* Jones, Schonlau, Welch (1998). *Efficient Global Optimization of
  Expensive Black-Box Functions.* -- the EI machinery this module reuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from discopt.doe.acquisition import call_acquisition, resolve_acquisition
from discopt.doe.optimize import (
    OptimizationCriterion,
    _chebyshev_scalarization,
    _objective_matrix,
    _pareto_mask,
    _sample_candidates,
)
from discopt.doe.workbook import Workbook
from discopt.estimate import Experiment


@dataclass
class ModelBasedRoundResult:
    """Outcome of one model-based active-learning round.

    Same shape as :class:`~discopt.doe.OptimizationRoundResult` plus
    fitted-parameter diagnostics.
    """

    next_designs: list[dict[str, float]]
    new_run_ids: list[int]
    incumbent_x: dict[str, float] | None
    incumbent_y: float | None
    acquisition_scores: list[float]
    surrogate_mode: str | None
    n_completed: int
    workbook_path: str
    parameters: dict[str, float] = field(default_factory=dict)
    parameter_se: dict[str, float] = field(default_factory=dict)
    fim_log_det: float | None = None
    log: list[str] = field(default_factory=list)
    pareto_front: list[dict[str, Any]] | None = None
    """With several objectives, the non-dominated completed runs (measured)."""
    predicted_front: list[dict[str, Any]] | None = None
    """With several objectives, the trade-off curve the *fitted model* predicts:
    the non-dominated candidates, each with its predicted objectives and their
    standard errors (``<name>_se``). Unlike ``pareto_front`` it covers the whole
    design space, including where no run has been made yet -- a model's
    extrapolation, to be trusted as far as the model is."""
    scalarization_weights: list[list[float]] | None = None
    """The objective weights drawn for each recommended run (ParEGO)."""


@dataclass
class _CompiledPredictor:
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    y_vec: Callable[[Any, Any], Any]
    """``y_vec(design_vector, theta_vector)`` -> the responses, in the order asked for."""


def _compile_predictor(
    experiment: Experiment,
    input_names: Sequence[str],
    response_names: Sequence[str],
    initial_guess: dict[str, float],
) -> _CompiledPredictor:
    """A JAX function of (design, parameters) for the named responses."""
    from discopt.doe.fim import _compile_response, _design_source_map, _require_jax
    from discopt.parametric import flatten_params, variable_slices

    _, jnp = _require_jax()

    em = experiment.create_model(**initial_guess)
    if _design_source_map(em) is None:
        # The compiled predictor below sets only the design inputs and the
        # parameters in x*; any other variable (a state fixed by a
        # constraint) would silently be evaluated at 0.
        raise ValueError(
            "ParametricSurrogate needs an explicit-response Experiment (no "
            "constraints; every variable an unknown parameter or a design input). "
            "Write the response as an expression of the parameters and inputs, or "
            "define implicit states with discopt.modeling.implicit."
        )
    for name in response_names:
        if name not in em.responses:
            raise ValueError(
                f"experiment has no response {name!r}; available: {list(em.responses)}"
            )
    missing = [n for n in input_names if n not in em.design_inputs]
    if missing:
        raise ValueError(
            f"workbook inputs {missing!r} not found in experiment.design_inputs "
            f"{list(em.design_inputs)!r}"
        )
    parameter_names = list(em.unknown_parameters.keys())

    # Parameter box bounds (for the bounded least-squares solver), aligned
    # with parameter_names. Missing bounds fall back to +/- inf.
    parameter_bounds = [
        (
            float(getattr(em.unknown_parameters[n], "lb", -np.inf)),
            float(getattr(em.unknown_parameters[n], "ub", np.inf)),
        )
        for n in parameter_names
    ]

    # Flat offsets for every variable in the model.
    slices = variable_slices(em.model)
    n_x = max((sl.stop for sl in slices.values()), default=0)

    design_idx = [slices[em.design_inputs[n].name].start for n in input_names]
    param_idx = [slices[em.unknown_parameters[n].name].start for n in parameter_names]

    # p_flat for any model Parameters (distinct from unknown parameters).
    p_flat_const = flatten_params(em.model)

    response_fns = [_compile_response(em.responses[n], em.model) for n in response_names]
    d_idx = jnp.asarray(design_idx, dtype=jnp.int32)
    p_idx = jnp.asarray(param_idx, dtype=jnp.int32)

    def y_vec(d_vec, theta_vec):
        x_flat = jnp.zeros(n_x, dtype=jnp.float64)
        x_flat = x_flat.at[d_idx].set(d_vec)
        x_flat = x_flat.at[p_idx].set(theta_vec)
        return jnp.stack([jnp.reshape(fn(x_flat, p_flat_const), ()) for fn in response_fns])

    return _CompiledPredictor(parameter_names, parameter_bounds, y_vec)


class ParametricSurrogate:
    """Mechanistic-model surrogate for active-learning optimization.

    Implements the :class:`~discopt.doe.surrogate.Surrogate` protocol on
    top of a :class:`~discopt.estimate.Experiment`. ``fit(X, y)`` refits
    the unknown parameters by maximum likelihood; ``predict(X)`` returns
    ``(μ, σ)`` where σ comes from linearizing the response in the
    parameters around the fit and adding measurement noise in
    quadrature.

    The class attribute ``_is_discopt_surrogate = True`` tells
    :func:`~discopt.doe.surrogate.coerce_surrogate` not to re-wrap this
    in the sklearn adapter.

    Parameters
    ----------
    experiment : Experiment
        Parametric model.
    input_names : sequence of str
        Names of the design-input columns in the X matrix, in column
        order. Each name must appear in ``experiment.design_inputs``.
    response_name : str
        Key in ``experiment.responses`` whose value is the predicted
        scalar response.
    initial_guess : dict[str, float], optional
        Starting values for the parameter NLS. Defaults to 1.0 for any
        parameter not provided.
    measurement_noise_var : float
        ``σ²_meas`` -- variance of the measurement noise. Added in
        quadrature to the linearized predictive variance.
    regularize : float, default 1e-12
        Ridge added to the FIM before inversion to keep the covariance
        well-defined in the early rounds.
    """

    _is_discopt_surrogate = True

    def __init__(
        self,
        experiment: Experiment,
        *,
        input_names: Sequence[str],
        response_name: str,
        initial_guess: dict[str, float] | None = None,
        measurement_noise_var: float = 1.0,
        regularize: float = 1e-12,
    ) -> None:
        self.experiment = experiment
        self.input_names = list(input_names)
        self.response_name = response_name
        self.initial_guess = dict(initial_guess or {})
        self.measurement_noise_var = float(measurement_noise_var)
        self.regularize = float(regularize)
        self.mode = "parametric"

        self.parameters_: dict[str, float] | None = None
        self.parameter_names_: list[str] = []
        self.covariance_: np.ndarray | None = None
        self.fim_: np.ndarray | None = None

        self._compile()

    @classmethod
    def from_symbolic(
        cls,
        model: Any,
        theta: dict[str, float] | None = None,
        *,
        measurement_noise_var: float | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        regularize: float = 1e-12,
    ) -> "ParametricSurrogate":
        """Build the surrogate straight from a :class:`~discopt.doe.SymbolicModel`.

        The equation is written once, as a string, and used both to fit and to
        predict: no separate :mod:`discopt.modeling` Experiment is needed. The
        model's inputs are the surrogate's input columns, in order.

        Parameters
        ----------
        model : SymbolicModel
        theta : dict, optional
            Starting parameter values for the first fit.
        measurement_noise_var : float, optional
            Defaults to the model's ``measurement_error ** 2``.
        parameter_bounds : dict name -> (lb, ub), optional
            Bounds for the least-squares refit (e.g. to keep a rate constant
            positive).
        regularize : float, default 1e-12
        """
        from discopt.doe.runs import symbolic_experiment

        noise = (
            float(model.measurement_error) ** 2
            if measurement_noise_var is None
            else float(measurement_noise_var)
        )
        return cls(
            symbolic_experiment(model, parameter_bounds=parameter_bounds),
            input_names=list(model.input_names),
            response_name=model.response_name,
            initial_guess=dict(theta or {}),
            measurement_noise_var=noise,
            regularize=regularize,
        )

    # ------------------------------------------------------------------
    # JIT compilation of f(design, theta) and its parameter Jacobian
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        from discopt.doe.fim import _require_jax

        jax, _ = _require_jax()

        compiled = _compile_predictor(
            self.experiment, self.input_names, [self.response_name], self.initial_guess
        )
        self.parameter_names_ = compiled.parameter_names
        self.parameter_bounds_ = compiled.parameter_bounds
        y_vec = compiled.y_vec

        def y_one(d_vec, theta_vec):
            return y_vec(d_vec, theta_vec)[0]

        self._y_one = jax.jit(y_one)
        self._y_batch = jax.jit(jax.vmap(y_one, in_axes=(0, None)))
        self._jac_batch = jax.jit(jax.vmap(jax.jacrev(y_one, argnums=1), in_axes=(0, None)))

        def residuals(theta_vec, D, Y):
            return jax.vmap(y_one, in_axes=(0, None))(D, theta_vec) - Y

        self._residuals = jax.jit(residuals)
        self._residuals_jac = jax.jit(jax.jacrev(residuals, argnums=0))

        self._n_params = len(self.parameter_names_)

    # ------------------------------------------------------------------
    # Surrogate protocol
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParametricSurrogate":
        """Refit the unknown parameters by nonlinear least squares.

        Computes the FIM and parameter covariance at the fitted point.
        """
        from discopt.doe.fim import _require_jax

        _, jnp = _require_jax()
        from scipy.optimize import least_squares

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[1] != len(self.input_names):
            raise ValueError(f"X has {X.shape[1]} columns but {len(self.input_names)} input names")

        n_obs = X.shape[0]
        n_par = self._n_params
        if n_obs < n_par:
            raise ValueError(
                f"cannot fit {n_par} parameter(s) from {n_obs} completed run(s): "
                f"need at least {n_par}. Complete more runs before designing the "
                "next batch."
            )

        theta0 = np.array(
            [float(self.initial_guess.get(n, 1.0)) for n in self.parameter_names_],
            dtype=float,
        )
        # Warm-start from the previous fit when available.
        if self.parameters_ is not None:
            theta0 = np.array([self.parameters_[n] for n in self.parameter_names_], dtype=float)

        # Clip the (possibly warm-started) guess strictly inside the bounds so
        # the bounded solver accepts it.
        lb = np.array([b[0] for b in self.parameter_bounds_], dtype=float)
        ub = np.array([b[1] for b in self.parameter_bounds_], dtype=float)
        theta0 = np.clip(theta0, lb, ub)

        D = jnp.asarray(X, dtype=jnp.float64)
        Y = jnp.asarray(y, dtype=jnp.float64)

        def f_resid(theta: np.ndarray) -> np.ndarray:
            return np.asarray(self._residuals(jnp.asarray(theta), D, Y), dtype=float)

        def f_jac(theta: np.ndarray) -> np.ndarray:
            return np.asarray(self._residuals_jac(jnp.asarray(theta), D, Y), dtype=float)

        # 'trf' (not 'lm') so the declared parameter bounds are respected and
        # the solver works when n_obs < n_params would otherwise be needed;
        # 'lm' supports neither.
        res = least_squares(f_resid, theta0, jac=f_jac, method="trf", bounds=(lb, ub))
        theta_hat = np.asarray(res.x, dtype=float)

        J = np.asarray(self._jac_batch(D, jnp.asarray(theta_hat)), dtype=float)
        sigma2 = max(self.measurement_noise_var, 1e-30)
        fim = (J.T @ J) / sigma2 + self.regularize * np.eye(self._n_params)
        try:
            cov = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fim)

        self.parameters_ = {n: float(theta_hat[i]) for i, n in enumerate(self.parameter_names_)}
        self.fim_ = fim
        self.covariance_ = cov
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior response mean and std at the candidate designs."""
        from discopt.doe.fim import _require_jax

        _, jnp = _require_jax()

        if self.parameters_ is None or self.covariance_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        theta = jnp.asarray([self.parameters_[n] for n in self.parameter_names_], dtype=jnp.float64)
        D = jnp.asarray(X, dtype=jnp.float64)
        mu = np.asarray(self._y_batch(D, theta), dtype=float).ravel()
        J = np.asarray(self._jac_batch(D, theta), dtype=float)
        # Diag of J Σ J^T per row.
        var_param = np.einsum("ki,ij,kj->k", J, self.covariance_, J)
        var_total = var_param + self.measurement_noise_var
        std = np.sqrt(np.clip(var_total, 0.0, None))
        return mu, std

    def predict_latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and the standard error of the mean response: parameter uncertainty only.

        ``predict`` adds the measurement noise, which describes a new
        observation; acquisition functions score the mean response instead
        (see :mod:`discopt.doe.acquisition`).
        """
        mu, std = self.predict(X)
        var_param = np.clip(std**2 - self.measurement_noise_var, 0.0, None)
        return mu, np.sqrt(var_param)


class MultiResponseParametricSurrogate:
    """One mechanistic model, several responses, one set of parameters.

    The multi-objective counterpart of :class:`ParametricSurrogate`. The
    responses share the parameters -- the yield and the by-product of one
    reaction network depend on the same rate constants -- so they are fitted
    *jointly*, every residual divided by its own measurement error. Fitting each
    response alone would throw that coupling away: a by-product measurement
    tells you about ``k1`` too.

    Parameters
    ----------
    experiment : Experiment
    input_names : sequence of str
        Design-input columns of the X matrix, in order.
    response_names : sequence of str
        Responses of ``experiment``; the columns of the Y matrix, in order.
    initial_guess : dict[str, float], optional
    measurement_noise_var : sequence of float
        ``σ²`` of each response, aligned with ``response_names``.
    regularize : float, default 1e-12
    """

    def __init__(
        self,
        experiment: Experiment,
        *,
        input_names: Sequence[str],
        response_names: Sequence[str],
        initial_guess: dict[str, float] | None = None,
        measurement_noise_var: Sequence[float],
        regularize: float = 1e-12,
    ) -> None:
        from discopt.doe.fim import _require_jax

        jax, jnp = _require_jax()

        self.experiment = experiment
        self.input_names = list(input_names)
        self.response_names = list(response_names)
        self.initial_guess = dict(initial_guess or {})
        self.measurement_noise_var = np.asarray(measurement_noise_var, dtype=float)
        if self.measurement_noise_var.shape != (len(self.response_names),):
            raise ValueError("measurement_noise_var needs one variance per response")
        if np.any(self.measurement_noise_var <= 0.0):
            raise ValueError("every response needs a positive measurement variance")
        self.regularize = float(regularize)
        self.mode = "parametric"

        self.parameters_: dict[str, float] | None = None
        self.covariance_: np.ndarray | None = None
        self.fim_: np.ndarray | None = None

        compiled = _compile_predictor(
            experiment, self.input_names, self.response_names, self.initial_guess
        )
        self.parameter_names_ = compiled.parameter_names
        self.parameter_bounds_ = compiled.parameter_bounds
        y_vec = compiled.y_vec
        self._y_batch = jax.jit(jax.vmap(y_vec, in_axes=(0, None)))
        self._jac_batch = jax.jit(jax.vmap(jax.jacrev(y_vec, argnums=1), in_axes=(0, None)))
        inv_sd = jnp.asarray(1.0 / np.sqrt(self.measurement_noise_var))

        def residuals(theta_vec, D, Y):
            return jnp.ravel((jax.vmap(y_vec, in_axes=(0, None))(D, theta_vec) - Y) * inv_sd)

        self._residuals = jax.jit(residuals)
        self._residuals_jac = jax.jit(jax.jacrev(residuals, argnums=0))

    def _theta(self) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.array([self.parameters_[n] for n in self.parameter_names_], dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiResponseParametricSurrogate":
        """Joint weighted least squares over every response; FIM and covariance at the fit."""
        from discopt.doe.fim import _require_jax

        _, jnp = _require_jax()
        from scipy.optimize import least_squares

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.shape != (X.shape[0], len(self.response_names)):
            raise ValueError(
                f"Y has shape {Y.shape}; expected ({X.shape[0]}, {len(self.response_names)})"
            )
        n_par = len(self.parameter_names_)
        if Y.size < n_par:
            raise ValueError(
                f"cannot fit {n_par} parameter(s) from {Y.size} measurement(s): "
                f"need at least {n_par}. Complete more runs before designing the next batch."
            )
        if self.parameters_ is not None:
            theta0 = self._theta()
        else:
            theta0 = np.array(
                [float(self.initial_guess.get(n, 1.0)) for n in self.parameter_names_]
            )
        lb = np.array([b[0] for b in self.parameter_bounds_], dtype=float)
        ub = np.array([b[1] for b in self.parameter_bounds_], dtype=float)
        theta0 = np.clip(theta0, lb, ub)

        D = jnp.asarray(X, dtype=jnp.float64)
        Yj = jnp.asarray(Y, dtype=jnp.float64)
        res = least_squares(
            lambda t: np.asarray(self._residuals(jnp.asarray(t), D, Yj), dtype=float),
            theta0,
            jac=lambda t: np.asarray(self._residuals_jac(jnp.asarray(t), D, Yj), dtype=float),
            method="trf",
            bounds=(lb, ub),
        )
        theta_hat = np.asarray(res.x, dtype=float)
        # The residual Jacobian is already noise-weighted, so J^T J is the FIM.
        Jr = np.asarray(self._residuals_jac(jnp.asarray(theta_hat), D, Yj), dtype=float)
        fim = Jr.T @ Jr + self.regularize * np.eye(n_par)
        try:
            cov = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fim)
        self.parameters_ = {n: float(theta_hat[i]) for i, n in enumerate(self.parameter_names_)}
        self.fim_ = fim
        self.covariance_ = cov
        return self

    def predict_all(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, std, jacobian)``: ``(n, m)``, ``(n, m)`` and ``(n, m, p)``.

        ``std`` is the standard error of each mean response (parameter
        uncertainty only, no measurement noise).
        """
        from discopt.doe.fim import _require_jax

        _, jnp = _require_jax()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        theta = jnp.asarray(self._theta())
        D = jnp.asarray(X, dtype=jnp.float64)
        mu = np.asarray(self._y_batch(D, theta), dtype=float)
        J = np.asarray(self._jac_batch(D, theta), dtype=float)
        var = np.einsum("kmi,ij,kmj->km", J, self.covariance_, J)
        return mu, np.sqrt(np.clip(var, 0.0, None)), J


class _ChebyshevScalarizedModel:
    """The ParEGO scalarization of a multi-response model, as one surrogate.

    The empirical ParEGO scalarizes the *data* and fits a fresh surrogate to it.
    With a mechanistic model there is no need: the scalarized objective is a
    known function of the model's predictions, so it is scalarized directly and
    its uncertainty follows by the same linearization the single-response
    surrogate uses, ``σ² = ∇θ g · Σθ · ∇θ gᵀ``. ``lo``/``span`` are the 0-1
    scaling of the observed (signed) objectives, held fixed for the round.
    """

    _is_discopt_surrogate = True
    mode = "parametric"

    def __init__(
        self,
        model: MultiResponseParametricSurrogate,
        weights: np.ndarray,
        signs: np.ndarray,
        lo: np.ndarray,
        span: np.ndarray,
        rho: float = 0.05,
    ) -> None:
        self.model = model
        self.weights = np.asarray(weights, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.lo = np.asarray(lo, dtype=float)
        self.span = np.asarray(span, dtype=float)
        self.rho = float(rho)

    def _mean_and_gradient(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, _, J = self.model.predict_all(X)
        cost = 1.0 - (self.signs * mu - self.lo) / self.span
        weighted = cost * self.weights
        g = -(np.max(weighted, axis=1) + self.rho * np.sum(weighted, axis=1))
        # dg/dcost_j = -(w_j [j = argmax] + rho w_j);  dcost_j/df_j = -sign_j / span_j
        active = np.zeros_like(weighted)
        active[np.arange(len(weighted)), np.argmax(weighted, axis=1)] = 1.0
        dg_df = (active + self.rho) * self.weights * self.signs / self.span
        return g, dg_df, J

    def predict_latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g, dg_df, J = self._mean_and_gradient(X)
        grad = np.einsum("km,kmp->kp", dg_df, J)
        var = np.einsum("ki,ij,kj->k", grad, self.model.covariance_, grad)
        return g, np.sqrt(np.clip(var, 0.0, None))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g, dg_df, J = self._mean_and_gradient(X)
        grad = np.einsum("km,kmp->kp", dg_df, J)
        var = np.einsum("ki,ij,kj->k", grad, self.model.covariance_, grad)
        var = var + (dg_df**2) @ self.model.measurement_noise_var
        return g, np.sqrt(np.clip(var, 0.0, None))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ChebyshevScalarizedModel":
        raise NotImplementedError("fit the underlying MultiResponseParametricSurrogate")


def model_based_optimize_round(
    workbook: str | Path | Workbook,
    *,
    experiment: Experiment | None = None,
    initial_guess: dict[str, float] | None = None,
    response: str | None = None,
    input_names: Sequence[str] | None = None,
    criterion: OptimizationCriterion | str = OptimizationCriterion.MAXIMIZE,
    acquisition: str | Callable = "expected_improvement",
    batch_size: int = 1,
    n_candidates: int = 2048,
    candidate_sampler: str = "sobol",
    bounds: Sequence[tuple[float, float]] | None = None,
    seed: int | None = None,
    acquisition_kwargs: dict[str, Any] | None = None,
    measurement_noise_override: float | Mapping[str, float] | None = None,
    objectives: Sequence[str] | Mapping[str, str] | None = None,
) -> ModelBasedRoundResult:
    """Propose the next batch of experiments using a parametric model.

    Parameters
    ----------
    workbook : path or Workbook
        Workbook containing the completed runs. The next batch is appended.
    experiment : Experiment, optional
        Parametric model. When omitted, the experiment is rebuilt from
        the workbook (for built-in templates and ``module_callable``).
    initial_guess : dict[str, float], optional
        Starting parameter values. Defaults to the
        ``param_initial_guess`` stored in the workbook.
    response : str, optional
        Response column name. Defaults to the workbook's stored value.
    input_names : sequence of str, optional
        Design-input column order. Defaults to the workbook's input_specs.
    criterion : OptimizationCriterion or str
        ``"maximize"`` or ``"minimize"``.
    acquisition : str or callable
        Looked up via :func:`~discopt.doe.acquisition.resolve_acquisition`.
    batch_size : int, default 1
        Number of new experiments to recommend.
    n_candidates : int, default 2048
        Size of the Sobol candidate pool.
    candidate_sampler : ``"sobol"`` or ``"uniform"``
        Sampler used to draw candidates inside the bounding box.
    bounds : sequence of (lo, hi), optional
        Per-input box. Defaults to the workbook's input_specs.
    seed : int, optional
        Reproducible candidate sampling.
    acquisition_kwargs : dict, optional
        Extra kwargs forwarded to the acquisition function.
    measurement_noise_override : float or mapping name -> float, optional
        Use this σ instead of the workbook's stored measurement_error. With
        ``objectives``, a mapping gives each response its own σ; responses it
        leaves out keep the experiment's ``measurement_error``.
    objectives : sequence of response names, or mapping name -> sense, optional
        Optimize several responses of the *same* model at once -- yield and
        selectivity, conversion and by-product -- with a model-based ParEGO
        (Knowles 2006). The responses share the parameters, so they are fitted
        jointly, each weighted by its own measurement error. Each recommended
        run draws a random weight vector, the model's predictions are
        scalarized with the augmented Chebyshev function, and the acquisition
        runs on that scalarized prediction and its linearized uncertainty.

        Every name must be a response of ``experiment`` and a numeric column of
        the completed runs. A sequence uses ``criterion`` for every objective; a
        mapping gives each its own ``"maximize"``/``"minimize"``. The result
        then carries ``pareto_front`` (measured), ``predicted_front`` (the
        model's whole trade-off curve) and ``scalarization_weights`` instead of
        a single incumbent.
    """
    crit = OptimizationCriterion(criterion) if isinstance(criterion, str) else criterion
    direction = crit.direction
    acq_fn = resolve_acquisition(acquisition)
    acq_kwargs = dict(acquisition_kwargs or {})

    wb = workbook if isinstance(workbook, Workbook) else Workbook.open(Path(workbook))

    if experiment is None:
        experiment, _ = wb.rebuild_experiment()
    if initial_guess is None:
        initial_guess = wb.param_initial_guess()

    specs = wb.input_specs()
    names = list(input_names) if input_names is not None else [s.name for s in specs]
    response_name = response or wb.response_name()
    sigma = (
        wb.measurement_error() if measurement_noise_override is None else measurement_noise_override
    )

    if bounds is None:
        bounds_arr = np.array([(s.lb, s.ub) for s in specs], dtype=float)
    else:
        bounds_arr = np.asarray(list(bounds), dtype=float)
    if bounds_arr.shape != (len(names), 2):
        raise ValueError(f"bounds shape {bounds_arr.shape} does not match {len(names)} input(s)")

    completed = wb.completed_runs()
    if not completed:
        raise ValueError(
            "no completed runs in workbook -- fill in at least one response "
            "before calling model_based_optimize_round"
        )

    X = np.array([[float(r[n]) for n in names] for r in completed], dtype=float)

    if objectives is not None:
        if isinstance(objectives, Mapping):
            senses = {str(k): str(v) for k, v in objectives.items()}
        else:
            senses = {str(n): crit.value for n in objectives}
        if len(senses) < 2:
            raise ValueError(
                "objectives needs at least two responses; for one use response= and criterion="
            )
        bad = {n: v for n, v in senses.items() if v not in ("maximize", "minimize")}
        if bad:
            raise ValueError(f"each objective's sense must be 'maximize' or 'minimize'; got {bad}")
        return _multi_objective_round(
            wb,
            experiment=experiment,
            initial_guess=initial_guess,
            names=names,
            X=X,
            completed=completed,
            senses=senses,
            noise_override=measurement_noise_override,
            acq_fn=acq_fn,
            acquisition=acquisition,
            acq_kwargs=acq_kwargs,
            batch_size=int(batch_size),
            candidates=_sample_candidates(
                bounds_arr, n_candidates, candidate_sampler, np.random.default_rng(seed)
            ),
            rng=np.random.default_rng(None if seed is None else seed + 1),
        )
    if isinstance(measurement_noise_override, Mapping):
        raise ValueError("a per-response measurement_noise_override needs objectives=")

    y = np.array([float(r[response_name]) for r in completed], dtype=float)

    s = ParametricSurrogate(
        experiment,
        input_names=names,
        response_name=response_name,
        initial_guess=initial_guess,
        measurement_noise_var=float(sigma) ** 2,
    )
    s.fit(X, y)

    # Snapshot the real-data fit for reporting. The batch loop below refits the
    # surrogate on fantasy (mean-imputed) phantom points to diversify picks;
    # reporting from that contaminated fit would shrink the standard errors and
    # inflate fim_log_det (extra rows, ~zero fantasy residuals), corrupting the
    # very diagnostics used to judge convergence.
    assert s.covariance_ is not None and s.fim_ is not None and s.parameters_ is not None
    real_parameters = dict(s.parameters_)
    real_parameter_names = list(s.parameter_names_)
    real_covariance = np.array(s.covariance_, copy=True)
    real_fim = np.array(s.fim_, copy=True)

    rng = np.random.default_rng(seed)
    candidates = _sample_candidates(bounds_arr, n_candidates, candidate_sampler, rng)

    incumbent_idx = int(np.argmax(direction * y))
    incumbent_y = float(y[incumbent_idx])
    incumbent_x = {n: float(X[incumbent_idx, i]) for i, n in enumerate(names)}

    chosen_idx: list[int] = []
    chosen_scores: list[float] = []
    X_fantasy = X.copy()
    y_fantasy = y.copy()
    incumbent_for_acq = incumbent_y

    for _ in range(int(batch_size)):
        scores = call_acquisition(
            acq_fn,
            s,
            candidates,
            direction=direction,
            y_best=incumbent_for_acq,
            acq_kwargs=acq_kwargs,
        )
        scores = np.asarray(scores, dtype=float).ravel()
        if chosen_idx:
            scores[chosen_idx] = -np.inf
        pick = int(np.argmax(scores))
        chosen_idx.append(pick)
        chosen_scores.append(float(scores[pick]))

        mu_pick, _ = s.predict(candidates[pick : pick + 1])
        X_fantasy = np.vstack([X_fantasy, candidates[pick : pick + 1]])
        y_fantasy = np.concatenate([y_fantasy, mu_pick])
        if direction * float(mu_pick[0]) > direction * incumbent_for_acq:
            incumbent_for_acq = float(mu_pick[0])
        s.fit(X_fantasy, y_fantasy)

    next_designs = [{n: float(candidates[i, j]) for j, n in enumerate(names)} for i in chosen_idx]
    batch_idx = wb.next_batch_index()
    new_run_ids = wb.append_runs(batch_idx, next_designs)

    # Report from the real-data snapshot, not the fantasy-contaminated fit.
    parameter_se = {
        n: float(np.sqrt(max(real_covariance[i, i], 0.0)))
        for i, n in enumerate(real_parameter_names)
    }
    sign, logdet = np.linalg.slogdet(real_fim)
    fim_log_det = float(logdet) if sign > 0 else float("-inf")

    acq_name = acquisition if isinstance(acquisition, str) else acq_fn.__name__
    log_lines = [
        f"fit {len(real_parameter_names)} parameter(s) on {len(completed)} completed run(s)",
        f"criterion={crit.value}, acquisition={acq_name}, batch_size={int(batch_size)}",
        f"recommended runs {new_run_ids}",
        f"fim_log_det={fim_log_det:.6g}",
    ]
    wb.log(
        "model_based_optimize",
        {
            "criterion": crit.value,
            "acquisition": acq_name,
            "batch_size": int(batch_size),
            "n_completed": len(completed),
            "fim_log_det": fim_log_det,
        },
    )
    wb.save()

    return ModelBasedRoundResult(
        next_designs=next_designs,
        new_run_ids=new_run_ids,
        incumbent_x=incumbent_x,
        incumbent_y=incumbent_y,
        acquisition_scores=chosen_scores,
        surrogate_mode="parametric",
        n_completed=len(completed),
        workbook_path=str(wb.path),
        parameters=real_parameters,
        parameter_se=parameter_se,
        fim_log_det=fim_log_det,
        log=log_lines,
    )


def _multi_objective_round(
    wb: Workbook,
    *,
    experiment: Experiment,
    initial_guess: dict[str, float],
    names: list[str],
    X: np.ndarray,
    completed: list[dict[str, Any]],
    senses: dict[str, str],
    noise_override: float | Mapping[str, float] | None,
    acq_fn: Callable,
    acquisition: str | Callable,
    acq_kwargs: dict[str, Any],
    batch_size: int,
    candidates: np.ndarray,
    rng: np.random.Generator,
) -> ModelBasedRoundResult:
    """The ``objectives=`` branch of :func:`model_based_optimize_round`."""
    response_names = list(senses)
    em = experiment.create_model(**initial_guess)
    unknown = [n for n in response_names if n not in em.responses]
    if unknown:
        raise ValueError(
            f"objective(s) {unknown} are not responses of the experiment; "
            f"its responses are {list(em.responses)}"
        )
    sigma = {n: float(em.measurement_error[n]) for n in response_names}
    if isinstance(noise_override, Mapping):
        sigma.update({str(k): float(v) for k, v in noise_override.items()})
    elif noise_override is not None:
        sigma = {n: float(noise_override) for n in response_names}

    signs = np.array(
        [
            1.0 if OptimizationCriterion(v) is OptimizationCriterion.MAXIMIZE else -1.0
            for v in senses.values()
        ]
    )
    Y_signed = _objective_matrix(completed, senses)
    Y = Y_signed * signs  # back to natural units, for the fit

    model = MultiResponseParametricSurrogate(
        experiment,
        input_names=names,
        response_names=response_names,
        initial_guess=initial_guess,
        measurement_noise_var=[sigma[n] ** 2 for n in response_names],
    )
    model.fit(X, Y)
    assert model.covariance_ is not None and model.fim_ is not None

    lo = Y_signed.min(axis=0)
    hi = Y_signed.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    chosen_idx: list[int] = []
    chosen_scores: list[float] = []
    weights_used: list[list[float]] = []
    for _ in range(batch_size):
        # A fresh weight vector per run: the batch spreads along the trade-off
        # because its weights do. No fantasy refit -- the model, unlike a GP fit
        # to scalarized data, does not change with the weights.
        w = rng.dirichlet(np.ones(len(response_names)))
        weights_used.append([float(v) for v in w])
        scalarized = _ChebyshevScalarizedModel(model, w, signs, lo, span)
        scores = call_acquisition(
            acq_fn,
            scalarized,
            candidates,
            direction=1.0,
            y_best=float(np.max(_chebyshev_scalarization(Y_signed, w))),
            acq_kwargs=acq_kwargs,
        )
        scores = np.asarray(scores, dtype=float).ravel()
        if chosen_idx:
            scores[chosen_idx] = -np.inf
        pick = int(np.argmax(scores))
        chosen_idx.append(pick)
        chosen_scores.append(float(scores[pick]))

    def _inputs(row: np.ndarray) -> dict[str, float]:
        return {n: float(row[j]) for j, n in enumerate(names)}

    pareto_front = [
        {**_inputs(X[i]), **{n: float(completed[i][n]) for n in response_names}}
        for i in np.flatnonzero(_pareto_mask(Y_signed))
    ]
    mu, se, _ = model.predict_all(candidates)
    front_idx = np.flatnonzero(_pareto_mask(mu * signs))
    front_idx = front_idx[np.argsort(mu[front_idx, 0])]
    predicted_front = [
        {
            **_inputs(candidates[i]),
            **{n: float(mu[i, j]) for j, n in enumerate(response_names)},
            **{f"{n}_se": float(se[i, j]) for j, n in enumerate(response_names)},
        }
        for i in front_idx
    ]

    next_designs = [_inputs(candidates[i]) for i in chosen_idx]
    new_run_ids = wb.append_runs(wb.next_batch_index(), next_designs)

    cov = model.covariance_
    parameter_se = {
        n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(model.parameter_names_)
    }
    sign, logdet = np.linalg.slogdet(model.fim_)
    fim_log_det = float(logdet) if sign > 0 else float("-inf")
    acq_name = acquisition if isinstance(acquisition, str) else acq_fn.__name__
    log_lines = [
        f"fit {len(model.parameter_names_)} parameter(s) jointly to {len(response_names)} "
        f"response(s) on {len(completed)} completed run(s)",
        f"objectives={senses}, acquisition={acq_name}, batch_size={batch_size}",
        f"recommended runs {new_run_ids}",
        f"{len(pareto_front)} measured run(s) on the Pareto front; "
        f"{len(predicted_front)} candidate(s) on the predicted front",
        f"fim_log_det={fim_log_det:.6g}",
    ]
    wb.log(
        "model_based_optimize",
        {
            "objectives": senses,
            "acquisition": acq_name,
            "batch_size": batch_size,
            "n_completed": len(completed),
            "fim_log_det": fim_log_det,
        },
    )
    wb.save()

    return ModelBasedRoundResult(
        next_designs=next_designs,
        new_run_ids=new_run_ids,
        incumbent_x=None,
        incumbent_y=None,
        acquisition_scores=chosen_scores,
        surrogate_mode="parametric",
        n_completed=len(completed),
        workbook_path=str(wb.path),
        parameters=dict(model.parameters_ or {}),
        parameter_se=parameter_se,
        fim_log_det=fim_log_det,
        log=log_lines,
        pareto_front=pareto_front,
        predicted_front=predicted_front,
        scalarization_weights=weights_used,
    )


__all__ = [
    "ModelBasedRoundResult",
    "MultiResponseParametricSurrogate",
    "ParametricSurrogate",
    "model_based_optimize_round",
]
