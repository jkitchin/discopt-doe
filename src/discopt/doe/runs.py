"""Campaigns: a per-run model plus the conditions each run was made at.

Most real experiments are a *campaign*: the same model run at several
conditions (temperatures, initial concentrations, sampling schedules), each run
giving one or more responses. The FIM, identifiability and estimability tools
all take a single :class:`~discopt.estimate.Experiment`, and the base
estimator merges data by response key, so without help a campaign has to be
hand-encoded as "one response per run", which is easy to get wrong: repeated
values under one key are silently fitted as replicates at one condition.

This module does the encoding once:

* :func:`campaign_experiment` stacks a per-run model over a list of run
  conditions into ONE :class:`CampaignExperiment` whose responses are indexed
  per run (``"y[0]"``, ``"A@2[3]"``, ...). Every tool that takes an
  Experiment then works on the whole campaign: :func:`~discopt.doe.compute_fim`,
  :func:`~discopt.doe.diagnose_identifiability`,
  :func:`~discopt.doe.estimability_rank`, :func:`~discopt.doe.d_optimal_subset`,
  :func:`~discopt.doe.profile_likelihood`, and estimation (the experiment
  carries its own :meth:`CampaignExperiment.estimate`, exact-Jacobian least
  squares).
* :func:`fit_campaign` fits a campaign directly from rows that hold both the
  conditions and the measured responses, and reports the uncertainty the way
  :func:`~discopt.doe.fit_least_squares` does.
* :func:`symbolic_experiment` turns a :class:`~discopt.doe.SymbolicModel`
  into an ordinary Experiment (design inputs and all), so a model written once
  as an equation can go anywhere an Experiment is needed.

The per-run model may be a :class:`~discopt.doe.SymbolicModel`, an
:class:`~discopt.doe.ODEExperiment`, or any pure explicit-response
:class:`~discopt.estimate.Experiment` (no constraints; every variable an
unknown parameter or a design input).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from discopt.estimate import EstimationResult, Experiment, ExperimentModel

__all__ = [
    "CampaignExperiment",
    "campaign_experiment",
    "fit_campaign",
    "symbolic_experiment",
]

# Bounds for variables that the model does not bound itself. discopt variables
# need finite bounds; these only matter to box-constrained solvers.
_BIG = 1e20


def _jax():
    from discopt.doe.fim import _require_jax

    return _require_jax()


# --------------------------------------------------------------------------
# A per-run model as one differentiable function h(theta, u)
# --------------------------------------------------------------------------


@dataclass
class _PerRun:
    """A per-run model reduced to ``h(theta_vec, u_vec) -> responses``."""

    parameter_names: list[str]
    parameter_bounds: dict[str, tuple[float, float]]
    nominal: dict[str, float]
    input_names: list[str]
    input_bounds: dict[str, tuple[float, float]]
    response_names: list[str]
    sigma: dict[str, float]
    h: Callable[[Any, Any], Any]


def _per_run(
    model: Any,
    *,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    measurement_error: float | Mapping[str, float] | None = None,
) -> _PerRun:
    from discopt.doe.dynamic import ODEExperiment
    from discopt.doe.symbolic import SymbolicModel

    pb = dict(parameter_bounds or {})
    if isinstance(model, SymbolicModel):
        import sympy as sp

        _, jnp = _jax()
        p_syms = [sp.Symbol(n, real=True) for n in model.parameter_names]
        x_syms = [sp.Symbol(n, real=True) for n in model.input_names]
        f = sp.lambdify((p_syms, x_syms), model.expression, modules="jax")

        def h_sym(theta, u, _f=f):
            return jnp.reshape(jnp.asarray(_f(theta, u), dtype=float), (1,))

        out = _PerRun(
            parameter_names=list(model.parameter_names),
            parameter_bounds={n: tuple(pb.get(n, (-_BIG, _BIG))) for n in model.parameter_names},
            nominal={},
            input_names=list(model.input_names),
            input_bounds={n: (-_BIG, _BIG) for n in model.input_names},
            response_names=[model.response_name],
            sigma={model.response_name: float(model.measurement_error)},
            h=h_sym,
        )
    elif isinstance(model, ODEExperiment):

        def h_ode(theta, u, _m=model):
            k = len(_m.parameter_names)
            return _m.response_function(
                *[theta[i] for i in range(k)], *[u[j] for j in range(u.size)]
            )

        out = _PerRun(
            parameter_names=list(model.parameter_names),
            parameter_bounds={
                n: tuple(pb.get(n, (float(lb), float(ub))))
                for n, (_, lb, ub) in model.parameter_specs.items()
            },
            nominal={n: float(spec[0]) for n, spec in model.parameter_specs.items()},
            input_names=list(model.design_names),
            input_bounds={n: tuple(map(float, b)) for n, b in model.design_bounds.items()},
            response_names=list(model.response_names),
            sigma={rn: float(model.sigma[rn.split("@")[0]]) for rn in model.response_names},
            h=h_ode,
        )
    elif isinstance(model, Experiment):
        out = _per_run_generic(model, pb)
    else:
        raise TypeError(
            "model must be a SymbolicModel, an ODEExperiment or a discopt Experiment, "
            f"got {type(model).__name__}"
        )

    if measurement_error is not None:
        if isinstance(measurement_error, Mapping):
            unknown = [k for k in measurement_error if k not in out.response_names]
            if unknown:
                raise ValueError(f"measurement_error keys {unknown} are not responses")
            out.sigma.update({k: float(v) for k, v in measurement_error.items()})
        else:
            out.sigma = {rn: float(measurement_error) for rn in out.response_names}
    bad = [rn for rn, s in out.sigma.items() if not (s > 0 and math.isfinite(s))]
    if bad:
        raise ValueError(f"measurement error must be positive and finite for {bad}")
    return out


def _per_run_generic(experiment: Experiment, pb: dict) -> _PerRun:
    """Reduce a pure explicit-response Experiment to ``h(theta, u)``."""
    from discopt.doe.fim import _compile_response, _design_source_map
    from discopt.parametric import flatten_params, variable_slices

    _, jnp = _jax()
    em = experiment.create_model()
    if _design_source_map(em) is None:
        raise ValueError(
            "campaigns need an explicit-response Experiment (no constraints; every "
            "variable an unknown parameter or a design input); this one has implicit "
            "states that need a solve"
        )
    slices = variable_slices(em.model)
    n_x = max((sl.stop for sl in slices.values()), default=0)
    for group, kind in ((em.unknown_parameters, "parameter"), (em.design_inputs, "design input")):
        for name, var in group.items():
            sl = slices[var.name]
            if sl.stop - sl.start != 1:
                raise ValueError(f"{kind} {name!r} is not a scalar; campaigns need scalars")
    p_names = list(em.unknown_parameters)
    u_names = list(em.design_inputs)
    p_idx = jnp.asarray([slices[em.unknown_parameters[n].name].start for n in p_names])
    u_idx = jnp.asarray([slices[em.design_inputs[n].name].start for n in u_names], dtype=int)
    p_flat = flatten_params(em.model)
    fns = [_compile_response(em.responses[rn], em.model) for rn in em.response_names]

    def h_gen(theta, u):
        x = jnp.zeros(n_x, dtype=float).at[p_idx].set(theta)
        if u_names:
            x = x.at[u_idx].set(u)
        return jnp.stack([jnp.reshape(jnp.asarray(f(x, p_flat), dtype=float), ()) for f in fns])

    def _bounds(var) -> tuple[float, float]:
        lb = float(np.asarray(getattr(var, "lb", -_BIG)).ravel()[0])
        ub = float(np.asarray(getattr(var, "ub", _BIG)).ravel()[0])
        return (max(lb, -_BIG), min(ub, _BIG))

    return _PerRun(
        parameter_names=p_names,
        parameter_bounds={n: tuple(pb.get(n, _bounds(em.unknown_parameters[n]))) for n in p_names},
        nominal={},
        input_names=u_names,
        input_bounds={n: _bounds(em.design_inputs[n]) for n in u_names},
        response_names=list(em.response_names),
        sigma={rn: float(em.measurement_error[rn]) for rn in em.response_names},
        h=h_gen,
    )


# --------------------------------------------------------------------------
# A SymbolicModel as an Experiment
# --------------------------------------------------------------------------


@dataclass
class _SymbolicExperiment(Experiment):
    """An Experiment built from a SymbolicModel (see :func:`symbolic_experiment`)."""

    per_run: _PerRun
    input_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    name: str = "symbolic"

    def create_model(self, **kwargs: Any) -> ExperimentModel:
        import discopt.modeling as dm

        _, jnp = _jax()
        pr = self.per_run
        m = dm.Model(self.name)
        params = {n: m.continuous(n, lb=lb, ub=ub) for n, (lb, ub) in pr.parameter_bounds.items()}
        ib = {**pr.input_bounds, **self.input_bounds}
        inputs = {n: m.continuous(n, lb=ib[n][0], ub=ib[n][1]) for n in pr.input_names}
        k = len(params)

        def fn(*args, _h=pr.h, _k=k):
            return _h(jnp.stack(args[:_k]), jnp.stack(args[_k:]))

        node = dm.custom(fn, name=self.name)(*params.values(), *inputs.values())
        return ExperimentModel(
            model=m,
            unknown_parameters=params,
            design_inputs=inputs,
            responses={rn: node[i] for i, rn in enumerate(pr.response_names)},
            measurement_error=dict(pr.sigma),
        )


def symbolic_experiment(
    model: Any,
    *,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    input_bounds: Mapping[str, tuple[float, float]] | None = None,
    name: str = "symbolic",
) -> Experiment:
    """Turn a :class:`~discopt.doe.SymbolicModel` into a discopt Experiment.

    The equation is differentiated by JAX (via sympy's JAX printer), and the
    model's inputs become the Experiment's design inputs, so the result works
    with :func:`~discopt.doe.compute_fim`, :func:`~discopt.doe.optimal_experiment`,
    :class:`~discopt.doe.ParametricSurrogate` and the rest without writing the
    model a second time in :mod:`discopt.modeling`.

    Parameters
    ----------
    model : SymbolicModel
    parameter_bounds, input_bounds : mapping name -> (lb, ub), optional
        Variable bounds. Unbounded by default, which is fine for FIMs; give
        input bounds when a design search will read them from the model.
    name : str, default "symbolic"

    Examples
    --------
    >>> from discopt.doe import SymbolicModel, compute_fim
    >>> m = SymbolicModel("a * exp(-b * x)", ("a", "b"), ("x",))
    >>> fim = compute_fim(symbolic_experiment(m), {"a": 1.0, "b": 0.5}, {"x": 2.0})
    """
    return _SymbolicExperiment(
        per_run=_per_run(model, parameter_bounds=parameter_bounds),
        input_bounds=dict(input_bounds or {}),
        name=name,
    )


# --------------------------------------------------------------------------
# Campaigns
# --------------------------------------------------------------------------


def _condition_matrix(per_run: _PerRun, runs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for i, run in enumerate(runs):
        missing = [n for n in per_run.input_names if n not in run]
        if missing:
            raise ValueError(f"run {i} is missing condition(s) {missing}")
        rows.append([float(run[n]) for n in per_run.input_names])
    return np.asarray(rows, dtype=float).reshape(len(runs), len(per_run.input_names))


@dataclass
class CampaignExperiment(Experiment):
    """One Experiment for a whole campaign of runs (see :func:`campaign_experiment`).

    Attributes
    ----------
    runs : list of dict
        The run conditions, in order.
    labels : list of str
        One label per run; response names carry it, as ``"<response>[<label>]"``.
    response_names : list of str
        Every response of every run, run by run.
    parameter_names : list of str
    """

    per_run: _PerRun
    runs: list[dict[str, Any]]
    labels: list[str]
    name: str = "campaign"
    response_names: list[str] = field(init=False)
    _conditions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.runs):
            raise ValueError(f"{len(self.labels)} labels for {len(self.runs)} runs")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("run labels must be unique")
        self._conditions = _condition_matrix(self.per_run, self.runs)
        self.response_names = [
            f"{rn}[{lab}]" for lab in self.labels for rn in self.per_run.response_names
        ]

    @property
    def parameter_names(self) -> list[str]:
        return list(self.per_run.parameter_names)

    # The whole campaign as one function of theta -------------------------

    def _campaign_fn(self):
        jax, jnp = _jax()
        h = self.per_run.h
        U = jnp.asarray(self._conditions, dtype=float)
        n_in = len(self.per_run.input_names)

        def F(theta):
            if n_in == 0:
                return jnp.tile(h(theta, jnp.zeros(0)), len(self.runs))
            return jax.vmap(lambda u: h(theta, u))(U).reshape(-1)

        return F

    def _sigma_vector(self) -> np.ndarray:
        per = [self.per_run.sigma[rn] for rn in self.per_run.response_names]
        return np.tile(np.asarray(per, dtype=float), len(self.runs))

    def create_model(self, **kwargs: Any) -> ExperimentModel:
        import discopt.modeling as dm

        _, jnp = _jax()
        m = dm.Model(self.name)
        params = {
            n: m.continuous(n, lb=lb, ub=ub)
            for n, (lb, ub) in self.per_run.parameter_bounds.items()
        }
        F = self._campaign_fn()

        def fn(*theta, _F=F):
            return _F(jnp.stack(theta))

        node = dm.custom(fn, name=self.name)(*params.values())
        sig = self._sigma_vector()
        return ExperimentModel(
            model=m,
            unknown_parameters=params,
            design_inputs={},
            responses={rn: node[i] for i, rn in enumerate(self.response_names)},
            measurement_error={rn: float(s) for rn, s in zip(self.response_names, sig)},
        )

    def predict(self, theta: Mapping[str, float]) -> dict[str, float]:
        """Every response of every run at parameters ``theta``."""
        _, jnp = _jax()
        th = jnp.asarray([float(theta[n]) for n in self.parameter_names], dtype=float)
        y = np.asarray(self._campaign_fn()(th))
        return dict(zip(self.response_names, map(float, y)))

    def data_from_runs(
        self, responses: Sequence[Mapping[str, Any]]
    ) -> dict[str, float | np.ndarray]:
        """Map per-run measured values (one mapping per run, in run order) to response names.

        Each mapping holds the per-run response names (``"y"``, ``"A@2"``);
        values may be a number or an array of replicates. Missing or NaN
        responses are skipped.
        """
        if len(responses) != len(self.runs):
            raise ValueError(f"{len(responses)} response rows for {len(self.runs)} runs")
        data: dict[str, float | np.ndarray] = {}
        for lab, row in zip(self.labels, responses):
            for rn in self.per_run.response_names:
                if rn not in row or row[rn] is None:
                    continue
                v = np.atleast_1d(np.asarray(row[rn], dtype=float))
                v = v[np.isfinite(v)]
                if v.size:
                    data[f"{rn}[{lab}]"] = v if v.size > 1 else float(v[0])
        return data

    # Estimation -------------------------------------------------------------

    def estimate(
        self,
        data: Mapping[str, float | Sequence[float]],
        *,
        initial_guess: Mapping[str, float] | None = None,
        fixed_parameters: Mapping[str, float] | None = None,
        solver_options: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> EstimationResult:
        """Weighted least squares on the campaign; same contract as the base estimator.

        ``objective`` is the deviance ``Σ((y - ŷ)/σ)²`` and ``fim``/``covariance``
        use the declared measurement errors, as
        :func:`discopt.estimate.estimate_parameters` does. The discopt-doe loops
        (:func:`~discopt.doe.profile_likelihood`, :func:`~discopt.doe.sequential_doe`)
        call this automatically.
        """
        fit = _weighted_fit(
            self,
            data,
            initial_guess=initial_guess,
            fixed_parameters=fixed_parameters,
            solver_options=solver_options,
        )
        cov = _safe_inverse(fit["fim"])
        return EstimationResult(
            parameters=fit["theta"],
            covariance=cov,
            fim=fit["fim"],
            objective=fit["deviance"],
            solve_result=_FitStatus(success=fit["success"], message=fit["message"]),
            parameter_names=self.parameter_names,
            n_observations=fit["n_obs"],
        )


@dataclass
class _FitStatus:
    success: bool
    message: str

    @property
    def status(self) -> str:
        return "optimal" if self.success else "failed"


def campaign_experiment(
    model: Any,
    runs: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[Any] | None = None,
    measurement_error: float | Mapping[str, float] | None = None,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    name: str = "campaign",
) -> CampaignExperiment:
    """Stack a per-run model over the conditions of every run in a campaign.

    Parameters
    ----------
    model : SymbolicModel, ODEExperiment, or Experiment
        The model of ONE run. Its design inputs (a SymbolicModel's inputs, an
        ODEExperiment's design inputs, an Experiment's ``design_inputs``) are
        the run conditions.
    runs : sequence of mapping
        One mapping per run giving every design input; extra keys (measured
        values, bookkeeping) are ignored.
    labels : sequence, optional
        Run labels used in response names (default ``0, 1, 2, ...``, or each
        run's ``run_id`` when every run has one).
    measurement_error : float or mapping, optional
        Override the model's measurement error: one σ for every response, or
        per per-run response name.
    parameter_bounds : mapping name -> (lb, ub), optional
        Parameter bounds (a SymbolicModel has none of its own).
    name : str, default "campaign"

    Returns
    -------
    CampaignExperiment
        An Experiment with the model's unknown parameters, no design inputs,
        and one response per (run, per-run response), named
        ``"<response>[<label>]"``.

    Examples
    --------
    >>> from discopt.doe import SymbolicModel, compute_fim
    >>> m = SymbolicModel("a * exp(-b * x)", ("a", "b"), ("x",), measurement_error=0.1)
    >>> camp = campaign_experiment(m, [{"x": 0.5}, {"x": 2.0}, {"x": 4.0}])
    >>> fim = compute_fim(camp, {"a": 1.0, "b": 0.5}).fim   # the whole campaign's FIM
    """
    runs = [dict(r) for r in runs]
    if not runs:
        raise ValueError("a campaign needs at least one run")
    if labels is None:
        if all("run_id" in r for r in runs):
            labels = [r["run_id"] for r in runs]
        else:
            labels = list(range(len(runs)))
    per_run = _per_run(
        model, parameter_bounds=parameter_bounds, measurement_error=measurement_error
    )
    return CampaignExperiment(
        per_run=per_run, runs=runs, labels=[str(v) for v in labels], name=name
    )


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _safe_inverse(m: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m)


def _weighted_fit(
    camp: CampaignExperiment,
    data: Mapping[str, Any],
    *,
    initial_guess: Mapping[str, float] | None,
    fixed_parameters: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    jax, jnp = _jax()
    unknown = [k for k in data if k not in camp.response_names]
    if unknown:
        raise ValueError(f"data keys {unknown[:5]} are not campaign responses")
    names = camp.parameter_names
    fixed = {k: float(v) for k, v in (fixed_parameters or {}).items()}
    for k in fixed:
        if k not in names:
            raise KeyError(f"{k!r} is not an unknown parameter ({names})")
    free = [n for n in names if n not in fixed]
    pb = dict(camp.per_run.parameter_bounds)
    pb.update({k: tuple(v) for k, v in (bounds or {}).items()})
    guess = {n: 1.0 for n in names}
    guess.update(camp.per_run.nominal)
    guess.update({k: float(v) for k, v in (initial_guess or {}).items() if k in guess})
    guess.update(fixed)

    sig = camp._sigma_vector()
    idx, obs, wts = [], [], []
    for i, rn in enumerate(camp.response_names):
        if rn not in data:
            continue
        for v in np.atleast_1d(np.asarray(data[rn], dtype=float)).ravel():
            if np.isfinite(v):
                idx.append(i)
                obs.append(v)
                wts.append(1.0 / sig[i])
    if not idx:
        raise ValueError("no finite data matched any campaign response")
    F = camp._campaign_fn()
    idx_a, obs_a, w_a = jnp.asarray(idx), jnp.asarray(obs), jnp.asarray(wts)
    free_pos = [names.index(n) for n in free]

    def theta_of(z):
        th = jnp.asarray([guess[n] for n in names], dtype=float)
        if free_pos:
            th = th.at[jnp.asarray(free_pos)].set(z)
        return th

    def resid(z):
        return (F(theta_of(z))[idx_a] - obs_a) * w_a

    res_j = jax.jit(resid)
    jac_j = jax.jit(jax.jacfwd(resid))
    lo = np.array([pb[n][0] for n in free], dtype=float)
    hi = np.array([pb[n][1] for n in free], dtype=float)
    z0 = np.clip(np.array([guess[n] for n in free], dtype=float), lo, hi)
    if free:
        sol = least_squares(
            lambda z: np.asarray(res_j(jnp.asarray(z))),
            z0,
            jac=lambda z: np.asarray(jac_j(jnp.asarray(z))),
            bounds=(lo, hi),
            x_scale="jac",
            **dict(solver_options or {}),
        )
        z_hat, success, message = np.asarray(sol.x), bool(sol.success), str(sol.message)
    else:
        z_hat, success, message = np.zeros(0), True, "all parameters fixed"
    theta_hat = {n: float(v) for n, v in zip(names, np.asarray(theta_of(jnp.asarray(z_hat))))}
    r = np.asarray(res_j(jnp.asarray(z_hat)))

    # Weighted Jacobian over ALL parameters (as the base estimator reports).
    def wF(th):
        return F(th)[idx_a] * w_a

    J = np.asarray(jax.jacfwd(wF)(jnp.asarray([theta_hat[n] for n in names])))
    return {
        "theta": theta_hat,
        "free": free,
        "J": J,
        "fim": J.T @ J,
        "deviance": float(r @ r),
        "residuals": r / np.asarray(wts),
        "n_obs": len(idx),
        "success": success,
        "message": message,
    }


def fit_campaign(
    model: Any,
    runs: Sequence[Mapping[str, Any]],
    initial: Mapping[str, float] | None = None,
    *,
    responses: Sequence[str] | None = None,
    measurement_error: float | Mapping[str, float] | None = None,
    sigma: str | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    level: float = 0.95,
) -> dict[str, Any]:
    """Fit a model to a campaign of runs, each row holding conditions and responses.

    Multi-experiment weighted least squares with the exact JAX Jacobian: every
    run is evaluated at its own conditions, every measured response is weighted
    by its measurement error, and the parameters are shared.

    Parameters
    ----------
    model : SymbolicModel, ODEExperiment, or Experiment
        The model of one run (see :func:`campaign_experiment`).
    runs : sequence of mapping
        One row per run: every design input, plus the measured values under the
        per-run response names (``"y"`` for a SymbolicModel, ``"A@2"`` for an
        ODE response). Values may be arrays of replicates; missing or NaN
        responses are skipped, so runs may measure different things.
    initial : mapping, optional
        Starting values (defaults: an ODEExperiment's nominal values, else 1.0).
    responses : sequence of str, optional
        Restrict the fit to these per-run responses.
    measurement_error : float or mapping, optional
        Override the model's measurement error (relative weights and, with
        ``sigma="declared"``, the absolute scale).
    sigma : {None, "residual", "declared"}, default None
        How the parameter uncertainty is scaled. ``"residual"`` multiplies the
        covariance by the reduced chi-square ``Σ(r/σ)²/(n - p)``, trusting the
        residuals for the absolute noise level (the relative weights still
        come from the measurement errors). ``"declared"`` trusts the declared
        measurement errors. ``None`` means ``"residual"`` when there are
        residual degrees of freedom, else ``"declared"``.
    bounds : mapping name -> (lb, ub), optional
    fixed_parameters : mapping, optional
        Hold these parameters at the given values.
    level : float, default 0.95
        Confidence level of ``ci_lower``/``ci_upper``.

    Returns
    -------
    dict
        ``parameter_names``, ``estimates``, ``std_errors``, ``ci_lower``,
        ``ci_upper``, ``covariance``, ``fim`` (``inv(fim) == covariance`` over
        the free parameters), ``sigma_scale`` and ``sigma_source``,
        ``residuals`` (in response units), ``deviance``, ``n_observations``,
        ``degrees_of_freedom``, ``level``, ``success``, ``message``, and the
        ``campaign`` (:class:`CampaignExperiment`).
    """
    from scipy import stats

    from discopt.doe.symbolic import check_jacobian_rank

    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if sigma not in (None, "residual", "declared"):
        raise ValueError(f"sigma must be None, 'residual' or 'declared', got {sigma!r}")
    camp = campaign_experiment(
        model, runs, measurement_error=measurement_error, parameter_bounds=bounds
    )
    keep = set(responses) if responses is not None else None
    rows = [
        {
            k: v
            for k, v in r.items()
            if keep is None or k in keep or k not in camp.per_run.response_names
        }
        for r in camp.runs
    ]
    data = camp.data_from_runs(rows)
    fit = _weighted_fit(
        camp,
        data,
        initial_guess=initial,
        fixed_parameters=fixed_parameters,
        bounds=bounds,
    )
    names = camp.parameter_names
    free = fit["free"]
    pos = [names.index(n) for n in free]
    Jf = fit["J"][:, pos]
    fim_free = Jf.T @ Jf
    dof = fit["n_obs"] - len(free)
    source = sigma or ("residual" if dof > 0 else "declared")
    if source == "residual" and dof <= 0:
        raise ValueError("sigma='residual' needs more observations than free parameters")
    scale2 = fit["deviance"] / dof if source == "residual" else 1.0
    check_jacobian_rank(Jf, free, stacklevel=3)
    cov = _safe_inverse(fim_free) * scale2
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    crit = (
        float(stats.t.ppf(0.5 + level / 2, dof))
        if dof > 0
        else float(stats.norm.ppf(0.5 + level / 2))
    )
    est = np.array([fit["theta"][n] for n in free])
    return {
        "parameter_names": free,
        "estimates": {n: fit["theta"][n] for n in names},
        "std_errors": {n: float(s) for n, s in zip(free, se)},
        "ci_lower": {n: float(e - crit * s) for n, e, s in zip(free, est, se)},
        "ci_upper": {n: float(e + crit * s) for n, e, s in zip(free, est, se)},
        "covariance": cov,
        "fim": fim_free / scale2,
        "sigma_scale": float(math.sqrt(scale2)),
        "sigma_source": source,
        "residuals": fit["residuals"],
        "deviance": fit["deviance"],
        "n_observations": fit["n_obs"],
        "degrees_of_freedom": dof,
        "level": float(level),
        "success": fit["success"],
        "message": fit["message"],
        "campaign": camp,
    }
