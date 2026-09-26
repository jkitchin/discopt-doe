"""Fisher Information Matrix computation via JAX autodiff.

Computes the FIM for model-based design of experiments using exact
Jacobian computation (no finite differences). The FIM quantifies how
much information an experiment provides about unknown parameters.

Mathematical background
-----------------------
For a model with responses ``y = f(θ, d)`` and measurement error
covariance ``Σ``, the Fisher Information Matrix is:

    FIM = J^T Σ^{-1} J

where ``J`` is the sensitivity Jacobian ``∂y/∂θ`` evaluated at the
nominal parameter values and design conditions.

The FIM is used to:
- Assess parameter identifiability (rank of FIM)
- Predict parameter estimation precision (Cov(θ) ≈ FIM^{-1})
- Optimize experimental design (maximize information content)
"""

from __future__ import annotations

import dataclasses
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from discopt.doe.linear_design import trace_inverse
from discopt.estimate import Experiment, ExperimentModel


def _require_jax():
    """Return ``(jax, jax.numpy)``, or raise with an actionable message.

    Mirrors ``_require_sklearn`` / ``_require_openpyxl`` elsewhere in the
    package. jax is a hard dependency of a normal install, so this fires only
    in environments where no jax wheel exists — Pyodide/WASM most notably. The
    classical designs, ANOVA, and OLS fitting all work there; FIM-based optimal
    design is what does not, and this says so rather than surfacing a bare
    ModuleNotFoundError from deep inside a Jacobian call.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:  # pragma: no cover - exercised only without jax
        raise ImportError(
            "FIM-based optimal design requires jax, which has no WebAssembly "
            "build. Install it with: pip install 'discopt-doe' (jax is a base "
            "dependency). In a browser/Pyodide environment use the classical "
            "designs (latin-hypercube, central-composite, box-behnken, "
            "factorial-2level, latin-square) and discopt.doe.linear_design "
            "instead, which are pure numpy/scipy."
        ) from e
    return jax, jnp


# A parameter axis whose squared projection onto the null-space basis
# exceeds this value is treated as lying *in* the null space — VIF is
# reported as infinite and the FIM-based standard error / correlations
# are masked to NaN. 1% captures "effectively unidentifiable" while
# avoiding spurious flagging from round-off in the right singular
# vectors.
_NULL_PROJECTION_THRESHOLD = 0.01


@dataclass
class FIMResult:
    """Result of Fisher Information Matrix computation.

    Attributes
    ----------
    fim : numpy.ndarray
        Fisher Information Matrix, shape ``(n_params, n_params)``.
    jacobian : numpy.ndarray
        Sensitivity Jacobian ``∂y/∂θ``, shape ``(n_responses, n_params)``.
    parameter_names : list[str]
        Ordered parameter names matching FIM rows/columns.
    response_names : list[str]
        Ordered response names matching Jacobian rows.
    """

    fim: np.ndarray
    jacobian: np.ndarray
    parameter_names: list[str]
    response_names: list[str]

    @property
    def d_optimal(self) -> float:
        """D-optimality criterion: ``log(det(FIM))``.

        Uses ``slogdet`` rather than ``log(det(...))``: for badly-scaled FIMs
        (parameters spanning many decades) ``det`` overflows to inf or
        underflows to 0, whereas ``slogdet`` computes the log-determinant
        directly and stably.
        """
        sign, logdet = np.linalg.slogdet(self.fim)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        return float(logdet)

    @property
    def a_optimal(self) -> float:
        """A-optimality criterion: ``trace(FIM^{-1})`` (``inf`` if not positive definite)."""
        return trace_inverse(self.fim)

    @property
    def e_optimal(self) -> float:
        """E-optimality criterion: minimum eigenvalue of FIM."""
        return float(np.min(np.linalg.eigvalsh(self.fim)))

    @property
    def me_optimal(self) -> float:
        """Modified E-optimality: condition number of FIM."""
        return float(np.linalg.cond(self.fim))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics as a dictionary."""
        return {
            "log_det_fim": self.d_optimal,
            "trace_fim_inv": self.a_optimal,
            "min_eigenvalue": self.e_optimal,
            "condition_number": self.me_optimal,
        }


def _compile_response(expr, model):
    """Compile a response expression into ``f(x_flat, p_flat)``.

    ``discopt.parametric.compile_expression`` threads model Parameters through
    ``p_flat`` but has no rule for an opaque :func:`discopt.modeling.custom`
    node (a ``CustomCall``), which is how an ODE integrator enters a model (see
    :mod:`discopt.doe.dynamic`). For those expressions fall back to the solver's
    own DAG compiler, which traces the callable through JAX (so every autodiff
    mode works) and evaluates shared nodes once; its Parameter values are
    snapshotted at compile time, which is equivalent here because the FIM
    differentiates with respect to variables, not Parameters.
    """
    from discopt.parametric import compile_expression

    try:
        return compile_expression(expr, model)
    except TypeError as exc:
        if "CustomCall" not in str(exc):
            raise
    from discopt._relax.dag_compiler import compile_expression as _dag_compile

    fn = _dag_compile(expr, model)

    def wrapped(x_flat, p_flat=None, _fn=fn):
        return _fn(x_flat)

    return wrapped


def _measurement_sigma(em: ExperimentModel) -> np.ndarray:
    """Validated per-response measurement std-devs (sigma > 0).

    A zero (or negative) measurement error makes ``1/sigma**2`` infinite, which
    silently poisons every FIM-based criterion. Fail loudly instead.
    """
    sigma = np.array([em.measurement_error[name] for name in em.response_names], dtype=np.float64)
    if np.any(sigma <= 0.0):
        bad = [n for n in em.response_names if float(em.measurement_error[n]) <= 0.0]
        raise ValueError(
            f"measurement_error must be positive; response(s) {bad} have <= 0 "
            "(a zero measurement error gives an infinite FIM)."
        )
    return sigma


def _check_nominal_in_bounds(name: str, var: Any, value: Any) -> None:
    """Refuse a nominal parameter value outside its Variable's bounds.

    The FIM is a local quantity: it must be evaluated *at* the nominal values.
    Clipping them into the bounds (or letting the bounded solve do so) would
    silently return the FIM of a different parameter vector.
    """
    arr = np.asarray(value, dtype=np.float64).ravel()
    lb = np.broadcast_to(np.asarray(var.lb, dtype=np.float64).ravel(), arr.shape)
    ub = np.broadcast_to(np.asarray(var.ub, dtype=np.float64).ravel(), arr.shape)
    tol = 1e-9 * np.maximum(1.0, np.abs(arr))
    if np.any(arr < lb - tol) or np.any(arr > ub + tol):
        raise ValueError(
            f"nominal value {arr.tolist() if arr.size > 1 else float(arr[0])} of parameter "
            f"{name!r} lies outside its variable bounds [{var.lb}, {var.ub}]; the FIM would "
            "be evaluated at a different point. Widen the bounds in create_model."
        )


def _design_source_map(em: ExperimentModel) -> dict | None:
    """Classify every model variable as a parameter or a design input.

    Returns ``{id(var): ("param"|"design", name, var)}`` when the model is a
    *pure explicit response model* — it has no constraints and every variable
    is either an unknown parameter or a design input, so the solution point
    ``x*`` is fully determined by the nominal parameters and the fixed design.
    Returns ``None`` when the model has constraints or any other variable (an
    implicit state that genuinely requires a solve).
    """
    from discopt.parametric import variable_slices

    if getattr(em.model, "_constraints", None):
        return None
    src: dict[str, tuple[str, str, Any]] = {}
    for name, var in em.unknown_parameters.items():
        src[var.name] = ("param", name, var)
    for name, var in em.design_inputs.items():
        src[var.name] = ("design", name, var)
    for vname in variable_slices(em.model):
        if vname not in src:
            return None
    return src


def _direct_var_values(
    src_entry: tuple[str, str, Any],
    param_values: dict[str, float],
    design_values: dict[str, float] | None,
) -> np.ndarray | None:
    """Values for one variable at ``x*`` without solving.

    Parameters take their nominal value (a value outside the variable bounds is
    refused, see :func:`_check_nominal_in_bounds`); design inputs
    take their fixed design value. Returns ``None`` on any shape mismatch or a
    missing design value, signalling the caller to fall back to the solve.
    """
    kind, name, var = src_entry
    size = int(getattr(var, "size", 1) or 1)
    if kind == "param":
        pv = np.asarray(param_values[name], dtype=np.float64).ravel()
        if pv.size == 1:
            arr = np.full(size, float(pv[0]))
        elif pv.size == size:
            arr = pv.astype(np.float64).copy()
        else:
            return None
        _check_nominal_in_bounds(name, var, arr)
        return arr
    if not design_values or name not in design_values:
        return None
    dv: np.ndarray = np.asarray(design_values[name], dtype=np.float64).ravel()
    if dv.size == 1:
        return np.full(size, float(dv[0]))
    if dv.size == size:
        out: np.ndarray = dv.copy()
        return out
    return None


def _assemble_x_flat_direct(em, param_values, design_values):
    """Assemble the flat solution vector ``x*`` directly, or ``None``.

    Bypasses the QP solve for pure explicit response models (see
    :func:`_design_source_map`). The result is identical to solving
    ``min Σ(θ - θ_nom)²`` with the design fixed, but with no solver call.
    """
    from discopt.parametric import variable_slices

    src = _design_source_map(em)
    if src is None:
        return None
    parts = []
    for vname in variable_slices(em.model):
        arr = _direct_var_values(src[vname], param_values, design_values)
        if arr is None:
            return None
        parts.append(arr)
    _, jnp = _require_jax()

    return jnp.array(np.concatenate(parts), dtype=jnp.float64)


def _assemble_x_flat_batch_direct(em, param_values, design_points):
    """Stack ``x*`` for many design points into one ``(B, n)`` array, or ``None``."""
    from discopt.parametric import variable_slices

    src = _design_source_map(em)
    if src is None:
        return None
    var_names = list(variable_slices(em.model))
    rows = []
    for dp in design_points:
        parts = []
        for vname in var_names:
            arr = _direct_var_values(src[vname], param_values, dp)
            if arr is None:
                return None
            parts.append(arr)
        rows.append(np.concatenate(parts))
    _, jnp = _require_jax()

    return jnp.asarray(np.stack(rows, axis=0), dtype=jnp.float64)


# ─────────────────────────────────────────────────────────────
# Compiled-Jacobian cache
# ─────────────────────────────────────────────────────────────
#
# Building the model and tracing the response Jacobian costs far more than
# evaluating it: about 1.4 s against well under a millisecond for an ODE
# experiment, whose integrator JAX re-traces on every un-jitted call. Every
# design search evaluates the FIM of the *same* experiment at the *same*
# nominal parameters for many design points, so the model and a jitted
# Jacobian are kept per experiment and reused.
#
# A compiled Jacobian does not depend on the nominal parameters at all: it maps
# the flat vector x* (which carries the parameter values) to the responses. Only
# the *model* could depend on them, since create_model receives them. So a
# kernel is reused for new nominal values whenever the model those values build
# is structurally identical -- same variables, bounds, response expressions and
# measurement errors, and the same underlying callables behind any custom node
# (see _model_signature). That is what makes a sweep over parameter draws -- a
# robust design, a profile, a Monte Carlo over the posterior -- pay for one
# trace instead of one per draw.
#
# The key also carries a fingerprint of the experiment's own state, so an
# experiment edited after a first call (say, more integration steps) is
# recompiled rather than served a stale function. Only pure explicit response
# models are cached: those are the models whose x* is assembled directly, so
# nothing on the cached model is ever mutated. Call :func:`clear_fim_cache` to
# drop everything explicitly.

_FIM_CACHE_ATTR = "_discopt_doe_fim_cache"
_FIM_CACHE_SIZE = 8
_CACHED_EXPERIMENTS: list[weakref.ref] = []


@dataclass
class _FIMKernel:
    """A built model plus its jitted response Jacobian, reusable across designs."""

    em: ExperimentModel
    src: dict
    var_names: list[str]
    param_indices: list[int]
    sigma_inv: np.ndarray
    jac: Callable
    batch_jac: Callable
    signature: Any = None

    def x_flat(self, param_values, design_values):
        """``x*`` for one design, or ``None`` on a shape mismatch / missing value."""
        parts = []
        for vname in self.var_names:
            arr = _direct_var_values(self.src[vname], param_values, design_values)
            if arr is None:
                return None
            parts.append(arr)
        return np.concatenate(parts)

    def result(self, J: np.ndarray, prior_fim: np.ndarray | None) -> FIMResult:
        fim = J.T @ self.sigma_inv @ J
        if prior_fim is not None:
            fim = fim + prior_fim
        return FIMResult(
            fim=np.asarray(fim),
            jacobian=np.asarray(J),
            parameter_names=fim_parameter_names(self.em),
            response_names=self.em.response_names,
        )


def _experiment_fingerprint(experiment: Experiment) -> Any:
    """A cheap summary of the experiment's state, to detect edits after caching."""
    try:
        if dataclasses.is_dataclass(experiment):
            items = [
                (f.name, getattr(experiment, f.name))
                for f in dataclasses.fields(experiment)
                if f.name != _FIM_CACHE_ATTR
            ]
        else:
            items = [(k, v) for k, v in vars(experiment).items() if k != _FIM_CACHE_ATTR]
        return repr(items)
    except Exception:  # noqa: BLE001 - an unfingerprintable experiment is not cached
        return None


def _callable_identities(node: Any, depth: int = 0, seen: set[int] | None = None) -> list:
    """Identities of the callables reachable from an expression.

    A custom node holds a Python function that the expression's ``repr`` cannot
    see, so two models whose responses print identically may still compute
    different things. Any dependence on the nominal parameters must be created
    inside ``create_model`` -- it is handed the values -- which makes a fresh
    function object, so comparing identities is enough to tell the two apart. A
    bound method is compared by its underlying function and instance, since
    attribute access builds a new wrapper every time.
    """
    seen = set() if seen is None else seen
    out: list = []
    if node is None or depth > 8 or id(node) in seen:
        return out
    seen.add(id(node))
    fn = getattr(node, "fn", None)
    if callable(fn):
        f = getattr(fn, "__func__", fn)
        out.append((getattr(f, "__qualname__", ""), id(f), id(getattr(fn, "__self__", None))))
    for value in getattr(node, "__dict__", {}).values():
        if isinstance(value, (list, tuple)):
            for item in value:
                out.extend(_callable_identities(item, depth + 1, seen))
        elif hasattr(value, "__dict__"):
            out.extend(_callable_identities(value, depth + 1, seen))
    return out


def _model_signature(em: ExperimentModel) -> Any:
    """What a built model computes, as a comparable value (or ``None``).

    Two models with the same signature differ at most in the nominal parameter
    values that built them -- values that enter the FIM through ``x*``, not
    through the compiled Jacobian -- so one traced kernel serves both.
    """
    from discopt.parametric import variable_slices

    def var_sig(var: Any) -> tuple:
        return (
            getattr(var, "name", None),
            int(getattr(var, "size", 1) or 1),
            np.asarray(var.lb, dtype=float).ravel().tolist(),
            np.asarray(var.ub, dtype=float).ravel().tolist(),
        )

    try:
        return (
            tuple(variable_slices(em.model)),
            tuple((n, var_sig(v)) for n, v in sorted(em.unknown_parameters.items())),
            tuple((n, var_sig(v)) for n, v in sorted(em.design_inputs.items())),
            tuple(em.response_names),
            tuple(repr(em.responses[n]) for n in em.response_names),
            tuple(_measurement_sigma(em).ravel().tolist()),
            tuple(tuple(_callable_identities(em.responses[n])) for n in em.response_names),
        )
    except Exception:  # noqa: BLE001 - an unsummarizable model is simply not reused
        return None


def _param_key(param_values: dict[str, Any]) -> tuple:
    return tuple(
        (k, tuple(np.asarray(v, dtype=np.float64).ravel().tolist()))
        for k, v in sorted(param_values.items())
    )


def _cache_for(experiment: Experiment) -> OrderedDict | None:
    cache = getattr(experiment, _FIM_CACHE_ATTR, None)
    if isinstance(cache, OrderedDict):
        return cache
    cache = OrderedDict()
    try:
        object.__setattr__(experiment, _FIM_CACHE_ATTR, cache)
        _CACHED_EXPERIMENTS.append(weakref.ref(experiment))
    except (AttributeError, TypeError):  # __slots__, no __dict__, or no weakref support
        return None
    return cache


def clear_fim_cache(experiment: Experiment | None = None) -> None:
    """Drop the compiled FIM Jacobians kept for ``experiment`` (or for all).

    :func:`compute_fim` and the design searches keep, per experiment, the built
    model and a jitted Jacobian, reused across designs and across nominal
    parameter values that build the same model, and recompiled automatically
    when the experiment's attributes or its model change. Clear the
    cache after changing something the fingerprint cannot see, such as the body
    of a function the experiment calls, or to release memory.
    """
    targets = [experiment] if experiment is not None else [r() for r in _CACHED_EXPERIMENTS]
    for exp in targets:
        if exp is not None and isinstance(getattr(exp, _FIM_CACHE_ATTR, None), OrderedDict):
            getattr(exp, _FIM_CACHE_ATTR).clear()
    if experiment is None:
        _CACHED_EXPERIMENTS.clear()


def _fim_kernel(experiment: Experiment, param_values: dict[str, float]) -> _FIMKernel | None:
    """The cached kernel for ``(experiment, param_values)``, building it if needed.

    Returns ``None`` for a model that needs a solve (constraints or implicit
    state); callers then take the general path.
    """
    fingerprint = _experiment_fingerprint(experiment)
    cache = _cache_for(experiment) if fingerprint is not None else None
    key = (_param_key(param_values), fingerprint)
    if cache is not None and key in cache:
        cache.move_to_end(key)
        return cache[key]

    em = None
    kernel = None
    if cache is not None:
        # New nominal values: build the model (cheap) and reuse the compiled
        # Jacobian of any cached kernel the same model structure produced.
        try:
            em = experiment.create_model(**param_values)
        except Exception:  # noqa: BLE001 - let the normal build path raise
            em = None
        signature = _model_signature(em) if em is not None else None
        if signature is not None:
            # Only entries under the current fingerprint: an experiment edited
            # after a first call (more integration steps, say) builds the same
            # model from the same callables, so the signature alone cannot tell
            # the stale kernel from a live one.
            for cached_key, cached in cache.items():
                if cached_key[1] == fingerprint and cached is not None:
                    if cached.signature == signature:
                        kernel = cached
                        break
    if kernel is None:
        kernel = _build_fim_kernel(experiment, param_values, em=em)
    if cache is not None:
        cache[key] = kernel
        while len(cache) > _FIM_CACHE_SIZE:
            cache.popitem(last=False)
    return kernel


def _build_fim_kernel(experiment: Experiment, param_values: dict[str, float], *, em=None):
    from discopt.parametric import flatten_params, variable_slices

    jax, jnp = _require_jax()
    if em is None:
        em = experiment.create_model(**param_values)
    src = _design_source_map(em)
    if src is None:
        return None
    response_fns = [_compile_response(em.responses[n], em.model) for n in em.response_names]
    p_flat = flatten_params(em.model)

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    jacobian = jax.jacobian(response_vector)
    sigma = _measurement_sigma(em)
    return _FIMKernel(
        em=em,
        src=src,
        var_names=list(variable_slices(em.model)),
        param_indices=_get_param_indices(em),
        sigma_inv=np.diag(1.0 / sigma**2),
        jac=jax.jit(jacobian),
        batch_jac=jax.jit(jax.vmap(jacobian)),
        signature=_model_signature(em),
    )


def _check_design_names(em: ExperimentModel, design_values: dict[str, float] | None) -> None:
    # Reject unknown design_values keys. Silently ignoring them (e.g. a typo
    # like "temperture") would leave the real design input free and compute the
    # FIM at an arbitrary point -- worse than a crash, since it propagates
    # meaningless "optima" through optimal_experiment.
    if design_values:
        unknown = [name for name in design_values if name not in em.design_inputs]
        if unknown:
            raise ValueError(
                f"unknown design input(s) {unknown} in design_values; "
                f"model design inputs are {sorted(em.design_inputs)}."
            )


def compute_fim(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    prior_fim: np.ndarray | None = None,
    method: str = "autodiff",
    fd_step: float = 1e-5,
) -> FIMResult:
    """Compute the Fisher Information Matrix via JAX autodiff.

    Uses ``jax.jacobian`` to compute exact sensitivities ``∂y/∂θ``, then:

        FIM = J^T Σ^{-1} J + FIM_prior

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal values for unknown parameters. These are set on the
        corresponding variables before computing the Jacobian.
    design_values : dict[str, float], optional
        Values for design input variables. If provided, these are fixed
        before computing the Jacobian.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previous experiments (for sequential DoE).
    method : str, default "autodiff"
        Sensitivity computation method: ``"autodiff"`` (exact JAX) or
        ``"finite_difference"`` (central differences, for validation).
    fd_step : float, default 1e-5
        Relative perturbation size for finite differences: the actual step
        for each parameter is ``fd_step * max(|value|, 1)`` (only used when
        ``method="finite_difference"``).

    Returns
    -------
    FIMResult
        FIM, Jacobian, and optimality metrics.

    Notes
    -----
    For a pure explicit response model (no constraints; every variable an
    unknown parameter or a design input) the built model and a jitted Jacobian
    are cached per experiment and nominal parameter values, so repeated calls
    at new design points cost a compiled evaluation rather than a rebuild and
    re-trace. See :func:`clear_fim_cache`.
    """

    result = _compute_fim(
        experiment, param_values, design_values, prior_fim=None, method=method, fd_step=fd_step
    )
    if prior_fim is not None:
        P = check_prior_fim(prior_fim, result.parameter_names)
        result = dataclasses.replace(result, fim=result.fim + P)
    if not np.all(np.isfinite(result.fim)):
        import warnings

        warnings.warn(
            "the FIM has non-finite entries: the responses or their sensitivities "
            "overflowed at this design (for an ODE experiment, typically explicit RK4 "
            "on a stiff system -- use more n_steps or method='trapezoid', and see "
            "check_accuracy).",
            stacklevel=2,
        )
    return result


def _compute_fim(experiment, param_values, design_values, *, prior_fim, method, fd_step):
    from discopt.parametric import extract_x_flat, flatten_params

    if method == "autodiff":
        kernel = _fim_kernel(experiment, param_values)
        if kernel is not None:
            _check_design_names(kernel.em, design_values)
            x = kernel.x_flat(param_values, design_values)
            if x is not None:
                J = np.asarray(kernel.jac(x))[:, kernel.param_indices]
                return kernel.result(J, prior_fim)

    # Build the model at nominal parameter values
    em = experiment.create_model(**param_values)
    _check_design_names(em, design_values)

    # Fast path: for a pure explicit response model (no constraints; every
    # variable is an unknown parameter or a design input) the solution point
    # x* is fully determined by the nominal parameters and the fixed design.
    # The QP solve below would merely reconstruct values we already know, so
    # assemble x* directly and skip it. ``x_flat`` here is identical (to solver
    # tolerance) to the solved one.
    x_flat = _assemble_x_flat_direct(em, param_values, design_values)

    if x_flat is None:
        # General path: a constrained / implicit-state model genuinely needs a
        # solve to recover x*. Fix the design, then minimise Σ(θ - θ_nom)².
        for n in em.parameter_names:
            _check_nominal_in_bounds(n, em.unknown_parameters[n], param_values[n])
        if design_values:
            for name, val in design_values.items():
                if name in em.design_inputs:
                    var = em.design_inputs[name]
                    # Fix design variable by setting lb = ub = val
                    val_arr = np.asarray(val, dtype=np.float64)
                    if var.shape:
                        val_arr = np.full(var.shape, val_arr)
                    var.lb = val_arr
                    var.ub = val_arr

        em.model.minimize(
            sum((em.unknown_parameters[n] - param_values[n]) ** 2 for n in em.parameter_names)
        )
        result = em.model.solve()

        x_flat = extract_x_flat(result, em.model)

    # Compile response functions
    response_fns = []
    for name in em.response_names:
        fn = _compile_response(em.responses[name], em.model)
        response_fns.append(fn)

    # Build p_flat for any model Parameters (distinct from unknown_parameters)
    p_flat = flatten_params(em.model)

    J = _total_jacobian(em, response_fns, x_flat, p_flat, method=method, fd_step=fd_step)

    # Measurement covariance (diagonal)
    sigma = _measurement_sigma(em)
    Sigma_inv = np.diag(1.0 / sigma**2)

    # FIM = J^T Σ^{-1} J
    fim = np.asarray(J.T @ Sigma_inv @ J)

    if prior_fim is not None:
        fim = fim + prior_fim

    return FIMResult(
        fim=fim,
        jacobian=np.asarray(J),
        parameter_names=fim_parameter_names(em),
        response_names=em.response_names,
    )


def compute_fim_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_points: list[dict[str, float]],
    *,
    prior_fim: np.ndarray | None = None,
    method: str = "autodiff",
    fd_step: float = 1e-5,
) -> list[FIMResult]:
    """Compute the FIM for a batch of design points (multi-RHS).

    Fast path — for a pure explicit response model (no constraints; every
    variable an unknown parameter or design input) the per-point ``x*`` is
    assembled directly (no QP solve) and the response Jacobian is evaluated for
    the whole batch in a single ``vmap`` pass. This is the multi-RHS analogue
    of :func:`compute_fim`: the model is built and the response functions
    compiled once, then reused across every design point.

    Returns one :class:`FIMResult` per design point, in input order. Falls back
    to a per-point :func:`compute_fim` loop for any model that needs a solve or
    a non-autodiff ``method``, so the result is always identical to calling
    :func:`compute_fim` on each point.
    """
    if not design_points:
        return []

    kernel = _fim_kernel(experiment, param_values) if method == "autodiff" else None
    X = None
    if kernel is not None:
        for dp in design_points:
            _check_design_names(kernel.em, dp)
        rows = [kernel.x_flat(param_values, dp) for dp in design_points]
        if all(r is not None for r in rows):
            X = np.stack(rows, axis=0)
    if X is None:
        return [
            compute_fim(
                experiment, param_values, dp, prior_fim=prior_fim, method=method, fd_step=fd_step
            )
            for dp in design_points
        ]

    # One compiled Jacobian, vmapped across the batch axis of x*.
    J_all = np.asarray(kernel.batch_jac(X))[:, :, kernel.param_indices]
    return [kernel.result(J_all[b], prior_fim) for b in range(J_all.shape[0])]


def _make_direct_fim_evaluator(
    experiment: Experiment,
    param_values: dict[str, float],
    *,
    prior_fim: np.ndarray | None = None,
) -> Callable[[dict[str, float] | None], FIMResult] | None:
    """Return a reusable FIM evaluator that compiles the response Jacobian once.

    For a *pure explicit response model* (see :func:`_design_source_map`) the
    model is built and the response Jacobian JIT-compiled a *single* time; the
    returned ``evaluator(design_values) -> FIMResult`` then reuses that compiled
    Jacobian for every design point, assembling ``x*`` directly (no solve). This
    is the single-point analogue of :func:`compute_fim_batch`, intended for the
    adaptive scipy refinement loop in :mod:`discopt.doe.design`, where design
    points are chosen one at a time and the same Jacobian is evaluated many
    times.

    Returns ``None`` when the model has constraints or implicit state (the
    caller falls back to per-call :func:`compute_fim`, which solves). For every
    design point the returned FIM is identical (to float tolerance) to calling
    :func:`compute_fim` on that point — only the per-call model rebuild and JAX
    re-trace are eliminated.
    """
    kernel = _fim_kernel(experiment, param_values)
    if kernel is None:
        return None

    def evaluator(design_values: dict[str, float] | None) -> FIMResult:
        _check_design_names(kernel.em, design_values)
        x_flat = kernel.x_flat(param_values, design_values)
        if x_flat is None:
            # Per-point shape/missing-design mismatch: fall back to the solve.
            return compute_fim(experiment, param_values, design_values, prior_fim=prior_fim)
        J = np.asarray(kernel.jac(x_flat))[:, kernel.param_indices]
        return kernel.result(J, prior_fim)

    return evaluator


@dataclass
class IdentifiabilityResult:
    """Minimal identifiability assessment (backwards-compatible).

    Attributes
    ----------
    is_identifiable : bool
        True if all parameters are identifiable (FIM is full rank).
    fim_rank : int
        Numerical rank of the FIM.
    n_parameters : int
        Total number of unknown parameters.
    problematic_parameters : list[str]
        Parameters with the largest component in the null directions.
    condition_number : float
        Condition number of the FIM.
    fim_result : FIMResult
        The underlying FIM computation result.
    """

    is_identifiable: bool
    fim_rank: int
    n_parameters: int
    problematic_parameters: list[str]
    condition_number: float
    fim_result: FIMResult

    @property
    def parameter_names(self) -> list[str]:
        """Parameter order of every matrix in this result."""
        return list(self.fim_result.parameter_names)


@dataclass
class IdentifiabilityDiagnostics:
    """Full Belsley/Gutenkunst identifiability diagnostic bundle.

    Returned by :func:`diagnose_identifiability`. Superset of
    :class:`IdentifiabilityResult`; includes everything needed to apply
    the regression-diagnostic rules of Belsley, Kuh & Welsch (1980) and
    the sloppy-model spectrum of Gutenkunst et al. (2007).

    Scaling conventions
    -------------------
    - ``singular_values``, ``condition_indices``, ``variance_decomposition``,
      ``vif``: computed on the *unit-column-length* scaled Jacobian (each
      column divided by its 2-norm). This is the Belsley convention; no
      mean-centering since there is no intercept in a sensitivity Jacobian.
    - ``log_eigenvalue_spectrum``, ``normalized_log_spectrum``,
      ``standard_errors``, ``correlation_matrix``: computed on the physical
      FIM = J^T Sigma^-1 J (unscaled).

    Notes
    -----
    Yao ranking and condition indices are *not* invariant under
    reparameterization (e.g. theta -> log theta). Profile likelihood is.
    If the condition number is large, try a log-scale reparameterization
    before concluding non-identifiability.

    Attributes
    ----------
    is_identifiable : bool
        True if all parameters are identifiable (FIM is full rank).
    fim_rank : int
        Numerical rank of the FIM.
    n_parameters : int
        Total number of unknown parameters.
    condition_number : float
        Condition number of the FIM (physical, unscaled).
    fim_result : FIMResult
        Underlying FIM computation result.
    singular_values : numpy.ndarray
        Singular values of the scaled Jacobian, descending.
    condition_indices : numpy.ndarray
        Belsley condition indices eta_k = sigma_max / sigma_k.
    vif : dict[str, float]
        Variance inflation factor per parameter; ``nan`` if undefined.
    variance_decomposition : numpy.ndarray
        Belsley pi_{jk}, shape ``(n_params, n_params)``. Rows sum to 1.
    correlation_matrix : numpy.ndarray
        Parameter correlation from FIM^-1 (pseudoinverse if singular).
        Entries touching a null direction are ``nan``.
    log_eigenvalue_spectrum : numpy.ndarray
        log10 of FIM eigenvalues, sorted descending.
    normalized_log_spectrum : numpy.ndarray
        log10(lambda_k / lambda_max); the Gutenkunst sloppy-model form.
    null_space : list[dict[str, float]]
        One entry per null direction (sigma_k < tol). Each entry maps
        parameter name to the (sign-normalized) coefficient in the
        right singular vector.
    standard_errors : dict[str, float]
        sqrt(diag(FIM^-1)); ``nan`` for parameters without identifiability.
    warnings : list[str]
        Human-readable flags for problematic diagnostics.
    problematic_parameters : list[str]
        Parameters with the largest component in a null direction
        (one per null direction; for backwards compatibility with
        :class:`IdentifiabilityResult`).
    """

    is_identifiable: bool
    fim_rank: int
    n_parameters: int
    condition_number: float
    fim_result: FIMResult
    singular_values: np.ndarray
    condition_indices: np.ndarray
    vif: dict[str, float]
    variance_decomposition: np.ndarray
    correlation_matrix: np.ndarray
    log_eigenvalue_spectrum: np.ndarray
    normalized_log_spectrum: np.ndarray
    null_space: list[dict[str, float]]
    standard_errors: dict[str, float]
    warnings: list[str]
    problematic_parameters: list[str]

    @property
    def parameter_names(self) -> list[str]:
        """Row/column order of ``correlation_matrix``, ``variance_decomposition``."""
        return list(self.fim_result.parameter_names)

    def correlation_frame(self) -> dict[str, dict[str, float]]:
        """The correlation matrix as ``{name: {name: value}}``, keyed by parameter."""
        names = self.parameter_names
        C = np.asarray(self.correlation_matrix)
        return {a: {b: float(C[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}

    def correlation(self, a: str, b: str) -> float:
        """Estimated correlation between parameters ``a`` and ``b`` (``nan`` if undefined)."""
        names = self.parameter_names
        for n in (a, b):
            if n not in names:
                raise KeyError(f"{n!r} is not a parameter ({names})")
        return float(np.asarray(self.correlation_matrix)[names.index(a), names.index(b)])


def diagnose_identifiability(
    experiment: Experiment,
    param_values: dict[str, float] | None = None,
    design_values: dict[str, float] | None = None,
    *,
    tol: float | None = None,
    estimation_result=None,
) -> IdentifiabilityDiagnostics:
    """Full identifiability diagnostics (Belsley + Gutenkunst).

    Computes the FIM and the scaled sensitivity Jacobian, then returns
    condition indices, variance-inflation factors, variance-decomposition
    proportions, the correlation matrix, the sloppy-model eigenvalue
    spectrum, and a null-space report.

    The function replaces :func:`check_identifiability` for new code;
    ``check_identifiability`` is kept as a thin wrapper.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float], optional
        Nominal parameter values (typically a fitted estimate). Either
        this or ``estimation_result`` must be supplied. If both are
        supplied, ``param_values`` wins.
    design_values : dict[str, float], optional
        Design input values.
    tol : float, optional
        Relative tolerance on singular values for the rank decision.
        Defaults to the LAPACK convention
        ``max(n_rows, n_params) * eps``.
    estimation_result : EstimationResult, optional
        A fit produced by :func:`discopt.estimate.estimate_parameters`.
        When supplied, its ``parameters`` dict is used as the nominal
        point. Convenience for the common pattern
        ``diagnose_identifiability(exp, estimation_result=res)``.

    Returns
    -------
    IdentifiabilityDiagnostics
        Full diagnostic bundle.
    """
    if param_values is None:
        if estimation_result is None:
            raise TypeError(
                "diagnose_identifiability requires either param_values or estimation_result"
            )
        param_values = dict(estimation_result.parameters)
    fim_result = compute_fim(experiment, param_values, design_values)
    return _diagnostics_from_fim_result(fim_result, tol=tol)


def _diagnostics_from_fim_result(
    fim_result: FIMResult,
    *,
    tol: float | None = None,
) -> IdentifiabilityDiagnostics:
    """Build diagnostics from an existing FIMResult.

    Factored out so both :func:`diagnose_identifiability` and
    :func:`check_identifiability` can use the same linear-algebra path.
    """
    fim = np.asarray(fim_result.fim, dtype=np.float64)
    jac = np.asarray(fim_result.jacobian, dtype=np.float64)
    names = list(fim_result.parameter_names)
    n_params = len(names)

    # Scaled Jacobian: unit-column-length. Columns with zero norm (a
    # parameter with no sensitivity) get a zero column; they will be
    # flagged as non-identifiable by the singular-value test below.
    col_norms = np.linalg.norm(jac, axis=0)
    safe_norms = np.where(col_norms > 0, col_norms, 1.0)
    J_s = jac / safe_norms
    J_s[:, col_norms == 0] = 0.0

    n_rows = max(J_s.shape[0], 1)
    if tol is None:
        tol = max(n_rows, n_params) * np.finfo(np.float64).eps

    # SVD of the scaled Jacobian.
    if J_s.shape[0] == 0:
        sv = np.zeros(n_params)
        Vt = np.eye(n_params)
    else:
        _, sv_raw, Vt = np.linalg.svd(J_s, full_matrices=False)
        sv = np.concatenate([sv_raw, np.zeros(n_params - sv_raw.size)])
        if Vt.shape[0] < n_params:
            # When J_s has fewer rows than columns, SVD returns only
            # rank-m right singular vectors. Complete them to an
            # orthonormal basis of R^{n_params}. A full-mode QR of V
            # (n_params × m) yields Q of shape (n_params, n_params)
            # whose first m columns match V's column space and whose
            # remaining n_params - m columns are an orthonormal basis
            # for the orthogonal complement — the true null space.
            # Using standard-basis rows directly would generally not be
            # orthogonal to the existing Vt.
            Q, _ = np.linalg.qr(Vt.T, mode="complete")
            extra = Q[:, Vt.shape[0] :].T
            Vt = np.vstack([Vt, extra])

    sv_max = sv[0] if sv.size and sv[0] > 0 else 0.0
    if sv_max > 0:
        rank = int(np.sum(sv > tol * sv_max))
    else:
        rank = 0

    # Condition indices: sigma_max / sigma_k (infinity for null directions).
    with np.errstate(divide="ignore"):
        condition_indices = np.where(sv > 0, sv_max / np.maximum(sv, np.finfo(float).tiny), np.inf)
    if sv_max == 0:
        condition_indices = np.full(n_params, np.inf)

    # Belsley variance-decomposition proportions.
    # phi_{j,k} = V_{j,k}^2 / sigma_k^2 ; pi_{j,k} = phi_{j,k} / sum_k phi_{j,k}
    V = Vt.T  # columns are right singular vectors
    sv_sq = np.where(sv > 0, sv**2, np.finfo(float).tiny)
    phi = (V**2) / sv_sq[np.newaxis, :]
    row_sums = phi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    vdp = phi / row_sums

    # VIF via the inverse of the correlation matrix of the unit-column-
    # length Jacobian: VIF_j = [C^-1]_{jj}.
    if J_s.shape[0] > 0:
        C = J_s.T @ J_s  # = correlation matrix since columns are unit length
    else:
        C = np.zeros((n_params, n_params))
    try:
        C_inv = np.linalg.inv(C)
        vif_array = np.diag(C_inv)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)
        vif_array = np.diag(C_inv)

    # Parameters whose direction is deficient → VIF is effectively infinite.
    # Detect by checking whether each parameter's axis vector lies
    # (almost) in the span of null singular vectors.
    null_indices = np.where(sv <= tol * max(sv_max, np.finfo(float).tiny))[0]
    null_directions = V[:, null_indices] if null_indices.size else np.zeros((n_params, 0))
    if null_directions.size:
        null_projections = np.sum(null_directions**2, axis=1)  # per parameter
    else:
        null_projections = np.zeros(n_params)
    in_null = null_projections > _NULL_PROJECTION_THRESHOLD
    if null_directions.size:
        vif_array = np.where(in_null, np.inf, vif_array)
    vif = {names[j]: float(vif_array[j]) for j in range(n_params)}

    # FIM-based correlation matrix and standard errors.
    try:
        fim_inv = np.linalg.inv(fim)
        singular_fim = False
    except np.linalg.LinAlgError:
        fim_inv = np.linalg.pinv(fim)
        singular_fim = True

    diag = np.diag(fim_inv)
    se_array = np.where(diag >= 0, np.sqrt(np.clip(diag, 0.0, None)), np.nan)
    if singular_fim and null_directions.size:
        # Parameters with large null-direction projection have no
        # meaningful standard error or correlation.
        se_array = np.where(in_null, np.nan, se_array)
    standard_errors = {names[j]: float(se_array[j]) for j in range(n_params)}

    # Correlation matrix.
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.clip(np.diag(fim_inv), 0.0, None))
        safe_d = np.where(d > 0, d, np.nan)
        corr = fim_inv / np.outer(safe_d, safe_d)
    if singular_fim and null_directions.size:
        corr[in_null, :] = np.nan
        corr[:, in_null] = np.nan

    # Eigenvalue spectrum of the physical FIM.
    eigvals = np.linalg.eigvalsh(fim)
    eigvals = np.sort(eigvals)[::-1]  # descending
    eig_max = eigvals[0] if eigvals.size and eigvals[0] > 0 else 0.0
    if eig_max > 0:
        clipped = np.clip(eigvals, eig_max * np.finfo(float).eps, None)
        log_spectrum = np.log10(clipped)
        normalized_log = np.log10(clipped / eig_max)
    else:
        log_spectrum = np.full(n_params, -np.inf)
        normalized_log = np.full(n_params, -np.inf)

    # Null-space report.
    null_space: list[dict[str, float]] = []
    problematic: list[str] = []
    for idx in null_indices:
        direction = V[:, idx].copy()
        # Sign normalization: largest-magnitude entry positive.
        max_mag = int(np.argmax(np.abs(direction)))
        if direction[max_mag] < 0:
            direction = -direction
        null_space.append({names[j]: float(direction[j]) for j in range(n_params)})
        problematic.append(names[max_mag])

    # Warnings.
    warnings_out: list[str] = []
    for k in range(n_params):
        eta = condition_indices[k]
        if eta > 30:
            warnings_out.append(
                f"serious collinearity: condition index eta_{k + 1} = {eta:.3g} > 30"
            )
        elif eta > 10 and not np.isinf(eta):
            warnings_out.append(f"mild collinearity: condition index eta_{k + 1} = {eta:.3g} > 10")
    for name, v in vif.items():
        if np.isfinite(v) and v > 10:
            warnings_out.append(f"VIF[{name}] = {v:.3g} > 10")
        elif np.isinf(v):
            warnings_out.append(f"VIF[{name}] is infinite (parameter lies in a null direction)")
    # Correlation warnings on finite entries only.
    for i in range(n_params):
        for j in range(i + 1, n_params):
            rho = corr[i, j]
            if np.isfinite(rho) and abs(rho) > 0.95:
                warnings_out.append(f"|rho[{names[i]},{names[j]}]| = {abs(rho):.3g} > 0.95")

    return IdentifiabilityDiagnostics(
        is_identifiable=(rank == n_params),
        fim_rank=rank,
        n_parameters=n_params,
        condition_number=fim_result.me_optimal,
        fim_result=fim_result,
        singular_values=sv,
        condition_indices=condition_indices,
        vif=vif,
        variance_decomposition=vdp,
        correlation_matrix=corr,
        log_eigenvalue_spectrum=log_spectrum,
        normalized_log_spectrum=normalized_log,
        null_space=null_space,
        standard_errors=standard_errors,
        warnings=warnings_out,
        problematic_parameters=problematic,
    )


def check_identifiability(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    tol: float = 1e-6,
) -> IdentifiabilityResult:
    """Minimal identifiability check (backwards-compatible).

    Computes the FIM and reports its rank plus a representative
    problematic parameter per null direction. For the full Belsley /
    Gutenkunst diagnostic toolkit, use :func:`diagnose_identifiability`.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_values : dict[str, float], optional
        Design input values.
    tol : float, default 1e-6
        Absolute-like tolerance (scaled by the top singular value of FIM)
        used to decide the rank. Kept for backwards compatibility.

    Returns
    -------
    IdentifiabilityResult
        Minimal identifiability assessment.
    """
    fim_result = compute_fim(experiment, param_values, design_values)
    fim = fim_result.fim
    n_params = len(fim_result.parameter_names)

    singular_values = np.linalg.svd(fim, compute_uv=False)
    if singular_values.size and singular_values[0] > 0:
        rank = int(np.sum(singular_values > tol * singular_values[0]))
    else:
        rank = 0

    _, _, Vt = np.linalg.svd(fim)
    problematic: list[str] = []
    for i in range(rank, n_params):
        direction = Vt[i]
        max_idx = int(np.argmax(np.abs(direction)))
        problematic.append(fim_result.parameter_names[max_idx])

    return IdentifiabilityResult(
        is_identifiable=(rank == n_params),
        fim_rank=rank,
        n_parameters=n_params,
        problematic_parameters=problematic,
        condition_number=fim_result.me_optimal,
        fim_result=fim_result,
    )


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────


def check_prior_fim(prior_fim: Any, parameter_names: list[str]) -> np.ndarray:
    """Validate a prior FIM against the FIM's parameters; return it as an array.

    It must be a finite, symmetric, positive semi-definite matrix with one
    row and column per FIM parameter (see :func:`fim_parameter_names`).
    """
    P = np.asarray(prior_fim, dtype=float)
    n = len(parameter_names)
    if P.shape != (n, n):
        raise ValueError(
            f"prior_fim has shape {P.shape}; expected ({n}, {n}) for the parameters "
            f"{parameter_names}"
        )
    if not np.all(np.isfinite(P)):
        raise ValueError("prior_fim contains non-finite entries")
    scale = max(float(np.max(np.abs(P))), 1e-300)
    if np.max(np.abs(P - P.T)) > 1e-8 * scale:
        raise ValueError("prior_fim is not symmetric")
    if np.min(np.linalg.eigvalsh(0.5 * (P + P.T))) < -1e-8 * scale:
        raise ValueError(
            "prior_fim is not positive semi-definite (a Fisher information matrix "
            "cannot have a negative eigenvalue)"
        )
    return P


def fim_parameter_names(em: ExperimentModel) -> list[str]:
    """One name per FIM row/column, in ``x*`` order.

    A scalar parameter keeps its name; a vector-valued one (a single
    ``Variable`` of size n) expands to ``name[0] .. name[n-1]``, matching the
    rows :func:`_get_param_indices` contributes. Using the bare
    ``ExperimentModel.parameter_names`` there gave one name for n rows, and
    everything that pairs names with rows (identifiability diagnostics,
    standard errors, warnings) broke or mislabelled them.
    """
    from discopt.parametric import variable_slices

    slices = variable_slices(em.model)
    names: list[str] = []
    for name, var in em.unknown_parameters.items():
        sl = slices[var.name]
        size = sl.stop - sl.start
        names.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
    return names


def _get_param_indices(em: ExperimentModel) -> list[int]:
    """Find indices of unknown parameter variables in the flat x vector."""
    from discopt.parametric import variable_slices

    slices = variable_slices(em.model)
    param_indices: list[int] = []
    for var in em.unknown_parameters.values():
        sl = slices[var.name]
        param_indices.extend(range(sl.start, sl.stop))
    return param_indices


def _total_jacobian(em, response_fns, x_flat, p_flat, *, method="autodiff", fd_step=1e-5):
    """``dy/dθ`` at a solved ``x*``, including the implicit states (see below).

    The one place a response Jacobian is taken for a model that needed a solve;
    everything that differentiates such a model goes through here.
    """
    param_indices = _get_param_indices(em)
    if method == "autodiff":
        jac = lambda fns, idx: _compute_jacobian_autodiff(fns, x_flat, p_flat, idx)  # noqa: E731
    elif method == "finite_difference":
        jac = lambda fns, idx: _compute_jacobian_fd(fns, x_flat, p_flat, idx, fd_step)  # noqa: E731
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'autodiff' or 'finite_difference'.")

    J = np.asarray(jac(response_fns, param_indices))
    state_indices = _get_state_indices(em)
    if state_indices:
        J = _add_implicit_sensitivity(
            em, J, jac, response_fns, param_indices, state_indices, x_flat, p_flat
        )
    return J


def _get_state_indices(em: ExperimentModel) -> list[int]:
    """Indices in x* of every variable that is neither a parameter nor a design input.

    These are the implicit states of a constrained model: their values at x*
    are functions of the parameters through the equality constraints.
    """
    from discopt.parametric import variable_slices

    known = {v.name for v in em.unknown_parameters.values()}
    known |= {v.name for v in em.design_inputs.values()}
    out: list[int] = []
    for vname, sl in variable_slices(em.model).items():
        if vname not in known:
            out.extend(range(sl.start, sl.stop))
    return out


def _add_implicit_sensitivity(
    em, J_theta, jac, response_fns, param_indices, state_indices, x_flat, p_flat
):
    """Total sensitivity ``dy/dθ`` for responses that depend on implicit states.

    Differentiating the response expressions with respect to the parameters
    alone holds every state fixed, which is wrong whenever a state is defined
    through a constraint (``z + k z^3 == x``, a mass balance, a discretized
    ODE): ``dz/dk`` is then silently dropped and the FIM is wrong -- all zeros
    when a response is a pure state. With the equality constraints
    ``g(θ, s) = 0`` determining the states ``s``, the implicit function theorem
    gives ``ds/dθ = -(∂g/∂s)^+ ∂g/∂θ`` and

        dy/dθ = ∂y/∂θ + ∂y/∂s · ds/dθ.

    Raises if the responses depend on states that the equality constraints do
    not determine, since no sensitivity can be computed for those.
    """
    import warnings

    _, jnp = _require_jax()

    J_s = np.asarray(jac(response_fns, state_indices))
    if not np.any(J_s):
        return J_theta

    eq_fns, ineq_fns = [], []
    for con in getattr(em.model, "_constraints", None) or []:
        body, sense = getattr(con, "body", None), getattr(con, "sense", None)
        if body is None or sense is None:
            raise NotImplementedError(
                f"cannot differentiate through a {type(con).__name__} constraint to "
                "compute the state sensitivities the FIM needs."
            )
        fn = _compile_response(body - con.rhs, em.model)
        (eq_fns if sense == "==" else ineq_fns).append(fn)

    if not eq_fns:
        raise ValueError(
            "the responses depend on model variables that are neither unknown "
            "parameters nor design inputs, and no equality constraint defines them, "
            "so their sensitivity to the parameters is undefined."
        )

    # Constraint bodies may be vector-valued: one residual vector for all rows.
    def g(x, p):
        return jnp.concatenate([jnp.ravel(jnp.asarray(f(x, p))) for f in eq_fns])

    rows = [lambda x, p, i=i: g(x, p)[i] for i in range(int(g(x_flat, p_flat).size))]
    G_theta = np.asarray(jac(rows, param_indices))
    G_s = np.asarray(jac(rows, state_indices))

    # Only the state directions the responses actually see must be determined.
    rank = int(np.linalg.matrix_rank(G_s))
    if rank < len(state_indices):
        null = np.linalg.svd(G_s)[2][rank:]
        if np.linalg.norm(J_s @ null.T) > 1e-8 * max(1.0, float(np.linalg.norm(J_s))):
            raise ValueError(
                f"the equality constraints determine only {rank} of the "
                f"{len(state_indices)} state variables the responses depend on, so "
                "dy/dθ (and the FIM) is undefined. Every variable that is not an "
                "unknown parameter or a design input must be fixed by the equality "
                "constraints."
            )

    for fn in ineq_fns:
        val = np.ravel(np.asarray(fn(x_flat, p_flat)))
        if np.any(np.abs(val) <= 1e-6 * np.maximum(1.0, np.abs(val))):
            warnings.warn(
                "an inequality constraint is active at the nominal point; the FIM "
                "sensitivities treat only the equality constraints as defining the "
                "states, so they ignore the active inequality.",
                stacklevel=3,
            )
            break

    dS = -np.linalg.lstsq(G_s, G_theta, rcond=None)[0]
    return J_theta + J_s @ dS


def _compute_jacobian_autodiff(response_fns, x_flat, p_flat, param_indices):
    """Compute Jacobian via JAX autodiff."""
    jax, jnp = _require_jax()

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    J_full = jax.jacobian(response_vector)(x_flat)
    return J_full[:, param_indices]


def _compute_jacobian_fd(response_fns, x_flat, p_flat, param_indices, step):
    """Compute Jacobian via central finite differences."""
    _, jnp = _require_jax()

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    n_responses = len(response_fns)
    n_params = len(param_indices)
    J = np.zeros((n_responses, n_params))

    for j, idx in enumerate(param_indices):
        # Scale the step by the parameter magnitude so it is a genuine relative
        # perturbation (as documented). A fixed absolute step causes
        # catastrophic cancellation for large parameters and a ~100%
        # perturbation for tiny ones.
        h = step * max(abs(float(x_flat[idx])), 1.0)
        x_plus = x_flat.at[idx].set(x_flat[idx] + h)
        x_minus = x_flat.at[idx].set(x_flat[idx] - h)
        J[:, j] = (response_vector(x_plus) - response_vector(x_minus)) / (2 * h)

    return J
