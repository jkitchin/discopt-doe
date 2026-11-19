"""Targeted (goal-oriented) design criteria: weighted precision on what you care about.

A model with many parameters, a small budget, and a question about a few of
them -- or about quantities computed from them -- does not need a design that
pins down every parameter. It needs one that makes the *targets* precise. This
module expresses every such goal through one matrix.

Mathematics
-----------
Let ``M`` be the Fisher information matrix of the ``p`` parameters ``θ`` and
``ψ = g(θ)`` the ``s`` target quantities. With ``K = ∂g/∂θ`` (``p × s``, taken
at the nominal ``θ``) the delta method gives the target covariance

    Σ_ψ = Kᵀ M⁻¹ K,

and ``C = (W^½ Σ_ψ W^½)⁻¹`` is the (weighted) information on the targets
(Pukelsheim's ``C_K(M)``). ``K`` depends on the nominal parameters, never on
the design, so it is built once and every candidate design costs one
factorization of ``M``. The criteria are the familiar ones, applied to the
targets instead of the parameters:

=====================  ==================================  ============================
criterion              value                               classical name
=====================  ==================================  ============================
``"trace"``            ``tr(W Σ_ψ) = Σ_j w_j Var(ψ_j)``    L / weighted-A (A_s, c, I)
``"average_variance"`` ``tr(W Σ_ψ) / Σ_j w_j``             same optimum as ``"trace"``
``"determinant"``      ``log det C``                       D_s, D_A
``"min_eigenvalue"``   ``λ_min(C)``                        E_s
``"condition_number"`` ``cond(C)``                         --
``"max_variance"``     ``max_j w_j Var(ψ_j)``              minimax (G on a region)
=====================  ==================================  ============================

Notes on the mathematics:

* **Nuisance parameters are marginalized, not dropped.** For a parameter
  subset ``S``, ``(Kᵀ M⁻¹ K)⁻¹ = M_SS - M_SN M_NN⁻¹ M_NS`` (the Schur
  complement), so experiments that decorrelate a target from a nuisance
  parameter are credited. Deleting the nuisance rows of ``M`` is wrong.
* **Weights.** Zero-weight targets are dropped. Under ``"determinant"`` a
  diagonal weight only shifts ``log det`` by a constant, so graded weights
  change nothing there (a warning says so); they matter for the trace,
  eigenvalue and minimax criteria.
* **Tolerances.** ``w_j = 1/τ_j²`` makes the trace criterion
  ``Σ (σ_j/τ_j)²`` and the minimax one ``max (σ_j/τ_j)²`` -- dimensionless,
  and "every ratio ≤ 1" is the natural stopping rule.
* **Singular M.** A target is estimable when ``range(K) ⊆ range(M)``. The
  covariance then uses the pseudo-inverse on the identifiable subspace, so a
  design can be optimal for the targets while ``M`` is singular overall --
  the usual situation with many parameters and few experiments.
* **Parameter scaling** does not change any targeted criterion (``K`` and
  ``M`` transform together), so ``scale_parameters`` is ignored with a target;
  use ``relative=True`` or tolerances to say what precision means.

Quick start
-----------
>>> from discopt.doe import Target, optimal_experiment
>>> t = Target.parameters(["k1", "Ea1"])                    # D_s: the rest are nuisance
>>> t = Target.parameters({"k1": 1.0, "k2": 0.1}, relative=True)
>>> t = Target.responses(at=[{"T": 350.0}], tolerances={"y": 0.01})
>>> res = optimal_experiment(exp, theta, bounds, criterion="trace", target=t)
>>> res.target_precision().summary()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Families of target. Targets from different families are in different units,
# so combining them needs tolerances (or relative=True) to be meaningful.
_PARAMETER_SPACE = "parameters"
_OUTPUT_SPACE = "outputs"
_FUNCTION_SPACE = "function"

# Same threshold as the singular-design warning in design.py: a correlation-form
# FIM eigenvalue below λ_max / 1e12 is round-off, so its direction is treated
# as unidentified.
_RANK_RTOL = 1e-12
# A target column whose part outside the identified subspace exceeds this
# fraction of its norm is not estimable.
_ESTIMABLE_RTOL = 1e-6

# Relative ridge of the smoothed criterion (see ResolvedTarget.criterion).
_SMOOTH_RIDGE = 1e-8

TARGET_CRITERIA = (
    "determinant",
    "trace",
    "average_variance",
    "min_eigenvalue",
    "condition_number",
    "max_variance",
)
# Criteria that need the targets to be linearly independent (C must exist).
_INFORMATION_CRITERIA = ("determinant", "min_eigenvalue", "condition_number")


# ---------------------------------------------------------------------------
# Parameter layout of an experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """How the FIM rows map to the experiment's (possibly vector) parameters."""

    names: list[str]  # one per FIM row ("k", "v[0]", "v[1]", ...)
    blocks: list[tuple[str, int, int]]  # (parameter, start, stop) in FIM order
    values: np.ndarray  # nominal value per FIM row
    scalar: dict[str, bool]  # parameter given as a scalar

    def index(self, name: str) -> list[int]:
        """FIM rows of ``name`` -- a scalar, a whole vector, or ``v[i]``."""
        if name in self.names:
            return [self.names.index(name)]
        for pname, a, b in self.blocks:
            if pname == name:
                return list(range(a, b))
        raise KeyError(
            f"{name!r} is not a parameter of the experiment; its parameters are {self.names}"
        )


def _layout(experiment: Any, param_values: Mapping[str, Any]) -> _Layout:
    from discopt.doe.fim import ParameterScaledExperiment, fim_parameter_names
    from discopt.parametric import variable_slices

    if isinstance(experiment, ParameterScaledExperiment):
        experiment = experiment.experiment
    em = experiment.create_model(**dict(param_values))
    slices = variable_slices(em.model)
    blocks, vals, scalar = [], [], {}
    start = 0
    for name, var in em.unknown_parameters.items():
        sl = slices[var.name]
        size = sl.stop - sl.start
        raw = np.asarray(param_values[name], dtype=float)
        scalar[name] = raw.ndim == 0
        vals.append(np.broadcast_to(raw.ravel(), (size,)))
        blocks.append((name, start, start + size))
        start += size
    values = np.concatenate(vals) if vals else np.zeros(0)
    return _Layout(fim_parameter_names(em), blocks, values, scalar)


# ---------------------------------------------------------------------------
# Weights and tolerances
# ---------------------------------------------------------------------------


def _check_positive(mapping: Mapping[str, float], what: str) -> None:
    for k, v in mapping.items():
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(f"{what} for {k!r} must be finite and >= 0, got {v}")


def _weights_and_tolerances(
    names: Sequence[str],
    keys: Sequence[str],
    weights: Mapping[str, float] | None,
    tolerances: Mapping[str, float] | None,
    scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column weight and absolute tolerance (``nan`` where none was given).

    ``keys[j]`` is the user-facing key column ``j`` is looked up under (a
    response name is shared by all its operating points). ``scale`` turns a
    relative tolerance / weight into an absolute one.
    """
    if weights is not None and tolerances is not None:
        raise ValueError("give weights or tolerances, not both (a tolerance τ is a weight 1/τ²)")
    s = len(names)
    scale = np.ones(s) if scale is None else np.asarray(scale, dtype=float)
    if tolerances is not None:
        _check_positive(tolerances, "tolerance")
        unknown = set(tolerances) - set(keys)
        if unknown:
            raise ValueError(f"tolerances for unknown targets {sorted(unknown)}")
        missing = sorted({k for k in keys if k not in tolerances})
        if missing:
            raise ValueError(f"no tolerance given for {missing}")
        tau = np.array([float(tolerances[k]) for k in keys]) * scale
        if np.any(tau <= 0):
            raise ValueError("tolerances must be > 0")
        return 1.0 / tau**2, tau
    if weights is None:
        w = np.ones(s)
    else:
        _check_positive(weights, "weight")
        unknown = set(weights) - set(keys)
        if unknown:
            raise ValueError(f"weights for unknown targets {sorted(unknown)}")
        w = np.array([float(weights.get(k, 0.0)) for k in keys])
    return w / scale**2, np.full(s, np.nan)


# ---------------------------------------------------------------------------
# Target specification (what the user writes)
# ---------------------------------------------------------------------------

_Builder = Callable[[Any, Mapping[str, Any], _Layout], tuple[np.ndarray, list, np.ndarray, Any]]


@dataclass(frozen=True)
class _Block:
    family: str
    commensurable: bool  # weights carry units (tolerances or relative)
    build: _Builder  # (experiment, θ, layout) -> (K, names, weights, tolerances)
    label: str


@dataclass(frozen=True)
class Target:
    """What a design should make precise: parameters, combinations, or outputs.

    Build one with the class methods and pass it as ``target=`` to
    :func:`~discopt.doe.optimal_experiment`,
    :func:`~discopt.doe.batch_optimal_experiment` or
    :func:`~discopt.doe.sequential_doe`. ``a + b`` combines targets; targets
    of different kinds (parameters vs. outputs) must each be given tolerances
    or ``relative=True`` so their precisions are comparable.

    A ``Target`` is a specification: it is turned into the matrix ``K`` at the
    nominal parameters by :meth:`resolve`, which the design functions call for
    you (and :func:`sequential_doe` re-calls every round, as ``K`` moves with
    the estimates). See :mod:`discopt.doe.targets` for the mathematics.
    """

    blocks: tuple[_Block, ...] = field(default_factory=tuple)

    # -- constructors -------------------------------------------------------

    @classmethod
    def parameters(
        cls,
        params: Sequence[str] | Mapping[str, float] | None = None,
        *,
        relative: bool = False,
        tolerances: Mapping[str, float] | None = None,
    ) -> Target:
        """Target parameters directly; every other parameter is a nuisance.

        Parameters
        ----------
        params : sequence of str or mapping name -> weight, optional
            A list targets those parameters with equal weight (``D_s`` /
            ``A_s``). A mapping gives each one a weight; a weight of 0 makes
            it a nuisance parameter. A vector parameter ``v`` stands for all
            its entries; ``"v[1]"`` names one. May be omitted when
            ``tolerances`` names the targets.
        relative : bool, default False
            Measure precision relative to the nominal value: the variance of
            ``log θ`` (weights are divided by ``θ²``; tolerances are fractions,
            ``0.05`` meaning 5 %).
        tolerances : mapping name -> τ, optional
            The standard error wanted on each target. Sets the weight to
            ``1/τ²`` and enables the "precise enough" check of
            :meth:`TargetPrecision.met`. Exclusive with weights in ``params``.
        """
        if isinstance(params, Mapping):
            spec = {str(k): float(v) for k, v in params.items()}
            explicit = True
        else:
            spec = {str(k): 1.0 for k in (params or ())}
            explicit = False
        if tolerances is not None:
            if explicit:
                raise ValueError("give weights or tolerances, not both")
            spec = {k: 1.0 for k in tolerances} if not spec else spec
        if not spec:
            raise ValueError("Target.parameters needs at least one parameter")

        def build(experiment, theta, lay: _Layout):
            cols, names, keys = [], [], []
            for key in spec:
                for i in lay.index(key):
                    cols.append(i)
                    names.append(lay.names[i])
                    keys.append(key)
            if len(set(cols)) != len(cols):
                raise ValueError(f"a parameter is targeted twice in {list(spec)}")
            K = np.zeros((len(lay.names), len(cols)))
            K[cols, range(len(cols))] = 1.0
            scale = np.abs(lay.values[cols]) if relative else None
            if relative and np.any(~(scale > 0)):
                raise ValueError("relative=True needs nonzero nominal values for the targets")
            w, tau = _weights_and_tolerances(
                names,
                keys,
                None if tolerances is not None else (spec if explicit else None),
                tolerances,
                scale,
            )
            return K, names, w, tau

        return cls(
            (
                _Block(
                    _PARAMETER_SPACE,
                    relative or tolerances is not None,
                    build,
                    f"parameters {list(spec)}",
                ),
            )
        )

    @classmethod
    def linear(
        cls,
        combinations: Mapping[str, Mapping[str, float]],
        *,
        weights: Mapping[str, float] | None = None,
        tolerances: Mapping[str, float] | None = None,
    ) -> Target:
        """Target linear combinations of parameters, ``ψ_j = Σ_i a_ij θ_i``.

        ``combinations`` maps a name to its coefficients, e.g.
        ``{"k1-k2": {"k1": 1, "k2": -1}}``. A single combination is
        c-optimality; several are ``D_A`` / L-optimality.
        """
        if not combinations:
            raise ValueError("Target.linear needs at least one combination")
        combos = {str(n): dict(c) for n, c in combinations.items()}

        def build(experiment, theta, lay: _Layout):
            K = np.zeros((len(lay.names), len(combos)))
            for j, coeffs in enumerate(combos.values()):
                for pname, a in coeffs.items():
                    rows = lay.index(pname)
                    if len(rows) != 1:
                        raise ValueError(
                            f"{pname!r} is a vector; name one entry, e.g. '{pname}[0]'"
                        )
                    K[rows[0], j] += float(a)
            names = list(combos)
            w, tau = _weights_and_tolerances(names, names, weights, tolerances, None)
            return K, names, w, tau

        return cls(
            (_Block(_PARAMETER_SPACE, tolerances is not None, build, f"linear {list(combos)}"),)
        )

    @classmethod
    def responses(
        cls,
        experiment: Any = None,
        at: Sequence[Mapping[str, float]] | None = None,
        *,
        responses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
        tolerances: Mapping[str, float] | None = None,
        relative: bool = False,
    ) -> Target:
        """Target model outputs (quantities of interest) at operating points.

        ``ψ`` is every selected response of ``experiment`` at every point in
        ``at``; ``K`` is its sensitivity ``∂ψ/∂θ`` at the nominal parameters.

        Parameters
        ----------
        experiment : Experiment, optional
            The model of the quantities of interest. It need not be measured
            at all -- a yield or selectivity at the plant's operating point --
            but its unknown parameters must be parameters of the experiment
            being designed (matched by name). Defaults to the experiment being
            designed, i.e. its own predictions at ``at``.
        at : sequence of mapping, optional
            Operating points: values of ``experiment``'s design inputs. Omit
            for an experiment without design inputs.
        responses : sequence of str, optional
            The responses to target (default: all of them, or the keys of
            ``weights`` / ``tolerances`` when given).
        weights, tolerances : mapping response -> value, optional
            Per response, shared by all its points.
        relative : bool, default False
            Precision relative to the predicted value (weights divided by
            ``q²``; tolerances are fractions).
        """
        points = [dict(p) for p in at] if at is not None else [None]
        if not points:
            raise ValueError("Target.responses got an empty list of points")

        def build(design_experiment, theta, lay: _Layout):
            from discopt.doe.fim import compute_fim, predict_responses

            qexp = experiment if experiment is not None else design_experiment
            chosen = list(responses) if responses is not None else None
            if chosen is None and (weights or tolerances):
                chosen = list(weights or tolerances)
            cols, names, keys, values = [], [], [], []
            for pt in points:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = compute_fim(qexp, dict(theta), pt)
                J = np.atleast_2d(np.asarray(res.jacobian, dtype=float))
                rnames = list(res.response_names)
                sel = chosen if chosen is not None else rnames
                missing = [r for r in sel if r not in rnames]
                if missing:
                    raise ValueError(f"unknown response(s) {missing}; the model has {rnames}")
                pred = predict_responses(qexp, dict(theta), pt) if relative else {}
                for r in sel:
                    row = np.zeros(len(lay.names))
                    for jq, pname in enumerate(res.parameter_names):
                        if pname not in lay.names:
                            raise ValueError(
                                f"the quantity-of-interest model depends on {pname!r}, which is "
                                "not a parameter of the experiment being designed"
                            )
                        row[lay.names.index(pname)] = J[rnames.index(r), jq]
                    cols.append(row)
                    where = ",".join(f"{k}={v:g}" for k, v in pt.items()) if pt else ""
                    names.append(f"{r}@{where}" if where and len(points) > 1 else r)
                    keys.append(r)
                    values.append(pred.get(r, 1.0))
            if len(set(names)) != len(names):
                names = [f"{n}#{i}" for i, n in enumerate(names)]
            K = np.stack(cols, axis=1)
            scale = np.abs(np.asarray(values)) if relative else None
            if relative and np.any(~(scale > 0)):
                raise ValueError("relative=True needs nonzero predictions at every point")
            w, tau = _weights_and_tolerances(names, keys, weights, tolerances, scale)
            return K, names, w, tau

        return cls((_Block(_OUTPUT_SPACE, relative or tolerances is not None, build, "responses"),))

    @classmethod
    def function(
        cls,
        fn: Callable[[dict[str, Any]], Any],
        names: Sequence[str],
        *,
        weights: Mapping[str, float] | None = None,
        tolerances: Mapping[str, float] | None = None,
    ) -> Target:
        """Target any differentiable function of the parameters.

        ``fn`` receives a mapping parameter name -> value (JAX scalars or
        arrays) and returns a scalar or a 1-D array of ``len(names)`` values,
        written with ``jax.numpy``. ``K`` is its JAX Jacobian at the nominal
        parameters, e.g. ``Target.function(lambda th: th["k1"] / th["k2"],
        ["ratio"])``.
        """
        names = [str(n) for n in names]
        if not names:
            raise ValueError("Target.function needs at least one name")

        def build(experiment, theta, lay: _Layout):
            import jax
            import jax.numpy as jnp

            def wrapped(x):
                th = {p: (x[a] if lay.scalar[p] else x[a:b]) for (p, a, b) in lay.blocks}
                return jnp.atleast_1d(jnp.asarray(fn(th))).ravel()

            x0 = jnp.asarray(lay.values, dtype=jnp.float64)
            J = np.atleast_2d(np.asarray(jax.jacfwd(wrapped)(x0), dtype=float))
            if J.shape[0] != len(names):
                raise ValueError(
                    f"the function returned {J.shape[0]} values for {len(names)} names"
                )
            w, tau = _weights_and_tolerances(names, names, weights, tolerances, None)
            return J.T, names, w, tau

        return cls((_Block(_FUNCTION_SPACE, tolerances is not None, build, f"function {names}"),))

    @classmethod
    def expression(
        cls,
        expressions: Mapping[str, str],
        *,
        weights: Mapping[str, float] | None = None,
        tolerances: Mapping[str, float] | None = None,
        relative: bool = False,
    ) -> Target:
        """Target functions of the parameters written as formulas.

        ``expressions`` maps a name to a formula in the (scalar) parameter
        names, e.g. ``{"ratio": "k1 / k2", "diff": "Ea1 - Ea2"}``, in the
        syntax of :func:`~discopt.doe.parse_expression`. The gradient is taken
        symbolically (sympy), so no JAX is needed. ``relative=True`` measures
        precision relative to the formula's value at the nominal parameters.
        This is the form the command line uses (``--target-expr``).
        """
        exprs = {str(n): str(e) for n, e in expressions.items()}
        if not exprs:
            raise ValueError("Target.expression needs at least one formula")

        def build(experiment, theta, lay: _Layout):
            from discopt.doe.symbolic import _require_sympy, parse_expression

            sp = _require_sympy()
            scalars = [p for (p, a, b) in lay.blocks if b - a == 1]
            subs = {sp.Symbol(p, real=True): float(lay.values[lay.index(p)[0]]) for p in scalars}
            K = np.zeros((len(lay.names), len(exprs)))
            values = []
            for j, src in enumerate(exprs.values()):
                expr = parse_expression(src, scalars)
                for p in scalars:
                    K[lay.index(p)[0], j] = float(sp.diff(expr, sp.Symbol(p, real=True)).subs(subs))
                values.append(float(expr.subs(subs)))
            names = list(exprs)
            scale = np.abs(np.asarray(values)) if relative else None
            if relative and np.any(~(scale > 0)):
                raise ValueError("relative=True needs every formula to be nonzero at the nominals")
            w, tau = _weights_and_tolerances(names, names, weights, tolerances, scale)
            return K, names, w, tau

        return cls(
            (
                _Block(
                    _FUNCTION_SPACE,
                    relative or tolerances is not None,
                    build,
                    f"expressions {list(exprs)}",
                ),
            )
        )

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Target:
        """Build a target from a JSON-able description (what workbooks store).

        ``spec`` has any of the keys ``parameters`` (name -> weight),
        ``expressions`` (name -> formula), ``responses`` (response name ->
        list of operating points, each a mapping of design inputs; an empty
        list for a model without inputs), ``tolerances`` (target name -> τ;
        when given, every target needs one) and ``relative`` (bool). Use
        :func:`validate_target_spec` to check it first.
        """
        spec = validate_target_spec(spec)
        tol = spec.get("tolerances") or None
        rel = bool(spec.get("relative", False))

        def pick(keys):
            return None if tol is None else {k: tol[k] for k in keys}

        parts: list[Target] = []
        if spec.get("parameters"):
            p = spec["parameters"]
            if tol is None:
                parts.append(cls.parameters(p, relative=rel))
            else:
                parts.append(cls.parameters(list(p), relative=rel, tolerances=pick(p)))
        if spec.get("expressions"):
            e = spec["expressions"]
            parts.append(cls.expression(e, relative=rel, tolerances=pick(e)))
        for r, pts in (spec.get("responses") or {}).items():
            parts.append(
                cls.responses(
                    at=[dict(x) for x in pts] or None,
                    responses=[r],
                    relative=rel,
                    tolerances=pick([r]),
                )
            )
        out = parts[0]
        for part in parts[1:]:
            out = out + part
        return out

    @classmethod
    def region(
        cls,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        *,
        points: Sequence[Mapping[str, float]] | None = None,
        n_points: int = 256,
        seed: int = 0,
    ) -> Target:
        """Predictions of the designed experiment over a region (I / G criteria).

        The targets are the experiment's responses at every region point,
        weighted by the region's integration weights (scaled to mean 1), so ``"average_variance"``
        on this target is the I-criterion and ``"max_variance"`` the
        G-criterion of :func:`~discopt.doe.experiment_region`. Useful to
        combine prediction quality with other targets.
        """

        def build(experiment, theta, lay: _Layout):
            from discopt.doe.design import experiment_region

            reg = experiment_region(
                experiment, theta, bounds=bounds, points=points, n_points=n_points, seed=seed
            )
            K = np.asarray(reg.moment_rows, dtype=float).T
            names = [f"region[{i}]" for i in range(K.shape[1])]
            # Integration weights scaled to mean 1: "average_variance" is then
            # the weighted mean (I) and "max_variance" the plain maximum (G).
            w = np.asarray(reg.weights, dtype=float)
            return K, names, w / np.mean(w), np.full(K.shape[1], np.nan)

        return cls((_Block(_OUTPUT_SPACE, False, build, "region"),))

    # -- algebra ------------------------------------------------------------

    def __add__(self, other: Target) -> Target:
        if not isinstance(other, Target):
            return NotImplemented
        combined = Target(self.blocks + other.blocks)
        families = {b.family for b in combined.blocks}
        if len(families) > 1 and not all(b.commensurable for b in combined.blocks):
            loose = [b.label for b in combined.blocks if not b.commensurable]
            raise ValueError(
                "targets of different kinds (parameters, outputs, functions) are in different "
                "units, so their variances cannot be added as given. Give each of "
                f"{loose} tolerances (or relative=True) to make them comparable."
            )
        return combined

    # -- resolution ---------------------------------------------------------

    def resolve(self, experiment: Any, param_values: Mapping[str, Any]) -> ResolvedTarget:
        """``K``, weights and names at the nominal parameters ``param_values``."""
        from discopt.doe.fim import ParameterScaledExperiment

        if not self.blocks:
            raise ValueError("empty Target")
        if isinstance(experiment, ParameterScaledExperiment):
            # K is always in the model's own parameter units; the design
            # functions convert it to the FIM's coordinates.
            experiment = experiment.experiment
        lay = _layout(experiment, param_values)
        Ks, names, ws, taus = [], [], [], []
        for b in self.blocks:
            K, n, w, tau = b.build(experiment, param_values, lay)
            Ks.append(K)
            names.extend(n)
            ws.append(w)
            taus.append(tau)
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"target names {dup} appear more than once")
        K = np.concatenate(Ks, axis=1)
        w = np.concatenate(ws)
        tau = np.concatenate(taus)
        keep = w > 0
        if not np.any(keep):
            raise ValueError("every target has weight 0")
        if not np.all(np.isfinite(K[:, keep])):
            raise ValueError("the target sensitivities are not finite at the nominal parameters")
        return ResolvedTarget(
            K=K[:, keep],
            weights=w[keep],
            names=[n for n, k in zip(names, keep) if k],
            parameter_names=list(lay.names),
            tolerances=tau[keep],
        )


# ---------------------------------------------------------------------------
# Resolved target: the numerical object the design searches evaluate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTarget:
    """A target at fixed nominal parameters: ``K`` (``p × s``) and weights ``w``.

    Created by :meth:`Target.resolve`, or directly from a matrix. Passed in
    place of a prediction region inside the design searches (it answers
    ``is_target``), so every search strategy evaluates the targeted criterion.
    """

    K: np.ndarray
    weights: np.ndarray
    names: list[str]
    parameter_names: list[str]
    tolerances: np.ndarray | None = None

    is_target = True

    def __post_init__(self) -> None:
        K = np.atleast_2d(np.asarray(self.K, dtype=float))
        w = np.asarray(self.weights, dtype=float).ravel()
        if K.shape != (len(self.parameter_names), len(self.names)) or w.shape != (len(self.names),):
            raise ValueError(
                f"K must be (n_parameters, n_targets) = "
                f"({len(self.parameter_names)}, {len(self.names)}), got {K.shape}; "
                f"weights {w.shape}"
            )
        if np.any(~(w > 0)) or not np.all(np.isfinite(w)):
            raise ValueError("target weights must be finite and > 0")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "weights", w)
        tau = self.tolerances
        tau = np.full(len(w), np.nan) if tau is None else np.asarray(tau, dtype=float).ravel()
        object.__setattr__(self, "tolerances", tau)

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        parameter_names: Sequence[str],
        names: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
    ) -> ResolvedTarget:
        """A target straight from a ``p × s`` matrix (``ψ = Kᵀθ`` locally)."""
        K = np.atleast_2d(np.asarray(K, dtype=float))
        if K.shape[0] != len(parameter_names) and K.shape[1] == len(parameter_names):
            K = K.T
        s = K.shape[1]
        return cls(
            K=K,
            weights=np.ones(s) if weights is None else np.asarray(weights, dtype=float),
            names=[f"psi[{j}]" for j in range(s)] if names is None else list(names),
            parameter_names=list(parameter_names),
        )

    @property
    def n_targets(self) -> int:
        return len(self.names)

    @property
    def full_rank(self) -> bool:
        """The targets are linearly independent (needed by D / E criteria)."""
        return bool(np.linalg.matrix_rank(self.K) == self.n_targets)

    @property
    def uniform_weights(self) -> bool:
        return bool(np.allclose(self.weights, self.weights[0]))

    # -- linear algebra -----------------------------------------------------

    def covariance(self, fim: np.ndarray) -> tuple[np.ndarray, bool]:
        """``(Σ_ψ, estimable)``: the target covariance ``Kᵀ M⁺ K``.

        Computed on the correlation form of ``M`` so parameters in very
        different units do not look singular. Directions of ``M`` below
        ``λ_max · 1e-12`` are unidentified; when a target has a component
        there it is not estimable, and ``Σ_ψ`` is returned as ``inf``.
        """
        M = np.asarray(fim, dtype=float)
        s = self.n_targets
        if M.shape != (self.K.shape[0],) * 2:
            raise ValueError(f"FIM is {M.shape}, the target expects {self.K.shape[0]} parameters")
        if not np.all(np.isfinite(M)):
            return np.full((s, s), np.inf), False
        M = 0.5 * (M + M.T)
        d = np.diag(M).copy()
        informed = d > 0
        K = self.K
        if np.any(np.abs(K[~informed]) > 0):
            return np.full((s, s), np.inf), False
        Mi = M[np.ix_(informed, informed)]
        scale = 1.0 / np.sqrt(d[informed])
        Kc = K[informed] * scale[:, None]
        Mc = Mi * scale[:, None] * scale[None, :]
        lam, V = np.linalg.eigh(Mc)
        if lam.size == 0 or lam[-1] <= 0:
            return np.full((s, s), np.inf), False
        keep = lam > lam[-1] * _RANK_RTOL
        Vr = V[:, keep]
        proj = Vr.T @ Kc
        resid = Kc - Vr @ proj
        norms = np.linalg.norm(Kc, axis=0)
        if np.any(np.linalg.norm(resid, axis=0) > _ESTIMABLE_RTOL * np.maximum(norms, 1e-300)):
            return np.full((s, s), np.inf), False
        sigma = (proj / lam[keep][:, None]).T @ proj
        return 0.5 * (sigma + sigma.T), True

    def _weighted(self, sigma: np.ndarray) -> np.ndarray:
        r = np.sqrt(self.weights)
        return sigma * r[:, None] * r[None, :]

    def information(self, fim: np.ndarray) -> np.ndarray | None:
        """``C = (W^½ Σ_ψ W^½)⁻¹``, or ``None`` if not estimable / singular."""
        sigma, ok = self.covariance(fim)
        if not ok:
            return None
        try:
            C = np.linalg.inv(self._weighted(sigma))
        except np.linalg.LinAlgError:
            return None
        return 0.5 * (C + C.T) if np.all(np.isfinite(C)) else None

    def criterion(self, fim: np.ndarray, criterion: str, *, smooth: bool = False) -> float:
        """Value of a targeted criterion (same sense as the untargeted one).

        ``smooth=True`` is for local search: the soft maximum for
        ``"max_variance"``, and a ridge far below the FIM's own scale. On a
        singular FIM a target can be estimable yet the exact criterion is
        discontinuous -- one more run helps only at exactly the right point --
        so a gradient search would never find it; the ridge makes the surface
        continuous and leads the search there. Report the exact value.
        """
        if criterion not in TARGET_CRITERIA:
            raise ValueError(f"criterion {criterion!r} is not available with a target")
        maximize = criterion in ("determinant", "min_eigenvalue")
        bad = -np.inf if maximize else np.inf
        if smooth:
            fim = np.asarray(fim, dtype=float)
            d = np.abs(np.diag(fim))
            floor = float(d.max()) if d.size and d.max() > 0 else 1.0
            fim = fim + np.diag(_SMOOTH_RIDGE * np.where(d > 0, d, floor))
        sigma, ok = self.covariance(fim)
        if not ok:
            return bad
        if criterion in ("trace", "average_variance"):
            val = float(np.sum(self.weights * np.diag(sigma)))
            if criterion == "average_variance":
                val /= float(np.sum(self.weights))
            return val if np.isfinite(val) else bad
        if criterion == "max_variance":
            v = self.weights * np.diag(sigma)
            if not np.all(np.isfinite(v)) or np.min(v) < 0:
                return bad
            vmax = float(np.max(v))
            if smooth and vmax > 0:
                q = 32.0
                return vmax * float(np.mean((v / vmax) ** q)) ** (1.0 / q)
            return vmax
        ev = np.linalg.eigvalsh(self._weighted(sigma))
        if not np.all(np.isfinite(ev)) or ev[0] <= 0:
            return bad
        if criterion == "determinant":
            return float(-np.sum(np.log(ev)))
        if criterion == "min_eigenvalue":
            return float(1.0 / ev[-1])
        return float(ev[-1] / ev[0])

    def precision(self, fim: np.ndarray) -> TargetPrecision:
        """Predicted standard errors of the targets under ``fim``."""
        sigma, ok = self.covariance(fim)
        return TargetPrecision(
            names=list(self.names),
            covariance=sigma,
            tolerances=np.asarray(self.tolerances, dtype=float),
            estimable=ok,
        )


@dataclass(frozen=True)
class TargetPrecision:
    """Predicted precision of the targets.

    Attributes
    ----------
    names : list[str]
        One per target.
    covariance : numpy.ndarray
        ``Σ_ψ = Kᵀ M⁺ K`` (``inf`` when not estimable).
    tolerances : numpy.ndarray
        Absolute tolerance per target, ``nan`` where none was given.
    estimable : bool
        Whether the FIM identifies every target.
    """

    names: list[str]
    covariance: np.ndarray
    tolerances: np.ndarray
    estimable: bool

    @property
    def std(self) -> dict[str, float]:
        """Standard error of each target."""
        return dict(zip(self.names, np.sqrt(np.abs(np.diag(self.covariance))).tolist()))

    @property
    def ratios(self) -> dict[str, float]:
        """``σ_j / τ_j`` for each target with a tolerance."""
        se = np.sqrt(np.abs(np.diag(self.covariance)))
        return {
            n: float(s / t) for n, s, t in zip(self.names, se, self.tolerances) if np.isfinite(t)
        }

    @property
    def met(self) -> bool | None:
        """Every toleranced target is within tolerance (``None``: no tolerances)."""
        r = self.ratios
        if not r:
            return None
        return bool(self.estimable and all(v <= 1.0 for v in r.values()))

    def summary(self) -> str:
        lines = ["Target precision" + ("" if self.estimable else " (NOT estimable)")]
        ratios = self.ratios
        for n, s in self.std.items():
            extra = f"   σ/τ = {ratios[n]:.3g}" if n in ratios else ""
            lines.append(f"  SE({n}) = {s:.4g}{extra}")
        if self.met is not None:
            lines.append(f"  all within tolerance: {self.met}")
        return "\n".join(lines)


_SPEC_KEYS = ("parameters", "expressions", "responses", "tolerances", "relative")


def validate_target_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Check a target description (see :meth:`Target.from_spec`); return a clean copy.

    Raises ``ValueError`` with a message naming the problem: an unknown key, a
    negative weight, a non-positive tolerance, a target name used twice, or
    tolerances that do not cover every target.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("a target spec must be a mapping")
    unknown = sorted(set(spec) - set(_SPEC_KEYS))
    if unknown:
        raise ValueError(f"unknown target spec key(s) {unknown}; allowed: {list(_SPEC_KEYS)}")
    params = {str(k): float(v) for k, v in (spec.get("parameters") or {}).items()}
    _check_positive(params, "weight")
    exprs = {str(k): str(v) for k, v in (spec.get("expressions") or {}).items()}
    responses: dict[str, list[dict[str, float]]] = {}
    for r, pts in (spec.get("responses") or {}).items():
        responses[str(r)] = [{str(k): float(v) for k, v in dict(pt).items()} for pt in pts or []]
    names = list(params) + list(exprs) + list(responses)
    if not names:
        raise ValueError("the target spec names no parameter, expression or response")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"target name(s) {dup} used more than once")
    out: dict[str, Any] = {"relative": bool(spec.get("relative", False))}
    if params:
        out["parameters"] = params
    if exprs:
        out["expressions"] = exprs
    if responses:
        out["responses"] = responses
    tol = spec.get("tolerances")
    if tol:
        tol = {str(k): float(v) for k, v in tol.items()}
        bad = sorted(k for k, v in tol.items() if not (np.isfinite(v) and v > 0))
        if bad:
            raise ValueError(f"tolerances must be finite and > 0 (got {bad})")
        extra = sorted(set(tol) - set(names))
        if extra:
            raise ValueError(f"tolerance(s) for {extra}, which are not targets")
        missing = sorted(set(names) - set(tol))
        if missing:
            raise ValueError(
                f"tolerances were given, so every target needs one; missing for {missing}"
            )
        if any(w != 1.0 for w in params.values()):
            raise ValueError(
                "give parameter weights or tolerances, not both (a tolerance τ is a weight 1/τ²)"
            )
        out["tolerances"] = tol
    return out


def basis_target(
    spec: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    basis: Callable[[np.ndarray], np.ndarray],
    param_values: Mapping[str, float] | None = None,
    response_name: str = "y",
) -> ResolvedTarget:
    """Resolve a target spec on a basis model, with no Experiment and no JAX.

    For the jax-free design path (:func:`~discopt.doe.linear_design.batch_design_from_basis`):
    ``basis(x)`` returns the Jacobian row ``∂y/∂θ`` at the input vector ``x``
    (ordered by ``input_names``) -- the regression basis of a linear model, or
    :func:`~discopt.doe.symbolic.basis_evaluator` for a nonlinear one. A
    response target's column of ``K`` is that row at its operating point; a
    formula's column is its sympy gradient at ``param_values`` (default 0 for
    every parameter, which is the right point for a linear model).
    """
    spec = validate_target_spec(spec)
    names_p = list(parameter_names)
    theta = {n: float((param_values or {}).get(n, 0.0)) for n in names_p}
    rel = bool(spec.get("relative", False))
    tol = spec.get("tolerances")
    cols: list[np.ndarray] = []
    names: list[str] = []
    keys: list[str] = []
    values: list[float] = []
    for pname in spec.get("parameters", {}):
        if pname not in names_p:
            raise ValueError(f"{pname!r} is not a parameter; the parameters are {names_p}")
        col = np.zeros(len(names_p))
        col[names_p.index(pname)] = 1.0
        cols.append(col)
        names.append(pname)
        keys.append(pname)
        values.append(theta[pname])
    if spec.get("expressions"):
        from discopt.doe.symbolic import _require_sympy, parse_expression

        sp = _require_sympy()
        subs = {sp.Symbol(n, real=True): v for n, v in theta.items()}
        for ename, src in spec["expressions"].items():
            expr = parse_expression(src, names_p)
            cols.append(
                np.array(
                    [float(sp.diff(expr, sp.Symbol(n, real=True)).subs(subs)) for n in names_p]
                )
            )
            names.append(ename)
            keys.append(ename)
            values.append(float(expr.subs(subs)))
    for rname, points in (spec.get("responses") or {}).items():
        if rname != response_name:
            raise ValueError(
                f"unknown response {rname!r}; the model's response is {response_name!r}"
            )
        for pt in points or [{}]:
            missing = [n for n in input_names if n not in pt]
            if missing:
                raise ValueError(f"operating point {pt} is missing inputs {missing}")
            row = np.asarray(basis(np.array([pt[n] for n in input_names], dtype=float)), float)
            cols.append(row.ravel())
            where = ",".join(f"{k}={v:g}" for k, v in pt.items())
            names.append(f"{rname}@{where}" if len(points) > 1 else rname)
            keys.append(rname)
            values.append(float(row.ravel() @ np.array([theta[n] for n in names_p])))
    K = np.stack(cols, axis=1)
    scale = np.abs(np.asarray(values)) if rel else None
    if rel and np.any(~(scale > 0)):
        raise ValueError(
            "relative precision needs every target to be nonzero at the nominal parameters"
        )
    if tol is not None:
        w, tau = _weights_and_tolerances(names, keys, None, tol, scale)
    else:
        weights = {k: float(spec.get("parameters", {}).get(k, 1.0)) for k in keys}
        w, tau = _weights_and_tolerances(names, keys, weights, None, scale)
    keep = w > 0
    if not np.any(keep):
        raise ValueError("every target has weight 0")
    return ResolvedTarget(
        K=K[:, keep],
        weights=w[keep],
        names=[n for n, k in zip(names, keep) if k],
        parameter_names=names_p,
        tolerances=tau[keep],
    )


def target_precision(
    fim: np.ndarray,
    target: Target | ResolvedTarget,
    experiment: Any = None,
    param_values: Mapping[str, Any] | None = None,
) -> TargetPrecision:
    """Predicted standard errors of ``target`` under ``fim``.

    A :class:`Target` is resolved first, which needs ``experiment`` and
    ``param_values``; a :class:`ResolvedTarget` is used as is.
    """
    if isinstance(target, Target):
        if experiment is None or param_values is None:
            raise ValueError("resolving a Target needs experiment and param_values")
        target = target.resolve(experiment, param_values)
    return target.precision(fim)


def _check_target_criterion(target: ResolvedTarget, criterion: str) -> None:
    """Refuse criteria the target cannot support; warn when weights do nothing."""
    if criterion not in TARGET_CRITERIA:
        raise ValueError(
            f"criterion {criterion!r} is not available with a target; use one of {TARGET_CRITERIA}"
        )
    if criterion in _INFORMATION_CRITERIA and not target.full_rank:
        raise ValueError(
            f"criterion {criterion!r} needs linearly independent targets, but these "
            f"{target.n_targets} targets span only {np.linalg.matrix_rank(target.K)} "
            "parameter directions (e.g. more output targets than parameters). Use "
            "'trace' or 'max_variance'."
        )
    if criterion == "determinant" and not target.uniform_weights:
        warnings.warn(
            "target weights do not change a 'determinant' design (they shift log det by a "
            "constant); only which targets are included matters. Use 'trace', "
            "'min_eigenvalue' or 'max_variance' for weights to take effect.",
            stacklevel=3,
        )
