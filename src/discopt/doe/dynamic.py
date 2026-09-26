"""discopt.doe.dynamic -- dynamic (ODE) experiments for model-based design.

Most kinetic, biological and process models are systems of ordinary
differential equations: a state trajectory ``x(t)`` from ``dx/dt = f(t, x; θ, u)``,
measured at a few sampling times. :func:`ode_experiment` turns such a model into
an ordinary :class:`~discopt.estimate.Experiment`, so everything in
:mod:`discopt.doe` that takes an experiment works on it unchanged:
:func:`~discopt.doe.compute_fim`, :func:`~discopt.doe.optimal_experiment` and
:func:`~discopt.doe.batch_optimal_experiment` (designing sampling times,
initial conditions and operating inputs), :func:`~discopt.doe.diagnose_identifiability`,
:func:`~discopt.doe.estimability_rank`, :func:`~discopt.doe.profile_likelihood`,
:func:`~discopt.doe.sequential_doe` and the discrimination tools.

How it works
------------
The ODE is integrated by a fixed-step scheme written in ``jax.numpy`` and
wrapped as a single opaque :func:`discopt.modeling.custom` node whose inputs
are the unknown parameters and the design inputs. The responses are therefore
*explicit, differentiable* functions of ``(θ, u, t_sample)``: exact
sensitivities ``∂y/∂θ`` come from autodiff through the integrator (no finite
differences, no sensitivity equations to write), and a sampling time can be a
continuous design variable because the step size is ``(t_j - t0) / n_steps``.

Two schemes are offered: classical fourth-order Runge-Kutta (``"rk4"``, the
default) and the implicit trapezoidal rule (``"trapezoid"``, for stiff systems;
each step solves its implicit equation with a few Newton iterations, unrolled
so every autodiff mode works). The trapezoidal rule is A-stable, so it never
blows up, but it is not L-stable: a very fast mode is damped slowly and
oscillates in sign, so resolve it (or its equilibrium) with enough steps.
Neither scheme adapts its step: choose ``n_steps`` so the answer no longer
changes when you double it.

A collocation transcription (:mod:`discopt.dae`) is *not* used here on purpose:
there the trajectory is a set of equality-constrained variables, so every FIM
evaluation needs a solve and an implicit-function-theorem correction (which
:func:`~discopt.doe.compute_fim` applies), where this module needs one compiled
Jacobian evaluation.

Example
-------
>>> exp = ode_experiment(
...     lambda t, x, p, u: {"A": -p["k"] * x["A"]},
...     states={"A": 1.0},
...     parameters={"k": (0.5, 1e-6, 10.0)},
...     measured=["A"],
...     sample_times=["t1"],
...     design_inputs={"t1": (0.05, 10.0)},
...     measurement_error=0.01,
... )
>>> r = compute_fim(exp, {"k": 0.5}, {"t1": 2.0})  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discopt.estimate import EstimationResult, Experiment, ExperimentModel

__all__ = ["ODEExperiment", "ode_experiment"]

_METHODS = ("rk4", "trapezoid")


def _jax():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


@dataclass
class ODEExperiment(Experiment):
    """An ODE model measured at sampling times, usable anywhere an Experiment is.

    Build one with :func:`ode_experiment`. Besides :meth:`create_model` (the
    :class:`~discopt.estimate.Experiment` interface) it offers:

    * :meth:`predict` and :meth:`simulate` for evaluating the model directly;
    * :meth:`jacobian` and :meth:`fim`, the sensitivity matrix and Fisher
      information from a compiled Jacobian (a fraction of a millisecond per
      call after the first), for custom design searches;
    * :meth:`check_accuracy`, which compares ``n_steps`` against twice as many;
    * :meth:`estimate`, a least-squares fit with the exact autodiff Jacobian
      that the discopt-doe loops use in place of the NLP-based estimator.

    Attributes
    ----------
    response_names : list of str
        One response per (sampling time, measured quantity), named
        ``"<measured>@<time>"``, e.g. ``"A@2"`` for a fixed time or ``"A@t1"``
        for a sampling time that is the design input ``t1``.
    response_times : dict
        Response name -> the sampling time (a float, or a design-input name).
    """

    rhs: Callable
    state_names: tuple[str, ...]
    initial: dict[str, float | str]
    parameter_specs: dict[str, tuple[float, float, float]]
    measured: dict[str, Callable | str]
    sample_times: tuple[float | str, ...]
    design_bounds: dict[str, tuple[float, float]]
    sigma: dict[str, float]
    n_steps: int = 50
    method: str = "rk4"
    t0: float = 0.0
    name: str = "ode"
    breakpoints: tuple[float, ...] = ()
    response_names: list[str] = field(init=False)
    response_times: dict[str, float | str] = field(init=False)

    def __post_init__(self) -> None:
        names, times = [], {}
        for t in self.sample_times:
            label = t if isinstance(t, str) else f"{float(t):g}"
            for m in self.measured:
                rn = f"{m}@{label}"
                if rn in times:
                    raise ValueError(f"duplicate response {rn!r}: sampling times must be distinct")
                names.append(rn)
                times[rn] = t
        self.response_names = names
        self.response_times = times

    # ------------------------------------------------------------------
    # The integrator
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameter_specs)

    @property
    def design_names(self) -> list[str]:
        return list(self.design_bounds)

    def _as_dict(self, names: Sequence[str], values: Sequence) -> dict[str, Any]:
        return dict(zip(names, values))

    def _rhs_vec(self, t, x, p: dict, u: dict):
        _, jnp = _jax()
        xs = self._as_dict(self.state_names, [x[i] for i in range(len(self.state_names))])
        out = self.rhs(t, xs, p, u)
        if isinstance(out, Mapping):
            missing = [s for s in self.state_names if s not in out]
            if missing:
                raise ValueError(f"rhs did not return derivatives for states {missing}")
            return jnp.stack([jnp.asarray(out[s], dtype=float) for s in self.state_names])
        return jnp.stack([jnp.asarray(v, dtype=float) for v in out])

    def _integrate(self, x0, t_end, p: dict, u: dict):
        """State at ``t_end`` from ``x0`` at ``t0``.

        ``n_steps`` fixed steps, per segment when there are ``breakpoints``:
        the interval is split at every breakpoint so no step straddles a jump
        in the right-hand side (a piecewise-constant temperature or feed),
        which would cut RK4 to first order. Segments past ``t_end`` have zero
        length and are no-ops, so ``t_end`` may be a traced design input. All
        segments run in one ``lax.scan`` over precomputed step times, so the
        integrator compiles once however many breakpoints there are.
        """
        jax, jnp = _jax()
        n = int(self.n_steps)
        t_end = jnp.asarray(t_end, dtype=float)
        t0 = jnp.asarray(self.t0, dtype=float)
        clamp = bool(self.breakpoints)
        edges = jnp.stack(
            [t0] + [jnp.clip(float(b), self.t0, t_end) for b in self.breakpoints] + [t_end]
        )
        lo, hi = edges[:-1], edges[1:]  # one row per segment
        h_seg = (hi - lo) / n
        k = jnp.arange(n, dtype=float)
        # Per step: its start time, size, and the segment it belongs to.
        t_steps = (lo[:, None] + k[None, :] * h_seg[:, None]).ravel()
        h_steps = jnp.repeat(h_seg, n)
        lo_steps, hi_steps = jnp.repeat(lo, n), jnp.repeat(hi, n)

        def rhs(t, x, a, b):
            if clamp:
                # Evaluate strictly inside the segment: at a breakpoint a
                # piecewise input is ambiguous (which piece owns the jump is
                # the model's convention), and either answer would put the
                # neighbouring piece into this segment.
                eps = 1e-12 * jnp.maximum(b - a, 1e-300)
                t = jnp.clip(t, a + eps, b - eps)
            return self._rhs_vec(t, x, p, u)

        if self.method == "rk4":

            def step(x, inp):
                t, h, a, b = inp
                k1 = rhs(t, x, a, b)
                k2 = rhs(t + h / 2, x + h / 2 * k1, a, b)
                k3 = rhs(t + h / 2, x + h / 2 * k2, a, b)
                k4 = rhs(t + h, x + h * k3, a, b)
                return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        else:  # implicit trapezoid, Newton iterations unrolled (all AD modes work)

            def step(x, inp):
                t, h, a, b = inp
                fx = rhs(t, x, a, b)
                y = x + h * fx  # explicit Euler predictor
                for _it in range(4):
                    g = y - x - h / 2 * (fx + rhs(t + h, y, a, b))
                    jac = jnp.eye(x.size) - h / 2 * jax.jacfwd(lambda z: rhs(t + h, z, a, b))(y)
                    y = y - jnp.linalg.solve(jac, g)
                return y, None

        x_end, _ = jax.lax.scan(step, x0, (t_steps, h_steps, lo_steps, hi_steps))
        return x_end

    def _measure(self, x, p: dict, u: dict):
        _, jnp = _jax()
        xs = self._as_dict(self.state_names, [x[i] for i in range(len(self.state_names))])
        vals = []
        for m, spec in self.measured.items():
            vals.append(xs[spec] if isinstance(spec, str) else spec(xs, p, u))
        return jnp.stack([jnp.asarray(v, dtype=float) for v in vals])

    def _response_fn(self, *args):
        """All responses, ordered as :attr:`response_names`, from (θ..., u...)."""
        _, jnp = _jax()
        k = len(self.parameter_specs)
        p = self._as_dict(self.parameter_names, args[:k])
        u = self._as_dict(self.design_names, args[k:])
        x0 = jnp.stack(
            [
                jnp.asarray(u[v] if isinstance(v, str) else v, dtype=float)
                for v in (self.initial[s] for s in self.state_names)
            ]
        )
        jax, _ = _jax()
        t_ends = jnp.stack(
            [jnp.asarray(u[t] if isinstance(t, str) else t, dtype=float) for t in self.sample_times]
        )
        # One integrator traced once and batched over the end times: each
        # response integrates from t0 with its own step (t_j - t0) / n_steps.
        states = jax.vmap(lambda te: self._integrate(x0, te, p, u))(t_ends)
        meas = jax.vmap(lambda x: self._measure(x, p, u))(states)
        return meas.reshape(-1)

    # ------------------------------------------------------------------
    # Experiment interface
    # ------------------------------------------------------------------

    def create_model(self, **kwargs) -> ExperimentModel:
        import discopt.modeling as dm

        m = dm.Model(self.name)
        params = {
            n: m.continuous(n, lb=lb, ub=ub) for n, (_, lb, ub) in self.parameter_specs.items()
        }
        inputs = {n: m.continuous(n, lb=lb, ub=ub) for n, (lb, ub) in self.design_bounds.items()}
        node = dm.custom(self._response_fn, name=self.name)(*params.values(), *inputs.values())
        responses = {rn: node[i] for i, rn in enumerate(self.response_names)}
        err = {rn: self.sigma[rn.split("@")[0]] for rn in self.response_names}
        return ExperimentModel(
            model=m,
            unknown_parameters=params,
            design_inputs=inputs,
            responses=responses,
            measurement_error=err,
        )

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def _args(self, theta: Mapping[str, float], design: Mapping[str, float] | None):
        design = dict(design or {})
        missing = [n for n in self.design_names if n not in design]
        if missing:
            raise ValueError(f"design values missing for {missing}")
        unknown = [n for n in design if n not in self.design_bounds]
        if unknown:
            raise ValueError(
                f"unknown design input(s) {unknown}; design inputs are {self.design_names}"
            )
        missing_p = [n for n in self.parameter_names if n not in theta]
        if missing_p:
            raise ValueError(f"parameter values missing for {missing_p}")
        return [float(theta[n]) for n in self.parameter_names] + [
            float(design[n]) for n in self.design_names
        ]

    def _compiled(self):
        """``(responses, jacobian)`` jitted over ``(theta_vector, design_vector)``.

        Compiled once per experiment state and reused; editing the experiment
        (e.g. ``n_steps``) triggers a recompile.
        """
        from discopt.doe.fim import _experiment_fingerprint

        jax, jnp = _jax()
        key = _experiment_fingerprint(self)
        store = self.__dict__.setdefault("_ode_compiled", {})
        if key not in store:
            store.clear()

            def resp(theta_vec, u_vec):
                return self._response_fn(*theta_vec, *u_vec)

            store[key] = (jax.jit(resp), jax.jit(jax.jacfwd(resp, argnums=0)))
        return store[key]

    def _vectors(self, theta, design):
        _, jnp = _jax()
        args = self._args(theta, design)
        k = len(self.parameter_specs)
        return jnp.asarray(args[:k], dtype=float), jnp.asarray(args[k:], dtype=float)

    @property
    def response_function(self):
        """The traceable response map ``f(θ_1, ..., θ_k, u_1, ..., u_m) -> y``.

        Pure ``jax.numpy``, so it can be composed, jitted or differentiated
        inside other JAX code (for example a campaign of several runs). Arguments
        are scalars in :attr:`parameter_names` then :attr:`design_names` order;
        the result is ordered as :attr:`response_names`. For one-off evaluations
        use :meth:`predict`, :meth:`jacobian` or :meth:`fim`, which are compiled.
        """
        return self._response_fn

    def predict(
        self, theta: Mapping[str, float], design: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Model responses at parameters ``theta`` and design ``design``."""
        resp, _ = self._compiled()
        y = np.asarray(resp(*self._vectors(theta, design)))
        return dict(zip(self.response_names, map(float, y)))

    def jacobian(
        self, theta: Mapping[str, float], design: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Sensitivity matrix ``∂y/∂θ``: rows :attr:`response_names`, columns parameters.

        Uses a Jacobian compiled once per experiment, so repeated calls at new
        parameter values or designs cost a compiled evaluation. Every design
        input must be given.
        """
        _, jac = self._compiled()
        return np.asarray(jac(*self._vectors(theta, design)))

    def fim(
        self,
        theta: Mapping[str, float],
        design: Mapping[str, float] | None = None,
        *,
        prior_fim: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fisher information ``Jᵀ Σ⁻¹ J`` (plus ``prior_fim``), parameters in order.

        Identical to ``compute_fim(self, theta, design).fim``, from the same
        compiled Jacobian as :meth:`jacobian`. Sum it over independent runs for
        a batch, which makes custom design searches (robust criteria, joint
        searches over many runs) fast.
        """
        J = self.jacobian(theta, design)
        w = np.array([1.0 / self.sigma[rn.split("@")[0]] ** 2 for rn in self.response_names])
        F = J.T @ (w[:, None] * J)
        return F + prior_fim if prior_fim is not None else F

    def check_accuracy(
        self,
        theta: Mapping[str, float],
        design: Mapping[str, float] | None = None,
        *,
        rtol: float = 1e-3,
        warn: bool = True,
    ) -> float:
        """Largest change in the responses and sensitivities when ``n_steps`` doubles.

        A fixed-step integrator is only as accurate as its step. This recomputes
        the responses and the Jacobian with ``2 * n_steps`` and returns the
        largest change relative to the largest magnitude of each. When it
        exceeds ``rtol`` (and ``warn``), a warning suggests more steps; a fast
        mode (a large rate constant, a stiff system) is the usual cause.
        """
        import dataclasses
        import warnings

        fine = dataclasses.replace(self, n_steps=2 * int(self.n_steps))
        change = 0.0
        for coarse_val, fine_val in (
            (
                np.array(list(self.predict(theta, design).values())),
                np.array(list(fine.predict(theta, design).values())),
            ),
            (self.jacobian(theta, design), fine.jacobian(theta, design)),
        ):
            scale = max(float(np.max(np.abs(fine_val))), 1e-300)
            change = max(change, float(np.max(np.abs(coarse_val - fine_val))) / scale)
        if warn and change > rtol:
            warnings.warn(
                f"n_steps={self.n_steps} is not converged: doubling it changes the responses "
                f"or sensitivities by {change:.2g} (relative), above rtol={rtol:g}. Increase "
                "n_steps, or use method='trapezoid' for a stiff system.",
                UserWarning,
                stacklevel=2,
            )
        return change

    def simulate(
        self,
        theta: Mapping[str, float],
        design: Mapping[str, float] | None = None,
        times: Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """State trajectories on a time grid (for plotting): ``{"t": ..., state: ...}``.

        Only the design inputs the trajectory actually uses must be given: a
        designed initial state, an input the ``rhs`` reads, and (when ``times``
        is omitted, so the grid runs to the last sampling time) the sampling-time
        inputs. A missing one raises a clear error naming it.
        """
        _, jnp = _jax()
        missing_p = [n for n in self.parameter_names if n not in theta]
        if missing_p:
            raise ValueError(f"parameter values missing for {missing_p}")
        given = dict(design or {})
        unknown = [n for n in given if n not in self.design_bounds]
        if unknown:
            raise ValueError(
                f"unknown design input(s) {unknown}; design inputs are {self.design_names}"
            )
        p = {n: float(theta[n]) for n in self.parameter_names}
        u = _NeededInputs({n: float(v) for n, v in given.items()})
        if times is None:
            needed = [t for t in self.sample_times if isinstance(t, str) and t not in u]
            if needed:
                raise ValueError(
                    f"pass times=..., or values for the sampling-time input(s) {needed} "
                    "(the default grid runs to the last sampling time)"
                )
            t_max = max(float(u[t]) if isinstance(t, str) else float(t) for t in self.sample_times)
            times = np.linspace(self.t0, t_max, 101)
        x0 = jnp.asarray(
            [
                float(u[v]) if isinstance(v, str) else float(v)
                for v in (self.initial[s] for s in self.state_names)
            ]
        )
        xs = np.array(
            [
                np.asarray(self._integrate(x0, float(t), p, u)) if t > self.t0 else np.asarray(x0)
                for t in times
            ]
        )
        out = {"t": np.asarray(times, dtype=float)}
        out.update({s: xs[:, i] for i, s in enumerate(self.state_names)})
        return out

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        data: Mapping[str, float | Sequence[float]],
        *,
        initial_guess: Mapping[str, float] | None = None,
        fixed_parameters: Mapping[str, float] | None = None,
        solver_options: Mapping[str, Any] | None = None,
        design: Mapping[str, float] | None = None,
    ) -> EstimationResult:
        """Weighted least-squares fit, returning a :class:`~discopt.estimate.EstimationResult`.

        Same contract as :func:`discopt.estimate.estimate_parameters`: ``data``
        maps response names to a value or an array of replicates, the returned
        ``objective`` is the deviance ``Σ ((y - ŷ)/σ)²``, and ``fim`` /
        ``covariance`` are at the estimate. Sampling times or inputs that are
        design inputs must be given in ``design``. The solve is
        ``scipy.optimize.least_squares`` with the exact autodiff Jacobian.
        """
        from scipy.optimize import least_squares

        jax, jnp = _jax()
        if self.design_names and not design:
            raise ValueError(
                f"this experiment has design inputs {self.design_names}; pass design=... "
                "with the conditions the data were measured at"
            )
        unknown = [k for k in data if k not in self.response_names]
        if unknown:
            raise ValueError(f"data keys {unknown} are not responses {self.response_names}")
        fixed = {k: float(v) for k, v in (fixed_parameters or {}).items()}
        for k in fixed:
            if k not in self.parameter_specs:
                raise KeyError(f"{k!r} is not an unknown parameter ({self.parameter_names})")
        free = [n for n in self.parameter_names if n not in fixed]
        guess = {n: float(spec[0]) for n, spec in self.parameter_specs.items()}
        guess.update({k: float(v) for k, v in (initial_guess or {}).items() if k in guess})
        guess.update(fixed)
        u = [float((design or {})[n]) for n in self.design_names]

        idx, obs, wts = [], [], []
        for i, rn in enumerate(self.response_names):
            if rn not in data:
                continue
            for v in np.atleast_1d(np.asarray(data[rn], dtype=float)).ravel():
                idx.append(i)
                obs.append(v)
                wts.append(1.0 / self.sigma[rn.split("@")[0]])
        if not idx:
            raise ValueError("No data matched any response names")
        idx_a, obs_a, w_a = jnp.asarray(idx), jnp.asarray(obs), jnp.asarray(wts)

        def full_theta(z):
            it = iter(z)
            return [next(it) if n in free else jnp.asarray(fixed[n]) for n in self.parameter_names]

        def resid(z):
            y = self._response_fn(*full_theta(z), *u)
            return (y[idx_a] - obs_a) * w_a

        res_j = jax.jit(resid)
        jac_j = jax.jit(jax.jacfwd(resid))
        lo = np.array([self.parameter_specs[n][1] for n in free], dtype=float)
        hi = np.array([self.parameter_specs[n][2] for n in free], dtype=float)
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
            z_hat, success, message = sol.x, bool(sol.success), str(sol.message)
        else:
            z_hat, success, message = np.array([]), True, "all parameters fixed"

        theta_hat = {n: float(v) for n, v in zip(free, z_hat)} | fixed
        theta_hat = {n: theta_hat[n] for n in self.parameter_names}
        r = np.asarray(res_j(jnp.asarray(z_hat)))

        # FIM over all parameters at the estimate (as the base estimator does).
        def all_resid(t_all):
            y = self._response_fn(*t_all, *u)
            return y[idx_a] * w_a

        J = np.asarray(
            jax.jacfwd(lambda t: all_resid(list(t)))(
                jnp.asarray([theta_hat[n] for n in self.parameter_names])
            )
        )
        fim = J.T @ J
        try:
            cov = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fim)
        return EstimationResult(
            parameters=theta_hat,
            covariance=cov,
            fim=fim,
            objective=float(r @ r),
            solve_result=_FitStatus(success=success, message=message),
            parameter_names=list(self.parameter_names),
            n_observations=len(idx),
        )


class _NeededInputs(dict):
    """Design-input values that name the missing input when one is read."""

    def __missing__(self, key):
        raise ValueError(
            f"design input {key!r} is needed for this simulation (an initial state or "
            "the rhs uses it); pass it in design=..."
        )


@dataclass
class _FitStatus:
    """Minimal stand-in for ``SolveResult`` on an :meth:`ODEExperiment.estimate` fit."""

    success: bool
    message: str

    @property
    def status(self) -> str:
        return "optimal" if self.success else "failed"


def ode_experiment(
    rhs: Callable,
    *,
    states: Mapping[str, float | str],
    parameters: Mapping[str, float | tuple[float, float, float]],
    measured: Sequence[str] | Mapping[str, Callable | str],
    sample_times: Sequence[float | str],
    design_inputs: Mapping[str, tuple[float, float]] | None = None,
    measurement_error: float | Mapping[str, float] = 1.0,
    n_steps: int = 50,
    method: str = "rk4",
    t0: float = 0.0,
    name: str = "ode",
    breakpoints: Sequence[float] = (),
) -> ODEExperiment:
    """Build an :class:`~discopt.estimate.Experiment` from an ODE model.

    Parameters
    ----------
    rhs : callable
        ``rhs(t, x, p, u) -> dict`` of time derivatives, one per state (or a
        sequence in state order). ``x``, ``p`` and ``u`` are dicts of states,
        parameters and design inputs. Write it with ``jax.numpy`` (or plain
        arithmetic) so it can be differentiated, e.g.
        ``lambda t, x, p, u: {"A": -p["k"] * jnp.exp(-p["E"] / u["T"]) * x["A"]}``.
    states : mapping name -> float or str
        Every state and its initial value. A string names a design input, so an
        initial concentration can be designed.
    parameters : mapping name -> nominal or (nominal, lower, upper)
        The unknown parameters. Bounds default to ``(0, +inf)`` for a positive
        nominal and ``(-inf, +inf)`` otherwise; estimation respects them.
    measured : sequence of state names, or mapping name -> state name or callable
        What is measured. A callable ``g(x, p, u)`` gives a derived quantity
        (a sum of states, a conversion, a signal with a response factor).
    sample_times : sequence of float or str
        When each measurement is taken. A string names a design input, which
        makes that sampling time something to optimize. Times need not be
        sorted; each response is integrated from ``t0`` on its own.
    design_inputs : mapping name -> (lower, upper), optional
        Every design input with its bounds: sampling times, designed initial
        conditions, and operating conditions used in ``rhs`` (temperature,
        feed rate, ...).
    measurement_error : float or mapping measured name -> float
        Standard deviation of each measured quantity.
    n_steps : int, default 50
        Fixed steps from ``t0`` to each sampling time.
    method : {"rk4", "trapezoid"}
        Explicit Runge-Kutta 4, or the A-stable implicit trapezoidal rule for
        stiff systems.
    t0 : float, default 0
        Start time of the experiment.
    name : str
        Model name.
    breakpoints : sequence of float, optional
        Times where the right-hand side jumps -- the switching times of a
        piecewise-constant temperature or feed profile. Integration is split
        there (``n_steps`` per segment), so no step straddles a jump. Without
        them RK4 drops to first order across each jump: for a 9-level
        temperature profile, 50 steps give a FIM ~3e-3 off, against 2e-8 with
        50 steps per segment.

    Returns
    -------
    ODEExperiment
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    design = {k: (float(v[0]), float(v[1])) for k, v in (design_inputs or {}).items()}
    for k, (lo, hi) in design.items():
        if not hi > lo:
            raise ValueError(f"design input {k!r}: upper bound must exceed lower bound")
    if not states:
        raise ValueError("an ODE experiment needs at least one state")
    for s, v in states.items():
        if isinstance(v, str) and v not in design:
            raise ValueError(f"initial value of {s!r} names {v!r}, which is not a design input")
    if not sample_times:
        raise ValueError("give at least one sampling time")
    for t in sample_times:
        if isinstance(t, str):
            if t not in design:
                raise ValueError(f"sampling time {t!r} is not a design input")
            if design[t][0] <= t0:
                raise ValueError(f"sampling time {t!r} must be bounded above t0={t0}")
        elif float(t) <= t0:
            raise ValueError(f"sampling time {t} must be after t0={t0}")

    bps = tuple(sorted(float(b) for b in breakpoints))
    if any(b <= t0 for b in bps):
        raise ValueError(f"breakpoints must be after t0={t0}")
    if len(set(bps)) != len(bps):
        raise ValueError("breakpoints must be distinct")

    specs: dict[str, tuple[float, float, float]] = {}
    for n, v in parameters.items():
        if isinstance(v, (tuple, list)):
            nom, lo, hi = (float(x) for x in v)
        else:
            nom = float(v)
            lo, hi = (0.0, np.inf) if nom > 0 else (-np.inf, np.inf)
        if not lo <= nom <= hi:
            raise ValueError(f"parameter {n!r}: nominal {nom} outside bounds ({lo}, {hi})")
        specs[n] = (nom, lo, hi)
    overlap = set(specs) & set(design)
    if overlap:
        raise ValueError(f"names used as both parameter and design input: {sorted(overlap)}")

    meas: dict[str, Callable | str]
    if isinstance(measured, Mapping):
        meas = dict(measured)
    else:
        meas = {m: m for m in measured}
    for m, spec in meas.items():
        if isinstance(spec, str) and spec not in states:
            raise ValueError(f"measured {m!r} refers to unknown state {spec!r}")
    if isinstance(measurement_error, Mapping):
        sigma = {m: float(measurement_error[m]) for m in meas}
    else:
        sigma = {m: float(measurement_error) for m in meas}
    if any(s <= 0 for s in sigma.values()):
        raise ValueError("measurement_error must be positive")

    return ODEExperiment(
        rhs=rhs,
        state_names=tuple(states),
        initial=dict(states),
        parameter_specs=specs,
        measured=meas,
        sample_times=tuple(sample_times),
        design_bounds=design,
        sigma=sigma,
        n_steps=int(n_steps),
        method=method,
        t0=float(t0),
        name=name,
        breakpoints=bps,
    )
