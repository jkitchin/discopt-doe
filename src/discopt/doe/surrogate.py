"""Surrogate-model protocol for active-learning optimization.

A surrogate is anything that, after seeing some ``(X, y)`` data, can
predict the response **with an uncertainty estimate** at new points.
That uncertainty is what acquisition functions like expected
improvement and UCB consume to decide where to run the next experiment.

Design philosophy
-----------------

There are three reasonable ways to hand a surrogate to
:func:`discopt.doe.optimize_round` (or :func:`discopt.doe.model_based_optimize_round`),
in increasing power:

1. **String preset** (zero knowledge required)::

       optimize_round(wb, surrogate="gp")

   Resolves to a sensible default (here, scikit-learn's
   ``GaussianProcessRegressor`` with a Matern(5/2) kernel + white
   noise). The string aliases live in :data:`PRESETS`.

2. **scikit-learn-compatible estimator** (auto-wrapped)::

       from sklearn.gaussian_process import GaussianProcessRegressor
       from sklearn.gaussian_process.kernels import Matern
       optimize_round(wb, surrogate=GaussianProcessRegressor(kernel=Matern(nu=2.5)))

       from pycse.sklearn.lpr import LinearLPR
       optimize_round(wb, surrogate=LinearLPR())

   Anything that quacks like sklearn (``fit``/``predict``) is wrapped
   in :class:`_SklearnUQAdapter`, which probes for UQ in this order:

   * ``predict(X, return_std=True)`` -- standard scikit-learn convention
     (GaussianProcessRegressor, BayesianRidge, ARDRegression, ...).
   * ``predict(X, return_interval=True)`` -- pycse linear local
     prediction regression returns a 95% interval; we convert to a
     pseudo-σ via ``(upper - lower) / (2 * 1.96)``.
   * Bootstrap residual fallback -- for plain regressors with no UQ,
     refit on resamples and take the per-point std of predictions.

3. **Custom object implementing the protocol** -- full escape hatch::

       class MyBespokeBayesian:
           def fit(self, X, y): ...; return self
           def predict(self, X): return mean, std
       optimize_round(wb, surrogate=MyBespokeBayesian())

For mechanistic models (you have ``y = f(d; θ)`` and want to fit ``θ``
between rounds), use :class:`discopt.doe.ParametricSurrogate` with
:func:`discopt.doe.model_based_optimize_round` instead.

You never need to subclass anything; the protocol is structural.

Conventions
-----------

* ``X`` is a 2D numpy array of shape ``(n_samples, n_features)``.
* ``y`` is a 1D numpy array of shape ``(n_samples,)``.
* ``predict(X)`` returns ``(mean, std)``, both 1D arrays of length
  ``n_samples``. ``std`` is non-negative; surrogates with no native
  UQ should not silently return zeros -- use the bootstrap adapter
  or raise.
* Optionally, ``predict_latent(X)`` returns ``(mean, std)`` where ``std``
  is the uncertainty of the *mean response* only, without the
  observation noise that ``predict`` includes. Acquisition functions use
  it when available: the chance that a new run *improves on the mean* is
  what matters for optimization, and a σ that includes noise never shrinks
  below the noise level, so expected improvement never decays.

Why the GP preset looks the way it does
---------------------------------------

The ``"gp"`` preset (:func:`gp_surrogate`) is tuned to give honest error
bars on small, noisy, unreplicated designs, where a plain maximum-likelihood
GP fails silently. With 20 runs the likelihood cannot tell noise from a
slightly wigglier surface; it drives the noise to ~0, interpolates the data,
and its 95% intervals cover the truth only ~79% of the time. The preset
therefore

* estimates the noise from **replicated runs** when the design has them
  (pure error, the classical answer), and otherwise fits it, never below
  10% of the response standard deviation; ``noise=0`` interpolates for
  deterministic simulators, and a float fixes the noise SD;
* fits the hyperparameters by **maximum a posteriori** with weak log-normal
  priors (``priors=True``). Maximum likelihood alone is fragile on small
  designs in two opposite ways. It collapses the noise to ~0 (the coverage
  failure above), and it can also do the reverse: with 8 runs spread
  between narrow features, "everything is noise" (noise SD = response SD, no
  signal) is the exact likelihood optimum, the surrogate goes flat, and
  Bayesian optimization degrades to random search. The priors, in the style of
  Hvarfner, Hellsten & Nardi (2024), are
  - on each length-scale, log-normal with median ``4.1 sqrt(d)`` times that
    input's span in the data and log-SD ``sqrt(3)`` (a preference for smooth
    surfaces that the data can override, scaled with the dimension so it
    stays weak in many dimensions);
  - on the noise variance, log-normal with median ``exp(-3) = 5%`` of the
    response variance and log-SD 1 (noise is expected to be a minority of the
    variation, as it is in a well-chosen experimental region).

  In the book's studies (Chapters 14-16) the priors keep held-out coverage
  of the 95% bands at ~0.97 on a 20-run unreplicated design, remove the
  all-noise fits, and cut Bayesian-optimization regret with 3 active factors
  among 12 from 0.74 to 0.18 over 30-run campaigns. The cost is
  overconfidence when the response really is mostly noise: coverage ~0.77
  when noise is two-thirds of the variation, ~0.70 for a response that is
  pure noise (maximum likelihood gets 0.82 and 0.76 there). Replicate a few
  runs in that situation; the noise is then taken from pure error;
* chooses between one shared length-scale and one per input (ARD) by the log
  posterior (``ard="auto"``). With the length-scale prior, ARD rarely
  overfits: it predicted better than a shared length-scale both with all six
  factors active and with inert ones, and "auto" mostly picks it. Without the
  priors (``priors=False``) the choice falls back to a BIC-penalized
  likelihood;
* floors the length-scales at 0.05 (in the standardized inputs
  :func:`~discopt.doe.optimize_round` uses) as a hard bound;
* is deterministic for a given ``random_state``.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Protocol, cast, runtime_checkable

import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Minimal interface every surrogate must satisfy.

    Implementations are expected to be *re-fittable*: each call to
    :meth:`fit` should reset the model to a fresh state trained on the
    new data, not incrementally update.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class _SklearnUQAdapter:
    """Adapt a scikit-learn-compatible estimator to the Surrogate protocol.

    The adapter probes for an uncertainty signal in this order:

    1. ``estimator.predict(X, return_std=True)`` -> ``(mean, std)``
    2. ``estimator.predict(X, return_interval=True)`` -> ``(mean, (lo, hi))``;
       converted to ``std = (hi - lo) / (2 * 1.96)`` (assumes 95%).
    3. Residual bootstrap: refit on ``n_bootstrap`` resamples, take the
       per-point std across predictions.

    The probe is done **once at fit time** and cached, so prediction
    is cheap.
    """

    def __init__(
        self,
        estimator,
        *,
        n_bootstrap: int = 32,
        random_state: int = 0,
        include_noise: bool = False,
    ):
        # Clone so fitting never mutates the caller's estimator (optimize_round
        # refits the surrogate each pick; a user-supplied instance would
        # otherwise end up trained on the last fantasy dataset). Fall back to the
        # original if clone is unavailable or the object isn't a sklearn
        # estimator (e.g. a bare Pipeline built here, which is safe to reuse).
        try:
            from sklearn.base import clone as _clone

            self._estimator = _clone(estimator)
        except Exception:  # noqa: BLE001
            self._estimator = estimator
        self._n_bootstrap = int(n_bootstrap)
        self._random_state = int(random_state)
        # Bootstrap spread is uncertainty about the fitted *mean*; a new
        # observation also carries the measurement noise. include_noise adds
        # the residual variance so predict() gives predictive intervals.
        self._include_noise = bool(include_noise)
        self._resid_var = 0.0
        self._mode: str | None = None
        self._bootstrap_models: list | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    @property
    def estimator(self):
        return self._estimator

    @property
    def mode(self) -> str | None:
        """One of ``"return_std"``, ``"return_interval"``, ``"bootstrap"``."""
        return self._mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnUQAdapter":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._X_train = X
        self._y_train = y
        self._estimator.fit(X, y)
        self._mode = self._probe_mode(X)
        if self._mode == "bootstrap":
            self._bootstrap_models = self._fit_bootstrap(X, y)
            resid = y - np.asarray(self._estimator.predict(X), dtype=float).ravel()
            self._resid_var = float(np.sum(resid**2) / max(len(y) - 1, 1))
        else:
            self._bootstrap_models = None
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if self._mode is None:
            raise RuntimeError("call fit() before predict()")
        if self._mode == "return_std":
            mean, std = self._estimator.predict(X, return_std=True)
            return np.asarray(mean, dtype=float).ravel(), np.asarray(std, dtype=float).ravel()
        if self._mode == "return_interval":
            out = self._estimator.predict(X, return_interval=True)
            mean, interval = out
            mean = np.asarray(mean, dtype=float).ravel()
            interval = np.asarray(interval, dtype=float)
            lo = interval[..., 0].ravel()
            hi = interval[..., 1].ravel()
            std = np.maximum(hi - lo, 0.0) / (2.0 * 1.959963984540054)
            return mean, std
        mean, std = self._bootstrap_predict(X)
        if self._include_noise:
            std = np.sqrt(std**2 + self._resid_var)
        return mean, std

    def predict_latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and the standard error of the mean response (no observation noise).

        For a scikit-learn GP whose kernel contains a ``WhiteKernel`` the fitted
        white-noise variance is removed from ``predict``'s std; the bootstrap
        spread is already a latent quantity. Estimators whose std has no known
        noise component are returned unchanged.
        """
        X = np.asarray(X, dtype=float)
        if self._mode == "bootstrap":
            return self._bootstrap_predict(X)
        mean, std = self.predict(X)
        noise_var = _white_noise_variance(self._estimator)
        if noise_var > 0.0:
            std = np.sqrt(np.clip(std**2 - noise_var, 0.0, None))
        return mean, std

    def _bootstrap_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._bootstrap_models is not None
        preds = np.stack([m.predict(X) for m in self._bootstrap_models], axis=0)
        mean = preds.mean(axis=0).ravel()
        std = preds.std(axis=0, ddof=1).ravel() if preds.shape[0] > 1 else np.zeros_like(mean)
        return mean, std

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _probe_mode(self, X: np.ndarray) -> str:
        sample = X[:1] if len(X) else X
        try:
            out = self._estimator.predict(sample, return_std=True)
        except TypeError:
            pass
        else:
            if isinstance(out, tuple) and len(out) == 2:
                return "return_std"
        try:
            out = self._estimator.predict(sample, return_interval=True)
        except TypeError:
            pass
        else:
            if isinstance(out, tuple) and len(out) == 2:
                interval = np.asarray(out[1])
                if interval.ndim >= 1 and interval.shape[-1] == 2:
                    return "return_interval"
        return "bootstrap"

    def _fit_bootstrap(self, X: np.ndarray, y: np.ndarray) -> list:
        from sklearn.base import clone

        rng = np.random.default_rng(self._random_state)
        n = len(X)
        models = []
        for _ in range(self._n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m = clone(self._estimator)
            m.fit(X[idx], y[idx])
            models.append(m)
        return models


def _require_sklearn() -> None:
    """Raise an actionable error when the optional ``ml`` extra is missing."""
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the built-in 'gp' and 'response-surface' surrogates require "
            "scikit-learn, which ships in the optional 'ml' extra. Install it "
            'with: pip install "discopt-doe[ml]"'
        ) from e


def _white_noise_variance(estimator: Any) -> float:
    """Fitted ``WhiteKernel`` variance of a scikit-learn GP, in response units (0 if none)."""
    kernel = getattr(estimator, "kernel_", None)
    if kernel is None:
        return 0.0
    try:
        from sklearn.gaussian_process.kernels import WhiteKernel
    except ImportError:  # pragma: no cover - sklearn is present if kernel_ exists
        return 0.0
    total = 0.0
    stack = [kernel]
    while stack:
        k = stack.pop()
        if isinstance(k, WhiteKernel):
            total += float(k.noise_level)
        for attr in ("k1", "k2"):
            child = getattr(k, attr, None)
            if child is not None and type(k).__name__ == "Sum":
                stack.append(child)
    if total and getattr(estimator, "normalize_y", False):
        y_std = np.asarray(getattr(estimator, "_y_train_std", 1.0), dtype=float)
        total *= float(np.mean(y_std**2))
    return total


def _pure_error_variance(X: np.ndarray, y: np.ndarray) -> tuple[float, int] | None:
    """Pooled within-group variance over exactly replicated rows, and its df."""
    keys = np.round(np.asarray(X, dtype=float), 12)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    df = int(np.sum(counts - 1))
    if df <= 0:
        return None
    means = np.bincount(inverse, weights=y) / counts
    ss = float(np.sum((y - means[inverse]) ** 2))
    return ss / df, df


class GPSurrogate:
    """Gaussian-process surrogate with calibrated defaults (the ``"gp"`` preset).

    A Matérn kernel times a constant, plus a white-noise term, fitted by
    scikit-learn with the response normalized. See the module docstring for
    why each default is what it is; build one with :func:`gp_surrogate`.

    Attributes (after ``fit``)
    --------------------------
    estimator : sklearn.gaussian_process.GaussianProcessRegressor
        The fitted regressor.
    ard_ : bool
        Whether the chosen model has one length-scale per input.
    length_scales_ : numpy.ndarray
        Fitted length-scale(s), in the units of the ``X`` passed to ``fit``.
    noise_sd_ : float
        Observation-noise standard deviation, in response units.
    noise_source_ : str
        ``"replicates"`` (pure error from replicated runs), ``"fitted"``,
        ``"fixed"`` (the ``noise`` argument) or ``"none"`` (interpolating).
    noise_at_floor_ : bool
        True when a fitted noise is not identified by the data: the likelihood
        alone would put it on its floor, so the data cannot tell noise from
        signal and the prior (or, with ``priors=False``, the floor) sets it.
    signal_sd_ : float
        Prior standard deviation of the latent function, in response units.
    """

    _is_discopt_surrogate = True
    # The UQ comes from GaussianProcessRegressor.predict(return_std=True);
    # kept for continuity with the sklearn adapter's reporting.
    mode = "return_std"

    def __init__(
        self,
        *,
        noise: str | float = "auto",
        ard: bool | str = "auto",
        nu: float = 2.5,
        length_scale_bounds: tuple[float, float] = (0.05, 1e2),
        noise_floor: float = 1e-2,
        n_restarts: int = 4,
        random_state: int | None = 0,
        priors: bool = True,
    ):
        if isinstance(noise, str):
            if noise != "auto":
                raise ValueError(f"noise must be 'auto' or a non-negative SD, got {noise!r}")
        elif float(noise) < 0.0:
            raise ValueError(f"noise SD must be >= 0, got {noise!r}")
        if ard not in ("auto", True, False):
            raise ValueError(f"ard must be 'auto', True or False, got {ard!r}")
        if not 0.0 < float(noise_floor) < 1.0:
            raise ValueError("noise_floor is a fraction of the response variance, in (0, 1)")
        self.noise = noise
        self.ard = ard
        self.nu = float(nu)
        self.length_scale_bounds = (float(length_scale_bounds[0]), float(length_scale_bounds[1]))
        self.noise_floor = float(noise_floor)
        self.n_restarts = int(n_restarts)
        self.random_state = random_state
        self.priors = bool(priors)
        self.estimator: Any = None
        self.ard_: bool | None = None
        self.length_scales_: np.ndarray | None = None
        self.noise_sd_: float | None = None
        self.noise_source_: str | None = None
        self.noise_at_floor_: bool = False
        self.signal_sd_: float | None = None

    # ------------------------------------------------------------------

    # Hyperparameter priors (see the module docstring). Length-scales:
    # log-normal, median exp(sqrt(2)) * sqrt(d) * span, log-SD sqrt(3), after
    # Hvarfner, Hellsten & Nardi (2024). Noise variance, as a fraction of the
    # response variance (normalize_y): log-normal, median exp(-3), log-SD 1.
    _LS_LOG_SD = float(np.sqrt(3.0))
    _NOISE_LOG_MEAN = -3.0
    _NOISE_LOG_SD = 1.0

    def _map_optimizer(self, d: int, ard: bool, has_noise: bool, span: np.ndarray):
        """A scipy L-BFGS-B optimizer over the log-posterior, for sklearn's ``optimizer=``.

        sklearn's objective is the negative log marginal likelihood of the log
        hyperparameters, laid out as [log signal, log length-scale(s), log noise].
        The log-normal priors add a quadratic penalty in those coordinates.
        """
        from scipy.optimize import minimize

        centres = np.log(span) + np.sqrt(2.0) + 0.5 * np.log(max(d, 1))
        ls_mean = centres if ard else np.array([float(np.mean(centres))])
        k = ls_mean.size
        ls_sd, n_mu, n_sd = self._LS_LOG_SD, self._NOISE_LOG_MEAN, self._NOISE_LOG_SD

        def penalty(theta: np.ndarray) -> tuple[float, np.ndarray]:
            grad = np.zeros_like(theta)
            z = (theta[1 : 1 + k] - ls_mean) / ls_sd
            value = 0.5 * float(z @ z)
            grad[1 : 1 + k] = z / ls_sd
            if has_noise:
                zn = (theta[-1] - n_mu) / n_sd
                value += 0.5 * zn * zn
                grad[-1] = zn / n_sd
            return value, grad

        def optimizer(obj_func, initial_theta, bounds):
            def f(theta):
                v, g = obj_func(theta, eval_gradient=True)
                pv, pg = penalty(theta)
                return v + pv, g + pg

            res = minimize(f, initial_theta, jac=True, method="L-BFGS-B", bounds=bounds)
            return res.x, float(res.fun)

        return optimizer

    def _build(
        self,
        d: int,
        ard: bool,
        fixed_noise: float | None,
        y_var: float,
        span: np.ndarray | None = None,
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        ls = np.ones(d) if ard else 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=ls, length_scale_bounds=self.length_scale_bounds, nu=self.nu
        )
        # normalize_y=True works in units of the response variance.
        if fixed_noise is None:
            start = max(0.1, 2.0 * self.noise_floor)
            kernel = kernel + WhiteKernel(start, (self.noise_floor, 1.0))
        elif fixed_noise > 0.0:
            kernel = kernel + WhiteKernel(max(fixed_noise / y_var, 1e-12), "fixed")
        extra: dict[str, Any] = {}
        if self.priors:
            if span is None:
                span = np.ones(d)
            extra["optimizer"] = self._map_optimizer(d, ard, fixed_noise is None, span)
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            alpha=1e-10,
            random_state=self.random_state,
            **extra,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        y_var = float(np.var(y)) if n > 1 and np.var(y) > 0 else 1.0
        span = np.ptp(X, axis=0) if n > 1 else np.ones(d)
        span = np.where(span > 0, span, 1.0)

        fixed_noise: float | None
        if isinstance(self.noise, str):  # "auto"
            pe = _pure_error_variance(X, y)
            if pe is not None:
                fixed_noise, self.noise_source_ = pe[0], "replicates"
            else:
                fixed_noise, self.noise_source_ = None, "fitted"
        elif float(self.noise) == 0.0:
            fixed_noise, self.noise_source_ = 0.0, "none"
        else:
            fixed_noise, self.noise_source_ = float(self.noise) ** 2, "fixed"

        options = [False, True] if (self.ard == "auto" and d > 1) else [self.ard is True]
        best = None
        with warnings.catch_warnings():
            # Hyperparameters on their bounds are expected here (that is what
            # the floors are for); report the one that matters below instead.
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                from sklearn.exceptions import ConvergenceWarning

                warnings.simplefilter("ignore", category=ConvergenceWarning)
            except ImportError:  # pragma: no cover
                pass
            for ard in options:
                gp = self._build(d, ard, fixed_noise, y_var, span).fit(X, y)
                # With priors, sklearn's stored value is the (negated) MAP
                # objective, i.e. the log posterior up to a constant, which is
                # the natural score for the ARD choice; the priors already
                # penalize extra length-scales. Without them, BIC does.
                score = gp.log_marginal_likelihood_value_
                if not self.priors and len(options) > 1:
                    score -= 0.5 * len(gp.kernel_.theta) * np.log(max(n, 2))
                if best is None or score > best[0]:
                    best = (score, gp, ard)
        assert best is not None
        _, gp, ard = best
        self.estimator, self.ard_ = gp, bool(ard)

        y_scale = float(np.mean(np.asarray(gp._y_train_std, dtype=float) ** 2))
        kern = gp.kernel_
        prod = kern.k1 if type(kern).__name__ == "Sum" else kern
        self.signal_sd_ = float(np.sqrt(prod.k1.constant_value * y_scale))
        self.length_scales_ = np.atleast_1d(np.asarray(prod.k2.length_scale, dtype=float))
        self.noise_sd_ = float(np.sqrt(_white_noise_variance(gp)))
        fitted = self.noise_source_ == "fitted"
        at_floor = fitted and kern.k2.noise_level <= self.noise_floor * (1.0 + 1e-6)
        if fitted and self.priors and not at_floor:
            # With the noise prior the estimate rarely touches the floor; what
            # matters is whether the *data* could have pinned it down. If the
            # likelihood alone (other hyperparameters as fitted) is higher at
            # the floor, the data cannot tell noise from signal and the prior
            # is setting the noise.
            theta = np.array(kern.theta, dtype=float)
            floor_theta = theta.copy()
            floor_theta[-1] = np.log(self.noise_floor)
            at_floor = bool(
                gp.log_marginal_likelihood(floor_theta) >= gp.log_marginal_likelihood(theta)
            )
        self.noise_at_floor_ = bool(self.noise_source_ == "fitted" and at_floor)
        if self.noise_at_floor_:
            how = (
                f"the preset's prior sets it instead (noise SD {self.noise_sd_:.3g}, "
                f"{np.sqrt(kern.k2.noise_level):.0%} of the response SD)"
                if self.priors
                else f"noise SD = {np.sqrt(self.noise_floor):.0%} of the response SD"
            )
            warnings.warn(
                "the fitted GP noise sits on its floor as far as the data can tell: they "
                f"cannot tell noise from signal, and {how}. Replicate a few runs so the "
                "noise can be estimated, pass noise=<known SD>, or noise=0 for a "
                "deterministic simulator.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and predictive standard deviation (includes observation noise)."""
        if self.estimator is None:
            raise RuntimeError("call fit() before predict()")
        mean, std = self.estimator.predict(
            np.atleast_2d(np.asarray(X, dtype=float)), return_std=True
        )
        return np.asarray(mean, dtype=float).ravel(), np.asarray(std, dtype=float).ravel()

    def predict_latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and the standard error of the mean response (no observation noise)."""
        mean, std = self.predict(X)
        noise_var = (self.noise_sd_ or 0.0) ** 2
        return mean, np.sqrt(np.clip(std**2 - noise_var, 0.0, None))

    def describe(self) -> dict[str, Any]:
        """The fitted hyperparameters in plain terms."""
        if self.estimator is None:
            raise RuntimeError("call fit() before describe()")
        est = self.estimator
        out = {
            "ard": self.ard_,
            "length_scales": None if self.length_scales_ is None else self.length_scales_.tolist(),
            "signal_sd": self.signal_sd_,
            "noise_sd": self.noise_sd_,
            "noise_source": self.noise_source_,
            "noise_at_floor": self.noise_at_floor_,
            "log_marginal_likelihood": float(est.log_marginal_likelihood(est.kernel_.theta)),
        }
        if self.priors:
            out["log_posterior"] = float(est.log_marginal_likelihood_value_)
        return out


def gp_surrogate(
    *,
    noise: str | float = "auto",
    ard: bool | str = "auto",
    nu: float = 2.5,
    length_scale_bounds: tuple[float, float] = (0.05, 1e2),
    noise_floor: float = 1e-2,
    n_restarts: int = 4,
    random_state: int | None = 0,
    priors: bool = True,
) -> GPSurrogate:
    """The ``"gp"`` preset: a Gaussian-process surrogate with calibrated defaults.

    Parameters
    ----------
    noise : "auto" or float, default "auto"
        ``"auto"`` estimates the noise from exactly replicated runs when there
        are any (pure error) and otherwise fits it, no lower than
        ``noise_floor``. A float fixes the noise standard deviation in response
        units; ``0`` interpolates the data (deterministic simulators).
    ard : "auto", True or False, default "auto"
        One length-scale per input (True), one shared (False), or choose by
        the log posterior ("auto"; a BIC-penalized likelihood when
        ``priors=False``).
    nu : float, default 2.5
        Matérn smoothness.
    length_scale_bounds : (float, float), default (0.05, 100)
        Bounds on the length-scales, in the units of the inputs passed to
        ``fit`` (standardized, inside :func:`~discopt.doe.optimize_round`).
    noise_floor : float, default 0.01
        Smallest fitted noise variance, as a fraction of the response
        variance (0.01 = a noise SD of 10% of the response SD).
    n_restarts : int, default 4
        Optimizer restarts for the hyperparameters.
    random_state : int or None, default 0
        Seed for the optimizer restarts, so fits are reproducible.
    priors : bool, default True
        Fit the hyperparameters by maximum a posteriori with weak log-normal
        priors on the length-scales and the noise (see the module docstring).
        ``False`` gives plain maximum likelihood, which can read a small
        design as all noise, or as noise-free.
    """
    _require_sklearn()
    return GPSurrogate(
        noise=noise,
        ard=ard,
        nu=nu,
        length_scale_bounds=length_scale_bounds,
        noise_floor=noise_floor,
        n_restarts=n_restarts,
        random_state=random_state,
        priors=priors,
    )


def _response_surface_preset() -> Surrogate:
    """Default response surface: degree-2 polynomial with BayesianRidge UQ."""
    _require_sklearn()
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        BayesianRidge(),
    )
    return _SklearnUQAdapter(pipe)


PRESETS: dict[str, Callable[..., Surrogate]] = {
    "gp": gp_surrogate,
    "response-surface": _response_surface_preset,
}


def coerce_surrogate(
    obj: object, *, random_state: int | None = None, include_noise: bool = False
) -> Surrogate:
    """Normalize a user-supplied surrogate spec to the Surrogate protocol.

    Accepts:

    * a string in :data:`PRESETS` (e.g. ``"gp"``);
    * any object with ``fit`` and ``predict`` (sklearn-style) -- wrapped
      in :class:`_SklearnUQAdapter`;
    * an object that already implements :class:`Surrogate`.

    Parameters
    ----------
    random_state : int, optional
        Seed forwarded to presets that accept one and to the bootstrap
        adapter, so a seeded optimization round is reproducible.
    include_noise : bool, default False
        For estimators wrapped with the bootstrap fallback: add the residual
        variance to ``predict``'s std, so it describes a new observation
        rather than the fitted mean (``predict_latent`` never includes it).

    Returns the surrogate ready for ``fit()``. Raises ``TypeError`` for
    anything else.
    """
    if isinstance(obj, str):
        try:
            factory = PRESETS[obj]
        except KeyError as e:
            raise ValueError(
                f"unknown surrogate preset {obj!r}; available: {sorted(PRESETS)}"
            ) from e
        if random_state is not None and "random_state" in inspect.signature(factory).parameters:
            return factory(random_state=int(random_state))
        return factory()
    if _matches_surrogate_protocol(obj):
        return cast(Surrogate, obj)  # already returns (mean, std)
    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        return _SklearnUQAdapter(
            obj,
            random_state=0 if random_state is None else int(random_state),
            include_noise=include_noise,
        )
    raise TypeError(
        f"surrogate {obj!r} must be a string preset, a sklearn-style estimator "
        "(with fit/predict), or implement the Surrogate protocol"
    )


def _matches_surrogate_protocol(obj: object) -> bool:
    """True when predict() is known to return a 2-tuple already.

    We can't tell from the signature alone, so the protocol is treated
    as opt-in via a class attribute ``_is_discopt_surrogate = True``.
    Anything else goes through the sklearn adapter, which is the
    correct path for ``GaussianProcessRegressor`` etc.
    """
    return bool(getattr(obj, "_is_discopt_surrogate", False))


__all__ = [
    "GPSurrogate",
    "PRESETS",
    "Surrogate",
    "coerce_surrogate",
    "gp_surrogate",
]
