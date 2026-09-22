"""Acquisition functions for active-learning optimization.

An acquisition function scores candidate design points by how
*valuable* they are to evaluate next. Higher scores are better; the
optimizer selects the top-``batch_size`` candidates.

Every function in this module has the signature::

    score(surrogate, X_candidates, *, y_best, direction, **kw) -> ndarray

* ``surrogate`` is a fitted object satisfying the
  :class:`~discopt.doe.surrogate.Surrogate` protocol.
* ``X_candidates`` is a 2D array of shape ``(n, d)``.
* ``y_best`` is the best observed response so far (incumbent).
* ``direction`` is ``+1`` for *maximize* or ``-1`` for *minimize*.

Implementations
---------------

* :func:`expected_improvement` -- the standard EI used in Bayesian
  optimization. Balances exploitation (high predicted mean in the
  desired direction) and exploration (high uncertainty).
* :func:`upper_confidence_bound` / :func:`lower_confidence_bound` --
  UCB / LCB. A direction-aware wrapper :func:`confidence_bound` picks
  the right sign automatically.
* :func:`steepest_ascent` -- Box-Wilson style: score points by the
  predicted improvement only (ignores uncertainty). Useful with a
  response-surface surrogate when you want classical RSM behaviour.
* :func:`max_variance` -- pure exploration: the candidate the surrogate
  is least sure about. For *active learning* (making the surrogate
  accurate everywhere) rather than optimization.

Latent versus predictive uncertainty
------------------------------------

EI, the confidence bounds and ``max_variance`` use the uncertainty of the
*mean response* (the surrogate's ``predict_latent``, when it has one), not
the predictive uncertainty of a new noisy observation. The question they
answer is "how much could the true response here improve on the incumbent?",
and noise does not make the true response better. With the predictive σ,
which never falls below the noise level, EI stays large at points that are
already well known and never decays to signal convergence. Pass
``latent=False`` to score on the predictive σ instead.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

from discopt.doe.surrogate import Surrogate

Direction = Literal[1, -1]


def _erf(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.vectorize(math.erf)(x), dtype=float)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi), dtype=float)


def _mean_std(surrogate: Surrogate, X: np.ndarray, latent: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` from the surrogate; the latent std when asked and available."""
    X = np.asarray(X, dtype=float)
    fn = getattr(surrogate, "predict_latent", None) if latent else None
    mu, sigma = fn(X) if callable(fn) else surrogate.predict(X)
    return np.asarray(mu, dtype=float).ravel(), np.asarray(sigma, dtype=float).ravel()


def expected_improvement(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    y_best: float,
    direction: Direction,
    xi: float = 0.0,
    latent: bool = True,
) -> np.ndarray:
    """Expected improvement over the incumbent ``y_best``.

    For maximization (``direction = +1``)::

        EI(x) = (μ - y_best - xi) Φ(z) + σ φ(z)
        z     = (μ - y_best - xi) / σ

    For minimization (``direction = -1``), the sign flips so EI is
    always non-negative for points that improve on the incumbent.

    ``xi`` is an exploration knob -- a small positive value
    (e.g. ``0.01``) demands a slightly stronger improvement and
    encourages exploration.
    """
    mu, sigma = _mean_std(surrogate, X_candidates, latent)
    dir_sign = int(direction)
    if dir_sign not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {dir_sign}")

    improvement = dir_sign * (mu - y_best) - xi
    safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
    z = improvement / safe_sigma
    ei = improvement * _norm_cdf(z) + safe_sigma * _norm_pdf(z)
    ei = np.where(sigma > 0.0, ei, np.maximum(improvement, 0.0))
    return ei


def upper_confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    kappa: float = 2.0,
    latent: bool = True,
) -> np.ndarray:
    """``μ + κ σ`` -- maximize when ``direction = +1``."""
    mu, sigma = _mean_std(surrogate, X_candidates, latent)
    return mu + float(kappa) * sigma


def lower_confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    kappa: float = 2.0,
    latent: bool = True,
) -> np.ndarray:
    """``-(μ - κ σ)`` -- larger is *better* under minimization."""
    mu, sigma = _mean_std(surrogate, X_candidates, latent)
    return -(mu - float(kappa) * sigma)


def confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    direction: Direction,
    kappa: float = 2.0,
    latent: bool = True,
) -> np.ndarray:
    """Direction-aware UCB/LCB. Higher score = better candidate."""
    if direction == 1:
        return upper_confidence_bound(surrogate, X_candidates, kappa=kappa, latent=latent)
    if direction == -1:
        return lower_confidence_bound(surrogate, X_candidates, kappa=kappa, latent=latent)
    raise ValueError(f"direction must be +1 or -1, got {direction}")


def max_variance(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    latent: bool = True,
) -> np.ndarray:
    """Pure exploration: score each candidate by the surrogate's uncertainty.

    The next run goes where the surrogate knows least, which is the classic
    uncertainty-sampling rule for active learning (Cohn, Ghahramani &
    Jordan 1996): it makes the surrogate accurate everywhere rather than
    finding an optimum, and ignores the response values entirely. With a
    batch, the fantasy refits in :func:`~discopt.doe.optimize_round` spread
    the picks out.
    """
    _mu, sigma = _mean_std(surrogate, X_candidates, latent)
    return sigma


def steepest_ascent(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    direction: Direction,
    y_best: float | None = None,
) -> np.ndarray:
    """Predicted improvement only -- ignores uncertainty.

    Score is ``direction * μ(x)`` (or, if ``y_best`` is supplied,
    ``direction * (μ(x) - y_best)``, which only shifts the score and
    does not change the ranking). Use this with a response-surface
    surrogate to reproduce classical Box-Wilson behaviour.
    """
    mu, _sigma = surrogate.predict(np.asarray(X_candidates, dtype=float))
    mu = np.asarray(mu, dtype=float).ravel()
    if y_best is None:
        return int(direction) * mu
    return int(direction) * (mu - float(y_best))


# Note: "ucb" and "lcb" both resolve to the *direction-aware* confidence_bound
# wrapper, which picks upper (maximize) or lower (minimize) from the round's
# direction. So under maximization "lcb" still computes the UCB (and vice
# versa) -- the names are aliases for "confidence bound", not a hard choice of
# upper vs lower. Use the raw upper_confidence_bound / lower_confidence_bound
# callables directly if you need a fixed side.
ACQUISITIONS: dict[str, Callable[..., np.ndarray]] = {
    "expected_improvement": expected_improvement,
    "ei": expected_improvement,
    "ucb": confidence_bound,
    "lcb": confidence_bound,
    "confidence_bound": confidence_bound,
    "steepest_ascent": steepest_ascent,
    "max_variance": max_variance,
    "uncertainty": max_variance,
}


def resolve_acquisition(name_or_fn):
    """Look up an acquisition function by string or pass through a callable."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return ACQUISITIONS[name_or_fn]
    except KeyError as e:
        raise ValueError(
            f"unknown acquisition {name_or_fn!r}; available: {sorted(set(ACQUISITIONS))}"
        ) from e


def call_acquisition(acq_fn, surrogate, X_candidates, *, direction, y_best, acq_kwargs=None):
    """Invoke an acquisition, passing only the arguments it accepts.

    Replaces a ``try/except TypeError`` dispatch that both masked genuine
    errors raised *inside* an acquisition and silently discarded the caller's
    tuning kwargs. ``direction``/``y_best`` are injected only when the callable
    declares them (or has ``**kwargs``), and user ``acq_kwargs`` that the
    callable cannot consume raise a clear error instead of being swallowed.
    """
    import inspect

    acq_kwargs = dict(acq_kwargs or {})
    params = inspect.signature(acq_fn).parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    named = {
        n
        for n, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if not has_var_kw:
        bad = [k for k in acq_kwargs if k not in named]
        if bad:
            tunable = sorted(named - {"surrogate", "X_candidates", "direction", "y_best"})
            raise TypeError(
                f"acquisition {getattr(acq_fn, '__name__', acq_fn)!r} got unexpected "
                f"keyword(s) {bad}; it accepts {tunable}."
            )
    call_kwargs = dict(acq_kwargs)
    if "direction" in named or has_var_kw:
        call_kwargs["direction"] = direction
    if "y_best" in named or has_var_kw:
        call_kwargs["y_best"] = y_best
    return acq_fn(surrogate, X_candidates, **call_kwargs)


__all__ = [
    "ACQUISITIONS",
    "confidence_bound",
    "expected_improvement",
    "lower_confidence_bound",
    "max_variance",
    "resolve_acquisition",
    "steepest_ascent",
    "upper_confidence_bound",
]
