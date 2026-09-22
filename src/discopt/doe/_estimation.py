"""Parameter-estimation dispatch for the discopt-doe loops.

:func:`discopt.estimate.estimate_parameters` builds and solves a discopt NLP,
then computes the covariance by compiling the response expressions. An
experiment whose responses come from an opaque JAX callable (a
:func:`discopt.modeling.custom` node, as in :mod:`discopt.doe.dynamic`) solves
fine but cannot go through that covariance compiler. Such experiments carry
their own ``estimate`` method, and the loops that fit models repeatedly
(:func:`~discopt.doe.profile_likelihood`, :func:`~discopt.doe.sequential_doe`,
:func:`~discopt.doe.sequential_discrimination`) call this dispatcher instead
of the base function.
"""

from __future__ import annotations

from typing import Any


def estimate_parameters(experiment: Any, data: Any, **kwargs: Any):
    """``experiment.estimate(data, **kwargs)`` when defined, else the base estimator.

    Takes and returns exactly what :func:`discopt.estimate.estimate_parameters`
    does, so it is a drop-in replacement.
    """
    own = getattr(experiment, "estimate", None)
    if callable(own):
        return own(data, **kwargs)
    from discopt.estimate import estimate_parameters as _base

    return _base(experiment, data, **kwargs)


__all__ = ["estimate_parameters"]
