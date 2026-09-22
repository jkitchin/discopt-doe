"""Designs that stay good when the parameter guess is wrong.

A model-based design for a nonlinear model is only *locally* optimal: it is
built for a nominal parameter value, and it can lose most of its efficiency
when the truth is elsewhere (Chapter 17 of the book shows a first-order decay
design at ``t = 1/k`` falling to 4 % efficiency when ``k`` is off by four).
Robust criteria average or guard against that loss over a set of plausible
parameter values, a sample from a prior or a grid over a region:

- **pseudo-Bayesian** (``robust="expected"``): maximize the average
  ``log det FIM(theta_s, design)`` over the samples (ED-optimality;
  Chaloner & Verdinelli 1995, Pronzato & Walter 1985);
- **max-min** (``robust="maximin"``): maximize the *worst* D-efficiency over
  the samples, each relative to that sample's own locally optimal design.

The information for each sample is computed with a response Jacobian
compiled once per sample, so a design evaluation costs one Jacobian call per
sample, not a model rebuild.

References
----------
Chaloner, K. & Verdinelli, I. Bayesian experimental design: a review.
*Statist. Sci.* 10, 273-304 (1995).

Pronzato, L. & Walter, E. Robust experiment design via stochastic
approximation. *Math. Biosci.* 75, 103-120 (1985).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from discopt.estimate import Experiment

_SINGULAR = -1e30


@dataclass
class RobustDesignResult:
    """Result of :func:`robust_optimal_experiment`.

    Attributes
    ----------
    designs : list[dict]
        The ``n_experiments`` design points.
    robust : str
        ``"expected"`` or ``"maximin"``.
    criterion_value : float
        Mean ``log det`` (expected) or minimum D-efficiency (maximin).
    log_dets : numpy.ndarray
        ``log det FIM`` of the design at each parameter sample.
    efficiencies : numpy.ndarray or None
        D-efficiency at each sample, relative to that sample's locally
        optimal design (computed for ``maximin``, or when ``reference_log_dets``
        is given).
    reference_log_dets : numpy.ndarray or None
        ``log det`` of each sample's locally optimal design.
    parameter_samples : list[dict]
    n_parameters : int
    """

    designs: list[dict[str, float]]
    robust: str
    criterion_value: float
    log_dets: np.ndarray
    efficiencies: np.ndarray | None
    reference_log_dets: np.ndarray | None
    parameter_samples: list[dict[str, float]]
    n_parameters: int

    def summary(self) -> str:
        lines = [f"robust design ({self.robust}), {len(self.designs)} run(s)"]
        for i, d in enumerate(self.designs):
            lines.append(f"  run {i + 1}: " + ", ".join(f"{k}={v:.6g}" for k, v in d.items()))
        lines.append(
            f"  mean log det over {len(self.log_dets)} samples: {self.log_dets.mean():.4f}"
        )
        if self.efficiencies is not None:
            e = self.efficiencies
            lines.append(
                f"  D-efficiency: mean {e.mean():.3f}, min {e.min():.3f}, "
                f"5th percentile {np.percentile(e, 5):.3f}"
            )
        return "\n".join(lines)


def _as_samples(param_samples: Any) -> list[dict[str, float]]:
    if isinstance(param_samples, Mapping):
        keys = list(param_samples)
        cols = [np.atleast_1d(np.asarray(param_samples[k], dtype=float)) for k in keys]
        n = len(cols[0])
        if any(len(c) != n for c in cols):
            raise ValueError("every parameter needs the same number of samples")
        return [{k: float(c[i]) for k, c in zip(keys, cols)} for i in range(n)]
    out = [{k: float(v) for k, v in s.items()} for s in param_samples]
    if not out:
        raise ValueError("param_samples is empty")
    return out


def _fim_functions(
    experiment: Experiment, samples: Sequence[Mapping[str, float]]
) -> list[Callable[[dict[str, float] | None], np.ndarray]]:
    """One ``design -> FIM`` function per sample, compiled once where possible."""
    from discopt.doe.fim import compute_fim

    try:
        from discopt.doe.fim import _make_direct_fim_evaluator
    except ImportError:  # pragma: no cover - internal helper renamed
        _make_direct_fim_evaluator = None  # type: ignore[assignment]

    fns = []
    for theta in samples:
        ev = None
        if _make_direct_fim_evaluator is not None:
            try:
                ev = _make_direct_fim_evaluator(experiment, dict(theta))
            except Exception:  # noqa: BLE001 - fall back to the per-call path
                ev = None
        if ev is not None:
            fns.append(lambda d, _ev=ev: np.asarray(_ev(d).fim, dtype=float))
        else:
            fns.append(
                lambda d, _t=dict(theta): np.asarray(
                    compute_fim(experiment, _t, d).fim, dtype=float
                )
            )
    return fns


def _logdet(m: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(m)
    return float(val) if sign > 0 and np.isfinite(val) else _SINGULAR


def robust_optimal_experiment(
    experiment: Experiment,
    param_samples: Sequence[Mapping[str, float]] | Mapping[str, Sequence[float]],
    design_bounds: Mapping[str, tuple[float, float]],
    *,
    criterion: str = "D",
    robust: str = "expected",
    n_experiments: int = 1,
    prior_fim: np.ndarray | None = None,
    reference_log_dets: Sequence[float] | None = None,
    n_starts: int = 10,
    seed: int = 0,
) -> RobustDesignResult:
    """Design experiments that are good across a set of parameter values.

    Parameters
    ----------
    experiment : Experiment
    param_samples : sequence of dict, or dict of arrays
        Plausible parameter values: draws from a prior, or a grid over the
        region you believe the parameters lie in. Tens of samples usually
        suffice; the cost grows linearly with their number.
    design_bounds : mapping name -> (lb, ub)
        Bounds on every design input.
    criterion : {"D"}, default "D"
        Only D-optimality is supported: its efficiency is invariant to how the
        parameters are scaled, which is what lets values from different
        samples be averaged or compared.
    robust : {"expected", "maximin"}, default "expected"
        Pseudo-Bayesian (maximize mean ``log det``) or max-min (maximize the
        worst D-efficiency).
    n_experiments : int, default 1
        Runs in the design; their information adds.
    prior_fim : numpy.ndarray, optional
        Information already collected (the same for every sample).
    reference_log_dets : sequence of float, optional
        ``log det`` of each sample's locally optimal design, for max-min.
        Computed when omitted (one local optimization per sample).
    n_starts : int, default 10
        Random starts of the design search.
    seed : int, default 0

    Returns
    -------
    RobustDesignResult

    Notes
    -----
    A max-min design guarantees its worst case only over the samples it was
    built on. With few samples it can over-fit them; check its efficiency on
    fresh draws (``design_efficiencies``) before relying on the guarantee.
    """
    from scipy.optimize import minimize

    if str(criterion).upper() not in ("D", "D_OPTIMAL", "DETERMINANT"):
        raise ValueError("robust designs support criterion='D' only")
    if robust not in ("expected", "maximin"):
        raise ValueError(f"robust must be 'expected' or 'maximin', got {robust!r}")
    samples = _as_samples(param_samples)
    names = list(design_bounds)
    lbs = np.array([float(design_bounds[n][0]) for n in names])
    ubs = np.array([float(design_bounds[n][1]) for n in names])
    n_exp = int(n_experiments)
    fims = _fim_functions(experiment, samples)
    p = fims[0]({n: float((lo + hi) / 2) for n, lo, hi in zip(names, lbs, ubs)}).shape[0]
    prior = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=float)

    def designs_of(z: np.ndarray) -> list[dict[str, float]]:
        z = np.clip(z.reshape(n_exp, len(names)), lbs, ubs)
        return [dict(zip(names, map(float, row))) for row in z]

    def log_dets(z: np.ndarray, which: Sequence[int] | None = None) -> np.ndarray:
        ds = designs_of(z)
        idx = range(len(samples)) if which is None else which
        out = []
        for s in idx:
            total = prior.copy()
            for d in ds:
                total = total + fims[s](d)
            out.append(_logdet(total))
        return np.asarray(out)

    rng = np.random.default_rng(seed)
    bounds = [(lo, hi) for _ in range(n_exp) for lo, hi in zip(lbs, ubs)]

    def best_of(obj: Callable[[np.ndarray], float], method: str) -> np.ndarray:
        best_z, best_f = None, np.inf
        for _ in range(max(1, int(n_starts))):
            z0 = np.concatenate([rng.uniform(lbs, ubs) for _ in range(n_exp)])
            res = minimize(obj, z0, method=method, bounds=bounds)
            if res.fun < best_f:
                best_f, best_z = float(res.fun), np.asarray(res.x)
        assert best_z is not None
        return np.clip(best_z, np.tile(lbs, n_exp), np.tile(ubs, n_exp))

    ref = None
    if robust == "maximin" or reference_log_dets is not None:
        if reference_log_dets is not None:
            ref = np.asarray(reference_log_dets, dtype=float)
            if ref.shape != (len(samples),):
                raise ValueError("reference_log_dets needs one value per parameter sample")
        else:
            # Each sample's locally optimal design sets its efficiency scale.
            ref = np.empty(len(samples))
            for s in range(len(samples)):
                z_s = best_of(lambda z, _s=s: -float(log_dets(z, [_s])[0]), "L-BFGS-B")
                ref[s] = log_dets(z_s, [s])[0]

    if robust == "expected":
        z = best_of(lambda z: -float(np.mean(log_dets(z))), "L-BFGS-B")
        ld = log_dets(z)
        value = float(np.mean(ld))
    else:
        assert ref is not None

        def neg_min_eff(z: np.ndarray) -> float:
            eff = np.exp((log_dets(z) - ref) / p)
            return -float(np.min(eff))

        # The worst case is not smooth; Powell copes with its kinks.
        z = best_of(neg_min_eff, "Powell")
        ld = log_dets(z)
        value = float(np.min(np.exp((ld - ref) / p)))

    eff = None if ref is None else np.exp((ld - ref) / p)
    return RobustDesignResult(
        designs=designs_of(z),
        robust=robust,
        criterion_value=value,
        log_dets=ld,
        efficiencies=eff,
        reference_log_dets=ref,
        parameter_samples=samples,
        n_parameters=int(p),
    )


def design_efficiencies(
    experiment: Experiment,
    designs: Sequence[Mapping[str, float]],
    param_samples: Sequence[Mapping[str, float]] | Mapping[str, Sequence[float]],
    reference_designs: Sequence[Sequence[Mapping[str, float]]] | None = None,
    *,
    reference_log_dets: Sequence[float] | None = None,
    prior_fim: np.ndarray | None = None,
) -> np.ndarray:
    """D-efficiency of ``designs`` at each parameter sample.

    Relative to ``reference_log_dets`` (one per sample) or to
    ``reference_designs`` (one design per sample, e.g. each sample's local
    optimum). Use it to check a robust design on fresh parameter draws that it
    was not built on.
    """
    samples = _as_samples(param_samples)
    fims = _fim_functions(experiment, samples)
    p = fims[0](dict(designs[0])).shape[0]
    prior = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=float)

    def ld(s: int, ds: Sequence[Mapping[str, float]]) -> float:
        total = prior.copy()
        for d in ds:
            total = total + fims[s](dict(d))
        return _logdet(total)

    if reference_log_dets is not None:
        ref = np.asarray(reference_log_dets, dtype=float)
    elif reference_designs is not None:
        ref = np.array([ld(s, reference_designs[s]) for s in range(len(samples))])
    else:
        raise ValueError("pass reference_log_dets or reference_designs")
    return np.exp((np.array([ld(s, designs) for s in range(len(samples))]) - ref) / p)


__all__ = ["RobustDesignResult", "design_efficiencies", "robust_optimal_experiment"]
