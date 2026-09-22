"""Space-filling designs and the metrics that compare them.

A space-filling design is judged by geometry alone, before any model is fitted,
so it needs numbers to be judged by. :func:`space_filling_metrics` reports the
standard ones on the unit cube:

* **minimum distance**: the closest pair of runs (maximin designs make this
  large, so that no two runs nearly duplicate each other);
* **mean nearest-neighbour distance**, the typical spacing;
* **phi_p** (Morris & Mitchell 1995), a smooth surrogate for the minimum
  distance that also counts the second-closest pairs and beyond (smaller is
  better);
* **centred L2 discrepancy** (Hickernell 1998), the deviation from a uniform
  spread over the box and over all its lower-dimensional projections
  (smaller is better);
* **maximum projection gap** per factor, the widest empty interval in each
  one-dimensional projection, edges included, which shows whether a factor
  that turns out to matter on its own was sampled evenly.

:func:`quasi_random_design` builds scrambled Sobol and Halton designs, the
low-discrepancy sequences that are the usual alternative to a Latin
hypercube. Latin hypercubes themselves, plain, discrepancy-optimized or
maximin, are :func:`discopt.doe.classical.latin_hypercube_design`.

numpy and scipy only, so it works in the browser build as well.
"""

from __future__ import annotations

import math
import warnings
from typing import Mapping, Sequence

import numpy as np

from discopt.doe.classical import ClassicalDesign, _finalize, _validate_factors


def _unit_matrix(
    design: ClassicalDesign | np.ndarray | Sequence[Sequence[float]],
    bounds: Sequence[tuple[float, float]] | None,
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    if isinstance(design, ClassicalDesign):
        return design.to_unit_matrix(), design.factors
    x = np.atleast_2d(np.asarray(design, dtype=np.float64))
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (x.shape[1], 2):
            raise ValueError(f"bounds must have shape ({x.shape[1]}, 2), got {b.shape}")
        if np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("every upper bound must exceed its lower bound")
        x = (x - b[:, 0]) / (b[:, 1] - b[:, 0])
    return x, None


def space_filling_metrics(
    design: ClassicalDesign | np.ndarray | Sequence[Sequence[float]],
    bounds: Sequence[tuple[float, float]] | None = None,
    *,
    p: float = 15.0,
) -> dict[str, object]:
    """Geometric quality of a design on the unit cube.

    Parameters
    ----------
    design : ClassicalDesign or array_like, shape (n_runs, n_factors)
        A design, or a run matrix. A :class:`ClassicalDesign` is scaled to the
        unit cube with its own bounds.
    bounds : sequence of (lb, ub), optional
        Factor ranges for scaling a raw matrix to the unit cube. Omit it when
        the matrix is already on ``[0, 1]``.
    p : float, default 15
        Exponent of the Morris-Mitchell criterion.

    Returns
    -------
    dict
        ``min_distance`` and ``mean_min_distance`` (larger is better);
        ``phi_p`` and ``centered_discrepancy`` (smaller is better);
        ``max_projection_gap`` (per factor, a dict when factor names are
        known, otherwise a list); ``n_runs`` and ``n_factors``.
        ``centered_discrepancy`` is NaN when a run lies outside the unit cube
        (e.g. a circumscribed central composite design), where it is undefined.

    Examples
    --------
    >>> m = space_filling_metrics([[0.0, 0.0], [1.0, 1.0]])
    >>> round(m["min_distance"], 4)
    1.4142
    """
    x, names = _unit_matrix(design, bounds)
    n, k = x.shape
    if n < 2:
        raise ValueError("space-filling metrics need at least 2 runs")

    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    pair = dist[np.triu_indices(n, 1)]
    with np.errstate(divide="ignore", over="ignore"):
        phi = float(np.sum(pair ** (-float(p))) ** (1.0 / float(p)))

    inside = bool(np.all((x >= -1e-12) & (x <= 1 + 1e-12)))
    if inside:
        from scipy.stats import qmc

        cd = float(qmc.discrepancy(np.clip(x, 0.0, 1.0), method="CD"))
    else:
        cd = math.nan

    gaps = []
    for j in range(k):
        col = np.concatenate([[0.0], np.sort(np.clip(x[:, j], 0.0, 1.0)), [1.0]])
        gaps.append(float(np.max(np.diff(col))))

    return {
        "n_runs": int(n),
        "n_factors": int(k),
        "min_distance": float(nearest.min()),
        "mean_min_distance": float(nearest.mean()),
        "phi_p": phi,
        "centered_discrepancy": cd,
        "max_projection_gap": dict(zip(names, gaps)) if names is not None else gaps,
    }


def quasi_random_design(
    factors: Mapping[str, tuple[float, float]],
    n: int,
    *,
    method: str = "sobol",
    scramble: bool = True,
    seed: int | None = None,
) -> ClassicalDesign:
    """Build a scrambled Sobol or Halton design over a continuous box.

    Quasi-random (low-discrepancy) sequences are deterministic constructions
    that fill the box evenly as points are added, so a design can be extended
    later by taking the next points of the same sequence. Scrambling
    randomizes them while keeping the low discrepancy.

    Parameters
    ----------
    factors : mapping name -> (lb, ub)
        Continuous range per factor.
    n : int
        Number of runs (at least 2). A Sobol sequence is balanced only at
        powers of two; other sizes work but draw a warning.
    method : {"sobol", "halton"}, default "sobol"
    scramble : bool, default True
        Randomize the sequence (Owen scrambling for Sobol, permutations for
        Halton). An unscrambled Sobol sequence starts at the origin corner.
    seed : int, optional
        Reproducible scrambling and run order.

    Returns
    -------
    ClassicalDesign
        ``kind`` is ``"sobol"`` or ``"halton"``.
    """
    from scipy.stats import qmc

    names, lbs, ubs = _validate_factors(factors)
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    key = method.strip().lower()
    k = len(names)
    if key == "sobol":
        sampler = qmc.Sobol(d=k, scramble=scramble, seed=seed)
        m = int(round(math.log2(n)))
        if 2**m == n:
            unit = sampler.random_base2(m)
        else:
            warnings.warn(
                f"a Sobol design is balanced only at powers of two; n={n} is not one "
                f"(nearest: {2 ** max(1, m)}). The points are still low-discrepancy.",
                UserWarning,
                stacklevel=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unit = sampler.random(n)
    elif key == "halton":
        unit = qmc.Halton(d=k, scramble=scramble, seed=seed).random(n)
    else:
        raise ValueError(f"method must be 'sobol' or 'halton', got {method!r}")

    values = lbs + unit * (ubs - lbs)
    return _finalize(key, names, lbs, ubs, values, is_center=[False] * n, seed=seed)


__all__ = ["quasi_random_design", "space_filling_metrics"]
