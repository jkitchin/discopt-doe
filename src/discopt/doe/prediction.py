"""Prediction variance of a design: where does it predict well, and where not?

A design's coefficient covariance is ``σ² (XᵀX)⁻¹``. The variance of the
*fitted response* at a point ``x`` follows from it:

    Var ŷ(x) = σ² · f(x)ᵀ (XᵀX)⁻¹ f(x)

where ``f(x)`` is the model's basis row (for a model nonlinear in its
parameters, its sensitivity row at nominal values). The quadratic form
``v(x) = f(x)ᵀ (XᵀX)⁻¹ f(x)`` depends only on the design and the model, never on
the data, so it can be mapped before a single run is made. This module maps it:

* :func:`prediction_variance` -- ``σ² · v(x)`` at a set of points.
* :func:`scaled_prediction_variance` -- ``N · v(x)``, the *scaled prediction
  variance* (SPV), which charges a design for its run count so that designs
  of different sizes compare fairly.
* :func:`fds_curve` -- the *fraction of design space* plot: the SPV
  distribution over a region, sorted, so one curve summarises how much of the
  region a design predicts to a given precision
  (Zahran, Anderson-Cook & Myers 2003).
* :func:`i_criterion` and :func:`g_criterion` -- the average and the maximum
  of ``v(x)`` over a region: the I- (integrated, also "V-") and G-optimality
  criteria. Unlike D, which is about the coefficients, both are about
  prediction.

The model can be given three ways, mirroring :mod:`discopt.doe.linear_design`:
a linear template name (``"linear"``, ``"polynomial-1d"``,
``"response-surface-2d"``, ``"scheffe-quadratic"``, the classical design
templates with a ``basis``, ...), any ``basis`` callable, or a
:class:`~discopt.doe.symbolic.SymbolicModel` with nominal parameters ``theta``.

Regions are either a box (``bounds``) or a mixture simplex (``mixture_total``),
sampled uniformly, or any explicit set of ``points``. Only numpy is needed, so
the module runs in the browser build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "FDSResult",
    "fds_curve",
    "g_criterion",
    "i_criterion",
    "prediction_variance",
    "region_points",
    "scaled_prediction_variance",
]

Rows = Sequence[Mapping[str, Any]] | np.ndarray


# --------------------------------------------------------------------------
# Model and point handling
# --------------------------------------------------------------------------


def _as_rows(points: Rows, input_names: Sequence[str]) -> list[dict[str, float]]:
    """Accept a sequence of mappings or an ``(n, k)`` array ordered by ``input_names``."""
    if isinstance(points, np.ndarray):
        arr = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if arr.shape[1] != len(input_names):
            raise ValueError(
                f"points array has {arr.shape[1]} columns but there are "
                f"{len(input_names)} inputs {list(input_names)}"
            )
        return [dict(zip(input_names, map(float, row))) for row in arr]
    out = []
    for r in points:
        missing = [n for n in input_names if n not in r]
        if missing:
            raise ValueError(f"point {dict(r)!r} is missing inputs {missing}")
        out.append({n: float(r[n]) for n in input_names})
    return out


def _infer_input_names(design_rows: Rows, input_names: Sequence[str] | None) -> list[str]:
    if input_names is not None:
        return list(input_names)
    if isinstance(design_rows, np.ndarray):
        raise ValueError("input_names is required when design_rows is an array")
    first = dict(design_rows[0])
    # Bookkeeping columns written by the design generators are not inputs.
    skip = {"run_order", "replicate", "is_center", "block", "run_id", "batch"}
    return [k for k, v in first.items() if k not in skip and isinstance(v, (int, float))]


def _row_function(
    *,
    template: str | None,
    template_args: Mapping[str, Any] | None,
    parameter_names: Sequence[str] | None,
    input_names: Sequence[str],
    basis: Callable[[np.ndarray], np.ndarray] | None,
    model: Any | None,
    theta: Mapping[str, float] | None,
) -> Callable[[Mapping[str, float]], np.ndarray]:
    """Return ``f(row) -> basis row`` for whichever way the model was given."""
    given = [template is not None, basis is not None, model is not None]
    if sum(given) != 1:
        raise ValueError("give exactly one of template=, basis=, or model=")

    if basis is not None:
        names = list(input_names)

        def from_basis(row: Mapping[str, float]) -> np.ndarray:
            return np.asarray(basis(np.array([row[n] for n in names])), dtype=np.float64)

        return from_basis

    if model is not None:
        if theta is None:
            raise ValueError("model= needs theta= (nominal parameter values)")

        def from_model(row: Mapping[str, float]) -> np.ndarray:
            return np.asarray(model.jacobian_row(theta, row), dtype=np.float64)

        return from_model

    from discopt.doe.linear_design import design_row

    args = dict(template_args or {})
    if parameter_names is None:
        from discopt.doe.templates import template_parameter_names

        parameter_names = template_parameter_names(
            template,
            len(input_names),
            degree=args.get("degree"),
            basis=args.get("basis"),
        )
    pnames = list(parameter_names)
    inames = list(input_names)

    def from_template(row: Mapping[str, float]) -> np.ndarray:
        return design_row(template, args, pnames, inames, row)

    return from_template


def _information_inverse(F_design: np.ndarray) -> np.ndarray:
    XtX = F_design.T @ F_design
    rank = np.linalg.matrix_rank(XtX)
    if rank < XtX.shape[0]:
        raise ValueError(
            f"the design cannot estimate the model: XᵀX has rank {rank} but the model has "
            f"{XtX.shape[0]} parameters, so the prediction variance is infinite somewhere. "
            "Add runs, or runs at more distinct points."
        )
    return np.linalg.inv(XtX)


def _relative_variance(
    design_rows: Rows,
    points: Rows,
    *,
    template: str | None,
    template_args: Mapping[str, Any] | None,
    parameter_names: Sequence[str] | None,
    input_names: Sequence[str] | None,
    basis: Callable[[np.ndarray], np.ndarray] | None,
    model: Any | None,
    theta: Mapping[str, float] | None,
) -> tuple[np.ndarray, int]:
    names = _infer_input_names(design_rows, input_names)
    f = _row_function(
        template=template,
        template_args=template_args,
        parameter_names=parameter_names,
        input_names=names,
        basis=basis,
        model=model,
        theta=theta,
    )
    design = _as_rows(design_rows, names)
    F_design = np.array([f(r) for r in design], dtype=np.float64)
    M_inv = _information_inverse(F_design)
    F = np.array([f(r) for r in _as_rows(points, names)], dtype=np.float64)
    v = np.einsum("ij,jk,ik->i", F, M_inv, F)
    return v, len(design)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def prediction_variance(
    design_rows: Rows,
    points: Rows,
    *,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[np.ndarray], np.ndarray] | None = None,
    model: Any | None = None,
    theta: Mapping[str, float] | None = None,
    sigma: float = 1.0,
) -> np.ndarray:
    """Variance of the fitted response, ``σ² · f(x)ᵀ (XᵀX)⁻¹ f(x)``, at each point.

    With the default ``sigma=1`` this is the *relative* prediction variance
    ``v(x)``, a property of the design and model alone.

    Parameters
    ----------
    design_rows : sequence of mapping, or array
        The design: one mapping per run (extra keys such as ``run_order`` are
        ignored), or an ``(n_runs, k)`` array ordered by ``input_names``.
    points : sequence of mapping, or array
        Where to evaluate the variance.
    template, template_args, parameter_names
        A linear template, as in :func:`~discopt.doe.linear_design.design_row`.
        ``parameter_names`` defaults to the template's own names.
    input_names : sequence of str, optional
        Factor order. Inferred from the numeric keys of the first design row
        when omitted; required when arrays are passed.
    basis : callable, optional
        ``f(x_vector) -> basis row`` instead of a template.
    model, theta : SymbolicModel and mapping, optional
        A user-defined model and nominal parameters, instead of a template.
        For a model nonlinear in its parameters the variance is the linearized
        one, local to ``theta``.
    sigma : float, default 1.0
        Measurement-error standard deviation.

    Returns
    -------
    numpy.ndarray
        One variance per point.

    Raises
    ------
    ValueError
        If the design cannot estimate the model (singular ``XᵀX``).
    """
    s = float(sigma)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    v, _ = _relative_variance(
        design_rows,
        points,
        template=template,
        template_args=template_args,
        parameter_names=parameter_names,
        input_names=input_names,
        basis=basis,
        model=model,
        theta=theta,
    )
    return s * s * v


def scaled_prediction_variance(
    design_rows: Rows,
    points: Rows,
    *,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[np.ndarray], np.ndarray] | None = None,
    model: Any | None = None,
    theta: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Scaled prediction variance ``SPV(x) = N · f(x)ᵀ (XᵀX)⁻¹ f(x)``.

    Multiplying by the run count ``N`` puts designs of different sizes on a
    per-run footing: a design that halves the variance by doubling its runs
    scores the same. For an orthogonal first-order design, SPV at the centre
    is 1. Arguments are as :func:`prediction_variance`.
    """
    v, n = _relative_variance(
        design_rows,
        points,
        template=template,
        template_args=template_args,
        parameter_names=parameter_names,
        input_names=input_names,
        basis=basis,
        model=model,
        theta=theta,
    )
    return n * v


def region_points(
    input_names: Sequence[str],
    n_samples: int,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_total: float | None = None,
    mixture_bounds: Mapping[str, tuple[float, float]] | None = None,
    seed: int | None = 0,
) -> list[dict[str, float]]:
    """Sample a design region uniformly: a box, or a mixture simplex.

    Give ``bounds`` for a box, or ``mixture_total`` for the simplex
    ``sum(x) = total, x >= 0`` (optionally with per-component
    ``mixture_bounds``; bounded samples are clipped and rescaled, which is
    not exactly uniform over a tightly constrained region).
    """
    names = list(input_names)
    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    if (bounds is None) == (mixture_total is None):
        raise ValueError("give exactly one of bounds= (a box) or mixture_total= (a simplex)")
    if bounds is not None:
        missing = [nm for nm in names if nm not in bounds]
        if missing:
            raise ValueError(f"bounds missing entries for {missing}")
        lo = np.array([float(bounds[nm][0]) for nm in names])
        hi = np.array([float(bounds[nm][1]) for nm in names])
        if np.any(hi <= lo):
            raise ValueError("bounds must satisfy ub > lb")
        X = rng.uniform(lo, hi, size=(n, len(names)))
        return [dict(zip(names, map(float, x))) for x in X]
    from discopt.doe.simplex import sample_simplex

    return [
        sample_simplex(
            names, float(mixture_total), rng, dict(mixture_bounds) if mixture_bounds else None
        )
        for _ in range(n)
    ]


@dataclass
class FDSResult:
    """A fraction-of-design-space curve.

    Attributes
    ----------
    fraction : numpy.ndarray
        Fraction of the region, from 0 to 1.
    spv : numpy.ndarray
        Scaled prediction variance, sorted ascending: ``spv[i]`` is the SPV
        that a fraction ``fraction[i]`` of the region does not exceed.
    quantiles : dict
        SPV at the 10/25/50/75/90 % fractions, plus ``"min"`` and ``"max"``.
    mean : float
        Mean SPV over the region (``N`` times the I-criterion).
    n_runs : int
        Runs in the design (the ``N`` in the scaling).
    """

    fraction: np.ndarray
    spv: np.ndarray
    quantiles: dict[str, float]
    mean: float
    n_runs: int

    def summary(self) -> str:
        q = self.quantiles
        return (
            f"FDS over {len(self.spv)} points, N = {self.n_runs}: "
            f"median SPV {q['0.5']:.3g}, 90% of region <= {q['0.9']:.3g}, "
            f"max {q['max']:.3g}, mean {self.mean:.3g}"
        )


def fds_curve(
    design_rows: Rows,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_total: float | None = None,
    mixture_bounds: Mapping[str, tuple[float, float]] | None = None,
    points: Rows | None = None,
    n_samples: int = 2000,
    seed: int | None = 0,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[np.ndarray], np.ndarray] | None = None,
    model: Any | None = None,
    theta: Mapping[str, float] | None = None,
) -> FDSResult:
    """Fraction-of-design-space curve of the scaled prediction variance.

    The region is a box (``bounds``), a mixture simplex (``mixture_total``),
    or explicit ``points``; boxes and simplices are sampled uniformly with
    ``n_samples`` points. Plot ``result.fraction`` against ``result.spv``: a
    flatter, lower curve is a design that predicts evenly and well over more of
    the region (Zahran, Anderson-Cook & Myers 2003). Model arguments are as
    :func:`prediction_variance`.
    """
    names = _infer_input_names(design_rows, input_names)
    if points is None:
        points = region_points(
            names,
            n_samples,
            bounds=bounds,
            mixture_total=mixture_total,
            mixture_bounds=mixture_bounds,
            seed=seed,
        )
    spv = np.sort(
        scaled_prediction_variance(
            design_rows,
            points,
            template=template,
            template_args=template_args,
            parameter_names=parameter_names,
            input_names=names,
            basis=basis,
            model=model,
            theta=theta,
        )
    )
    m = len(spv)
    fraction = np.arange(1, m + 1) / m if m > 1 else np.array([1.0])
    n_runs = len(design_rows)
    quantiles = {f"{q}": float(np.quantile(spv, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    quantiles["min"] = float(spv[0])
    quantiles["max"] = float(spv[-1])
    return FDSResult(
        fraction=fraction, spv=spv, quantiles=quantiles, mean=float(spv.mean()), n_runs=n_runs
    )


def _criterion_points(
    design_rows: Rows,
    names: list[str],
    points: Rows | None,
    bounds,
    mixture_total,
    mixture_bounds,
    n_samples: int,
    seed,
) -> Rows:
    if points is not None:
        return points
    return region_points(
        names,
        n_samples,
        bounds=bounds,
        mixture_total=mixture_total,
        mixture_bounds=mixture_bounds,
        seed=seed,
    )


def i_criterion(
    design_rows: Rows,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_total: float | None = None,
    mixture_bounds: Mapping[str, tuple[float, float]] | None = None,
    points: Rows | None = None,
    n_samples: int = 4000,
    seed: int | None = 0,
    scaled: bool = False,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[np.ndarray], np.ndarray] | None = None,
    model: Any | None = None,
    theta: Mapping[str, float] | None = None,
) -> float:
    """I-criterion: the average of ``v(x)`` over the region (smaller is better).

    Estimated by the mean over ``points`` or over a uniform sample of the
    region. ``scaled=True`` multiplies by the run count (mean SPV).
    """
    names = _infer_input_names(design_rows, input_names)
    pts = _criterion_points(
        design_rows, names, points, bounds, mixture_total, mixture_bounds, n_samples, seed
    )
    v, n = _relative_variance(
        design_rows,
        pts,
        template=template,
        template_args=template_args,
        parameter_names=parameter_names,
        input_names=names,
        basis=basis,
        model=model,
        theta=theta,
    )
    return float(v.mean() * (n if scaled else 1))


def g_criterion(
    design_rows: Rows,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_total: float | None = None,
    mixture_bounds: Mapping[str, tuple[float, float]] | None = None,
    points: Rows | None = None,
    n_samples: int = 4000,
    seed: int | None = 0,
    scaled: bool = False,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[np.ndarray], np.ndarray] | None = None,
    model: Any | None = None,
    theta: Mapping[str, float] | None = None,
) -> float:
    """G-criterion: the maximum of ``v(x)`` over the region (smaller is better).

    The maximum is taken over ``points``, or over a uniform sample of the region
    together with the box corners (where polynomial prediction variance
    usually peaks). A sample can only under-estimate a maximum, so pass an
    explicit candidate grid when the exact value matters. ``scaled=True``
    multiplies by the run count.
    """
    names = _infer_input_names(design_rows, input_names)
    pts = _criterion_points(
        design_rows, names, points, bounds, mixture_total, mixture_bounds, n_samples, seed
    )
    if points is None and bounds is not None:
        from itertools import product

        corners = [
            dict(zip(names, c))
            for c in product(*[(float(bounds[n][0]), float(bounds[n][1])) for n in names])
        ]
        pts = list(pts) + corners
    v, n = _relative_variance(
        design_rows,
        pts,
        template=template,
        template_args=template_args,
        parameter_names=parameter_names,
        input_names=names,
        basis=basis,
        model=model,
        theta=theta,
    )
    return float(v.max() * (n if scaled else 1))
