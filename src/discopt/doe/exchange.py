"""Choose runs from a finite candidate list, and compare designs by efficiency.

The continuous searches in :mod:`discopt.doe.linear_design` place runs anywhere
in a box. Often the runs must come from a list instead: the settings a machine
actually offers, the materials on the shelf, the grid points of an irregular
region, or the level combinations of categorical factors. The classical tool
for that is **point exchange** (Fedorov 1972): start from some ``n`` candidates
and repeatedly swap a design run for the candidate that most improves the
criterion, until no swap helps. :func:`candidate_exchange_design` implements the
modified Fedorov exchange (Cook & Nachtsheim 1980) with random restarts, using
the rank-one determinant and inverse updates so that every sweep scores all
candidates at once.

The candidates can be anything the model can turn into a row: mappings of
factor values (including text levels, with a ``basis`` that one-hot codes
them), an ``(n, k)`` array, or precomputed model rows.

:func:`relative_efficiency` (with :func:`d_efficiency` and
:func:`i_efficiency`) answers the practical question: *how many runs does this
design waste compared with that one?* An efficiency of 0.8 means the design
needs about 25 % more runs to match the reference on that criterion.

Only numpy is needed, so the module runs in the browser build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from discopt.doe.linear_design import (
    A_OPTIMAL,
    D_OPTIMAL,
    G_OPTIMAL,
    I_OPTIMAL,
    DesignRegion,
    design_region,
    design_row,
    evaluate_criterion,
    is_maximized,
    normalize_criterion,
)

__all__ = [
    "ExchangeDesignResult",
    "candidate_exchange_design",
    "d_efficiency",
    "i_efficiency",
    "relative_efficiency",
]

_EXCHANGE_CRITERIA = (D_OPTIMAL, A_OPTIMAL, I_OPTIMAL, G_OPTIMAL)
_RIDGE = 1e-8


@dataclass
class ExchangeDesignResult:
    """A design chosen from a candidate list.

    Attributes
    ----------
    designs : list
        The chosen candidates, one per run, in the form they were given
        (mappings stay mappings). A candidate chosen twice appears twice.
    indices : list of int
        Index of each run in the candidate list.
    joint_fim : numpy.ndarray
        Information of the design, ``prior_fim`` included.
    criterion_value : float
        Criterion at the returned design.
    criterion : str
        Canonical criterion name.
    model_rows : numpy.ndarray
        ``(n, p)`` model rows of the chosen runs.
    n_sweeps : int
        Exchange sweeps used by the best restart.
    parameter_names : list of str
    """

    designs: list[Any]
    indices: list[int]
    joint_fim: np.ndarray
    criterion_value: float
    criterion: str
    model_rows: np.ndarray
    n_sweeps: int
    parameter_names: list[str] = field(default_factory=list)

    @property
    def n_experiments(self) -> int:
        return len(self.designs)

    @property
    def parameter_covariance(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.joint_fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.joint_fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        return np.sqrt(np.diag(self.parameter_covariance))


# --------------------------------------------------------------------------
# Model rows
# --------------------------------------------------------------------------


def _rows_function(
    *,
    template: str | None,
    template_args: Mapping[str, Any] | None,
    parameter_names: Sequence[str] | None,
    input_names: Sequence[str] | None,
    basis: Callable[[Any], np.ndarray] | None,
) -> Callable[[Any], np.ndarray]:
    if (template is None) == (basis is None):
        raise ValueError("give exactly one of template= or basis= (or pass model_rows=)")
    if basis is not None:
        return lambda c: np.asarray(basis(c), dtype=np.float64)
    if input_names is None:
        raise ValueError("template= needs input_names=")
    args = dict(template_args or {})
    names = list(input_names)
    if parameter_names is None:
        from discopt.doe.templates import template_parameter_names

        parameter_names = template_parameter_names(
            template, len(names), degree=args.get("degree"), basis=args.get("basis")
        )
    pnames = list(parameter_names)

    def f(c: Any) -> np.ndarray:
        row = (
            {n: float(c[n]) for n in names}
            if isinstance(c, Mapping)
            else dict(zip(names, map(float, np.asarray(c, dtype=np.float64).ravel())))
        )
        return design_row(template, args, pnames, names, row)

    return f


def _candidate_list(candidates: Any) -> list[Any]:
    if isinstance(candidates, np.ndarray):
        return list(np.atleast_2d(candidates))
    return list(candidates)


# --------------------------------------------------------------------------
# Exchange
# --------------------------------------------------------------------------


def _score(M: np.ndarray, criterion: str, region: DesignRegion | None) -> float:
    value = evaluate_criterion(M, criterion, region=region)
    if not np.isfinite(value):
        return -np.inf
    return value if is_maximized(criterion) else -value


def _swap_scores(
    Minv: np.ndarray,
    fi: np.ndarray,
    Fc: np.ndarray,
    criterion: str,
    W: np.ndarray | None,
) -> np.ndarray:
    """Change in the (maximized) criterion from swapping row ``fi`` for each candidate.

    Closed-form rank-one updates. For D the determinant lemma gives the ratio
    ``det M' / det M``; for A and I, Sherman-Morrison twice gives
    ``trace(W M'^-1)`` for every candidate at once.
    """
    d_i = fi @ Minv @ fi
    if criterion == D_OPTIMAL:
        MF = Fc @ Minv  # (m, p)
        d_j = np.einsum("ij,ij->i", MF, Fc)
        d_ij = MF @ fi
        ratio = (1.0 + d_j) * (1.0 - d_i) + d_ij**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ratio > 0, np.log(ratio), -np.inf)
    # A and I: remove fi, then add each candidate.
    denom = 1.0 - d_i
    if denom <= 1e-12:
        return np.full(Fc.shape[0], -np.inf)
    u = Minv @ fi
    Mminus = Minv + np.outer(u, u) / denom
    Wm = np.eye(Minv.shape[0]) if W is None else W
    base = float(np.trace(Wm @ Mminus))
    G = Fc @ Mminus  # (m, p)
    num = np.einsum("ij,jk,ik->i", G, Wm, G)
    den = 1.0 + np.einsum("ij,ij->i", G, Fc)
    new_trace = base - num / den
    old_trace = float(np.trace(Wm @ Minv))
    return old_trace - new_trace  # positive = improvement (trace decreases)


def candidate_exchange_design(
    candidates: Sequence[Any] | np.ndarray,
    n: int,
    *,
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[Any], np.ndarray] | None = None,
    model_rows: np.ndarray | None = None,
    criterion: str = "D",
    prior_fim: np.ndarray | None = None,
    measurement_error: float = 1.0,
    region: DesignRegion | Sequence[Any] | np.ndarray | None = None,
    allow_repeats: bool = True,
    n_starts: int = 10,
    max_sweeps: int = 100,
    seed: int = 0,
) -> ExchangeDesignResult:
    """Choose ``n`` runs from a finite candidate list by point exchange.

    Modified Fedorov exchange (Fedorov 1972; Cook & Nachtsheim 1980): from a
    random start of ``n`` candidates, sweep through the design runs and replace
    each by the candidate that most improves the criterion, until a full sweep
    finds no improvement. ``n_starts`` random restarts guard against local
    optima; the best result is returned.

    Parameters
    ----------
    candidates : sequence or array
        The runs you may choose from: mappings of factor values, an ``(m, k)``
        array ordered by ``input_names``, or any objects ``basis`` accepts
        (e.g. mappings with text levels for categorical factors).
    n : int
        Runs to choose.
    template, template_args, parameter_names, input_names
        A linear template, as in :func:`~discopt.doe.linear_design.design_row`.
    basis : callable, optional
        ``basis(candidate) -> model row``, instead of a template.
    model_rows : numpy.ndarray, optional
        Precomputed ``(m, p)`` model rows, one per candidate, instead of either.
    criterion : str, default ``"D"``
        ``"D"``, ``"A"``, ``"I"`` or ``"G"`` (or their canonical names). I and
        G need a region: ``region`` (a :class:`DesignRegion`, or region points
        in candidate form) or, by default, the candidate set itself.
    prior_fim : numpy.ndarray, optional
        Information from runs already made; the new runs complement it.
    measurement_error : float, default 1.0
        Response standard deviation.
    allow_repeats : bool, default True
        Whether a candidate may be chosen more than once (replication).
    n_starts, max_sweeps, seed
        Random restarts, a cap on sweeps per restart, and the seed.

    Returns
    -------
    ExchangeDesignResult

    Notes
    -----
    Starts with fewer independent candidates than parameters are singular; the
    first sweeps then maximize a lightly regularized determinant, which fills
    in the missing directions, before the requested criterion takes over.
    """
    crit = normalize_criterion(criterion)
    if crit not in _EXCHANGE_CRITERIA:
        raise ValueError(f"candidate exchange supports D, A, I and G, not {criterion!r}")
    items = _candidate_list(candidates)
    m = len(items)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not allow_repeats and n > m:
        raise ValueError(f"cannot choose {n} distinct runs from {m} candidates")

    if model_rows is not None:
        Fc = np.asarray(model_rows, dtype=np.float64)
        if Fc.shape[0] != m:
            raise ValueError(f"model_rows has {Fc.shape[0]} rows for {m} candidates")
        row_fn = None
    else:
        row_fn = _rows_function(
            template=template,
            template_args=template_args,
            parameter_names=parameter_names,
            input_names=input_names,
            basis=basis,
        )
        Fc = np.array([row_fn(c) for c in items], dtype=np.float64)
    p = Fc.shape[1]
    inv_var = 1.0 / float(measurement_error) ** 2
    Fs = Fc * np.sqrt(inv_var)  # rows scaled so that M = prior + Fs[idx]^T Fs[idx]
    prior = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=np.float64)
    if prior.shape != (p, p):
        raise ValueError(f"prior_fim shape {prior.shape} does not match {p} parameters")

    region_obj: DesignRegion | None = None
    if crit in (I_OPTIMAL, G_OPTIMAL):
        if isinstance(region, DesignRegion):
            region_obj = region
        else:
            pts = items if region is None else _candidate_list(region)
            if region is not None and row_fn is None:
                raise ValueError("region points need template= or basis= to become model rows")
            R = Fc if region is None else np.array([row_fn(c) for c in pts])  # type: ignore[misc]
            region_obj = DesignRegion(
                R,
                np.full(R.shape[0], 1.0 / R.shape[0]),
                R,
                np.zeros((R.shape[0], 0)),
                np.zeros((R.shape[0], 0)),
                "points",
            )
    W = region_obj.moment_matrix if (crit == I_OPTIMAL and region_obj is not None) else None

    typical = np.mean(Fs**2, axis=0)
    typical = np.where(typical > 0, typical, max(float(typical.max()), 1.0))
    ridge = np.diag(_RIDGE * typical)

    rng = np.random.default_rng(seed)
    best: tuple[float, list[int], int] | None = None
    for _ in range(max(1, int(n_starts))):
        idx = list(rng.choice(m, size=n, replace=bool(allow_repeats) or n > m))
        sweeps = 0
        for stage in dict.fromkeys([D_OPTIMAL if crit == G_OPTIMAL else crit, crit]):
            for _sweep in range(int(max_sweeps)):
                sweeps += 1
                improved = False
                for i in range(n):
                    M = prior + Fs[idx].T @ Fs[idx]
                    singular = np.linalg.matrix_rank(M) < p
                    use = D_OPTIMAL if singular else stage
                    Mr = M + ridge if singular else M
                    try:
                        Minv = np.linalg.inv(Mr)
                    except np.linalg.LinAlgError:
                        Minv = np.linalg.pinv(Mr)
                    if use == G_OPTIMAL:
                        gains = _g_gains(M, Fs, idx, i, region_obj)
                    else:
                        gains = _swap_scores(Minv, Fs[idx[i]], Fs, use, W)
                    if not allow_repeats:
                        taken = set(idx) - {idx[i]}
                        gains = np.where(np.isin(np.arange(m), list(taken)), -np.inf, gains)
                    j = int(np.argmax(gains))
                    if j != idx[i] and gains[j] > 1e-10:
                        idx[i] = j
                        improved = True
                if not improved:
                    break
        M = prior + Fs[idx].T @ Fs[idx]
        score = _score(M, crit, region_obj)
        if best is None or score > best[0]:
            best = (score, list(idx), sweeps)

    assert best is not None
    score, idx, sweeps = best
    M = prior + Fs[idx].T @ Fs[idx]
    return ExchangeDesignResult(
        designs=[items[i] for i in idx],
        indices=[int(i) for i in idx],
        joint_fim=M,
        criterion_value=evaluate_criterion(M, crit, region=region_obj),
        criterion=crit,
        model_rows=Fc[idx],
        n_sweeps=sweeps,
        parameter_names=list(parameter_names) if parameter_names is not None else [],
    )


def _g_gains(
    M: np.ndarray, Fs: np.ndarray, idx: list[int], i: int, region: DesignRegion | None
) -> np.ndarray:
    """Improvement in G (max variance) from swapping run ``i`` for each candidate."""
    assert region is not None
    cur = region.max_variance(M)
    fi = Fs[idx[i]]
    out = np.full(Fs.shape[0], -np.inf)
    for j in range(Fs.shape[0]):
        Mj = M - np.outer(fi, fi) + np.outer(Fs[j], Fs[j])
        v = region.max_variance(Mj)
        if np.isfinite(v):
            out[j] = cur - v
    return out


# --------------------------------------------------------------------------
# Efficiency
# --------------------------------------------------------------------------


def _design_fim(
    design: Any,
    row_fn: Callable[[Any], np.ndarray] | None,
) -> tuple[np.ndarray, int]:
    """Per-run information ``XᵀX / N`` and ``N`` for a design in any accepted form."""
    if hasattr(design, "model_rows"):
        F = np.asarray(design.model_rows, dtype=np.float64)
    else:
        items = design.designs if hasattr(design, "designs") else design
        items = _candidate_list(items)
        if row_fn is None:
            F = np.atleast_2d(np.asarray(items, dtype=np.float64))
        else:
            F = np.array([row_fn(c) for c in items], dtype=np.float64)
    N = F.shape[0]
    return F.T @ F / N, N


def relative_efficiency(
    design: Any,
    reference: Any,
    *,
    criterion: str = "D",
    template: str | None = None,
    template_args: Mapping[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    basis: Callable[[Any], np.ndarray] | None = None,
    region: DesignRegion | Sequence[Any] | np.ndarray | None = None,
    region_bounds: Mapping[str, tuple[float, float]] | None = None,
    per_run: bool = True,
) -> float:
    """Efficiency of ``design`` relative to ``reference`` on one criterion.

    ``> 1`` means ``design`` is better. With ``per_run=True`` (the usual
    definition) both are normalized by their run counts, so the efficiency reads
    as a ratio of runs: 0.8 means ``design`` needs about 1/0.8 = 1.25 times as
    many runs to match ``reference``.

    * D: ``(det M_design / det M_ref)^(1/p)`` with ``M = XᵀX / N``.
    * A: ``trace(M_ref⁻¹) / trace(M_design⁻¹)``.
    * I: ``I_ref / I_design``, average prediction variance over ``region`` (or a
      box ``region_bounds``).
    * G: ``G_ref / G_design``, maximum prediction variance over the region.

    Designs may be run rows (mappings or an array), model rows (an array, with
    neither ``template`` nor ``basis``), or any result object with
    ``.designs`` or ``.model_rows``.
    """
    crit = normalize_criterion(criterion)
    row_fn = (
        _rows_function(
            template=template,
            template_args=template_args,
            parameter_names=parameter_names,
            input_names=input_names,
            basis=basis,
        )
        if (template is not None or basis is not None)
        else None
    )
    Ma, Na = _design_fim(design, row_fn)
    Mb, Nb = _design_fim(reference, row_fn)
    if not per_run:
        Ma, Mb = Ma * Na, Mb * Nb
    p = Ma.shape[0]
    if crit == D_OPTIMAL:
        la, lb = np.linalg.slogdet(Ma), np.linalg.slogdet(Mb)
        if la[0] <= 0 or lb[0] <= 0:
            raise ValueError("a design cannot estimate the model (singular information)")
        return float(np.exp((la[1] - lb[1]) / p))
    if crit == A_OPTIMAL:
        return float(np.trace(np.linalg.inv(Mb)) / np.trace(np.linalg.inv(Ma)))
    if crit in (I_OPTIMAL, G_OPTIMAL):
        if isinstance(region, DesignRegion):
            reg = region
        elif region is not None:
            if row_fn is None:
                raise ValueError("region points need template= or basis=")
            R = np.array([row_fn(c) for c in _candidate_list(region)])
            reg = DesignRegion(
                R,
                np.full(R.shape[0], 1.0 / R.shape[0]),
                R,
                np.zeros((R.shape[0], 0)),
                np.zeros((R.shape[0], 0)),
                "points",
            )
        else:
            if region_bounds is None or template is None or input_names is None:
                raise ValueError(
                    "I/G efficiency needs region= or region_bounds= with template= and input_names="
                )
            names = list(input_names)

            def vec_basis(x: np.ndarray) -> np.ndarray:
                return row_fn(dict(zip(names, map(float, x))))  # type: ignore[misc]

            reg = design_region(vec_basis, names, bounds=region_bounds)
        va = evaluate_criterion(Ma, crit, region=reg)
        vb = evaluate_criterion(Mb, crit, region=reg)
        return float(vb / va)
    raise ValueError(f"efficiency is defined here for D, A, I and G, not {criterion!r}")


def d_efficiency(design: Any, reference: Any, **kwargs: Any) -> float:
    """D-efficiency of ``design`` relative to ``reference`` (see :func:`relative_efficiency`)."""
    return relative_efficiency(design, reference, criterion="D", **kwargs)


def i_efficiency(design: Any, reference: Any, **kwargs: Any) -> float:
    """I-efficiency of ``design`` relative to ``reference`` (see :func:`relative_efficiency`)."""
    return relative_efficiency(design, reference, criterion="I", **kwargs)
