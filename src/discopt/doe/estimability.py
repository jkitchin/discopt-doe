"""Parameter estimability ranking and subset selection.

Tools for the chemical-engineering estimability literature:

- ``estimability_rank`` : Yao et al. (2003) orthogonalization ranking via
  rank-revealing QR on the scaled sensitivity matrix.
- ``collinearity_index`` : Brun, Reichert & Kuensch (2001) collinearity
  index gamma_K for a user-specified parameter subset.
- ``mse_subset_selection`` : Wu, McAuley & Harris (2011) choice of *how
  many* ranked parameters to estimate, from data, by the corrected critical
  ratio (a mean-squared-error trade-off between bias and variance).
- ``d_optimal_subset`` : Chu & Hahn (2007, 2012) D-optimal subset
  selection. ``method="auto"`` dispatches to enumeration for small
  problems and to the greedy Yao ranking for larger ones. A MINLP
  variant is reserved for a future release; see the ``method`` argument.

Scaling convention
------------------
The scaled sensitivity matrix Z follows Brun's recipe,

    Z = Sigma^{-1/2} J diag(s_theta)

so each column j measures "observable change per meaningful parameter
perturbation" and each row is noise-weighted. ``s_theta_j`` defaults to
``|theta_j|`` (relative scaling) with a floor of ``eps``; callers can
override via ``parameter_scales``. Measurement errors come from
``ExperimentModel.measurement_error``.

Reparameterization warning
--------------------------
The Yao ranking and the Brun collinearity index are *not* invariant
under a reparameterization ``theta -> log theta`` or any other nonlinear
change of variable. The ranking is a statement about the user's
parameterization. Profile likelihood (:mod:`discopt.doe.profile`) *is*
reparameterization-invariant and is the tool of choice when the
parameterization itself is in question.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import scipy.linalg

from discopt.doe.fim import compute_fim
from discopt.estimate import Experiment


@dataclass
class EstimabilityResult:
    """Result of a Yao-style estimability analysis.

    Attributes
    ----------
    ranking : list[str]
        Parameters ordered from most to least estimable.
    projected_norms : numpy.ndarray
        ``|diag(R)|`` from the pivoted QR, in ``ranking`` order. Matches
        Yao's projected 2-norms numerically.
    recommended_subset : list[str]
        Parameters above the user-specified cutoff.
    collinearity_index : float
        Brun gamma_K of the recommended subset.
    parameter_names : list[str]
        Original (unranked) parameter order, for reference.
    raw_norms : numpy.ndarray
        Unprojected 2-norms of the scaled sensitivity columns, in ``ranking``
        order. Compare with ``projected_norms``: a parameter weak in both has a
        small effect; one strong raw but weak projected is collinear with
        parameters ranked above it.
    method : str
        ``"cutoff"`` (the recommended subset uses ``cutoff``) or ``"mse"``
        (it uses :func:`mse_subset_selection`).
    mse : MSESubsetResult or None
        The mean-squared-error table when ``method="mse"``.
    """

    ranking: list[str]
    projected_norms: np.ndarray
    recommended_subset: list[str]
    collinearity_index: float
    parameter_names: list[str]
    raw_norms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = "cutoff"
    mse: MSESubsetResult | None = None


@dataclass
class MSESubsetResult:
    """How many ranked parameters to estimate: Wu et al. (2011).

    For each ``k``, the top ``k`` ranked parameters are estimated and the rest
    held at their nominal values, giving the weighted least-squares objective
    ``J_k``. With ``p`` parameters and ``N`` observations,

    - critical ratio ``r_C,k = (J_k - J_p) / (p - k)``;
    - truncated estimate ``r_CKub,k = max(r_C,k - 1, 2 r_C,k / (p - k + 2))``;
    - corrected critical ratio ``r_CC,k = (p - k) / N * (r_CKub,k - 1)``,
      with ``r_CC,p = 0`` (estimate everything).

    The recommended ``k`` has the lowest ``r_CC,k``: it is the simplified model
    expected to give the most accurate predictions, trading the bias of fixing
    parameters against the variance of estimating them. ``r_C,k`` near 1 means
    the fixed values are consistent with the data; much larger means fixing
    them biases the fit. The objective must be weighted by the true measurement
    uncertainties (the ``measurement_error`` of the experiment) for the ratios
    to be calibrated.

    Attributes
    ----------
    ranking : list[str]
        Parameters in estimability order.
    k : numpy.ndarray
        ``1 .. p``.
    objectives : numpy.ndarray
        ``J_k``.
    critical_ratios : numpy.ndarray
        ``r_C,k`` (``nan`` at ``k = p``).
    corrected_ratios : numpy.ndarray
        ``r_CC,k``.
    recommended_k : int
    recommended_subset : list[str]
    n_observations : int
    """

    ranking: list[str]
    k: np.ndarray
    objectives: np.ndarray
    critical_ratios: np.ndarray
    corrected_ratios: np.ndarray
    recommended_k: int
    recommended_subset: list[str]
    n_observations: int

    def summary(self) -> str:
        lines = [f"{'k':>3s} {'estimated (added)':<22s} {'J_k':>12s} {'r_C':>10s} {'r_CC':>10s}"]
        for i, kk in enumerate(self.k):
            mark = "  <- recommended" if int(kk) == self.recommended_k else ""
            rc = self.critical_ratios[i]
            rc_s = "       ---" if not np.isfinite(rc) else f"{rc:10.3f}"
            lines.append(
                f"{int(kk):3d} {'+' + self.ranking[int(kk) - 1]:<22s} "
                f"{self.objectives[i]:12.4f} {rc_s} {self.corrected_ratios[i]:10.4f}{mark}"
            )
        return "\n".join(lines)


def _scaled_sensitivity(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None,
    parameter_scales: dict[str, float] | None,
    noise_covariance: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Return the Brun-scaled sensitivity matrix Z and parameter order.

    Uses :func:`compute_fim` for the Jacobian, then applies
    ``Sigma^{-1/2}`` (diagonal noise-weighting) and ``diag(s_theta)``.
    """
    fim_result = compute_fim(experiment, param_values, design_values)
    J = np.asarray(fim_result.jacobian, dtype=np.float64)
    names = list(fim_result.parameter_names)

    em = experiment.create_model(**param_values)
    if noise_covariance is None:
        sigma = np.array(
            [em.measurement_error[name] for name in fim_result.response_names],
            dtype=np.float64,
        )
        J_weighted = J / sigma[:, np.newaxis]
    else:
        Sigma = np.asarray(noise_covariance, dtype=np.float64)
        L = np.linalg.cholesky(Sigma)
        J_weighted = scipy.linalg.solve_triangular(L, J, lower=True)

    # Parameter-axis scales. A nominal value of 0 (common for offsets) has no
    # meaningful magnitude scale; the old ``abs(...) or eps`` fallback used
    # machine-eps, which zeroed that parameter's Z column and always ranked it
    # unestimable regardless of its true sensitivity. Fall back to 1.0 with a
    # warning instead so the QR sees the real sensitivity.
    scales_list: list[float] = []
    for name in names:
        override = (parameter_scales or {}).get(name)
        s = abs(float(override)) if override is not None else abs(float(param_values[name]))
        if s == 0.0:
            warnings.warn(
                f"parameter {name!r} has scale 0 (nominal value 0 and no "
                "parameter_scales override); using 1.0. Pass parameter_scales "
                "for a meaningful axis scale.",
                stacklevel=2,
            )
            s = 1.0
        scales_list.append(s)
    scales = np.array(scales_list, dtype=np.float64)
    Z = J_weighted * scales[np.newaxis, :]
    return Z, names


def estimability_rank(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    cutoff: float = 0.04,
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
    method: Literal["cutoff", "mse"] = "cutoff",
    data: Mapping[str, Any] | None = None,
    _cache: tuple[np.ndarray, list[str]] | None = None,
) -> EstimabilityResult:
    """Rank parameters by estimability (Yao et al. 2003).

    Uses rank-revealing QR on the Brun-scaled sensitivity matrix Z.
    The permutation returned by ``scipy.linalg.qr(..., pivoting=True)``
    coincides with the Yao orthogonalization order, and ``|diag(R_kk)|``
    equals Yao's projected 2-norm at step k exactly.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_values : dict[str, float], optional
        Fixed design conditions.
    cutoff : float, default 0.04
        Cutoff for the recommended subset. A parameter is included if
        ``|R_kk| / |R_11| >= cutoff`` -- a *relative* (scale-invariant)
        variant of Yao's rule, thresholding each projected column against the
        largest pivot rather than against an absolute value. Yao et al. (2003)
        apply the 0.04 threshold to the projected magnitude ``|R_kk|`` itself;
        the two agree only when ``|R_11| ~ 1`` (well-scaled Z). The 0.04
        default is carried over as a rule of thumb; adjust it if you rely on
        the absolute Yao criterion.
    parameter_scales : dict[str, float], optional
        Override parameter-axis scales ``s_theta``. Defaults to
        ``|param_values|``.
    noise_covariance : numpy.ndarray, optional
        Full noise covariance. Defaults to the diagonal
        ``ExperimentModel.measurement_error`` from ``experiment``.
    method : {"cutoff", "mse"}, default "cutoff"
        How the recommended subset is chosen. ``"cutoff"`` applies ``cutoff``
        to the projected norms (needs no data). ``"mse"`` fits the ranked
        subsets to ``data`` and picks the size with the lowest corrected
        critical ratio (Wu et al. 2011; see :func:`mse_subset_selection`).
    data : mapping, optional
        Observed responses, required for ``method="mse"``.

    Returns
    -------
    EstimabilityResult
        Ranking, projected and raw norms, recommended subset, collinearity
        index (and the MSE table for ``method="mse"``).
    """
    if method not in ("cutoff", "mse"):
        raise ValueError(f"method must be 'cutoff' or 'mse', got {method!r}")
    if method == "mse" and data is None:
        raise ValueError("method='mse' needs the observed data (data=...)")
    if _cache is not None:
        Z, names = _cache
    else:
        Z, names = _scaled_sensitivity(
            experiment, param_values, design_values, parameter_scales, noise_covariance
        )
    n_params = len(names)
    if n_params == 0:
        return EstimabilityResult([], np.zeros(0), [], 1.0, [])

    _, R, piv = scipy.linalg.qr(Z, pivoting=True, mode="economic")
    k = min(R.shape)
    projected = np.zeros(n_params)
    projected[:k] = np.abs(np.diag(R)[:k])

    ranking = [names[i] for i in piv]
    raw = np.linalg.norm(Z, axis=0)[piv]
    top = projected[0] if projected[0] > 0 else 1.0
    recommended = [ranking[i] for i in range(n_params) if projected[i] / top >= cutoff]
    mse_result = None
    if method == "mse":
        assert data is not None
        mse_result = mse_subset_selection(
            experiment, data, param_values, ranking=ranking, design_values=design_values
        )
        recommended = list(mse_result.recommended_subset)

    coll_idx = (
        collinearity_index(
            experiment,
            param_values,
            recommended,
            design_values,
            parameter_scales=parameter_scales,
            noise_covariance=noise_covariance,
            _cache=(Z, names),
        )
        if recommended
        else float("inf")
    )

    return EstimabilityResult(
        ranking=ranking,
        projected_norms=projected,
        recommended_subset=recommended,
        collinearity_index=coll_idx,
        parameter_names=names,
        raw_norms=raw,
        method=method,
        mse=mse_result,
    )


def _subset_deviances(
    experiment: Experiment,
    data: Mapping[str, Any],
    nominal: dict[str, float],
    ranking: Sequence[str],
    *,
    design_values: Mapping[str, float] | None,
    n_starts: int,
    extra: Mapping[str, Any],
) -> np.ndarray:
    """Minimized deviance with the top ``k`` of ``ranking`` free, for every ``k``.

    Uses the compiled deviance where the experiment has one, and the base
    estimator otherwise, so an experiment that needs a solve still works.
    """
    from scipy.optimize import minimize

    from discopt.doe._estimation import DevianceFunction, estimate_parameters, parameter_bounds

    p = len(ranking)
    J = np.zeros(p)

    dev: DevianceFunction | None = None
    try:
        dev = DevianceFunction(experiment, data, nominal, design=design_values)
    except Exception:  # noqa: BLE001 - any build problem: use the estimator
        dev = None
    if dev is not None and dev.path == "fallback":
        # The fallback path evaluates through the estimator anyway, so it would
        # be the same work with a worse optimizer.
        dev = None

    if dev is None:
        for k in range(1, p + 1):
            fixed = {name: nominal[name] for name in ranking[k:]}
            res = estimate_parameters(
                experiment,
                data,
                initial_guess=nominal,
                fixed_parameters=fixed or None,
                n_starts=n_starts,
                **dict(extra),
            )
            J[k - 1] = float(res.objective)
        return J

    names = list(dev.names)
    base = dev.vector(nominal)
    bounds_map = parameter_bounds(experiment, nominal)
    rng = np.random.default_rng(0)

    for k in range(1, p + 1):
        free = [names.index(n) for n in ranking[:k] if n in names]
        if not free:
            J[k - 1] = float(dev(base))
            continue
        box = [bounds_map.get(names[i], (-np.inf, np.inf)) for i in free]

        def value(z: np.ndarray, free=free) -> float:
            full = base.copy()
            full[free] = z
            out = float(dev(full))
            return out if np.isfinite(out) else 1e300

        def gradient(z: np.ndarray, free=free) -> np.ndarray | None:
            full = base.copy()
            full[free] = z
            g = dev.gradient(full)
            return None if g is None else np.asarray(g)[free]

        jac = gradient if dev.gradient(base) is not None else None
        starts = [base[free].copy()]
        for _ in range(max(0, int(n_starts) - 1)):
            starts.append(
                np.array(
                    [
                        rng.uniform(
                            max(lo, -abs(v) * 10 - 1.0) if np.isfinite(lo) else -abs(v) * 10 - 1.0,
                            min(hi, abs(v) * 10 + 1.0) if np.isfinite(hi) else abs(v) * 10 + 1.0,
                        )
                        for (lo, hi), v in zip(box, base[free])
                    ]
                )
            )
        best = np.inf
        for z0 in starts:
            res = minimize(
                value,
                z0,
                jac=jac,
                method="L-BFGS-B",
                bounds=[(lo, hi) for lo, hi in box],
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
            )
            best = min(best, float(res.fun))
        J[k - 1] = best
    return J


def mse_subset_selection(
    experiment: Experiment,
    data: Mapping[str, Any],
    param_values: Mapping[str, float],
    *,
    ranking: Sequence[str] | None = None,
    design_values: Mapping[str, float] | None = None,
    n_starts: int = 1,
) -> MSESubsetResult:
    """Choose how many ranked parameters to estimate (Wu, McAuley & Harris 2011).

    Fits the model ``p`` times: with the top ``k`` parameters of ``ranking``
    estimated and the rest held at ``param_values``, for ``k = 1 .. p``. The
    corrected critical ratio ``r_CC,k`` of each fit (see
    :class:`MSESubsetResult`) estimates how much the simplified model's
    prediction error exceeds the full model's, and the ``k`` with the lowest
    value is recommended. Unlike a fixed cutoff on the ranking, this uses the
    data: it keeps a weakly estimable parameter fixed when fixing it costs less
    in bias than estimating it costs in variance, and estimates it when the
    fixed value is visibly wrong.

    Parameters
    ----------
    experiment : Experiment
    data : mapping
        Observed responses (as for :func:`discopt.estimate.estimate_parameters`).
    param_values : mapping
        Nominal values; the non-estimated parameters are held here, and they
        start each fit.
    ranking : sequence of str, optional
        Estimability order. Defaults to :func:`estimability_rank` at
        ``param_values``.
    design_values : mapping, optional
        Design conditions for experiments whose fit needs them.
    n_starts : int, default 1
        Starting points per fit (see :func:`discopt.doe._estimation.estimate_parameters`).

    References
    ----------
    Wu, S., McAuley, K. B. & Harris, T. J. Selection of simplified models: II.
    Development of a model selection criterion based on mean squared error.
    *Can. J. Chem. Eng.* 89, 325-336 (2011).
    """
    if ranking is None:
        ranking = estimability_rank(experiment, dict(param_values), design_values).ranking
    ranking = list(ranking)
    p = len(ranking)
    if p == 0:
        raise ValueError("no parameters to rank")
    extra: dict[str, Any] = {"design": dict(design_values)} if design_values else {}
    nominal = {k: float(v) for k, v in param_values.items()}
    n_obs = int(sum(np.atleast_1d(np.asarray(v)).size for v in data.values()))

    # Every one of the p fits minimizes the *same* deviance over a different
    # subset of its coordinates, so the function is built once and reused. On a
    # compiled path that turns p full estimator solves -- each rebuilding the
    # model and going through the base solver -- into p bounded minimizations of
    # an already-jitted objective with an exact gradient.
    J = _subset_deviances(
        experiment,
        data,
        nominal,
        ranking,
        design_values=design_values,
        n_starts=n_starts,
        extra=extra,
    )

    Jp = J[-1]
    k_arr = np.arange(1, p + 1)
    rc = np.full(p, np.nan)
    rcc = np.zeros(p)
    for i, k in enumerate(k_arr[:-1]):
        r = (J[i] - Jp) / (p - k)
        rc[i] = r
        r_kub = max(r - 1.0, 2.0 * r / (p - k + 2.0))
        rcc[i] = (p - k) / n_obs * (r_kub - 1.0)
    best = int(k_arr[int(np.argmin(rcc))])
    return MSESubsetResult(
        ranking=ranking,
        k=k_arr,
        objectives=J,
        critical_ratios=rc,
        corrected_ratios=rcc,
        recommended_k=best,
        recommended_subset=ranking[:best],
        n_observations=n_obs,
    )


def collinearity_index(
    experiment: Experiment,
    param_values: dict[str, float],
    subset: list[str],
    design_values: dict[str, float] | None = None,
    *,
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
    _cache: tuple[np.ndarray, list[str]] | None = None,
) -> float:
    """Brun-Reichert-Kuensch collinearity index for a parameter subset.

    Defined as ``gamma_K = 1 / sqrt(lambda_min(Z_K^T Z_K))`` where Z_K
    has columns of Z restricted to ``subset`` and further rescaled to
    unit column length (Brun's choice). ``gamma_K`` above ~10 indicates
    collinearity so severe that the subset cannot be jointly estimated.

    Parameters
    ----------
    subset : list[str]
        Parameter names to include. Duplicates and unknown names raise
        ``ValueError``.
    Other parameters
        See :func:`estimability_rank`.

    Returns
    -------
    float
        The collinearity index. ``inf`` if Z_K is rank-deficient.

    Raises
    ------
    ValueError
        If ``subset`` contains duplicate or unrecognized parameter names.
    """
    if _cache is not None:
        Z, names = _cache
    else:
        Z, names = _scaled_sensitivity(
            experiment, param_values, design_values, parameter_scales, noise_covariance
        )
    duplicates = sorted({n for n in subset if subset.count(n) > 1})
    if duplicates:
        raise ValueError(f"subset contains duplicate parameter names: {duplicates}")
    unknown = [n for n in subset if n not in names]
    if unknown:
        raise ValueError(f"subset contains names not in parameter_names {names}: {unknown}")
    idx = [names.index(name) for name in subset]
    Z_K = Z[:, idx]
    if Z_K.size == 0:
        return float("inf")
    col_norms = np.linalg.norm(Z_K, axis=0)
    if np.any(col_norms == 0):
        return float("inf")
    Z_K = Z_K / col_norms
    lam = np.linalg.eigvalsh(Z_K.T @ Z_K)
    lam_min = float(lam[0])
    if lam_min <= 0:
        return float("inf")
    return float(1.0 / np.sqrt(lam_min))


def d_optimal_subset(
    experiment: Experiment,
    param_values: dict[str, float],
    k: int,
    design_values: dict[str, float] | None = None,
    *,
    method: Literal["auto", "enumerate", "greedy", "minlp"] = "auto",
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
) -> list[str]:
    """D-optimal subset of size ``k`` (Chu & Hahn 2007, 2012).

    Picks the size-``k`` subset S of parameters that maximizes
    ``log det(Z_S^T Z_S)``. Available methods:

    - ``"enumerate"``: exact, iterates all C(p, k) subsets. Uses
      ``numpy.linalg.slogdet``. Practical for p up to about 20.
    - ``"greedy"``: top-``k`` parameters from :func:`estimability_rank`.
      Cheap and typically close to optimal.
    - ``"auto"`` (default): enumerate for ``p <= 20``, greedy otherwise.
    - ``"minlp"``: reserved. Writing ``log det`` of a binary-masked
      matrix as an algebraic MINLP in discopt is non-trivial and
      outside the scope of the initial release. Raises
      :class:`NotImplementedError`; the Chu-Hahn paper uses
      combinatorial branch-and-bound with rank-one determinant
      updates, which is a better fit for a dedicated implementation.

    Parameters
    ----------
    k : int
        Subset size. Must satisfy ``0 < k <= n_parameters``.
    method : {"auto", "enumerate", "greedy", "minlp"}
        Solver.
    Other parameters
        See :func:`estimability_rank`.

    Returns
    -------
    list[str]
        Selected parameter names.

    Raises
    ------
    ValueError
        If ``k <= 0``, ``k > n_parameters``, or ``method`` is unknown.
    RuntimeError
        If ``method="enumerate"`` and no subset has positive determinant
        (Z is rank-deficient at rank < k).
    NotImplementedError
        If ``method="minlp"`` — see the method list above.
    """
    # Fail fast on invalid method/k before computing the Jacobian.
    if method not in ("auto", "enumerate", "greedy", "minlp"):
        raise ValueError(
            f"Unknown method {method!r}. Use 'auto', 'enumerate', 'greedy', or 'minlp'."
        )
    if method == "minlp":
        raise NotImplementedError(
            "d_optimal_subset(method='minlp') is reserved for a future release. "
            "The algebraic log-det of a binary-masked Gram matrix is not a "
            "clean MINLP nonlinearity; Chu & Hahn's combinatorial B&B is a "
            "better fit and will be added as a separate implementation. "
            "Use method='enumerate' (exact, p<=20) or method='greedy' (approx)."
        )
    if k <= 0:
        raise ValueError(f"k must be in (0, n_parameters], got {k}")

    Z, names = _scaled_sensitivity(
        experiment, param_values, design_values, parameter_scales, noise_covariance
    )
    p = len(names)
    if k > p:
        raise ValueError(f"k must be in (0, {p}], got {k}")

    chosen_method = method
    if chosen_method == "auto":
        chosen_method = "enumerate" if p <= 20 else "greedy"

    if chosen_method == "enumerate":
        return _dopt_enumerate(Z, names, k)
    # greedy
    res = estimability_rank(
        experiment,
        param_values,
        design_values,
        parameter_scales=parameter_scales,
        noise_covariance=noise_covariance,
        _cache=(Z, names),
    )
    return res.ranking[:k]


def _dopt_enumerate(Z: np.ndarray, names: list[str], k: int) -> list[str]:
    """Exact D-optimal subset by enumeration.

    Raises
    ------
    RuntimeError
        If no size-``k`` subset has a positive-determinant Gram matrix,
        i.e. the Brun-scaled sensitivity matrix Z has rank strictly
        less than ``k`` so no joint estimate is possible.
    """
    p = len(names)
    best_logdet = -np.inf
    best_subset: tuple[int, ...] | None = None
    for S in combinations(range(p), k):
        Z_S = Z[:, list(S)]
        sign, logabsdet = np.linalg.slogdet(Z_S.T @ Z_S)
        if sign > 0 and logabsdet > best_logdet:
            best_logdet = logabsdet
            best_subset = S
    if best_subset is None:
        raise RuntimeError(
            f"No size-{k} subset has a positive-determinant Gram matrix; "
            f"the scaled sensitivity matrix has rank < {k}. Try a smaller k "
            "or run diagnose_identifiability to locate the null directions."
        )
    return [names[i] for i in best_subset]
