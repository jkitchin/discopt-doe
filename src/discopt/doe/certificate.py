"""Optimal approximate targeted designs and certified efficiency bounds.

How good is a design? For a targeted criterion this module answers with a
*certificate*: a lower bound on the efficiency of any design, relative to the
best design that can be built from a candidate set, which holds whether or not
the optimizer behind the design found the global optimum.

Mathematics
-----------
An *approximate* design puts weights ``w_i ≥ 0``, ``Σ w_i = 1`` on candidate
conditions ``x_i``; with ``n`` runs and prior information ``P`` its
information is ``M(w) = P + n Σ w_i I(x_i)``. Both targeted criteria used here
are convex in ``M`` (written to be minimized):

* ``"trace"``:       ``φ(M) = tr(L M⁻¹)`` with ``L = K W Kᵀ``  (L-optimality)
* ``"determinant"``: ``φ(M) = log det(Kᵀ M⁻¹ K)``              (D_s / D_A)

Let the *sensitivity* of candidate ``x`` be ``d(x) = -<∇φ(M), n I(x)>``:
``n tr(I(x) M⁻¹ L M⁻¹)`` for the trace and ``n tr(I(x) M⁻¹ K Σ_ψ⁻¹ Kᵀ M⁻¹)``
for the determinant. Convexity gives, for every other weight vector ``w*``,

    φ(M(w*)) ≥ φ(M(w)) - (max_x d(x) - Σ_i w_i d(x_i)),

so ``gap = max d - Σ w d`` is zero exactly at the optimum (the general
equivalence theorem; Kiefer & Wolfowitz 1960, Pukelsheim 2006) and
``φ(w) - gap`` is a certified lower bound on the optimal value. Every exact
``n``-run design on the candidates is an approximate design, so its
efficiency is bounded below by

* ``trace``:       ``eff ≥ (φ(w) - gap) / φ(exact)``
* ``determinant``: ``eff ≥ exp(-(φ(exact) - φ(w) + gap) / s)``  (``s`` targets)

The weights are found by alternating the multiplicative algorithm
``w_i ← w_i (d_i / Σ w d)^ρ`` (Silvey, Titterington & Torsney 1978; Yu 2010;
``ρ = 1`` for the determinant, ``1/2`` for the trace) with Frank-Wolfe
vertex-direction steps and away steps under an exact line search (Wynn 1970;
Todd & Yildirim 2007), in the spirit of Yu's cocktail algorithm (2011): the
multiplicative step moves all weights at once, the away step removes poor
support points that it only shrinks geometrically. Exact designs come
from the weights by efficient rounding (Pukelsheim & Rieder 1992).

Minimax criteria
----------------
``"max_variance"`` (``max_j w_j Var ψ_j``) and ``"min_eigenvalue"``
(``λ_max(S̃)`` with ``S̃ = W^½ Σ_ψ W^½``; maximizing the smallest eigenvalue of
the target information is minimizing this) are convex but not differentiable.
They are maxima of trace criteria,

    max_j S̃_jj = max_{π ∈ simplex} Σ_j π_j S̃_jj,     λ_max(S̃) = max_{Z ⪰ 0, tr Z = 1} tr(Z S̃),

so for every fixed dual weight ``Z`` the smooth criterion ``ψ_Z = tr(Z S̃)``
(an L-criterion with ``L = K W^½ Z W^½ Kᵀ``) satisfies ``ψ_Z ≤ φ``, and its
own certified bound ``ψ_Z(w) - gap_Z(w)`` bounds the minimax optimum from
below. The dual is improved by exponentiated-gradient ascent (the
supergradient of ``min_w ψ_Z`` is ``S̃`` at the inner optimum); by strong
duality the bound closes on the optimum, and the reported gap is the duality
gap between the best design found and the best bound. The final ``Z`` tells
which targets (or target directions) bind.

The bound is relative to designs on the candidate set. Use a fine grid (the
default) or add the points you care about; the designs being certified are
always added to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

_CRITERIA = {
    "trace": "trace",
    "average_variance": "trace",
    "determinant": "determinant",
    "max_variance": "max_variance",
    "min_eigenvalue": "min_eigenvalue",
}
# Criteria certified through their dual (a max over trace criteria).
_MINIMAX = ("max_variance", "min_eigenvalue")
# Relative ridge that keeps M invertible in directions no candidate informs
# (nuisance combinations the target does not need); far below any real scale.
_RIDGE = 1e-10


def _criterion_name(criterion: str) -> str:
    try:
        return _CRITERIA[criterion]
    except KeyError:
        raise ValueError(
            f"certificates are available for the 'trace', 'determinant', 'max_variance' and "
            f"'min_eigenvalue' target criteria, got {criterion!r}"
        ) from None


def default_candidates(
    design_bounds: Mapping[str, tuple[float, float]], n: int | None = None, seed: int = 0
) -> list[dict[str, float]]:
    """A candidate grid over the design box.

    A regular grid in one or two inputs (201 or 41 x 41 points), a scrambled
    Sobol sample plus the box corners in more. Inputs spanning two or more
    decades with a positive lower bound get log-spaced points.
    """
    names = list(design_bounds)
    k = len(names)

    def axis(name: str, m: int) -> np.ndarray:
        lo, hi = map(float, design_bounds[name])
        if lo > 0 and hi / lo >= 100:
            return np.geomspace(lo, hi, m)
        return np.linspace(lo, hi, m)

    if k == 1:
        m = n or 201
        return [{names[0]: float(v)} for v in axis(names[0], m)]
    if k == 2:
        m = int(round(np.sqrt(n))) if n else 41
        a, b = axis(names[0], m), axis(names[1], m)
        return [{names[0]: float(x), names[1]: float(y)} for x in a for y in b]
    from scipy.stats import qmc

    m = n or 1024
    u = qmc.Sobol(k, scramble=True, seed=seed).random(m)
    corners = np.array(np.meshgrid(*[[0.0, 1.0]] * k)).reshape(k, -1).T
    u = np.vstack([u, corners])
    out = []
    for row in u:
        point = {}
        for name, ui in zip(names, row):
            lo, hi = map(float, design_bounds[name])
            if lo > 0 and hi / lo >= 100:
                point[name] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
            else:
                point[name] = float(lo + ui * (hi - lo))
        out.append(point)
    return out


def _regularized(M: np.ndarray) -> np.ndarray:
    d = np.abs(np.diag(M))
    scale = np.where(d > 0, d, max(float(d.max()) if d.size else 1.0, 1.0))
    return M + np.diag(_RIDGE * scale)


def _weighted_cov(M: np.ndarray, K: np.ndarray, Wd: np.ndarray) -> np.ndarray:
    """``Σ̃ = W^½ Kᵀ M⁻¹ K W^½`` on the regularized ``M``."""
    Kw = K * np.sqrt(Wd)
    S = Kw.T @ np.linalg.solve(_regularized(M), Kw)
    return 0.5 * (S + S.T)


def _minimax_value(S: np.ndarray, crit: str) -> float:
    """``max_j Σ̃_jj`` (max_variance) or ``λ_max(Σ̃)`` (min_eigenvalue), minimized."""
    if crit == "max_variance":
        return float(np.max(np.diag(S)))
    return float(np.linalg.eigvalsh(S)[-1])


def _phi_and_grad(
    M: np.ndarray, K: np.ndarray, Wd: np.ndarray, crit: str, Z: np.ndarray | None = None
):
    """``φ(M)`` (minimized) and ``G = -∇φ`` (so ``d(x) = n tr(I(x) G)``).

    ``Z`` (``s × s``, PSD, trace 1) turns ``"trace"`` into the dual criterion
    ``tr(Z Σ̃)`` of the minimax ones. A minimax criterion itself returns its
    value and no gradient.
    """
    if crit in _MINIMAX:
        return _minimax_value(_weighted_cov(M, K, Wd), crit), None
    Mi = np.linalg.inv(_regularized(M))
    Mi = 0.5 * (Mi + Mi.T)
    MiK = Mi @ K
    if crit == "trace":
        if Z is None:
            L = (K * Wd) @ K.T
        else:
            Kw = K * np.sqrt(Wd)
            L = Kw @ Z @ Kw.T
        phi = float(np.trace(L @ Mi))
        G = Mi @ L @ Mi
    else:
        S = K.T @ MiK
        sign, logdet = np.linalg.slogdet(S)
        phi = float(logdet) if sign > 0 else np.inf
        G = MiK @ np.linalg.solve(S, MiK.T)
    return phi, 0.5 * (G + G.T)


def _frank_wolfe_step(w: np.ndarray, d: np.ndarray, phi_of) -> np.ndarray:
    """One vertex-direction / away step on the simplex, with a line search."""
    from scipy.optimize import minimize_scalar

    mean_d = float(w @ d)
    j = int(np.argmax(d))
    on = np.flatnonzero(w > 0)
    k = int(on[np.argmin(d[on])])
    toward = d[j] - mean_d
    away = mean_d - d[k]
    if toward >= away:
        e = np.zeros_like(w)
        e[j] = 1.0
        direction, top = e - w, 1.0
    else:
        e = np.zeros_like(w)
        e[k] = 1.0
        direction = w - e
        top = w[k] / (1.0 - w[k]) if w[k] < 1.0 else 0.0
    if top <= 0:
        return w
    res = minimize_scalar(
        lambda a: phi_of(np.maximum(w + a * direction, 0.0)),
        bounds=(0.0, top),
        method="bounded",
        options={"xatol": 1e-12 * max(top, 1e-12)},
    )
    a = float(res.x)
    # The bounded search never returns the end point itself; a full away step
    # (dropping the point) is often the right one.
    if phi_of(np.maximum(w + top * direction, 0.0)) <= res.fun:
        a = top
    out = np.maximum(w + a * direction, 0.0)
    return out / out.sum()


@dataclass
class ApproximateTargetDesign:
    """Optimal approximate design for a target, with its equivalence-theorem check.

    Attributes
    ----------
    candidates : list[dict]
        The candidate conditions.
    weights : numpy.ndarray
        Optimal weight on each candidate (sums to 1).
    n_runs : float
        Number of runs the weights are spread over (scales the information
        relative to ``prior_fim``).
    criterion : str
        ``"trace"``, ``"determinant"``, ``"max_variance"`` or ``"min_eigenvalue"``.
    value : float
        ``φ`` at the weights: the weighted target variance sum (trace),
        ``log det Σ_ψ`` (determinant), the largest weighted target variance
        (max_variance) or ``λ_max(W^½ Σ_ψ W^½)`` (min_eigenvalue, whose
        efficiency is the ratio of these).
    lower_bound : float
        Certified lower bound on the optimal ``φ`` over designs on the
        candidates.
    sensitivity : numpy.ndarray
        ``d(x)`` at each candidate. At the optimum it touches its maximum
        exactly on the support -- plot it against the candidates to see where
        the information about the target is.
    gap : float
        ``value - lower_bound``: for the smooth criteria the equivalence-theorem
        gap ``max d - Σ w d``, for the minimax ones the duality gap.
    dual : numpy.ndarray or None
        Minimax criteria: the dual weight ``Z`` (``s × s``) on the targets --
        which targets (or target directions) bind at the optimum.
    iterations : int
    target : ResolvedTarget
    candidate_fims : numpy.ndarray
        ``I(x_i)``, one per candidate (``(m, p, p)``).
    prior_fim : numpy.ndarray
    """

    candidates: list[dict[str, float]]
    weights: np.ndarray
    n_runs: float
    criterion: str
    value: float
    lower_bound: float
    sensitivity: np.ndarray
    gap: float
    iterations: int
    target: Any
    candidate_fims: np.ndarray = field(repr=False)
    prior_fim: np.ndarray = field(repr=False)
    dual: np.ndarray | None = None

    def support(self, tol: float = 1e-3) -> list[tuple[dict[str, float], float]]:
        """Candidates with weight above ``tol``, as ``(condition, weight)`` pairs."""
        idx = np.flatnonzero(self.weights > tol)
        return [(dict(self.candidates[i]), float(self.weights[i])) for i in idx]

    def phi(self, fim: np.ndarray) -> float:
        """The criterion (minimized form) at an information matrix."""
        Wd = np.asarray(self.target.weights, dtype=float)
        return _phi_and_grad(np.asarray(fim, dtype=float), self.target.K, Wd, self.criterion)[0]

    def efficiency_bound(self, fim: np.ndarray) -> float:
        """Certified lower bound on the efficiency of a design with information ``fim``.

        ``fim`` must include the same prior and come from ``n_runs`` runs.
        Relative to the best ``n_runs``-run design on the candidates.
        """
        phi = self.phi(fim)
        if not np.isfinite(phi):
            return 0.0
        if self.criterion != "determinant":
            return float(max(self.lower_bound, 0.0) / phi) if phi > 0 else 0.0
        s = self.target.n_targets
        return float(min(1.0, np.exp(-(phi - self.lower_bound) / s)))

    def rounded(self, n: int | None = None) -> list[dict[str, float]]:
        """An exact ``n``-run design from the weights (efficient rounding)."""
        n = int(round(self.n_runs if n is None else n))
        idx = np.flatnonzero(self.weights > 1e-4)
        w = self.weights[idx] / self.weights[idx].sum()
        ell = len(idx)
        if n < ell:  # keep the heaviest points
            order = np.argsort(-w)[:n]
            return [dict(self.candidates[idx[i]]) for i in sorted(order)]
        counts = np.ceil((n - ell / 2.0) * w).astype(int)
        while counts.sum() < n:
            j = int(np.argmin(counts / w))
            counts[j] += 1
        while counts.sum() > n:
            j = int(np.argmax((counts - 1) / w))
            counts[j] -= 1
        out: list[dict[str, float]] = []
        for i, c in zip(idx, counts):
            out += [dict(self.candidates[i])] * int(c)
        return out

    def summary(self) -> str:
        lines = [
            f"Approximate {self.criterion}-optimal design for {self.target.names} "
            f"({self.n_runs:g} runs, {len(self.candidates)} candidates)",
            f"  criterion {self.value:.6g}  (certified optimum >= {self.lower_bound:.6g}, "
            f"gap {self.gap:.2e}, {self.iterations} iterations)",
            "  support:",
        ]
        for point, wt in self.support():
            lines.append(
                "    " + ", ".join(f"{k}={v:.4g}" for k, v in point.items()) + f"   w={wt:.3f}"
            )
        return "\n".join(lines)


def approximate_targeted_design(
    experiment: Any,
    param_values: Mapping[str, Any],
    target: Any,
    *,
    candidates: Sequence[Mapping[str, float]] | None = None,
    design_bounds: Mapping[str, tuple[float, float]] | None = None,
    n_candidates: int | None = None,
    criterion: str = "trace",
    n_runs: float = 1.0,
    prior_fim: np.ndarray | None = None,
    max_iter: int = 20000,
    tol: float = 1e-7,
) -> ApproximateTargetDesign:
    """The optimal approximate design for a target, with its optimality gap.

    Parameters
    ----------
    experiment, param_values
        The model and the nominal parameters.
    target : Target or ResolvedTarget
    candidates : sequence of mapping, optional
        Candidate conditions; default :func:`default_candidates` over
        ``design_bounds``.
    criterion : {"trace", "determinant", "max_variance", "min_eigenvalue"}, default "trace"
        The minimax criteria (worst weighted target variance, and the worst
        weighted target direction) are certified through their dual; see the
        module docstring.
    n_runs : float, default 1
        Runs the weights are spread over. Matters only with a ``prior_fim``
        (it sets how much new information competes with the old); certify an
        ``N``-run design against ``n_runs=N``.
    prior_fim : numpy.ndarray, optional
        Information already collected.
    max_iter, tol
        Multiplicative-algorithm budget and the relative gap to stop at.
    """
    from discopt.doe.fim import compute_fim_batch
    from discopt.doe.targets import ResolvedTarget, Target, _check_target_criterion

    crit = _criterion_name(criterion)
    if candidates is None:
        if not design_bounds:
            raise ValueError("give candidates or design_bounds")
        candidates = default_candidates(design_bounds, n_candidates)
    cands = [dict(c) for c in candidates]
    if isinstance(target, Target):
        resolved = target.resolve(experiment, param_values)
    elif isinstance(target, ResolvedTarget):
        resolved = target
    else:
        raise TypeError("target must be a Target or ResolvedTarget")
    _check_target_criterion(resolved, crit)

    results = compute_fim_batch(experiment, dict(param_values), cands)
    infos = np.stack([np.asarray(r.fim, dtype=float) for r in results])
    ok = np.all(np.isfinite(infos.reshape(len(cands), -1)), axis=1)
    if not np.any(ok):
        raise RuntimeError("the FIM could not be evaluated at any candidate")
    cands = [c for c, keep in zip(cands, ok) if keep]
    infos = infos[ok]
    p = infos.shape[1]
    P = np.zeros((p, p)) if prior_fim is None else np.asarray(prior_fim, dtype=float)
    K, Wd = resolved.K, np.asarray(resolved.weights, dtype=float)
    n = float(n_runs)
    rho = 1.0 if crit == "determinant" else 0.5

    m = len(cands)
    w0 = np.full(m, 1.0 / m)

    def info(v: np.ndarray) -> np.ndarray:
        return P + n * np.einsum("i,ijk->jk", v, infos)

    def smooth_solver(kind: str, Z: np.ndarray | None):
        def evaluate(v: np.ndarray):
            phi, G = _phi_and_grad(info(v), K, Wd, kind, Z)
            return phi, n * np.einsum("ijk,jk->i", infos, G)

        def phi_of(v: np.ndarray) -> float:
            return _phi_and_grad(info(v), K, Wd, kind, Z)[0]

        return evaluate, phi_of

    if crit not in _MINIMAX:
        evaluate, phi_of = smooth_solver(crit, None)
        phi, d = evaluate(w0)
        if not np.isfinite(phi):
            raise RuntimeError(
                f"the targets {resolved.names} are not estimable from any mixture of the "
                "candidates (and the prior); widen the design region or add a prior_fim"
            )
        w, phi, d, gap, it = _optimize_weights(w0, evaluate, phi_of, rho, max_iter, tol)
        return ApproximateTargetDesign(
            candidates=cands,
            weights=w,
            n_runs=n,
            criterion=crit,
            value=float(phi),
            lower_bound=float(phi - gap),
            sensitivity=d,
            gap=gap,
            iterations=it,
            target=resolved,
            candidate_fims=infos,
            prior_fim=P,
        )

    # Minimax criteria through their dual. With S̃ = W^½ Σ_ψ W^½,
    #   max_j S̃_jj = max_{π ∈ simplex} Σ_j π_j S̃_jj,   λ_max(S̃) = max_{Z ⪰ 0, tr Z = 1} tr(Z S̃).
    # For every fixed Z, ψ_Z(M) = tr(Z S̃(M)) is a smooth trace criterion with
    # ψ_Z ≤ φ, so its own certified bound (value - equivalence gap) bounds the
    # minimax optimum from below. Exponentiated-gradient ascent on Z (the
    # supergradient of min_w ψ_Z is S̃ at the inner optimum) tightens it to the
    # optimum (strong duality); the best primal design seen, including the
    # running average of the inner optima, gives the upper end.
    s_t = resolved.n_targets
    diagonal = crit == "max_variance"
    H = np.zeros((s_t, s_t))  # log of Z (diagonal for max_variance)

    def z_of(H: np.ndarray) -> np.ndarray:
        if diagonal:
            h = np.diag(H)
            e = np.exp(h - h.max())
            return np.diag(e / e.sum())
        lam, V = np.linalg.eigh(0.5 * (H + H.T))
        e = np.exp(lam - lam.max())
        Z = (V * (e / e.sum())) @ V.T
        return 0.5 * (Z + Z.T)

    def primal(v: np.ndarray) -> float:
        return _minimax_value(_weighted_cov(info(v), K, Wd), crit)

    if not np.isfinite(primal(w0)):
        raise RuntimeError(
            f"the targets {resolved.names} are not estimable from any mixture of the "
            "candidates (and the prior); widen the design region or add a prior_fim"
        )
    w = w0
    w_avg = np.zeros(m)
    best_w, best_val = w0, primal(w0)
    best_lb, best_Z, best_d = -np.inf, z_of(H), None
    inner_iter = max(200, int(max_iter) // 20)
    it_total = 0
    outer = 0
    for outer in range(1, 301):
        Z = z_of(H)
        evaluate, phi_of = smooth_solver("trace", Z)
        # The bound psi - g_in is valid at any inner iterate, so the inner
        # solves need not be tight: the outer loop does the converging.
        w, psi, d, g_in, it = _optimize_weights(
            w, evaluate, phi_of, 0.5, inner_iter, max(tol, 1e-5)
        )
        it_total += it
        lb = psi - g_in
        if lb > best_lb:
            best_lb, best_Z = lb, Z
        w_avg = w_avg + (w - w_avg) / outer
        for cand in (w, w_avg):
            val = primal(cand)
            if val < best_val:
                best_val, best_w = val, cand.copy()
        if best_val - best_lb <= 1e-4 * abs(best_val):
            break
        S = _weighted_cov(info(w), K, Wd)
        scale = float(np.max(np.abs(np.linalg.eigvalsh(S)))) or 1.0
        H = H + (2.0 / np.sqrt(outer)) * (np.diag(np.diag(S)) if diagonal else S) / scale
    # The sensitivity of the best dual at the best design: where the binding
    # targets' information is.
    evaluate, _ = smooth_solver("trace", best_Z)
    _, best_d = evaluate(best_w)
    lb = float(min(best_lb, best_val))
    return ApproximateTargetDesign(
        candidates=cands,
        weights=np.where(best_w < 1e-12, 0.0, best_w)
        / np.sum(np.where(best_w < 1e-12, 0.0, best_w)),
        n_runs=n,
        criterion=crit,
        value=float(best_val),
        lower_bound=lb,
        sensitivity=best_d,
        gap=float(best_val - lb),
        iterations=it_total,
        target=resolved,
        candidate_fims=infos,
        prior_fim=P,
        dual=best_Z,
    )


def _optimize_weights(w, evaluate, phi_of, rho: float, max_iter: int, tol: float):
    """Minimize a smooth convex design criterion over the weights.

    Alternates multiplicative steps with Frank-Wolfe vertex/away steps (see the
    module docstring). Returns ``(w, φ, d, gap, iterations)``.
    """
    phi, d = evaluate(w)
    it = 0
    for it in range(1, int(max_iter) + 1):
        mean_d = float(w @ d)
        gap = float(d.max() - mean_d)
        if gap <= tol * max(abs(phi), 1e-300) or mean_d <= 0:
            break
        if it % 2:
            # Multiplicative step: fast early, slow to drop poor support points.
            w = w * (np.maximum(d, 0.0) / mean_d) ** rho
            w = w / w.sum()
        else:
            # Frank-Wolfe step with away steps and an exact line search: toward
            # the most sensitive candidate, or mass away from the least
            # sensitive support point -- whichever promises more.
            w = _frank_wolfe_step(w, d, phi_of)
        phi, d = evaluate(w)
    gap = float(max(d.max() - w @ d, 0.0))
    w = np.where(w < 1e-12, 0.0, w)
    return w / w.sum(), float(phi), d, gap, it


@dataclass
class DesignCertificate:
    """How close an exact design is to the best possible, for a target.

    Attributes
    ----------
    efficiency_lower_bound : float
        Certified: the design is at least this efficient relative to the best
        design with the same number of runs on the candidate set (1 = optimal).
    value : float
        The design's criterion (minimized form; see :mod:`discopt.doe.certificate`).
    optimum_lower_bound : float
        Certified lower bound on the best achievable criterion.
    approximate : ApproximateTargetDesign
        The approximate optimum the bound came from (its support shows where
        the ideal runs are).
    """

    efficiency_lower_bound: float
    value: float
    optimum_lower_bound: float
    approximate: ApproximateTargetDesign

    def summary(self) -> str:
        return (
            f"design criterion {self.value:.6g}; best possible >= {self.optimum_lower_bound:.6g}; "
            f"certified efficiency >= {self.efficiency_lower_bound:.1%}"
        )


def certify_design(
    experiment: Any,
    param_values: Mapping[str, Any],
    designs: Sequence[Mapping[str, float]],
    target: Any,
    *,
    criterion: str = "trace",
    design_bounds: Mapping[str, tuple[float, float]] | None = None,
    candidates: Sequence[Mapping[str, float]] | None = None,
    prior_fim: np.ndarray | None = None,
    **kwargs: Any,
) -> DesignCertificate:
    """Certified efficiency of an exact design (a list of runs) for a target.

    Solves the approximate design problem for ``len(designs)`` runs on the
    candidates (default: a grid over ``design_bounds``, plus the runs
    themselves) and bounds the design's efficiency against it. The bound holds
    for the best exact design too, since every exact design is an approximate
    one. ``kwargs`` go to :func:`approximate_targeted_design`.
    """
    from discopt.doe.fim import compute_fim

    runs = [dict(d) for d in designs]
    if not runs:
        raise ValueError("no runs to certify")
    if candidates is None:
        if not design_bounds:
            raise ValueError("give candidates or design_bounds")
        candidates = default_candidates(design_bounds, kwargs.pop("n_candidates", None))
    cands = list(candidates) + runs
    approx = approximate_targeted_design(
        experiment,
        param_values,
        target,
        candidates=cands,
        criterion=criterion,
        n_runs=len(runs),
        prior_fim=prior_fim,
        **kwargs,
    )
    fim = approx.prior_fim.copy()
    for d in runs:
        fim = fim + np.asarray(compute_fim(experiment, dict(param_values), d).fim, dtype=float)
    return DesignCertificate(
        efficiency_lower_bound=approx.efficiency_bound(fim),
        value=approx.phi(fim),
        optimum_lower_bound=approx.lower_bound,
        approximate=approx,
    )


__all__ = [
    "ApproximateTargetDesign",
    "DesignCertificate",
    "approximate_targeted_design",
    "certify_design",
    "default_candidates",
]
