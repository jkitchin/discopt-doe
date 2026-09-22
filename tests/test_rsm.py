"""Classical RSM analysis: steepest ascent, canonical analysis, stationary-point
CI, ridge analysis and desirability, against constructed truths."""

from __future__ import annotations

import numpy as np
import pytest

from discopt.doe.linear_design import basis_parameter_names
from discopt.doe.rsm import (
    canonical_analysis,
    desirability,
    overall_desirability,
    quadratic_form,
    ridge_analysis,
    stationary_point_ci,
    steepest_ascent_path,
)


def _coeffs(b0, b, B):
    """Quadratic-basis coefficient dict from (b0, b, B)."""
    k = len(b)
    names = basis_parameter_names("quadratic", k)
    vals = [b0, *b, *np.diag(B)]
    vals += [2 * B[i][j] for i in range(k) for j in range(i + 1, k)]
    return dict(zip(names, vals))


# --- quadratic_form -------------------------------------------------------


def test_quadratic_form_round_trips() -> None:
    B = np.array([[-2.0, 0.5], [0.5, -1.0]])
    b0, b, B2 = quadratic_form(_coeffs(3.0, [1.0, -2.0], B))
    assert b0 == 3.0 and b == pytest.approx([1.0, -2.0])
    np.testing.assert_allclose(B2, B)


def test_quadratic_form_rejects_wrong_counts() -> None:
    with pytest.raises(ValueError, match="not a full quadratic"):
        quadratic_form([1.0, 2.0, 3.0, 4.0])


# --- steepest ascent ------------------------------------------------------


def test_steepest_ascent_path_follows_the_gradient() -> None:
    path = steepest_ascent_path({"b0": 50.0, "b1": 3.0, "b2": 4.0}, n_steps=4, step=0.5)
    assert path.direction == pytest.approx([0.6, 0.8])
    assert path.coded[0] == pytest.approx([0, 0])
    assert np.linalg.norm(path.coded[4]) == pytest.approx(2.0)


def test_steepest_ascent_base_factor_and_natural_units() -> None:
    path = steepest_ascent_path(
        {"b0": 0.0, "b1": 2.0, "b2": -1.0},
        n_steps=2,
        step=1.0,
        base_factor="T",
        input_names=["T", "t"],
        center={"T": 100.0, "t": 30.0},
        half_range={"T": 10.0, "t": 5.0},
    )
    # T moves one coded unit per step, t moves b2/|b1| = -0.5 of one.
    assert path.coded[1] == pytest.approx([1.0, -0.5])
    assert path.natural[2] == pytest.approx([120.0, 25.0])
    assert path.rows()[1] == pytest.approx({"T": 110.0, "t": 27.5})


def test_steepest_descent_reverses() -> None:
    up = steepest_ascent_path([1.0, 1.0], n_steps=1)
    down = steepest_ascent_path([1.0, 1.0], n_steps=1, direction="descent")
    assert down.coded[1] == pytest.approx(-up.coded[1])


# --- canonical analysis ---------------------------------------------------


def test_canonical_analysis_recovers_a_known_maximum() -> None:
    xs_true = np.array([0.4, -0.3])
    B = np.array([[-2.0, 0.6], [0.6, -1.5]])
    b = -2 * B @ xs_true  # so that -½ B⁻¹ b = xs_true
    ca = canonical_analysis(_coeffs(10.0, b, B))
    assert ca.stationary_point == pytest.approx(xs_true)
    assert ca.nature == "maximum"
    assert ca.response_at_stationary == pytest.approx(10.0 + xs_true @ b + xs_true @ B @ xs_true)
    assert ca.eigenvalues == pytest.approx(np.sort(np.linalg.eigvalsh(B)))
    assert "maximum" in ca.summary()


@pytest.mark.parametrize(
    "B, nature",
    [
        (np.diag([1.0, 2.0]), "minimum"),
        (np.diag([-1.0, 2.0]), "saddle"),
        (np.diag([-1.0, -0.001]), "ridge"),
    ],
)
def test_canonical_analysis_classifies(B, nature) -> None:
    assert canonical_analysis(b=[0.1, 0.2], B=B).nature == nature


def test_canonical_analysis_singular_B() -> None:
    ca = canonical_analysis(b=[1.0, 0.0], B=np.diag([-1.0, 0.0]))
    assert ca.nature == "ridge" and ca.stationary_point is None


# --- stationary-point CI --------------------------------------------------


def _ccd_rows():
    a = np.sqrt(2.0)
    pts = [(-1, -1), (1, -1), (-1, 1), (1, 1), (a, 0), (-a, 0), (0, a), (0, -a)] + [(0, 0)] * 5
    return np.array(pts, dtype=float)


def _X(pts):
    x1, x2 = pts[:, 0], pts[:, 1]
    return np.column_stack([np.ones(len(pts)), x1, x2, x1**2, x2**2, x1 * x2])


def test_stationary_point_ci_covers_by_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    xs_true = np.array([0.3, -0.2])
    B = np.array([[-3.0, 0.5], [0.5, -2.5]])
    b = -2 * B @ xs_true
    beta = np.array([50.0, *b, B[0, 0], B[1, 1], 2 * B[0, 1]])
    pts = _ccd_rows()
    X = _X(pts)
    names = basis_parameter_names("quadratic", 2)
    hits = np.zeros(2)
    n_rep = 300
    for _ in range(n_rep):
        y = X @ beta + rng.normal(0, 0.5, len(pts))
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(y) - X.shape[1]
        s2 = np.sum((y - X @ coef) ** 2) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        ci = stationary_point_ci(dict(zip(names, coef)), cov, dof=dof)
        hits += (ci.lower <= xs_true) & (xs_true <= ci.upper)
    coverage = hits / n_rep
    assert np.all((coverage > 0.89) & (coverage < 0.99)), coverage


def test_stationary_point_ci_accepts_a_fit_result() -> None:
    from discopt.doe.symbolic import SymbolicModel, fit_least_squares

    rng = np.random.default_rng(1)
    pts = _ccd_rows()
    beta = np.array([50.0, 1.0, -0.5, -3.0, -2.0, 0.4])
    y = _X(pts) @ beta + rng.normal(0, 0.3, len(pts))
    model = SymbolicModel(
        source="b0 + b1*x1 + b2*x2 + b11*x1**2 + b22*x2**2 + b12*x1*x2",
        parameter_names=("b0", "b1", "b2", "b11", "b22", "b12"),
        input_names=("x1", "x2"),
    )
    fit = fit_least_squares(model, [{"x1": p[0], "x2": p[1], "y": v} for p, v in zip(pts, y)])
    ci = stationary_point_ci(fit)
    assert np.all(ci.lower < ci.point) and np.all(ci.point < ci.upper)
    assert ci.point == pytest.approx(canonical_analysis(fit).stationary_point)


def test_stationary_point_ci_respects_parameter_order() -> None:
    names = basis_parameter_names("quadratic", 2)
    coef = dict(zip(names, [1.0, 0.5, -0.2, -1.0, -2.0, 0.3]))
    cov = np.diag(np.arange(1, 7) / 100.0)
    a = stationary_point_ci(coef, cov)
    perm = [5, 0, 3, 1, 4, 2]
    shuffled = [names[i] for i in perm]
    b = stationary_point_ci(
        {n: coef[n] for n in shuffled}, cov[np.ix_(perm, perm)], parameter_names=shuffled
    )
    assert a.std_errors == pytest.approx(b.std_errors)


# --- ridge analysis -------------------------------------------------------


def test_ridge_points_are_constrained_maxima() -> None:
    B = np.array([[-1.0, 0.8], [0.8, 0.5]])  # a saddle
    b = np.array([1.0, 0.5])
    res = ridge_analysis(b=b, B=B, radii=[0.0, 0.5, 1.0, 1.5])
    angles = np.linspace(0, 2 * np.pi, 20001)
    for R, p, yhat in zip(res.radii, res.points, res.response):
        assert np.linalg.norm(p) == pytest.approx(R, abs=1e-9)
        circle = R * np.column_stack([np.cos(angles), np.sin(angles)])
        brute = np.max(circle @ b + np.einsum("ij,jk,ik->i", circle, B, circle))
        assert yhat == pytest.approx(brute, abs=1e-6)


def test_ridge_minimize() -> None:
    B = np.diag([1.0, 2.0])
    b = np.array([1.0, -1.0])
    res = ridge_analysis(b=b, B=B, radii=[0.3], maximize=False)
    angles = np.linspace(0, 2 * np.pi, 20001)
    circle = 0.3 * np.column_stack([np.cos(angles), np.sin(angles)])
    brute = np.min(circle @ b + np.einsum("ij,jk,ik->i", circle, B, circle))
    assert res.response[0] == pytest.approx(brute, abs=1e-6)


# --- desirability ---------------------------------------------------------


def test_desirability_shapes() -> None:
    y = np.array([0.0, 5.0, 10.0, 15.0])
    assert desirability(y, "maximize", 5, 15) == pytest.approx([0, 0, 0.5, 1])
    assert desirability(y, "minimize", 5, 15) == pytest.approx([1, 1, 0.5, 0])
    assert desirability(10, "maximize", 5, 15, weight=2) == pytest.approx(0.25)
    t = desirability([5, 7.5, 10, 12.5, 15, 16], "target", 5, 15, target=10)
    assert t == pytest.approx([0, 0.5, 1, 0.5, 0, 0])
    t2 = desirability(12.5, "target", 5, 15, target=10, weight_upper=2)
    assert t2 == pytest.approx(0.25)


def test_desirability_validation() -> None:
    with pytest.raises(ValueError, match="high > low"):
        desirability(1, "maximize", 2, 1)
    with pytest.raises(ValueError, match="target"):
        desirability(1, "target", 0, 2)
    with pytest.raises(ValueError, match="kind"):
        desirability(1, "best", 0, 2)


def test_overall_desirability_is_a_weighted_geometric_mean() -> None:
    assert overall_desirability([0.25, 1.0]) == pytest.approx(0.5)
    assert overall_desirability([0.25, 1.0], importance=[1, 3]) == pytest.approx(0.25**0.25)
    assert overall_desirability([0.0, 1.0]) == pytest.approx(0.0)
    both = overall_desirability([np.array([0.25, 0.0]), np.array([1.0, 1.0])])
    assert both == pytest.approx([0.5, 0.0])
