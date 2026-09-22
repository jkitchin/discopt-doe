"""Prediction variance, SPV, FDS and the I/G criteria, against closed forms."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from discopt.doe.prediction import (
    fds_curve,
    g_criterion,
    i_criterion,
    prediction_variance,
    region_points,
    scaled_prediction_variance,
)

FACTORIAL_2x2 = [{"x1": a, "x2": b} for a, b in product((-1.0, 1.0), repeat=2)]


def test_first_order_factorial_spv_is_one_at_the_centre() -> None:
    """XᵀX = 4I for a 2² factorial and a first-order model, so v(0) = 1/4 and
    SPV(0) = N·v = 1; at a corner v = 3/4 and SPV = 3."""
    pts = [{"x1": 0.0, "x2": 0.0}, {"x1": 1.0, "x2": 1.0}]
    v = prediction_variance(FACTORIAL_2x2, pts, template="linear")
    assert v == pytest.approx([0.25, 0.75])
    assert scaled_prediction_variance(FACTORIAL_2x2, pts, template="linear") == pytest.approx(
        [1.0, 3.0]
    )
    # sigma scales the absolute variance by sigma^2
    assert prediction_variance(FACTORIAL_2x2, pts, template="linear", sigma=2.0) == pytest.approx(
        [1.0, 3.0]
    )


def test_line_design_variance_matches_the_textbook_formula() -> None:
    """Var ŷ(x)/σ² = 1/n + (x - x̄)² / Sxx for a straight line."""
    design = [{"x": 0.0}] * 3 + [{"x": 10.0}] * 3
    xs = np.linspace(-2, 12, 15)
    v = prediction_variance(
        design,
        xs.reshape(-1, 1),
        template="polynomial-1d",
        template_args={"degree": 1},
        input_names=["x"],
    )
    expected = 1 / 6 + (xs - 5.0) ** 2 / (6 * 25.0)
    assert v == pytest.approx(expected)


def test_rotatable_ccd_has_constant_spv_on_circles() -> None:
    alpha = np.sqrt(2.0)  # (2^k)^(1/4) for k = 2
    design = (
        FACTORIAL_2x2
        + [
            {"x1": alpha, "x2": 0.0},
            {"x1": -alpha, "x2": 0.0},
            {"x1": 0.0, "x2": alpha},
            {"x1": 0.0, "x2": -alpha},
        ]
        + [{"x1": 0.0, "x2": 0.0}] * 5
    )
    for r in (0.3, 0.8, 1.2):
        angles = np.linspace(0, 2 * np.pi, 13)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        spv = scaled_prediction_variance(
            design, pts, template="response-surface-2d", input_names=["x1", "x2"]
        )
        assert np.ptp(spv) < 1e-9 * spv.mean()


def test_basis_and_symbolic_model_agree_with_template() -> None:
    from discopt.doe.symbolic import SymbolicModel

    design = FACTORIAL_2x2 + [{"x1": 0.0, "x2": 0.0}]
    pts = [{"x1": 0.3, "x2": -0.7}]
    v_template = prediction_variance(design, pts, template="linear")
    v_basis = prediction_variance(
        design, pts, basis=lambda x: np.array([1.0, x[0], x[1]]), input_names=["x1", "x2"]
    )
    model = SymbolicModel(
        source="b0 + b1*x1 + b2*x2", parameter_names=("b0", "b1", "b2"), input_names=("x1", "x2")
    )
    v_model = prediction_variance(
        design, pts, model=model, theta={"b0": 0, "b1": 0, "b2": 0}, input_names=["x1", "x2"]
    )
    assert v_basis == pytest.approx(v_template)
    assert v_model == pytest.approx(v_template)


def test_rows_with_bookkeeping_columns_are_accepted() -> None:
    design = [dict(r, run_order=i, is_center=False) for i, r in enumerate(FACTORIAL_2x2)]
    assert prediction_variance(design, [{"x1": 0.0, "x2": 0.0}], template="linear") == (
        pytest.approx([0.25])
    )


def test_singular_design_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot estimate"):
        prediction_variance(FACTORIAL_2x2, FACTORIAL_2x2, template="response-surface-2d")


def test_fds_curve_is_sorted_and_summarised() -> None:
    res = fds_curve(
        FACTORIAL_2x2, bounds={"x1": (-1, 1), "x2": (-1, 1)}, template="linear", n_samples=500
    )
    assert np.all(np.diff(res.spv) >= 0)
    assert res.fraction[0] > 0 and res.fraction[-1] == 1.0
    assert 1.0 <= res.quantiles["min"] <= res.quantiles["0.5"] <= res.quantiles["max"] <= 3.0
    assert res.n_runs == 4
    assert "median SPV" in res.summary()


def test_i_and_g_criteria_on_a_known_design() -> None:
    """For the 2² factorial and a first-order model, v(x) = (1 + x1² + x2²)/4.
    Its mean over the square is (1 + 2/3)/4 and its maximum (corners) is 3/4."""
    bounds = {"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)}
    mid = (np.arange(200) + 0.5) / 100.0 - 1.0  # midpoint grid: unbiased mean
    grid = [{"x1": a, "x2": b} for a in mid for b in mid]
    i_grid = i_criterion(FACTORIAL_2x2, points=grid, template="linear")
    assert i_grid == pytest.approx((1 + 2 / 3) / 4, rel=1e-3)
    i_mc = i_criterion(FACTORIAL_2x2, bounds=bounds, template="linear", n_samples=20000, seed=1)
    assert i_mc == pytest.approx((1 + 2 / 3) / 4, rel=0.02)
    assert g_criterion(FACTORIAL_2x2, bounds=bounds, template="linear") == pytest.approx(0.75)
    assert g_criterion(FACTORIAL_2x2, bounds=bounds, template="linear", scaled=True) == (
        pytest.approx(3.0)
    )


def test_mixture_region_and_scheffe_template() -> None:
    from discopt.doe.templates import simplex_lattice_points

    lattice = simplex_lattice_points(["A", "B", "C"], 2)
    pts = region_points(["A", "B", "C"], 200, mixture_total=1.0, seed=0)
    assert all(abs(sum(p.values()) - 1.0) < 1e-12 and min(p.values()) >= 0 for p in pts)
    res = fds_curve(lattice, mixture_total=1.0, template="scheffe-quadratic", n_samples=300)
    # A saturated design: SPV is exactly N = 6 at every lattice point, less inside.
    at_design = scaled_prediction_variance(lattice, lattice, template="scheffe-quadratic")
    assert at_design == pytest.approx(np.full(6, 6.0))
    assert res.quantiles["max"] <= 6.0 + 1e-9


def test_region_requires_exactly_one_shape() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        region_points(["x"], 5)
    with pytest.raises(ValueError, match="exactly one"):
        region_points(["x"], 5, bounds={"x": (0, 1)}, mixture_total=1.0)
