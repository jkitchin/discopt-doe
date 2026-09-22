"""I- and G-optimal design search in discopt.doe.linear_design.

I (average prediction variance over the region) and G (its maximum) are not
functions of the FIM alone: they need the region, carried by a DesignRegion.
The checks below hold the searches against brute force and against the
Kiefer-Wolfowitz equivalence theorem.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from discopt.doe.linear_design import (
    CRITERIA,
    D_OPTIMAL,
    G_OPTIMAL,
    I_OPTIMAL,
    DesignRegion,
    _basis_evaluator,
    design_matrix,
    design_region,
    evaluate_criterion,
    is_maximized,
    linear_batch_design,
    linear_fim,
    linear_optimal_design,
    normalize_criterion,
)
from discopt.doe.prediction import i_criterion

P = ["b0", "b1", "b2"]
QUAD = dict(
    parameter_names=P,
    input_names=["x"],
    design_bounds={"x": (-1.0, 1.0)},
    template_args={"degree": 2},
)


def _basis():
    return _basis_evaluator("polynomial-1d", {"degree": 2}, P, ["x"])


def _fim(xs):
    F = np.array([_basis()(np.array([x])) for x in xs])
    return F.T @ F


def test_criterion_names_and_aliases() -> None:
    assert normalize_criterion("I") == I_OPTIMAL
    assert normalize_criterion("V") == I_OPTIMAL
    assert normalize_criterion("g") == G_OPTIMAL
    assert normalize_criterion("D") == D_OPTIMAL
    assert not is_maximized("I") and not is_maximized("G")
    # The FIM-only tuple is unchanged, so its users (and the jax pin) are too.
    assert I_OPTIMAL not in CRITERIA and G_OPTIMAL not in CRITERIA
    with pytest.raises(ValueError, match="unknown criterion"):
        normalize_criterion("g_optimal")
    with pytest.raises(ValueError, match="region"):
        evaluate_criterion(np.eye(3), "I")


def test_gauss_legendre_moment_matrix_is_exact() -> None:
    """W = ∫ f fᵀ dx / 2 over [-1, 1] for f = (1, x, x²): [[1,0,1/3],[0,1/3,0],[1/3,0,1/5]]."""
    reg = design_region(_basis(), ["x"], bounds={"x": (-1.0, 1.0)})
    assert reg.method == "gauss-legendre"
    expected = np.array([[1, 0, 1 / 3], [0, 1 / 3, 0], [1 / 3, 0, 1 / 5]])
    np.testing.assert_allclose(reg.moment_matrix, expected, atol=1e-12)


def test_i_criterion_matches_prediction_module() -> None:
    """The closed-form trace(M⁻¹W) agrees with averaging v(x) over sampled points."""
    reg = design_region(_basis(), ["x"], bounds={"x": (-1.0, 1.0)})
    xs = [-1.0, -0.3, 0.2, 0.9, 1.0]
    closed = reg.average_variance(_fim(xs))
    sampled = i_criterion(
        [{"x": x} for x in xs],
        points=[{"x": x} for x in np.linspace(-1, 1, 20001)],
        template="polynomial-1d",
        template_args={"degree": 2},
        parameter_names=P,
    )
    assert closed == pytest.approx(sampled, rel=1e-3)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_i_optimal_quadratic_matches_brute_force(n: int) -> None:
    result = linear_batch_design("polynomial-1d", n, criterion="I", **QUAD)
    reg = design_region(_basis(), ["x"], bounds={"x": (-1.0, 1.0)})
    grid = np.linspace(-1, 1, 21)
    best = np.inf
    for comb in itertools.combinations_with_replacement(grid, n):
        M = _fim(comb)
        if abs(np.linalg.det(M)) > 1e-10:
            best = min(best, reg.average_variance(M))
    # A continuous search can only match or beat the grid.
    assert result.criterion_value <= best + 1e-9
    assert result.criterion_value == pytest.approx(reg.average_variance(result.joint_fim))


def test_i_optimal_quadratic_approaches_the_known_weights() -> None:
    """The approximate I-optimal design for a quadratic on [-1, 1] puts weights
    1/4, 1/2, 1/4 on -1, 0, 1; with 8 runs that is exactly 2, 4, 2."""
    result = linear_batch_design("polynomial-1d", 8, criterion="I", **QUAD)
    xs = np.array(sorted(d["x"] for d in result.designs))
    np.testing.assert_allclose(xs, [-1, -1, 0, 0, 0, 0, 1, 1], atol=1e-3)


def test_g_optimal_matches_d_optimal_where_the_theorem_says() -> None:
    """For n = 6 the exact D-optimal quadratic design (2, 2, 2 on -1, 0, 1) is also
    G-optimal, with maximum relative variance p/N = 0.5 (equivalence theorem)."""
    g = linear_batch_design("polynomial-1d", 6, criterion="G", **QUAD)
    d = linear_batch_design("polynomial-1d", 6, criterion="D", **QUAD)
    assert g.criterion_value == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(
        sorted(x["x"] for x in g.designs), sorted(x["x"] for x in d.designs), atol=1e-3
    )


@pytest.mark.parametrize("n, sym", [(4, [-1.0, 1.0]), (5, [-1.0, 0.0, 1.0])])
def test_g_optimal_small_designs_match_a_fine_scan(n, sym) -> None:
    """With too few runs for the 2,2,2 pattern, G moves the inner runs off -1/0/1.
    The search must match a fine scan over the symmetric family."""
    g = linear_batch_design("polynomial-1d", n, criterion="G", **QUAD)
    reg = design_region(_basis(), ["x"], points=np.linspace(-1, 1, 4001)[:, None])
    scan = min(reg.max_variance(_fim(sorted(sym + [-t, t]))) for t in np.linspace(0.01, 0.99, 981))
    assert reg.max_variance(g.joint_fim) <= scan + 2e-3


def test_region_criteria_in_two_dimensions_and_single_point() -> None:
    kw = dict(
        parameter_names=["b0", "b1", "b2", "b11", "b22", "b12"],
        input_names=["x1", "x2"],
        design_bounds={"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)},
    )
    names = kw["parameter_names"]
    fac = [{"x1": a, "x2": b} for a in (-1, 0, 1) for b in (-1, 0, 1)]
    prior = linear_fim(design_matrix("response-surface-2d", {}, names, ["x1", "x2"], fac), 1.0)
    one = linear_optimal_design("response-surface-2d", criterion="I", prior_fim=prior, **kw)
    assert np.isfinite(one.criterion_value)
    i9 = linear_batch_design("response-surface-2d", 9, criterion="I", **kw)
    d9 = linear_batch_design("response-surface-2d", 9, criterion="D", **kw)
    reg = design_region(
        _basis_evaluator("response-surface-2d", {}, names, ["x1", "x2"]),
        ["x1", "x2"],
        bounds=kw["design_bounds"],
    )
    # The I search is at least as good as the D-optimal design on I.
    assert i9.criterion_value <= reg.average_variance(d9.joint_fim) + 1e-9


def test_explicit_region_points_and_constraints() -> None:
    """A region cut by x1 + x2 <= 1: the I design stays feasible and uses the region."""
    names = ["b0", "b1", "b2"]
    cons = [lambda d: 1.0 - d["x1"] - d["x2"]]
    res = linear_batch_design(
        "linear",
        6,
        parameter_names=names,
        input_names=["x1", "x2"],
        design_bounds={"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)},
        criterion="I",
        inequality_constraints=cons,
    )
    assert all(d["x1"] + d["x2"] <= 1.0 + 1e-6 for d in res.designs)
    pts = np.array([[a, b] for a in np.linspace(-1, 1, 21) for b in np.linspace(-1, 1, 21)])
    reg = design_region(
        _basis_evaluator("linear", {}, names, ["x1", "x2"]), ["x1", "x2"], points=pts
    )
    assert isinstance(reg, DesignRegion) and reg.method == "points"
    assert np.isfinite(evaluate_criterion(res.joint_fim, "I", region=reg))


def test_fim_criteria_unchanged_by_region_plumbing() -> None:
    """D designs are identical with and without the new region arguments."""
    a = linear_batch_design("polynomial-1d", 6, criterion="D", **QUAD)
    b = linear_batch_design(
        "polynomial-1d", 6, criterion="determinant", region_bounds={"x": (-1, 1)}, **QUAD
    )
    assert [d["x"] for d in a.designs] == [d["x"] for d in b.designs]
