"""Tests for maximin Latin hypercubes, quasi-random designs and space-filling metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from discopt.doe.classical import latin_hypercube_design
from discopt.doe.spacefilling import quasi_random_design, space_filling_metrics

FACTORS = {f"x{i}": (0.0, 1.0) for i in range(4)}


def _is_latin(u: np.ndarray) -> bool:
    n = u.shape[0]
    return all(
        sorted(np.floor(u[:, j] * n).astype(int)) == list(range(n)) for j in range(u.shape[1])
    )


def test_maximin_lhs_is_latin_and_spreads_runs_further_apart() -> None:
    plain, maximin = [], []
    for seed in range(10):
        d0 = latin_hypercube_design(FACTORS, 30, optimize=False, seed=seed)
        d1 = latin_hypercube_design(FACTORS, 30, optimize="maximin", seed=seed)
        assert _is_latin(d1.to_unit_matrix())
        plain.append(d0.metrics()["min_distance"])
        maximin.append(d1.metrics()["min_distance"])
    assert np.mean(maximin) > 1.5 * np.mean(plain)


def test_maximin_is_reproducible() -> None:
    a = latin_hypercube_design(FACTORS, 20, optimize="maximin", seed=4).to_matrix()
    b = latin_hypercube_design(FACTORS, 20, optimize="maximin", seed=4).to_matrix()
    np.testing.assert_array_equal(a, b)


def test_optimize_true_still_means_discrepancy() -> None:
    a = latin_hypercube_design(FACTORS, 16, optimize=True, seed=1).to_matrix()
    b = latin_hypercube_design(FACTORS, 16, optimize="discrepancy", seed=1).to_matrix()
    np.testing.assert_array_equal(a, b)


def test_bad_optimize_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="optimize"):
        latin_hypercube_design(FACTORS, 8, optimize="best")


def test_unit_matrix_scales_by_bounds() -> None:
    d = latin_hypercube_design({"T": (300.0, 400.0), "P": (1.0, 5.0)}, 10, seed=0)
    u = d.to_unit_matrix()
    np.testing.assert_allclose(u[:, 0], (d.to_matrix()[:, 0] - 300.0) / 100.0)
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_metrics_match_hand_computation() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]])
    m = space_filling_metrics(x)
    assert m["min_distance"] == pytest.approx(0.5)
    assert m["mean_min_distance"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    d = np.array([1.0, 0.5, np.hypot(1.0, 0.5)])
    assert m["phi_p"] == pytest.approx(np.sum(d**-15.0) ** (1 / 15.0))
    assert m["max_projection_gap"] == pytest.approx([1.0, 0.5])
    from scipy.stats import qmc

    assert m["centered_discrepancy"] == pytest.approx(qmc.discrepancy(x, method="CD"))


def test_metrics_scale_raw_matrices_and_name_design_factors() -> None:
    raw = np.array([[300.0, 1.0], [400.0, 5.0]])
    m = space_filling_metrics(raw, bounds=[(300.0, 400.0), (1.0, 5.0)])
    assert m["min_distance"] == pytest.approx(np.sqrt(2.0))
    d = latin_hypercube_design({"T": (300.0, 400.0), "P": (1.0, 5.0)}, 8, seed=0)
    assert set(d.metrics()["max_projection_gap"]) == {"T", "P"}


def test_discrepancy_is_nan_outside_the_unit_cube() -> None:
    m = space_filling_metrics(np.array([[-0.5, 0.2], [0.4, 0.9]]))
    assert np.isnan(m["centered_discrepancy"])


def test_sobol_power_of_two_is_balanced() -> None:
    d = quasi_random_design(FACTORS, 64, method="sobol", seed=0)
    assert d.kind == "sobol" and len(d) == 64
    u = d.to_unit_matrix()
    # A (t, m, s)-net: each factor's projection has one point per 1/64 interval.
    for j in range(u.shape[1]):
        assert sorted(np.floor(u[:, j] * 64).astype(int)) == list(range(64))


def test_sobol_non_power_of_two_warns_and_halton_does_not() -> None:
    with pytest.warns(UserWarning, match="powers of two"):
        quasi_random_design(FACTORS, 50, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = quasi_random_design(FACTORS, 50, method="halton", seed=0)
    assert d.kind == "halton" and len(d) == 50


def test_quasi_random_respects_bounds_and_method_names() -> None:
    d = quasi_random_design({"T": (300.0, 400.0)}, 16, seed=2)
    x = d.to_matrix()
    assert x.min() >= 300.0 and x.max() <= 400.0
    with pytest.raises(ValueError, match="method"):
        quasi_random_design(FACTORS, 8, method="grid")


def test_metrics_warn_on_natural_units_without_bounds() -> None:
    from discopt.doe import space_filling_metrics

    with pytest.warns(UserWarning, match="bounds"):
        space_filling_metrics([[300.0, 1.0], [400.0, 5.0], [350.0, 3.0]])
    m = space_filling_metrics(
        [[300.0, 1.0], [400.0, 5.0], [350.0, 3.0]], bounds=[(300.0, 400.0), (1.0, 5.0)]
    )
    assert m["min_distance"] == pytest.approx(np.sqrt(0.5))
