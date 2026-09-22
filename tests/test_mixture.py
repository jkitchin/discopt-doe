"""Tests for constrained mixture regions and the bounded simplex projection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from discopt.doe.mixture import (
    check_mixture_bounds,
    cox_direction_trace,
    extreme_vertices,
    extreme_vertices_design,
    from_pseudo_components,
    to_pseudo_components,
)
from discopt.doe.simplex import project_to_simplex

BOUNDS = {"A": (0.2, 0.7), "B": (0.1, 0.6), "C": (0.1, 0.5)}

# McLean & Anderson (1966) railroad-flare example: magnesium, sodium nitrate,
# strontium nitrate, binder. Their design has 8 extreme vertices, 6 centroids
# of two-dimensional faces and the overall centroid (15 runs).
FLARE = {"mg": (0.40, 0.60), "nano3": (0.10, 0.50), "srno3": (0.10, 0.50), "binder": (0.03, 0.08)}


def test_consistent_bounds_pass_unchanged() -> None:
    r = check_mixture_bounds(BOUNDS)
    assert r.feasible and r.adjusted == ()
    assert r.bounds == BOUNDS


def test_unreachable_bounds_are_tightened() -> None:
    r = check_mixture_bounds({"A": (0.0, 1.0), "B": (0.0, 0.2), "C": (0.0, 0.2)})
    assert r.feasible and r.adjusted == ("A",)
    assert r.bounds["A"] == pytest.approx((0.6, 1.0))


def test_infeasible_bounds_are_reported_and_refused() -> None:
    bad = {"A": (0.5, 0.6), "B": (0.5, 0.6), "C": (0.1, 0.2)}
    assert not check_mixture_bounds(bad).feasible
    with pytest.raises(ValueError, match="infeasible"):
        extreme_vertices(bad)


def test_extreme_vertices_of_a_hexagon() -> None:
    ev = extreme_vertices(BOUNDS)
    assert len(ev) == 6
    for v in ev:
        assert sum(v.values()) == pytest.approx(1.0)
        for name, (lo, hi) in BOUNDS.items():
            assert lo - 1e-12 <= v[name] <= hi + 1e-12
    assert {"A": 0.2, "B": 0.3, "C": 0.5} in [{k: round(x, 9) for k, x in v.items()} for v in ev]


def test_mclean_anderson_flare_design() -> None:
    assert len(extreme_vertices(FLARE)) == 8
    design = extreme_vertices_design(FLARE, face_dimensions=(2,))
    counts = Counter(r["point_type"] for r in design)
    assert counts == {"vertex": 8, "face-2": 6, "centroid": 1}
    edges = extreme_vertices_design(FLARE, face_dimensions=(1,), include_centroid=False)
    assert Counter(r["point_type"] for r in edges)["face-1"] == 12


def test_hexagon_design_matches_the_textbook_recipe() -> None:
    design = extreme_vertices_design(BOUNDS)
    assert Counter(r["point_type"] for r in design) == {"vertex": 6, "face-1": 6, "centroid": 1}
    centroid = design[-1]
    ev = np.array([[v[n] for n in BOUNDS] for v in extreme_vertices(BOUNDS)])
    np.testing.assert_allclose([centroid[n] for n in BOUNDS], ev.mean(axis=0))


def test_pseudo_components_round_trip_and_map_to_the_simplex() -> None:
    lower = {"A": 0.2, "B": 0.1, "C": 0.1}
    x = {"A": 0.5, "B": 0.3, "C": 0.2}
    xp = to_pseudo_components(x, lower)
    assert sum(xp.values()) == pytest.approx(1.0)
    assert to_pseudo_components(lower | {"A": 0.8}, lower)["A"] == pytest.approx(1.0)
    back = from_pseudo_components(xp, lower)
    assert back == pytest.approx(x)


def test_cox_direction_keeps_the_sum_and_the_proportions() -> None:
    ref = {"A": 0.5, "B": 0.3, "C": 0.2}
    trace = cox_direction_trace(ref, "A", [-0.5, -0.1, 0.0, 0.25, 0.5])
    for row in trace:
        assert row["A"] + row["B"] + row["C"] == pytest.approx(1.0)
        if row["A"] < 1.0:
            assert row["B"] / row["C"] == pytest.approx(1.5)
    assert trace[-1]["A"] == pytest.approx(1.0)
    assert len(cox_direction_trace(ref, "B")) == 51
    with pytest.raises(ValueError, match="leaves the simplex"):
        cox_direction_trace(ref, "A", [0.6])


# ---- project_to_simplex with bounds ----------------------------------------


@pytest.mark.parametrize(
    "point",
    [
        {"A": 0.9, "B": 0.9, "C": 0.9},
        {"A": 1.0, "B": 0.0, "C": 0.0},
        {"A": -1.0, "B": 2.0, "C": 0.1},
        {"A": 0.3, "B": 0.3, "C": 0.4},
    ],
)
def test_bounded_projection_satisfies_bounds_and_sum(point) -> None:
    w = project_to_simplex(point, list(BOUNDS), 1.0, BOUNDS)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    for name, (lo, hi) in BOUNDS.items():
        assert lo - 1e-12 <= w[name] <= hi + 1e-12


def test_bounded_projection_is_the_nearest_feasible_point() -> None:
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    names = list(BOUNDS)
    for _ in range(20):
        v = rng.normal(0.33, 0.5, 3)
        w = project_to_simplex(dict(zip(names, v)), names, 1.0, BOUNDS)
        ref = minimize(
            lambda z: np.sum((z - v) ** 2),
            np.array([0.4, 0.3, 0.3]),
            bounds=[BOUNDS[n] for n in names],
            constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1.0}],
            method="SLSQP",
            options={"ftol": 1e-14},
        )
        np.testing.assert_allclose([w[n] for n in names], ref.x, atol=1e-6)


def test_inactive_bounds_give_the_unbounded_projection() -> None:
    p = {"A": 0.5, "B": 0.2, "C": 0.1}
    loose = {n: (0.0, 1.0) for n in "ABC"}
    assert project_to_simplex(p, list("ABC"), 1.0, loose) == pytest.approx(
        project_to_simplex(p, list("ABC"), 1.0)
    )


def test_bounded_projection_refuses_infeasible_bounds() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        project_to_simplex({"A": 0.5, "B": 0.5}, ["A", "B"], 1.0, {"A": (0, 0.3), "B": (0, 0.3)})


def test_projection_leaves_other_keys_alone() -> None:
    w = project_to_simplex(
        {"A": 0.9, "B": 0.9, "T": 350.0}, ["A", "B"], 1.0, {"A": (0, 1), "B": (0, 1)}
    )
    assert w["T"] == 350.0


def test_sample_simplex_respects_bounds_and_total() -> None:
    """Clip-and-rescale could push a component back outside its bounds; the
    pseudo-component draw with rejection never does, and stays uniform."""
    from discopt.doe import sample_simplex

    rng = np.random.default_rng(0)
    names = ["A", "B", "C"]
    bounds = {"A": (0.2, 0.7), "B": (0.1, 0.6), "C": (0.05, 0.5)}
    pts = np.array(
        [
            [p[n] for n in names]
            for p in (sample_simplex(names, 1.0, rng, bounds) for _ in range(2000))
        ]
    )
    assert np.allclose(pts.sum(axis=1), 1.0)
    for j, n in enumerate(names):
        assert pts[:, j].min() >= bounds[n][0] - 1e-12
        assert pts[:, j].max() <= bounds[n][1] + 1e-12
    # Unbounded draws are unchanged: same stream as a flat Dirichlet.
    a = sample_simplex(names, 2.0, np.random.default_rng(5))
    w = np.random.default_rng(5).dirichlet(np.ones(3))
    assert [a[n] for n in names] == pytest.approx(list(2.0 * w))


def test_check_mixture_bounds_message_states_the_new_bounds() -> None:
    from discopt.doe import check_mixture_bounds

    r = check_mixture_bounds({"A": (0.0, 1.0), "B": (0.0, 0.2), "C": (0.0, 0.2)})
    assert "A [0, 1] -> [0.6, 1]" in r.message
