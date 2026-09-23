"""Ratio bounds on a mixture region.

Formulations are specified in ratios more often than in component bounds -- an
emulsifier that has to be two to three times the oil -- and a ratio is not a
box constraint, so the region stops being a slice of a box and the corners move.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.doe import ratio_constraints
from discopt.doe.mixture import extreme_vertices, extreme_vertices_design

BOUNDS = {"water": (0.2, 0.7), "oil": (0.1, 0.5), "surf": (0.05, 0.4)}
TOL = 1e-9


def _ratio(point, num="surf", den="oil"):
    return point[num] / point[den]


def test_a_ratio_bound_removes_the_corners_that_break_it() -> None:
    free = extreme_vertices(BOUNDS)
    held = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (0.5, 1.5)})

    assert any(_ratio(p) > 1.5 + TOL for p in free)  # the free region does break it
    assert all(0.5 - TOL <= _ratio(p) <= 1.5 + TOL for p in held)
    assert all(abs(sum(p.values()) - 1.0) < TOL for p in held)


def test_the_new_corners_sit_on_the_ratio_boundary() -> None:
    """A corner is where constraints meet, so a ratio bound makes its own."""
    held = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (0.5, 1.5)})
    on_boundary = [p for p in held if min(abs(_ratio(p) - 0.5), abs(_ratio(p) - 1.5)) < 1e-9]
    assert len(on_boundary) >= 2, held


def test_a_one_sided_ratio_works() -> None:
    lower_only = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (0.5, None)})
    upper_only = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (None, 1.5)})
    assert all(_ratio(p) >= 0.5 - TOL for p in lower_only)
    assert all(_ratio(p) <= 1.5 + TOL for p in upper_only)
    # Each is a weaker statement than the pair, so neither region is smaller.
    both = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (0.5, 1.5)})
    assert len(lower_only) >= len(both) or len(upper_only) >= len(both)


def test_a_ratio_that_changes_nothing_leaves_the_region_alone() -> None:
    """A bound wider than the region is not a constraint, and must not move a corner."""
    free = extreme_vertices(BOUNDS)
    wide = extreme_vertices(BOUNDS, ratios={("surf", "oil"): (0.0, 100.0)})
    assert len(free) == len(wide)
    for a, b in zip(free, wide):
        assert a == pytest.approx(b)


def test_the_design_keeps_its_faces_and_respects_the_ratio() -> None:
    design = extreme_vertices_design(BOUNDS, ratios={("surf", "oil"): (0.5, 1.5)})
    kinds = {p["point_type"] for p in design}
    assert "vertex" in kinds and "centroid" in kinds and "face-1" in kinds
    for point in design:
        blend = {k: v for k, v in point.items() if k != "point_type"}
        assert abs(sum(blend.values()) - 1.0) < TOL
        assert 0.5 - TOL <= blend["surf"] / blend["oil"] <= 1.5 + TOL


def test_an_impossible_ratio_says_which_constraint_closed_the_region() -> None:
    with pytest.raises(ValueError, match="ratios are what closes the region"):
        # oil is at least 0.1 and surf at most 0.4, so surf/oil cannot reach 20.
        extreme_vertices(BOUNDS, ratios={("surf", "oil"): (20.0, None)})


def test_ratio_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        extreme_vertices(BOUNDS, ratios={("surf", "nope"): (1.0, 2.0)})
    with pytest.raises(ValueError, match="ratio to itself"):
        extreme_vertices(BOUNDS, ratios={("oil", "oil"): (1.0, 2.0)})
    with pytest.raises(ValueError, match="exceeds upper bound"):
        extreme_vertices(BOUNDS, ratios={("surf", "oil"): (2.0, 1.0)})
    with pytest.raises(ValueError, match="non-negative"):
        extreme_vertices(BOUNDS, ratios={("surf", "oil"): (-1.0, 1.0)})


def test_four_components_with_two_ratios() -> None:
    bounds = {
        "water": (0.3, 0.6),
        "oil": (0.1, 0.4),
        "surf": (0.05, 0.3),
        "salt": (0.0, 0.1),
    }
    ratios = {("surf", "oil"): (0.5, 1.0), ("salt", "water"): (None, 0.2)}
    verts = extreme_vertices(bounds, ratios=ratios)
    assert verts
    for p in verts:
        assert abs(sum(p.values()) - 1.0) < TOL
        assert 0.5 - TOL <= p["surf"] / p["oil"] <= 1.0 + TOL
        assert p["salt"] <= 0.2 * p["water"] + TOL
        for name, (lo, hi) in bounds.items():
            assert lo - TOL <= p[name] <= hi + TOL


# ---------------------------------------------------------------------------
# The same ratio for a search rather than a classical design
# ---------------------------------------------------------------------------


def test_ratio_constraints_agree_with_the_region() -> None:
    """The callables and the geometry must mean the same thing."""
    cons = ratio_constraints("surf", "oil", 0.5, 1.5)
    inside = extreme_vertices_design(BOUNDS, ratios={("surf", "oil"): (0.5, 1.5)})
    for point in inside:
        blend = {k: v for k, v in point.items() if k != "point_type"}
        assert all(c(blend) >= -1e-9 for c in cons)

    outside = [p for p in extreme_vertices(BOUNDS) if not 0.5 <= _ratio(p) <= 1.5]
    assert outside
    for blend in outside:
        assert any(c(blend) < 0 for c in cons)


def test_a_ratio_constraint_is_well_behaved_at_a_zero_denominator() -> None:
    """The ratio is undefined there; the linear form still says something sane."""
    cons = ratio_constraints("surf", "oil", 0.5, 1.5)
    at_zero = {"surf": 0.2, "oil": 0.0}
    values = [c(at_zero) for c in cons]
    assert np.isfinite(values).all()
    assert values[1] < 0  # surf above 1.5 * 0 is still a violation, not a NaN


def test_ratio_constraints_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="give low, high, or both"):
        ratio_constraints("a", "b")
    with pytest.raises(ValueError, match="back to front"):
        ratio_constraints("a", "b", 2.0, 1.0)
    with pytest.raises(ValueError, match="ratio to itself"):
        ratio_constraints("a", "a", 1.0)
    assert len(ratio_constraints("a", "b", low=1.0)) == 1
    assert len(ratio_constraints("a", "b", high=2.0)) == 1
    assert len(ratio_constraints("a", "b", 1.0, 2.0)) == 2
