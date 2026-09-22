"""Candidate-set (Fedorov) exchange and design-efficiency helpers."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from discopt.doe.exchange import (
    candidate_exchange_design,
    d_efficiency,
    i_efficiency,
    relative_efficiency,
)
from discopt.doe.templates import template_parameter_names

GRID5 = [{"x1": a, "x2": b} for a in np.linspace(-1, 1, 5) for b in np.linspace(-1, 1, 5)]
KW2 = dict(template="response-surface-2d", input_names=["x1", "x2"])
FACTORIAL3 = sorted((a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0))


@pytest.mark.parametrize("criterion", ["D", "A", "I", "G"])
def test_exchange_recovers_the_3x3_factorial(criterion: str) -> None:
    """9 runs for a full quadratic from a 5×5 grid: the 3² factorial is optimal."""
    r = candidate_exchange_design(GRID5, 9, criterion=criterion, seed=1, **KW2)
    assert sorted((d["x1"], d["x2"]) for d in r.designs) == FACTORIAL3
    assert r.n_experiments == 9 and len(r.indices) == 9


def test_exchange_with_categorical_candidates_matches_brute_force() -> None:
    cands = [{"cat": c, "x": x} for c in "ABC" for x in (-1.0, 0.0, 1.0)]

    def basis(r):
        return np.array([1.0, r["cat"] == "B", r["cat"] == "C", r["x"]], dtype=float)

    r = candidate_exchange_design(cands, 6, basis=basis, criterion="D", seed=0)
    F = np.array([basis(c) for c in cands])
    best = max(
        np.linalg.slogdet(F[list(c)].T @ F[list(c)])[1]
        for c in itertools.combinations_with_replacement(range(len(cands)), 6)
        if np.linalg.slogdet(F[list(c)].T @ F[list(c)])[0] > 0
    )
    assert r.criterion_value == pytest.approx(best)
    assert all(d["x"] in (-1.0, 1.0) for d in r.designs)


def test_exchange_without_repeats_and_with_prior() -> None:
    r = candidate_exchange_design(GRID5, 12, allow_repeats=False, seed=2, **KW2)
    assert len(set(r.indices)) == 12
    names = template_parameter_names("response-surface-2d", 2)
    from discopt.doe.linear_design import design_matrix, linear_fim

    prior = linear_fim(
        design_matrix("response-surface-2d", {}, names, ["x1", "x2"], GRID5[:6]), 1.0
    )
    aug = candidate_exchange_design(GRID5, 4, prior_fim=prior, seed=3, **KW2)
    assert np.isfinite(aug.criterion_value)
    np.testing.assert_allclose(aug.joint_fim, prior + aug.model_rows.T @ aug.model_rows)


def test_exchange_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        candidate_exchange_design(GRID5[:3], 5, allow_repeats=False, **KW2)
    with pytest.raises(ValueError, match="supports"):
        candidate_exchange_design(GRID5, 9, criterion="E", **KW2)


def test_d_efficiency_closed_form() -> None:
    """Adding a centre run to a 2² first-order design: per-run information goes
    from diag(1,1,1) to diag(1, 4/5, 4/5), so D-efficiency is (16/25)^(1/3)."""
    corners = [{"x1": a, "x2": b} for a in (-1, 1) for b in (-1, 1)]
    eff = d_efficiency(
        corners + [{"x1": 0, "x2": 0}], corners, template="linear", input_names=["x1", "x2"]
    )
    assert eff == pytest.approx((16 / 25) ** (1 / 3))
    assert d_efficiency(
        corners, corners, template="linear", input_names=["x1", "x2"]
    ) == pytest.approx(1.0)


def test_efficiency_of_ccd_against_d_optimal() -> None:
    """A rotatable CCD (alpha = sqrt 2) scaled into the unit box versus the 3²
    factorial for the 2-D quadratic: both efficiencies are below 1, and the
    per-run and absolute forms differ only by the run-count factor."""
    s = 1 / np.sqrt(2)
    ccd = [{"x1": a * s, "x2": b * s} for a in (-1, 1) for b in (-1, 1)] + [
        {"x1": 1.0, "x2": 0.0},
        {"x1": -1.0, "x2": 0.0},
        {"x1": 0.0, "x2": 1.0},
        {"x1": 0.0, "x2": -1.0},
        {"x1": 0.0, "x2": 0.0},
    ]
    fac = [{"x1": a, "x2": b} for a, b in FACTORIAL3]
    d = d_efficiency(ccd, fac, **KW2)
    i = i_efficiency(ccd, fac, region_bounds={"x1": (-1, 1), "x2": (-1, 1)}, **KW2)
    assert 0 < d < 1 and 0 < i
    absolute = relative_efficiency(ccd, fac, criterion="D", per_run=False, **KW2)
    assert absolute == pytest.approx(d)  # both have 9 runs


def test_efficiency_accepts_result_objects() -> None:
    a = candidate_exchange_design(GRID5, 9, seed=1, **KW2)
    b = candidate_exchange_design(GRID5, 9, criterion="I", seed=1, **KW2)
    assert d_efficiency(a, b) == pytest.approx(1.0)
