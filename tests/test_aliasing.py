"""Tests for discopt.doe.aliasing: words, resolution, and effect correlations."""

from __future__ import annotations

import numpy as np
import pytest

from discopt.doe.aliasing import alias_structure
from discopt.doe.fractional import fractional_factorial_design
from discopt.doe.screening import factorial_2level_design
from discopt.doe.screening_designs import definitive_screening_design, plackett_burman_design

SEVEN = {f: (-1, 1) for f in "ABCDEFG"}


def test_saturated_2_7_4_defining_relation():
    """2^(7-4) with D=AB, E=AC, F=BC, G=ABC: 16 elements in the defining
    contrast subgroup (15 words), seven of length three, resolution III
    (Box, Hunter & Hunter 2005, ch. 6)."""
    d = fractional_factorial_design(SEVEN, generators=["D=AB", "E=AC", "F=BC", "G=ABC"], seed=0)
    a = alias_structure(d)
    assert a.resolution == 3
    assert a.regular
    assert len(a.words) == 15
    assert sum(len(w) - 1 == 3 for w in a.words) == 7
    assert {"+ABD", "+ACE", "+BCF"} <= set(a.words)
    # Each main effect carries exactly three 2FIs at full strength.
    for f in "ABCDEFG":
        hits = a.aliases(f)
        assert len(hits) == 3
        assert all(abs(abs(r) - 1) < 1e-12 for _, r in hits)
    assert ("A", "B:D", "C:E", "F:G") in a.alias_groups()


def test_resolution_v_half_fraction_alias_groups():
    """2^(5-1) with I = ABCDE: resolution V; each 2FI is aliased with a 3FI."""
    d = fractional_factorial_design(
        {f: (-1, 1) for f in "ABCDE"}, generators=["E=ABCD"], resolution=5, seed=0
    )
    a = alias_structure(d, max_order=3)
    assert a.resolution == 5
    assert a.words == ("+ABCDE",)
    assert a.aliases("A") == []  # clear of all 2FIs and 3FIs
    groups = set(a.alias_groups())
    assert ("A:B", "C:D:E") in groups
    assert ("D:E", "A:B:C") in groups


def test_negative_generator_sign():
    d = fractional_factorial_design({f: (-1, 1) for f in "ABCD"}, generators=["D=-ABC"], seed=0)
    a = alias_structure(d)
    assert a.words == ("-ABCD",)
    assert a.resolution == 4


def test_full_factorial_has_no_words():
    d = factorial_2level_design({f: (0.0, 1.0) for f in "ABC"}, seed=0)
    a = alias_structure(d)
    assert a.words == ()
    assert a.resolution is None
    assert a.regular
    assert a.alias_groups() == []


def test_center_points_ignored_for_words_and_quadratics_aliased():
    d = factorial_2level_design({f: (0.0, 1.0) for f in "AB"}, center_points=3, seed=0)
    a = alias_structure(d)
    assert a.words == ()
    # A centre point cannot tell one pure quadratic from another.
    assert ("A^2", "B^2") in a.alias_groups()


def test_plackett_burman_is_nonregular_with_third_correlations():
    """PB-12: mains orthogonal, every 2FI correlated +-1/3 with the mains it
    does not involve (Hamada & Wu 1992)."""
    d = plackett_burman_design({f"x{i}": (-1, 1) for i in range(1, 12)}, seed=0)
    a = alias_structure(d)
    assert not a.regular
    # Saturated, the only complete word is the product of all 11 columns
    # (every row has five minus signs); any 7 columns have none.
    assert len(a.words) == 1 and a.words[0].count(":") == 10
    assert (
        alias_structure(plackett_burman_design({f: (-1, 1) for f in "ABCDEFG"}, n_runs=12)).words
        == ()
    )
    k = 11
    R = a.correlation
    assert np.allclose(R[:k, :k], np.eye(k))
    main_vs_2fi = np.abs(R[:k, k:])
    assert set(np.round(np.unique(main_vs_2fi), 12)) <= {0.0, round(1 / 3, 12)}


def test_definitive_screening_mains_clear_of_2fi_and_quadratics():
    d = definitive_screening_design({f: (0.0, 10.0) for f in "ABCDEF"}, seed=0)
    a = alias_structure(d)
    k = 6
    assert np.allclose(a.correlation[:k, :k], np.eye(k))
    assert np.max(np.abs(a.correlation[:k, k:])) < 1e-12
    assert any(lab.endswith("^2") for lab in a.labels)
    # No two 2FIs are completely aliased.
    twofi = [i for i, lab in enumerate(a.labels) if lab.count(":") == 1]
    sub = np.abs(a.correlation[np.ix_(twofi, twofi)]) - np.eye(len(twofi))
    assert sub.max() < 1 - 1e-9


def test_accepts_rows_and_matrix_inputs():
    d = fractional_factorial_design(SEVEN, generators=["D=AB", "E=AC", "F=BC", "G=ABC"], seed=0)
    from_design = alias_structure(d)
    rows = [{f: r[f] for f in "ABCDEFG"} for r in d.rows]
    from_rows = alias_structure(rows)
    X = np.array([[r[f] for f in "ABCDEFG"] for r in d.rows], dtype=float)
    from_matrix = alias_structure(X)
    assert from_rows.words == from_design.words
    assert from_matrix.words == from_design.words
    assert from_matrix.factors == tuple("ABCDEFG")


def test_row_dicts_code_categorical_and_numeric():
    rows = [
        {"cat": "A", "T": 80.0, "y": 1.0},
        {"cat": "B", "T": 80.0, "y": 2.0},
        {"cat": "A", "T": 120.0, "y": 3.0},
        {"cat": "B", "T": 120.0, "y": 4.0},
    ]
    a = alias_structure(rows, ["cat", "T"])
    assert a.words == ()
    assert a.correlation[0, 1] == 0.0


def test_summary_text():
    d = fractional_factorial_design(SEVEN, generators=["D=AB", "E=AC", "F=BC", "G=ABC"], seed=0)
    text = alias_structure(d).summary()
    assert "Resolution: III" in text
    assert "I = +ABD" in text
    pb = alias_structure(plackett_burman_design({f: (-1, 1) for f in "ABCDE"}, n_runs=12))
    assert "partial aliasing" in pb.summary()


def test_bad_inputs():
    with pytest.raises(ValueError, match="entries in"):
        alias_structure(np.array([[2.0, 1.0], [1.0, -1.0]]))
    with pytest.raises(ValueError, match="max_order"):
        alias_structure(np.eye(2), max_order=0)
    a = alias_structure(np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]))
    with pytest.raises(ValueError, match="unknown effect"):
        a.aliases("Z")
