"""Tests for the closed-form classical designs."""

from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pytest

from discopt.doe.classical import (
    ClassicalDesign,
    box_behnken_design,
    central_composite_design,
    latin_hypercube_design,
)

BOX_3 = {"x": (0.0, 10.0), "y": (0.0, 10.0), "z": (0.0, 10.0)}


def coded(design: ClassicalDesign) -> np.ndarray:
    """Design points mapped back to coded units, where the box is [-1, 1]."""
    lbs = np.array([b[0] for b in design.bounds])
    ubs = np.array([b[1] for b in design.bounds])
    return (design.to_matrix() - (lbs + ubs) / 2.0) / ((ubs - lbs) / 2.0)


def center_mask(design: ClassicalDesign) -> np.ndarray:
    return np.array([bool(r["is_center"]) for r in design.rows])


# ─────────────────────────── shared behaviour ───────────────────────────


def _all_designs():
    return [
        latin_hypercube_design(BOX_3, 12, seed=0),
        central_composite_design(BOX_3, seed=0),
        box_behnken_design(BOX_3, seed=0),
    ]


@pytest.mark.parametrize("design", _all_designs())
def test_run_order_is_a_permutation(design: ClassicalDesign) -> None:
    orders = [r["run_order"] for r in design.rows]
    assert sorted(orders) == list(range(len(design)))


@pytest.mark.parametrize("design", _all_designs())
def test_design_rows_drop_bookkeeping_keys(design: ClassicalDesign) -> None:
    """design_rows() yields only operating conditions, ready for the workbook."""
    for row in design.design_rows():
        assert set(row) == set(design.factors)
        assert all(isinstance(v, float) for v in row.values())


@pytest.mark.parametrize("design", _all_designs())
def test_to_matrix_shape(design: ClassicalDesign) -> None:
    assert design.to_matrix().shape == (len(design), len(design.factors))


@pytest.mark.parametrize(
    "generator",
    [latin_hypercube_design, central_composite_design, box_behnken_design],
)
def test_rejects_empty_factors(generator) -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        generator({}, 4) if generator is latin_hypercube_design else generator({})


@pytest.mark.parametrize(
    "bounds, match",
    [
        ((5.0, 5.0), "upper bound must exceed"),
        ((10.0, 0.0), "upper bound must exceed"),
        ((0.0, float("inf")), "must be finite"),
    ],
)
def test_rejects_degenerate_bounds(bounds, match) -> None:
    factors = {"x": bounds, "y": (0.0, 1.0), "z": (0.0, 1.0)}
    with pytest.raises(ValueError, match=match):
        box_behnken_design(factors)


# ────────────────────────── Latin hypercube ──────────────────────────


def test_lhs_run_count_and_bounds() -> None:
    d = latin_hypercube_design({"T": (300.0, 400.0), "P": (1.0, 5.0)}, 8, seed=0)
    assert len(d) == 8
    m = d.to_matrix()
    assert ((m[:, 0] >= 300.0) & (m[:, 0] <= 400.0)).all()
    assert ((m[:, 1] >= 1.0) & (m[:, 1] <= 5.0)).all()
    assert not center_mask(d).any()


@pytest.mark.parametrize("optimize", [True, False])
def test_lhs_puts_exactly_one_point_per_stratum(optimize: bool) -> None:
    """The defining property: n strata per factor, one point in each."""
    n = 16
    d = latin_hypercube_design({"a": (0.0, 1.0), "b": (-5.0, 5.0)}, n, optimize=optimize, seed=3)
    m = d.to_matrix()
    for j, (lb, ub) in enumerate(d.bounds):
        idx = np.floor((m[:, j] - lb) / (ub - lb) * n).astype(int)
        idx = np.clip(idx, 0, n - 1)  # a point exactly on ub lands in the last stratum
        assert sorted(idx) == list(range(n)), f"factor {j} strata not uniquely occupied"


def test_lhs_is_reproducible_and_seed_sensitive() -> None:
    a = latin_hypercube_design(BOX_3, 10, seed=7).to_matrix()
    b = latin_hypercube_design(BOX_3, 10, seed=7).to_matrix()
    c = latin_hypercube_design(BOX_3, 10, seed=8).to_matrix()
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_lhs_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be >= 2"):
        latin_hypercube_design(BOX_3, 1)


def test_lhs_falls_back_without_scipy_qmc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without scipy.stats.qmc the pure-numpy stratified path still holds."""
    import scipy.stats

    monkeypatch.delattr(scipy.stats, "qmc", raising=False)
    monkeypatch.setitem(sys.modules, "scipy.stats.qmc", None)

    n = 12
    d = latin_hypercube_design({"a": (0.0, 1.0), "b": (0.0, 1.0)}, n, seed=5)
    assert len(d) == n
    m = d.to_matrix()
    assert ((m >= 0.0) & (m <= 1.0)).all()
    for j in range(2):
        idx = np.clip(np.floor(m[:, j] * n).astype(int), 0, n - 1)
        assert sorted(idx) == list(range(n))


# ──────────────────────── central composite ────────────────────────


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ccd_run_count_and_blocks(k: int) -> None:
    factors = {f"x{i}": (0.0, 10.0) for i in range(k)}
    d = central_composite_design(factors, center_points=4, seed=0)
    assert len(d) == 2**k + 2 * k + 4

    blocks = [r["block"] for r in d.rows]
    assert blocks.count("factorial") == 2**k
    assert blocks.count("axial") == 2 * k
    assert blocks.count("center") == 4
    assert center_mask(d).sum() == 4


def test_ccd_default_keeps_every_run_within_bounds() -> None:
    """The axial points define the range, so nothing escapes the user's box."""
    d = central_composite_design({"a": (0.0, 10.0), "b": (0.0, 10.0)}, seed=1)
    m = d.to_matrix()
    assert m.min() == 0.0 and m.max() == 10.0


def test_ccd_textbook_scaling_pushes_axial_points_outside() -> None:
    """within_bounds=False means lb/ub are the ±1 corners, not the extremes."""
    d = central_composite_design({"a": (0.0, 10.0), "b": (0.0, 10.0)}, within_bounds=False, seed=1)
    m = d.to_matrix()
    alpha = 2 ** (2 / 4)
    assert m.min() == pytest.approx(5.0 - 5.0 * alpha)
    assert m.max() == pytest.approx(5.0 + 5.0 * alpha)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ccd_rotatable_alpha_is_the_fourth_root_of_the_factorial_count(k: int) -> None:
    """Rotatability fixes the axial distance at ``(2**k)**0.25``.

    Note this is *not* the same as putting the axial points on the sphere
    through the factorial corners (radius ``sqrt(k)``); the two coincide only
    at k=2. Rotatability is a statement about prediction variance being a
    function of distance from the centre, not about a common radius.
    """
    factors = {f"x{i}": (0.0, 10.0) for i in range(k)}
    d = central_composite_design(factors, alpha="rotatable", within_bounds=False, seed=0)
    c = coded(d)
    axial = c[[r["block"] == "axial" for r in d.rows]]
    assert np.abs(axial).max(axis=1) == pytest.approx((2**k) ** 0.25)


def test_ccd_face_centred_puts_axials_on_the_faces() -> None:
    d = central_composite_design(BOX_3, alpha="face", seed=0)
    c = coded(d)
    axial = c[[r["block"] == "axial" for r in d.rows]]
    # One coordinate at ±1, the rest at the centre.
    assert np.allclose(np.abs(axial).max(axis=1), 1.0)
    assert np.allclose(np.abs(axial).sum(axis=1), 1.0)


def test_ccd_face_scaling_is_convention_independent() -> None:
    """At alpha=1 the two `within_bounds` conventions coincide."""
    a = central_composite_design(BOX_3, alpha="face", within_bounds=True, seed=0)
    b = central_composite_design(BOX_3, alpha="face", within_bounds=False, seed=0)
    np.testing.assert_allclose(a.to_matrix(), b.to_matrix())


def test_ccd_accepts_explicit_alpha() -> None:
    d = central_composite_design(BOX_3, alpha=1.5, within_bounds=False, seed=0)
    c = coded(d)
    axial = c[[r["block"] == "axial" for r in d.rows]]
    assert np.allclose(np.abs(axial).max(axis=1), 1.5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": "spherical"}, "alpha must be"),
        ({"alpha": -1.0}, "positive finite"),
        ({"center_points": 0}, "center_points must be >= 1"),
    ],
)
def test_ccd_rejects_bad_options(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        central_composite_design(BOX_3, **kwargs)


@pytest.mark.parametrize("k", [1, 7])
def test_ccd_rejects_unsupported_factor_counts(k: int) -> None:
    factors = {f"x{i}": (0.0, 1.0) for i in range(k)}
    with pytest.raises(ValueError, match="2 to 6 factors"):
        central_composite_design(factors)


# ───────────────────────────── Box-Behnken ─────────────────────────────


@pytest.mark.parametrize("k, center_points, expected", [(3, 3, 15), (4, 3, 27), (5, 6, 46)])
def test_bbd_matches_published_run_counts(k: int, center_points: int, expected: int) -> None:
    factors = {f"x{i}": (0.0, 10.0) for i in range(k)}
    d = box_behnken_design(factors, center_points=center_points, seed=0)
    assert len(d) == expected
    assert center_mask(d).sum() == center_points


def test_bbd_visits_no_corner() -> None:
    """The defining property: no run sets every factor to an extreme."""
    d = box_behnken_design(BOX_3, seed=0)
    c = coded(d)
    assert not np.any(np.all(np.abs(c) == 1.0, axis=1))


def test_bbd_points_are_edge_midpoints() -> None:
    """Each non-centre run has exactly two factors at ±1 and the rest centred."""
    d = box_behnken_design(BOX_3, seed=0)
    c = coded(d)[~center_mask(d)]
    assert np.all(np.count_nonzero(np.abs(c) == 1.0, axis=1) == 2)
    assert np.all(np.count_nonzero(c == 0.0, axis=1) == 1)


def test_bbd_points_lie_on_a_common_sphere() -> None:
    """Every non-centre run is the same coded distance from the centre."""
    d = box_behnken_design(BOX_3, seed=0)
    radii = np.linalg.norm(coded(d)[~center_mask(d)], axis=1)
    assert np.allclose(radii, np.sqrt(2.0))


def test_bbd_stays_within_bounds() -> None:
    d = box_behnken_design({"a": (300.0, 400.0), "b": (1.0, 5.0), "c": (-2.0, 2.0)}, seed=0)
    m = d.to_matrix()
    for j, (lb, ub) in enumerate(d.bounds):
        assert m[:, j].min() >= lb and m[:, j].max() <= ub


@pytest.mark.parametrize("k", [2, 6])
def test_bbd_rejects_unsupported_factor_counts(k: int) -> None:
    """Six factors would silently build a non-standard all-pairs design."""
    factors = {f"x{i}": (0.0, 1.0) for i in range(k)}
    with pytest.raises(ValueError, match="3 to 5 factors"):
        box_behnken_design(factors)


def test_bbd_rejects_zero_center_points() -> None:
    with pytest.raises(ValueError, match="center_points must be >= 1"):
        box_behnken_design(BOX_3, center_points=0)


# ─────────────────────── CLI / workbook integration ───────────────────────


class TestClassicalWorkbooks:
    """The design → fill → fit loop the browser app drives, via the pure API."""

    @staticmethod
    def _new(tmp_path, template, inputs, **kw):
        from discopt.doe.cli import NewParams, do_new

        return do_new(
            NewParams(
                output=tmp_path / f"{template}.xlsx",
                n=kw.pop("n", 8),
                inputs=inputs,
                response_name="resp",
                measurement_error=1.0,
                criterion="determinant",
                seed=0,
                n_starts=1,
                template=template,
                **kw,
            )
        )

    def test_box_behnken_workbook_has_the_published_run_count(self, tmp_path) -> None:
        out = self._new(
            tmp_path,
            "box-behnken",
            [("T", 300.0, 400.0), ("P", 1.0, 5.0), ("F", 0.1, 2.0)],
            basis="quadratic",
        )
        assert len(out["new_run_ids"]) == 15
        assert out["criterion"] == "classical"
        assert out["parameter_names"] == [
            "b0",
            "b1",
            "b2",
            "b3",
            "b11",
            "b22",
            "b33",
            "b12",
            "b13",
            "b23",
        ]

    def test_central_composite_supports_more_factors_than_the_rsm_templates(self, tmp_path) -> None:
        """k=4 has no response-surface-Nd template, but the quadratic basis covers it."""
        out = self._new(
            tmp_path,
            "central-composite",
            [(n, 0.0, 1.0) for n in ("a", "b", "c", "d")],
            basis="quadratic",
        )
        assert len(out["new_run_ids"]) == 2**4 + 2 * 4 + 4
        assert out["n_parameters"] == 15  # 1 + 4 main + 4 square + 6 cross

    def test_fit_recovers_a_known_quadratic(self, tmp_path) -> None:
        """Box-Behnken is saturated for a 3-factor quadratic: a noise-free fit is exact."""
        import openpyxl

        from discopt.doe.cli import do_fit

        truth = {
            "b0": 5.0,
            "b1": 0.02,
            "b2": -0.5,
            "b3": 1.5,
            "b11": -1e-4,
            "b22": 0.1,
            "b33": -0.3,
            "b12": 1e-3,
            "b13": 2e-3,
            "b23": -0.05,
        }

        def predict(T, P, F):
            return (
                truth["b0"]
                + truth["b1"] * T
                + truth["b2"] * P
                + truth["b3"] * F
                + truth["b11"] * T * T
                + truth["b22"] * P * P
                + truth["b33"] * F * F
                + truth["b12"] * T * P
                + truth["b13"] * T * F
                + truth["b23"] * P * F
            )

        out = self._new(
            tmp_path,
            "box-behnken",
            [("T", 300.0, 400.0), ("P", 1.0, 5.0), ("F", 0.1, 2.0)],
            basis="quadratic",
        )
        path = out["workbook_path"]

        book = openpyxl.load_workbook(path)
        sheet = book["runs"]
        headers = [c.value for c in sheet[1]]
        resp_col = headers.index("resp") + 1
        for row in sheet.iter_rows(min_row=2):
            values = dict(zip(headers, [c.value for c in row]))
            if values.get("run_id") is None:
                continue
            row[resp_col - 1].value = predict(values["T"], values["P"], values["F"])
        book.save(path)

        fitted = do_fit({"workbook": path})
        estimates = {p["name"]: p["estimate"] for p in fitted["parameters"]}
        for name, expected in truth.items():
            assert estimates[name] == pytest.approx(expected, rel=1e-6, abs=1e-9)
        assert fitted["objective"] < 1e-18
        # `extend` has no meaning for a closed-form design, so don't suggest it.
        assert "anova" in fitted["next_command"]

    def test_extend_is_rejected_with_an_actionable_message(self, tmp_path) -> None:
        from discopt.doe.cli import DoEError, ExtendParams, do_extend

        out = self._new(
            tmp_path, "central-composite", [("a", 0.0, 1.0), ("b", 0.0, 1.0)], basis="quadratic"
        )
        with pytest.raises((DoEError, ValueError), match="classical design template"):
            do_extend(ExtendParams(workbook=Path(out["workbook_path"]), n=2, n_starts=1))

    def test_latin_hypercube_defaults_to_a_linear_basis(self, tmp_path) -> None:
        out = self._new(
            tmp_path, "latin-hypercube", [("x", 0.0, 10.0), ("z", 0.0, 10.0)], n=12, basis="linear"
        )
        assert len(out["new_run_ids"]) == 12
        assert out["parameter_names"] == ["b0", "b1", "b2"]

    def test_generator_validation_surfaces_as_a_cli_error(self, tmp_path) -> None:
        from discopt.doe.cli import DoEError

        with pytest.raises(DoEError, match="3 to 5 factors"):
            self._new(tmp_path, "box-behnken", [("a", 0.0, 1.0), ("b", 0.0, 1.0)])
