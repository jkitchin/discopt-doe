"""Tests for ``quadratic_from_fit``: somebody else's fit, in this module's convention.

The translation has to be exact and it has to be *checkable*: a cross term
matched to the wrong pair still analyses cleanly and reports the wrong
stationary point, so the failure mode being guarded against here is silence.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.doe import canonical_analysis, quadratic_from_fit, stationary_point_ci

# A quadratic in three coded factors with a distinct value per coefficient, so
# any mis-mapping shows up as a wrong number rather than a coincidence.
TRUE = {
    "b0": 50.0,
    "b1": 2.0,
    "b2": -1.0,
    "b3": 0.5,
    "b11": -3.0,
    "b22": -2.0,
    "b33": -1.5,
    "b12": 0.7,
    "b13": -0.4,
    "b23": 0.2,
}
PATSY = {
    "Intercept": "b0",
    "x1": "b1",
    "x2": "b2",
    "x3": "b3",
    "I(x1 ** 2)": "b11",
    "I(x2 ** 2)": "b22",
    "I(x3 ** 2)": "b33",
    "x1:x2": "b12",
    "x1:x3": "b13",
    "x2:x3": "b23",
}


class FakeFit:
    """The duck type: named coefficients, a covariance, residual dof."""

    def __init__(self, params, cov=None, df_resid=30):
        self.params = params
        self._cov = cov
        self.df_resid = df_resid

    def cov_params(self):
        return self._cov


def _series(mapping):
    pd = pytest.importorskip("pandas")
    return pd.Series(list(mapping.values()), index=list(mapping.keys()))


def _patsy_params() -> dict[str, float]:
    return {term: TRUE[name] for term, name in PATSY.items()}


def test_translates_formula_terms_to_the_b_convention() -> None:
    fit = FakeFit(_patsy_params())
    out = quadratic_from_fit(fit, ["x1", "x2", "x3"])

    assert out["estimates"] == TRUE
    assert out["parameter_names"] == list(TRUE)
    assert out["degrees_of_freedom"] == 30
    # It says what it matched, so the mapping can be eyeballed once.
    assert out["terms"]["b12"] == "x1:x2"
    assert out["terms"]["b11"] == "I(x1 ** 2)"
    assert out["terms"]["b0"] == "Intercept"


def test_factor_order_fixes_the_meaning_of_every_coefficient() -> None:
    """Swapping the factor order swaps the coefficients, and the cross terms follow."""
    fit = FakeFit(_patsy_params())
    swapped = quadratic_from_fit(fit, ["x2", "x1", "x3"])["estimates"]

    assert swapped["b1"] == TRUE["b2"] and swapped["b2"] == TRUE["b1"]
    assert swapped["b11"] == TRUE["b22"] and swapped["b22"] == TRUE["b11"]
    # b12 is still the x1-x2 interaction, whichever order they are given in.
    assert swapped["b12"] == TRUE["b12"]
    assert swapped["b13"] == TRUE["b23"]  # now factor1=x2, factor3=x3
    assert swapped["b23"] == TRUE["b13"]


def test_covariance_is_reordered_to_match_the_coefficients() -> None:
    pd = pytest.importorskip("pandas")
    terms = list(PATSY)
    # A covariance whose entries encode their own position, so any mis-ordering
    # is visible in the result.
    raw = np.arange(100, dtype=float).reshape(10, 10)
    raw = raw + raw.T
    cov = pd.DataFrame(raw, index=terms, columns=terms)
    fit = FakeFit(_series(_patsy_params()), cov=cov)

    out = quadratic_from_fit(fit, ["x1", "x2", "x3"])
    order = [out["terms"][n] for n in out["parameter_names"]]
    expected = cov.loc[order, order].to_numpy()
    np.testing.assert_array_equal(out["covariance"], expected)


def test_unlabelled_covariance_follows_the_fits_own_order() -> None:
    terms = list(PATSY)
    raw = np.diag(np.arange(1.0, 11.0))
    fit = FakeFit(_series(_patsy_params()), cov=raw)

    out = quadratic_from_fit(fit, ["x1", "x2", "x3"])
    for name, term in out["terms"].items():
        i = out["parameter_names"].index(name)
        assert out["covariance"][i, i] == raw[terms.index(term), terms.index(term)]


def test_result_feeds_the_analysis_functions_directly() -> None:
    pd = pytest.importorskip("pandas")
    terms = list(PATSY)
    cov = pd.DataFrame(np.eye(10) * 0.01, index=terms, columns=terms)
    fit = FakeFit(_series(_patsy_params()), cov=cov, df_resid=30)
    out = quadratic_from_fit(fit, ["x1", "x2", "x3"])

    # Same answers as passing the coefficients in this module's own convention.
    assert canonical_analysis(out["estimates"]).stationary_point == pytest.approx(
        canonical_analysis(TRUE).stationary_point
    )
    ci = stationary_point_ci(out)
    direct = stationary_point_ci(
        list(TRUE.values()),
        np.eye(10) * 0.01,
        parameter_names=list(TRUE),
        dof=30,
    )
    np.testing.assert_allclose(ci.point, direct.point, rtol=1e-12)
    np.testing.assert_allclose(ci.lower, direct.lower, rtol=1e-12)


def test_alternative_spellings_are_recognized() -> None:
    params = {
        "const": TRUE["b0"],
        "A": TRUE["b1"],
        "B": TRUE["b2"],
        "A_sq": TRUE["b11"],
        "B^2": TRUE["b22"],
        "A*B": TRUE["b12"],
    }
    out = quadratic_from_fit(FakeFit(params, df_resid=None), ["A", "B"])
    assert out["estimates"] == {k: TRUE[k] for k in ("b0", "b1", "b2", "b11", "b22", "b12")}
    assert out["degrees_of_freedom"] is None


def test_whitespace_in_a_term_does_not_matter() -> None:
    params = dict(_patsy_params())
    params["I(x1**2)"] = params.pop("I(x1 ** 2)")
    out = quadratic_from_fit(FakeFit(params), ["x1", "x2", "x3"])
    assert out["estimates"]["b11"] == TRUE["b11"]


def test_an_unrecognized_term_is_named_not_guessed() -> None:
    params = dict(_patsy_params())
    params["weird_x1_x2"] = params.pop("x1:x2")
    with pytest.raises(ValueError, match="could not find a term for b12"):
        quadratic_from_fit(FakeFit(params), ["x1", "x2", "x3"])

    out = quadratic_from_fit(FakeFit(params), ["x1", "x2", "x3"], aliases={"b12": "weird_x1_x2"})
    assert out["estimates"] == TRUE


def test_missing_terms_list_what_was_searched_for() -> None:
    params = {k: v for k, v in _patsy_params().items() if k != "I(x2 ** 2)"}
    with pytest.raises(ValueError) as excinfo:
        quadratic_from_fit(FakeFit(params), ["x1", "x2", "x3"])
    message = str(excinfo.value)
    assert "b22" in message and "I(x2 ** 2)" in message
    assert "aliases" in message  # tells the caller what to do about it


def test_repeated_or_unusable_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="distinct"):
        quadratic_from_fit(FakeFit(_patsy_params()), ["x1", "x1"])
    with pytest.raises(TypeError, match="named coefficients"):
        quadratic_from_fit(object(), ["x1", "x2"])


def test_matches_a_real_statsmodels_fit() -> None:
    """The hand-built two-list translation, and this, must agree exactly."""
    pd = pytest.importorskip("pandas")
    smf = pytest.importorskip("statsmodels.formula.api")

    rng = np.random.default_rng(0)
    n = 40
    d = pd.DataFrame({f"x{i}": rng.uniform(-1, 1, n) for i in (1, 2, 3)})
    d["y"] = (
        TRUE["b0"]
        + TRUE["b1"] * d.x1
        + TRUE["b2"] * d.x2
        + TRUE["b3"] * d.x3
        + TRUE["b11"] * d.x1**2
        + TRUE["b22"] * d.x2**2
        + TRUE["b33"] * d.x3**2
        + TRUE["b12"] * d.x1 * d.x2
        + TRUE["b13"] * d.x1 * d.x3
        + TRUE["b23"] * d.x2 * d.x3
        + rng.normal(0, 0.05, n)
    )
    quad = smf.ols(
        "y ~ x1 + x2 + x3 + I(x1**2) + I(x2**2) + I(x3**2) + x1:x2 + x1:x3 + x2:x3", d
    ).fit()

    idx = list(PATSY)
    keys = [PATSY[t] for t in idx]
    by_hand = dict(zip(keys, quad.params[idx].to_numpy()))
    cov_by_hand = quad.cov_params().loc[idx, idx].to_numpy()

    out = quadratic_from_fit(quad, ["x1", "x2", "x3"])
    assert out["estimates"] == by_hand
    np.testing.assert_array_equal(out["covariance"], cov_by_hand)
    assert out["degrees_of_freedom"] == int(quad.df_resid)

    a = stationary_point_ci(out)
    b = stationary_point_ci(list(by_hand.values()), cov_by_hand, parameter_names=keys, dof=30)
    np.testing.assert_allclose(a.point, b.point, rtol=1e-12)
    np.testing.assert_allclose(a.upper, b.upper, rtol=1e-12)
