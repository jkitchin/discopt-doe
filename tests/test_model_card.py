"""Tests for the JSON model card and the fit metadata a workbook stores.

A card has to be enough, on its own, to use a fitted model: the same
prediction, the same intervals, no code executed on open. The workbook side has
to make the noise estimate recoverable, since the information matrix it stores
is deliberately on a different scale.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from discopt.doe import ModelCard, SymbolicModel
from discopt.doe.card import SCHEMA_VERSION, digest_rows
from discopt.doe.symbolic import ModelSyntaxError, fit_least_squares

TRUE = {"Vmax": 10.0, "Km": 2.0}
SIGMA = 0.25


def _model() -> SymbolicModel:
    return SymbolicModel(
        source="Vmax * S / (Km + S)",
        parameter_names=("Vmax", "Km"),
        input_names=("S",),
        response_name="rate",
        measurement_error=SIGMA,
    )


def _rows(n: int = 12, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    model = _model()
    out = []
    for s in np.linspace(0.5, 20.0, n):
        y = model.predict(TRUE, {"S": float(s)}) + rng.normal(0.0, SIGMA)
        out.append({"S": float(s), "rate": float(y)})
    return out


def _fitted() -> tuple[SymbolicModel, dict, list[dict[str, float]]]:
    model, rows = _model(), _rows()
    fit = fit_least_squares(model, rows, TRUE)
    return model, fit, rows


# ---------------------------------------------------------------------------
# SymbolicModel on its own
# ---------------------------------------------------------------------------


def test_model_json_round_trip_keeps_response_and_sigma() -> None:
    """``to_metadata`` drops both, because a workbook stores them separately."""
    model = _model()
    back = SymbolicModel.from_json(model.to_json())
    assert back.source == model.source
    assert back.parameter_names == model.parameter_names
    assert back.input_names == model.input_names
    assert back.response_name == "rate"
    assert back.measurement_error == pytest.approx(SIGMA)
    assert back.predict(TRUE, {"S": 4.0}) == model.predict(TRUE, {"S": 4.0})


def test_model_json_is_parsed_not_executed() -> None:
    payload = json.loads(_model().to_json())
    payload["expression"] = "__import__('os').system('true')"
    with pytest.raises(ModelSyntaxError):
        SymbolicModel.from_json(json.dumps(payload))


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_card_from_fit_carries_the_whole_fit() -> None:
    model, fit, rows = _fitted()
    card = ModelCard.from_fit(model, fit, design_region={"S": (0.5, 20.0)}, rows=rows, note="n=12")

    assert card.estimates == {k: pytest.approx(fit["estimates"][k]) for k in fit["estimates"]}
    np.testing.assert_allclose(card.covariance, fit["covariance"], rtol=1e-12)
    assert card.sigma == pytest.approx(fit["sigma"])
    assert card.sigma_source == fit["sigma_source"]
    assert card.n_observations == fit["n_observations"]
    assert card.degrees_of_freedom == fit["degrees_of_freedom"]
    assert card.design_region == {"S": (0.5, 20.0)}
    assert card.data_digest == digest_rows(rows)
    assert "discopt-doe" in card.created_with
    # The standard errors it reports are the ones the fit reported.
    for name, se in card.standard_errors().items():
        assert se == pytest.approx(fit["std_errors"][name], rel=1e-10)


def test_card_round_trip_predicts_identically(tmp_path) -> None:
    model, fit, rows = _fitted()
    card = ModelCard.from_fit(model, fit, design_region={"S": (0.5, 20.0)}, rows=rows)
    path = card.save(tmp_path / "card.json")

    back = ModelCard.load(path)
    for s in (1.0, 7.5, 19.0):
        x = {"S": s}
        assert back.predict(x) == card.predict(x)  # exactly, not approximately
        assert back.standard_error(x) == pytest.approx(card.standard_error(x), rel=1e-12)
        assert back.interval(x) == pytest.approx(card.interval(x), rel=1e-12)
    assert back.data_digest == card.data_digest
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_card_intervals_match_the_delta_method_by_hand() -> None:
    model, fit, rows = _fitted()
    card = ModelCard.from_fit(model, fit, rows=rows)
    x = {"S": 6.0}

    j = np.asarray(model.jacobian_row(card.estimates, x), dtype=float)
    se = math.sqrt(j @ np.asarray(fit["covariance"]) @ j)
    assert card.standard_error(x) == pytest.approx(se, rel=1e-12)

    from scipy.stats import t as t_dist

    crit = float(t_dist.ppf(0.975, df=fit["degrees_of_freedom"]))
    centre = card.predict(x)
    np.testing.assert_allclose(
        card.interval(x), (centre - crit * se, centre + crit * se), rtol=1e-12
    )
    # A prediction interval carries the measurement noise as well, so it is wider.
    wide = card.interval(x, kind="prediction")
    assert wide[0] < centre - crit * se and wide[1] > centre + crit * se
    np.testing.assert_allclose(
        wide,
        (
            centre - crit * math.sqrt(se**2 + card.sigma**2),
            centre + crit * math.sqrt(se**2 + card.sigma**2),
        ),
        rtol=1e-12,
    )


def test_card_uses_the_normal_quantile_for_a_known_sigma() -> None:
    """A declared sigma is not estimated from the fit, so there is no t penalty."""
    model, fit, rows = _fitted()
    known = dict(fit, sigma=SIGMA, sigma_source="declared")
    card = ModelCard.from_fit(model, known, rows=rows)
    x = {"S": 6.0}
    from scipy.stats import norm

    half = float(norm.ppf(0.975)) * card.standard_error(x)
    lo, hi = card.interval(x)
    assert (hi - lo) / 2 == pytest.approx(half, rel=1e-12)


def test_card_fim_inverts_the_covariance() -> None:
    model, fit, rows = _fitted()
    card = ModelCard.from_fit(model, fit, rows=rows)
    np.testing.assert_allclose(card.fim() @ card.covariance, np.eye(2), atol=1e-8)


def test_design_region_tells_interpolation_from_extrapolation() -> None:
    model, fit, rows = _fitted()
    card = ModelCard.from_fit(model, fit, design_region={"S": (0.5, 20.0)}, rows=rows)
    assert card.inside_design_region({"S": 10.0})
    assert not card.inside_design_region({"S": 40.0})
    # With no region recorded there is nothing to judge against.
    assert ModelCard.from_fit(model, fit, rows=rows).inside_design_region({"S": 40.0})


def test_digest_identifies_the_data_and_ignores_key_order() -> None:
    rows = _rows()
    assert digest_rows(rows) == digest_rows([dict(reversed(list(r.items()))) for r in rows])
    changed = [dict(r) for r in rows]
    changed[0]["rate"] += 1e-9
    assert digest_rows(changed) != digest_rows(rows)


def test_card_rejects_a_covariance_that_does_not_match_the_parameters() -> None:
    model, fit, _ = _fitted()
    with pytest.raises(ValueError, match="covariance must be 2x2"):
        ModelCard(model=model, estimates=fit["estimates"], covariance=np.eye(3), sigma=0.3)
    with pytest.raises(ValueError, match="missing parameter"):
        ModelCard(model=model, estimates={"Vmax": 1.0}, covariance=np.eye(2), sigma=0.3)
    with pytest.raises(ValueError, match="sigma must be positive"):
        ModelCard(model=model, estimates=fit["estimates"], covariance=np.eye(2), sigma=0.0)


def test_card_refuses_a_future_schema() -> None:
    model, fit, rows = _fitted()
    payload = ModelCard.from_fit(model, fit, rows=rows).to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than this version"):
        ModelCard.from_dict(payload)


def test_card_from_fit_names_what_a_bad_fit_dict_is_missing() -> None:
    model = _model()
    with pytest.raises(ValueError, match="missing"):
        ModelCard.from_fit(model, {"estimates": {"Vmax": 1.0, "Km": 1.0}})


# ---------------------------------------------------------------------------
# What the workbook stores about a fit
# ---------------------------------------------------------------------------


def _campaign(tmp_path, n: int = 8):
    """A user-defined-model campaign, filled in and fitted through the CLI."""
    import openpyxl
    from discopt.doe.cli import NewParams, do_fit, do_new

    path = tmp_path / "campaign.xlsx"
    do_new(
        NewParams(
            output=path,
            n=n,
            inputs=[("S", 0.5, 20.0)],
            response_name="rate",
            measurement_error=SIGMA,
            criterion="determinant",
            seed=0,
            n_starts=4,
            template="symbolic",
            expression="Vmax * S / (Km + S)",
            param_initial_guess=dict(TRUE),
        )
    )
    rng = np.random.default_rng(3)
    model = _model()
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    head = [c.value for c in sheet[1]]
    col = head.index("rate")
    for row in sheet.iter_rows(min_row=2):
        values = dict(zip(head, [c.value for c in row]))
        if values.get("run_id") is None:
            continue
        y = model.predict(TRUE, {"S": float(values["S"])}) + rng.normal(0.0, SIGMA)
        row[col].value = float(y)
    book.save(path)
    return path, do_fit({"workbook": str(path)})


def test_workbook_records_the_sigma_behind_its_standard_errors(tmp_path) -> None:
    """Without it, the stored information matrix cannot be read as a covariance."""
    from discopt.doe.workbook import Workbook

    path, result = _campaign(tmp_path)
    assert result["sigma"] > 0
    assert result["sigma_source"] in ("residual", "declared")

    wb = Workbook.open(path)
    sigma_hat, source = wb.fitted_sigma()
    assert sigma_hat == pytest.approx(result["sigma"])
    assert source == result["sigma_source"]

    cov, names = wb.read_covariance()
    stored = {p["name"]: p["std_error"] for p in wb.read_parameters()}
    for i, name in enumerate(names):
        assert math.sqrt(cov[i, i]) == pytest.approx(stored[name], rel=1e-6)


def test_a_card_can_be_rebuilt_from_a_workbook_alone(tmp_path) -> None:
    """The file is now self-describing: model, estimates, covariance, sigma."""
    from discopt.doe.workbook import Workbook

    path, _ = _campaign(tmp_path)
    wb = Workbook.open(path)
    cov, names = wb.read_covariance()
    sigma_hat, source = wb.fitted_sigma()
    pars = {p["name"]: p for p in wb.read_parameters()}

    card = ModelCard(
        model=wb.symbolic_model(),
        estimates={n: pars[n]["estimate"] for n in names},
        covariance=cov,
        sigma=sigma_hat,
        sigma_source=source,
        n_observations=len(wb.completed_runs()),
        degrees_of_freedom=len(wb.completed_runs()) - len(names),
    )
    for name in names:
        assert card.standard_errors()[name] == pytest.approx(pars[name]["std_error"], rel=1e-6)
    assert card.predict({"S": 10.0}) > 0
    assert ModelCard.from_json(card.to_json()).predict({"S": 10.0}) == card.predict({"S": 10.0})


def test_read_covariance_is_none_before_a_fit(tmp_path) -> None:
    from discopt.doe.cli import NewParams, do_new
    from discopt.doe.workbook import Workbook

    path = tmp_path / "fresh.xlsx"
    do_new(
        NewParams(
            output=path,
            n=6,
            inputs=[("S", 0.5, 20.0)],
            response_name="rate",
            measurement_error=SIGMA,
            criterion="determinant",
            seed=0,
            n_starts=4,
            template="symbolic",
            expression="Vmax * S / (Km + S)",
            param_initial_guess=dict(TRUE),
        )
    )
    wb = Workbook.open(path)
    assert wb.fitted_sigma() == (None, None)
    assert wb.read_covariance() is None
