"""Categorical factors and several objectives in the BO loop.

Both were things the book had to do by hand in a notebook. The tests that earn
their place here are the ones covering the silent failures: a categorical coded
as a number invents an ordering nobody meant, and a scalarization written in
the wrong direction quietly optimizes the objective you down-weighted.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.doe.cli import NewParams, do_new
from discopt.doe.optimize import (
    _chebyshev_scalarization,
    _InputEncoder,
    _pareto_mask,
    optimize_round,
)
from discopt.doe.workbook import Workbook

# The default surrogate is the GP preset, which lives in the optional 'ml'
# extra. Skip the module on a core install rather than failing on the import.
pytest.importorskip("sklearn")
openpyxl = pytest.importorskip("openpyxl")

LEVELS = ["A", "B", "C"]


def _campaign(tmp_path, inputs, response="y", n=8):
    path = tmp_path / "w.xlsx"
    do_new(
        NewParams(
            output=path,
            n=n,
            inputs=inputs,
            response_name=response,
            measurement_error=0.1,
            criterion="determinant",
            seed=0,
            n_starts=2,
            template="latin-hypercube",
        )
    )
    return path


def _fill(path, fn, extra_columns=()):
    """Write responses (and any extra objective columns) by header name."""
    book = openpyxl.load_workbook(path)
    sheet = book["runs"]
    header = [c.value for c in sheet[1]]
    for name in extra_columns:
        if name not in header:
            header.append(name)
            sheet.cell(row=1, column=len(header), value=name)
    for row in sheet.iter_rows(min_row=2):
        if row[0].value is None:
            continue
        values = dict(zip(header, [c.value for c in row]))
        for name, value in fn(values).items():
            sheet.cell(row=row[0].row, column=header.index(name) + 1, value=value)
    book.save(path)


# ---------------------------------------------------------------------------
# Categorical factors
# ---------------------------------------------------------------------------


def test_one_hot_keeps_every_level_equidistant() -> None:
    """Coding levels 0, 1, 2 would put B between A and C. None of them is."""
    encoder = _InputEncoder(["x", "cat"], {"cat": LEVELS}, standardize=False)
    rows = [{"x": 0.0, "cat": level} for level in LEVELS]
    encoder.fit(rows)
    features = encoder.encode(rows)

    assert features.shape == (3, 4)  # x + three indicators
    distances = {
        (i, j): float(np.linalg.norm(features[i] - features[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    }
    assert len(set(round(d, 12) for d in distances.values())) == 1


def test_an_unknown_level_is_refused() -> None:
    encoder = _InputEncoder(["cat"], {"cat": LEVELS}, standardize=False)
    encoder.fit([{"cat": "A"}])
    with pytest.raises(ValueError, match="not one of its levels"):
        encoder.encode([{"cat": "D"}])


def test_a_categorical_must_be_an_input() -> None:
    with pytest.raises(ValueError, match="are not inputs"):
        _InputEncoder(["x"], {"cat": LEVELS}, standardize=False)


def test_levels_must_be_distinct_and_plural() -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        _InputEncoder(["cat"], {"cat": ["A"]}, standardize=False)
    with pytest.raises(ValueError, match="repeated levels"):
        _InputEncoder(["cat"], {"cat": ["A", "A"]}, standardize=False)


def test_a_round_proposes_across_the_levels_and_writes_labels(tmp_path) -> None:
    """The level is written back as itself, not as an index into a list."""
    path = _campaign(tmp_path, [("x", 0.0, 1.0), ("cat", 0.0, 1.0)], n=9)

    def fill(values):
        level = LEVELS[int(values["run_id"]) % 3]
        x = float(values["x"])
        peak = {"A": 1.0, "B": 2.0, "C": 0.5}[level]
        return {"cat": level, "y": peak * np.exp(-((x - 0.5) ** 2) / 0.05)}

    _fill(path, fill)
    result = optimize_round(
        path, batch_size=6, seed=1, n_candidates=300, categorical={"cat": LEVELS}
    )

    assert all(d["cat"] in LEVELS for d in result.next_designs)
    assert isinstance(result.incumbent_x["cat"], str)
    # The written rows carry the label, so the next round reads it back.
    rows = {int(r["run_id"]): r for r in Workbook.open(path).all_runs()}
    for run_id, design in zip(result.new_run_ids, result.next_designs):
        assert rows[run_id]["cat"] == design["cat"]


def test_every_level_is_offered_at_the_same_conditions(tmp_path) -> None:
    """Otherwise a level can lose because it was sampled at worse conditions."""
    from discopt.doe.optimize import _candidate_rows

    encoder = _InputEncoder(["x", "cat"], {"cat": LEVELS}, standardize=False)
    encoder.fit([{"x": 0.5, "cat": "A"}])
    rows = _candidate_rows(
        candidates=None,
        candidate_sampler="sobol",
        n_candidates=30,
        names=["x", "cat"],
        bounds_arr=np.array([[0.0, 1.0], [0.0, 1.0]]),
        encoder=encoder,
        rng=np.random.default_rng(0),
    )
    by_level = {level: sorted(r["x"] for r in rows if r["cat"] == level) for level in LEVELS}
    assert set(by_level) == set(LEVELS)
    assert by_level["A"] == by_level["B"] == by_level["C"]


# ---------------------------------------------------------------------------
# Several objectives
# ---------------------------------------------------------------------------


def test_pareto_mask_keeps_only_the_non_dominated() -> None:
    values = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_pareto_mask(values), [True, False, False, False, True])
    # Duplicated rows do not dominate each other.
    both = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(_pareto_mask(both), [True, True])


def test_a_bigger_weight_means_more_of_that_objective() -> None:
    """The inversion this guards against is silent: it still returns a front.

    ParEGO is written for minimization, so the weighted *cost* that is largest
    is the one the max picks up. Scalarizing the rewards instead flips which
    objective a weight buys.
    """
    # Two rows: the first is better on objective 0, the second on objective 1.
    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    favour_first = _chebyshev_scalarization(values, np.array([0.9, 0.1]))
    favour_second = _chebyshev_scalarization(values, np.array([0.1, 0.9]))

    assert favour_first[0] > favour_first[1]
    assert favour_second[1] > favour_second[0]


def test_a_round_returns_a_front_and_spreads_its_weights(tmp_path) -> None:
    path = _campaign(tmp_path, [("x", 0.0, 1.0)], response="yield", n=8)

    def fill(values):
        x = float(values["x"])
        return {"yield": 90.0 - 40.0 * (x - 0.75) ** 2, "impurity": 2.0 + 18.0 * x**2}

    _fill(path, fill, extra_columns=["impurity"])
    result = optimize_round(
        path,
        batch_size=4,
        seed=2,
        n_candidates=256,
        objectives={"yield": "maximize", "impurity": "minimize"},
    )

    assert result.incumbent_x is None and result.incumbent_y is None
    assert result.pareto_front and len(result.pareto_front) >= 2
    front = sorted(result.pareto_front, key=lambda r: r["yield"])
    # A real trade-off: more yield costs impurity, all along the front.
    assert all(front[i]["impurity"] < front[i + 1]["impurity"] for i in range(len(front) - 1))
    assert len(result.scalarization_weights) == 4
    assert all(abs(sum(w) - 1.0) < 1e-9 for w in result.scalarization_weights)
    assert len({tuple(np.round(w, 3)) for w in result.scalarization_weights}) > 1


def test_objectives_are_validated(tmp_path) -> None:
    path = _campaign(tmp_path, [("x", 0.0, 1.0)], n=6)
    _fill(path, lambda v: {"y": float(v["x"])})

    with pytest.raises(ValueError, match="at least two columns"):
        optimize_round(path, objectives=["y"], seed=0)
    with pytest.raises(ValueError, match="missing or not numeric"):
        optimize_round(path, objectives=["y", "not_a_column"], seed=0)
