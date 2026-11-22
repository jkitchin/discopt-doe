"""The I/G criteria answer to every spelling the package uses.

``linear_design`` spells them ``average_prediction_variance`` /
``max_prediction_variance``, while ``DesignCriterion``, the CLI's ``--criterion``
choices and the design targets spell them ``average_variance`` /
``max_variance``. The CLI maps ``I``/``G`` to the latter, so before the basis
search accepted both spellings, ``discopt doe new symbolic --criterion I``
failed with "unknown criterion 'average_variance'".
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("openpyxl")

from discopt.doe import DesignCriterion  # noqa: E402
from discopt.doe.cli import (  # noqa: E402
    ExtendParams,
    add_subparser,
    do_extend,
)
from discopt.doe.linear_design import (  # noqa: E402
    G_OPTIMAL,
    I_OPTIMAL,
    linear_batch_design,
    normalize_criterion,
    target_criterion_name,
)
from discopt.doe.workbook import Workbook  # noqa: E402

SPELLINGS = {
    I_OPTIMAL: ["I", "V", "IV", "average_prediction_variance", "average_variance"],
    G_OPTIMAL: ["G", "max_prediction_variance", "max_variance"],
}


@pytest.mark.parametrize(
    "name, canonical",
    [(name, canon) for canon, names in SPELLINGS.items() for name in names],
)
def test_every_spelling_normalizes(name, canonical):
    assert normalize_criterion(name) == canonical


def test_design_criterion_spellings_are_accepted():
    """The jax-side enum's names work on the basis path, and back again."""
    assert normalize_criterion(DesignCriterion.I_OPTIMAL) == I_OPTIMAL
    assert normalize_criterion(DesignCriterion.G_OPTIMAL) == G_OPTIMAL
    assert target_criterion_name(DesignCriterion.I_OPTIMAL) == "average_variance"
    assert target_criterion_name(DesignCriterion.G_OPTIMAL) == "max_variance"


@pytest.mark.parametrize("criterion", ["average_variance", "max_variance"])
def test_linear_batch_design_takes_the_package_wide_names(criterion):
    """The same design as the canonical name, not a ValueError."""
    kw = dict(
        parameter_names=["b0", "b1", "b2"],
        input_names=["x"],
        design_bounds={"x": (-1.0, 1.0)},
        template_args={"degree": 2},
        n_starts=2,
        seed=0,
    )
    canonical = normalize_criterion(criterion)
    a = linear_batch_design("polynomial-1d", 6, criterion=criterion, **kw)
    b = linear_batch_design("polynomial-1d", 6, criterion=canonical, **kw)
    assert a.criterion_value == pytest.approx(b.criterion_value)


def _run_cli(argv: list[str]) -> int:
    top = argparse.ArgumentParser()
    add_subparser(top.add_subparsers(dest="cmd"))
    args = top.parse_args(argv)
    return args.doe_func(args)


@pytest.mark.parametrize("criterion", ["I", "G", "average_variance", "max_variance"])
def test_cli_new_symbolic_with_a_prediction_criterion(tmp_path, criterion):
    """`new symbolic --criterion I` (and G, and the long names) builds a design
    and `extend` continues it under the stored criterion."""
    path = tmp_path / "s.xlsx"
    rc = _run_cli(
        [
            "doe",
            "new",
            "symbolic",
            "--expr",
            "a + b*x + c*x**2",
            "--param",
            "a=1",
            "--param",
            "b=1",
            "--param",
            "c=1",
            "--input",
            "x:-1:1",
            "--n",
            "6",
            "--n-starts",
            "2",
            "--criterion",
            criterion,
            "-o",
            str(path),
        ]
    )
    assert rc == 0
    wb = Workbook.open(path)
    assert len(wb.pending_runs()) == 6
    wb.record_responses({int(r["run_id"]): 1.0 + 0.1 * i for i, r in enumerate(wb.pending_runs())})
    wb.save()
    ext = do_extend(ExtendParams(workbook=path, n=1, n_starts=2))
    assert len(ext["designs"]) == 1
