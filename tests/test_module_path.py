"""``--module`` models resolve from the working directory and the workbook's directory.

Issue #48: run as a console script, ``sys.path[0]`` is the script's bin
directory, so a ``my_model.py`` next to the user (or the workbook) was not
importable without ``PYTHONPATH``. These tests drive the real ``discopt``
console script with ``PYTHONPATH`` unset, so they exercise exactly that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("jax")

from discopt.doe.workbook import _load_module_callable  # noqa: E402

_MODEL = '''
import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel


class KExp(Experiment):
    """y = k * x."""

    def create_model(self, **kwargs):
        m = dm.Model("k_exp")
        k = m.continuous("k", lb=0.01, ub=20.0)
        x = m.continuous("x", lb=0.1, ub=10.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"x": x},
            responses={"y": k * x},
            measurement_error={"y": 0.1},
        )


def make():
    return KExp()
'''


def _discopt() -> str:
    exe = Path(sys.executable).with_name("discopt")
    found = str(exe) if exe.exists() else shutil.which("discopt")
    if not found:
        pytest.skip("discopt console script not installed")
    return found


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [_discopt(), "doe", *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _fill(path: Path) -> None:
    wb = openpyxl.load_workbook(path)
    runs = wb["runs"]
    headers = [c.value for c in runs[1]]
    xi, yi = headers.index("x"), headers.index("y")
    for row in runs.iter_rows(min_row=2):
        if row[0].value is not None and row[yi].value is None:
            row[yi].value = 2.5 * float(row[xi].value)
    wb.save(path)


@pytest.fixture
def campaign_dir(tmp_path):
    d = tmp_path / "campaign"
    d.mkdir()
    (d / "issue48_model.py").write_text(_MODEL)
    (tmp_path / "elsewhere").mkdir()
    return d


NEW = ["new", "module", "--module", "issue48_model:make", "--bounds", "x:0.1:10"]
NEW += ["--params", "k=1.5", "--n", "2"]


def test_module_resolves_from_cwd(campaign_dir):
    _run([*NEW, "-o", "w.xlsx"], cwd=campaign_dir)
    _fill(campaign_dir / "w.xlsx")
    assert "2.5" in _run(["fit", "w.xlsx"], cwd=campaign_dir).stdout
    _run(["extend", "w.xlsx", "--n", "1"], cwd=campaign_dir)
    _run(["status", "w.xlsx"], cwd=campaign_dir)


def test_module_resolves_from_workbook_dir(campaign_dir):
    elsewhere = campaign_dir.parent / "elsewhere"
    wb = campaign_dir / "w.xlsx"
    # `new` run from another directory finds the model next to the output.
    _run([*NEW, "-o", str(wb)], cwd=elsewhere)
    _fill(wb)
    assert "2.5" in _run(["fit", str(wb)], cwd=elsewhere).stdout
    _run(["extend", str(wb), "--n", "1"], cwd=elsewhere)
    _run(["status", str(wb)], cwd=elsewhere)


def test_sys_path_restored(tmp_path):
    (tmp_path / "issue48_inproc.py").write_text(_MODEL)
    sys.modules.pop("issue48_inproc", None)
    before = list(sys.path)
    try:
        exp = _load_module_callable("issue48_inproc:make", search_dirs=[tmp_path])
    finally:
        sys.modules.pop("issue48_inproc", None)
    assert type(exp).__name__ == "KExp"
    assert sys.path == before
