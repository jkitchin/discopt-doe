"""Guard the browser-importable subset of the package.

A WASM/Pyodide build can install ``discopt-doe`` (it is a pure-Python wheel) but
*cannot* install the base ``discopt`` package: it ships only platform wheels and
depends on jax, jaxlib, and a native solver, none of which have WebAssembly
builds. In that environment ``discopt`` exists purely as a PEP 420 namespace
portion contributed by this distribution, and ``discopt.estimate`` /
``discopt.modeling`` / ``jax`` are simply absent.

The tests below reproduce that environment in a subprocess and assert that the
dependency-light modules still import and work. They are the regression guard
for the lazy-import structure in ``discopt/doe/__init__.py``: a stray
module-level ``from discopt.estimate import ...`` anywhere in the modules listed
in ``BROWSER_SAFE_MODULES`` will fail here and nowhere else in the suite.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

# Modules that must import with no base-discopt and no jax present.
BROWSER_SAFE_MODULES = [
    "discopt.doe.anova",
    # cli and templates are on the list because the browser app drives the same
    # do_new/do_fit/do_anova entry points the CLI does. templates in particular
    # is easy to regress: it builds discopt Models, and the import that does so
    # has to stay inside the functions that need it.
    "discopt.doe.cli",
    "discopt.doe.classical",
    "discopt.doe.latin",
    "discopt.doe.linear_design",
    "discopt.doe.screening",
    "discopt.doe.simplex",
    "discopt.doe.symbolic",
    "discopt.doe.templates",
    "discopt.doe.workbook",
]

# Importing any of the above must not drag these in.
FORBIDDEN = ["jax", "jaxlib", "discopt.estimate", "discopt.modeling", "discopt.parametric"]

# Prelude that turns the interpreter into a stand-in for the browser: `discopt`
# becomes a bare namespace package rooted at src/, and jax is unimportable.
PRELUDE = f"""
import sys, types
from importlib.machinery import ModuleSpec

SRC = {str(SRC)!r}
sys.path.insert(0, SRC)

# `discopt` as a namespace portion only -- never the installed package, whose
# __init__ eagerly imports discopt.modeling and would mask the very coupling
# this test exists to detect.
_ns = types.ModuleType("discopt")
_ns.__path__ = [SRC + "/discopt"]
_ns.__spec__ = ModuleSpec("discopt", loader=None, is_package=True)
_ns.__spec__.submodule_search_locations = _ns.__path__
sys.modules["discopt"] = _ns


class _Blocker:
    \"\"\"Make jax unimportable, as it is under Pyodide.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError(f"no WebAssembly build of {{name}} (simulated)")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a fresh interpreter set up as the browser stand-in."""
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        # cwd matters: running from the repo root would put the real src/ on
        # sys.path[0] via the rootdir conftest and defeat the namespace seeding.
        cwd=str(Path(__file__).resolve().parent),
    )


@pytest.mark.parametrize("module", BROWSER_SAFE_MODULES)
def test_module_imports_without_discopt_or_jax(module: str) -> None:
    """Each dependency-light module imports with no base package and no jax."""
    proc = _run(f"""
        import importlib
        importlib.import_module({module!r})
        print("OK")
    """)
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_no_forbidden_modules_are_pulled_in() -> None:
    """Importing the whole browser-safe set leaves jax and the base package out."""
    proc = _run(f"""
        import importlib, sys
        for name in {BROWSER_SAFE_MODULES!r}:
            importlib.import_module(name)
        leaked = sorted(m for m in {FORBIDDEN!r} if m in sys.modules)
        print("LEAKED:", leaked)
        sys.exit(1 if leaked else 0)
    """)
    assert proc.returncode == 0, f"heavy dependencies leaked in:\n{proc.stdout}{proc.stderr}"


def test_package_import_is_lazy() -> None:
    """`import discopt.doe` itself must not eagerly load the jax-bound modules."""
    proc = _run("""
        import sys
        import discopt.doe
        eager = sorted(
            m for m in sys.modules
            if m.startswith("discopt.doe.") and m.split(".")[-1] in {"fim", "design", "model_based"}
        )
        print("EAGER:", eager)
        sys.exit(1 if eager else 0)
    """)
    assert proc.returncode == 0, f"submodules imported eagerly:\n{proc.stdout}{proc.stderr}"


def test_lazy_attribute_access_still_resolves() -> None:
    """The lazy __getattr__ resolves a jax-free export and rejects unknown names."""
    proc = _run("""
        import discopt.doe as d
        assert callable(d.anova_report), "anova_report did not resolve"
        assert callable(d.latin_square_design), "latin_square_design did not resolve"
        assert "anova_report" in dir(d), "__dir__ omitted a public name"
        try:
            d.definitely_not_a_real_export
        except AttributeError:
            print("OK")
        else:
            raise AssertionError("unknown attribute did not raise AttributeError")
    """)
    assert proc.returncode == 0, f"lazy access broke:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_mixture_design_without_base_package(tmp_path: Path) -> None:
    """A Scheffé design builds with no base discopt and no jax.

    Importing ``discopt.doe.cli`` was never enough to catch this: the mixture
    branch reached for ``project_to_simplex`` and ``sum_constraint`` *inside*
    the function, through ``discopt.doe``, whose package-level names resolved
    via ``discopt.doe.design`` — which imports the FIM machinery and
    ``discopt.estimate``. Every mixture template in the browser app therefore
    failed at design time on ``No module named 'discopt.estimate'``, while
    every test in this suite passed. So run the design, do not just import it.
    """
    target = tmp_path / "mixture.xlsx"
    proc = _run(f"""
        from discopt.doe.cli import NewParams, do_new

        out = do_new(NewParams(
            output={str(target)!r},
            n=8,
            inputs=[("a", 0.0, 1.0), ("b", 0.0, 1.0), ("c", 0.0, 1.0)],
            response_name="y",
            measurement_error=1.0,
            criterion="determinant",
            n_starts=4,
            seed=0,
            template="scheffe-quadratic",
            mixture_total=1.0,
            use_linear_design=True,
            force=True,
        ))
        assert len(out["new_run_ids"]) == 8, out["new_run_ids"]
        # The components of every run must lie on the simplex it was
        # constrained to — that is what the helpers are for.
        for design in out["designs"]:
            total = design["a"] + design["b"] + design["c"]
            assert abs(total - 1.0) < 1e-6, (design, total)
        print("OK")
    """)
    assert proc.returncode == 0, f"mixture design failed:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_workbook_round_trip_without_base_package(tmp_path: Path) -> None:
    """A campaign workbook can be created and reopened with no base discopt.

    This is the web app's core loop: build a design in the browser, hand the
    user a .xlsx, read it back after they fill it in.
    """
    target = tmp_path / "campaign.xlsx"
    proc = _run(f"""
        from discopt.doe.workbook import InputSpec, Workbook

        wb = Workbook.create(
            {str(target)!r},
            input_specs=[InputSpec("T", 300.0, 400.0)],
            response_name="y",
            template="linear",
            template_args={{}},
            criterion="determinant",
            measurement_error=1.0,
            seed=42,
        )
        wb.append_runs(0, [{{"T": 325.0}}, {{"T": 375.0}}])
        wb.save()

        reopened = Workbook.open({str(target)!r})
        assert [s.name for s in reopened.input_specs()] == ["T"]
        assert len(reopened.pending_runs()) == 2
        # The base package is absent, so its version stamp degrades gracefully
        # rather than raising.
        assert reopened.metadata()["discopt_version"] == "unknown"
        print("OK")
    """)
    assert proc.returncode == 0, f"workbook round-trip failed:\n{proc.stderr}"
    assert "OK" in proc.stdout
    assert target.is_file()
