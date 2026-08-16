"""Execute the docs notebooks in place and commit their outputs.

``docs/_config.yml`` sets ``execute_notebooks: "off"``, so whatever outputs are
committed are exactly what the published book shows. That is fast and keeps the
docs build hermetic, but it also means nothing notices when the code drifts away
from the numbers on the page -- which is how ``latin-designs.ipynb`` came to ship
a table the library now refuses to compute.

This is the missing half of the workflow. ``scripts/build_*_notebook.py`` write
the notebook *source*; this executes it and writes the outputs back:

    uv run python scripts/execute_notebooks.py                 # all of them
    uv run python scripts/execute_notebooks.py latin-designs   # just one
    uv run python scripts/execute_notebooks.py --check         # fail if any errors

``--check`` executes without saving, so CI can gate on "every notebook still
runs" without carrying a diff.

Each notebook runs with its own directory as the working directory, matching
how Jupyter Book executes them, so relative paths in a notebook mean the same
thing here as they do there. Workbook files a notebook writes as a side effect
are removed afterwards -- they are build artifacts, not sources.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "docs" / "notebooks"

# Terminal colour codes from the solver's tracing layer. They render as literal
# escape sequences in the published HTML, so strip them from stored outputs.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Side-effect files a notebook writes into its own directory. Kept out of the
# repo: the notebook that needs one creates it.
_ARTIFACT_SUFFIXES = (".xlsx", ".xlsx.bak", ".csv")


def _tidy(nb: nbformat.NotebookNode) -> None:
    """Strip colour codes and wall-clock timings from the stored outputs.

    The timings differ on every run, so leaving them in means a notebook shows
    a diff whether or not any number on the page actually changed.
    """
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = _ANSI.sub("", out["text"])
            for key, val in (out.get("data") or {}).items():
                if key.startswith("text/") and isinstance(val, str):
                    out["data"][key] = _ANSI.sub("", val)


def _artifacts() -> set[Path]:
    return {
        p
        for p in NOTEBOOK_DIR.iterdir()
        if p.is_file() and any(p.name.endswith(s) for s in _ARTIFACT_SUFFIXES)
    }


def execute(path: Path, *, save: bool, timeout: int) -> bool:
    """Execute one notebook. Return True on success."""
    before = _artifacts()
    nb = nbformat.read(path, as_version=4)
    # Hand-edited notebooks can be missing per-cell ``id`` fields, which
    # nbformat warns about now and will reject later.
    nbformat.validator.normalize(nb)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    try:
        client.execute()
    except CellExecutionError as err:
        print(f"FAIL  {path.name}\n{err}", file=sys.stderr)
        return False
    finally:
        for leftover in _artifacts() - before:
            leftover.unlink(missing_ok=True)

    if save:
        _tidy(nb)
        nbformat.write(nb, str(path))
    print(f"ok    {path.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="notebook stems (default: all)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="execute without writing outputs back; exit non-zero on any failure",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout, seconds")
    args = ap.parse_args()

    if args.names:
        paths = [NOTEBOOK_DIR / f"{n.removesuffix('.ipynb')}.ipynb" for n in args.names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            ap.error(f"no such notebook(s): {', '.join(p.name for p in missing)}")
    else:
        paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    failed = [p.name for p in paths if not execute(p, save=not args.check, timeout=args.timeout)]
    if failed:
        print(f"\n{len(failed)} notebook(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n{len(paths)} notebook(s) executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
