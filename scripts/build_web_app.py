"""Assemble the browser app into a servable directory.

Builds the wheel, copies ``web/`` next to it, and writes the manifest
``app.js`` reads to find the wheel. The same steps run in CI (see
``.github/workflows/deploy-book.yml``); this script is what you run locally.

    uv run python scripts/build_web_app.py --serve

Then open the printed URL. ``--serve`` starts a plain HTTP server because
Pyodide needs real HTTP requests — opening ``index.html`` over ``file://``
fails on the module import and the ``fetch`` calls.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_SRC = ROOT / "web"
DEFAULT_OUT = ROOT / "docs" / "_build" / "web"


def build_wheel(dist: Path) -> Path:
    """Build the wheel into ``dist`` and return its path."""
    if dist.exists():
        shutil.rmtree(dist)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv not found on PATH; install it from https://docs.astral.sh/uv/")
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True)
    wheels = sorted(dist.glob("discopt_doe-*.whl"))
    if not wheels:
        raise SystemExit(f"no wheel produced in {dist}")
    return wheels[-1]


def assemble(out_dir: Path) -> Path:
    """Copy the app and its wheel into ``out_dir``, returning the wheel name."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(WEB_SRC, out_dir)

    wheel_dir = out_dir / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = build_wheel(ROOT / "dist")
    shutil.copy2(wheel, wheel_dir / wheel.name)
    (wheel_dir / "manifest.json").write_text(json.dumps({"wheel": wheel.name}, indent=2) + "\n")
    return wheel.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--serve", action="store_true", help="serve the result over HTTP")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    wheel_name = assemble(args.out)
    print(f"built {args.out} (wheel: {wheel_name})")

    if args.serve:
        import http.server
        import socketserver

        handler = type(
            "Handler",
            (http.server.SimpleHTTPRequestHandler,),
            {"directory": str(args.out)},
        )
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
            print(f"serving on http://127.0.0.1:{args.port}/ — Ctrl-C to stop")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
