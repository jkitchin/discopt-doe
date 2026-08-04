"""Python glue between the browser UI and ``discopt.doe``.

Loaded into Pyodide by ``app.js`` and driven entirely through the functions at
the bottom of this file. Every one takes and returns a JSON string, so nothing
but plain text crosses the JS boundary and no ``PyProxy`` needs to be kept
alive on the JavaScript side.

Two things make this thin rather than a reimplementation:

* Pyodide provides a virtual filesystem, so ``Workbook`` needs no changes —
  uploaded bytes are written to ``/work`` and read back from it. The app is
  stateless in the sense that matters (nothing is retained between visits, all
  state lives in the file the user holds), without the workbook layer having to
  become path-free.
* The base ``discopt`` package cannot be installed here — it ships only
  platform wheels and depends on jax, jaxlib, and a native solver, none of
  which have WebAssembly builds. So this drives only the dependency-light
  half of the package: the classical designs, the closed-form linear FIM,
  OLS fitting, and ANOVA. ``NewParams.use_linear_design`` is what routes the
  parametric templates away from autodiff; the designs it produces are
  identical, because every built-in template is linear in its parameters.
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable

WORK_DIR = Path("/work")

# Templates the browser build offers, grouped for the UI. The active-learning
# "optimize" template is absent: its surrogate round needs scikit-learn, which
# Pyodide does have, but the round itself is out of scope for phase 1.
TEMPLATE_GROUPS = [
    {
        "label": "Your own model",
        "hint": "Write the response formula; it is differentiated symbolically.",
        "templates": ["symbolic"],
    },
    {
        "label": "Space-filling & response surface",
        "hint": "Closed-form classical designs. No model assumed up front.",
        "templates": ["latin-hypercube", "central-composite", "box-behnken"],
    },
    {
        "label": "Screening & blocking",
        "hint": "Which factors matter, and how to block out nuisance variation.",
        "templates": ["factorial-2level", "latin-square", "graeco-latin", "hyper-graeco-latin"],
    },
    {
        "label": "Model-based optimal",
        "hint": "Maximize information about a specific model's parameters.",
        "templates": [
            "linear",
            "polynomial-1d",
            "response-surface-2d",
            "response-surface-3d",
            "scheffe-linear",
            "scheffe-quadratic",
            "scheffe-special-cubic",
        ],
    },
]

# Per-template UI metadata: which factor input style applies, and which extra
# options to render. Kept here rather than in JS so the CLI stays the single
# source of truth for what each template actually accepts.
TEMPLATE_UI: dict[str, dict[str, Any]] = {
    "symbolic": {
        "factors": "bounds",
        "options": ["n", "criterion"],
        "min": 1,
        "max": 6,
        # Drives the model editor: an expression box plus a parameter table.
        "model_editor": True,
        "example": {
            "expression": "k0 * exp(-Ea / (8.314 * T))",
            "parameters": [
                {"name": "k0", "value": 2.0},
                {"name": "Ea", "value": 5000.0},
            ],
            "factors": [{"name": "T", "low": 300.0, "high": 500.0}],
            "response": "rate",
            "n": 6,
        },
    },
    "latin-hypercube": {"factors": "bounds", "options": ["n", "basis"], "min": 1, "max": 12},
    "central-composite": {
        "factors": "bounds",
        "options": ["center_points", "alpha", "outside_bounds"],
        "min": 2,
        "max": 6,
    },
    "box-behnken": {"factors": "bounds", "options": ["center_points"], "min": 3, "max": 5},
    "factorial-2level": {
        "factors": "levels2",
        "options": ["center_points", "replicates"],
        "min": 2,
        "max": 8,
    },
    "latin-square": {"factors": "levels", "options": ["replicates"], "min": 3, "max": 3},
    "graeco-latin": {"factors": "levels", "options": ["replicates"], "min": 4, "max": 4},
    "hyper-graeco-latin": {"factors": "levels", "options": ["replicates"], "min": 5, "max": 5},
    "linear": {"factors": "bounds", "options": ["n", "criterion"], "min": 1, "max": 8},
    "polynomial-1d": {
        "factors": "bounds",
        "options": ["n", "degree", "criterion"],
        "min": 1,
        "max": 1,
    },
    "response-surface-2d": {"factors": "bounds", "options": ["n", "criterion"], "min": 2, "max": 2},
    "response-surface-3d": {"factors": "bounds", "options": ["n", "criterion"], "min": 3, "max": 3},
    "scheffe-linear": {
        "factors": "bounds",
        "options": ["n", "mixture_total", "criterion"],
        "min": 2,
        "max": 8,
    },
    "scheffe-quadratic": {
        "factors": "bounds",
        "options": ["n", "mixture_total", "criterion"],
        "min": 2,
        "max": 8,
    },
    "scheffe-special-cubic": {
        "factors": "bounds",
        "options": ["n", "mixture_total", "criterion"],
        "min": 3,
        "max": 8,
    },
}

DESIGN_PATH = WORK_DIR / "design.xlsx"
UPLOAD_PATH = WORK_DIR / "upload.xlsx"


def _sanitize(obj: Any) -> Any:
    """Replace NaN/±inf with None so ``json.dumps`` emits valid JSON.

    Mirrors the CLI's ``_sanitize_json``: JavaScript has no literal for these,
    and ``JSON.parse`` rejects the bare tokens Python would otherwise write.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps({"ok": True, **_sanitize(payload)})


def _err(message: str, *, detail: str = "") -> str:
    return json.dumps({"ok": False, "error": message, "detail": detail})


def _guard(fn: Callable[..., str]) -> Callable[..., str]:
    """Turn any exception into a JSON error rather than a JS-side throw."""

    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - the boundary must not leak
            return _err(str(e) or e.__class__.__name__, detail=traceback.format_exc())

    wrapper.__name__ = fn.__name__
    return wrapper


@_guard
def describe_templates() -> str:
    """Return the template catalogue plus the UI metadata for each."""
    from discopt.doe.cli import do_templates

    descriptions = {t["name"]: t["description"] for t in do_templates()["templates"]}
    groups = []
    for group in TEMPLATE_GROUPS:
        entries = [
            {
                "name": name,
                "description": descriptions.get(name, ""),
                **TEMPLATE_UI.get(name, {}),
            }
            for name in group["templates"]
            if name in descriptions
        ]
        groups.append({"label": group["label"], "hint": group["hint"], "templates": entries})
    return _ok({"groups": groups})


@_guard
def create_design(spec_json: str) -> str:
    """Build a design workbook at :data:`DESIGN_PATH` and return its summary.

    ``spec_json`` carries the template name, the factor rows, and whatever
    template-specific options the UI collected.
    """
    from discopt.doe.cli import NewParams, do_new
    from discopt.doe.linear_design import LINEAR_TEMPLATES

    spec = json.loads(spec_json)
    template = spec["template"]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_PATH.unlink(missing_ok=True)

    inputs: list[tuple[str, float, float]] = []
    levels: dict[str, list[object]] | None = None
    factor_pairs: dict[str, tuple[object, object]] | None = None

    style = TEMPLATE_UI.get(template, {}).get("factors", "bounds")
    if style == "bounds":
        for row in spec["factors"]:
            inputs.append((row["name"], float(row["low"]), float(row["high"])))
    elif style == "levels2":
        factor_pairs = {row["name"]: (row["low"], row["high"]) for row in spec["factors"]}
    else:  # "levels" — the Latin-square family
        levels = {
            row["name"]: [v.strip() for v in str(row["levels"]).split(",") if v.strip()]
            for row in spec["factors"]
        }

    # A user-defined model: the parameter table carries both the names (whose
    # order fixes the FIM layout) and the nominal values the design is centred on.
    nominal: dict[str, float] = {}
    for row in spec.get("parameters") or []:
        name = str(row.get("name") or "").strip()
        if name:
            nominal[name] = float(row.get("value") or 0.0)

    params = NewParams(
        output=DESIGN_PATH,
        n=int(spec.get("n") or 1),
        inputs=inputs,
        response_name=spec.get("response") or "y",
        measurement_error=float(spec.get("error") or 1.0),
        criterion=spec.get("criterion") or "determinant",
        seed=int(spec.get("seed") or 42),
        n_starts=int(spec.get("n_starts") or 8),
        template=template,
        degree=int(spec["degree"]) if spec.get("degree") not in (None, "") else None,
        mixture_total=(
            float(spec["mixture_total"]) if spec.get("mixture_total") not in (None, "") else None
        ),
        levels=levels,
        replicates=int(spec.get("replicates") or 1),
        factor_pairs=factor_pairs,
        center_points=int(spec.get("center_points") or 0),
        basis=spec.get("basis") or "quadratic",
        alpha=str(spec.get("alpha") or "rotatable"),
        within_bounds=not bool(spec.get("outside_bounds")),
        # The autodiff path cannot run here; the closed form is exact for every
        # one of these templates, so route the parametric family through it.
        use_linear_design=template in LINEAR_TEMPLATES,
        expression=(spec.get("expression") or None),
        param_initial_guess=nominal,
        force=True,
    )

    out = do_new(params)
    out["download_name"] = f"{template}-campaign.xlsx"
    out["file_path"] = str(DESIGN_PATH)
    return _ok(out)


@_guard
def check_model(spec_json: str) -> str:
    """Validate a model expression and report its symbolic derivatives.

    Drives the live feedback under the editor: a typo becomes a message while
    you are still typing, rather than an error when you press Generate. Also
    returns ∂y/∂θ so you can see what the design is actually being built from.
    """
    from discopt.doe.symbolic import SymbolicModel

    spec = json.loads(spec_json)
    names = [str(p.get("name") or "").strip() for p in (spec.get("parameters") or [])]
    names = [n for n in names if n]
    inputs = [str(f.get("name") or "").strip() for f in (spec.get("factors") or [])]
    inputs = [n for n in inputs if n]

    if not names:
        return _err("declare at least one parameter")
    if not inputs:
        return _err("declare at least one factor")

    model = SymbolicModel(
        source=str(spec.get("expression") or ""),
        parameter_names=tuple(names),
        input_names=tuple(inputs),
        response_name=spec.get("response") or "y",
        measurement_error=float(spec.get("error") or 1.0),
    )
    return _ok(
        {
            "expression": str(model.expression),
            "parameters": list(model.parameter_names),
            "inputs": list(model.input_names),
            "derivatives": [
                {"parameter": n, "expression": str(d)}
                for n, d in zip(model.parameter_names, model.jacobian_expressions)
            ],
        }
    )


@_guard
def inspect_workbook(path: str = str(UPLOAD_PATH)) -> str:
    """Return the campaign's status plus its runs, for the results table."""
    from discopt.doe.cli import do_status
    from discopt.doe.workbook import Workbook

    status = do_status({"workbook": path})
    wb = Workbook.open(Path(path))
    runs = wb.all_runs()
    response = wb.response_name()
    columns = ["run_id", *[s.name for s in wb.input_specs()], response]
    if any(r.get("replicate") is not None for r in runs):
        columns.insert(-1, "replicate")
    return _ok(
        {
            "status": status,
            "columns": columns,
            "rows": [{c: r.get(c) for c in columns} for r in runs],
            "response": response,
        }
    )


@_guard
def run_fit(path: str = str(UPLOAD_PATH)) -> str:
    """Fit the campaign's model to the completed runs (OLS over the basis)."""
    from discopt.doe.cli import do_fit

    return _ok({"fit": do_fit({"workbook": path})})


@_guard
def run_anova(path: str = str(UPLOAD_PATH), interactions_json: str = "[]") -> str:
    """Run ANOVA over the completed runs."""
    from discopt.doe.cli import do_anova

    pairs = [tuple(p) for p in json.loads(interactions_json or "[]")]
    return _ok(
        {"anova": do_anova({"workbook": path, "interactions": pairs, "include_replicate": False})}
    )


@_guard
def environment() -> str:
    """Report the runtime, for the footer and for bug reports."""
    import sys

    import numpy
    import scipy

    from discopt.doe.workbook import Workbook  # noqa: F401 - import must succeed

    version = "unknown"
    try:
        from importlib.metadata import version as _dist_version

        version = _dist_version("discopt-doe")
    except Exception:  # noqa: BLE001 - a missing dist is not fatal here
        pass

    return _ok(
        {
            "discopt_doe": version,
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        }
    )
