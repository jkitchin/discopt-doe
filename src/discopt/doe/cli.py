"""``discopt doe`` — Excel-workbook-driven optimal experimental design.

Eight verbs make up the CLI:

* ``discopt doe templates`` — list the available template models.
* ``discopt doe new TEMPLATE [args] -o file.xlsx --n N`` — start a
  campaign with N optimal initial runs.
* ``discopt doe status file.xlsx`` — show how many runs are complete,
  pending, and what to do next.
* ``discopt doe fit file.xlsx`` — estimate parameters from completed
  runs and refresh the FIM.
* ``discopt doe anova file.xlsx`` — ANOVA F-table for latin/factorial designs.
* ``discopt doe extend file.xlsx --n M`` — append M more optimal runs
  using the cumulative FIM as the prior.
* ``discopt doe optimize file.xlsx`` — one active-learning round on an
  optimize-template workbook.
* ``discopt doe gui file.xlsx`` — launch the Streamlit workbook GUI.

Each verb is split into a pure ``do_<verb>(params: dict) -> dict``
function and a ``_cmd_<verb>(args)`` argparse wrapper. The pure
functions raise typed exceptions on failure and are the contract a
future GUI binds to. The wrappers add argparse parsing, ``--json``
output formatting, and exit codes — nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from discopt.doe.templates import (
    TEMPLATE_NAMES,
    build_template,
    template_parameter_names,
)
from discopt.doe.linear_design import design_row
from discopt.doe.workbook import (
    InputSpec,
    ResponseSpec,
    Workbook,
    _load_module_callable,
    module_experiment_with_sigmas,
)

# Ridge added to the prior FIM whenever no fitted prior exists. Keeps
# log-det-FIM finite for the very first batch of a linear-in-parameters
# template (a rank-1 single-design FIM is otherwise singular). It is a
# *fraction of the information already present*, never an absolute floor —
# see _scaled_ridge for why that distinction decides which design you get.
_RIDGE = 1e-6

# Points spanning the factor box, used only to measure how large the
# information in each parameter direction typically is.
_RIDGE_SAMPLES = 5


def _scaled_ridge(reference: Any, n_p: int) -> np.ndarray:
    """Return the ridge to add to a FIM, scaled to the information in it.

    As an *absolute* constant, ``_RIDGE`` does more than keep the log-det
    finite: it silently outvotes the data whenever a parameter's information
    is smaller than 1e-6 in its own units. That is not exotic — an Arrhenius
    activation energy in J/mol has ``∂y/∂Ea ~ 1e-4``, so its information is
    ~1e-8 and the ridge is a hundred times larger than the thing it is
    supposed to nudge. The search then maximizes the ridge's criterion rather
    than the design's: for ``k0·exp(-Ea/RT)`` over T ∈ [300, 500] K at σ = 1,
    a 6-run design collapsed onto a single temperature scored *higher*
    (log-det −14.43) than the correct two-point design (−14.93), and the
    returned design could not identify either parameter.

    Scaling each direction's ridge to the information available in that
    direction keeps the guard and removes the vote: 1e-6 of a quantity is
    negligible next to it whatever its units. Directions with no usable
    reference (zero, NaN, or no reference at all) keep the old absolute value,
    which is the best available answer when there is nothing to scale to.
    """
    ridge = np.full(n_p, _RIDGE, dtype=np.float64)
    if reference is not None:
        diag = np.abs(np.diag(np.asarray(reference, dtype=np.float64)))
        usable = np.isfinite(diag) & (diag > 0.0)
        ridge[usable] = _RIDGE * diag[usable]
    return np.diag(ridge)


def _ridge_sample_designs(
    inputs: Sequence[tuple[str, float, float]], n: int = _RIDGE_SAMPLES
) -> list[dict[str, float]]:
    """Design points spanning the factor box, for scale reference only.

    Walks the box diagonal, so every factor is seen across its whole range.
    These are never returned to the user and need not be feasible under a
    mixture constraint — they exist to answer "how big is the information
    around here?", and :func:`_scaled_ridge` falls back to the absolute ridge
    for any direction where the answer comes out zero.
    """
    return [{name: lo + t * (hi - lo) for name, lo, hi in inputs} for t in np.linspace(0.0, 1.0, n)]


def _basis_reference_fim(
    basis: Any, inputs: Sequence[tuple[str, float, float]], sigma: float
) -> np.ndarray | None:
    """Information a basis carries across the factor box, for ridge scaling.

    Same ``f(x) -> ∂y/∂θ`` contract the design search itself uses, so this
    measures the very quantity the ridge has to stay below. Returns None if the
    basis cannot be evaluated, leaving the absolute ridge in place.
    """
    from discopt.doe.linear_design import linear_fim

    names = [name for name, _, _ in inputs]
    try:
        rows = [
            np.asarray(basis([point[name] for name in names]), dtype=np.float64)
            for point in _ridge_sample_designs(inputs)
        ]
        return linear_fim(np.vstack(rows), sigma)
    except Exception:  # noqa: BLE001 - a scale hint is optional, never fatal
        return None


def _template_reference_fim(
    template: str,
    template_args: dict[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    inputs: Sequence[tuple[str, float, float]],
    sigma: float,
) -> np.ndarray | None:
    """The same measurement as :func:`_basis_reference_fim`, for a template."""
    from discopt.doe.linear_design import design_matrix, linear_fim

    try:
        X = design_matrix(
            template,
            template_args,
            list(parameter_names),
            list(input_names),
            _ridge_sample_designs(inputs),
        )
        return linear_fim(X, sigma)
    except Exception:  # noqa: BLE001 - a scale hint is optional, never fatal
        return None


def _experiment_reference_fim(
    experiment: Any,
    param_values: dict[str, float],
    inputs: Sequence[tuple[str, float, float]],
) -> np.ndarray | None:
    """The same measurement again, through the autodiff FIM.

    A module callable can fail anywhere in the box (or be expensive), so
    anything that goes wrong here simply forfeits the scale hint.
    """
    from discopt.doe.fim import compute_fim

    try:
        total = None
        for design in _ridge_sample_designs(inputs):
            fim = np.asarray(compute_fim(experiment, param_values, design).fim, dtype=np.float64)
            total = fim if total is None else total + fim
        return total
    except Exception:  # noqa: BLE001 - a scale hint is optional, never fatal
        return None


_DEFAULT_CRITERION = "determinant"
# The last two are about *predictions* rather than parameters: the average (I)
# and worst-case (G) variance of the fitted response over the design box. Every
# design path builds that region from the factor bounds when none is given.
_CRITERION_CHOICES = (
    "determinant",
    "trace",
    "min_eigenvalue",
    "condition_number",
    "average_variance",
    "max_variance",
)
_PREDICTION_CRITERIA = ("average_variance", "max_variance")
_CRITERION_ALIASES = {
    "D": "determinant",
    "A": "trace",
    "E": "min_eigenvalue",
    "ME": "condition_number",
    "I": "average_variance",
    "G": "max_variance",
}
# How `extend` spreads a batch over the design space (see
# discopt.doe.batch_optimal_experiment). Only models designed through an
# Experiment have the choice; the basis-driven paths are greedy.
_BATCH_STRATEGIES = ("greedy", "joint", "penalized")


class DoEError(Exception):
    """Raised by ``do_*`` functions on user-facing failures."""


# Mirrors discopt.doe.templates.CLASSICAL_TEMPLATES. Duplicated as a literal so
# the hot paths that only need the membership test don't import the templates
# module (which reaches discopt.modeling) just to branch on a name.
_CLASSICAL_TEMPLATE_NAMES = frozenset(
    {"latin-hypercube", "central-composite", "box-behnken", "definitive-screening"}
)
# Mirrors discopt.doe.templates.TWO_LEVEL_TEMPLATES, for the same reason.
_TWO_LEVEL_TEMPLATE_NAMES = frozenset(
    {"factorial-2level", "fractional-factorial", "plackett-burman"}
)


# ──────────────────────────────────────────────────────────────────
# argparse parsing helpers
# ──────────────────────────────────────────────────────────────────


def _parse_input_spec(s: str) -> tuple[str, float, float]:
    """Parse ``name:lb:ub`` (used by ``--input`` / ``--bounds``)."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'name:lb:ub', got {s!r}")
    name, lb_s, ub_s = parts
    if not name:
        raise argparse.ArgumentTypeError(f"input name is empty in {s!r}")
    try:
        lb = float(lb_s)
        ub = float(ub_s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad bound in {s!r}: {e}") from e
    if not (ub > lb):
        raise argparse.ArgumentTypeError(f"input {name!r}: upper bound must exceed lower bound")
    return (name, lb, ub)


def _parse_levels_spec(s: str) -> tuple[str, list[object]]:
    """Parse ``name:L1,L2,L3`` for Latin-family --levels arguments.

    Levels are interpreted as floats when all parts parse as numeric,
    otherwise as strings.
    """
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"expected 'name:L1,L2,...', got {s!r}")
    name, _, levels_s = s.partition(":")
    if not name:
        raise argparse.ArgumentTypeError(f"factor name is empty in {s!r}")
    raw = [p.strip() for p in levels_s.split(",") if p.strip()]
    if len(raw) < 2:
        raise argparse.ArgumentTypeError(f"factor {name!r}: need at least 2 levels")
    try:
        levels: list[object] = [float(p) for p in raw]
    except ValueError:
        levels = list(raw)
    return (name, levels)


def _parse_factor_pair(s: str) -> tuple[str, object, object]:
    """Parse ``name:LOW:HIGH`` where LOW/HIGH may be numeric or string."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'name:LOW:HIGH', got {s!r}")
    name, lo_s, hi_s = parts
    if not name:
        raise argparse.ArgumentTypeError(f"factor name is empty in {s!r}")

    def _maybe_num(v: str) -> object:
        try:
            return float(v)
        except ValueError:
            return v

    lo, hi = _maybe_num(lo_s), _maybe_num(hi_s)
    if lo == hi:
        raise argparse.ArgumentTypeError(f"factor {name!r}: LOW and HIGH must differ")
    return (name, lo, hi)


def _parse_kv_float(s: str) -> tuple[str, float]:
    """Parse ``name=value`` (used by ``--params``)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected 'name=value', got {s!r}")
    name, _, val = s.partition("=")
    try:
        return (name, float(val))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad numeric value in {s!r}: {e}") from e


def _normalize_criterion(c: str) -> str:
    return _CRITERION_ALIASES.get(c, c)


# ──────────────────────────────────────────────────────────────────
# Design targets (--target, --target-expr, --target-response, --tolerance)
# ──────────────────────────────────────────────────────────────────


def _parse_target(s: str) -> tuple[str, float]:
    """Parse ``NAME`` or ``NAME=WEIGHT`` (``--target``)."""
    name, eq, val = s.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"expected 'NAME' or 'NAME=WEIGHT', got {s!r}")
    if not eq:
        return (name, 1.0)
    try:
        return (name, float(val))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad weight in {s!r}: {e}") from e


def _parse_target_expr(s: str) -> tuple[str, str]:
    """Parse ``NAME=FORMULA`` (``--target-expr``), e.g. ``ratio=k1/k2``."""
    name, eq, expr = s.partition("=")
    if not eq or not name.strip() or not expr.strip():
        raise argparse.ArgumentTypeError(f"expected 'NAME=FORMULA', got {s!r}")
    return (name.strip(), expr.strip())


def _parse_target_response(s: str) -> tuple[str, dict[str, float] | None]:
    """Parse ``RESPONSE`` or ``RESPONSE@INPUT=V[,INPUT=V...]`` (``--target-response``)."""
    name, at, rest = s.partition("@")
    name = name.strip()
    if not name or (at and not rest.strip()):
        raise argparse.ArgumentTypeError(
            f"expected 'RESPONSE' or 'RESPONSE@INPUT=VALUE[,INPUT=VALUE]', got {s!r}"
        )
    if not at:
        return (name, None)
    point: dict[str, float] = {}
    for item in rest.split(","):
        k, eq, v = item.partition("=")
        try:
            if not eq or not k.strip():
                raise ValueError("missing '='")
            point[k.strip()] = float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad operating point {item!r} in {s!r}") from e
    return (name, point)


def _target_spec_from_args(args) -> dict[str, Any] | None:
    """The target spec the command-line flags describe; ``None`` without any."""
    params = dict(getattr(args, "target", None) or [])
    exprs = dict(getattr(args, "target_expr", None) or [])
    responses: dict[str, list[dict[str, float]]] = {}
    for name, point in getattr(args, "target_response", None) or []:
        responses.setdefault(name, [])
        if point is not None:
            responses[name].append(point)
    tolerances = dict(getattr(args, "tolerance", None) or [])
    relative = bool(getattr(args, "relative_precision", False))
    if not (params or exprs or responses):
        if tolerances or relative:
            raise DoEError(
                "--tolerance / --relative-precision qualify a target; name one with "
                "--target, --target-expr or --target-response"
            )
        return None
    spec: dict[str, Any] = {"relative": relative}
    if params:
        spec["parameters"] = params
    if exprs:
        spec["expressions"] = exprs
    if responses:
        spec["responses"] = responses
    if tolerances:
        spec["tolerances"] = tolerances
    return spec


def target_spec_from_text(
    targets: str | Sequence[str] | None,
    tolerances: str | Sequence[str] | None = None,
    relative: bool = False,
) -> dict[str, Any] | None:
    """A target spec from free text, for the GUI and the browser app.

    ``targets`` holds items separated by semicolons or new lines (commas belong
    to operating points and formulas):

    * ``k1`` or ``k1=2`` -- a parameter, optionally weighted;
    * ``ratio=k1/k2`` -- a formula (anything after ``=`` that is not a number);
    * ``y@T=365,t=4`` -- a prediction at an operating point.

    ``tolerances`` is ``name=tau`` items separated by commas, semicolons or new
    lines. Returns ``None`` when no target is named.
    """
    import re

    def items(text: str | Sequence[str] | None, seps: str) -> list[str]:
        if text is None:
            return []
        parts = [text] if isinstance(text, str) else list(text)
        out: list[str] = []
        for part in parts:
            out += [x.strip() for x in re.split(seps, str(part)) if x.strip()]
        return out

    params: dict[str, float] = {}
    exprs: dict[str, str] = {}
    responses: dict[str, list[dict[str, float]]] = {}
    try:
        for item in items(targets, r"[;\n]"):
            if "@" in item:
                name, point = _parse_target_response(item)
                responses.setdefault(name, [])
                if point is not None:
                    responses[name].append(point)
                continue
            name, eq, rhs = item.partition("=")
            name, rhs = name.strip(), rhs.strip()
            if not eq:
                params[name] = 1.0
                continue
            try:
                params[name] = float(rhs)
            except ValueError:
                exprs[name] = rhs
        tol = dict(_parse_kv_float(x) for x in items(tolerances, r"[;,\n]"))
    except argparse.ArgumentTypeError as e:
        raise DoEError(f"bad design target: {e}") from e
    if not (params or exprs or responses):
        if tol or relative:
            raise DoEError("tolerances and relative precision qualify a target; name one")
        return None
    spec: dict[str, Any] = {"relative": bool(relative)}
    if params:
        spec["parameters"] = params
    if exprs:
        spec["expressions"] = exprs
    if responses:
        spec["responses"] = responses
    if tol:
        spec["tolerances"] = tol
    return spec


def _check_target_spec(
    spec: Mapping[str, Any],
    parameter_names: Sequence[str],
    response_names: Sequence[str],
    input_names: Sequence[str],
) -> dict[str, Any]:
    """Validate a target spec against the campaign's model; a clean copy or DoEError."""
    from discopt.doe.targets import validate_target_spec

    try:
        spec = validate_target_spec(spec)
    except ValueError as e:
        raise DoEError(f"bad design target: {e}") from e
    unknown = [p for p in spec.get("parameters", {}) if p not in parameter_names]
    if unknown:
        raise DoEError(
            f"--target names unknown parameter(s) {unknown}; the model's parameters are "
            f"{list(parameter_names)}"
        )
    for r, points in spec.get("responses", {}).items():
        if r not in response_names:
            raise DoEError(
                f"--target-response names unknown response {r!r}; the model's responses are "
                f"{list(response_names)}"
            )
        if input_names and not points:
            raise DoEError(
                f"--target-response {r} needs an operating point: write "
                f"'{r}@" + ",".join(f"{n}=VALUE" for n in input_names) + "'"
            )
        for pt in points:
            missing = [n for n in input_names if n not in pt]
            extra = [n for n in pt if n not in input_names]
            if missing or extra:
                raise DoEError(
                    f"--target-response {r}@...: the operating point must give exactly the "
                    f"design inputs {list(input_names)} (missing {missing}, unknown {extra})"
                )
    return spec


def _target_from_spec(spec: Mapping[str, Any]) -> Any:
    """``Target.from_spec`` with its errors as DoEError (e.g. kinds mixed without tolerances)."""
    from discopt.doe.targets import Target

    try:
        return Target.from_spec(spec)
    except ValueError as e:
        raise DoEError(f"bad design target: {e}") from e


def _target_precision_without_ridge(
    target: Any, fim: np.ndarray | None, ridge: np.ndarray | None
) -> Any:
    """Target precision on the information the runs carry, not the search's ridge.

    The design searches add a tiny ridge to keep early FIMs invertible; it must
    not count as data, or a target that the runs cannot identify reports a
    finite (ridge-sized) standard error instead of "not estimable".
    """
    if target is None or fim is None:
        return None
    info = np.asarray(fim, dtype=float)
    if ridge is not None:
        info = info - np.asarray(ridge, dtype=float)
    return target.precision(info)


def _precision_out(prec: Any) -> dict[str, Any]:
    """A TargetPrecision as JSON-able output."""
    return {
        "estimable": bool(prec.estimable),
        "std_errors": prec.std,
        "ratios": prec.ratios,
        "met": prec.met,
    }


def _print_target_precision(out: Mapping[str, Any], label: str) -> None:
    prec = out.get("target_precision")
    if not prec:
        return
    note = "" if prec["estimable"] else "  (not estimable yet)"
    print(f"  {label}{note}")
    for name, se in prec["std_errors"].items():
        ratio = prec["ratios"].get(name)
        extra = f"   SE/tolerance = {ratio:.3g}" if ratio is not None else ""
        print(f"    {name:>12s}  SE = {se:.4g}{extra}")
    if prec["met"] is not None:
        print(f"    every target within tolerance: {'yes' if prec['met'] else 'no'}")


def _no_target_for(what: str) -> DoEError:
    return DoEError(
        f"design targets (--target, --target-expr, --target-response) need a model the "
        f"design is optimized for: the parametric templates, --module, symbolic, or "
        f"`extend` of a classical design. {what}"
    )


def _basis_target_or_error(
    spec: Mapping[str, Any],
    parameter_names: Sequence[str],
    input_names: Sequence[str],
    basis: Any,
    response_name: str,
    param_values: Mapping[str, float] | None = None,
) -> Any:
    """A target on the jax-free basis path; errors as DoEError."""
    from discopt.doe.targets import basis_target

    try:
        return basis_target(
            spec, parameter_names, input_names, basis, param_values, response_name=response_name
        )
    except ValueError as e:
        raise DoEError(f"bad design target: {e}") from e


# ──────────────────────────────────────────────────────────────────
# Typed parameter / result records
# ──────────────────────────────────────────────────────────────────


@dataclass
class NewParams:
    output: Path
    n: int
    inputs: list[tuple[str, float, float]]
    response_name: str
    measurement_error: float
    criterion: str
    seed: int
    n_starts: int
    template: str | None = None
    degree: int | None = None
    mixture_total: float | None = None
    module_callable: str | None = None
    param_initial_guess: dict[str, float] = field(default_factory=dict)
    levels: dict[str, list[object]] | None = None
    replicates: int = 1
    factor_pairs: dict[str, tuple[object, object]] | None = None
    center_points: int = 0
    # Classical design template options (latin-hypercube / central-composite /
    # box-behnken). `basis` is the regression model the design is meant to be
    # analysed with, and travels in the workbook metadata for `fit`.
    basis: str = "quadratic"
    alpha: str = "rotatable"
    # Screening-design options: fractional-factorial generators / run count /
    # resolution, Plackett-Burman run count (both via `runs`), definitive
    # screening fake factors, and the Latin-hypercube optimization criterion.
    generators: list[str] | None = None
    runs: int | None = None
    resolution: int = 3
    fake_factors: int = 0
    lhs_optimize: str = "discrepancy"
    within_bounds: bool = True
    # Build parametric designs from the closed-form linear FIM instead of jax
    # autodiff. Identical results for every built-in template; exists for
    # environments with no jax (Pyodide/WASM), where the default path cannot run.
    use_linear_design: bool = False
    # The response expression for the `symbolic` template. Parameter names and
    # their nominal values come from `param_initial_guess`.
    expression: str | None = None
    # Active-learning ("optimize") template options
    optimize_criterion: str = "maximize"
    optimize_surrogate: str = "gp"
    optimize_acquisition: str = "expected_improvement"
    # Overwrite an existing output workbook (the CLI maps --force here).
    force: bool = False
    # Extra bookkeeping columns on the runs sheet, beyond the design inputs:
    # somewhere to write the operator, the lot number, the instrument. They
    # travel with the runs and `anova_report` can pick them up as blocking
    # factors. "replicate" is added automatically by the designs that need it.
    extra_columns: list[str] = field(default_factory=list)
    # `--module` only: per-response measurement-error overrides. A module
    # experiment takes its response names and errors from the ExperimentModel
    # it returns (response_name / measurement_error above are ignored); these
    # replace the model's sigma for the named responses, in the design and in
    # every later fit/extend.
    sigma_overrides: dict[str, float] = field(default_factory=dict)
    # Design target (a Target spec, see discopt.doe.Target.from_spec): design
    # for the precision of these parameters / formulas / predictions instead
    # of all parameters. Stored in the workbook so `extend` reuses it.
    target: dict[str, Any] | None = None


@dataclass
class OptimizeParams:
    """Inputs to :func:`do_optimize`.

    ``criterion``, ``surrogate`` and ``acquisition`` default to ``None``,
    meaning "use whatever the workbook was created with" (the values stored
    in ``template_args`` by ``discopt doe new optimize``). Pass an explicit
    value to override the stored setting for this round.
    """

    workbook: Path
    criterion: str | None = None
    surrogate: str | None = None
    acquisition: str | None = None
    batch_size: int = 4
    n_candidates: int = 2048
    seed: int | None = None
    acquisition_kwargs: dict[str, float] | None = None
    custom_surrogate_path: str | None = None
    custom_surrogate_kwargs: dict[str, Any] | None = None


@dataclass
class ExtendParams:
    workbook: Path
    n: int
    n_starts: int
    # "greedy" | "joint" | "penalized"; see batch_optimal_experiment.
    batch_strategy: str = "greedy"
    # A new design target replacing the stored one (None: keep the stored
    # target, if any); clear_target drops it and designs for all parameters.
    target: dict[str, Any] | None = None
    clear_target: bool = False


@dataclass
class FoldoverParams:
    """Inputs to :func:`do_foldover`.

    ``factor=None`` reverses every factor (the full fold-over); a factor name
    reverses only that one.
    """

    workbook: Path
    factor: str | None = None


# ──────────────────────────────────────────────────────────────────
# Internal: build Experiment + parameter-name list from spec
# ──────────────────────────────────────────────────────────────────


def _build_experiment_from_new(params: NewParams):
    """Return ``(experiment, parameter_names)`` for a ``do_new`` request."""
    if params.template is not None and params.module_callable is not None:
        raise DoEError("specify either --template or --module, not both")
    if params.template is None and params.module_callable is None:
        raise DoEError("specify a template name or --module")

    if params.template is not None:
        if params.template not in TEMPLATE_NAMES:
            raise DoEError(
                f"unknown template {params.template!r}; choose from {list(TEMPLATE_NAMES)}"
            )
        exp = build_template(
            params.template,
            inputs=params.inputs,
            response_name=params.response_name,
            measurement_error=params.measurement_error,
            degree=params.degree,
            mixture_total=params.mixture_total,
        )
        names = template_parameter_names(
            params.template, degree=params.degree, n_inputs=len(params.inputs)
        )
        return exp, names

    exp, _specs = _module_experiment_from_new(params)
    em = exp.create_model(**params.param_initial_guess)
    return exp, em.parameter_names


def _module_experiment_from_new(params: NewParams) -> tuple[Any, list[ResponseSpec]]:
    """Load a ``--module`` experiment with any ``--sigma`` overrides applied.

    Returns the experiment to design with and the response columns it implies:
    one per response of the model, with the model's own measurement error unless
    ``sigma_overrides`` names it.
    """
    # The working directory (searched by default) plus the output's directory,
    # which is where every later command will look for the model (issue #48).
    out_dir = Path(params.output).resolve().parent
    exp = _load_module_callable(params.module_callable, search_dirs=[out_dir])  # type: ignore[arg-type]
    em = exp.create_model(**params.param_initial_guess)
    names = list(em.response_names)
    overrides = dict(params.sigma_overrides or {})
    unknown = [k for k in overrides if k not in names]
    if unknown:
        raise DoEError(f"--sigma names {unknown} that are not responses of the model ({names})")
    sigmas = {n: float(overrides.get(n, em.measurement_error[n])) for n in names}
    try:
        exp = module_experiment_with_sigmas(exp, sigmas, params.param_initial_guess)
    except ValueError as e:
        raise DoEError(str(e)) from e
    missing_inputs = [n for n in em.design_inputs if n not in {s[0] for s in params.inputs}]
    if missing_inputs:
        raise DoEError(
            f"the model's design input(s) {missing_inputs} have no --bounds; give "
            "NAME:LB:UB for every design input"
        )
    return exp, [ResponseSpec(n, sigmas[n]) for n in names]


def _mixture_constraints(
    template: str | None,
    inputs: list[tuple[str, float, float]],
    mixture_total: float | None,
):
    """Return ``(equality_constraints, feasible_projection)`` for Scheffé
    templates, or ``(None, None)`` for any other template.

    Mixture designs require an equality constraint ``sum(components) ==
    total`` plus a Dirichlet-style projection so the multi-start lands
    on the simplex from the start.
    """
    if not template or not template.startswith("scheffe-"):
        return None, None
    # From discopt.doe.simplex, not discopt.doe: the package-level name resolves
    # through discopt.doe.design, which needs the FIM machinery and the base
    # package. Every mixture design in the browser died on that import.
    from discopt.doe.simplex import project_to_simplex, sum_constraint

    from discopt.doe.mixture import check_mixture_bounds

    components = [name for name, _, _ in inputs]
    bounds = {name: (lb, ub) for name, lb, ub in inputs}
    total = float(mixture_total) if mixture_total is not None else 1.0
    # Reject impossible component bounds before any search runs. They used to
    # be clipped away silently, which returned a design that broke the very
    # bounds it was given.
    check = check_mixture_bounds(bounds, total)
    if not check.feasible:
        raise DoEError(
            f"{check.message} Component bounds are in the same units as the "
            f"mixture total ({total:g}); widen them or change --mixture-total."
        )
    g = sum_constraint(components, total=total)

    def proj(point: dict[str, float]) -> dict[str, float]:
        return project_to_simplex(point, components, total=total, bounds=bounds)

    return [g], proj


def _design_bounds(specs: list[tuple[str, float, float]]) -> dict[str, tuple[float, float]]:
    return {name: (lb, ub) for name, lb, ub in specs}


def _param_values_for_design(
    parameter_names: list[str],
    workbook: Workbook | None,
) -> dict[str, float]:
    """Pick the parameter point at which to evaluate FIMs.

    Order of precedence: fitted parameters from the workbook
    (if available), then ``param_initial_guess`` metadata, then zeros.
    """
    if workbook is not None:
        fitted = workbook.read_parameters()
        if fitted:
            out: dict[str, float] = {}
            for row in fitted:
                if row["estimate"] is not None:
                    out[row["name"]] = float(row["estimate"])
            if out:
                for name in parameter_names:
                    out.setdefault(name, 0.0)
                return out
        guesses = workbook.param_initial_guess()
        if guesses:
            out = {name: float(guesses.get(name, 0.0)) for name in parameter_names}
            return out
    return {name: 0.0 for name in parameter_names}


def _cumulative_fim_from_completed(
    experiment: Any,
    parameter_names: list[str],
    completed_runs: list[dict[str, Any]],
    input_names: list[str],
    param_values: dict[str, float] | None = None,
    *,
    ridge: bool = True,
) -> np.ndarray:
    """Sum FIMs over every completed run. Adds a ridge so first-batch
    FIMs stay non-singular when fewer runs than parameters have come in.

    ``param_values`` is the point at which each per-run FIM is evaluated; for a
    nonlinear model it must be the fitted/guessed parameters (the FIM at θ=0 is
    meaningless there). Rows whose FIM cannot be evaluated are skipped with a
    warning rather than dropped silently.
    """
    from discopt.doe.fim import compute_fim

    n_p = len(parameter_names)
    fim = np.zeros((n_p, n_p), dtype=np.float64)
    if param_values is None:
        param_values = {name: 0.0 for name in parameter_names}
    for row in completed_runs:
        design = {nm: float(row[nm]) for nm in input_names}
        try:
            r = compute_fim(experiment, param_values, design)
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: skipping run {row.get('run_id')} in prior-FIM sum ({e})",
                file=sys.stderr,
            )
            continue
        fim = fim + np.asarray(r.fim)
    # Ridge last, scaled to what the completed runs actually carry — the sum
    # is its own best scale reference.
    return fim + _scaled_ridge(fim, n_p) if ridge else fim


def _module_campaign_runs(
    wb: Workbook, completed: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Completed rows as ``campaign_experiment`` runs, plus the count of blank cells.

    Each run carries its ``run_id``, every design input, and every response;
    a blank response is ``None``, which ``data_from_runs`` drops from the
    likelihood (a missing value, not a zero).
    """
    input_names = [s.name for s in wb.input_specs()]
    response_names = wb.response_names()
    runs: list[dict[str, Any]] = []
    n_missing = 0
    for row in completed:
        run: dict[str, Any] = {"run_id": row.get("run_id")}
        for nm in input_names:
            v = row.get(nm)
            if v is None or (isinstance(v, str) and not v.strip()):
                raise DoEError(
                    f"run {row.get('run_id')} has a blank value for input {nm!r}; "
                    "fill it in (or clear its responses to mark the run pending)."
                )
            run[nm] = float(v)
        for rn in response_names:
            value = _coerce_response(row.get(rn), row.get("run_id"), rn)
            n_missing += value is None
            run[rn] = value
        runs.append(run)
    return runs, n_missing


def _coerce_response(v: Any, run_id: Any, name: str) -> float | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise DoEError(f"run {run_id}: response {name!r} is {v!r}, not a number") from e


def _measured_fim(
    experiment: Any,
    runs: list[dict[str, Any]],
    param_values: dict[str, float],
) -> np.ndarray:
    """Information in the responses actually measured in ``runs``, at ``param_values``.

    Summing ``compute_fim`` over runs counts every response of every run; with
    blank cells that overstates what is known. The campaign holds every run at
    its own conditions, and its weighted Jacobian over just the measured
    entries gives the information they carry.
    """
    from discopt.doe.runs import _weighted_fit, campaign_experiment

    camp = campaign_experiment(experiment, runs)
    data = camp.data_from_runs(runs)
    fixed = {n: float(param_values.get(n, 0.0)) for n in camp.parameter_names}
    return np.asarray(_weighted_fit(camp, data, initial_guess=None, fixed_parameters=fixed)["fim"])


# ──────────────────────────────────────────────────────────────────
# do_templates
# ──────────────────────────────────────────────────────────────────


_TEMPLATE_DESCRIPTIONS = {
    "linear": (
        "y = b0 + sum_i bi * xi.  Inputs: --input NAME:LB:UB (repeatable). "
        "Parameters: 1 + n_inputs."
    ),
    "polynomial-1d": (
        "y = sum_{j=0..d} bj * x**j.  Inputs: --input NAME:LB:UB (single), "
        "--degree D.  Parameters: D + 1."
    ),
    "response-surface-2d": (
        "Full quadratic in 2 factors (intercept, 2 main, 2 square, 1 cross). "
        "Inputs: --input NAME:LB:UB (exactly 2).  Parameters: 6."
    ),
    "response-surface-3d": (
        "Full quadratic in 3 factors (intercept, 3 main, 3 square, 3 cross). "
        "Inputs: --input NAME:LB:UB (exactly 3).  Parameters: 10."
    ),
    "scheffe-linear": (
        "Scheffé canonical linear mixture: y = sum_i bi * xi, with "
        "sum(components) == --mixture-total.  Inputs: --input NAME:LB:UB "
        "(>= 2 components), --mixture-total T (default 1.0).  Parameters: q."
    ),
    "scheffe-quadratic": (
        "Scheffé canonical quadratic mixture: pure-blend + pairwise "
        "blending terms.  Inputs: --input NAME:LB:UB (>= 2), "
        "--mixture-total T.  Parameters: q + q(q-1)/2."
    ),
    "scheffe-special-cubic": (
        "Scheffé canonical special-cubic mixture: adds three-way blending "
        "to the quadratic.  Inputs: --input NAME:LB:UB (>= 3), "
        "--mixture-total T.  Parameters: q + q(q-1)/2 + q(q-1)(q-2)/6."
    ),
    "latin-square": (
        "Latin-square design for ANOVA on 3 factors (1 treatment + 2 blocks). "
        "Inputs: --levels NAME:L1,L2,... (exactly 3 factors, same level count). "
        "Optional: --replicates R."
    ),
    "graeco-latin": (
        "Graeco-Latin square for ANOVA on 4 factors. Inputs: --levels "
        "NAME:L1,L2,... (exactly 4 factors, k>=3 and k!=6). "
        "Optional: --replicates R."
    ),
    "hyper-graeco-latin": (
        "Hyper-Graeco-Latin square for ANOVA on 5 factors (3 MOLS). Inputs: "
        "--levels NAME:L1,L2,... (exactly 5 factors, k in {4,5,7,...}). "
        "Optional: --replicates R."
    ),
    "factorial-2level": (
        "2-level full factorial for screening. Inputs: --factor "
        "NAME:LOW:HIGH (repeatable, 2-8 factors; LOW/HIGH may be "
        "numeric or string). Optional: --center-points N (repeated at each "
        "level of any string factor), --replicates R. Use this to answer 'does each "
        "factor matter?' before fitting a response surface."
    ),
    "fractional-factorial": (
        "2-level fractional factorial for screening many factors in few runs. "
        "Inputs: --factor NAME:LOW:HIGH (repeatable, 3-15 factors) and either "
        "--generator 'D=ABC' (repeatable; written in the factor names -- "
        "single-letter names concatenate, longer ones join with '*', e.g. "
        "'cat=temp*time') or --runs N [--resolution R] to search for a "
        "fraction (the search needs the base discopt solver). Optional: "
        "--center-points N, --replicates R. Check what is aliased with what "
        "using discopt.doe.alias_structure before you run it."
    ),
    "plackett-burman": (
        "Plackett-Burman screening design: main effects of up to N-1 two-level "
        "factors in N runs (N a multiple of 4). Inputs: --factor NAME:LOW:HIGH "
        "(repeatable, 2-23 factors). Optional: --runs N (default: the smallest "
        "multiple of 4 above the factor count), --center-points N, --replicates R. "
        "Main effects are partially aliased with two-factor interactions."
    ),
    "definitive-screening": (
        "Definitive screening design (Jones & Nachtsheim 2011): three levels per "
        "factor in 2m+1 runs; main effects are clear of every two-factor "
        "interaction and quadratic effect. Inputs: --input NAME:LB:UB (3-12 "
        "factors). Optional: --fake-factors F, --center-points N (default 1), "
        "--basis (default linear). Use `fit` on the completed runs."
    ),
    "latin-hypercube": (
        "Latin hypercube: space-filling stratified sample over a continuous "
        "box, with the run count set by you rather than by the factor count. "
        "Inputs: --input NAME:LB:UB (repeatable), --n RUNS. Optional: "
        "--basis {linear,quadratic} (default linear here; the model `fit` "
        "will estimate), --optimize {discrepancy,maximin,none} (default "
        "discrepancy). Use this for surrogate fitting or when you want "
        "coverage without assuming a model form."
    ),
    "central-composite": (
        "Central composite design for a full quadratic response surface: "
        "2**k factorial core + 2k axial points + centre replicates. Inputs: "
        "--input NAME:LB:UB (2-6 factors). Optional: --center-points N "
        "(default 4), --alpha {rotatable,face} or a number, --outside-bounds "
        "to use the textbook scaling where the axial points fall beyond "
        "LB/UB. By default every run stays within the bounds you give."
    ),
    "box-behnken": (
        "Box-Behnken design for a full quadratic response surface without "
        "visiting any corner of the design box — every run sits at an edge "
        "midpoint, so no run combines the extreme level of every factor. "
        "Inputs: --input NAME:LB:UB (3-5 factors). Optional: "
        "--center-points N (default 3)."
    ),
    "symbolic": (
        "Your own model, differentiated symbolically. Inputs: --expr "
        "'EXPRESSION' (e.g. 'k0 * exp(-Ea / (8.314 * T))'), --param NAME=VALUE "
        "(repeatable; the nominal guess a locally-optimal design is built "
        "around), --input NAME:LB:UB (repeatable), --n RUNS. Optional: "
        "--criterion, --n-starts. `fit` estimates the parameters by nonlinear "
        "least squares and `extend` designs the next batch around them. "
        "Allowed in the expression: + - * / **, numbers, your declared names, "
        "pi/E, and exp log ln log10 sqrt sin cos tan asin acos atan sinh cosh "
        "tanh abs erf."
    ),
    "optimize": (
        "Active-learning workbook for sequential optimization. Inputs: "
        "--input NAME:LB:UB (repeatable). The seed batch is sampled by "
        "Sobol; subsequent rounds are produced by 'discopt doe optimize "
        "WORKBOOK' which fits a surrogate and proposes new runs via the "
        "chosen acquisition. Optional: --optimize-criterion "
        "{maximize,minimize}, --optimize-surrogate (preset name), "
        "--optimize-acquisition."
    ),
}


def do_templates(_params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "templates": [
            {"name": name, "description": _TEMPLATE_DESCRIPTIONS[name]} for name in TEMPLATE_NAMES
        ]
    }


def _cmd_templates(args) -> int:
    out = do_templates()
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        for entry in out["templates"]:
            print(f"{entry['name']}")
            print(f"    {entry['description']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# do_new
# ──────────────────────────────────────────────────────────────────


def _validate_new_column_names(params: NewParams) -> None:
    """Reject duplicate factor/input names or collisions with reserved columns.

    Runs before any design work so all templates fail with the same clear
    message instead of an opaque downstream error (or silent corruption).
    """
    from discopt.doe.templates import COMBINATORIAL_TEMPLATES

    if params.template in _TWO_LEVEL_TEMPLATE_NAMES:
        names = list((params.factor_pairs or {}).keys())
    elif params.template in COMBINATORIAL_TEMPLATES:
        names = list((params.levels or {}).keys())
    else:
        names = [s[0] for s in params.inputs]

    # A module experiment's response columns come from its model, so they are
    # checked against the inputs when the model is loaded (Workbook.create).
    reserved = {"run_id", "batch", "measured_at", "replicate"}
    if not params.module_callable:
        reserved.add(params.response_name)
    seen: set[str] = set()
    for name in names:
        if name in reserved:
            raise DoEError(
                f"column name {name!r} is reserved (response is "
                f"{params.response_name!r}); rename the input/factor."
            )
        if name in seen:
            raise DoEError(f"duplicate factor/input name {name!r}; names must be unique.")
        seen.add(name)


def _extra_columns(params: Any) -> list[str]:
    """Validated ``extra_columns`` from the params, if it carries any."""
    names = [str(n).strip() for n in getattr(params, "extra_columns", None) or []]
    bad = [n for n in names if not n or n in ("run_id", "batch", "run_order", "measured_at")]
    if bad:
        raise DoEError(
            f"extra column name(s) {bad} clash with the workbook's own columns "
            "(run_id, batch, run_order, measured_at) or are blank"
        )
    return names


def do_new(params: NewParams) -> dict[str, Any]:
    # batch_optimal_experiment is imported at its use site below, not here: it
    # reaches jax through discopt.doe.fim, and the combinatorial, classical, and
    # use_linear_design branches all return before needing it. Hoisting it would
    # make every template require jax.
    from discopt.doe.templates import CLASSICAL_TEMPLATES, COMBINATORIAL_TEMPLATES

    # Guard here too (not only in _cmd_new) so direct callers such as the GUI
    # don't silently clobber an existing campaign.
    if Path(params.output).exists() and not params.force:
        raise DoEError(f"{params.output} already exists; pass force=True to overwrite.")

    _validate_new_column_names(params)

    if params.mixture_total is not None and float(params.mixture_total) <= 0.0:
        raise DoEError(
            f"mixture-total must be positive (got {params.mixture_total}); it is the "
            "required sum of the component values."
        )

    if params.target and (
        params.template in _TWO_LEVEL_TEMPLATE_NAMES
        or params.template == "optimize"
        or params.template in COMBINATORIAL_TEMPLATES
        or params.template in CLASSICAL_TEMPLATES
    ):
        raise _no_target_for(
            f"Template {params.template!r} is a fixed or model-free design (to target "
            "coefficients of a classical design, `extend` it with --target)."
        )

    if params.template in _TWO_LEVEL_TEMPLATE_NAMES:
        return _do_new_factorial(params)
    if params.template == "optimize":
        return _do_new_optimize(params)
    if params.template in COMBINATORIAL_TEMPLATES:
        return _do_new_latin(params)
    if params.template in CLASSICAL_TEMPLATES:
        return _do_new_classical(params)
    if params.template == "symbolic":
        return _do_new_symbolic(params)

    if params.use_linear_design:
        return _do_new_linear(params)

    response_specs: list[ResponseSpec] | None = None
    if params.module_callable and params.template is None:
        experiment, response_specs = _module_experiment_from_new(params)
        parameter_names = experiment.create_model(**params.param_initial_guess).parameter_names
    else:
        experiment, parameter_names = _build_experiment_from_new(params)
    bounds = _design_bounds(params.inputs)
    input_names = [s[0] for s in params.inputs]
    _check_prediction_criterion(params.criterion, params.template)
    target_spec, target = None, None
    if params.target:
        target_spec = _check_target_spec(
            params.target,
            parameter_names,
            [r.name for r in response_specs] if response_specs else [params.response_name],
            input_names,
        )
        target = _target_from_spec(target_spec)

    n_p = len(parameter_names)

    # For module-callable (likely nonlinear) experiments, use any
    # supplied parameter guesses; for templates, zeros are fine (FIM
    # independent of params in linear-in-parameters models).
    param_values = (
        {name: float(params.param_initial_guess.get(name, 0.0)) for name in parameter_names}
        if params.module_callable
        else {name: 0.0 for name in parameter_names}
    )

    prior_fim = _scaled_ridge(
        _experiment_reference_fim(experiment, param_values, params.inputs), n_p
    )

    eq_cons, proj = _mixture_constraints(params.template, params.inputs, params.mixture_total)

    if params.n == 1:
        from discopt.doe import optimal_experiment

        single = optimal_experiment(
            experiment,
            param_values,
            bounds,
            criterion=params.criterion,
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            target=target,
            seed=params.seed,
        )
        designs = [single.design]
        criterion_value = float(single.criterion_value)
        precision = _target_precision_without_ridge(single.target, single.fim, prior_fim)
    else:
        from discopt.doe import batch_optimal_experiment

        batch = batch_optimal_experiment(
            experiment,
            param_values,
            bounds,
            n_experiments=params.n,
            criterion=params.criterion,
            # A target the first runs cannot identify on their own (fewer runs
            # than parameters) leaves the greedy picks arbitrary, and exchange
            # cannot restore the joint structure it needs (e.g. two runs placed
            # symmetrically). The joint search starts from the greedy batch,
            # so it is never worse.
            strategy="joint" if target is not None else "greedy",
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=params.seed,
            target=target,
        )
        designs = list(batch.designs)
        criterion_value = float(batch.criterion_value)
        precision = _target_precision_without_ridge(batch.target, batch.joint_fim, prior_fim)

    template_args: dict[str, Any] = {}
    if params.degree is not None:
        template_args["degree"] = int(params.degree)
    if params.mixture_total is not None:
        template_args["mixture_total"] = float(params.mixture_total)

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion=params.criterion,
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=params.module_callable,
        param_initial_guess=params.param_initial_guess or None,
        extra_columns=_extra_columns(params),
        responses=response_specs,
    )
    if target_spec:
        wb.set_target_spec(target_spec)
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": params.template,
            "module_callable": params.module_callable,
            "n": params.n,
            "criterion": params.criterion,
            "target": target_spec,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": params.template,
        "module_callable": params.module_callable,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": params.criterion,
        "criterion_value": criterion_value,
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
        "response_names": wb.response_names(),
        "target": target_spec,
        "target_precision": _precision_out(precision) if precision is not None else None,
        "next_command": f"discopt doe status {wb.path}",
    }


def _check_prediction_criterion(criterion: str, template: str | None) -> None:
    """Reject I/G on a mixture template designed through an Experiment.

    That path samples the prediction region from the factor box, which for a
    mixture is mostly off the simplex the runs live on, so the criterion would
    average over blends that cannot be made.
    """
    if criterion in _PREDICTION_CRITERIA and template and template.startswith("scheffe-"):
        raise DoEError(
            f"criterion {criterion!r} needs a prediction region on the mixture simplex; "
            "use determinant/trace/min_eigenvalue/condition_number for a Scheffé "
            "template, or build the design in Python with prediction_points="
        )


def _do_new_factorial(params: NewParams) -> dict[str, Any]:
    """Build a 2-level screening design and persist it as a workbook.

    Handles the full factorial, fractional factorials (from generators or the
    MILP fraction search) and Plackett-Burman designs; all are analysed with
    ``discopt doe anova``.
    """
    from discopt.doe import factorial_2level_design

    template = params.template or "factorial-2level"
    pairs = params.factor_pairs
    if not pairs:
        raise DoEError(f"{template} requires --factor NAME:LOW:HIGH (repeatable)")
    if len(pairs) < 2:
        raise DoEError(f"{template} needs at least 2 factors")
    limits = {"factorial-2level": 8, "fractional-factorial": 15, "plackett-burman": 23}
    if len(pairs) > limits[template]:
        raise DoEError(f"{template} supports up to {limits[template]} factors")

    extra_args: dict[str, Any] = {}
    try:
        if template == "factorial-2level":
            design = factorial_2level_design(
                pairs,
                center_points=params.center_points,
                replicates=params.replicates,
                seed=params.seed,
            )
        elif template == "plackett-burman":
            from discopt.doe.screening_designs import plackett_burman_design

            design = plackett_burman_design(
                pairs,
                n_runs=params.runs,
                center_points=params.center_points,
                replicates=params.replicates,
                seed=params.seed,
            )
            extra_args["runs"] = len([r for r in design.rows if not r.get("is_center")]) // max(
                1, int(params.replicates)
            )
        else:  # fractional-factorial
            from discopt.doe.fractional import fractional_factorial_design

            if not params.generators and not params.runs:
                raise DoEError(
                    "fractional-factorial needs --generator 'D=ABC' (repeatable) or --runs N"
                )
            design = fractional_factorial_design(
                pairs,
                n_runs=params.runs,
                resolution=int(params.resolution),
                generators=list(params.generators) if params.generators else None,
                center_points=params.center_points,
                replicates=params.replicates,
                seed=params.seed,
            )
            if params.generators:
                extra_args["generators"] = list(params.generators)
            else:
                extra_args["runs"] = int(params.runs or 0)
                extra_args["resolution"] = int(params.resolution)
    except ValueError as e:
        raise DoEError(str(e)) from e

    factor_names = list(pairs.keys())
    input_specs_data: list[tuple[str, float, float]] = []
    for name in factor_names:
        lo, hi = pairs[name]
        if (
            isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and not isinstance(lo, bool)
        ):
            lb, ub = float(min(lo, hi)), float(max(lo, hi))
            if ub == lb:
                ub = lb + 1.0
        else:
            lb, ub = 0.0, 1.0
        input_specs_data.append((name, lb, ub))

    template_args: dict[str, Any] = {
        "levels": {name: [pairs[name][0], pairs[name][1]] for name in factor_names},
        "replicates": int(params.replicates),
        "center_points": int(params.center_points),
        "family": template,
        **extra_args,
    }

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in input_specs_data],
        criterion="anova",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
        extra_columns=["replicate", *_extra_columns(params)],
    )

    designs = [
        {**{n: row[n] for n in factor_names}, "replicate": row.get("replicate")}
        for row in design.rows
    ]
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": template,
            "n": len(designs),
            "replicates": params.replicates,
            "center_points": params.center_points,
            "k": len(factor_names),
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": template,
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{n: design.rows[i][n] for n in factor_names}}
            for i, rid in enumerate(new_ids)
        ],
        "criterion": "anova",
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe anova {wb.path}",
    }


def _symbolic_start_values(wb: Workbook, model) -> dict[str, float]:
    """Starting parameter values for a nonlinear fit or a design refresh.

    Prefers estimates already in the workbook's ``parameters`` sheet — after a
    first fit those are far better than the original guess — and falls back to
    the nominal values the campaign was created with.
    """
    nominal = dict(wb.param_initial_guess() or {})
    try:
        fitted = {
            row["name"]: row.get("estimate") for row in wb.read_parameters() if row.get("name")
        }
    except Exception:  # noqa: BLE001 - an unfitted workbook is the normal case
        fitted = {}
    out: dict[str, float] = {}
    for name in model.parameter_names:
        value = fitted.get(name)
        if value is None or not math.isfinite(float(value)):
            value = nominal.get(name, 1.0)
        out[name] = float(value)
    return out


def _do_fit_symbolic(wb: Workbook, completed: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit a user-defined model by nonlinear least squares."""
    from discopt.doe.symbolic import fit_least_squares

    model = wb.symbolic_model()
    start = _symbolic_start_values(wb, model)

    missing = [n for n in model.input_names if any(r.get(n) is None for r in completed)]
    if missing:
        raise DoEError(f"completed runs have blank values for input(s) {missing}")

    try:
        out = fit_least_squares(model, completed, start)
    except (ValueError, RuntimeError) as e:
        raise DoEError(str(e)) from e

    names = out["parameter_names"]
    wb.write_parameters(
        names,
        out["estimates"],
        out["std_errors"],
        {n: (out["ci_lower"][n], out["ci_upper"][n]) for n in names},
        sigma=float(out["sigma"]),
        sigma_source=str(out["sigma_source"]),
    )
    # `extend` adds each new run's information at the declared measurement
    # error, so the stored prior must be on that same scale. fit_least_squares
    # reports the FIM at the sigma behind its standard errors (often the
    # residual estimate); rescale it to the declared one.
    declared = float(model.measurement_error)
    design_fim = np.asarray(out["fim"]) * (float(out["sigma"]) / declared) ** 2
    wb.write_fim(design_fim, names)

    # The same regression statistics the linear path reports, so a user-defined
    # model is not the one kind of fit that cannot tell you whether its
    # coefficients are distinguishable from zero. For a nonlinear model these
    # are the asymptotic (Wald) forms — the footing the confidence intervals
    # above already stand on.
    response = wb.response_name()
    coefficients, anova_rows, fit_summary = _compute_anova(
        y=np.array([float(r[response]) for r in completed], dtype=np.float64),
        beta=np.array([out["estimates"][n] for n in names], dtype=np.float64),
        residual_ss=float(out["residual_sum_of_squares"]),
        std_errs=np.array([out["std_errors"][n] for n in names], dtype=np.float64),
        parameter_names=list(names),
        n_obs=int(out["n_observations"]),
        n_p=len(names),
        sigma=float(model.measurement_error),
        inputs=_input_matrix(wb, completed),
    )
    wb.write_anova(coefficients=coefficients, anova_rows=anova_rows, fit_summary=fit_summary)
    wb.log(
        "fit",
        {
            "template": "symbolic",
            "n_observations": out["n_observations"],
            "rss": out["residual_sum_of_squares"],
            "converged": out["success"],
        },
    )
    wb.save()

    log_det = float(np.linalg.slogdet(design_fim)[1])
    return {
        "workbook_path": str(wb.path),
        "parameters": [
            {
                "name": n,
                "estimate": out["estimates"][n],
                "std_error": out["std_errors"][n],
                "ci_lower_95": out["ci_lower"][n],
                "ci_upper_95": out["ci_upper"][n],
            }
            for n in names
        ],
        "parameter_names": names,
        "n_observations": out["n_observations"],
        "objective": out["residual_sum_of_squares"],
        "sigma": float(out["sigma"]),
        "sigma_source": str(out["sigma_source"]),
        "degrees_of_freedom": int(out["degrees_of_freedom"]),
        "log_det_fim": log_det,
        "coefficients": coefficients,
        "regression_anova": anova_rows,
        "summary": dict(fit_summary),
        "expression": model.source,
        "converged": out["success"],
        "solver_message": out["message"],
        "next_command": f"discopt doe extend {wb.path} --n N",
    }


def _do_fit_module(wb: Workbook, completed: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit a user-defined ``--module`` experiment to the completed runs.

    This is the estimate :func:`discopt.doe.sequential_doe` makes from
    ``initial_runs``: the per-run model is stacked over every completed run's
    conditions (:func:`discopt.doe.campaign_experiment`), blank responses are
    dropped from the likelihood, and the parameters are found by weighted least
    squares starting from the campaign's parameter guesses. Uncertainty uses the
    declared measurement errors (``covariance = inv(FIM)``), as the Python API
    does; the reduced chi-square in the summary says whether those errors look
    right for the scatter actually seen.
    """
    from scipy.stats import t as t_dist

    from discopt.doe.runs import campaign_experiment

    experiment, names = wb.rebuild_experiment()
    runs, n_missing = _module_campaign_runs(wb, completed)
    try:
        camp = campaign_experiment(experiment, runs)
        data = camp.data_from_runs(runs)
        est = camp.estimate(data, initial_guess=wb.param_initial_guess())
    except (ValueError, RuntimeError, KeyError) as e:
        raise DoEError(f"fit failed: {e}") from e

    names = list(est.parameter_names)
    n_obs = int(est.n_observations)
    n_p = len(names)
    dof = n_obs - n_p
    se = est.standard_errors
    cis = est.confidence_intervals
    std_errors = {n: float(se[n]) for n in names}
    ci_map = {n: (float(cis[n][0]), float(cis[n][1])) for n in names}
    estimates = {n: float(est.parameters[n]) for n in names}

    # The FIM is on the declared-sigma scale and the standard errors come from
    # it directly, so the "fitted sigma" that turns one into the other is the
    # declared one (read_covariance then returns inv(FIM) unchanged).
    declared = wb.measurement_error()
    wb.write_parameters(
        names, estimates, std_errors, ci_map, sigma=declared, sigma_source="declared"
    )
    fim = np.asarray(est.fim, dtype=np.float64)
    wb.write_fim(fim, names)

    deviance = float(est.objective)
    reduced_chi2 = deviance / dof if dof > 0 else float("nan")
    coefficients: list[dict[str, Any]] = []
    for n in names:
        s_ = std_errors[n]
        t_stat = estimates[n] / s_ if s_ > 0 and np.isfinite(s_) else float("nan")
        p_val = (
            float(2.0 * t_dist.sf(abs(t_stat), df=dof))
            if dof > 0 and np.isfinite(t_stat)
            else float("nan")
        )
        coefficients.append(
            {
                "name": n,
                "estimate": estimates[n],
                "std_error": s_,
                "t_statistic": float(t_stat),
                "p_value": p_val,
                "ci_lower_95": ci_map[n][0],
                "ci_upper_95": ci_map[n][1],
            }
        )
    fit_summary: list[tuple[str, Any]] = [
        ("Observations", n_obs),
        ("Missing responses", n_missing),
        ("Parameters", n_p),
        ("Residual df", dof),
        ("Weighted SSR (chi-square)", deviance),
        ("Reduced chi-square", reduced_chi2),
        ("Uncertainty from", "declared measurement errors"),
    ]
    wb.write_anova(coefficients=coefficients, anova_rows=[], fit_summary=fit_summary)
    converged = bool(getattr(est.solve_result, "success", True))
    wb.log(
        "fit",
        {
            "module_callable": wb.module_callable(),
            "n_observations": n_obs,
            "n_missing": n_missing,
            "chi_square": deviance,
            "converged": converged,
        },
    )
    wb.save()

    warnings_out: list[str] = []
    if dof <= 0:
        warnings_out.append(
            f"{n_obs} measured value(s) for {n_p} parameter(s): no residual degrees of "
            "freedom, so the confidence intervals rest entirely on the declared errors."
        )
    if not converged:
        warnings_out.append(
            "the least-squares solver did not report convergence: "
            f"{getattr(est.solve_result, 'message', '')}"
        )

    sign, logdet = np.linalg.slogdet(fim)
    return {
        "workbook_path": str(wb.path),
        "parameters": [
            {
                "name": n,
                "estimate": estimates[n],
                "std_error": std_errors[n],
                "ci_lower_95": ci_map[n][0],
                "ci_upper_95": ci_map[n][1],
            }
            for n in names
        ],
        "parameter_names": names,
        "response_names": wb.response_names(),
        "n_observations": n_obs,
        "n_missing": n_missing,
        "objective": deviance,
        "reduced_chi_square": reduced_chi2,
        "sigma": declared,
        "sigma_source": "declared",
        "degrees_of_freedom": max(0, dof),
        "log_det_fim": float(logdet) if sign > 0 else float("-inf"),
        "covariance": np.asarray(est.covariance, dtype=np.float64).tolist(),
        "coefficients": coefficients,
        "regression_anova": [],
        "summary": dict(fit_summary),
        "converged": converged,
        "solver_message": str(getattr(est.solve_result, "message", "")),
        "warnings": warnings_out,
        "next_command": f"discopt doe extend {wb.path} --n N",
    }


def _do_extend_symbolic(wb: Workbook, params: ExtendParams) -> dict[str, Any]:
    """Design the next batch for a user-defined model.

    The information already collected enters as a prior FIM, and the Jacobian is
    re-evaluated at the *current* parameter estimates — so each round re-centres
    the local optimality of the design on what the data now say.
    """
    from discopt.doe.linear_design import batch_design_from_basis
    from discopt.doe.symbolic import basis_evaluator

    model = wb.symbolic_model()
    theta = _symbolic_start_values(wb, model)
    specs = wb.input_specs()
    bounds = {s.name: (s.lb, s.ub) for s in specs}
    completed = wb.completed_runs()
    n_p = len(model.parameter_names)

    warnings_out: list[str] = []
    pending = wb.pending_runs()
    if pending:
        warnings_out.append(
            f"{len(pending)} run(s) are still pending; they contribute no "
            "information to the prior FIM, so the recommended batch may overlap them."
        )
    if not completed:
        warnings_out.append(
            "no completed runs yet, so the design is still centred on the nominal "
            "parameter guess rather than on measured data."
        )

    # With completed runs the information they carry is both the prior and its
    # own scale reference; without them, fall back to sampling the box.
    if completed:
        designs_done = [{n: float(r[n]) for n in model.input_names} for r in completed]
        collected = model.fim(theta, designs_done)
        ridge = _scaled_ridge(collected, n_p)
        prior = collected + ridge
    else:
        ridge = _scaled_ridge(
            _basis_reference_fim(
                basis_evaluator(model, theta),
                [(s.name, s.lb, s.ub) for s in specs],
                wb.measurement_error(),
            ),
            n_p,
        )
        prior = ridge

    batch_idx = wb.next_batch_index()
    target_spec = _extend_target_spec(
        wb, params, list(model.parameter_names), [model.response_name], list(model.input_names)
    )
    criterion = wb.criterion() or _DEFAULT_CRITERION
    precision = None
    try:
        if target_spec:
            designs, criterion_value, precision = _symbolic_target_design(
                model,
                theta,
                bounds,
                int(params.n),
                target_spec,
                criterion=criterion,
                prior_fim=prior,
                ridge=ridge,
                n_starts=params.n_starts,
                seed=wb.seed() + batch_idx,
                strategy=str(params.batch_strategy or "greedy"),
            )
        else:
            batch = batch_design_from_basis(
                basis_evaluator(model, theta),
                int(params.n),
                parameter_names=list(model.parameter_names),
                input_names=list(model.input_names),
                design_bounds=bounds,
                measurement_error=wb.measurement_error(),
                criterion=criterion,
                prior_fim=prior,
                n_starts=params.n_starts,
                seed=wb.seed() + batch_idx,
            )
            designs, criterion_value = list(batch.designs), float(batch.criterion_value)
            criterion = batch.criterion
    except (ValueError, RuntimeError) as e:
        raise DoEError(str(e)) from e

    new_ids = wb.append_runs(batch_idx, designs)
    wb.log(
        "extend",
        {
            "template": "symbolic",
            "n": params.n,
            "batch": batch_idx,
            "theta": theta,
            "target": target_spec,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "batch": batch_idx,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in model.input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": criterion,
        "criterion_value": criterion_value,
        "parameter_names": list(model.parameter_names),
        "design_parameters": theta,
        "target": target_spec,
        "target_precision": _precision_out(precision) if precision is not None else None,
        "warnings": warnings_out,
        "next_command": f"discopt doe status {wb.path}",
    }


def _build_symbolic_model(params: NewParams):
    """Construct the SymbolicModel a `new symbolic` invocation describes."""
    from discopt.doe.symbolic import ModelSyntaxError, SymbolicModel

    if not params.expression:
        raise DoEError("symbolic requires --expr 'EXPRESSION' (the response formula)")
    if not params.param_initial_guess:
        raise DoEError(
            "symbolic requires --param NAME=VALUE (repeatable). A nonlinear model's "
            "information depends on the parameter values, so the design is built "
            "around a nominal guess."
        )
    if not params.inputs:
        raise DoEError("symbolic requires --input NAME:LB:UB (repeatable)")

    try:
        return SymbolicModel(
            source=params.expression,
            parameter_names=tuple(params.param_initial_guess),
            input_names=tuple(n for n, _, _ in params.inputs),
            response_name=params.response_name,
            measurement_error=params.measurement_error,
        )
    except (ModelSyntaxError, ValueError) as e:
        raise DoEError(str(e)) from e


def _symbolic_target_design(
    model: Any,
    theta: Mapping[str, float],
    bounds: dict[str, tuple[float, float]],
    n: int,
    spec: Mapping[str, Any],
    *,
    criterion: str,
    prior_fim: np.ndarray,
    ridge: np.ndarray | None,
    n_starts: int,
    seed: int,
    strategy: str = "joint",
    use_jax: bool | None = None,
) -> tuple[list[dict[str, float]], float, Any]:
    """A targeted design for a symbolic model.

    Through its jax Experiment (:func:`symbolic_experiment`) where jax and the
    base discopt package are available: every batch strategy, and replicate
    candidates. Otherwise -- the browser app, Pyodide -- through the jax-free
    basis search with :func:`~discopt.doe.targets.basis_target`.
    ``use_jax=None`` picks automatically.
    """
    if use_jax is None:
        try:
            import jax  # noqa: F401

            from discopt.doe.runs import symbolic_experiment  # noqa: F401

            use_jax = True
        except ImportError:
            use_jax = False
    if not use_jax:
        from discopt.doe.linear_design import batch_design_from_basis
        from discopt.doe.symbolic import basis_evaluator

        basis = basis_evaluator(model, dict(theta))
        target = _basis_target_or_error(
            spec,
            list(model.parameter_names),
            list(model.input_names),
            basis,
            model.response_name,
            dict(theta),
        )
        batch = batch_design_from_basis(
            basis,
            int(n),
            parameter_names=list(model.parameter_names),
            input_names=list(model.input_names),
            design_bounds=bounds,
            measurement_error=model.measurement_error,
            criterion=criterion,
            prior_fim=prior_fim,
            n_starts=n_starts,
            seed=seed,
            target=target,
        )
        precision = _target_precision_without_ridge(target, batch.joint_fim, ridge)
        return list(batch.designs), float(batch.criterion_value), precision

    from discopt.doe.design import batch_optimal_experiment, optimal_experiment
    from discopt.doe.runs import symbolic_experiment

    exp = symbolic_experiment(model, input_bounds=bounds)
    target = _target_from_spec(spec)
    theta = {k: float(v) for k, v in theta.items()}
    if n == 1:
        res = optimal_experiment(
            exp,
            theta,
            bounds,
            criterion=criterion,
            prior_fim=prior_fim,
            n_starts=n_starts,
            seed=seed,
            target=target,
        )
        precision = _target_precision_without_ridge(res.target, res.fim, ridge)
        return [res.design], float(res.criterion_value), precision
    batch = batch_optimal_experiment(
        exp,
        theta,
        bounds,
        n_experiments=n,
        criterion=criterion,
        strategy=strategy,
        prior_fim=prior_fim,
        n_starts=n_starts,
        seed=seed,
        target=target,
    )
    precision = _target_precision_without_ridge(batch.target, batch.joint_fim, ridge)
    return list(batch.designs), float(batch.criterion_value), precision


def _do_new_symbolic(params: NewParams) -> dict[str, Any]:
    """Design a locally-optimal batch for a user-defined model.

    The Jacobian comes from sympy rather than jax, so this runs anywhere numpy
    and scipy do. Because the model is nonlinear in its parameters the design is
    only optimal *around the nominal values* — which is why `--param` is
    required and why the natural workflow is design, fit, then `extend` around
    the improved estimates.
    """
    from discopt.doe.linear_design import batch_design_from_basis
    from discopt.doe.symbolic import basis_evaluator

    model = _build_symbolic_model(params)
    nominal = dict(params.param_initial_guess)
    input_names = list(model.input_names)
    n_p = len(model.parameter_names)
    target_spec = (
        _check_target_spec(
            params.target, list(model.parameter_names), [model.response_name], input_names
        )
        if params.target
        else None
    )

    # With a target, fewer runs than parameters is the point: only the targets
    # have to be identifiable.
    if int(params.n) < n_p and target_spec is None:
        raise DoEError(
            f"a model with {n_p} parameters needs at least {n_p} runs to be "
            f"identifiable; got --n {params.n} (or name the parameters you need with "
            "--target)"
        )

    basis = basis_evaluator(model, nominal)
    prior = _scaled_ridge(_basis_reference_fim(basis, params.inputs, params.measurement_error), n_p)
    precision = None
    try:
        if target_spec is not None:
            designs, criterion_value, precision = _symbolic_target_design(
                model,
                nominal,
                _design_bounds(params.inputs),
                int(params.n),
                target_spec,
                criterion=params.criterion,
                prior_fim=prior,
                ridge=prior,
                n_starts=params.n_starts,
                seed=params.seed,
            )
        else:
            batch = batch_design_from_basis(
                basis,
                int(params.n),
                parameter_names=list(model.parameter_names),
                input_names=input_names,
                design_bounds=_design_bounds(params.inputs),
                measurement_error=params.measurement_error,
                criterion=params.criterion,
                prior_fim=prior,
                n_starts=params.n_starts,
                seed=params.seed,
            )
            designs, criterion_value = list(batch.designs), float(batch.criterion_value)
    except (ValueError, RuntimeError) as e:
        raise DoEError(str(e)) from e

    wb = Workbook.create(
        params.output,
        template="symbolic",
        template_args=model.to_metadata(),
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion=params.criterion,
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=nominal,
        extra_columns=_extra_columns(params),
    )
    if target_spec:
        wb.set_target_spec(target_spec)
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": "symbolic",
            "expression": model.source,
            "n": params.n,
            "criterion": params.criterion,
            "nominal": nominal,
            "target": target_spec,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": "symbolic",
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": params.criterion,
        "criterion_value": criterion_value,
        "parameter_names": list(model.parameter_names),
        "n_parameters": n_p,
        "expression": model.source,
        "nominal_parameters": nominal,
        "target": target_spec,
        "target_precision": _precision_out(precision) if precision is not None else None,
        "next_command": f"discopt doe status {wb.path}",
    }


def _do_new_linear(params: NewParams) -> dict[str, Any]:
    """Build a parametric design without jax, using the closed-form FIM.

    Same result as the autodiff path for every built-in template — they are all
    linear in their parameters, so ``XᵀX/σ²`` *is* the Fisher information (see
    :mod:`discopt.doe.linear_design`). Opt-in via ``NewParams.use_linear_design``
    because it exists for environments where jax cannot be installed at all;
    the desktop CLI keeps the autodiff path as its default.
    """
    from discopt.doe.linear_design import LINEAR_TEMPLATES, linear_batch_design
    from discopt.doe.templates import template_parameter_names

    template = params.template or ""
    if template not in LINEAR_TEMPLATES:
        raise DoEError(
            f"use_linear_design requires a template that is linear in its parameters; "
            f"{template!r} is not one of {sorted(LINEAR_TEMPLATES)}"
        )
    if not params.inputs:
        raise DoEError(f"{template} requires --input NAME:LB:UB (repeatable)")

    input_names = [s[0] for s in params.inputs]
    parameter_names = template_parameter_names(
        template,
        degree=int(params.degree) if params.degree is not None else None,
        n_inputs=len(input_names),
    )
    eq_cons, proj = _mixture_constraints(template, params.inputs, params.mixture_total)

    template_args: dict[str, Any] = {}
    if params.degree is not None:
        template_args["degree"] = int(params.degree)
    if params.mixture_total is not None:
        template_args["mixture_total"] = float(params.mixture_total)

    ridge = _scaled_ridge(
        _template_reference_fim(
            template,
            template_args,
            parameter_names,
            input_names,
            params.inputs,
            params.measurement_error,
        ),
        len(parameter_names),
    )
    target_spec, target = None, None
    if params.target:
        from discopt.doe.linear_design import _basis_evaluator

        target_spec = _check_target_spec(
            params.target, parameter_names, [params.response_name], input_names
        )
        target = _basis_target_or_error(
            target_spec,
            parameter_names,
            input_names,
            _basis_evaluator(template, template_args, parameter_names, input_names),
            params.response_name,
        )

    try:
        batch = linear_batch_design(
            template,
            max(int(params.n), 1),
            parameter_names=parameter_names,
            input_names=input_names,
            design_bounds=_design_bounds(params.inputs),
            template_args=template_args,
            measurement_error=params.measurement_error,
            criterion=params.criterion,
            # The same ridge the autodiff path seeds (see the parametric branch
            # of do_new). It keeps the FIM non-singular from the first round, so
            # the requested criterion is well-posed for every pick rather than
            # only once the design has accumulated full rank — and it is scaled
            # to the information in each direction so it cannot decide the
            # design (see _scaled_ridge).
            prior_fim=ridge,
            equality_constraints=eq_cons or None,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=params.seed,
            target=target,
        )
    except (ValueError, RuntimeError) as e:
        raise DoEError(str(e)) from e

    designs = list(batch.designs)
    wb = Workbook.create(
        params.output,
        template=template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion=params.criterion,
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
        extra_columns=_extra_columns(params),
    )
    if target_spec:
        wb.set_target_spec(target_spec)
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": template,
            "n": params.n,
            "criterion": params.criterion,
            "fim": "linear",
            "target": target_spec,
        },
    )
    wb.save()
    precision = _target_precision_without_ridge(target, batch.joint_fim, ridge)

    return {
        "workbook_path": str(wb.path),
        "template": template,
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": params.criterion,
        "criterion_value": float(batch.criterion_value),
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
        "target": target_spec,
        "target_precision": _precision_out(precision) if precision is not None else None,
        "next_command": f"discopt doe status {wb.path}",
    }


def _do_new_classical(params: NewParams) -> dict[str, Any]:
    """Build a classical space-filling / response-surface design as a workbook.

    These skip the FIM search entirely — the run list is closed-form — but
    unlike the combinatorial family they record a regression ``basis`` so
    ``discopt doe fit`` can estimate a model from the completed runs.
    """
    from discopt.doe.classical import (
        box_behnken_design,
        central_composite_design,
        latin_hypercube_design,
    )
    from discopt.doe.linear_design import BASES, basis_parameter_names

    template = params.template or ""
    if not params.inputs:
        raise DoEError(f"{template} requires --input NAME:LB:UB (repeatable)")
    if params.basis not in BASES:
        raise DoEError(f"--basis must be one of {list(BASES)}, got {params.basis!r}")

    factors = {name: (lb, ub) for name, lb, ub in params.inputs}
    input_names = list(factors)
    template_args: dict[str, Any] = {"basis": params.basis, "family": template}
    dsd_rows: list[dict[str, float]] = []

    try:
        if template == "latin-hypercube":
            if int(params.n) < 2:
                raise DoEError("latin-hypercube requires --n >= 2")
            optimize: bool | str = {"none": False, "discrepancy": True}.get(
                params.lhs_optimize, params.lhs_optimize
            )
            design = latin_hypercube_design(
                factors, int(params.n), optimize=optimize, seed=params.seed
            )
            if params.lhs_optimize != "discrepancy":
                template_args["optimize"] = params.lhs_optimize
        elif template == "central-composite":
            alpha: float | str = params.alpha
            if params.alpha not in ("rotatable", "face"):
                try:
                    alpha = float(params.alpha)
                except ValueError as e:
                    raise DoEError(
                        f"--alpha must be 'rotatable', 'face', or a number, got {params.alpha!r}"
                    ) from e
            design = central_composite_design(
                factors,
                alpha=alpha,
                center_points=params.center_points if params.center_points else 4,
                within_bounds=params.within_bounds,
                seed=params.seed,
            )
            template_args["alpha"] = params.alpha
            template_args["within_bounds"] = bool(params.within_bounds)
            template_args["center_points"] = int(params.center_points or 4)
        elif template == "box-behnken":
            design = box_behnken_design(
                factors,
                center_points=params.center_points if params.center_points else 3,
                seed=params.seed,
            )
            template_args["center_points"] = int(params.center_points or 3)
        elif template == "definitive-screening":
            from discopt.doe.screening_designs import definitive_screening_design

            if len(factors) < 3:
                raise DoEError("definitive-screening needs at least 3 factors")
            dsd = definitive_screening_design(
                factors,
                fake_factors=int(params.fake_factors),
                center_points=params.center_points if params.center_points else 1,
                seed=params.seed,
            )
            template_args["fake_factors"] = int(params.fake_factors)
            template_args["center_points"] = int(params.center_points or 1)
            dsd_rows = [{n: float(r[n]) for n in input_names} for r in dsd.rows]
            n_params = len(basis_parameter_names(params.basis, len(input_names)))
            if n_params > len(dsd_rows):
                raise DoEError(
                    f"a {params.basis} model has {n_params} coefficients but this "
                    f"definitive screening design has only {len(dsd_rows)} runs; use "
                    "--basis linear (the usual first analysis) or add --fake-factors"
                )

        else:  # pragma: no cover - do_new only routes the classical names here
            raise DoEError(f"unknown classical template {template!r}")
    except ValueError as e:
        # The generators validate factor counts and bounds; surface those as
        # ordinary CLI failures rather than tracebacks.
        raise DoEError(str(e)) from e

    parameter_names = basis_parameter_names(params.basis, len(input_names))

    wb = Workbook.create(
        params.output,
        template=template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion="classical",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
        extra_columns=_extra_columns(params),
    )

    designs = dsd_rows if template == "definitive-screening" else design.design_rows()
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": template,
            "n": len(designs),
            "basis": params.basis,
            **{k: v for k, v in template_args.items() if k not in ("basis", "family")},
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": template,
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{n: float(d[n]) for n in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": "classical",
        "criterion_value": float("nan"),
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
        "next_command": f"discopt doe status {wb.path}",
    }


def _do_new_optimize(params: NewParams) -> dict[str, Any]:
    """Create a workbook seeded with an initial Sobol batch for active learning.

    The workbook stores the box bounds (``input_specs``) so subsequent
    rounds of :func:`do_optimize` can sample candidates inside them.
    The seed batch is generated by Sobol sampling and gives the
    surrogate something to fit against.
    """
    if not params.inputs:
        raise DoEError("optimize template requires --input NAME:LB:UB (at least one)")
    if int(params.n) < 2:
        raise DoEError("optimize seed batch must have n >= 2 runs")

    bounds = np.array([(lb, ub) for _, lb, ub in params.inputs], dtype=float)
    input_names = [n_ for n_, _, _ in params.inputs]

    try:
        from scipy.stats import qmc

        engine = qmc.Sobol(d=len(input_names), scramble=True, seed=int(params.seed))
        m = max(1, int(math.ceil(math.log2(int(params.n)))))
        unit = engine.random_base2(m=m)[: int(params.n)]
    except ImportError:
        rng = np.random.default_rng(int(params.seed))
        unit = rng.uniform(size=(int(params.n), len(input_names)))

    lo, hi = bounds[:, 0], bounds[:, 1]
    seed_points = lo + (hi - lo) * unit
    designs = [
        {nm: float(seed_points[i, j]) for j, nm in enumerate(input_names)}
        for i in range(int(params.n))
    ]

    template_args: dict[str, Any] = {
        "family": "optimize",
        "criterion": params.optimize_criterion,
        "surrogate": params.optimize_surrogate,
        "acquisition": params.optimize_acquisition,
    }

    wb = Workbook.create(
        params.output,
        template="optimize",
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion="active-learning",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
        extra_columns=_extra_columns(params),
    )
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": "optimize",
            "n": int(params.n),
            "criterion": params.optimize_criterion,
            "surrogate": params.optimize_surrogate,
            "acquisition": params.optimize_acquisition,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": "optimize",
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [{"run_id": rid, **designs[i]} for i, rid in enumerate(new_ids)],
        "criterion": params.optimize_criterion,
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe optimize {wb.path}",
    }


def do_optimize(params: OptimizeParams) -> dict[str, Any]:
    """Run one active-learning round on an optimize-template workbook.

    Reads completed runs from ``params.workbook``, fits the chosen
    surrogate, scores Sobol candidates with the chosen acquisition,
    and appends the recommended batch as pending runs.
    """
    from discopt.doe import (
        OptimizationCriterion,
    )
    from discopt.doe import (
        optimize_round as _optimize_round,
    )

    wb = Workbook.open(Path(params.workbook))
    if (wb.template_name() or "") != "optimize":
        raise DoEError(
            f"workbook template is {wb.template_name()!r}, not 'optimize'; "
            "use `discopt doe new optimize` to create one"
        )

    # Resolve criterion/surrogate/acquisition from the settings the workbook
    # was created with, unless the caller explicitly overrode them. Without
    # this, a workbook made with `--optimize-criterion minimize` would be
    # silently *maximized* by the argparse default.
    stored = wb.template_args()
    warnings: list[str] = []

    def _resolve(name: str, given: object, fallback: str) -> str:
        stored_val = stored.get(name, fallback)
        if given is None:
            return str(stored_val)
        if str(given) != str(stored_val):
            warnings.append(
                f"{name} overridden: workbook was created with "
                f"{stored_val!r}, using {given!r} for this round."
            )
        return str(given)

    criterion = _resolve("criterion", params.criterion, "maximize")
    acquisition = _resolve("acquisition", params.acquisition, "expected_improvement")
    surrogate = _resolve("surrogate", params.surrogate, "gp")

    surrogate_obj: object = surrogate
    if params.custom_surrogate_path:
        surrogate_obj = _instantiate_dotted(
            params.custom_surrogate_path, params.custom_surrogate_kwargs or {}
        )

    result = _optimize_round(
        workbook=wb,
        criterion=OptimizationCriterion(criterion),
        surrogate=surrogate_obj,
        acquisition=acquisition,
        batch_size=int(params.batch_size),
        n_candidates=int(params.n_candidates),
        seed=params.seed,
        acquisition_kwargs=params.acquisition_kwargs,
    )

    return {
        "workbook_path": str(wb.path),
        "criterion": criterion,
        "acquisition": acquisition,
        "surrogate": (params.custom_surrogate_path if params.custom_surrogate_path else surrogate),
        "warnings": warnings,
        "surrogate_mode": result.surrogate_mode,
        "batch_size": int(params.batch_size),
        "n_completed": result.n_completed,
        "incumbent_x": result.incumbent_x,
        "incumbent_y": result.incumbent_y,
        "new_run_ids": result.new_run_ids,
        "next_designs": [
            {"run_id": rid, **d} for rid, d in zip(result.new_run_ids, result.next_designs)
        ],
        "acquisition_scores": result.acquisition_scores,
    }


def _instantiate_dotted(spec: str, kwargs: dict[str, Any]) -> object:
    """Import and instantiate ``module.path:ClassName`` or ``module.path.ClassName``.

    ``spec`` may use either ``:`` (preferred) or the last ``.`` as the
    separator between module path and class name. ``kwargs`` are
    forwarded to the constructor.
    """
    import importlib

    if ":" in spec:
        mod_path, cls_name = spec.split(":", 1)
    else:
        if "." not in spec:
            raise DoEError(
                f"custom surrogate spec {spec!r} must be 'module.path:ClassName' "
                "or 'module.path.ClassName'"
            )
        mod_path, _, cls_name = spec.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise DoEError(f"could not import {mod_path!r}: {e}") from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise DoEError(f"{cls_name!r} not found in {mod_path!r}") from e
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise DoEError(f"could not instantiate {spec} with kwargs {kwargs!r}: {e}") from e


_LATIN_FACTOR_COUNT = {
    "latin-square": 3,
    "graeco-latin": 4,
    "hyper-graeco-latin": 5,
}


def _do_new_latin(params: NewParams) -> dict[str, Any]:
    """Build a Latin-family design (closed-form) and persist as a workbook.

    Latin family designs do not have a FIM-based optimization step;
    runs come from the combinatorial generator and the workbook is
    used only for capturing responses + downstream ANOVA.
    """
    from discopt.doe import latin_square_design

    if not params.template or params.template not in _LATIN_FACTOR_COUNT:
        raise DoEError(f"unsupported latin template {params.template!r}")
    if not params.levels:
        raise DoEError(f"{params.template} requires --levels NAME:L1,L2,... per factor")

    expected = _LATIN_FACTOR_COUNT[params.template]
    if len(params.levels) != expected:
        raise DoEError(
            f"{params.template} requires exactly {expected} factors, got {len(params.levels)}"
        )

    level_counts = {len(v) for v in params.levels.values()}
    if len(level_counts) != 1:
        raise DoEError("all factors must have the same number of levels")
    k = level_counts.pop()
    if k < 2:
        raise DoEError("each factor needs at least 2 levels")
    if params.template != "latin-square" and k == 6:
        raise DoEError(
            f"{params.template}: no orthogonal Latin squares exist for k = 6 (Euler exception)"
        )

    design = latin_square_design(params.levels, replicates=params.replicates, seed=params.seed)

    # Synthesize numeric (lb, ub) input specs for compatibility with the
    # rest of the workbook code. For numeric levels we use min/max; for
    # categorical levels we use 0..k-1 placeholders.
    factor_names = list(params.levels.keys())
    input_specs_data: list[tuple[str, float, float]] = []
    for name in factor_names:
        vals = params.levels[name]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            numeric_vals = [float(cast(float, v)) for v in vals]
            lb, ub = min(numeric_vals), max(numeric_vals)
            if ub == lb:
                ub = lb + 1.0
        else:
            lb, ub = 0.0, float(len(vals) - 1)
        input_specs_data.append((name, lb, ub))

    template_args: dict[str, Any] = {
        "levels": {name: list(params.levels[name]) for name in factor_names},
        "replicates": int(params.replicates),
        "family": design.family,
    }

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in input_specs_data],
        criterion="anova",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
        extra_columns=["replicate", *_extra_columns(params)],
    )

    designs = [
        {**{n: row[n] for n in factor_names}, "replicate": row.get("replicate")}
        for row in design.rows
    ]
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": params.template,
            "n": len(designs),
            "replicates": params.replicates,
            "k": k,
            "family": design.family,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": params.template,
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{n: design.rows[i][n] for n in factor_names}}
            for i, rid in enumerate(new_ids)
        ],
        "criterion": "anova",
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe anova {wb.path}",
    }


def _cmd_new(args) -> int:
    output = Path(args.output)
    if output.exists() and not args.force:
        return _fail(
            args,
            f"{output} already exists. Pass --force to overwrite.",
            workbook_path=str(output),
        )
    is_module = bool(getattr(args, "_is_module", False))
    if is_module and (args.response is not None or args.error is not None):
        return _fail(
            args,
            "a --module experiment takes its response names and measurement errors from "
            "the ExperimentModel it returns; drop --response/--error and override a "
            "response's error with --sigma NAME=VALUE.",
            workbook_path=str(output),
        )
    error = 1.0 if args.error is None else float(args.error)
    if error <= 0.0:
        return _fail(
            args,
            f"--error must be positive (got {args.error}); it is the "
            "measurement standard deviation and appears as 1/sigma^2 in the FIM.",
            workbook_path=str(output),
        )
    sigma_overrides = dict(getattr(args, "sigma", None) or [])
    bad_sigma = [k for k, v in sigma_overrides.items() if not v > 0.0]
    if bad_sigma:
        return _fail(args, f"--sigma must be positive (got {bad_sigma})", workbook_path=str(output))
    inputs: list[tuple[str, float, float]] = list(getattr(args, "input", None) or []) + list(
        getattr(args, "bounds", None) or []
    )
    levels_arg = getattr(args, "levels", None)
    levels_dict: dict[str, list[object]] | None = None
    if levels_arg:
        levels_dict = {name: list(vals) for name, vals in levels_arg}
    factor_arg = getattr(args, "factor", None)
    factor_pairs: dict[str, tuple[object, object]] | None = None
    if factor_arg:
        factor_pairs = {name: (lo, hi) for name, lo, hi in factor_arg}
    params = NewParams(
        output=output,
        # --n / --criterion / --n-starts are only registered for the templates
        # that use them (see the subparser wiring); default the rest.
        n=int(getattr(args, "n", 1) or 1),
        inputs=inputs,
        response_name=args.response or "y",
        measurement_error=error,
        criterion=_normalize_criterion(getattr(args, "criterion", _DEFAULT_CRITERION)),
        seed=int(args.seed),
        n_starts=int(getattr(args, "n_starts", 10) or 10),
        template=None if is_module else getattr(args, "template", None),
        degree=getattr(args, "degree", None),
        mixture_total=getattr(args, "mixture_total", None),
        module_callable=getattr(args, "module", None) if is_module else None,
        # --param carries the nominal values for both the --module escape hatch
        # and the symbolic template; for symbolic it also fixes parameter order.
        param_initial_guess=dict(getattr(args, "params", None) or []),
        expression=getattr(args, "expr", None),
        levels=levels_dict,
        replicates=int(getattr(args, "replicates", 1) or 1),
        factor_pairs=factor_pairs,
        center_points=int(getattr(args, "center_points", 0) or 0),
        basis=getattr(args, "basis", "quadratic") or "quadratic",
        alpha=str(getattr(args, "alpha", "rotatable") or "rotatable"),
        # --outside-bounds is the opt-in for the textbook CCD scaling; the
        # default keeps every run inside the stated factor range.
        within_bounds=not bool(getattr(args, "outside_bounds", False)),
        generators=getattr(args, "generators", None),
        runs=getattr(args, "runs", None),
        resolution=int(getattr(args, "resolution", 3) or 3),
        fake_factors=int(getattr(args, "fake_factors", 0) or 0),
        lhs_optimize=getattr(args, "lhs_optimize", "discrepancy") or "discrepancy",
        extra_columns=list(getattr(args, "extra_column", None) or []),
        optimize_criterion=getattr(args, "optimize_criterion", "maximize") or "maximize",
        optimize_surrogate=getattr(args, "optimize_surrogate", "gp") or "gp",
        optimize_acquisition=(
            getattr(args, "optimize_acquisition", "expected_improvement") or "expected_improvement"
        ),
        force=bool(getattr(args, "force", False)),
        sigma_overrides=sigma_overrides,
    )
    try:
        params.target = _target_spec_from_args(args)
        out = do_new(params)
    except (DoEError, ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError) as e:
        # RuntimeError: the optimal-design search can fail to find a non-singular
        # FIM (too few runs, unidentifiable model); surface it as a clean error.
        return _fail(args, str(e), workbook_path=str(output))
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        _print_new_human(out)
    return 0


def _print_new_human(out: dict[str, Any]) -> None:
    print(f"created workbook: {out['workbook_path']}")
    label = out["template"] or out["module_callable"]
    print(f"  model:         {label}")
    print(f"  parameters:    {out['n_parameters']} ({', '.join(out['parameter_names'])})")
    print(f"  criterion:     {out['criterion']}")
    if out.get("target"):
        print(f"  target:        {_describe_target(out['target'])}")
        _print_target_precision(out, "predicted target precision (at the nominal parameters):")
    new_ids = out["new_run_ids"]
    print(f"  batch:         {out['batch']}  (run_ids {min(new_ids)}-{max(new_ids)})")
    print("  recommended runs:")
    if out["designs"]:
        keys = [k for k in out["designs"][0].keys() if k != "run_id"]
        header = "    run_id  " + "  ".join(f"{k:>10s}" for k in keys)
        print(header)
        for row in out["designs"]:
            cells = []
            for k in keys:
                v = row[k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cells.append(f"{float(v):>10.4f}")
                else:
                    cells.append(f"{str(v):>10s}")
            print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
    print(f"  next:          {out['next_command']}")


# ──────────────────────────────────────────────────────────────────
# do_status
# ──────────────────────────────────────────────────────────────────


def do_status(params: dict[str, Any]) -> dict[str, Any]:
    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    response_specs = wb.response_specs()
    response_label = ", ".join(r.name for r in response_specs)
    all_runs = wb.all_runs()
    pending = wb.pending_runs()
    completed = wb.completed_runs()
    fitted = wb.read_parameters()
    fim_data = wb.read_fim()

    # The analysis verb depends on the template family: combinatorial designs
    # use `anova` (fit/extend explicitly reject them), the active-learning
    # template uses `optimize`, and parametric templates use `fit`/`extend`.
    from discopt.doe.templates import COMBINATORIAL_TEMPLATES

    template = wb.template_name() or ""
    if template == "optimize":
        analyze_verb = "optimize"
    elif template in COMBINATORIAL_TEMPLATES:
        analyze_verb = "anova"
    else:
        analyze_verb = "fit"

    if pending:
        if completed:
            next_command = (
                f"discopt doe {analyze_verb} {wb.path}  # {len(pending)} run(s) still pending"
            )
        else:
            next_command = (
                f"# fill in '{response_label}' for run_ids "
                f"{', '.join(str(r['run_id']) for r in pending)}, save, then: "
                f"discopt doe {analyze_verb} {wb.path}"
            )
    elif completed:
        # Parametric campaigns fit, then extend for the next batch; anova and
        # optimize workbooks just re-run their single analysis verb.
        if analyze_verb == "fit" and fitted:
            next_command = f"discopt doe extend {wb.path} --n N"
        else:
            next_command = f"discopt doe {analyze_verb} {wb.path}"
    else:
        next_command = f"discopt doe new ... -o {wb.path}"

    target_spec = wb.target_spec()
    target_out: dict[str, Any] | None = None
    target_error: str | None = None
    if target_spec and fim_data is not None and fitted:
        try:
            target_out = _precision_out(_status_target_precision(wb, target_spec, fim_data))
        except Exception as e:  # noqa: BLE001 - status must still report the rest
            target_error = f"{type(e).__name__}: {e}"
        if target_out and target_out["met"] and not pending:
            next_command = (
                f"# every target is within its tolerance; more runs are optional "
                f"(discopt doe extend {wb.path} --n N)"
            )

    target_profile: dict[str, Any] | None = None
    if params.get("profile") and target_spec:
        if not completed:
            target_error = "no completed runs to profile"
        else:
            try:
                target_profile = _status_target_profile(wb, target_spec, completed)
            except Exception as e:  # noqa: BLE001 - status must still report the rest
                target_error = f"profile failed: {type(e).__name__}: {e}"

    return {
        "workbook_path": str(wb.path),
        "template": wb.template_name(),
        "template_args": wb.template_args(),
        "module_callable": wb.module_callable(),
        "response_name": response,
        "response_names": [r.name for r in response_specs],
        "responses": [r.to_dict() for r in response_specs],
        "seed": wb.seed(),
        "input_specs": [s.to_dict() for s in wb.input_specs()],
        "n_total": len(all_runs),
        "n_completed": len(completed),
        "n_pending": len(pending),
        "next_batch_index": wb.next_batch_index(),
        "parameters": fitted,
        "has_fim": fim_data is not None,
        "fim_size": fim_data[0].shape[0] if fim_data is not None else 0,
        "target": target_spec,
        "target_precision": target_out,
        "target_precision_error": target_error,
        "target_profile": target_profile,
        "next_command": next_command,
    }


def _status_target_profile(
    wb: Workbook, spec: Mapping[str, Any], completed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Each target's linearized interval next to its profile-likelihood one.

    Refits the campaign (``campaign_experiment`` over the completed runs) and
    profiles every target with :func:`~discopt.doe.profile_target`. The
    classical designs are linear in their coefficients, where the two agree
    exactly, so they are not profiled.
    """
    from discopt.doe.profile import profile_target
    from discopt.doe.runs import campaign_experiment, symbolic_experiment
    from discopt.doe.templates import CLASSICAL_TEMPLATES

    template = wb.template_name() or ""
    if template in CLASSICAL_TEMPLATES:
        return {
            "adequate": True,
            "targets": {},
            "note": "linear in its coefficients: the linearized precision is exact",
        }
    runs, _ = _module_campaign_runs(wb, completed)
    # The check refits and profiles, so it is cached in the workbook against
    # exactly what it depends on: the data, the target and the model.
    import hashlib

    key = hashlib.sha256(
        json.dumps(
            {
                "runs": [{k: v for k, v in r.items() if k != "run_id"} for r in runs],
                "target": spec,
                "template": template,
                "template_args": wb.template_args(),
                "module": wb.module_callable(),
                "sigmas": wb.response_sigmas(),
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    cached = wb.metadata().get("target_profile_cache")
    if cached:
        try:
            entry = json.loads(cached)
            if entry.get("key") == key:
                return {**entry["result"], "cached": True}
        except (ValueError, TypeError):
            pass
    if template == "symbolic":
        model: Any = wb.symbolic_model()
        bounds = {s.name: (s.lb, s.ub) for s in wb.input_specs()}
        per_run: Any = symbolic_experiment(model, input_bounds=bounds)
        guess = _symbolic_start_values(wb, model)
    else:
        per_run, names = wb.rebuild_experiment()
        model = per_run
        guess = _param_values_for_design(names, wb)
    camp = campaign_experiment(model, runs)
    data = camp.data_from_runs(runs)
    est = camp.estimate(data, initial_guess=guess)
    check = profile_target(camp, data, _target_from_spec(spec), model=per_run, initial_estimate=est)
    result = {
        "adequate": check.adequate,
        "confidence_level": check.confidence_level,
        "targets": {
            name: {
                "estimate": p.estimate,
                "std_error": p.std_error,
                "linear_ci": list(p.linear_ci),
                "profile_ci": None if p.profile_ci is None else list(p.profile_ci),
                "half_width_ratios": list(p.half_width_ratios),
                "method": p.method,
                "adequate": p.adequate,
                "note": p.note,
            }
            for name, p in check.profiles.items()
        },
    }
    wb.set_target_profile_cache({"key": key, "result": _sanitize_json(result)})
    wb.save()
    return {**result, "cached": False}


def _status_target_precision(
    wb: Workbook, spec: Mapping[str, Any], fim_data: tuple[np.ndarray, list[str]]
) -> Any:
    """Precision of the stored target under the FIM `fit` stored (data so far).

    That FIM is on the declared measurement error -- the scale `extend` designs
    and predicts on -- so the tolerance check does not jump with a residual-based
    sigma estimate from a handful of runs (the parameters table may use one).
    """
    fim, names = fim_data
    from discopt.doe.templates import CLASSICAL_TEMPLATES

    if wb.template_name() in CLASSICAL_TEMPLATES:
        from discopt.doe.linear_design import _basis_evaluator

        input_names = [s.name for s in wb.input_specs()]
        pnames = list(wb.parameter_names())
        target = _basis_target_or_error(
            spec,
            pnames,
            input_names,
            _basis_evaluator(wb.template_name() or "", wb.template_args(), pnames, input_names),
            wb.response_name(),
        )
        if list(pnames) != list(names):
            raise ValueError(f"the stored FIM is over {list(names)}, the basis over {pnames}")
        return target.precision(fim)
    if wb.template_name() == "symbolic":
        model = wb.symbolic_model()
        theta = _symbolic_start_values(wb, model)
        try:
            from discopt.doe.runs import symbolic_experiment
        except ImportError:  # no base discopt (the browser): the basis path
            from discopt.doe.symbolic import basis_evaluator

            target = _basis_target_or_error(
                spec,
                list(model.parameter_names),
                list(model.input_names),
                basis_evaluator(model, theta),
                model.response_name,
                theta,
            )
            return target.precision(fim)
        bounds = {s.name: (s.lb, s.ub) for s in wb.input_specs()}
        experiment: Any = symbolic_experiment(model, input_bounds=bounds)
    else:
        experiment, parameter_names = wb.rebuild_experiment()
        theta = _param_values_for_design(parameter_names, wb)
    resolved = _target_from_spec(spec).resolve(experiment, theta)
    if list(resolved.parameter_names) != list(names):
        raise ValueError(
            f"the stored FIM is over {list(names)}, the model's parameters are "
            f"{list(resolved.parameter_names)}; re-run `discopt doe fit`"
        )
    return resolved.precision(fim)


def _cmd_status(args) -> int:
    try:
        out = do_status(
            {"workbook": args.workbook, "profile": bool(getattr(args, "profile", False))}
        )
    except (FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        label = out["template"] or out["module_callable"] or "(unknown model)"
        print(f"{out['workbook_path']}")
        print(f"  model:       {label}")
        # For combinatorial designs the factors are categorical/level-based;
        # show the actual levels instead of the synthesized numeric bounds
        # (which would misleadingly print e.g. 'treatment in [0.0, 3.0]').
        levels = (out.get("template_args") or {}).get("levels") or {}
        parts = []
        for s in out["input_specs"]:
            if s["name"] in levels:
                parts.append(f"{s['name']} in {{{', '.join(map(str, levels[s['name']]))}}}")
            else:
                parts.append(f"{s['name']} in [{s['lb']}, {s['ub']}]")
        inputs_str = ", ".join(parts)
        print(f"  inputs:      {inputs_str}")
        if len(out["responses"]) > 1:
            print(
                "  responses:   "
                + ", ".join(f"{r['name']} (sigma {r['sigma']:g})" for r in out["responses"])
            )
        else:
            print(f"  response:    {out['response_name']}")
        print(
            f"  runs:        {out['n_completed']} completed / "
            f"{out['n_pending']} pending / {out['n_total']} total"
        )
        if out["parameters"]:
            print("  parameters:")
            for p in out["parameters"]:
                est = p["estimate"]
                se = p["std_error"]
                if est is None:
                    continue
                se_s = f"{se:.4g}" if se is not None else "n/a"
                print(f"    {p['name']:>8s} = {est:>12.6g}  ± {se_s}")
        else:
            print("  parameters:  (not fit yet)")
        if out.get("target"):
            print(f"  target:      {_describe_target(out['target'])}")
            if out.get("target_precision"):
                _print_target_precision(
                    out, "target precision (data so far, at the declared measurement error):"
                )
            elif out.get("target_precision_error"):
                print(f"    (could not evaluate: {out['target_precision_error']})")
            else:
                print("    (precision available after `discopt doe fit`)")
            prof = out.get("target_profile")
            if prof:
                level = f"{prof.get('confidence_level', 0.95):.0%}"
                print(f"  profile likelihood check ({level} intervals):")
                if prof.get("note"):
                    print(f"    {prof['note']}")
                fmt = lambda v: "unbounded" if v is None else f"{v:.5g}"  # noqa: E731
                for name, t in prof["targets"].items():
                    if t.get("profile_ci"):
                        lo, hi = t["profile_ci"]
                        shown = f"profile [{fmt(lo)}, {fmt(hi)}]"
                    else:
                        r = t["half_width_ratios"]
                        shown = f"two-point check {fmt(r[0])}x / {fmt(r[1])}x"
                    print(
                        f"    {name:>12s}  linear [{t['linear_ci'][0]:.5g}, {t['linear_ci'][1]:.5g}]"
                        f"  {shown}  {'ok' if t['adequate'] else 'CHECK'}"
                    )
                    print(f"      {t['note']}")
                if prof.get("cached"):
                    print("    (cached: same data and target as the last check)")
        print(f"  next:        {out['next_command']}")
    return 0


def _describe_target(spec: Mapping[str, Any]) -> str:
    parts = list(spec.get("parameters") or {})
    parts += [
        f"{n}={e}" if len(e) <= 40 else f"{n} (formula)"
        for n, e in (spec.get("expressions") or {}).items()
    ]
    for r, pts in (spec.get("responses") or {}).items():
        parts += (
            [r]
            if not pts
            else [f"{r}@" + ",".join(f"{k}={v:g}" for k, v in p.items()) for p in pts]
        )
    tol = spec.get("tolerances") or {}
    text = ", ".join(parts)
    if tol:
        text += "  (tolerances " + ", ".join(f"{k}={v:g}" for k, v in tol.items()) + ")"
    if spec.get("relative"):
        text += "  [relative]"
    return text


# ──────────────────────────────────────────────────────────────────
# do_fit
# ──────────────────────────────────────────────────────────────────


def _linear_ols(wb: Workbook, completed: list[dict[str, Any]], template: str) -> dict[str, Any]:
    """Ordinary least squares of a built-in (linear-in-parameters) template.

    Shared by ``fit`` and the regression form of ``anova``. Returns the design
    matrix, response, estimates, residual SS, standard errors, CIs,
    ``(XᵀX)⁻¹`` and the matrix of input settings (for the pure-error split).
    """
    response = wb.response_name()
    input_names = [s.name for s in wb.input_specs()]
    # Derived from metadata rather than rebuild_experiment(): the fit below is
    # plain OLS over _design_row, so it needs the name ordering but not a built
    # model — and building one would drag in discopt.modeling for no benefit.
    parameter_names = wb.parameter_names()
    sigma = wb.measurement_error()
    n_p = len(parameter_names)
    n_obs = len(completed)

    # A completed row (non-blank response) with a blanked-out input would raise
    # a bare TypeError from float(None) deep in the design-matrix build; give an
    # actionable message naming the run and column instead.
    for row in completed:
        for nm in input_names:
            v = row.get(nm)
            if v is None or (isinstance(v, str) and not v.strip()):
                raise DoEError(
                    f"run {row.get('run_id')} has a blank value for input {nm!r}; "
                    "fill it in (or clear the response to mark the run pending)."
                )

    X = np.array(
        [
            _design_row(template, wb.template_args(), parameter_names, input_names, row)
            for row in completed
        ],
        dtype=np.float64,
    )
    y = np.array([float(row[response]) for row in completed], dtype=np.float64)

    rank = int(np.linalg.matrix_rank(X)) if n_obs else 0
    if rank < n_p:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Undetermined fit — too few runs, or runs whose settings cannot
        # separate every term (a full quadratic from a 3-factor definitive
        # screening design's 9 distinct points). The estimates are then one of
        # infinitely many equally good solutions and covariance / SE are not
        # meaningful (inverting the singular XᵀX gave SE = 0, i.e. false
        # certainty), but we still return what we can so the user sees progress.
        residual_ss = float(np.sum((y - X @ beta) ** 2))
        sigma_hat_sq = residual_ss / (n_obs - rank) if n_obs > rank else sigma**2
        xtx_inv = np.full((n_p, n_p), np.nan)
        std_errs = np.full(n_p, np.nan)
        cis_arr = np.full((n_p, 2), np.nan)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual_ss = float(np.sum((y - X @ beta) ** 2))
        dof = max(1, n_obs - n_p)
        sigma_hat_sq = residual_ss / dof if n_obs > n_p else sigma**2
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(X.T @ X)
        cov = sigma_hat_sq * xtx_inv
        std_errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        from scipy.stats import t as t_dist

        t_val = float(t_dist.ppf(0.975, df=dof)) if n_obs > n_p else float("nan")
        cis_arr = np.column_stack([beta - t_val * std_errs, beta + t_val * std_errs])
    inputs = np.array(
        [[float(row[nm]) for nm in input_names] for row in completed], dtype=np.float64
    )
    return {
        "input_names": input_names,
        "parameter_names": parameter_names,
        "sigma": sigma,
        "n_p": n_p,
        "n_obs": n_obs,
        "X": X,
        "y": y,
        "inputs": inputs,
        "beta": beta,
        "residual_ss": residual_ss,
        "std_errs": std_errs,
        "cis_arr": cis_arr,
        "xtx_inv": xtx_inv,
        "sigma_hat_sq": sigma_hat_sq,
        "rank": rank,
    }


def do_fit(params: dict[str, Any]) -> dict[str, Any]:
    """Estimate parameters from the completed runs in a workbook.

    For the four built-in templates the model is linear in the
    parameters, so this is ordinary least squares: ``β̂ = (XᵀX)⁻¹ Xᵀy``
    with ``cov(β̂) = σ̂² (XᵀX)⁻¹``. The cumulative Fisher Information
    Matrix is ``Σᵢ FIM(dᵢ) = XᵀX / σ²`` and is written to the workbook
    for use as ``prior_fim`` by a subsequent ``extend``.

    A ``--module`` experiment is fitted by weighted nonlinear least squares
    over every measured response of every completed run (see
    :func:`_do_fit_module`).
    """
    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    completed = wb.completed_runs()
    if not completed:
        raise DoEError(f"no completed runs in {wb.path}; fill in the '{response}' column first")

    template = wb.template_name()
    if not template:
        return _do_fit_module(wb, completed)

    if template == "symbolic":
        return _do_fit_symbolic(wb, completed)

    ols = _linear_ols(wb, completed, template)
    parameter_names = ols["parameter_names"]
    sigma = ols["sigma"]
    n_p, n_obs = ols["n_p"], ols["n_obs"]
    X, y, beta = ols["X"], ols["y"], ols["beta"]
    residual_ss, std_errs, cis_arr = ols["residual_ss"], ols["std_errs"], ols["cis_arr"]
    sigma_hat_sq = ols["sigma_hat_sq"]
    rank = ols["rank"]
    estimable = rank == n_p
    warnings_out: list[str] = []
    if not estimable and n_obs >= n_p:
        n_settings = len(np.unique(np.round(ols["inputs"], 12), axis=0))
        warnings_out.append(
            f"the {n_p}-parameter model is not estimable from these runs: "
            f"{n_settings} distinct settings support only {rank} independent terms, so "
            "the estimates are one of many equally good solutions and have no "
            "standard errors. Use a smaller basis, or add runs (`discopt doe extend`)."
        )

    estimates = {name: float(beta[i]) for i, name in enumerate(parameter_names)}
    std_errors = {name: float(std_errs[i]) for i, name in enumerate(parameter_names)}
    cis = {
        name: (float(cis_arr[i, 0]), float(cis_arr[i, 1])) for i, name in enumerate(parameter_names)
    }

    # FIM = XᵀX / σ² is the design's information about the parameters
    # under the user-declared measurement_error. Use the declared σ
    # (not σ̂) so the prior FIM reflects design quality rather than fit
    # residuals. This is what a later `extend` reads back as its prior, so the
    # ridge has to stay negligible against it (see _scaled_ridge).
    information = (X.T @ X) / (sigma**2)
    cum_fim = information + _scaled_ridge(information, n_p)

    # The sigma behind the standard errors above: the residual estimate when
    # there were degrees of freedom to spare, else the declared value. Stored so
    # the declared-scale FIM can be turned back into this covariance later.
    if estimable and n_obs > n_p:
        sigma_hat, sigma_source = float(np.sqrt(sigma_hat_sq)), "residual"
    else:
        sigma_hat, sigma_source = float(sigma), "declared"
    wb.write_parameters(
        parameter_names, estimates, std_errors, cis, sigma=sigma_hat, sigma_source=sigma_source
    )
    wb.write_fim(cum_fim, parameter_names)
    coefficients, anova_rows, fit_summary = _compute_anova(
        y=y,
        beta=beta,
        residual_ss=residual_ss,
        std_errs=std_errs,
        parameter_names=parameter_names,
        n_obs=n_obs,
        n_p=n_p,
        sigma=sigma,
        inputs=ols["inputs"],
        rank=rank,
    )
    wb.write_anova(
        coefficients=coefficients,
        anova_rows=anova_rows,
        fit_summary=fit_summary,
    )
    wb.log("fit", {"n_completed": n_obs})
    wb.save()

    sign, logdet = np.linalg.slogdet(cum_fim)
    log_det_fim = float(logdet) if sign > 0 else float("-inf")

    return {
        "workbook_path": str(wb.path),
        "parameters": [
            {
                "name": name,
                "estimate": estimates[name],
                "std_error": std_errors[name],
                "ci_lower_95": cis[name][0],
                "ci_upper_95": cis[name][1],
            }
            for name in parameter_names
        ],
        "parameter_names": parameter_names,
        "n_observations": n_obs,
        "objective": residual_ss,
        "sigma": sigma_hat,
        "sigma_source": sigma_source,
        "degrees_of_freedom": max(0, n_obs - rank),
        "estimable": estimable,
        "warnings": warnings_out,
        "log_det_fim": log_det_fim,
        # The regression statistics that already go into the workbook's ANOVA
        # sheet, returned as well: whether a coefficient is distinguishable
        # from zero is the question a fit is usually asked, and a caller that
        # only sees `parameters` has to recompute the t-tests to answer it.
        # `regression_anova` decomposes the response the *model* explains,
        # which is a different question from `do_anova`'s comparison of
        # factor-level means and is the right one for a continuous design.
        "coefficients": coefficients,
        "regression_anova": anova_rows,
        "summary": dict(fit_summary),
        # A classical design has no model to search against, so `extend` is
        # rejected for it; point at the verb that does apply.
        "next_command": (
            f"discopt doe anova {wb.path}"
            if template in _CLASSICAL_TEMPLATE_NAMES and estimable
            else f"discopt doe extend {wb.path} --n N"
        ),
    }


def _compute_anova(
    *,
    y: np.ndarray,
    beta: np.ndarray,
    residual_ss: float,
    std_errs: np.ndarray,
    parameter_names: list[str],
    n_obs: int,
    n_p: int,
    sigma: float,
    inputs: np.ndarray | None = None,
    rank: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, Any]]]:
    """Compute coefficient table, ANOVA decomposition, and fit summary.

    ``inputs`` (one row of input values per observation) enables the classical
    split of the residual into **lack of fit** and **pure error** whenever some
    input settings were run more than once -- the reason replicated center
    points are in a central composite or Box-Behnken design. Pure error is the
    scatter among runs at identical settings and does not depend on the model;
    lack of fit is the rest of the residual, and its F-test against pure error
    asks whether the model's form is adequate.

    Assumes the model includes an intercept (true for every built-in
    template), so ``df_regression = n_p - 1``. Returns NaN-filled entries
    for fields that aren't statistically meaningful (e.g. when
    ``n_obs <= n_p``).

    Everything here is a function of the fitted values and their standard
    errors, never of the design matrix, which is what lets the nonlinear fit
    reuse it: for a model nonlinear in its parameters the same quantities are
    the usual asymptotic (Wald) ones, on the same footing as the confidence
    intervals that fit already reports.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist

    # ``rank`` (of the design matrix) is the number of terms the runs actually
    # determine; below ``n_p`` the regression and residual degrees of freedom
    # follow it, and the coefficient tests are undefined (their SEs are NaN).
    n_eff = n_p if rank is None else int(rank)
    dof_resid = n_obs - n_eff
    has_dof = dof_resid > 0
    nan = float("nan")

    if has_dof:
        t_crit = float(t_dist.ppf(0.975, df=dof_resid))
        t_stats = np.where(std_errs > 0, beta / np.where(std_errs > 0, std_errs, 1.0), nan)
        p_vals = np.where(
            std_errs > 0,
            2.0 * t_dist.sf(np.abs(t_stats), df=dof_resid),
            nan,
        )
    else:
        t_crit = nan
        t_stats = np.full(n_p, nan)
        p_vals = np.full(n_p, nan)

    coefficients: list[dict[str, Any]] = []
    for i, name in enumerate(parameter_names):
        se = float(std_errs[i])
        est = float(beta[i])
        ci_half = t_crit * se if has_dof and np.isfinite(se) else nan
        coefficients.append(
            {
                "name": name,
                "estimate": est,
                "std_error": se if np.isfinite(se) else nan,
                "t_statistic": float(t_stats[i]),
                "p_value": float(p_vals[i]),
                "ci_lower_95": est - ci_half if np.isfinite(ci_half) else nan,
                "ci_upper_95": est + ci_half if np.isfinite(ci_half) else nan,
            }
        )

    # Overall ANOVA (corrected total; intercept-aware).
    y_bar = float(np.mean(y))
    ss_total = float(np.sum((y - y_bar) ** 2))
    ss_residual = float(residual_ss)
    ss_regression = max(ss_total - ss_residual, 0.0)
    df_regression = max(n_eff - 1, 0)
    df_total = n_obs - 1

    if has_dof and df_regression > 0:
        ms_regression = ss_regression / df_regression
        ms_residual = ss_residual / dof_resid
        f_stat = ms_regression / ms_residual if ms_residual > 0 else nan
        p_f = float(f_dist.sf(f_stat, df_regression, dof_resid)) if np.isfinite(f_stat) else nan
    else:
        ms_regression = nan
        ms_residual = nan
        f_stat = nan
        p_f = nan

    anova_rows: list[dict[str, Any]] = [
        {
            "source": "Regression",
            "ss": ss_regression,
            "df": df_regression,
            "ms": ms_regression,
            "f_statistic": f_stat,
            "p_value": p_f,
        },
        {
            "source": "Residual",
            "ss": ss_residual,
            "df": dof_resid,
            "ms": ms_residual,
            "f_statistic": None,
            "p_value": None,
        },
        {
            "source": "Total (corrected)",
            "ss": ss_total,
            "df": df_total,
            "ms": None,
            "f_statistic": None,
            "p_value": None,
        },
    ]

    lof = _lack_of_fit(inputs, y, ss_residual, dof_resid) if has_dof else None
    if lof is not None:
        anova_rows[2:2] = lof["rows"]

    if ss_total > 0:
        r2 = 1.0 - ss_residual / ss_total
        if has_dof:
            r2_adj = 1.0 - (1.0 - r2) * (n_obs - 1) / dof_resid
        else:
            r2_adj = nan
    else:
        r2 = nan
        r2_adj = nan
    rmse = float(np.sqrt(ms_residual)) if np.isfinite(ms_residual) else nan
    sigma_hat = rmse  # RMSE is the estimated σ from residuals

    fit_summary: list[tuple[str, Any]] = [
        ("n_observations", int(n_obs)),
        ("n_parameters", int(n_p)),
        ("degrees_of_freedom", int(dof_resid) if has_dof else "n/a (under-determined)"),
        ("R_squared", float(r2)),
        ("adjusted_R_squared", float(r2_adj)),
        ("RMSE (sigma_hat)", float(sigma_hat)),
        ("sigma_declared", float(sigma)),
        ("F_statistic", float(f_stat)),
        ("F_p_value", float(p_f)),
    ]
    if lof is not None:
        fit_summary += [
            ("lack_of_fit_F", float(lof["f"])),
            ("lack_of_fit_p_value", float(lof["p"])),
            ("pure_error_df", int(lof["df_pe"])),
        ]
    return coefficients, anova_rows, fit_summary


def _input_matrix(wb: Workbook, completed: list[dict[str, Any]]) -> np.ndarray | None:
    """Input settings of the completed runs, one row each, or ``None``."""
    names = [s.name for s in wb.input_specs()]
    try:
        return np.array([[float(r[n]) for n in names] for r in completed], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None


def _lack_of_fit(
    inputs: np.ndarray | None, y: np.ndarray, ss_residual: float, dof_resid: int
) -> dict[str, Any] | None:
    """Split a residual into lack of fit and pure error, or ``None``.

    Pure error pools the scatter within groups of runs at identical input
    settings. ``None`` when there are no replicated settings (no pure error) or
    when every residual degree of freedom is pure error (nothing left to test).
    """
    from scipy.stats import f as f_dist

    if inputs is None:
        return None
    X = np.asarray(inputs, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(y) or not np.all(np.isfinite(X)):
        return None
    keys = np.round(X, 12)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    df_pe = int(np.sum(counts - 1))
    df_lof = int(dof_resid) - df_pe
    if df_pe <= 0 or df_lof <= 0:
        return None
    means = np.bincount(inverse, weights=y) / counts
    ss_pe = float(np.sum((y - means[inverse]) ** 2))
    ss_lof = max(float(ss_residual) - ss_pe, 0.0)
    ms_pe = ss_pe / df_pe
    ms_lof = ss_lof / df_lof
    if ms_pe > 0:
        f_lof = ms_lof / ms_pe
        p_lof = float(f_dist.sf(f_lof, df_lof, df_pe))
    else:
        f_lof, p_lof = float("nan"), float("nan")
    rows = [
        {
            "source": "  Lack of fit",
            "ss": ss_lof,
            "df": df_lof,
            "ms": ms_lof,
            "f_statistic": f_lof,
            "p_value": p_lof,
        },
        {
            "source": "  Pure error",
            "ss": ss_pe,
            "df": df_pe,
            "ms": ms_pe,
            "f_statistic": None,
            "p_value": None,
        },
    ]
    return {"rows": rows, "f": f_lof, "p": p_lof, "df_pe": df_pe, "df_lof": df_lof}


# The basis-row expansion moved to discopt.doe.linear_design, where it does
# double duty as the sensitivity Jacobian for jax-free optimal design. Kept
# under the original private name: the Streamlit GUI imports it from here.
_design_row = design_row


def _cmd_fit(args) -> int:
    try:
        out = do_fit({"workbook": args.workbook})
    except (DoEError, FileNotFoundError, OSError, ValueError, TypeError, RuntimeError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    for _w in out.get("warnings", []):
        print(f"warning: {_w}", file=sys.stderr)
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        print(f"fit complete: {out['workbook_path']}")
        print(f"  observations: {out['n_observations']}")
        if out.get("n_missing"):
            print(f"  missing:      {out['n_missing']} blank response cell(s), not fitted")
        print(f"  objective:    {out['objective']:.6g}")
        if out.get("reduced_chi_square") is not None and math.isfinite(out["reduced_chi_square"]):
            print(f"  reduced chi2: {out['reduced_chi_square']:.4g}")
        print(f"  log-det FIM:  {out['log_det_fim']:.6g}")
        fs = out.get("summary") or {}
        p_lof = fs.get("lack_of_fit_p_value")
        if p_lof is not None and math.isfinite(p_lof):
            print(
                f"  lack of fit:  F = {fs['lack_of_fit_F']:.4g}, p = {p_lof:.4g} "
                f"(pure error from {fs['pure_error_df']} replicate df)"
            )
        print("  parameters:")
        for p in out["parameters"]:
            est = p["estimate"]
            se = p["std_error"]
            if est is None:
                continue
            se_s = f"{se:.4g}" if se is not None and math.isfinite(se) else "n/a"
            print(f"    {p['name']:>8s} = {est:>12.6g}  ± {se_s}")
        print(f"  next:         {out['next_command']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# do_anova
# ──────────────────────────────────────────────────────────────────


def do_anova(params: dict[str, Any]) -> dict[str, Any]:
    """Run ANOVA on the completed runs of a Latin-family workbook."""
    from discopt.doe import anova_report

    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    completed = wb.completed_runs()
    if not completed:
        raise DoEError(f"no completed runs in {wb.path}; fill in '{response}' first")

    template = wb.template_name()
    template_args = wb.template_args()
    if _uses_regression_anova(wb, template):
        return _regression_anova(wb, completed, template or "", params)
    if template and template in {"latin-square", "graeco-latin", "hyper-graeco-latin"}:
        factors = list(template_args.get("levels", {}).keys())
    else:
        factors = [s.name for s in wb.input_specs()]

    include_replicate = params.get("include_replicate", False)
    if include_replicate:
        if not any(r.get("replicate") is not None for r in completed):
            raise DoEError(
                "--include-replicate was given but this workbook has no "
                "replicate column with data. Recreate the design with "
                "--replicates > 1."
            )
    interactions = params.get("interactions") or None
    # A two-level template knows its centre runs: flag them, since unflagged
    # data with one numeric factor and a categorical one cannot be told apart
    # from a 2 x 3 factorial.
    two_level_levels: dict[str, tuple[Any, Any]] | None = None
    if template in _TWO_LEVEL_TEMPLATE_NAMES:
        lv = template_args.get("levels") or {}
        two_level_levels = {f: (lv[f][0], lv[f][1]) for f in factors if f in lv}
    rows: list[dict[str, Any]] = []
    for r in completed:
        d = {f: r[f] for f in factors if f in r}
        d[response] = r[response]
        if "replicate" in r and r["replicate"] is not None:
            d["replicate"] = r["replicate"]
        if two_level_levels:
            d["is_center"] = bool(r.get("is_center")) or _is_center_row(d, two_level_levels)
        rows.append(d)

    # Two-level screening designs also get signed effect estimates. A saturated
    # design (e.g. 7 factors in an 8-run Plackett-Burman) leaves no residual
    # degrees of freedom, so there is no ANOVA F-test at all; the effects are
    # then judged against Lenth's pseudo standard error instead.
    effects: list[dict[str, Any]] | None = None
    if template in _TWO_LEVEL_TEMPLATE_NAMES:
        from discopt.doe import effects_estimates

        lv = template_args.get("levels") or {}
        levels = {f: (lv[f][0], lv[f][1]) for f in factors if f in lv}
        fx_rows = [
            dict(r, is_center=bool(src.get("is_center")) or _is_center_row(r, levels))
            for r, src in zip(rows, completed)
        ]
        effects = [
            {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in e.items()
            }
            for e in effects_estimates(
                fx_rows,
                response,
                factors=factors,
                levels=levels,
                interactions=interactions,
                blocks=["replicate"] if include_replicate else None,
            )
        ]
    try:
        table = anova_report(
            rows,
            response=response,
            factors=factors,
            interactions=interactions,
            include_replicate=include_replicate,
        )
    except ValueError as e:
        if effects is None or "no residual degrees of freedom" not in str(e):
            raise
        return {
            "workbook_path": str(wb.path),
            "response": response,
            "n_observations": len(rows),
            "grand_mean": float(np.mean([float(r[response]) for r in rows])),
            "balanced": True,
            "rows": [],
            "effects": effects,
            "summary": _effects_summary(effects, saturated=True),
        }
    return {
        "workbook_path": str(wb.path),
        "response": response,
        "effects": effects,
        "n_observations": table.n_obs,
        "grand_mean": table.grand_mean,
        "balanced": table.balanced,
        "rows": [
            {
                "source": r.source,
                "ss": r.ss,
                "df": r.df,
                "ms": r.ms,
                "f": r.f,
                "p": r.p,
            }
            for r in table.rows
        ],
        "summary": table.summary()
        + ("\n\n" + _effects_summary(effects, saturated=False) if effects else ""),
    }


def _is_center_row(row: dict[str, Any], levels: dict[str, tuple[Any, Any]]) -> bool:
    """True when every numeric factor sits at the midpoint of its two levels.

    Categorical factors have no midpoint; center runs are made at one of their
    levels, so they are not checked. At least one factor must be numeric.
    """

    def _num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    numeric = {f: lh for f, lh in levels.items() if _num(lh[0]) and _num(lh[1])}
    if not numeric:
        return False
    try:
        return all(
            abs(float(row[f]) - (float(lo) + float(hi)) / 2.0) < 1e-12
            for f, (lo, hi) in numeric.items()
        )
    except (TypeError, ValueError):
        return False


def _effects_summary(effects: list[dict[str, Any]], *, saturated: bool) -> str:
    """Fixed-width table of signed effects with their standard errors."""
    method = effects[0].get("method", "") if effects else ""
    head = "Effects (high minus low)"
    if saturated:
        head += " -- saturated design: no residual df, so no ANOVA F-test"
    if method == "lenth":
        head += "; SEs are Lenth's pseudo standard error"
    lines = [head, f"{'Effect':<20s} {'estimate':>10s} {'SE':>9s} {'t':>8s} {'p':>9s}"]
    for e in effects:
        p = e.get("p")
        p_str = "      ---" if p is None or p != p else f"{p:9.4g}"
        lines.append(
            f"{str(e.get('effect_name', e.get('factor'))):<20s} {e['effect']:10.4f} "
            f"{e['se']:9.4f} {e['t']:8.3f} {p_str}"
        )
    return "\n".join(lines)


def _uses_regression_anova(wb: Workbook, template: str | None) -> bool:
    """True for a continuous-factor template with an intercept model.

    Comparing factor-level means (the Latin/factorial analysis) is the wrong
    question for a central composite, Box-Behnken, definitive screening or
    other regression design: its factors are continuous, its levels are
    unequally replicated (center points), and in a CCD the axial levels are
    confounded with each other, so that analysis is unbalanced at best and
    fails outright at worst. These get the regression ANOVA of their model.
    """
    from discopt.doe.templates import COMBINATORIAL_TEMPLATES, SYMBOLIC_TEMPLATE

    if not template or template in COMBINATORIAL_TEMPLATES:
        return False
    if template in (SYMBOLIC_TEMPLATE, "optimize"):
        return False
    try:
        return "b0" in wb.parameter_names()
    except ValueError:
        return False


def _term_label(powers: Mapping[str, int]) -> str:
    """``{"T": 1, "P": 1}`` -> ``"T*P"``; ``{"T": 2}`` -> ``"T^2"``."""
    return "*".join(f"{n}^{p}" if p > 1 else n for n, p in powers.items()) or "1"


def _regression_anova(
    wb: Workbook, completed: list[dict[str, Any]], template: str, params: dict[str, Any]
) -> dict[str, Any]:
    """ANOVA of a regression template's model: one row per model term.

    Each term's SS is its *adjusted* (partial, Type III) sum of squares,
    ``b_j^2 / [(XᵀX)⁻¹]_jj`` -- the increase in residual SS from dropping that
    term alone -- so the rows do not depend on term order (they need not sum to
    the model SS when terms are correlated). The residual is split into lack of
    fit and pure error when settings were replicated (e.g. center points).
    """
    from scipy.stats import f as f_dist

    from discopt.doe.anova import AnovaEffect, AnovaTable
    from discopt.doe.linear_design import basis_terms

    if params.get("interactions"):
        raise DoEError(
            f"--interaction does not apply to a '{template}' workbook: its model terms "
            "(including interactions) come from the template's regression basis"
        )
    if params.get("include_replicate"):
        raise DoEError(f"--include-replicate does not apply to a '{template}' workbook")

    response = wb.response_name()
    ols = _linear_ols(wb, completed, template)
    names, beta, y = ols["parameter_names"], ols["beta"], ols["y"]
    n_obs, n_p = ols["n_obs"], ols["n_p"]
    if n_obs <= n_p:
        raise DoEError(
            f"{n_obs} completed run(s) for {n_p} model parameters leaves no residual "
            "degrees of freedom for an ANOVA; complete more runs"
        )
    rank = int(np.linalg.matrix_rank(ols["X"]))
    if rank < n_p:
        n_settings = len(np.unique(np.round(ols["inputs"], 12), axis=0))
        raise DoEError(
            f"the {n_p}-parameter '{wb.template_args().get('basis', 'model')}' basis is not "
            f"estimable from these runs ({n_settings} distinct settings support only "
            f"{rank} independent terms); use a smaller basis or add runs "
            "(e.g. `discopt doe extend`)"
        )
    coefficients, anova_rows, fit_summary = _compute_anova(
        y=y,
        beta=beta,
        residual_ss=ols["residual_ss"],
        std_errs=ols["std_errs"],
        parameter_names=names,
        n_obs=n_obs,
        n_p=n_p,
        sigma=ols["sigma"],
        inputs=ols["inputs"],
    )
    dof_resid = n_obs - n_p
    ms_resid = ols["residual_ss"] / dof_resid
    try:
        labels = [
            _term_label(t)
            for t in basis_terms(template, wb.template_args(), names, ols["input_names"])
        ]
    except (ValueError, KeyError):
        labels = list(names)

    rows: list[dict[str, Any]] = []
    model = anova_rows[0]
    rows.append(
        {
            "source": "Model",
            "ss": model["ss"],
            "df": model["df"],
            "ms": model["ms"],
            "f": model["f_statistic"],
            "p": model["p_value"],
        }
    )
    xtx_inv = ols["xtx_inv"]
    for j, name in enumerate(names):
        if name == "b0":
            continue
        d = float(xtx_inv[j, j])
        ss = float(beta[j] ** 2 / d) if d > 0 and np.isfinite(d) else float("nan")
        f = ss / ms_resid if ms_resid > 0 and np.isfinite(ss) else float("nan")
        p = float(f_dist.sf(f, 1, dof_resid)) if np.isfinite(f) else float("nan")
        rows.append(
            {"source": f"  {labels[j]}", "term": name, "ss": ss, "df": 1, "ms": ss, "f": f, "p": p}
        )
    for r in anova_rows[1:]:
        rows.append(
            {
                "source": "Error" if r["source"] == "Residual" else r["source"],
                "ss": r["ss"],
                "df": r["df"],
                "ms": r["ms"],
                "f": r["f_statistic"],
                "p": r["p_value"],
            }
        )

    def _opt(v: Any) -> float | None:
        return None if v is None or not np.isfinite(v) else float(v)

    table = AnovaTable(
        rows=[
            AnovaEffect(
                r["source"].replace("Total (corrected)", "Total"),
                float(r["ss"]),
                int(r["df"]),
                float(r["ms"]) if r["ms"] is not None and np.isfinite(r["ms"]) else 0.0,
                _opt(r["f"]),
                _opt(r["p"]),
            )
            for r in rows
        ],
        response=response,
        n_obs=n_obs,
        grand_mean=float(np.mean(y)),
        balanced=True,
    )
    summary = dict(fit_summary)
    notes = [
        "",
        f"Regression ANOVA of the '{template}' model (adjusted SS per term).",
        f"R^2 = {summary['R_squared']:.4f}, adjusted R^2 = "
        f"{summary['adjusted_R_squared']:.4f}, RMSE = {summary['RMSE (sigma_hat)']:.4g}",
    ]
    if "lack_of_fit_p_value" in summary:
        p_lof = summary["lack_of_fit_p_value"]
        if np.isfinite(p_lof):
            verdict = (
                "significant lack of fit at the 5% level -- the model form may be "
                "inadequate (consider more terms or a transform)"
                if p_lof < 0.05
                else "no significant lack of fit at the 5% level"
            )
            notes.append(f"Lack of fit p = {p_lof:.4g}: {verdict}.")
    elif len(np.unique(np.round(ols["inputs"], 12), axis=0)) < n_obs:
        notes.append(
            "No lack-of-fit test: every residual degree of freedom is pure error "
            "(the model has as many terms as there are distinct settings)."
        )
    else:
        notes.append(
            "No lack-of-fit test: no input settings were replicated (add replicated "
            "center points to get a pure-error estimate)."
        )
    return {
        "workbook_path": str(wb.path),
        "response": response,
        "analysis": "regression",
        "effects": None,
        "n_observations": n_obs,
        "grand_mean": float(np.mean(y)),
        "balanced": True,
        "rows": rows,
        "coefficients": coefficients,
        "fit_summary": summary,
        "summary": table.summary() + "\n" + "\n".join(notes),
    }


def _cmd_anova(args) -> int:
    try:
        interactions: list[tuple[str, ...]] | None = None
        if args.interaction:
            interactions = [tuple(s.split(":")) for s in args.interaction]
        out = do_anova(
            {
                "workbook": args.workbook,
                "include_replicate": args.include_replicate,
                "interactions": interactions,
            }
        )
    except (DoEError, FileNotFoundError, OSError, ValueError, TypeError, RuntimeError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(_dump_json({k: v for k, v in out.items() if k != "summary"}, indent=2))
    else:
        print(f"ANOVA on {out['workbook_path']}")
        print(f"  response:   {out['response']}")
        print(f"  n:          {out['n_observations']}")
        print(f"  grand mean: {out['grand_mean']:.6g}")
        print()
        print(out["summary"])
    return 0


# ──────────────────────────────────────────────────────────────────
# do_extend
# ──────────────────────────────────────────────────────────────────


def do_extend(params: ExtendParams) -> dict[str, Any]:
    from discopt.doe.templates import CLASSICAL_TEMPLATES

    strategy = str(params.batch_strategy or "greedy")
    if strategy not in _BATCH_STRATEGIES:
        raise DoEError(
            f"unknown batch strategy {strategy!r}; choose from {list(_BATCH_STRATEGIES)}"
        )
    wb = Workbook.open(params.workbook)
    # A targeted symbolic campaign is designed through its Experiment, where
    # every strategy is available.
    symbolic_targeted = wb.template_name() == "symbolic" and bool(
        (params.target or wb.target_spec()) and not params.clear_target
    )
    if strategy != "greedy" and (
        (wb.template_name() == "symbolic" and not symbolic_targeted)
        or wb.template_name() in CLASSICAL_TEMPLATES
    ):
        raise DoEError(
            f"batch strategy {strategy!r} is only available for models designed through an "
            f"Experiment (the built-in parametric templates and --module); template "
            f"{wb.template_name()!r} designs its batches greedily."
        )
    if params.target and params.clear_target:
        raise DoEError("give a new target or clear_target (--clear-target), not both")
    if params.clear_target:
        wb.set_target_spec(None)
    if wb.template_name() == "symbolic":
        return _do_extend_symbolic(wb, params)
    if wb.template_name() in CLASSICAL_TEMPLATES:
        return _do_extend_classical(wb, params)
    if wb.template_name() in FOLDOVER_TEMPLATES:
        raise DoEError(
            f"workbook uses screening template {wb.template_name()!r}; adding 'optimal' "
            "runs does not undo its aliasing. Use `discopt doe foldover` to append the "
            "mirror-image runs, or `discopt doe anova` to analyse what you have."
        )

    from discopt.doe import batch_optimal_experiment

    experiment, parameter_names = wb.rebuild_experiment()
    input_specs = wb.input_specs()
    input_names = [s.name for s in input_specs]
    bounds = {s.name: (s.lb, s.ub) for s in input_specs}
    completed = wb.completed_runs()

    # Offset the search seed by the batch index so two consecutive extends
    # (with no new data) don't recommend the identical set of points.
    batch_idx = wb.next_batch_index()
    design_seed = wb.seed() + batch_idx

    warnings_out: list[str] = []
    pending = wb.pending_runs()
    if pending:
        warnings_out.append(
            f"{len(pending)} run(s) are still pending; they contribute no "
            "information to the prior FIM, so the recommended batch may overlap "
            "them. Fill in their responses before extending for best results."
        )

    param_values = _param_values_for_design(parameter_names, wb)

    template = wb.template_name()
    _check_prediction_criterion(wb.criterion(), template)
    target_spec = _extend_target_spec(wb, params, parameter_names, wb.response_names(), input_names)
    target = None
    replicates = None
    if target_spec:
        target = _target_from_spec(target_spec)
        # Replicating a completed run is always a candidate: while the data
        # leave the FIM singular it can be the only run that helps a target.
        replicates = [
            {n: float(r[n]) for n in input_names} for r in completed if _all_inputs(r, input_names)
        ] or None

    cached = wb.read_fim()
    ridge: np.ndarray | None = None
    if cached is not None and cached[1] == parameter_names:
        # Written by `fit` at the estimates the design below is centred on.
        prior_fim = cached[0]
    elif not template and completed and _any_blank_response(wb, completed):
        # Not fitted yet and some responses are missing: count only what was
        # measured (a per-run compute_fim would count every response).
        runs, _ = _module_campaign_runs(wb, completed)
        try:
            info = _measured_fim(experiment, runs, param_values)
        except (ValueError, RuntimeError) as e:
            raise DoEError(f"could not evaluate the information in the completed runs: {e}") from e
        ridge = _scaled_ridge(info, len(parameter_names))
        prior_fim = info + ridge
    else:
        info = _cumulative_fim_from_completed(
            experiment, parameter_names, completed, input_names, param_values, ridge=False
        )
        ridge = _scaled_ridge(info, len(parameter_names))
        prior_fim = info + ridge

    template_args = wb.template_args()
    mixture_total = template_args.get("mixture_total")
    eq_cons, proj = _mixture_constraints(
        template,
        [(s.name, s.lb, s.ub) for s in input_specs],
        float(mixture_total) if mixture_total is not None else None,
    )
    if params.n == 1:
        from discopt.doe import optimal_experiment

        single = optimal_experiment(
            experiment,
            param_values,
            bounds,
            criterion=wb.criterion(),
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=design_seed,
            target=target,
            initial_designs=replicates,
        )
        designs = [single.design]
        criterion_value = float(single.criterion_value)
        precision = _target_precision_without_ridge(single.target, single.fim, ridge)
    else:
        batch = batch_optimal_experiment(
            experiment,
            param_values,
            bounds,
            n_experiments=params.n,
            criterion=wb.criterion(),
            strategy=strategy,
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=design_seed,
            target=target,
            initial_designs=replicates,
        )
        designs = list(batch.designs)
        criterion_value = float(batch.criterion_value)
        precision = _target_precision_without_ridge(batch.target, batch.joint_fim, ridge)

    new_ids = wb.append_runs(batch_idx, designs)
    wb.log(
        "extend",
        {"n": params.n, "batch": batch_idx, "batch_strategy": strategy, "target": target_spec},
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "batch": batch_idx,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": wb.criterion(),
        "criterion_value": criterion_value,
        "batch_strategy": strategy,
        "parameter_names": parameter_names,
        "design_parameters": param_values,
        "target": target_spec,
        "target_precision": _precision_out(precision) if precision is not None else None,
        "warnings": warnings_out,
        "next_command": f"discopt doe status {wb.path}",
    }


def _extend_target_spec(
    wb: Workbook,
    params: ExtendParams,
    parameter_names: Sequence[str],
    response_names: Sequence[str],
    input_names: Sequence[str],
) -> dict[str, Any] | None:
    """The target this extend designs for: a new one (stored for next time) or the stored one."""
    if params.clear_target:
        return None
    if params.target:
        spec = _check_target_spec(params.target, parameter_names, response_names, input_names)
        wb.set_target_spec(spec)
        return spec
    stored = wb.target_spec()
    if not stored:
        return None
    return _check_target_spec(stored, parameter_names, response_names, input_names)


def _all_inputs(row: Mapping[str, Any], input_names: Sequence[str]) -> bool:
    try:
        return all(math.isfinite(float(row[n])) for n in input_names)
    except (KeyError, TypeError, ValueError):
        return False


def _any_blank_response(wb: Workbook, rows: list[dict[str, Any]]) -> bool:
    names = wb.response_names()
    return len(names) > 1 and any(
        r.get(n) is None or (isinstance(r.get(n), str) and not r.get(n).strip())
        for r in rows
        for n in names
    )


def _do_extend_classical(wb: Workbook, params: ExtendParams) -> dict[str, Any]:
    """Augment a classical design with D-optimal runs for its own regression basis.

    A central composite, Box-Behnken, definitive screening or Latin hypercube
    records a regression basis (``--basis``) rather than a model, so there is no
    ``Experiment`` to rebuild; the basis is enough. Its information matrix does
    not depend on the responses, so *every* run already in the workbook --
    measured or still pending -- counts as prior information, and the new runs
    go where the existing design leaves the coefficients least certain.
    """
    from discopt.doe.linear_design import design_row, linear_batch_design

    template = wb.template_name() or ""
    args = wb.template_args()
    names = list(wb.parameter_names())
    specs = wb.input_specs()
    input_names = [s.name for s in specs]
    sigma = float(wb.measurement_error() or 1.0)
    rows = wb.all_runs()
    batch_idx = wb.next_batch_index()

    X = np.array([design_row(template, args, names, input_names, row) for row in rows], dtype=float)
    prior = (X.T @ X) / sigma**2 if len(rows) else None
    criterion = wb.criterion()
    if criterion not in _CRITERION_CHOICES:
        # Classical workbooks record no optimality criterion of their own.
        criterion = "determinant"
    target_spec = _extend_target_spec(wb, params, names, wb.response_names(), input_names)
    target = None
    if target_spec:
        from discopt.doe.linear_design import _basis_evaluator

        target = _basis_target_or_error(
            target_spec,
            names,
            input_names,
            _basis_evaluator(template, args, names, input_names),
            wb.response_name(),
        )
        if wb.criterion() not in _CRITERION_CHOICES:
            # Augmenting for a target: the weighted variance sum unless the
            # workbook names a criterion.
            criterion = "trace"

    result = linear_batch_design(
        template,
        int(params.n),
        parameter_names=names,
        input_names=input_names,
        design_bounds={s.name: (s.lb, s.ub) for s in specs},
        template_args=args,
        measurement_error=sigma,
        criterion=criterion,
        prior_fim=prior,
        n_starts=int(params.n_starts),
        seed=wb.seed() + batch_idx,
        target=target,
    )
    designs = [{nm: float(d[nm]) for nm in input_names} for d in result.designs]
    new_ids = wb.append_runs(batch_idx, designs)
    wb.log(
        "extend",
        {"n": params.n, "batch": batch_idx, "mode": "classical-augment", "target": target_spec},
    )
    wb.save()

    # Per-parameter geometric precision gain: how much the confidence intervals
    # shrink on average, from the change in log det of the information.
    gain = None
    if prior is not None:
        s0, ld0 = np.linalg.slogdet(prior)
        s1, ld1 = np.linalg.slogdet(np.asarray(result.joint_fim, dtype=float))
        if s0 > 0 and s1 > 0:
            gain = float(np.exp((ld1 - ld0) / (2 * len(names))))

    return {
        "workbook_path": str(wb.path),
        "batch": batch_idx,
        "new_run_ids": new_ids,
        "designs": [{"run_id": rid, **d} for rid, d in zip(new_ids, designs)],
        "criterion": result.criterion,
        "criterion_value": float(result.criterion_value),
        "parameter_names": names,
        "precision_gain": gain,
        "target": target_spec,
        "target_precision": (
            _precision_out(target.precision(np.asarray(result.joint_fim, dtype=float)))
            if target is not None
            else None
        ),
        "warnings": [],
        "next_command": f"discopt doe status {wb.path}",
    }


#: Two-level screening templates that `foldover` can mirror.
FOLDOVER_TEMPLATES = frozenset({"factorial-2level", "fractional-factorial", "plackett-burman"})


def do_foldover(params: FoldoverParams) -> dict[str, Any]:
    """Append the fold-over of a two-level screening campaign.

    The full fold-over (``factor=None``) reverses every factor's level in each
    original corner run: in a resolution III design it separates every main
    effect from every two-factor interaction. Reversing one factor clears that
    factor's main effect and all its two-factor interactions. Centre runs are not
    mirrored (their mirror is themselves). The new runs form batch 2, in a fresh
    seeded random order, so a shift between the two halves shows up as a batch
    difference -- run them as a block.

    The workbook counterpart of :func:`discopt.doe.fold_over`.
    """
    wb = Workbook.open(params.workbook)
    template = wb.template_name() or ""
    if template not in FOLDOVER_TEMPLATES:
        raise DoEError(
            f"fold-over applies to two-level screening designs "
            f"({', '.join(sorted(FOLDOVER_TEMPLATES))}); this workbook uses {template!r}"
        )
    rows = wb.all_runs()
    if any(int(r.get("batch") or 1) > 1 for r in rows):
        raise DoEError(
            "this campaign has already been followed up; a second fold-over only "
            "duplicates runs. Analyse what you have with `discopt doe anova`."
        )
    input_names = [s.name for s in wb.input_specs()]
    if params.factor is not None and params.factor not in input_names:
        raise DoEError(f"unknown factor {params.factor!r}; known: {input_names}")

    def extremes(name: str) -> tuple[Any, Any] | None:
        seen: list[Any] = []
        for r in rows:
            v = r.get(name)
            if v is not None and v not in seen:
                seen.append(v)
        numeric = [v for v in seen if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and len(numeric) == len(seen):
            lo, hi = min(numeric), max(numeric)
            return (lo, hi) if lo != hi else None
        return (seen[0], seen[1]) if len(seen) == 2 else None

    levels = {name: extremes(name) for name in input_names}
    flipped = [params.factor] if params.factor is not None else input_names

    new: list[dict[str, Any]] = []
    for r in rows:
        point = {name: r.get(name) for name in input_names}
        # Only corner runs are mirrored: every factor must sit at one of its two
        # extreme levels.
        if any(levels[n] is None or point[n] not in levels[n] for n in input_names):
            continue
        for name in flipped:
            lo, hi = levels[name]  # type: ignore[misc]
            point[name] = hi if point[name] == lo else lo
        # Each mirrored run keeps its source's replicate index; a blank one
        # made `anova --include-replicate` refuse the whole workbook.
        if r.get("replicate") is not None:
            point["replicate"] = r["replicate"]
        new.append(point)
    if not new:
        raise DoEError("no two-level corner runs to mirror in this workbook")

    import random

    batch_idx = wb.next_batch_index()
    random.Random(wb.seed() + batch_idx).shuffle(new)
    new_ids = wb.append_runs(batch_idx, new)
    wb.log("foldover", {"factor": params.factor, "batch": batch_idx})
    wb.save()

    resolution = None
    try:
        from discopt.doe.aliasing import alias_structure

        resolution = alias_structure(
            [dict(r) for r in Workbook.open(wb.path).all_runs()], input_names
        ).resolution
    except Exception:  # noqa: BLE001 -- best-effort report, never a failure
        resolution = None

    return {
        "workbook_path": str(wb.path),
        "batch": batch_idx,
        "factor": params.factor,
        "new_run_ids": new_ids,
        "designs": [{"run_id": rid, **d} for rid, d in zip(new_ids, new)],
        "resolution": resolution,
        "warnings": [
            "Run the fold-over as its own block: any shift since the first half then "
            "shows up as a difference between batches rather than in the effects."
        ],
        "next_command": f"discopt doe anova {wb.path}",
    }


def _cmd_foldover(args) -> int:
    try:
        out = do_foldover(FoldoverParams(workbook=Path(args.workbook), factor=args.factor))
    except (DoEError, FileNotFoundError, OSError, ValueError, TypeError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    for _w in out.get("warnings", []):
        print(f"warning: {_w}", file=sys.stderr)
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        what = f"factor {out['factor']!r} reversed" if out["factor"] else "every factor reversed"
        new_ids = out["new_run_ids"]
        print(f"folded workbook: {out['workbook_path']}")
        print(f"  batch:        {out['batch']}  (run_ids {min(new_ids)}-{max(new_ids)}, {what})")
        if out["resolution"] is not None:
            print(f"  resolution:   {out['resolution']} (original and fold-over together)")
        print(f"  next:         {out['next_command']}")
    return 0


def _cmd_extend(args) -> int:
    try:
        out = do_extend(
            ExtendParams(
                workbook=Path(args.workbook),
                n=int(args.n),
                n_starts=int(args.n_starts),
                batch_strategy=str(getattr(args, "batch_strategy", "greedy") or "greedy"),
                target=_target_spec_from_args(args),
                clear_target=bool(getattr(args, "clear_target", False)),
            )
        )
    except (DoEError, FileNotFoundError, OSError, ValueError, TypeError, RuntimeError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    for _w in out.get("warnings", []):
        print(f"warning: {_w}", file=sys.stderr)
    if args.json:
        print(_dump_json(out, indent=2))
    else:
        print(f"extended workbook: {out['workbook_path']}")
        new_ids = out["new_run_ids"]
        print(f"  batch:        {out['batch']}  (run_ids {min(new_ids)}-{max(new_ids)})")
        print(f"  criterion:    {out['criterion']} = {out['criterion_value']:.6g}")
        if out.get("target"):
            print(f"  target:       {_describe_target(out['target'])}")
            _print_target_precision(out, "predicted target precision after these runs:")
        if out.get("batch_strategy") and out["batch_strategy"] != "greedy":
            print(f"  strategy:     {out['batch_strategy']}")
        if out.get("precision_gain"):
            shrink = (1 - 1 / out["precision_gain"]) * 100
            print(f"  precision:    confidence intervals ~{shrink:.0f}% narrower on average")
        print("  new runs:")
        if out["designs"]:
            keys = [k for k in out["designs"][0].keys() if k != "run_id"]
            print("    run_id  " + "  ".join(f"{k:>10s}" for k in keys))
            for row in out["designs"]:
                cells = [f"{row[k]:>10.4f}" for k in keys]
                print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
        print(f"  next:         {out['next_command']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# optimize
# ──────────────────────────────────────────────────────────────────


def _parse_kv_float_eq(s: str) -> tuple[str, float]:
    """Parse a ``key=value`` string with a float value."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {s!r}")
    k, _, v = s.partition("=")
    try:
        return k.strip(), float(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"value for {k!r} is not a float: {v!r}") from e


def _cmd_optimize(args) -> int:
    try:
        acq_kwargs: dict[str, float] | None = None
        if getattr(args, "acquisition_kwarg", None):
            acq_kwargs = dict(args.acquisition_kwarg)
        custom_kwargs: dict[str, Any] | None = None
        if getattr(args, "surrogate_kwarg", None):
            custom_kwargs = {k: v for k, v in args.surrogate_kwarg}
        out = do_optimize(
            OptimizeParams(
                workbook=Path(args.workbook),
                criterion=args.criterion,
                surrogate=args.surrogate,
                acquisition=args.acquisition,
                batch_size=int(args.batch_size),
                n_candidates=int(args.n_candidates),
                seed=args.seed,
                acquisition_kwargs=acq_kwargs,
                custom_surrogate_path=args.custom_surrogate,
                custom_surrogate_kwargs=custom_kwargs,
            )
        )
    except (DoEError, FileNotFoundError, OSError, ValueError, TypeError, RuntimeError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    for _w in out.get("warnings", []):
        print(f"warning: {_w}", file=sys.stderr)
    if args.json:
        print(_dump_json(out, indent=2, default=str))
    else:
        print(f"optimized workbook: {out['workbook_path']}")
        print(
            f"  criterion:    {out['criterion']}   acquisition: {out['acquisition']}   "
            f"surrogate: {out['surrogate']} ({out['surrogate_mode']})"
        )
        print(f"  completed:    {out['n_completed']} runs")
        if out["incumbent_x"] is not None:
            x_str = ", ".join(f"{k}={v:.4g}" for k, v in out["incumbent_x"].items())
            print(f"  incumbent:    y={out['incumbent_y']:.6g} at ({x_str})")
        new_ids = out["new_run_ids"]
        if new_ids:
            print(f"  batch:        {len(new_ids)}  (run_ids {min(new_ids)}-{max(new_ids)})")
            print("  new runs:")
            keys = [k for k in out["next_designs"][0].keys() if k != "run_id"]
            print("    run_id  " + "  ".join(f"{k:>10s}" for k in keys))
            for row in out["next_designs"]:
                cells = [
                    f"{row[k]:>10.4g}" if isinstance(row[k], (int, float)) else f"{row[k]:>10s}"
                    for k in keys
                ]
                print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
    return 0


# ──────────────────────────────────────────────────────────────────
# gui
# ──────────────────────────────────────────────────────────────────


def _cmd_gui(args) -> int:
    from discopt.doe.gui.launcher import launch

    return launch(
        workbook=args.workbook,
        port=args.port,
        open_browser=not args.no_browser,
    )


# ──────────────────────────────────────────────────────────────────
# error formatting
# ──────────────────────────────────────────────────────────────────


def _sanitize_json(obj: Any) -> Any:
    """Replace non-finite floats (NaN, +/-inf) with None, recursively.

    ``json.dumps`` emits the literals ``NaN``/``Infinity`` by default, which
    are invalid JSON that ``jq`` and strict parsers reject. DoE outputs contain
    NaN criterion values (factorial/latin `new`) and -inf log-dets, so sanitize
    before dumping.
    """
    import math as _math

    if isinstance(obj, float):
        return obj if _math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _dump_json(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` that never emits NaN/Infinity (invalid JSON)."""
    return json.dumps(_sanitize_json(obj), allow_nan=False, **kwargs)


def _fail(args, msg: str, *, workbook_path: str | None = None) -> int:
    if getattr(args, "json", False):
        payload: dict[str, Any] = {"error": msg}
        if workbook_path is not None:
            payload["workbook_path"] = workbook_path
        print(_dump_json(payload, indent=2), file=sys.stdout)
    else:
        print(f"error: {msg}", file=sys.stderr)
    return 1


# ──────────────────────────────────────────────────────────────────
# argparse registration
# ──────────────────────────────────────────────────────────────────


def add_subparser(subparsers) -> None:
    """Register the ``doe`` subcommand on the top-level ``discopt`` parser."""
    p = subparsers.add_parser(
        "doe",
        help="Design of experiments with Excel-workbook campaigns",
        description=(
            "Optimal experimental design. Verbs: templates, new, status, "
            "fit, anova, extend, foldover, optimize, gui. Each accepts --json for "
            "machine-readable output."
        ),
    )
    doe_sub = p.add_subparsers(dest="doe_cmd", metavar="<verb>", required=True)

    # --- templates ---
    p_tmpl = doe_sub.add_parser("templates", help="List built-in experiment templates.")
    _add_json(p_tmpl)
    p_tmpl.set_defaults(doe_func=_cmd_templates)

    # --- new ---
    p_new = doe_sub.add_parser("new", help="Create a workbook with N optimal initial runs.")
    new_sub = p_new.add_subparsers(dest="template", metavar="<template>", required=True)

    from discopt.doe.templates import CLASSICAL_TEMPLATES, COMBINATORIAL_TEMPLATES

    for tmpl in TEMPLATE_NAMES:
        sp = new_sub.add_parser(tmpl, help=_TEMPLATE_DESCRIPTIONS[tmpl])
        if tmpl in _TWO_LEVEL_TEMPLATE_NAMES:
            sp.add_argument(
                "--factor",
                action="append",
                type=_parse_factor_pair,
                required=True,
                help="Factor as NAME:LOW:HIGH (repeatable; LOW/HIGH numeric or string).",
            )
            sp.add_argument(
                "--center-points",
                type=int,
                default=0,
                help=(
                    "Center-point runs per replicate (default 0); with categorical "
                    "factors, made at each combination of their levels."
                ),
            )
            sp.add_argument(
                "--replicates",
                type=int,
                default=1,
                help="Whole-design replications (default 1).",
            )
            if tmpl in ("fractional-factorial", "plackett-burman"):
                sp.add_argument(
                    "--runs",
                    type=int,
                    default=None,
                    help=(
                        "Runs per replicate (before centre points). Plackett-Burman: a "
                        "multiple of 4 (default: smallest above the factor count). "
                        "Fractional factorial: a power of 2, searched by MILP."
                    ),
                )
            if tmpl == "fractional-factorial":
                sp.add_argument(
                    "--generator",
                    dest="generators",
                    action="append",
                    default=None,
                    metavar="'D=ABC'",
                    help=(
                        "Generator for an added factor, written in the factor names "
                        "(e.g. 'D=ABC', or 'cat=temp*time' for longer names); "
                        "repeatable. No solver needed."
                    ),
                )
                sp.add_argument(
                    "--resolution",
                    type=int,
                    default=3,
                    help="Minimum resolution for the --runs search (default 3).",
                )
        elif tmpl in COMBINATORIAL_TEMPLATES:
            sp.add_argument(
                "--levels",
                action="append",
                type=_parse_levels_spec,
                required=True,
                help="Factor levels as NAME:L1,L2,... (repeatable).",
            )
            sp.add_argument(
                "--replicates",
                type=int,
                default=1,
                help="Whole-design replications (default 1).",
            )
        else:
            sp.add_argument(
                "--input",
                action="append",
                type=_parse_input_spec,
                required=True,
                help="Design factor as NAME:LB:UB (repeatable).",
            )
        if tmpl == "symbolic":
            sp.add_argument(
                "--expr",
                required=True,
                metavar="EXPRESSION",
                help=(
                    "Response formula over your --param and --input names, e.g. "
                    "'k0 * exp(-Ea / (8.314 * T))'."
                ),
            )
            sp.add_argument(
                "--param",
                dest="params",
                action="append",
                type=_parse_kv_float,
                required=True,
                metavar="NAME=VALUE",
                help=(
                    "Unknown parameter and its nominal value (repeatable). The order "
                    "given fixes the parameter order; the values centre the design, "
                    "which for a nonlinear model is only locally optimal."
                ),
            )
        if tmpl in CLASSICAL_TEMPLATES:
            sp.add_argument(
                "--basis",
                default=(
                    "linear" if tmpl in ("latin-hypercube", "definitive-screening") else "quadratic"
                ),
                choices=("linear", "quadratic"),
                help=(
                    "Regression model `discopt doe fit` will estimate from the "
                    "completed runs. Defaults to 'quadratic' for the response-surface "
                    "designs and 'linear' for a Latin hypercube or definitive screening."
                ),
            )
        if tmpl == "latin-hypercube":
            sp.add_argument(
                "--optimize",
                dest="lhs_optimize",
                default="discrepancy",
                choices=("discrepancy", "maximin", "none"),
                help=(
                    "Space-filling criterion: centered discrepancy (default), maximin "
                    "distance, or none (a plain random Latin hypercube)."
                ),
            )
        if tmpl == "definitive-screening":
            sp.add_argument(
                "--fake-factors",
                type=int,
                default=0,
                help="Extra inert columns, for more runs and a better error estimate.",
            )
            sp.add_argument(
                "--center-points",
                type=int,
                default=0,
                help="Centre runs (default 1).",
            )
        if tmpl in ("central-composite", "box-behnken"):
            sp.add_argument(
                "--center-points",
                type=int,
                default=0,
                help=(
                    "Replicated centre runs, for the pure-error estimate "
                    "(default 4 for central-composite, 3 for box-behnken)."
                ),
            )
        if tmpl == "central-composite":
            sp.add_argument(
                "--alpha",
                default="rotatable",
                help=("Axial distance in coded units: 'rotatable' (default), 'face', or a number."),
            )
            sp.add_argument(
                "--outside-bounds",
                action="store_true",
                help=(
                    "Use the textbook scaling where LB/UB are the factorial corners "
                    "and the axial points fall outside them. By default the axial "
                    "points land on LB/UB so every run stays within range."
                ),
            )
        if tmpl == "polynomial-1d":
            sp.add_argument("--degree", type=int, required=True, help="Polynomial degree (>= 1).")
        if tmpl == "optimize":
            sp.add_argument(
                "--optimize-criterion",
                default="maximize",
                choices=("maximize", "minimize"),
                help="Active-learning direction (default 'maximize').",
            )
            sp.add_argument(
                "--optimize-surrogate",
                default="gp",
                help="Surrogate preset (default 'gp'); see SURROGATE_PRESETS.",
            )
            sp.add_argument(
                "--optimize-acquisition",
                default="expected_improvement",
                help="Acquisition (default 'expected_improvement').",
            )
        if tmpl.startswith("scheffe-"):
            sp.add_argument(
                "--mixture-total",
                type=float,
                default=1.0,
                help="Required sum of the component values (default 1.0).",
            )
        _add_common_new_options(sp)
        # --n applies to parametric + optimize (initial batch) + latin-hypercube,
        # not to combinatorial designs (run count = levels/factors x replicates)
        # nor to central-composite / box-behnken, whose run count follows from
        # the factor count and centre points. The optimize template needs >= 2
        # seed points to fit a surrogate.
        if tmpl not in COMBINATORIAL_TEMPLATES and tmpl not in (
            "central-composite",
            "box-behnken",
            "definitive-screening",
        ):
            _add_run_count_option(
                sp, default=4 if tmpl == "optimize" else (8 if tmpl == "latin-hypercube" else 1)
            )
        # --criterion / --n-starts govern the parametric D-optimal search only.
        if (
            tmpl not in COMBINATORIAL_TEMPLATES
            and tmpl not in CLASSICAL_TEMPLATES
            and tmpl != "optimize"
        ):
            _add_design_search_options(sp)
        sp.set_defaults(doe_func=_cmd_new, _is_module=False, bounds=None, params=None, module=None)

    # Escape-hatch: --module
    p_module = new_sub.add_parser(
        "module",
        help="Use a user-defined Experiment loaded via 'pkg.mod:callable'.",
    )
    p_module.add_argument(
        "--module",
        required=True,
        help="Spec of the form 'pkg.mod:callable' returning an Experiment.",
    )
    p_module.add_argument(
        "--bounds",
        action="append",
        type=_parse_input_spec,
        required=True,
        help="Design factor as NAME:LB:UB (repeatable).",
    )
    p_module.add_argument(
        "--params",
        action="append",
        type=_parse_kv_float,
        required=True,
        help="Prior parameter value as NAME=VALUE (repeatable).",
    )
    p_module.add_argument(
        "--sigma",
        action="append",
        type=_parse_kv_float,
        default=None,
        metavar="RESPONSE=VALUE",
        help=(
            "Override one response's measurement standard deviation (repeatable). "
            "Response names and their default errors come from the model; the "
            "runs sheet gets one column per response."
        ),
    )
    _add_common_new_options(p_module)
    _add_run_count_option(p_module)
    _add_design_search_options(p_module)
    # Responses come from the model: --response/--error are rejected for a
    # module, which needs to tell "not given" from the template defaults.
    p_module.set_defaults(
        doe_func=_cmd_new, _is_module=True, input=None, degree=None, response=None, error=None
    )

    # --- status ---
    p_status = doe_sub.add_parser("status", help="Show campaign state and suggested next step.")
    p_status.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_status.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Check the design target's linearized precision against its profile "
            "likelihood (refits the campaign; slower). Flags targets whose real "
            "uncertainty the SE understates."
        ),
    )
    _add_json(p_status)
    p_status.set_defaults(doe_func=_cmd_status)

    # --- fit ---
    p_fit = doe_sub.add_parser("fit", help="Estimate parameters from completed runs.")
    p_fit.add_argument("workbook", help="Path to the .xlsx workbook.")
    _add_json(p_fit)
    p_fit.set_defaults(doe_func=_cmd_fit)

    # --- anova ---
    p_anova = doe_sub.add_parser(
        "anova",
        help="Run ANOVA on a completed Latin-family workbook.",
    )
    p_anova.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_anova.add_argument(
        "--interaction",
        action="append",
        default=None,
        help="Interaction term as 'A:B' or 'A:B:C' (repeatable).",
    )
    p_anova.add_argument(
        "--include-replicate",
        action="store_true",
        help="Treat the replicate column as a blocking factor.",
    )
    _add_json(p_anova)
    p_anova.set_defaults(doe_func=_cmd_anova)

    # --- optimize ---
    p_opt = doe_sub.add_parser(
        "optimize",
        help="Run one active-learning round on an optimize-template workbook.",
        description=(
            "Fit the chosen surrogate to completed runs, score Sobol "
            "candidates with the chosen acquisition, and append the "
            "recommended batch as pending runs."
        ),
    )
    p_opt.add_argument("workbook", help="Path to the .xlsx workbook (template=optimize).")
    p_opt.add_argument(
        "--criterion",
        default=None,
        choices=("maximize", "minimize"),
        help="Optimization direction (default: whatever the workbook was created with).",
    )
    p_opt.add_argument(
        "--surrogate",
        default=None,
        help=(
            "Surrogate preset name ('gp' or 'response-surface'); see "
            "SURROGATE_PRESETS. Default: whatever the workbook was created with."
        ),
    )
    p_opt.add_argument(
        "--custom-surrogate",
        default=None,
        help=(
            "Dotted import path of a custom surrogate class, e.g. "
            "'mypkg.bayes:MyBayes'. Overrides --surrogate when given."
        ),
    )
    p_opt.add_argument(
        "--surrogate-kwarg",
        action="append",
        type=_parse_kv_float_eq,
        default=None,
        help="kwarg KEY=VALUE for --custom-surrogate constructor (repeatable).",
    )
    p_opt.add_argument(
        "--acquisition",
        default=None,
        help=(
            "Acquisition function: 'expected_improvement', 'confidence_bound', "
            "'steepest_ascent'. Default: whatever the workbook was created with."
        ),
    )
    p_opt.add_argument(
        "--acquisition-kwarg",
        action="append",
        type=_parse_kv_float_eq,
        default=None,
        help="kwarg KEY=VALUE for the acquisition (e.g. beta=2.0); repeatable.",
    )
    p_opt.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of new runs to recommend (default 4).",
    )
    p_opt.add_argument(
        "--n-candidates",
        type=int,
        default=2048,
        help="Sobol candidate pool size (default 2048).",
    )
    p_opt.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducible candidate-sampling seed.",
    )
    _add_json(p_opt)
    p_opt.set_defaults(doe_func=_cmd_optimize)

    # --- extend ---
    p_ext = doe_sub.add_parser(
        "extend", help="Append M more optimal runs using the cumulative FIM as prior."
    )
    p_ext.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_ext.add_argument("--n", type=int, required=True, help="Number of additional runs.")
    p_ext.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Multi-start budget for each single-design search (default 10).",
    )
    p_ext.add_argument(
        "--batch-strategy",
        choices=_BATCH_STRATEGIES,
        default="greedy",
        help=(
            "How a batch of --n > 1 runs is searched: 'greedy' adds one run at a time "
            "(then exchange passes), 'joint' optimizes all runs together, 'penalized' "
            "is greedy with a minimum spacing. Parametric templates and --module only "
            "(default greedy)."
        ),
    )
    _add_target_options(p_ext)
    p_ext.add_argument(
        "--clear-target",
        action="store_true",
        help="Drop the stored design target and design for all parameters again.",
    )
    _add_json(p_ext)
    p_ext.set_defaults(doe_func=_cmd_extend)

    # --- foldover ---
    p_fold = doe_sub.add_parser(
        "foldover",
        help="Append the mirror-image runs of a 2-level screening design (de-aliasing).",
    )
    p_fold.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_fold.add_argument(
        "--factor",
        default=None,
        help="Reverse only this factor (clears it and its 2FIs); default reverses every factor.",
    )
    _add_json(p_fold)
    p_fold.set_defaults(doe_func=_cmd_foldover)

    # --- gui ---
    p_gui = doe_sub.add_parser(
        "gui",
        help="Launch the Streamlit GUI over a workbook (requires discopt-doe[gui]).",
    )
    p_gui.add_argument(
        "workbook",
        nargs="?",
        default=None,
        help="Optional path to an existing .xlsx workbook to open on startup.",
    )
    p_gui.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port to bind on 127.0.0.1 (default: pick a free port).",
    )
    p_gui.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    p_gui.set_defaults(doe_func=_cmd_gui)


def _add_common_new_options(sp) -> None:
    """Options that apply to every ``new`` template."""
    sp.add_argument("-o", "--output", required=True, help="Output .xlsx path.")
    sp.add_argument(
        "--response",
        default="y",
        help="Name of the response column in the workbook (default 'y').",
    )
    sp.add_argument(
        "--error",
        type=float,
        default=1.0,
        help="Measurement error stdev (default 1.0).",
    )
    sp.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    sp.add_argument("--force", action="store_true", help="Overwrite existing output file.")
    sp.add_argument(
        "--extra-column",
        action="append",
        metavar="NAME",
        help=(
            "Add a bookkeeping column to the runs sheet (repeatable): operator, "
            "lot, instrument. It travels with the runs and `doe anova` can use it "
            "as a blocking factor."
        ),
    )
    _add_json(sp)


def _add_run_count_option(sp, *, default: int = 1) -> None:
    """``--n`` applies to parametric and optimize templates (initial batch).

    Combinatorial designs derive their run count from levels/factors x
    replicates, so they do not accept ``--n``.
    """
    sp.add_argument(
        "--n", type=int, default=default, help=f"Number of initial runs (default {default})."
    )


def _add_design_search_options(sp) -> None:
    """``--criterion``/``--n-starts`` govern the parametric D-optimal search."""
    sp.add_argument(
        "--criterion",
        default=_DEFAULT_CRITERION,
        choices=(*_CRITERION_CHOICES, *_CRITERION_ALIASES.keys()),
        help=(
            "Optimality criterion (default determinant aka D). average_variance (I) and "
            "max_variance (G) target prediction variance over the factor box."
        ),
    )
    sp.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Multi-start budget for each single-design search (default 10).",
    )
    _add_target_options(sp)


def _add_target_options(sp) -> None:
    """Design for the precision of chosen targets instead of all parameters."""
    g = sp.add_argument_group(
        "design target",
        "Design for the precision of chosen quantities instead of every parameter "
        "(the others become nuisance parameters). --criterion is then applied to the "
        "targets: trace = weighted sum of their variances, determinant = D_s, max_variance "
        "= the worst one. The target is stored in the workbook and reused by `extend`.",
    )
    g.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        default=None,
        metavar="PARAM[=WEIGHT]",
        help="A parameter to make precise, optionally weighted (repeatable).",
    )
    g.add_argument(
        "--target-expr",
        action="append",
        type=_parse_target_expr,
        default=None,
        metavar="NAME=FORMULA",
        help="A function of the parameters, e.g. 'ratio=k1/k2' (repeatable).",
    )
    g.add_argument(
        "--target-response",
        action="append",
        type=_parse_target_response,
        default=None,
        metavar="RESPONSE@INPUT=V[,...]",
        help=(
            "A model prediction at an operating point, e.g. 'y@T=350' (repeatable; "
            "repeat a response for several points)."
        ),
    )
    g.add_argument(
        "--tolerance",
        action="append",
        type=_parse_kv_float,
        default=None,
        metavar="NAME=TAU",
        help=(
            "The standard error wanted on a target (repeatable). Sets its weight to "
            "1/TAU^2, and `status` reports whether every target meets its tolerance. "
            "If any is given every target needs one; required to mix parameters, "
            "formulas and responses."
        ),
    )
    g.add_argument(
        "--relative-precision",
        action="store_true",
        help=(
            "Measure precision relative to each target's value: tolerances become "
            "fractions (0.05 = 5%%), weights apply to relative variances."
        ),
    )


def _add_json(sp) -> None:
    sp.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )


def run(args) -> int:
    """Dispatch ``discopt doe ...`` after argparse parsing."""
    func = getattr(args, "doe_func", None)
    if func is None:
        print("no doe subcommand given; try `discopt doe --help`", file=sys.stderr)
        return 1
    return int(func(args) or 0)


__all__ = [
    "DoEError",
    "ExtendParams",
    "NewParams",
    "OptimizeParams",
    "add_subparser",
    "do_anova",
    "do_extend",
    "do_fit",
    "do_new",
    "do_optimize",
    "do_status",
    "do_templates",
    "run",
]
