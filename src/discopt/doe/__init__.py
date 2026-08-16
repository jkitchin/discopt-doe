"""discopt.doe -- Design of Experiments.

This package has three complementary entry points, each tailored to
a different question:

1. **"What is the best operating condition?"**
   :func:`optimize_round` runs one active-learning round: fit a
   surrogate to the experiments completed so far, recommend the
   next batch via an acquisition function (expected improvement,
   UCB, steepest ascent), append the batch to a workbook for
   execution. See the ``active-learning`` tutorial notebook.

2. **"Does this factor matter?"**
   :func:`factorial_2level_design` builds 2-level full factorial
   screening designs; :func:`effects_estimates` gives signed
   main-effect estimates; :func:`anova_report` produces the
   F-table.

3. **"How precisely can I estimate the model parameters?"**
   :func:`optimal_experiment` and :func:`batch_optimal_experiment`
   solve for an exact D/A/E-optimal design using the Fisher
   Information Matrix computed with JAX autodiff;
   :func:`diagnose_identifiability` and
   :func:`estimability_rank` warn when the chosen experiments
   cannot identify the parameters.

Quick start (FIM-based parameter-estimation design)
---------------------------------------------------
>>> from discopt.doe import compute_fim, optimal_experiment, DesignCriterion
>>> fim_result = compute_fim(experiment, param_values, design_values)
>>> design = optimal_experiment(experiment, param_values, design_bounds)
>>> print(design.summary())

Quick start (active-learning optimization)
------------------------------------------
>>> from discopt.doe import optimize_round, OptimizationCriterion
>>> result = optimize_round(
...     workbook="opt.xlsx",
...     criterion=OptimizationCriterion.MAXIMIZE,
...     surrogate="gp",                       # or sklearn estimator, or Surrogate
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.next_designs)

Identifiability and estimability diagnostics
--------------------------------------------
>>> from discopt.doe import diagnose_identifiability, estimability_rank, profile_likelihood
>>> diag = diagnose_identifiability(experiment, param_values)
>>> est = estimability_rank(experiment, param_values)
>>> profile = profile_likelihood(experiment, data, "k")

See Also
--------
discopt.estimate : Parameter estimation using the same Experiment interface.
discopt.doe.surrogate : Surrogate model protocol + sklearn adapter.
discopt.doe.acquisition : Acquisition functions (EI, UCB, steepest ascent).
"""

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

# discopt-doe consumes the public discopt.parametric API and the
# "discopt.cli" plugin hook, both introduced in discopt 0.6, and the solver
# correctness fixes that landed in 0.8 (false `infeasible` on a GDP
# disjunction, false `Unbounded` on a bounded LP, an ignored non-zero
# `Constraint.rhs`). A stale base install fails here with a clear message
# instead of a wrong answer or an AttributeError later. Two-component compare
# so 0.8.0.dev0 local builds pass.
_MIN_DISCOPT = (0, 8)


def _version_prefix(found: str) -> tuple[int, ...]:
    """(major, minor) integer prefix, tolerant of pre-release suffixes.

    ``"0.6rc1"`` -> ``(0, 6)``, ``"0.7b2"`` -> ``(0, 7)``. The old
    ``int(p) for p in ... if p.isdigit()`` dropped a component like ``"6rc1"``
    entirely and spuriously rejected valid pre-releases.
    """
    import re as _re

    out: list[int] = []
    for p in found.split(".")[:2]:
        m = _re.match(r"\d+", p)
        if m:
            out.append(int(m.group()))
    return tuple(out)


try:
    _found = _dist_version("discopt")
    _v = _version_prefix(_found)
    if _v < _MIN_DISCOPT:
        raise ImportError(
            f"discopt-doe requires discopt>={'.'.join(map(str, _MIN_DISCOPT))} "
            f"(found {_found}); upgrade with `pip install -U discopt`."
        )
except _PkgNotFound:  # odd dev setups (no dist metadata): don't block
    pass

# Re-exports are resolved lazily (PEP 562). Importing this package used to pull
# all 18 submodules eagerly, which dragged in `discopt.estimate` — and through it
# scipy and, on the FIM paths, jax. That makes the jax-free half of the package
# (Latin squares, factorial screening, ANOVA, OLS fitting, the workbook layer)
# unusable anywhere jax cannot be installed, notably Pyodide/WASM, where no jax
# wheel exists. Deferring the imports keeps `from discopt.doe import X` working
# exactly as before while charging each caller only for what it actually touches.
# tests/test_import_hygiene.py pins the modules that must stay dependency-light.

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "acquisition": (
        "ACQUISITIONS",
        "confidence_bound",
        "expected_improvement",
        "resolve_acquisition",
        "steepest_ascent",
    ),
    "anova": ("AnovaEffect", "AnovaTable", "anova_report"),
    "classical": (
        "ClassicalDesign",
        "box_behnken_design",
        "central_composite_design",
        "latin_hypercube_design",
    ),
    "design": (
        "BatchDesignResult",
        "BatchStrategy",
        "DesignConstraint",
        "DesignCriterion",
        "DesignResult",
        "batch_optimal_experiment",
        "optimal_experiment",
        "project_to_simplex",
        "sample_simplex",
        "sum_constraint",
    ),
    "discrimination": (
        "DiscriminationCriterion",
        "DiscriminationDesignResult",
        "discriminate_compound",
        "discriminate_design",
        "evaluate_discrimination_criterion",
    ),
    "discrimination_sequential": ("DiscriminationRound", "sequential_discrimination"),
    "estimability": (
        "EstimabilityResult",
        "collinearity_index",
        "d_optimal_subset",
        "estimability_rank",
    ),
    "exploration": ("ExplorationResult", "explore_design_space"),
    "fim": (
        "FIMResult",
        "IdentifiabilityDiagnostics",
        "IdentifiabilityResult",
        "check_identifiability",
        "compute_fim",
        "diagnose_identifiability",
    ),
    "fractional": ("fractional_factorial_design",),
    "latin": (
        "LatinDesign",
        "graeco_latin_square",
        "hyper_graeco_latin_square",
        "latin_square",
        "latin_square_design",
    ),
    "linear_design": (
        "LinearBatchDesignResult",
        "LinearDesignResult",
        "batch_design_from_basis",
        "design_matrix",
        "design_row",
        "linear_batch_design",
        "linear_fim",
        "linear_optimal_design",
    ),
    "symbolic": (
        "ModelSyntaxError",
        "SymbolicModel",
        "fit_least_squares",
        "parse_expression",
    ),
    "model_based": (
        "ModelBasedRoundResult",
        "ParametricSurrogate",
        "model_based_optimize_round",
    ),
    "optimize": ("OptimizationCriterion", "OptimizationRoundResult", "optimize_round"),
    "profile": ("ProfileLikelihoodResult", "profile_all", "profile_likelihood"),
    "screening": ("FactorialDesign", "effects_estimates", "factorial_2level_design"),
    "selection": (
        "ModelSelectionResult",
        "likelihood_ratio_test",
        "model_selection",
        "vuong_test",
    ),
    "sequential": ("DoERound", "sequential_doe"),
    # The same three names as "design" above, from the module that actually
    # defines them. Resolving them through `design` needs the FIM machinery and
    # the base package; this route needs neither, which is what lets a mixture
    # design run in the browser.
    "simplex": (
        "DesignConstraint",
        "project_to_simplex",
        "sample_simplex",
        "sum_constraint",
    ),
    "surrogate": ("Surrogate", "coerce_surrogate"),
    "templates": (
        "TEMPLATE_NAMES",
        "build_template",
        "linear_template",
        "polynomial_1d_template",
        "response_surface_template",
        "scheffe_linear_template",
        "scheffe_quadratic_template",
        "scheffe_special_cubic_template",
        "simplex_centroid_points",
        "simplex_lattice_points",
    ),
}

# Public name -> submodule it lives in.
_EXPORTS: dict[str, str] = {
    name: module for module, names in _SUBMODULE_EXPORTS.items() for name in names
}

# Public names that differ from the attribute inside the submodule.
_ALIASES: dict[str, tuple[str, str]] = {"SURROGATE_PRESETS": ("surrogate", "PRESETS")}

__all__ = [
    "AnovaEffect",
    "AnovaTable",
    "BatchDesignResult",
    "ClassicalDesign",
    "BatchStrategy",
    "DesignConstraint",
    "DesignCriterion",
    "DesignResult",
    "FactorialDesign",
    "LatinDesign",
    "DiscriminationCriterion",
    "DiscriminationDesignResult",
    "DiscriminationRound",
    "DoERound",
    "EstimabilityResult",
    "ExplorationResult",
    "FIMResult",
    "IdentifiabilityDiagnostics",
    "IdentifiabilityResult",
    "LinearBatchDesignResult",
    "LinearDesignResult",
    "ModelBasedRoundResult",
    "ModelSyntaxError",
    "ModelSelectionResult",
    "OptimizationCriterion",
    "OptimizationRoundResult",
    "ParametricSurrogate",
    "ProfileLikelihoodResult",
    "SURROGATE_PRESETS",
    "Surrogate",
    "SymbolicModel",
    "ACQUISITIONS",
    "TEMPLATE_NAMES",
    "anova_report",
    "batch_design_from_basis",
    "batch_optimal_experiment",
    "box_behnken_design",
    "build_template",
    "central_composite_design",
    "coerce_surrogate",
    "confidence_bound",
    "expected_improvement",
    "optimize_round",
    "resolve_acquisition",
    "steepest_ascent",
    "check_identifiability",
    "graeco_latin_square",
    "hyper_graeco_latin_square",
    "latin_square",
    "latin_square_design",
    "collinearity_index",
    "compute_fim",
    "d_optimal_subset",
    "design_matrix",
    "design_row",
    "diagnose_identifiability",
    "discriminate_compound",
    "effects_estimates",
    "factorial_2level_design",
    "fit_least_squares",
    "fractional_factorial_design",
    "discriminate_design",
    "estimability_rank",
    "evaluate_discrimination_criterion",
    "explore_design_space",
    "latin_hypercube_design",
    "likelihood_ratio_test",
    "linear_batch_design",
    "linear_fim",
    "linear_optimal_design",
    "linear_template",
    "model_based_optimize_round",
    "model_selection",
    "optimal_experiment",
    "parse_expression",
    "polynomial_1d_template",
    "project_to_simplex",
    "response_surface_template",
    "profile_all",
    "profile_likelihood",
    "sample_simplex",
    "scheffe_linear_template",
    "scheffe_quadratic_template",
    "scheffe_special_cubic_template",
    "sequential_discrimination",
    "sequential_doe",
    "simplex_centroid_points",
    "simplex_lattice_points",
    "sum_constraint",
    "vuong_test",
]


def __getattr__(name: str):
    """Import the submodule owning ``name`` on first access (PEP 562)."""
    if name in _ALIASES:
        module_name, attr = _ALIASES[name]
    elif name in _EXPORTS:
        module_name, attr = _EXPORTS[name], name
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), attr)
    # Cache on the module so repeat lookups bypass __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
