"""Tests for the jax-free optimal-design path.

The load-bearing claim of :mod:`discopt.doe.linear_design` is that for a model
linear in its parameters, ``XᵀX/σ²`` *is* the Fisher information the autodiff
path computes — not an approximation of it. Most of this file exists to hold
the two implementations against each other; if they ever diverge, the browser
build silently starts designing different experiments from the desktop CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from discopt.doe.design import DesignCriterion, project_to_simplex, sum_constraint
from discopt.doe.fim import compute_fim
from discopt.doe.linear_design import (
    A_OPTIMAL,
    CRITERIA,
    D_OPTIMAL,
    E_OPTIMAL,
    ME_OPTIMAL,
    design_matrix,
    design_row,
    evaluate_criterion,
    is_maximized,
    linear_batch_design,
    linear_fim,
    linear_optimal_design,
)
from discopt.doe.templates import build_template, template_parameter_names

# (template, inputs, degree, sample design points)
TEMPLATE_CASES = [
    (
        "linear",
        [("T", 300.0, 400.0), ("P", 1.0, 5.0)],
        None,
        [{"T": 350.0, "P": 3.0}, {"T": 300.0, "P": 5.0}, {"T": 399.9, "P": 1.2}],
    ),
    (
        "polynomial-1d",
        [("x", 0.0, 10.0)],
        3,
        [{"x": 7.5}, {"x": 0.0}, {"x": 10.0}, {"x": 2.25}],
    ),
    (
        "response-surface-2d",
        [("a", 0.0, 1.0), ("b", 0.0, 1.0)],
        None,
        [{"a": 0.3, "b": 0.8}, {"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}],
    ),
    (
        "response-surface-3d",
        [("a", 0.0, 1.0), ("b", 0.0, 1.0), ("c", 0.0, 1.0)],
        None,
        [{"a": 0.3, "b": 0.8, "c": 0.1}, {"a": 1.0, "b": 1.0, "c": 1.0}],
    ),
    (
        "scheffe-linear",
        [("p", 0.0, 1.0), ("q", 0.0, 1.0), ("r", 0.0, 1.0)],
        None,
        [{"p": 0.2, "q": 0.3, "r": 0.5}, {"p": 1.0, "q": 0.0, "r": 0.0}],
    ),
    (
        "scheffe-quadratic",
        [("p", 0.0, 1.0), ("q", 0.0, 1.0), ("r", 0.0, 1.0)],
        None,
        [{"p": 0.2, "q": 0.3, "r": 0.5}, {"p": 0.5, "q": 0.5, "r": 0.0}],
    ),
    (
        "scheffe-special-cubic",
        [("p", 0.0, 1.0), ("q", 0.0, 1.0), ("r", 0.0, 1.0)],
        None,
        [{"p": 0.2, "q": 0.3, "r": 0.5}, {"p": 1 / 3, "q": 1 / 3, "r": 1 / 3}],
    ),
]

CASE_IDS = [c[0] for c in TEMPLATE_CASES]
SIGMA = 2.5


def _setup(template, inputs, degree):
    """Build the Experiment, parameter names, and nominal parameter values."""
    experiment = build_template(
        template,
        inputs=inputs,
        measurement_error=SIGMA,
        degree=degree,
        mixture_total=1.0 if template.startswith("scheffe") else None,
    )
    names = template_parameter_names(template, degree=degree, n_inputs=len(inputs))
    # Values are arbitrary: for a linear model the FIM does not depend on them,
    # which is exactly what test_fim_is_independent_of_parameter_values checks.
    param_values = {n: 1.0 + 0.1 * i for i, n in enumerate(names)}
    return experiment, names, param_values


# ──────────────────── equivalence with the autodiff path ────────────────────


@pytest.mark.parametrize("template, inputs, degree, points", TEMPLATE_CASES, ids=CASE_IDS)
def test_linear_fim_matches_jax_fim_at_each_point(template, inputs, degree, points) -> None:
    """XᵀX/σ² reproduces the autodiff FIM to machine precision."""
    experiment, names, param_values = _setup(template, inputs, degree)
    template_args = {"degree": degree} if degree is not None else {}
    input_names = [i[0] for i in inputs]

    for point in points:
        expected = compute_fim(experiment, param_values, point).fim
        row = design_row(template, template_args, names, input_names, point)
        actual = linear_fim(row.reshape(1, -1), SIGMA)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("template, inputs, degree, points", TEMPLATE_CASES, ids=CASE_IDS)
def test_multi_run_fim_matches_accumulated_jax_fim(template, inputs, degree, points) -> None:
    """Information adds: the stacked design matrix equals the summed per-run FIMs."""
    experiment, names, param_values = _setup(template, inputs, degree)
    template_args = {"degree": degree} if degree is not None else {}
    input_names = [i[0] for i in inputs]

    expected = sum(compute_fim(experiment, param_values, p).fim for p in points)
    X = design_matrix(template, template_args, names, input_names, points)
    np.testing.assert_allclose(linear_fim(X, SIGMA), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("template, inputs, degree, points", TEMPLATE_CASES, ids=CASE_IDS)
def test_fim_is_independent_of_parameter_values(template, inputs, degree, points) -> None:
    """The property that makes the closed form valid: ∂y/∂θ does not involve θ."""
    experiment, names, _ = _setup(template, inputs, degree)
    low = {n: 0.01 for n in names}
    high = {n: 500.0 for n in names}
    point = points[0]
    np.testing.assert_allclose(
        compute_fim(experiment, low, point).fim,
        compute_fim(experiment, high, point).fim,
        rtol=1e-10,
        atol=1e-12,
    )


def test_criterion_constants_match_the_jax_module() -> None:
    """The duplicated criterion strings must not drift from DesignCriterion."""
    assert D_OPTIMAL == DesignCriterion.D_OPTIMAL
    assert A_OPTIMAL == DesignCriterion.A_OPTIMAL
    assert E_OPTIMAL == DesignCriterion.E_OPTIMAL
    assert ME_OPTIMAL == DesignCriterion.ME_OPTIMAL


@pytest.mark.parametrize("criterion", CRITERIA)
def test_evaluate_criterion_matches_fim_result_properties(criterion: str) -> None:
    """Criterion evaluation agrees with the FIMResult properties it mirrors."""
    experiment, names, param_values = _setup(*TEMPLATE_CASES[2][:3])
    points = TEMPLATE_CASES[2][3]
    fim_result = compute_fim(experiment, param_values, points[0])
    for extra in points[1:] * 3:
        fim_result_extra = compute_fim(experiment, param_values, extra)
        fim_result.fim = fim_result.fim + fim_result_extra.fim

    expected = {
        D_OPTIMAL: fim_result.d_optimal,
        A_OPTIMAL: fim_result.a_optimal,
        E_OPTIMAL: fim_result.e_optimal,
        ME_OPTIMAL: fim_result.me_optimal,
    }[criterion]
    assert evaluate_criterion(fim_result.fim, criterion) == pytest.approx(expected, rel=1e-10)


# ─────────────────────────── criterion mechanics ───────────────────────────


def test_is_maximized_flags_the_right_criteria() -> None:
    assert is_maximized(D_OPTIMAL) and is_maximized(E_OPTIMAL)
    assert not is_maximized(A_OPTIMAL) and not is_maximized(ME_OPTIMAL)


def test_singular_fim_gives_degenerate_d_and_a_values() -> None:
    """A rank-deficient FIM must not silently produce a finite score."""
    singular = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert evaluate_criterion(singular, D_OPTIMAL) == -np.inf
    assert evaluate_criterion(singular, E_OPTIMAL) == pytest.approx(0.0)


def test_evaluate_criterion_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown criterion"):
        evaluate_criterion(np.eye(2), "g_optimal")


def test_linear_fim_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        linear_fim(np.eye(3), 0.0)


def test_design_row_rejects_a_nonlinear_template() -> None:
    with pytest.raises(ValueError, match="unknown template"):
        design_row("module", {}, ["a"], ["x"], {"x": 1.0})


# ──────────────────────────── design searches ────────────────────────────


def test_batch_design_reaches_full_rank_and_improves() -> None:
    """A from-scratch D-optimal batch escapes the rank-1 degeneracy."""
    result = linear_batch_design(
        "response-surface-2d",
        8,
        parameter_names=template_parameter_names("response-surface-2d", n_inputs=2),
        input_names=["a", "b"],
        design_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        measurement_error=1.0,
        criterion=D_OPTIMAL,
        n_starts=6,
        seed=0,
    )
    assert result.n_experiments == 8
    assert np.linalg.matrix_rank(result.joint_fim) == 6
    assert np.isfinite(result.criterion_value)
    # Once full rank is reached, more runs can only add information.
    finite = [v for v in result.per_round_criterion if np.isfinite(v)]
    assert finite == sorted(finite)
    assert np.all(result.predicted_standard_errors > 0)


def test_batch_design_respects_bounds() -> None:
    result = linear_batch_design(
        "linear",
        6,
        parameter_names=["b0", "b1", "b2"],
        input_names=["T", "P"],
        design_bounds={"T": (300.0, 400.0), "P": (1.0, 5.0)},
        n_starts=4,
        seed=1,
    )
    for design in result.designs:
        assert 300.0 <= design["T"] <= 400.0
        assert 1.0 <= design["P"] <= 5.0


def test_batch_design_honours_a_mixture_constraint() -> None:
    """Scheffé designs must land on the simplex."""
    names = ["p", "q", "r"]
    result = linear_batch_design(
        "scheffe-quadratic",
        7,
        parameter_names=template_parameter_names("scheffe-quadratic", n_inputs=3),
        input_names=names,
        design_bounds={n: (0.0, 1.0) for n in names},
        equality_constraints=[sum_constraint(names, 1.0)],
        feasible_projection=lambda d: project_to_simplex(d, names, 1.0),
        n_starts=5,
        seed=2,
    )
    for design in result.designs:
        assert sum(design[n] for n in names) == pytest.approx(1.0, abs=1e-6)
        assert all(design[n] >= -1e-9 for n in names)


def test_single_point_design_extends_a_prior_fim() -> None:
    """The `extend` use case: add one run to information already collected."""
    names = template_parameter_names("response-surface-2d", n_inputs=2)
    seed_batch = linear_batch_design(
        "response-surface-2d",
        7,
        parameter_names=names,
        input_names=["a", "b"],
        design_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        n_starts=5,
        seed=3,
    )
    prior = seed_batch.joint_fim
    result = linear_optimal_design(
        "response-surface-2d",
        parameter_names=names,
        input_names=["a", "b"],
        design_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        prior_fim=prior,
        n_starts=6,
        seed=4,
    )
    assert result.criterion_value > evaluate_criterion(prior, D_OPTIMAL)
    assert 0.0 <= result.design["a"] <= 1.0
    assert "SE(" in result.summary()


def test_single_point_design_reports_rank_one_degeneracy() -> None:
    """One run cannot identify six parameters; say so instead of returning junk.

    Worth pinning: ``slogdet`` reports a finite log-determinant for a
    numerically rank-1 matrix, because the determinant underflows to ~1e-81
    rather than to exactly zero. Without the explicit rank check the search
    returns a confident-looking score for a hopeless design.
    """
    with pytest.raises(RuntimeError, match="cannot reach the 6 parameters"):
        linear_optimal_design(
            "response-surface-2d",
            parameter_names=template_parameter_names("response-surface-2d", n_inputs=2),
            input_names=["a", "b"],
            design_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
            n_starts=3,
            seed=5,
        )


def test_design_rejects_a_nonlinear_template() -> None:
    with pytest.raises(ValueError, match="not linear in its parameters"):
        linear_optimal_design(
            "module",
            parameter_names=["k"],
            input_names=["T"],
            design_bounds={"T": (0.0, 1.0)},
        )


def test_design_rejects_mismatched_prior_fim() -> None:
    with pytest.raises(ValueError, match="does not match"):
        linear_optimal_design(
            "linear",
            parameter_names=["b0", "b1"],
            input_names=["x"],
            design_bounds={"x": (0.0, 1.0)},
            prior_fim=np.eye(5),
        )


def test_design_rejects_missing_bounds() -> None:
    with pytest.raises(ValueError, match="missing entries"):
        linear_optimal_design(
            "linear",
            parameter_names=["b0", "b1"],
            input_names=["x", "y"],
            design_bounds={"x": (0.0, 1.0)},
        )


@pytest.mark.slow
def test_batch_criterion_is_competitive_with_the_jax_search() -> None:
    """The jax-free search finds designs of comparable quality to the autodiff one.

    Compared on criterion value rather than on design coordinates: the
    underlying SLSQP/L-BFGS-B searches are BLAS-sensitive and multi-modal, so
    two runs legitimately land on different but equally good optima.
    """
    from discopt.doe.design import batch_optimal_experiment

    inputs = [("a", 0.0, 1.0), ("b", 0.0, 1.0)]
    experiment, names, param_values = _setup("response-surface-2d", inputs, None)
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}

    jax_result = batch_optimal_experiment(
        experiment, param_values, bounds, 8, criterion=DesignCriterion.D_OPTIMAL, n_starts=6, seed=0
    )
    linear_result = linear_batch_design(
        "response-surface-2d",
        8,
        parameter_names=names,
        input_names=["a", "b"],
        design_bounds=bounds,
        measurement_error=SIGMA,
        criterion=D_OPTIMAL,
        n_starts=6,
        seed=0,
    )
    # Both should be sane, full-rank designs within a couple of nats of each
    # other in log-determinant.
    assert np.linalg.matrix_rank(linear_result.joint_fim) == len(names)
    assert linear_result.criterion_value == pytest.approx(jax_result.criterion_value, abs=2.0)


@pytest.mark.parametrize(
    "template, inputs, degree, n",
    [
        ("response-surface-2d", [("a", 0.0, 1.0), ("b", 0.0, 1.0)], None, 8),
        ("linear", [("T", 300.0, 400.0), ("P", 1.0, 5.0)], None, 6),
        ("polynomial-1d", [("x", 0.0, 10.0)], 3, 6),
        ("scheffe-quadratic", [("p", 0.0, 1.0), ("q", 0.0, 1.0), ("r", 0.0, 1.0)], None, 8),
    ],
)
def test_do_new_linear_path_matches_the_jax_path(tmp_path, template, inputs, degree, n) -> None:
    """`use_linear_design` designs are as informative as the autodiff ones.

    Compared on criterion value, not coordinates: both searches are multi-start
    over a multi-modal surface, so they legitimately land on different optima of
    equal quality. The tolerance is loose enough for that and tight enough to
    catch a genuinely worse design.
    """
    from discopt.doe.cli import NewParams, do_new

    common = dict(
        inputs=inputs,
        response_name="resp",
        measurement_error=1.0,
        criterion=D_OPTIMAL,
        seed=0,
        n_starts=8,
        template=template,
        degree=degree,
        mixture_total=1.0 if template.startswith("scheffe") else None,
    )
    jax_out = do_new(NewParams(output=tmp_path / "jax.xlsx", n=n, **common))
    linear_out = do_new(
        NewParams(output=tmp_path / "linear.xlsx", n=n, use_linear_design=True, **common)
    )

    assert jax_out["parameter_names"] == linear_out["parameter_names"]
    assert len(jax_out["new_run_ids"]) == len(linear_out["new_run_ids"]) == n
    # The linear path must not be materially worse (it may be slightly better,
    # since the two searches explore differently).
    assert linear_out["criterion_value"] > jax_out["criterion_value"] - 0.5


def test_do_new_linear_path_rejects_a_nonlinear_experiment(tmp_path) -> None:
    """A --module experiment may be nonlinear, so the closed form does not apply."""
    from discopt.doe.cli import DoEError, NewParams, do_new

    with pytest.raises(DoEError, match="linear in its parameters"):
        do_new(
            NewParams(
                output=tmp_path / "x.xlsx",
                n=4,
                inputs=[("T", 0.0, 1.0)],
                response_name="y",
                measurement_error=1.0,
                criterion=D_OPTIMAL,
                seed=0,
                n_starts=2,
                template=None,
                module_callable="some.module:build",
                use_linear_design=True,
            )
        )
