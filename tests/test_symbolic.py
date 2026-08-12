"""Tests for user-defined models differentiated with sympy.

The claim under test is the same shape as the one in test_linear_design.py, but
harder: for a model that is genuinely *nonlinear* in its parameters, a
sympy-derived Jacobian must reproduce what jax autodiff computes. If it does,
model-based design for arbitrary models works without jax — which is what makes
the browser model editor possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from discopt.doe.linear_design import D_OPTIMAL, batch_design_from_basis, evaluate_criterion
from discopt.doe.symbolic import (
    ModelSyntaxError,
    SymbolicModel,
    basis_evaluator,
    fit_least_squares,
    parse_expression,
)

R_GAS = 8.314

# (label, source, parameters, inputs, nominal theta, design bounds)
NONLINEAR_MODELS = [
    (
        "arrhenius",
        f"k0 * exp(-Ea / ({R_GAS} * T))",
        ("k0", "Ea"),
        ("T",),
        {"k0": 2.0, "Ea": 5000.0},
        {"T": (300.0, 500.0)},
    ),
    (
        "michaelis-menten",
        "vmax * S / (Km + S)",
        ("vmax", "Km"),
        ("S",),
        {"vmax": 3.0, "Km": 0.8},
        {"S": (0.05, 10.0)},
    ),
    (
        "exponential-decay",
        "A * exp(-t / tau) + C",
        ("A", "tau", "C"),
        ("t",),
        {"A": 5.0, "tau": 2.5, "C": 0.4},
        {"t": (0.0, 12.0)},
    ),
    (
        "two-factor-nonlinear",
        "a * exp(-b * x) * (1 + c * z**2)",
        ("a", "b", "c"),
        ("x", "z"),
        {"a": 1.5, "b": 0.3, "c": 0.2},
        {"x": (0.0, 8.0), "z": (-2.0, 2.0)},
    ),
]
MODEL_IDS = [m[0] for m in NONLINEAR_MODELS]
SIGMA = 0.5


def build(label, source, params, inputs, theta, bounds) -> SymbolicModel:
    return SymbolicModel(
        source=source,
        parameter_names=params,
        input_names=inputs,
        response_name="y",
        measurement_error=SIGMA,
    )


# ─────────────────────── the expression parser ───────────────────────


def test_parses_arithmetic_and_allowed_functions() -> None:
    expr = parse_expression("k0 * exp(-Ea / (8.314 * T)) + sqrt(T)", ["k0", "Ea", "T"])
    assert {s.name for s in expr.free_symbols} == {"k0", "Ea", "T"}


def test_rejects_unknown_names() -> None:
    with pytest.raises(ModelSyntaxError, match="unknown name 'Q'"):
        parse_expression("k * Q", ["k", "T"])


@pytest.mark.parametrize(
    "source, match",
    [
        # Rejected at the attribute access: the call target is not a bare name.
        ("__import__('os').system('ls')", "only direct calls"),
        ("__import__('os')", "is not allowed"),
        ("().__class__", "not allowed"),
        ("open('/etc/passwd')", "not allowed"),
        ("[1, 2, 3]", "not allowed"),
        ("lambda x: x", "not allowed"),
        ("T if T else T", "not allowed"),
        ("{'a': 1}", "not allowed"),
        ("T.real", "not allowed"),
        ("x[0]", "not allowed"),
        ("exp(x, base=2)", "keyword arguments"),
        ("'a string'", "numeric literals"),
        ("T and T", "not allowed"),
        ("T > 1", "not allowed"),
    ],
)
def test_rejects_everything_that_is_not_a_model(source: str, match: str) -> None:
    """The parser is the trust boundary for expressions arriving in workbooks."""
    with pytest.raises(ModelSyntaxError, match=match):
        parse_expression(source, ["T", "x"])


def test_parser_never_evaluates_its_input(tmp_path) -> None:
    """A payload that would have side effects must not have them."""
    canary = tmp_path / "canary.txt"
    with pytest.raises(ModelSyntaxError):
        parse_expression(f"__import__('pathlib').Path({str(canary)!r}).touch()", ["T"])
    assert not canary.exists()


def test_rejects_empty_and_malformed() -> None:
    with pytest.raises(ModelSyntaxError, match="empty"):
        parse_expression("   ", ["T"])
    with pytest.raises(ModelSyntaxError, match="could not parse"):
        parse_expression("k0 * (T", ["k0", "T"])


# ──────────────── agreement with the jax autodiff path ────────────────


@pytest.mark.parametrize("case", NONLINEAR_MODELS, ids=MODEL_IDS)
def test_sympy_jacobian_matches_jax_autodiff(case) -> None:
    """The load-bearing claim: sympy reproduces the autodiff FIM for nonlinear models.

    Builds the same model twice — once as a real discopt Experiment evaluated
    by ``compute_fim`` (jax), once symbolically — and compares the Fisher
    information at several parameter values and design points.
    """
    import discopt.modeling as dm
    from discopt.estimate import Experiment, ExperimentModel

    from discopt.doe.fim import compute_fim

    label, source, params, inputs, theta, bounds = case
    model = build(*case)

    # The equivalent discopt model, built with the base package's own API.
    dm_builders = {
        "arrhenius": lambda v: v["k0"] * dm.exp(-v["Ea"] / (R_GAS * v["T"])),
        "michaelis-menten": lambda v: v["vmax"] * v["S"] / (v["Km"] + v["S"]),
        "exponential-decay": lambda v: v["A"] * dm.exp(-v["t"] / v["tau"]) + v["C"],
        "two-factor-nonlinear": lambda v: (
            v["a"] * dm.exp(-v["b"] * v["x"]) * (1 + v["c"] * v["z"] ** 2)
        ),
    }

    class _Exp(Experiment):
        def create_model(self, **kw):
            m = dm.Model(label)
            vs = {}
            for n in params:
                vs[n] = m.continuous(n, lb=-1e6, ub=1e6)
            for n in inputs:
                lo, hi = bounds[n]
                vs[n] = m.continuous(n, lb=lo, ub=hi)
            return ExperimentModel(
                model=m,
                unknown_parameters={n: vs[n] for n in params},
                design_inputs={n: vs[n] for n in inputs},
                responses={"y": dm_builders[label](vs)},
                measurement_error={"y": SIGMA},
            )

    experiment = _Exp()

    # A spread of parameter values and design points, since for a nonlinear
    # model the FIM depends on both.
    thetas = [theta, {k: v * 1.7 for k, v in theta.items()}, {k: v * 0.4 for k, v in theta.items()}]
    grids = []
    for frac in (0.15, 0.5, 0.85):
        grids.append({n: bounds[n][0] + frac * (bounds[n][1] - bounds[n][0]) for n in inputs})

    for th in thetas:
        for point in grids:
            expected = compute_fim(experiment, th, point).fim
            actual = model.fim(th, [point])
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("case", NONLINEAR_MODELS, ids=MODEL_IDS)
def test_fim_depends_on_parameter_values(case) -> None:
    """What separates these from the linear templates, and why a guess is needed."""
    _, _, _, inputs, theta, bounds = case
    model = build(*case)
    point = {n: (bounds[n][0] + bounds[n][1]) / 2 for n in inputs}
    a = model.fim(theta, [point])
    b = model.fim({k: v * 2.5 for k, v in theta.items()}, [point])
    assert not np.allclose(a, b)


def test_information_accumulates_over_runs() -> None:
    model = build(*NONLINEAR_MODELS[0])
    theta = NONLINEAR_MODELS[0][4]
    points = [{"T": 320.0}, {"T": 400.0}, {"T": 480.0}]
    total = model.fim(theta, points)
    summed = sum(model.fim(theta, [p]) for p in points)
    np.testing.assert_allclose(total, summed, rtol=1e-12)


# ───────────────────────── model mechanics ─────────────────────────


def test_predict_matches_hand_evaluation() -> None:
    model = build(*NONLINEAR_MODELS[0])
    got = model.predict({"k0": 2.0, "Ea": 5000.0}, {"T": 350.0})
    assert got == pytest.approx(2.0 * np.exp(-5000.0 / (R_GAS * 350.0)))


def test_jacobian_row_is_ordered_by_parameter_names() -> None:
    """A constant derivative must still occupy its own column."""
    model = SymbolicModel(
        source="A * exp(-t / tau) + C",
        parameter_names=("A", "tau", "C"),
        input_names=("t",),
        measurement_error=1.0,
    )
    row = model.jacobian_row({"A": 5.0, "tau": 2.5, "C": 0.4}, {"t": 3.0})
    assert row.shape == (3,)
    # dy/dC is structurally 1; lambdify would otherwise collapse it.
    assert row[2] == pytest.approx(1.0)
    assert row[0] == pytest.approx(np.exp(-3.0 / 2.5))


def test_rejects_overlapping_and_missing_names() -> None:
    with pytest.raises(ValueError, match="both parameter and input"):
        SymbolicModel(source="a * T", parameter_names=("a", "T"), input_names=("T",))
    with pytest.raises(ValueError, match="at least one unknown parameter"):
        SymbolicModel(source="T", parameter_names=(), input_names=("T",))


def test_bad_expression_fails_at_construction() -> None:
    """Not later, inside an optimizer callback."""
    with pytest.raises(ModelSyntaxError):
        SymbolicModel(source="k * nope", parameter_names=("k",), input_names=("T",))


def test_missing_values_name_the_culprit() -> None:
    model = build(*NONLINEAR_MODELS[0])
    with pytest.raises(KeyError, match="Ea"):
        model.predict({"k0": 1.0}, {"T": 350.0})
    with pytest.raises(KeyError, match="T"):
        model.predict({"k0": 1.0, "Ea": 100.0}, {})


def test_metadata_round_trip() -> None:
    """A workbook stores the expression as data; reading it back evaluates nothing."""
    model = build(*NONLINEAR_MODELS[0])
    meta = model.to_metadata()
    assert meta["expression"] == model.source
    assert meta["parameters"] == list(model.parameter_names)

    rebuilt = SymbolicModel.from_metadata(meta, response_name="y", measurement_error=SIGMA)
    theta, point = {"k0": 2.0, "Ea": 5000.0}, {"T": 375.0}
    assert rebuilt.predict(theta, point) == pytest.approx(model.predict(theta, point))
    np.testing.assert_allclose(rebuilt.fim(theta, [point]), model.fim(theta, [point]))


def test_metadata_rejects_incomplete_input() -> None:
    with pytest.raises(ValueError, match="missing"):
        SymbolicModel.from_metadata({"expression": "a * T", "parameters": ["a"]})


# ──────────────────────── design and fitting ────────────────────────


@pytest.mark.parametrize("case", NONLINEAR_MODELS, ids=MODEL_IDS)
def test_design_search_drives_a_nonlinear_model(case) -> None:
    """The linear_design search runs unchanged on a sympy Jacobian."""
    _, _, params, inputs, theta, bounds = case
    model = build(*case)
    n_p = len(params)

    result = batch_design_from_basis(
        basis_evaluator(model, theta),
        n_p + 3,
        parameter_names=list(params),
        input_names=list(inputs),
        design_bounds=bounds,
        measurement_error=SIGMA,
        criterion=D_OPTIMAL,
        n_starts=6,
        seed=0,
    )
    assert result.n_experiments == n_p + 3
    assert np.linalg.matrix_rank(result.joint_fim) == n_p
    assert np.isfinite(result.criterion_value)
    for d in result.designs:
        for n in inputs:
            assert bounds[n][0] - 1e-9 <= d[n] <= bounds[n][1] + 1e-9


def test_designed_experiment_beats_a_naive_grid() -> None:
    """The point of designing: more information for the same number of runs."""
    label, source, params, inputs, theta, bounds = NONLINEAR_MODELS[0]
    model = build(*NONLINEAR_MODELS[0])
    n = 6

    designed = batch_design_from_basis(
        basis_evaluator(model, theta),
        n,
        parameter_names=list(params),
        input_names=list(inputs),
        design_bounds=bounds,
        measurement_error=SIGMA,
        n_starts=8,
        seed=1,
    )
    grid = [{"T": t} for t in np.linspace(bounds["T"][0], bounds["T"][1], n)]
    grid_value = evaluate_criterion(model.fim(theta, grid), D_OPTIMAL)
    assert designed.criterion_value > grid_value


def test_fit_recovers_known_parameters_from_noise_free_data() -> None:
    label, source, params, inputs, theta, bounds = NONLINEAR_MODELS[0]
    model = build(*NONLINEAR_MODELS[0])

    designs = [{"T": t} for t in np.linspace(300.0, 500.0, 12)]
    rows = [{**d, "y": model.predict(theta, d)} for d in designs]

    out = fit_least_squares(model, rows, {"k0": 1.0, "Ea": 3000.0})
    assert out["success"]
    for name, want in theta.items():
        assert out["estimates"][name] == pytest.approx(want, rel=1e-6)
    assert out["residual_sum_of_squares"] < 1e-16
    assert out["n_observations"] == 12


def test_fit_reports_uncertainty_on_noisy_data() -> None:
    label, source, params, inputs, theta, bounds = NONLINEAR_MODELS[1]
    model = build(*NONLINEAR_MODELS[1])

    rng = np.random.default_rng(0)
    designs = [{"S": s} for s in np.linspace(0.05, 10.0, 25)]
    rows = [{**d, "y": model.predict(theta, d) + rng.normal(0.0, 0.02)} for d in designs]

    out = fit_least_squares(model, rows, {"vmax": 1.0, "Km": 1.0})
    assert out["success"]
    for name, want in theta.items():
        assert out["estimates"][name] == pytest.approx(want, rel=0.1)
        # A finite, positive standard error and an interval that brackets truth.
        assert out["std_errors"][name] > 0
        assert out["ci_lower"][name] < want < out["ci_upper"][name]
    assert out["degrees_of_freedom"] == 23


def test_fit_uses_the_analytic_jacobian() -> None:
    """Not finite differences: the Jacobian passed to least_squares is exact."""
    model = build(*NONLINEAR_MODELS[0])
    theta = NONLINEAR_MODELS[0][4]
    designs = [{"T": t} for t in (320.0, 400.0, 480.0)]
    analytic = model.design_matrix(theta, designs)

    eps = 1e-6
    numeric = np.zeros_like(analytic)
    for j, name in enumerate(model.parameter_names):
        hi = dict(theta)
        lo = dict(theta)
        step = eps * max(abs(theta[name]), 1.0)
        hi[name] += step
        lo[name] -= step
        for i, d in enumerate(designs):
            numeric[i, j] = (model.predict(hi, d) - model.predict(lo, d)) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5)


def test_fit_rejects_empty_data() -> None:
    model = build(*NONLINEAR_MODELS[0])
    with pytest.raises(ValueError, match="no completed runs"):
        fit_least_squares(model, [], {"k0": 1.0, "Ea": 1.0})


def test_pretty_shows_the_derivatives() -> None:
    model = build(*NONLINEAR_MODELS[1])
    text = model.pretty()
    assert "y =" in text
    assert "dy/dvmax" in text and "dy/dKm" in text


# ───────────────── CLI / workbook integration ─────────────────


class TestSymbolicCampaigns:
    """The design → fit → extend loop for a user-defined model."""

    TRUTH = {"k0": 3.7, "Ea": 6200.0}
    EXPR = "k0 * exp(-Ea / (8.314 * T))"

    @staticmethod
    def _new(tmp_path, **kw):
        from discopt.doe.cli import NewParams, do_new

        params = dict(
            output=tmp_path / "campaign.xlsx",
            n=6,
            inputs=[("T", 300.0, 500.0)],
            response_name="rate",
            measurement_error=0.05,
            criterion="determinant",
            seed=0,
            n_starts=6,
            template="symbolic",
            expression=TestSymbolicCampaigns.EXPR,
            param_initial_guess={"k0": 2.0, "Ea": 5000.0},
        )
        params.update(kw)
        return do_new(NewParams(**params))

    @classmethod
    def _fill(cls, path, *, noise=0.0, seed=0):
        import openpyxl

        rng = np.random.default_rng(seed)
        book = openpyxl.load_workbook(path)
        sheet = book["runs"]
        head = [c.value for c in sheet[1]]
        col = head.index("rate") + 1
        for row in sheet.iter_rows(min_row=2):
            values = dict(zip(head, [c.value for c in row]))
            if values.get("run_id") is None or values.get("rate") is not None:
                continue
            T = float(values["T"])
            y = cls.TRUTH["k0"] * np.exp(-cls.TRUTH["Ea"] / (8.314 * T))
            row[col - 1].value = float(y + (rng.normal(0.0, noise) if noise else 0.0))
        book.save(path)

    def test_new_designs_at_the_support_points(self, tmp_path) -> None:
        """A 2-parameter model's D-optimal design collapses onto 2 support points."""
        out = self._new(tmp_path)
        assert out["parameter_names"] == ["k0", "Ea"]
        assert len(out["new_run_ids"]) == 6
        temps = sorted({round(d["T"], 6) for d in out["designs"]})
        assert len(temps) == 2
        assert temps[0] == pytest.approx(300.0, abs=1e-3)
        assert temps[1] == pytest.approx(500.0, abs=1e-3)

    @pytest.mark.parametrize("sigma", [1.0, 0.5, 0.1, 0.05])
    def test_the_design_does_not_depend_on_the_measurement_error(self, tmp_path, sigma) -> None:
        """D-optimality is scale-free in σ, and the design must be too.

        ``FIM = JᵀJ/σ²`` is a positive rescaling, so σ shifts log-det by a
        constant and cannot move the argmax. It did once: the regularizing
        ridge was an absolute ``1e-6``, while this model's information in the
        Ea direction is ~1e-7 at σ = 1 — a hundredfold smaller than its own
        guard. The ridge then outvoted the data and the search returned all six
        runs at a single temperature, a rank-deficient design that identifies
        neither parameter, while σ = 0.05 (what the rest of this class uses)
        happened to be small enough to hide it.
        """
        out = self._new(tmp_path, measurement_error=sigma)

        temps = sorted({round(d["T"], 6) for d in out["designs"]})
        assert temps == [pytest.approx(300.0, abs=1e-3), pytest.approx(500.0, abs=1e-3)]

        # And the design is actually identifiable: full-rank information with
        # no ridge propping it up.
        model = SymbolicModel(
            source=self.EXPR,
            parameter_names=("k0", "Ea"),
            input_names=("T",),
            response_name="rate",
            measurement_error=sigma,
        )
        fim = model.fim({"k0": 2.0, "Ea": 5000.0}, [{"T": d["T"]} for d in out["designs"]])
        assert np.linalg.matrix_rank(fim) == 2

    def test_parameter_order_follows_the_param_flags(self, tmp_path) -> None:
        """Not alphabetical: the order given is the FIM layout."""
        out = self._new(
            tmp_path,
            expression="A * exp(-t / tau) + C",
            inputs=[("t", 0.0, 12.0)],
            param_initial_guess={"tau": 2.0, "C": 0.1, "A": 4.0},
            n=5,
        )
        assert out["parameter_names"] == ["tau", "C", "A"]

    def test_fit_recovers_truth_from_a_distant_guess(self, tmp_path) -> None:
        from discopt.doe.cli import do_fit

        out = self._new(tmp_path)
        self._fill(out["workbook_path"])
        fitted = do_fit({"workbook": out["workbook_path"]})

        assert fitted["converged"]
        estimates = {p["name"]: p["estimate"] for p in fitted["parameters"]}
        for name, want in self.TRUTH.items():
            assert estimates[name] == pytest.approx(want, rel=1e-5)
        # Machine precision for a response of order 1 over 6 runs. Unlike the
        # OLS templates this is an iterative solve, so it lands at solver
        # tolerance rather than exactly zero.
        assert fitted["objective"] < 1e-12
        assert fitted["expression"] == self.EXPR

    def test_extend_recentres_on_the_fitted_parameters(self, tmp_path) -> None:
        """Sequential design: each round uses what the data now say."""
        from discopt.doe.cli import ExtendParams, do_extend, do_fit

        out = self._new(tmp_path)
        path = out["workbook_path"]
        self._fill(path, noise=0.005)
        do_fit({"workbook": path})

        extended = do_extend(ExtendParams(workbook=Path(path), n=4, n_starts=6))
        assert len(extended["new_run_ids"]) == 4
        # The design parameters are the fitted values, not the original guess.
        assert extended["design_parameters"]["Ea"] == pytest.approx(6200.0, rel=1e-2)
        assert extended["design_parameters"]["Ea"] != pytest.approx(5000.0, rel=1e-3)
        for d in extended["designs"]:
            assert 300.0 <= d["T"] <= 500.0

    def test_extend_improves_the_criterion(self, tmp_path) -> None:
        from discopt.doe.cli import ExtendParams, do_extend, do_fit
        from discopt.doe.linear_design import D_OPTIMAL, evaluate_criterion

        out = self._new(tmp_path)
        path = out["workbook_path"]
        self._fill(path, noise=0.005)
        first = do_fit({"workbook": path})
        before = first["log_det_fim"]

        extended = do_extend(ExtendParams(workbook=Path(path), n=4, n_starts=6))
        assert extended["criterion_value"] > before

        from discopt.doe.workbook import Workbook

        model = Workbook.open(Path(path)).symbolic_model()
        assert (
            evaluate_criterion(
                model.fim(extended["design_parameters"], extended["designs"]), D_OPTIMAL
            )
            > -np.inf
        )

    def test_workbook_stores_the_expression_not_code(self, tmp_path) -> None:
        """What round-trips is data; reopening runs nothing."""
        from discopt.doe.workbook import Workbook

        out = self._new(tmp_path)
        wb = Workbook.open(Path(out["workbook_path"]))
        args = wb.template_args()
        assert args["expression"] == self.EXPR
        assert args["parameters"] == ["k0", "Ea"]
        assert args["inputs"] == ["T"]

        model = wb.symbolic_model()
        assert model.predict({"k0": 1.0, "Ea": 0.0}, {"T": 400.0}) == pytest.approx(1.0)
        assert wb.parameter_names() == ["k0", "Ea"]

    def test_a_hostile_expression_in_a_workbook_is_refused_on_open(self, tmp_path) -> None:
        """The threat model that motivated storing expressions rather than source."""
        import openpyxl

        from discopt.doe.workbook import Workbook

        out = self._new(tmp_path)
        path = Path(out["workbook_path"])
        canary = tmp_path / "canary.txt"

        book = openpyxl.load_workbook(path)
        sheet = book["metadata"]
        for row in sheet.iter_rows(min_row=2):
            if row[0].value == "template_args":
                row[1].value = (
                    '{"family": "symbolic", "inputs": ["T"], "parameters": ["k0"], '
                    f'"expression": "__import__(\'pathlib\').Path({str(canary)!r}).touch()"}}'
                )
        book.save(path)

        with pytest.raises(ModelSyntaxError):
            Workbook.open(path).symbolic_model()
        assert not canary.exists()

    def test_new_rejects_an_underdetermined_run_count(self, tmp_path) -> None:
        from discopt.doe.cli import DoEError

        with pytest.raises(DoEError, match="at least 2 runs"):
            self._new(tmp_path, n=1)

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"expression": None}, "requires --expr"),
            ({"param_initial_guess": {}}, "requires --param"),
            ({"expression": "k0 * bogus"}, "unknown name 'bogus'"),
            ({"expression": "k0 * ("}, "could not parse"),
        ],
    )
    def test_new_reports_bad_models_clearly(self, tmp_path, kw, match) -> None:
        from discopt.doe.cli import DoEError

        with pytest.raises(DoEError, match=match):
            self._new(tmp_path, **kw)
