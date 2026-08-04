"""User-defined response models, differentiated symbolically.

The built-in templates in :mod:`discopt.doe.templates` are all linear in their
parameters, which is what lets :mod:`discopt.doe.linear_design` compute their
Fisher information in closed form. Real kinetics are usually not: an Arrhenius
rate ``k0 * exp(-Ea / (R * T))`` has a Jacobian that genuinely depends on the
parameter values, so designing for it needs derivatives of an arbitrary
expression.

This module provides them with sympy rather than jax:

    ∂y/∂θ  via  sympy.diff,  evaluated through  sympy.lambdify(..., "numpy")

The result is exactly what the autodiff path computes — ``tests/test_symbolic.py``
pins the two together on a nonlinear model — but it needs only sympy and numpy,
both of which have WebAssembly builds where jax does not. It is what lets you
write your own model in the browser app.

Expressions as data, not code
-----------------------------
A model has to survive a round trip through the campaign workbook so that
``fit`` and ``extend`` can rebuild it, which means model definitions arrive in
*files*, not just from the person who typed them. Storing executable source and
running it on open would make an emailed workbook a code-execution vector, so
what is stored is the expression itself.

:func:`parse_expression` reads that expression without ``eval`` at any point:
it walks Python's own AST and constructs the sympy tree node by node, accepting
only arithmetic, the functions in :data:`ALLOWED_FUNCTIONS`, and names declared
as parameters or inputs. Anything else — an attribute access, a call to an
unlisted name, a lambda, a subscript — is a parse error, not a silent success.
``sympy.sympify`` and ``sympy.parsing.parse_expr`` are deliberately *not* used;
both ultimately call ``eval``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def _require_sympy():
    """Return the sympy module, or raise with an actionable message."""
    try:
        import sympy
    except ImportError as e:  # pragma: no cover - sympy is a declared dependency
        raise ImportError(
            "user-defined models need sympy. Install it with: pip install sympy "
            "(it ships as a dependency of discopt-doe)."
        ) from e
    return sympy


def _function_table() -> dict[str, Any]:
    sp = _require_sympy()
    return {
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "erf": sp.erf,
    }


#: Function names a model expression may call.
ALLOWED_FUNCTIONS = (
    "exp", "log", "ln", "log10", "sqrt",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "abs", "Abs", "erf",
)  # fmt: skip

#: Constants a model expression may reference by name.
ALLOWED_CONSTANTS = ("pi", "E")

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


class ModelSyntaxError(ValueError):
    """Raised when a model expression cannot be parsed or uses a forbidden construct."""


def parse_expression(source: str, names: Sequence[str]):
    """Parse ``source`` into a sympy expression over ``names``.

    No ``eval`` is involved at any stage: the string is parsed to a Python AST,
    which is then walked and rebuilt as a sympy tree, rejecting every node type
    that is not arithmetic, a number, an allowed function call, or one of
    ``names``.

    Parameters
    ----------
    source : str
        The response expression, e.g. ``"k0 * exp(-Ea / (8.314 * T))"``.
    names : sequence of str
        Symbol names the expression may use — the model's parameters and design
        inputs. A name outside this list (and outside
        :data:`ALLOWED_CONSTANTS`) is an error, which is what turns a typo into
        a message rather than a silently-introduced free variable.

    Returns
    -------
    sympy.Expr

    Raises
    ------
    ModelSyntaxError
        On a syntax error, an unknown name, or a forbidden construct.
    """
    sp = _require_sympy()
    text = (source or "").strip()
    if not text:
        raise ModelSyntaxError("model expression is empty")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ModelSyntaxError(f"could not parse model expression: {e.msg}") from e

    symbols = {n: sp.Symbol(n, real=True) for n in names}
    constants = {"pi": sp.pi, "E": sp.E}
    functions = _function_table()

    def build(node: ast.AST):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ModelSyntaxError(f"only numeric literals are allowed, got {node.value!r}")
            return sp.nsimplify(node.value, rational=False)

        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in constants:
                return constants[node.id]
            known = ", ".join(sorted(names))
            raise ModelSyntaxError(
                f"unknown name {node.id!r} in model expression; declared symbols are: {known}"
            )

        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ModelSyntaxError(
                    f"operator {type(node.op).__name__} is not allowed in a model expression"
                )
            return op(build(node.left), build(node.right))

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -build(node.operand)
            if isinstance(node.op, ast.UAdd):
                return build(node.operand)
            raise ModelSyntaxError(
                f"unary {type(node.op).__name__} is not allowed in a model expression"
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ModelSyntaxError("only direct calls to the allowed functions are permitted")
            if node.func.id not in functions:
                allowed = ", ".join(sorted(set(ALLOWED_FUNCTIONS)))
                raise ModelSyntaxError(
                    f"function {node.func.id!r} is not allowed; available: {allowed}"
                )
            if node.keywords:
                raise ModelSyntaxError(f"{node.func.id}() takes no keyword arguments here")
            return functions[node.func.id](*[build(a) for a in node.args])

        raise ModelSyntaxError(
            f"{type(node).__name__} is not allowed in a model expression; "
            "use arithmetic, numbers, the declared symbols, and the allowed functions"
        )

    return build(tree.body)


@dataclass
class SymbolicModel:
    """A user-defined response model with symbolic derivatives.

    Attributes
    ----------
    source : str
        The expression as written, kept verbatim for display and round-tripping.
    parameter_names : tuple of str
        Unknown parameters, in the order that fixes the FIM row/column layout.
    input_names : tuple of str
        Design factors.
    response_name : str
        Name of the measured response column.
    measurement_error : float
        Response standard deviation σ.
    """

    source: str
    parameter_names: tuple[str, ...]
    input_names: tuple[str, ...]
    response_name: str = "y"
    measurement_error: float = 1.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.parameter_names = tuple(self.parameter_names)
        self.input_names = tuple(self.input_names)
        if not self.parameter_names:
            raise ValueError("a model needs at least one unknown parameter")
        if not self.input_names:
            raise ValueError("a model needs at least one design input")
        overlap = set(self.parameter_names) & set(self.input_names)
        if overlap:
            raise ValueError(f"names used as both parameter and input: {sorted(overlap)}")
        if float(self.measurement_error) <= 0.0:
            raise ValueError(f"measurement error must be positive, got {self.measurement_error!r}")
        # Parse eagerly: a bad expression should fail where it was entered, not
        # later inside an optimizer callback.
        self.expression  # noqa: B018

    # ── compiled artefacts, built once and reused ──────────────────────

    @property
    def expression(self):
        """The parsed sympy expression."""
        if "expr" not in self._cache:
            names = (*self.parameter_names, *self.input_names)
            self._cache["expr"] = parse_expression(self.source, names)
        return self._cache["expr"]

    @property
    def jacobian_expressions(self) -> list[Any]:
        """``[∂y/∂θ₁, …]`` as sympy expressions, ordered by ``parameter_names``."""
        if "jac_expr" not in self._cache:
            sp = _require_sympy()
            syms = [sp.Symbol(n, real=True) for n in self.parameter_names]
            self._cache["jac_expr"] = [sp.diff(self.expression, s) for s in syms]
        return self._cache["jac_expr"]

    def _lambdified(self, key: str, exprs):
        if key not in self._cache:
            sp = _require_sympy()
            p = [sp.Symbol(n, real=True) for n in self.parameter_names]
            x = [sp.Symbol(n, real=True) for n in self.input_names]
            self._cache[key] = sp.lambdify((p, x), exprs, "numpy")
        return self._cache[key]

    # ── evaluation ─────────────────────────────────────────────────────

    def _vectors(
        self, theta: Mapping[str, float], design: Mapping[str, float]
    ) -> tuple[list[float], list[float]]:
        try:
            p = [float(theta[n]) for n in self.parameter_names]
        except KeyError as e:
            raise KeyError(f"missing parameter value for {e.args[0]!r}") from e
        try:
            x = [float(design[n]) for n in self.input_names]
        except KeyError as e:
            raise KeyError(f"missing design value for {e.args[0]!r}") from e
        return p, x

    def predict(self, theta: Mapping[str, float], design: Mapping[str, float]) -> float:
        """Evaluate the response at one design point."""
        fn = self._lambdified("predict", self.expression)
        p, x = self._vectors(theta, design)
        return float(fn(p, x))

    def jacobian_row(self, theta: Mapping[str, float], design: Mapping[str, float]) -> np.ndarray:
        """Return ``∂y/∂θ`` at one design point, ordered by ``parameter_names``.

        Unlike the linear templates, this genuinely depends on ``theta`` — which
        is why designing a nonlinear experiment needs a nominal parameter guess
        and is only locally optimal around it.
        """
        fn = self._lambdified("jac", self.jacobian_expressions)
        p, x = self._vectors(theta, design)
        row = np.asarray(fn(p, x), dtype=np.float64).ravel()
        if row.size != len(self.parameter_names):
            # lambdify collapses a derivative that is structurally constant to a
            # scalar; broadcast it back to the full row.
            row = np.broadcast_to(row, (len(self.parameter_names),)).astype(np.float64)
        if not np.all(np.isfinite(row)):
            raise ValueError(
                f"model Jacobian is not finite at {dict(design)} with parameters "
                f"{dict(theta)}; check for division by zero or overflow"
            )
        return row

    def design_matrix(
        self, theta: Mapping[str, float], designs: Iterable[Mapping[str, float]]
    ) -> np.ndarray:
        """Stack :meth:`jacobian_row` over ``designs``."""
        rows = [self.jacobian_row(theta, d) for d in designs]
        if not rows:
            return np.zeros((0, len(self.parameter_names)), dtype=np.float64)
        return np.vstack(rows)

    def fim(self, theta: Mapping[str, float], designs: Iterable[Mapping[str, float]]) -> np.ndarray:
        """Fisher information ``JᵀJ/σ²`` accumulated over ``designs``."""
        J = self.design_matrix(theta, designs)
        return J.T @ J / (float(self.measurement_error) ** 2)

    # ── persistence ────────────────────────────────────────────────────

    def to_metadata(self) -> dict[str, Any]:
        """Return the JSON-able form stored in a workbook's ``template_args``.

        Only the expression text and the name lists — never executable source.
        Reading it back goes through :func:`parse_expression`, so opening a
        workbook someone sent you evaluates nothing.
        """
        return {
            "family": "symbolic",
            "expression": self.source,
            "parameters": list(self.parameter_names),
            "inputs": list(self.input_names),
        }

    @classmethod
    def from_metadata(
        cls,
        meta: Mapping[str, Any],
        *,
        response_name: str = "y",
        measurement_error: float = 1.0,
    ) -> "SymbolicModel":
        """Rebuild a model from workbook metadata."""
        missing = [k for k in ("expression", "parameters", "inputs") if not meta.get(k)]
        if missing:
            raise ValueError(
                f"workbook metadata is missing {missing} for a user-defined model; "
                "it may have been written by an older version"
            )
        return cls(
            source=str(meta["expression"]),
            parameter_names=tuple(meta["parameters"]),
            input_names=tuple(meta["inputs"]),
            response_name=response_name,
            measurement_error=float(measurement_error),
        )

    def pretty(self) -> str:
        """A readable one-line rendering of the model and its derivatives."""
        lines = [f"{self.response_name} = {self.expression}"]
        for name, expr in zip(self.parameter_names, self.jacobian_expressions):
            lines.append(f"  d{self.response_name}/d{name} = {expr}")
        return "\n".join(lines)


def fit_least_squares(
    model: SymbolicModel,
    rows: Sequence[Mapping[str, Any]],
    initial: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    max_nfev: int | None = None,
) -> dict[str, Any]:
    """Fit a user-defined model to completed runs by nonlinear least squares.

    Uses the analytic Jacobian from sympy rather than finite differences, which
    makes the solve both faster and better conditioned.

    Parameters
    ----------
    model : SymbolicModel
    rows : sequence of mapping
        Completed runs; each must carry every input name and the response.
    initial : mapping
        Starting parameter values. A nonlinear fit needs them, and a poor guess
        can converge to a different local optimum.
    bounds : mapping name -> (lo, hi), optional
        Per-parameter bounds.
    max_nfev : int, optional
        Cap on residual evaluations.

    Returns
    -------
    dict
        ``estimates``, ``std_errors``, ``ci_lower``/``ci_upper`` (95%),
        ``residual_sum_of_squares``, ``n_observations``, ``fim``, ``success``,
        and the solver ``message``.
    """
    from scipy.optimize import least_squares

    names = list(model.parameter_names)
    n_p = len(names)
    obs = list(rows)
    if not obs:
        raise ValueError("no completed runs to fit")

    y = np.array([float(r[model.response_name]) for r in obs], dtype=np.float64)
    designs = [{n: float(r[n]) for n in model.input_names} for r in obs]

    def unpack(vec: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(names, vec)}

    def residuals(vec: np.ndarray) -> np.ndarray:
        theta = unpack(vec)
        return np.array([model.predict(theta, d) for d in designs], dtype=np.float64) - y

    def jac(vec: np.ndarray) -> np.ndarray:
        return model.design_matrix(unpack(vec), designs)

    x0 = np.array([float(initial.get(n, 1.0)) for n in names], dtype=np.float64)

    kwargs: dict[str, Any] = {"jac": jac}
    if bounds:
        lo = np.array([float(bounds.get(n, (-np.inf, np.inf))[0]) for n in names])
        hi = np.array([float(bounds.get(n, (-np.inf, np.inf))[1]) for n in names])
        kwargs["bounds"] = (lo, hi)
        x0 = np.clip(x0, lo, hi)
    if max_nfev is not None:
        kwargs["max_nfev"] = int(max_nfev)

    result = least_squares(residuals, x0, **kwargs)
    theta = unpack(result.x)
    rss = float(np.sum(result.fun**2))
    n_obs = len(obs)

    # Covariance from the Jacobian at the solution. Prefer the residual-based
    # variance estimate when there are degrees of freedom left, since the
    # declared sigma is often a guess; fall back to the declared one otherwise.
    J = model.design_matrix(theta, designs)
    dof = n_obs - n_p
    sigma2 = rss / dof if dof > 0 else float(model.measurement_error) ** 2
    fim = J.T @ J / (float(model.measurement_error) ** 2)

    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J) * sigma2

    diag = np.diag(cov)
    std_errors = np.sqrt(np.where(diag >= 0, diag, np.nan))

    if dof > 0:
        from scipy.stats import t as t_dist

        crit = float(t_dist.ppf(0.975, dof))
    else:
        crit = float("nan")

    estimates = np.array([theta[n] for n in names], dtype=np.float64)
    return {
        "parameter_names": names,
        "estimates": {n: float(v) for n, v in zip(names, estimates)},
        "std_errors": {n: float(s) for n, s in zip(names, std_errors)},
        "ci_lower": {n: float(e - crit * s) for n, e, s in zip(names, estimates, std_errors)},
        "ci_upper": {n: float(e + crit * s) for n, e, s in zip(names, estimates, std_errors)},
        "residual_sum_of_squares": rss,
        "n_observations": n_obs,
        "degrees_of_freedom": dof,
        "fim": fim,
        "success": bool(result.success),
        "message": str(result.message),
    }


def basis_evaluator(
    model: SymbolicModel, theta: Mapping[str, float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(x_vector) -> ∂y/∂θ`` at fixed nominal parameters.

    This is the adapter that lets the design search in
    :mod:`discopt.doe.linear_design` drive a nonlinear model unchanged: that
    search only ever asks for the Jacobian row at a candidate point, and does
    not care whether it came from a basis function or from sympy.
    """
    names = list(model.input_names)
    nominal = dict(theta)

    def basis(x: np.ndarray) -> np.ndarray:
        return model.jacobian_row(nominal, {n: float(v) for n, v in zip(names, x)})

    return basis


__all__ = [
    "ALLOWED_CONSTANTS",
    "ALLOWED_FUNCTIONS",
    "ModelSyntaxError",
    "SymbolicModel",
    "basis_evaluator",
    "fit_least_squares",
    "parse_expression",
]
