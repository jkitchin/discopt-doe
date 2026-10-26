"""Trade-off curves for optimal design: information against a second objective.

When to use this module
-----------------------

:func:`~discopt.doe.optimal_experiment` answers "which experiment is most
informative?". Real campaigns rarely have only that objective. The most
informative experiment may run for thirty hours, sit at the hottest corner the
rig allows, or produce material nobody can sell. The question a planner
actually asks is *how much information does each hour, degree or gram cost?*

:func:`design_tradeoff` answers it with the **ε-constraint method**
(Haimes, Lasdon & Wismer 1971; Miettinen 1999): hold the second objective to a
level ε, maximize the information criterion subject to it, and sweep ε. Each
solve is an ordinary constrained :func:`optimal_experiment` call, so every
point on the curve is an optimal design in its own right, and the curve is the
Pareto front of (information, second objective). Unlike a weighted sum of the
two, the ε-constraint reaches every point of a non-convex front, and its axes
are in units a planner can read -- "hours", "kelvin", "percent yield" -- rather
than in an arbitrary weight.

Information is reported as **efficiency** relative to the unconstrained optimum,
so the front reads as "this design keeps 90 % of the information at a third of
the duration". For D-optimality the efficiency is
``(det F / det F*)^(1/p)``: an efficiency of 0.5 means the design needs twice
as many replicates to match the optimum's precision.

The second objective is either

* a **response** of the experiment (a string), predicted at the nominal
  parameters with :func:`~discopt.doe.predict_responses` -- a yield the run
  must also deliver, a concentration it must stay under; or
* any **function of the design** (a callable ``g(design) -> float``) --
  experiment duration, reagent cost, a temperature setting, or a prediction
  from a second experiment that models something not measured.

Quick start
-----------

>>> from discopt.doe import design_tradeoff
>>> front = design_tradeoff(
...     experiment, theta, {"t1": (0.1, 30), "t2": (0.1, 30), "t3": (0.1, 30)},
...     tradeoff=lambda d: max(d.values()),      # experiment duration
...     sense="minimize",
...     n_levels=8,
... )
>>> front.knee                                   # the best-value compromise
>>> front.plot()

References
----------

* Haimes, Lasdon, Wismer (1971). *On a bicriterion formulation of the problems
  of integrated system identification and system optimization.* IEEE Trans.
  Syst. Man Cybern. 1:296-297. -- the ε-constraint method.
* Miettinen (1999). *Nonlinear Multiobjective Optimization.* Kluwer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discopt.doe.design import (
    DesignCriterion,
    DesignResult,
    _is_maximization,
    _multi_start_candidates,
    _response_bound_constraints,
    optimal_experiment,
)
from discopt.doe.fim import predict_responses
from discopt.estimate import Experiment


@dataclass
class TradeoffPoint:
    """One optimal design on the trade-off curve."""

    level: float
    """The bound ε the second objective was held to."""
    tradeoff_value: float
    """The second objective at the design found (at or inside ``level``)."""
    criterion_value: float
    """The information criterion at the design (as :func:`optimal_experiment` reports it)."""
    efficiency: float
    """Information relative to the unconstrained optimum, in (0, 1]."""
    design: dict[str, float]
    result: DesignResult
    on_front: bool = True
    """False when another point is at least as good on both objectives and
    better on one: the solver found a local optimum at this level."""


@dataclass
class TradeoffResult:
    """The information-vs-second-objective front from :func:`design_tradeoff`.

    ``points`` are ordered from the unconstrained optimum (efficiency 1) to the
    tightest level; ``reference`` is the unconstrained optimal design.
    """

    points: list[TradeoffPoint]
    reference: DesignResult
    criterion: str
    tradeoff_name: str
    sense: str
    failed_levels: list[float] = field(default_factory=list)
    """Levels at which no feasible, informative design was found."""

    @property
    def front(self) -> list[TradeoffPoint]:
        """The non-dominated points only."""
        return [p for p in self.points if p.on_front]

    @property
    def knee(self) -> TradeoffPoint:
        """The point where giving up more of the second objective stops buying information.

        The front point farthest from the straight line joining its two ends, with
        both axes scaled to 0-1 (the "maximum distance to the chord" knee). It is
        a starting point for a decision, not the decision: whether an hour is worth
        two percent of efficiency is a judgement the data cannot make.
        """
        pts = self.front
        if len(pts) < 3:
            return pts[0]
        x = np.array([p.tradeoff_value for p in pts], dtype=float)
        y = np.array([p.efficiency for p in pts], dtype=float)
        xs = (x - x.min()) / (np.ptp(x) or 1.0)
        ys = (y - y.min()) / (np.ptp(y) or 1.0)
        a = np.array([xs[0], ys[0]])
        b = np.array([xs[-1], ys[-1]])
        chord = b - a
        norm = np.hypot(*chord) or 1.0
        dist = np.abs(chord[0] * (ys - a[1]) - chord[1] * (xs - a[0])) / norm
        return pts[int(np.argmax(dist))]

    def rows(self) -> list[dict[str, Any]]:
        """One dict per point: the level, both objectives and the design."""
        return [
            {
                "level": p.level,
                self.tradeoff_name: p.tradeoff_value,
                "criterion_value": p.criterion_value,
                "efficiency": p.efficiency,
                "on_front": p.on_front,
                **p.design,
            }
            for p in self.points
        ]

    def summary(self) -> str:
        """A text table of the front."""
        names = list(self.reference.design)
        head = f"{self.tradeoff_name:>14s} {'efficiency':>10s}  " + "  ".join(
            f"{n:>9s}" for n in names
        )
        lines = [
            f"Trade-off: {self.criterion} information vs {self.tradeoff_name} ({self.sense})",
            head,
            "-" * len(head),
        ]
        knee = self.knee
        for p in self.points:
            mark = " <- knee" if p is knee else ("" if p.on_front else " (dominated)")
            lines.append(
                f"{p.tradeoff_value:14.4g} {p.efficiency:10.3f}  "
                + "  ".join(f"{p.design[n]:9.4g}" for n in names)
                + mark
            )
        if self.failed_levels:
            lines.append(f"no feasible design at level(s) {self.failed_levels}")
        return "\n".join(lines)

    def plot(self, ax: Any = None, *, annotate_knee: bool = True, **kwargs: Any) -> Any:
        """Efficiency against the second objective; returns the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.6))
        front = self.front
        x = [p.tradeoff_value for p in front]
        y = [p.efficiency for p in front]
        style = {"marker": "o", "lw": 1.8, **kwargs}
        ax.plot(x, y, **style)
        if annotate_knee and len(front) >= 3:
            k = self.knee
            ax.plot([k.tradeoff_value], [k.efficiency], "o", ms=12, mfc="none", mec="C3", mew=2)
            ax.annotate(
                f"knee: {k.efficiency:.0%} at {k.tradeoff_value:.3g}",
                (k.tradeoff_value, k.efficiency),
                textcoords="offset points",
                xytext=(10, -14),
                color="C3",
            )
        ax.set_xlabel(self.tradeoff_name)
        ax.set_ylabel(f"{_criterion_letter(self.criterion)}-efficiency")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        return ax


_LETTERS = {
    DesignCriterion.D_OPTIMAL: "D",
    DesignCriterion.A_OPTIMAL: "A",
    DesignCriterion.E_OPTIMAL: "E",
    DesignCriterion.ME_OPTIMAL: "ME",
    DesignCriterion.I_OPTIMAL: "I",
    DesignCriterion.G_OPTIMAL: "G",
}


def _criterion_letter(criterion: str) -> str:
    return _LETTERS.get(criterion, str(criterion))


def _efficiency(value: float, reference: float, criterion: str, p: int) -> float:
    """Information at ``value`` relative to the unconstrained ``reference``."""
    if not np.isfinite(value) or not np.isfinite(reference):
        return 0.0
    if criterion == DesignCriterion.D_OPTIMAL:
        return float(np.exp((value - reference) / p))  # (det F / det F*)^(1/p)
    if _is_maximization(criterion):  # E: smallest eigenvalue
        return float(value / reference) if reference > 0 else 0.0
    return float(reference / value) if value > 0 else 0.0  # A, ME, I, G: smaller is better


def _resolve_tradeoff(
    experiment: Experiment,
    param_values: Mapping[str, float],
    tradeoff: str | Callable[[dict[str, float]], float],
) -> tuple[Callable[[dict[str, float]], float], str]:
    if callable(tradeoff):
        return (
            lambda d: float(tradeoff(dict(d))),
            getattr(tradeoff, "__name__", "tradeoff").replace("<lambda>", "tradeoff"),
        )
    name = str(tradeoff)
    responses = experiment.create_model(**dict(param_values)).response_names
    if name not in responses:
        raise ValueError(
            f"tradeoff {name!r} is not a response of the experiment ({list(responses)}); "
            "pass a callable g(design) -> float for anything else"
        )
    theta = dict(param_values)
    return (lambda d: float(predict_responses(experiment, theta, dict(d))[name])), name


def design_tradeoff(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    tradeoff: str | Callable[[dict[str, float]], float],
    sense: str = "minimize",
    levels: Sequence[float] | None = None,
    n_levels: int = 8,
    criterion: str = DesignCriterion.D_OPTIMAL,
    tradeoff_name: str | None = None,
    inequality_constraints: Sequence[Callable[[dict[str, float]], float]] | None = None,
    seed: int = 42,
    **optimal_experiment_kwargs: Any,
) -> TradeoffResult:
    """Sweep the information-vs-``tradeoff`` Pareto front by the ε-constraint method.

    For each level ε, solves ``optimal_experiment`` with the extra constraint
    ``tradeoff(design) <= ε`` (``sense="minimize"``: duration, cost, impurity)
    or ``>= ε`` (``sense="maximize"``: a yield the run must also deliver).

    Parameters
    ----------
    experiment, param_values, design_bounds, criterion
        As for :func:`~discopt.doe.optimal_experiment`.
    tradeoff : str or callable
        A response name of ``experiment`` (predicted at ``param_values``), or a
        function ``g(design: dict) -> float`` of the design.
    sense : {"minimize", "maximize"}, default "minimize"
        Which direction of ``tradeoff`` is desirable.
    levels : sequence of float, optional
        The ε values to solve at. By default ``n_levels`` evenly spaced levels
        from the value at the unconstrained optimum to the best value found on a
        Sobol sample of the design box.
    n_levels : int, default 8
        Number of levels when ``levels`` is not given (the unconstrained
        optimum is added as the first point either way).
    tradeoff_name : str, optional
        Axis label; defaults to the response name or the callable's name.
    inequality_constraints : sequence of callable, optional
        Constraints ``h(design) >= 0`` that hold at every level.
    seed : int, default 42
    **optimal_experiment_kwargs
        Passed to every :func:`optimal_experiment` call (``prior_fim``,
        ``response_bounds``, ``n_starts``, ``scale_parameters``, ...).

    Returns
    -------
    TradeoffResult
        ``points`` from the unconstrained optimum to the tightest level, each an
        optimal design with its efficiency; ``knee``, ``summary()`` and ``plot()``.

    Notes
    -----
    The levels are solved in order from loose to tight, each warm-started from
    the previous design (continuation), and a point that a tighter one beats on
    both axes is re-solved from that tighter design. Points that are still
    dominated afterwards are kept but flagged ``on_front=False``.

    With ``prior_fim`` the criterion, and so the efficiency, is that of the prior
    plus the new design: the precision the campaign ends up with, not what the
    new experiment alone adds. ``response_bounds`` hold at every level, and the
    automatic level range is taken over designs that satisfy them.
    """
    if sense not in ("minimize", "maximize"):
        raise ValueError(f"sense must be 'minimize' or 'maximize', got {sense!r}")
    if "initial_designs" in optimal_experiment_kwargs:
        raise TypeError("design_tradeoff sets initial_designs itself (continuation)")
    g, default_name = _resolve_tradeoff(experiment, param_values, tradeoff)
    name = tradeoff_name or default_name
    sign = 1.0 if sense == "minimize" else -1.0  # internally: smaller sign*g is better
    base_ineq = list(inequality_constraints or [])
    response_bounds = optimal_experiment_kwargs.pop("response_bounds", None)
    if response_bounds:
        # Applied here rather than passed through, so that the automatic level
        # range is taken over designs that satisfy them too.
        base_ineq += _response_bound_constraints(experiment, param_values, response_bounds)

    def solve(level: float | None, starts: list[dict[str, float]]) -> DesignResult:
        ineq = list(base_ineq)
        if level is not None:
            ineq.append(lambda d, eps=float(level): sign * (eps - g(d)))
        return optimal_experiment(
            experiment,
            param_values,
            design_bounds,
            criterion=criterion,
            inequality_constraints=ineq or None,
            initial_designs=starts or None,
            seed=seed,
            **optimal_experiment_kwargs,
        )

    reference = solve(None, [])
    g_ref = g(reference.design)
    if levels is None:
        sample = _multi_start_candidates(design_bounds, 512, seed)
        feasible = [d for d in sample if all(h(d) >= 0.0 for h in base_ineq)]
        best = min(sign * g(d) for d in feasible or sample)
        if best >= sign * g_ref:
            raise ValueError(
                f"the unconstrained optimum already has the best {name} on the design "
                "box: there is no trade-off to sweep. Pass levels= to probe it anyway."
            )
        # The extreme level itself usually admits only a degenerate experiment
        # (every sample at t = 0), so stop just short of it.
        span = sign * g_ref - best
        sweep = [sign * (sign * g_ref - span * f) for f in np.linspace(0.0, 0.97, n_levels + 1)[1:]]
    else:
        # Loose to tight, so each solve can warm-start from the last.
        sweep = sorted((float(v) for v in levels), key=lambda v: -sign * v)

    p = reference.fim.shape[0]

    def point(level: float, res: DesignResult) -> TradeoffPoint:
        return TradeoffPoint(
            level=float(level),
            tradeoff_value=g(res.design),
            criterion_value=float(res.criterion_value),
            efficiency=_efficiency(res.criterion_value, reference.criterion_value, criterion, p),
            design=dict(res.design),
            result=res,
        )

    points = [point(g_ref, reference)]
    failed: list[float] = []
    for level in sweep:
        if sign * g_ref <= sign * level:  # the optimum already satisfies it
            points.append(point(level, reference))
            continue
        try:
            res = solve(level, [points[-1].design, reference.design])
        except RuntimeError:
            failed.append(float(level))
            continue
        pt = point(level, res)
        if pt.efficiency <= 0.0:
            failed.append(float(level))
            continue
        points.append(pt)

    # A tighter design is feasible at every looser level, so a looser point that
    # it beats is a local optimum: re-solve it from the tighter design.
    for i in range(len(points) - 2, 0, -1):
        tighter = max(points[i + 1 :], key=lambda q: q.efficiency)
        if tighter.efficiency > points[i].efficiency + 1e-9:
            try:
                res = solve(points[i].level, [tighter.design, points[i].design])
            except RuntimeError:
                continue
            candidate = point(points[i].level, res)
            if candidate.efficiency > points[i].efficiency:
                points[i] = candidate

    for i, pi in enumerate(points):
        pi.on_front = not any(
            q.efficiency >= pi.efficiency
            and sign * q.tradeoff_value <= sign * pi.tradeoff_value
            and (q.efficiency > pi.efficiency or sign * q.tradeoff_value < sign * pi.tradeoff_value)
            for j, q in enumerate(points)
            if j != i
        )

    return TradeoffResult(
        points=points,
        reference=reference,
        criterion=criterion,
        tradeoff_name=name,
        sense=sense,
        failed_levels=failed,
    )


__all__ = ["TradeoffPoint", "TradeoffResult", "design_tradeoff"]
