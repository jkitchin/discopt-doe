"""A fitted model as a file: the model form, the estimates, and their covariance.

A fitted model is only reusable if everything a prediction needs travels with
it. That is four things: the model form, the parameter estimates, their
covariance, and the noise estimate behind that covariance.
Keep three and drop the fourth and the model still *predicts* -- it just can no
longer say how well.

:class:`ModelCard` is that bundle, written as JSON:

    card = ModelCard.from_fit(model, fit, design_region=BOUNDS, rows=rows)
    card.save("kinetics.json")

    later = ModelCard.load("kinetics.json")
    later.predict({"S": 10.0, "T": 330.0})
    later.interval({"S": 10.0, "T": 330.0}, kind="prediction")

Reading a card back parses the expression rather than executing it (see
:func:`~discopt.doe.symbolic.parse_expression`), so opening one someone sent you
runs none of their code. That is the reason the model travels as *text* and not
as a pickle.

What a card deliberately does not carry is the data. The runs belong in the
workbook (or a CSV) beside it: a card is enough to *use* a model, never enough
to re-fit or re-check one, and pretending otherwise invites a reader to trust a
number whose provenance they cannot audit. What it does carry is a digest of
the rows it was fitted to, so a card and a data file can be matched up later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from discopt.doe.symbolic import SymbolicModel

SCHEMA_VERSION = 1

__all__ = ["ModelCard", "SCHEMA_VERSION"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _version() -> str:
    """``discopt-doe <version>``, or just the name if it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"discopt-doe {version('discopt-doe')}"
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "discopt-doe"


def digest_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """A short, order-independent digest of the runs a model was fitted to.

    Not a substitute for keeping the data: it identifies which data, so a card
    and a workbook can be matched up long after both were written.
    """
    payload = json.dumps(
        [{k: _plain(v) for k, v in sorted(row.items())} for row in rows],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _plain(value: Any) -> Any:
    """JSON-able form of a cell value (numpy scalars, dates, everything else)."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ModelCard:
    """A fitted model, its uncertainty, and where it is allowed to be used.

    Attributes
    ----------
    model : SymbolicModel
        The model form, carrying its response name and declared sigma.
    estimates : dict
        Fitted parameter values, keyed by parameter name.
    covariance : numpy.ndarray
        Parameter covariance, rows and columns in ``model.parameter_names``
        order. This is the covariance *behind the reported standard errors*,
        on the ``sigma`` scale below -- not the declared-sigma information
        matrix a design search uses.
    sigma : float
        The noise level behind ``covariance``.
    sigma_source : str
        ``"residual"`` (estimated from this fit), ``"declared"`` (the model's
        own measurement error) or ``"given"``.
    n_observations, degrees_of_freedom : int
        How much data stands behind the fit, and how much of it was left over
        after estimating the parameters. ``degrees_of_freedom`` sets the
        critical value in :meth:`interval`.
    level : float
        Confidence level for :meth:`interval` unless one is passed.
    design_region : dict or None
        ``{input: (low, high)}`` over which the model was fitted. The only
        thing that lets a later reader tell interpolation from extrapolation;
        :meth:`inside_design_region` is the check.
    note, data_digest, created_at, created_with : str
        Provenance.
    """

    model: SymbolicModel
    estimates: dict[str, float]
    covariance: np.ndarray
    sigma: float
    sigma_source: str = "residual"
    n_observations: int = 0
    degrees_of_freedom: int = 0
    level: float = 0.95
    design_region: dict[str, tuple[float, float]] | None = None
    note: str = ""
    data_digest: str | None = None
    created_at: str = field(default_factory=_now_iso)
    created_with: str = ""

    def __post_init__(self) -> None:
        names = list(self.model.parameter_names)
        missing = [n for n in names if n not in self.estimates]
        if missing:
            raise ValueError(f"estimates are missing parameter(s) {missing}")
        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (len(names), len(names)):
            raise ValueError(
                f"covariance must be {len(names)}x{len(names)} for parameters {names}, "
                f"got shape {cov.shape}"
            )
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "estimates", {n: float(self.estimates[n]) for n in names})
        if float(self.sigma) <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma!r}")
        if not 0.0 < float(self.level) < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level!r}")

    # -- building -----------------------------------------------------------

    @classmethod
    def from_fit(
        cls,
        model: SymbolicModel,
        fit: Mapping[str, Any],
        *,
        design_region: Mapping[str, tuple[float, float]] | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
        note: str = "",
    ) -> "ModelCard":
        """Build a card from a :func:`~discopt.doe.symbolic.fit_least_squares` result.

        ``rows`` (the runs that were fitted) is only digested, never stored.
        """
        required = ("estimates", "covariance", "sigma")
        missing = [k for k in required if k not in fit]
        if missing:
            raise ValueError(
                f"fit result is missing {missing}; pass the dict returned by "
                "fit_least_squares (or build the ModelCard directly)"
            )
        return cls(
            model=model,
            estimates=dict(fit["estimates"]),
            covariance=np.asarray(fit["covariance"], dtype=float),
            sigma=float(fit["sigma"]),
            sigma_source=str(fit.get("sigma_source", "residual")),
            n_observations=int(fit.get("n_observations", 0)),
            degrees_of_freedom=int(fit.get("degrees_of_freedom", 0)),
            level=float(fit.get("level", 0.95)),
            design_region={k: (float(v[0]), float(v[1])) for k, v in (design_region or {}).items()}
            or None,
            note=note,
            data_digest=digest_rows(rows) if rows is not None else None,
            created_with=_version(),
        )

    # -- using the model ----------------------------------------------------

    def predict(self, x: Mapping[str, float]) -> float:
        """The fitted response at design point ``x``."""
        return float(self.model.predict(self.estimates, x))

    def sensitivity(self, x: Mapping[str, float]) -> np.ndarray:
        """``dy/dtheta`` at ``x``, in parameter order."""
        return np.asarray(self.model.jacobian_row(self.estimates, x), dtype=float)

    def standard_error(self, x: Mapping[str, float]) -> float:
        """Standard error of the *mean* response at ``x`` (the delta method).

        ``sqrt(J Cov Jᵀ)``: how well the fitted surface itself is known there.
        For the spread of a single future measurement, see :meth:`interval`
        with ``kind="prediction"``.
        """
        j = self.sensitivity(x)
        return float(np.sqrt(max(j @ self.covariance @ j, 0.0)))

    def interval(
        self,
        x: Mapping[str, float],
        *,
        kind: str = "mean",
        level: float | None = None,
    ) -> tuple[float, float]:
        """Confidence interval for the mean response, or for the next measurement.

        ``kind="mean"`` gives ``yhat ± t·SE``: where the surface is. Repeating
        the experiment many times at ``x`` and averaging lands inside it at the
        stated rate. ``kind="prediction"`` gives ``yhat ± t·sqrt(SE² + sigma²)``:
        where the *next single run* will land, which is the wider question and
        usually the one being asked.
        """
        if kind not in ("mean", "prediction"):
            raise ValueError(f"kind must be 'mean' or 'prediction', got {kind!r}")
        lvl = self.level if level is None else float(level)
        if not 0.0 < lvl < 1.0:
            raise ValueError(f"level must be in (0, 1), got {lvl!r}")
        se = self.standard_error(x)
        if kind == "prediction":
            se = float(np.sqrt(se**2 + self.sigma**2))
        centre = self.predict(x)
        half = self._critical(lvl) * se
        return (centre - half, centre + half)

    def _critical(self, level: float) -> float:
        """t with the fit's degrees of freedom, or normal when there are none.

        A residual sigma from two or three degrees of freedom is itself very
        uncertain, and the t quantile is what carries that into the interval.
        With a known sigma there is nothing extra to pay for, so the normal
        quantile is right.
        """
        tail = 1.0 - (1.0 - level) / 2.0
        if self.sigma_source == "residual" and self.degrees_of_freedom > 0:
            from scipy.stats import t as t_dist

            return float(t_dist.ppf(tail, df=self.degrees_of_freedom))
        from scipy.stats import norm

        return float(norm.ppf(tail))

    def fim(self) -> np.ndarray:
        """The information matrix behind the covariance, ``inv(Cov)``.

        On the ``sigma`` scale of this card. A design search wants information
        on the *declared* scale instead -- see
        :meth:`~discopt.doe.workbook.Workbook.read_fim`.
        """
        try:
            return np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.covariance)

    def standard_errors(self) -> dict[str, float]:
        """Parameter standard errors, keyed by name."""
        sd = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        return {n: float(sd[i]) for i, n in enumerate(self.model.parameter_names)}

    def inside_design_region(self, x: Mapping[str, float]) -> bool:
        """Whether ``x`` is inside the region the model was fitted over.

        ``True`` when no region was recorded: nothing is known against it.
        """
        if not self.design_region:
            return True
        for name, (lo, hi) in self.design_region.items():
            if name in x and not (lo <= float(x[name]) <= hi):
                return False
        return True

    # -- files --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The JSON-able form, ready for :func:`json.dumps`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model.to_dict(),
            "estimates": dict(self.estimates),
            "covariance": self.covariance.tolist(),
            "standard_errors": self.standard_errors(),
            "sigma": float(self.sigma),
            "sigma_source": self.sigma_source,
            "n_observations": int(self.n_observations),
            "degrees_of_freedom": int(self.degrees_of_freedom),
            "level": float(self.level),
            "design_region": (
                {k: list(v) for k, v in self.design_region.items()} if self.design_region else None
            ),
            "note": self.note,
            "data_digest": self.data_digest,
            "created_at": self.created_at,
            "created_with": self.created_with,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelCard":
        """Rebuild a card. ``standard_errors`` is derived, so it is not read back."""
        version = int(payload.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"model card schema version {version} is newer than this version of "
                f"discopt-doe understands ({SCHEMA_VERSION}); upgrade discopt-doe"
            )
        missing = [k for k in ("model", "estimates", "covariance", "sigma") if k not in payload]
        if missing:
            raise ValueError(f"model card is missing {missing}")
        region = payload.get("design_region") or None
        return cls(
            model=SymbolicModel.from_dict(payload["model"]),
            estimates=dict(payload["estimates"]),
            covariance=np.asarray(payload["covariance"], dtype=float),
            sigma=float(payload["sigma"]),
            sigma_source=str(payload.get("sigma_source", "residual")),
            n_observations=int(payload.get("n_observations", 0)),
            degrees_of_freedom=int(payload.get("degrees_of_freedom", 0)),
            level=float(payload.get("level", 0.95)),
            design_region=(
                {k: (float(v[0]), float(v[1])) for k, v in region.items()} if region else None
            ),
            note=str(payload.get("note", "")),
            data_digest=payload.get("data_digest"),
            created_at=str(payload.get("created_at", _now_iso())),
            created_with=str(payload.get("created_with", "")),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """The card as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "ModelCard":
        """Read a card from JSON text."""
        return cls.from_dict(json.loads(text))

    def save(self, path: Path | str) -> Path:
        """Write the card to ``path`` and return it."""
        p = Path(path)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "ModelCard":
        """Read a card written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- display ------------------------------------------------------------

    def summary(self) -> str:
        """A readable rendering: the model, the estimates with SEs, the noise."""
        se = self.standard_errors()
        lines = [f"{self.model.response_name} = {self.model.source}"]
        for name in self.model.parameter_names:
            lines.append(f"  {name:>10s} = {self.estimates[name]:>12.6g}  ± {se[name]:.4g}")
        lines.append(
            f"  sigma = {self.sigma:.6g} ({self.sigma_source}), "
            f"n = {self.n_observations}, dof = {self.degrees_of_freedom}"
        )
        if self.design_region:
            region = ", ".join(
                f"{k} in [{v[0]:g}, {v[1]:g}]" for k, v in self.design_region.items()
            )
            lines.append(f"  fitted over: {region}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)
