"""Silence the base solver's bookkeeping during repeated fits.

Loops that solve many small estimation problems (profile likelihood,
multi-start fitting, sequential design) otherwise flood the output with the
base ``discopt`` solver's per-solve messages, for example the "Duals withheld"
warning emitted when a box-constrained least-squares fit ends near its bounds.
None of it bears on the estimate. :func:`quiet_solver` raises the ``discopt``
logger to ``ERROR`` for the duration of a block and restores it afterwards.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

_SOLVER_LOGGER = "discopt"


@contextmanager
def quiet_solver(enabled: bool = True, *, level: int = logging.ERROR) -> Iterator[None]:
    """Context manager that hides the base solver's log messages.

    Parameters
    ----------
    enabled : bool, default True
        ``False`` makes the block a no-op, so callers can expose a
        ``quiet=`` switch without branching.
    level : int, default ``logging.ERROR``
        Minimum level still shown from the ``discopt`` logger.

    Notes
    -----
    The native (Rust) layer reads ``RUST_LOG`` once, when it first loads. The
    variable is set to ``error`` here only if it is unset, which affects a
    first solve inside the block; set ``RUST_LOG=error`` before importing
    ``discopt`` to silence native messages from the start.

    Examples
    --------
    >>> from discopt.doe import quiet_solver
    >>> with quiet_solver():
    ...     pass  # fits here print no solver bookkeeping
    """
    if not enabled:
        yield
        return
    logger = logging.getLogger(_SOLVER_LOGGER)
    old_level = logger.level
    set_env = "RUST_LOG" not in os.environ
    if set_env:
        os.environ["RUST_LOG"] = "error"
    logger.setLevel(max(level, old_level) if old_level else level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
        if set_env:
            os.environ.pop("RUST_LOG", None)


__all__ = ["quiet_solver"]
