"""Shared test configuration.

Force JAX onto CPU with 64-bit floats before any test imports discopt (the
FIM autodiff pipeline assumes float64; see the same setup in the base repo's
python/tests/conftest.py).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
