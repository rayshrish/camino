"""CAMINO: Computational Aberration Modelling and Inference for NIRCam Observations.

The core optics, exposure and fitting helpers are re-exported here, so
``import camino; camino.calc_throughput(...)`` works directly. The heavier
workflow modules are imported on demand:

- ``camino.fitting``: single-pair WLP8/WLM8 wavefront fitting.
- ``camino.data_utils``: MAST download helpers (needs network access).
- ``camino.abcdlux_patch``: ABCD-matrix / linear canonical transform propagation.
"""

from importlib.metadata import PackageNotFoundError, version

from . import core
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all

try:
    __version__ = version("jwst-camino")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0+unknown"

__all__ = ["core", "__version__", *_core_all]
