"""
This library supports the management of RTDC-training datasets, ML-models and the
pre- and postprocessing of RTDC data.
"""
# flake8: noqa: F401

try:
    from importlib.metadata import version as _pkg_version
except ImportError:  # Python <3.8 fallback
    from importlib_metadata import version as _pkg_version  # type: ignore

try:
    __version__ = _pkg_version("dcml")
except Exception:
    __version__ = "0+unknown"

from . import utils
from . import preprocessing, training
