"""This package contains various tools for working with WSIs."""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__)

graph = _lazy_import("graph", location)
patchextraction = _lazy_import("patchextraction", location)
pyramid = _lazy_import("pyramid", location)
registration = _lazy_import("registration", location)
stainaugment = _lazy_import("stainaugment", location)
stainextract = _lazy_import("stainextract", location)
stainnorm = _lazy_import("stainnorm", location)
tissuemask = _lazy_import("tissuemask", location)
