"""This package contains various tools for working with WSIs."""
from pathlib import Path

from tiatoolbox import _lazy_import
from tiatoolbox.tools import (
    graph,
    patchextraction,
    pyramid,
    stainaugment,
    stainextract,
    stainnorm,
    tissuemask,
)

location = Path(__file__).parent
registration = _lazy_import("registration", location)
