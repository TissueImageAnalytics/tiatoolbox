"""Models package for the models implemented in tiatoolbox."""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

wsi_registration = _lazy_import("wsi_registration", location)
