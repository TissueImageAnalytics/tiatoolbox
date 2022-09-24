"""Package to read whole slide images"""
from pathlib import Path

from tiatoolbox import _lazy_import
from tiatoolbox.wsicore import save_tiles, slide_info, wsimeta, wsireader

location = Path(__file__).parent

metadata = _lazy_import("metadata", location)
