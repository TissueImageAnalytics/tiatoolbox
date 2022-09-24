"""Package to read whole slide images"""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__)

metadata = _lazy_import("metadata", location)
save_tiles = _lazy_import("save_tiles", location)
slide_info = _lazy_import("slide_info", location)
wsimeta = _lazy_import("wsimeta", location)
wsireader = _lazy_import("wsireader", location)
