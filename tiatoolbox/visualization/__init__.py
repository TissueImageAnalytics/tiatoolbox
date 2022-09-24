from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__)

tileserver = _lazy_import("tileserver", location)
