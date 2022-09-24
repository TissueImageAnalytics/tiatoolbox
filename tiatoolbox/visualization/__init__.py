from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

tileserver = _lazy_import("tileserver", location)
