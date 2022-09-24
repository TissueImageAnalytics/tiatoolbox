from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

ngff = _lazy_import("ngff", location)
