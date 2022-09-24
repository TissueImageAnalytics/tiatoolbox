"""Module initialisation."""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent
dsl = _lazy_import("dsl", location)
storage = _lazy_import("storage", location)
