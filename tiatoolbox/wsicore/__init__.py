"""Package to read whole slide images."""

from tiatoolbox.wsicore import metadata, wsimeta, wsireader

from .wsimeta import WSIMeta
from .wsireader import WSIReader

# Top level imports
__all__ = [
    "WSIMeta",
    "WSIReader",
]
