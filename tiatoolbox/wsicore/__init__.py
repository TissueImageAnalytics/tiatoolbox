"""Package to read whole slide images."""

from typing import TypedDict

from tiatoolbox.wsicore import metadata, wsimeta, wsireader

from .wsimeta import WSIMeta
from .wsireader import Number, WSIReader

# Top level imports
__all__ = [
    "WSIMeta",
    "WSIReader",
    "WSIReaderParams",
]


class WSIReaderParams(TypedDict, total=False):
    """Parameters for reading whole slide images."""

    meta: WSIMeta | None
    mpp: tuple[Number, Number] | Number
    power: Number
    storage_options: dict  # For FsspecStore
