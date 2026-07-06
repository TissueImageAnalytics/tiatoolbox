"""Define TypedDict for WSIReader input params."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:  # pragma: no cover
    from numbers import Number

    from tiatoolbox.wsicore.wsimeta import WSIMeta


class WSIReaderParams(TypedDict, total=False):
    """Parameters for reading whole slide images."""

    meta: WSIMeta | None
    mpp: tuple[Number, Number] | Number | None
    power: Number | None
    storage_options: dict  # For FsspecStore
