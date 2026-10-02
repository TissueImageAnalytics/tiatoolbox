"""Define TypedDict for WSIReader input params."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from numbers import Number

    from tiatoolbox.wsicore.wsimeta import WSIMeta


class WSIReaderParams(TypedDict, total=False):
    """Parameters for reading whole slide images."""

    meta: WSIMeta | None
    mpp: tuple[Number, Number] | Number | None
    power: Number | None
    storage_options: dict  # For FsspecStore
    post_proc: str | Callable[..., Any] | None


class WSIReaderExtraParams(TypedDict, total=False):
    """Extra kwargs when mpp, power, and post_proc are explicit parameters."""

    meta: WSIMeta | None
    storage_options: dict  # For FsspecStore
