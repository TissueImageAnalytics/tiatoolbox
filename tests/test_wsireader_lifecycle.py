"""Resource ownership and cleanup for WSI reader contexts."""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.wsicore.wsireader import (
    AnnotationStoreReader,
    DICOMWSIReader,
    FsspecJsonWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    VirtualWSIReader,
    WSIReader,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("reader_type", "attribute"),
    [(OpenSlideWSIReader, "openslide_wsi"), (DICOMWSIReader, "wsi")],
)
def test_native_reader_closes_on_context_failure(
    reader_type: type, attribute: str
) -> None:
    """Native handles are released even when reading within the context fails."""
    reader = reader_type.__new__(reader_type)
    handle = Mock()
    setattr(reader, attribute, handle)
    failure = RuntimeError("read failed")
    with pytest.raises(RuntimeError, match="read failed"), reader:
        raise failure
    handle.close.assert_called_once_with()


@pytest.mark.parametrize("reader_type", [TIFFWSIReader, FsspecJsonWSIReader])
@pytest.mark.parametrize("state", ["absent", "none", "open"])
def test_zarr_reader_closes_available_handles(reader_type: type, state: str) -> None:
    """Cleanup tolerates partial initialization and closes every acquired handle."""
    reader = reader_type.__new__(reader_type)
    handles = {name: Mock() for name in ("_zarr_cache", "_zarr_store", "tiff")}
    for name, handle in handles.items():
        if state != "absent":
            setattr(reader, name, handle if state == "open" else None)
    reader.close()
    for name, handle in handles.items():
        if state == "open" and (name != "tiff" or reader_type is TIFFWSIReader):
            handle.close.assert_called_once_with()
        else:
            handle.close.assert_not_called()


def test_annotation_reader_owns_base_opened_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close owned base images while keeping borrowed annotation stores open."""
    base = VirtualWSIReader(np.zeros((8, 8, 3), np.uint8))
    close = Mock(wraps=base.close)
    monkeypatch.setattr(base, "close", close)
    path = tmp_path / "base.png"
    open_reader = Mock(return_value=base)
    monkeypatch.setattr(WSIReader, "open", open_reader)
    with closing(SQLiteStore()) as store:
        with AnnotationStoreReader(store, base.info, base_wsi=path) as reader:
            assert reader.base_wsi is base
        open_reader.assert_called_once_with(path)
        close.assert_called_once_with()
        assert store.cons
