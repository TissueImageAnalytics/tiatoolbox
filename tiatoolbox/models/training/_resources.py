"""Process-local resource handling for training datasets and samplers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tiatoolbox.annotation import AnnotationStore, DictionaryStore, SQLiteStore
from tiatoolbox.wsicore.wsireader import (
    DICOMWSIReader,
    FsspecJsonWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    WSIReader,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable

    from tiatoolbox.wsicore.wsimeta import WSIMeta

DEFAULT_CACHE_SIZE = 16
# File-backed readers that can be reopened from their path in a worker.
_REOPENABLE_READERS = (
    OpenSlideWSIReader,
    JP2WSIReader,
    TIFFWSIReader,
    DICOMWSIReader,
    NGFFWSIReader,
    FsspecJsonWSIReader,
)


@dataclass
class _ReaderSpec:
    """Reopen a standard file reader in a worker with its original metadata."""

    reader_type: type[WSIReader]
    path: str | Path
    info: WSIMeta
    kwargs: dict

    def open(self) -> WSIReader:
        """Open the configured reader and restore its metadata."""
        reader = self.reader_type(self.path, **self.kwargs)
        reader.info = self.info
        return reader


def _worker_spec(spec: object, memo: dict[int, object]) -> object:
    """Serialize each shared runtime resource only once per dataset snapshot."""
    key = id(spec)
    if key not in memo:
        memo[key] = _serialize_resource(spec)
    return memo[key]


def _serialize_resource(spec: object) -> object:
    """Serialize borrowed resources without discarding their meaning or contents."""
    if isinstance(spec, SQLiteStore):
        if str(spec.connection) == ":memory:":
            snapshot = DictionaryStore()
            snapshot.append_many(list(spec.values()), keys=list(spec.keys()))
            return snapshot
        if spec.con.in_transaction:
            msg = (
                "Commit file-backed annotation stores before starting dataset workers."
            )
            raise ValueError(msg)
        return Path(spec.path)
    # Exact types: subclasses may need constructor arguments a path cannot give.
    if type(spec) in _REOPENABLE_READERS:
        kwargs = {"series": spec.series_n} if hasattr(spec, "series_n") else {}
        kwargs["post_proc"] = spec.post_proc
        return _ReaderSpec(type(spec), spec.input_path, spec.info, kwargs)
    return spec


def _cache_resource[Resource: (AnnotationStore, WSIReader)](
    cache: OrderedDict[str | int, Resource],
    key: str | int,
    factory: Callable[[], Resource],
    capacity: int = DEFAULT_CACHE_SIZE,
) -> Resource:
    """Maintain a small LRU of owned handles in each process."""
    if key not in cache:
        if len(cache) >= capacity:
            _, resource = cache.popitem(last=False)
            resource.close()
        cache[key] = factory()
    cache.move_to_end(key)
    return cache[key]
