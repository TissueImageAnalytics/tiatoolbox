"""Storage of annotations.

This module contains a collection of classes for handling storage of
annotations in memory in addition to serialization/deserialization to/from
disk.

Definitions
-----------

For the sake of clarity it is helpful to define a few terms used throughout
this documentation.

Annotation
    A geometry and associated properties.
Geometry
    One of: a point, a polygon, or a line string.

    .. figure:: ../images/geometries.png
            :width: 512

Properties
    Key-value pairs associated with a geometry.

"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import pickle
import sqlite3
import struct
import sys
import tempfile
import threading
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (
    Generator,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MutableMapping,
    ValuesView,
)
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import shapely
from shapely import wkb as shapely_wkb
from shapely import wkt as shapely_wkt
from shapely.affinity import scale, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import mapping as geometry2feature
from shapely.geometry import shape as feature2geometry

import tiatoolbox
from tiatoolbox import DuplicateFilter, logger
from tiatoolbox.annotation.dsl import (
    PY_GLOBALS,
    SQL_GLOBALS,
    json_contains,
    json_list_sum,
    py_regexp,
)
from tiatoolbox.enums import GeometryType
from tiatoolbox.typing import CallablePredicate, CallableSelect, Geometry

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.typing import (
        Predicate,
        Properties,
        QueryGeometry,
        Select,
    )

sqlite3.enable_callback_tracebacks(True)  # noqa: FBT003

WKB_POINT_STRUCT = struct.Struct("<BIdd")

# Only Python 3.10+ supports using slots for dataclasses
# https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass
# therefore we use the following workaround to only use them when available.
# Using slots gives a performance boost at object creation time.
USE_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}  # pragma: no cover


@dataclass(frozen=True, init=False, eq=True, **USE_SLOTS)
class Annotation:
    """An annotation: a geometry and associated properties.

    Attributes:
        geometry (Geometry):
            The geometry of the annotation as a Shapely object.
        properties (dict):
            The properties of the annotation.
        wkb (bytes):
            The WKB representation of the geometry.

    """

    _geometry: Geometry = field()
    properties: dict[str, Properties] = field(default_factory=dict, hash=True)
    _wkb: bytes = field(default_factory=bytes, hash=False)

    @property
    def geometry(self: Annotation) -> Geometry:
        """Return the shapely geometry of the annotation."""
        if self._geometry is None:
            # Lazy creation of Shapely object when first requested. This
            # is memoized under _geometry. object.__setattr__ must be
            # used because the class is frozen and will disallow normal
            # assignment.
            object.__setattr__(self, "_geometry", shapely_wkb.loads(self._wkb))
        # Return memoized geometry
        return self._geometry

    @property
    def wkb(self: Annotation) -> bytes:
        """Return the WKB representation of the annotation."""
        if self._wkb is None:
            object.__setattr__(self, "_wkb", self.geometry.wkb)
        return self._wkb

    @property
    def geometry_type(self: Annotation) -> GeometryType:
        """Return the geometry type of the annotation."""
        if self._geometry:
            return GeometryType(self.geometry.type)
        return GeometryType(
            int.from_bytes(
                self._wkb[1:4],
                byteorder="big" if self._wkb[0] == "b\0" else "little",
            ),
        )

    @property
    def coords(  # noqa: PLR0911 - 7 > 6 returns
        self: Annotation,
    ) -> np.ndarray | list[np.ndarray] | list[list[np.ndarray]]:
        """The annotation geometry as a flat array of 2D coordinates.

        Returns a numpy array of coordinates for point and line string.
        For polygons, returns a list of numpy arrays, one for each ring.
        For multi-geometries, returns a list with one element for each
        geometry.

        Returns:
            np.array or list:
                The coordinates of the annotation geometry.

        """
        if self._geometry is None:
            return Annotation.decode_wkb(self._wkb, self.geometry_type.value)
        geom_type = self.geometry_type
        if geom_type == GeometryType.POINT:
            return np.array(self.geometry.coords)
        if geom_type == GeometryType.LINE_STRING:
            return np.array(self.geometry.coords)
        if geom_type == GeometryType.POLYGON:
            return [np.array(ring.coords) for ring in shapely.get_rings(self.geometry)]
        if geom_type == GeometryType.MULTI_POINT:
            return [np.array(part.coords) for part in self.geometry.geoms]
        if geom_type == GeometryType.MULTI_LINE_STRING:
            return [np.array(part.coords) for part in self.geometry.geoms]
        if geom_type == GeometryType.MULTI_POLYGON:
            return [
                [np.array(ring.coords) for ring in shapely.get_rings(poly)]
                for poly in self.geometry.geoms
            ]
        msg = f"Unknown geometry type: {self.geometry_type}"
        raise ValueError(msg)

    def to_feature(self: Annotation) -> dict:
        """Return a feature representation of this annotation.

        A feature representation is a Python dictionary with the
        same schema as a geoJSON feature.

        Returns:
            dict:
                A feature representation of this annotation.
        """
        return {
            "type": "Feature",
            "geometry": geometry2feature(self.geometry),
            "properties": self.properties,
        }

    def to_geojson(self: Annotation) -> str:
        """Return a GeoJSON string representation of this annotation.

        Returns:
            str:
                A GeoJSON representation of this annotation.

        """
        return json.dumps(self.to_feature())

    def to_wkb(self: Annotation) -> bytes:
        """Returns the geometry as Well-Known Binary (WKB).

        Returns:
            Annotation:
                The annotation as a WKB geometry.

        """
        return copy.copy(self.wkb)

    def to_wkt(self: Annotation) -> str:
        """Returns the geometry as Well-Know Text (WKT).

        Returns:
            Annotation:
                The annotation as a WKT geometry.

        """
        return self.geometry.wkt

    def __init__(
        self: Annotation,
        geometry: Geometry | int | str | None = None,
        properties: Properties | None = None,
        wkb: bytes | None = None,
    ) -> None:
        """Create a new annotation.

        Must be initialized with either a Shapely geometry object or WKB
        bytes.

        Args:
            geometry (Geometry):
                The geometry of the annotation.
            properties (dict):
                The properties of the annotation. Optional, defaults to
                {}.
            wkb (bytes):
                The WKB representation of a geometry. Optional.

        """
        if wkb is not None and geometry is not None:
            msg = "Cannot init with both geometry and wkb."
            raise ValueError(msg)
        if wkb is None and geometry is None:
            msg = "Either geometry or wkb must be given."
            raise ValueError(msg)
        if wkb is not None:
            object.__setattr__(self, "_wkb", wkb)
            object.__setattr__(self, "_geometry", None)
        if geometry is not None:
            object.__setattr__(self, "_wkb", None)
            object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(self, "properties", properties or {})

    def __repr__(self: Annotation) -> str:
        """Return a string representation of the object."""
        return f"Annotation({self.geometry}, {self.properties})"

    def __hash__(self: Annotation) -> int:
        """Compute the hash value of the object.

        Returns:
            int:
                The hash value.

        """
        return hash((self.geometry, json.dumps(self.properties, sort_keys=True)))

    def __eq__(self: Annotation, other: object) -> bool:
        """Compare this annotation to another.

        Args:
            other (object):
                The object to compare to.

        Returns:
            bool:
                True if the objects are equal, False otherwise.

        """
        if not isinstance(other, Annotation):
            return False
        return self.geometry == other.geometry and self.properties == other.properties

    @staticmethod
    def decode_wkb(
        wkb: bytes,
        geom_type: int,
    ) -> np.ndarray | list[np.ndarray] | list[list[np.ndarray]]:
        r"""Decode WKB to a NumPy array of flat (alternating x, y) coordinates.

        Polygons return a list of NumPy arrays (one array per ring) and
        multi-part geometries return a list with one element per child
        geometry e.g. a list of arrays for Multi-Line and a list of
        lists of arrays for multi-polyon.

        Args:
            wkb (bytes):
                The WKB representation of a geometry.
            geom_type (int):
                The type of geometry to decode. Where 1 = point, 2 =
                line, 3 = polygon, 4 = multi-point, 5 = multi-line, 6 =
                multi-polygon.

        Examples:
            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> # Point(1, 2).wkb
            >>> wkb = (
            ...     b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ...     b"\xf0?\x00\x00\x00\x00\x00\x00\x00@"
            ... )
            >>> AnnotationStore.decode_wkb(wkb, 1)
            array([0., 0.])

            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> # Polygon([[0, 0], [1, 1], [1, 0]]).wkb
            >>> wkb = (
            ...     b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00"
            ...     b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ...     b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00"
            ...     b"\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00"
            ...     b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ...     b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ... )
            >>> AnnotationStore.decode_wkb(wkb, 3)
            array([[0., 0.],
                [1., 1.],
                [1., 0.],
                [0., 0.]])

        Raises:
            ValueError:
                If the geometry type is not supported.

        Returns:
            np.ndarray or list:
                An array of coordinates, a list of coordinates, or a list
                of lists of coordinates.

        """

        def decode_polygon(offset: int = 0) -> tuple[list[np.ndarray], int]:
            """Decode a polygon from WKB.

            Args:
                offset (int, optional):
                    The starting offset in the WKB representation.
                    Defaults to 0.

            Returns:
                Tuple[np.ndarray, int]: A tuple containing the decoded
                polygon rings as numpy arrays and the new offset in the
                WKB representation.

            """
            offset += 5  # byte order and geom type at start of each polygon
            n_rings = np.frombuffer(wkb, np.int32, 1, offset)[0]
            offset += 4

            rings = []
            for _ in range(n_rings):
                n_points = np.frombuffer(wkb, np.int32, 1, offset)[0]
                offset += 4
                rings.append(
                    np.frombuffer(wkb, np.double, n_points * 2, offset).reshape(-1, 2),
                )
                offset += n_points * 16
            return rings, offset

        # Using magic numbers instead of GeometryType enums for future
        # numba compilation (numba doesn't like enums).
        if geom_type == 1:
            # Point
            return np.frombuffer(wkb, np.double, -1, 5).reshape(1, 2)
        if geom_type == 2:  # noqa: PLR2004 - Intentional magic number
            # Line
            return np.frombuffer(wkb, np.double, -1, 9).reshape(-1, 2)
        if geom_type == 3:  # noqa: PLR2004 - Intentional magic number
            # Polygon
            return decode_polygon()[0]
        if geom_type == 4:  # noqa: PLR2004 - Intentional magic number
            # Multi-point
            n_points = np.frombuffer(wkb, np.int32, 1, 5)[0]
            return [
                np.frombuffer(wkb, np.double, 2, 14 + i * 21).reshape(1, 2)
                # each point is 21 bytes
                for i in range(n_points)
            ]
        if geom_type == 5:  # noqa: PLR2004 - Intentional magic number
            # Multi-line
            n_lines = np.frombuffer(wkb, np.int32, 1, 5)[0]
            lines = []
            offset = 9
            for _ in range(n_lines):
                offset += 5
                n_points = np.frombuffer(wkb, np.int32, n_lines, offset)[0]
                offset += 4
                lines.append(
                    np.frombuffer(wkb, np.double, n_points * 2, offset).reshape(-1, 2),
                )
                offset += n_points * 16
            return lines

        if geom_type == 6:  # noqa: PLR2004 - Intentional magic number
            # Multi-polygon
            n_polygons = np.frombuffer(wkb, np.int32, 1, 5)[0]
            polygons = []
            offset = 9
            for _ in range(n_polygons):
                rings, offset = decode_polygon(offset)
                polygons.append(rings)
            return polygons

        msg = f"Unknown geometry type: {geom_type}"
        raise ValueError(msg)


StoreInstanceType = TypeVar("StoreInstanceType", bound="AnnotationStore")


class AnnotationStore(ABC, MutableMapping[str, Annotation]):
    """Annotation store abstract base class."""

    def __new__(
        cls: type[StoreInstanceType],
        *args: str,  # noqa: ARG003
        **kwargs: int,  # noqa: ARG003
    ) -> StoreInstanceType:
        """Return an instance of a subclass of AnnotationStore."""
        if cls is AnnotationStore:
            msg = (
                "AnnotationStore is an abstract class and cannot be instantiated."
                "Use a subclass such as DictionaryStore or SQLiteStore instead."
            )
            raise TypeError(
                msg,
            )
        return super().__new__(cls)

    @staticmethod
    def _is_right_angle(a: list[float], b: list[float], c: list[float]) -> bool:
        """Return True if three points make a right angle.

        Used for optimising queries.

        This function will have positional only arguments when support
        for Python 3.7 is dropped.

        Args:
            a (Sequence[Number]):
                First coordinate.
            b (Sequence[Number]):
                Second coordinate.
            c (Sequence[Number]):
                Third coordinate.


        """
        return np.dot(np.subtract(a, b), np.subtract(b, c)) == 0

    @staticmethod
    def _is_rectangle(
        a: list[float],
        b: list[float],
        c: list[float],
        d: list[float],
        *args: str,
    ) -> bool:
        """Determine if a set of coordinates form a rectangle.

        Used for optimising queries. If more than five points are given,
        or if the optional fifth point is not equal to `a` then this
        returns False.

        Args:
            a (Sequence[Number]):
                First coordinate.
            b (Sequence[Number]):
                Second coordinate.
            c (Sequence[Number])::
                Third coordinate.
            d (Sequence[Number]):
                Fourth coordinate.
            args (list):
                Non-Keyword arguments.

        Returns:
            True if the coordinates form a rectangle, False otherwise.

        """
        # Only allow one extra coordinate for looping back to the first point
        if (len(args) == 1 and not np.array_equal(args[:1], [a])) or len(args) > 1:
            return False
        # Check that all angles are right angles
        return all(
            AnnotationStore._is_right_angle(*xyz)
            for xyz in ((a, b, c), (b, c, d), (c, d, a))
        )

    @staticmethod
    def _connection_to_path(connection: str | Path | IO) -> Path:
        """Normalise a connection object to a Path.

        Here we refer to a 'connection' as anything which references a
        file e.g. a string, a pathlibPath, or a file-like object (IO).

        Args:
            connection (Union[str, Path, IO]):
                The connection object to normalise.

        Returns:
            Path:
                The normalised path.

        """
        if not isinstance(
            connection,
            (
                str,
                Path,
                io.IOBase,
                io.TextIOBase,
                tempfile._TemporaryFileWrapper,  # skipcq: PYL-W0212  # noqa: SLF001
            ),
        ):
            msg = (
                f"Connection must be a string, Path, or an IO object, "
                f"not {type(connection)}"
            )
            raise TypeError(
                msg,
            )
        if isinstance(
            connection,
            (
                io.IOBase,
                io.TextIOBase,
                tempfile._TemporaryFileWrapper,  # skipcq: PYL-W0212  # noqa: SLF001
            ),
        ):
            connection = connection.name
        return Path(connection)

    @staticmethod
    def _validate_equal_lengths(*args: list | None) -> None:
        """Validate that all given args are either None or have the same length."""
        lengths = [len(v) for v in args if v is not None]
        if lengths and any(length != lengths[0] for length in lengths):
            msg = "All arguments must be None or of equal length."
            raise ValueError(msg)

    @staticmethod
    def _geometry_predicate(name: str, a: Geometry, b: Geometry) -> bool:
        """Apply a binary geometry predicate.

        For more information on geometric predicates see the `Shapely
        documentation <https://shapely.readthedocs.io/en/stable/
        manual.html#binary-predicates>`_.

        Args:
            name(str):
                Name of the predicate to apply.
            a(Geometry):
                The first geometry.
            b(Geometry):
                The second geometry.

        Returns:
            bool:
                True if the geometry predicate holds.

        """
        return getattr(a, name)(b)

    # All valid shapely binary predicates
    _geometry_predicate_names: ClassVar[list[str]] = [
        "equals",
        "contains",
        "covers",
        "covered_by",
        "crosses",
        "disjoint",
        "intersects",
        "overlaps",
        "touches",
        "within",
        # Special non-shapely case, bounding-boxes intersect.
        "bbox_intersects",
        # Special non-shapely case, query centroid within k of
        # annotation bounds center.
        "centers_within_k",
    ]

    @classmethod
    @abstractmethod
    def open(cls: type[AnnotationStore], fp: Path | str | IO) -> ABC:
        """Load a store object from a path or file-like object.

        Args:
            fp(Path or str or IO): The file path or file handle.

        Returns:
            AnnotationStoreABC:
                An instance of an annotation store.

        """

    @staticmethod
    def serialise_geometry(geometry: Geometry) -> str | bytes:
        """Serialise a geometry to a string or bytes.

        This defaults to well-known text (WKT) but may be overridden to
        any other format which a Shapely geometry could be serialised to
        e.g. well-known binary (WKB) or geoJSON etc.

        Args:
            geometry(Geometry):
                The Shapely geometry to be serialised.

        Returns:
            bytes or str: The serialised geometry.

        """
        return geometry.wkt

    @staticmethod
    @lru_cache(32)
    def deserialize_geometry(data: str | bytes) -> Geometry:
        """Deserialize a geometry from a string or bytes.

        This default implementation will deserialize bytes as well-known
        binary (WKB) and strings as well-known text (WKT). This can be
        overridden to deserialize other formats such as geoJSON etc.

        Args:
            data(bytes or str):
                The serialised representation of a Shapely geometry.

        Returns:
            Geometry: The deserialized Shapely geometry.

        """
        return (
            shapely_wkt.loads(data)
            if isinstance(data, str)
            else shapely_wkb.loads(data)
        )

    @abstractmethod
    def commit(self: AnnotationStore) -> None:
        """Commit any in-memory changes to disk."""

    @abstractmethod
    def dump(self: AnnotationStore, fp: Path | str | IO) -> None:
        """Serialise a copy of the whole store to a file-like object.

        Args:
            fp(Path or str or IO):
                A file path or file handle object for output to disk.

        """

    @abstractmethod
    def dumps(self: AnnotationStore) -> str | bytes:
        """Serialise and return a copy of store as a string or bytes.

        Returns:
            str or bytes:
                The serialised store.

        """

    def append(
        self: AnnotationStore,
        annotation: Annotation,
        key: str | None = None,
    ) -> str:
        """Insert a new annotation, returning the key.

        Args:
            annotation (Annotation):
                The shapely annotation to insert.
            key (str):
                Optional. The unique key used to identify the annotation in the
                store. If not given a new UUID4 will be generated and returned
                instead.

        Returns:
            str:
                The unique key of the newly inserted annotation.

        """
        if not isinstance(annotation.geometry, (Polygon, Point, LineString)):
            msg = "Invalid geometry type. Must be one of Point, LineString, Polygon."
            raise TypeError(msg)
        keys = key if key is None else [key]
        return self.append_many([annotation], keys)[0]

    def append_many(
        self: AnnotationStore,
        annotations: Iterable[Annotation],
        keys: Iterable[str] | None = None,
    ) -> list[str]:
        """Bulk append of annotations.

        This may be more performant than repeated calls to `append`.

        Args:
            annotations (iter(Annotation)):
                An iterable of annotations.
            keys (iter(str)):
                An iterable of unique keys associated with each geometry being
                inserted. If None, a new UUID4 is generated for each geometry.

        Returns:
            list(str):
                A list of unique keys for the inserted geometries.

        """
        annotations = list(annotations)
        keys = list(keys) if keys else None
        self._validate_equal_lengths(keys, annotations)
        result: list[str] = []
        if keys:
            result.extend(
                self.append(annotation, key)
                for key, annotation in zip(keys, annotations)
            )
            return result
        result.extend(self.append(annotation) for annotation in annotations)
        return result

    def patch(
        self: AnnotationStore,
        key: str,
        geometry: Geometry | None = None,
        properties: dict[str, Properties] | None = None,
    ) -> None:
        """Patch an annotation at given key.

        Partial update of an annotation. Providing only a geometry will update
        the geometry and leave properties unchanged. Providing a properties
        dictionary applies a patch operation to the properties. Only updating
        the properties which are given and leaving the rest unchanged. To
        completely replace an annotation use `__setitem__`.

        Args:
            key(str):
                The key of the annotation to update.
            geometry(Geometry):
                The new geometry. If None, the geometry is not updated.
            properties(dict):
                A dictionary of properties to patch and their new values.
                If None, the existing properties are not altered.

        """
        if key not in self:
            self.append(Annotation(geometry, properties), key)
            return
        geometry = geometry if geometry is None else [geometry]
        properties_list = properties if properties is None else [properties]
        self.patch_many([key], geometry, properties_list)

    def patch_many(
        self: AnnotationStore,
        keys: Iterable[str],
        geometries: Iterable[Geometry] | None = None,
        properties_iter: Iterable[dict[str, Properties]] | None = None,
    ) -> None:
        """Bulk patch of annotations.

        This may be more efficient than calling `patch` repeatedly
        in a loop.

        Args:
            geometries (iter(Geometry)):
                An iterable of geometries to update.
            properties_iter (iter(dict)):
                An iterable of properties to update.
            keys (iter(str)):
                An iterable of keys for each annotation to be updated.

        """
        # Validate inputs
        if not any([geometries, properties_iter]):
            msg = "At least one of geometries or properties_iter must be given"
            raise ValueError(
                msg,
            )
        keys = list(keys)
        geometries = list(geometries) if geometries else None
        properties_iter = list(properties_iter) if properties_iter else None
        self._validate_equal_lengths(keys, geometries, properties_iter)
        properties_iter = properties_iter or ({} for _ in keys)  # pragma: no branch
        geometries = geometries or (None for _ in keys)  # pragma: no branch
        # Update the store
        for key, geometry, properties in zip(keys, geometries, properties_iter):
            properties_ = cast(dict[str, Any], copy.deepcopy(properties))
            self.patch(key, geometry, properties_)

    def remove(self: AnnotationStore, key: str) -> None:
        """Remove annotation from the store with its unique key.

        Args:
            key (str):
                The key of the annotation to be removed.

        """
        self.remove_many([key])

    def remove_many(self: AnnotationStore, keys: Iterable[str]) -> None:
        """Bulk removal of annotations by keys.

        Args:
            keys (iter(str)):
                An iterable of keys for the annotation to be removed.

        """
        for key in keys:
            self.remove(key)

    def setdefault(
        self: AnnotationStore,
        key: str,
        default: Annotation | None = None,
    ) -> Annotation:
        """Return the value of the annotation with the given key.

        If the key does not exist, insert the default value and return
        it.

        Args:
            key (str):
                The key of the annotation to be fetched.
            default (Annotation):
                The value to return if the key is not found.

        Returns:
            Annotation:
                The annotation or default if the key is not found.

        """
        if not isinstance(default, Annotation):
            msg = "default value must be an Annotation instance."
            raise TypeError(msg)
        return super().setdefault(key, default)

    def __delitem__(self: AnnotationStore, key: str) -> None:
        """Delete an annotation by key.

        An alias of `remove`.

        Args:
            key (str):
                The key of the annotation to be removed.

        """
        self.remove(key)

    def keys(self: AnnotationStore) -> KeysView[str]:
        """Return an iterable (usually generator) of all keys in the store.

        Returns:
            Iterable[str]:
                An iterable of keys.

        """
        keys_dict: dict[str, None] = {}
        for key, _ in self.items():  # noqa: PERF102
            keys_dict[key] = None
        return keys_dict.keys()

    def values(self: AnnotationStore) -> ValuesView[Annotation]:
        """Return an iterable of all annotation in the store.

        Returns:
            Iterable[Annotation]:
                An iterable of annotations.

        """
        values_dict: dict[int, Annotation] = {}

        for i, (_, annotation) in enumerate(self.items()):
            values_dict[i] = annotation
        return values_dict.values()

    def __iter__(self: AnnotationStore) -> Iterator[str]:
        """Return an iterable of keys in the store.

        An alias of `keys`.

        Returns:
            Iterator[str]:
                An iterable of keys.

        """
        yield from self.keys()

    @staticmethod
    def _eval_where(
        predicate: Predicate | None,
        properties: Properties,
    ) -> bool:
        """Evaluate properties predicate against properties.

        Args:
            predicate (str or bytes or Callable):
                The predicate to evaluate on properties. The predicate may be a
                string, pickled bytes, or a Callable (e.g. a function).
            properties (dict):
                A dictionary of JSON serializable
                properties on which to evaluate the predicate.

        Returns:
            bool:
                Return True if the predicate holds.

        """
        if predicate is None:
            return True
        if isinstance(predicate, str):
            return bool(
                eval(  # skipcq: PYL-W0123,  # noqa: S307
                    predicate,
                    PY_GLOBALS,
                    {"props": properties},
                ),
            )
        if isinstance(predicate, bytes):
            predicate = pickle.loads(predicate)  # skipcq: BAN-B301  # noqa: S301

        # predicate is Callable
        predicate = cast(Callable, predicate)
        return bool(predicate(properties))

    @staticmethod
    def _centers_within_k(
        annotation_geometry: Geometry,
        query_point: Point,
        distance: float,
    ) -> bool:
        """Checks if centre of annotation is within k of query geometry center.

        Here the "center" is the centroid of the bounds.

        Args:
            annotation_geometry (Geometry): Annotation geometry
            query_point (Point): Query point
            distance (float): distance

        Returns:
            bool:
                True if centre of annotation within k of query geometry center

        """
        ann_centre = Polygon.from_bounds(*annotation_geometry.bounds).centroid
        return query_point.dwithin(ann_centre, distance)

    @staticmethod
    def _bbox_intersects(
        annotation_geometry: Geometry,
        query_geometry: Geometry,
    ) -> bool:
        """Checks if bounding box of the annotation intersects the query geometry.

        Args:
            annotation_geometry (Geometry): Annotation geometry
            query_geometry (Geometry): Query geometry

        Returns:
            bool:
                True if bounding box of the annotation intersects the query geometry
        """
        return Polygon.from_bounds(*query_geometry.bounds).intersects(
            Polygon.from_bounds(*annotation_geometry.bounds),
        )

    def _validate_query_inputs(
        self: AnnotationStore,
        geometry: QueryGeometry | None,
        where: Predicate | None,
        geometry_predicate: str,
    ) -> None:
        """Validates query input.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon).
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore `and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.
            geometry_predicate (str):
                A string defining which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.

        Returns:
            None:
                Raises ValueError if query input is not valid

        """
        if all(x is None for x in (geometry, where)):
            msg = "At least one of geometry or where must be set."
            raise ValueError(msg)
        if geometry_predicate not in self._geometry_predicate_names:
            allowed_values = ", ".join(self._geometry_predicate_names)
            msg = f"Invalid geometry predicate. Allowed values are: {allowed_values}."
            raise ValueError(msg)

    @staticmethod
    def _process_geometry(
        geometry: QueryGeometry | None, geometry_predicate: str
    ) -> QueryGeometry | None:
        """Processes input query geometry.

        Processes input query geometry into appropriate
        geometry type according to geometry_predicate.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon).
            geometry_predicate (str):
                A string defining which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.

        Returns:
            QueryGeometry | None:
                Returns the processed geometry, None if input geometry is None

        """
        query_geometry = geometry
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)

        if geometry_predicate == "centers_within_k" and isinstance(
            query_geometry, (Polygon, Point, LineString)
        ):
            query_geometry = Polygon.from_bounds(*query_geometry.bounds).centroid

        return query_geometry

    def query(
        self: AnnotationStore,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        min_area: float | None = None,
        distance: float = 0,
    ) -> dict[str, Annotation]:
        """Query the store for annotations.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon).
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore `and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.
            geometry_predicate (str):
                A string defining which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.
            min_area (float):
                Minimum area of the annotation geometry. Only
                annotations with an area greater than or equal to this
                value will be returned. Defaults to None (no min).
            distance (float):
                Distance used when performing a distance based query.
                E.g. "centers_within_k" geometry predicate.

        Returns:
                list:
                    A list of Annotation objects.

        """
        self._validate_query_inputs(geometry, where, geometry_predicate)

        query_geometry = self._process_geometry(geometry, geometry_predicate)

        def filter_function(annotation: Annotation) -> bool:
            """Filter function for querying annotations.

            Args:
                annotation (Annotation):
                    The annotation to filter.

            Returns:
                bool:
                    True if the annotation should be included in the
                    query result.

            """
            if min_area is not None and annotation.geometry.area < min_area:
                return False
            return (  # Geometry is None or the geometry predicate matches
                query_geometry is None
                or any(
                    [
                        (
                            geometry_predicate == "bbox_intersects"
                            and self._bbox_intersects(
                                annotation.geometry, query_geometry
                            )
                        ),
                        (
                            geometry_predicate == "centers_within_k"
                            and self._centers_within_k(
                                annotation.geometry,
                                query_geometry,
                                distance,
                            )
                        ),
                        (
                            geometry_predicate
                            not in ("bbox_intersects", "centers_within_k")
                            and self._geometry_predicate(
                                geometry_predicate,
                                query_geometry,
                                annotation.geometry,
                            )
                        ),
                    ],
                )
            ) and self._eval_where(where, annotation.properties)

        return {
            key: annotation
            for key, annotation in self.items()
            if filter_function(annotation)
        }

    def iquery(
        self: AnnotationStore,
        geometry: QueryGeometry,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
    ) -> list[str]:
        """Query the store for annotation keys.

        Acts the same as `AnnotationStore.query` except returns keys
        instead of annotations.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon).
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore `and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.
            geometry_predicate:
                A string which define which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.

        Returns:
                list:
                    A list of keys for each Annotation.

        """
        if geometry_predicate not in self._geometry_predicate_names:
            msg = (
                "Invalid geometry predicate. Allowed values are: "
                f"{', '.join(self._geometry_predicate_names)}."
            )
            raise ValueError(
                msg,
            )
        query_geometry = geometry
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            key
            for key, annotation in self.items()
            if (
                self._geometry_predicate(
                    geometry_predicate,
                    query_geometry,
                    annotation.geometry,
                )
                and self._eval_where(where, annotation.properties)
            )
        ]

    def bquery(
        self: AnnotationStore,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Query the store for annotation bounding boxes.

        Acts similarly to `AnnotationStore.query` except it checks for
        intersection between stored and query geometry bounding boxes.
        This may be faster than a regular query in some cases, e.g. for
        SQliteStore with a large number of annotations.

        Note that this method only checks for bounding box intersection
        and therefore may give a different result to using
        `AnnotationStore.query` with a box polygon and the "intersects"
        geometry predicate. Also note that geometry predicates are not
        supported for this method.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon). If a geometry is provided, the bounds of the
                geometry will be used for the query. Full geometry
                intersection is not used for the query method.
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore` and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.

        Returns:
                list:
                    A list of bounding boxes for each Annotation.

        Example:
                >>> from tiatoolbox.annotation.storage import DictionaryStore
                >>> from shapely.geometry import Polygon
                >>> store = DictionaryStore()
                >>> store.append(
                ...     Annotation(
                ...         geometry=Polygon.from_bounds(0, 0, 1, 1),
                ...         properties={"class": 42},
                ...     ),
                ...     key="foo",
                ... )
                >>> store.bquery(where="props['class'] == 42")
                {'foo': (0.0, 0.0, 1.0, 1.0)}

        """
        query_geometry = geometry
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return {
            key: annotation.geometry.bounds
            for key, annotation in self.items()
            if (
                query_geometry is None
                or isinstance(query_geometry, (Polygon, Point, LineString))
                and Polygon.from_bounds(*annotation.geometry.bounds).intersects(
                    Polygon.from_bounds(*query_geometry.bounds),
                )
                and self._eval_where(where, annotation.properties)
            )
        }

    def pquery(
        self: AnnotationStore,
        select: Select,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
        *,
        unique: bool = True,
        squeeze: bool = True,
    ) -> dict[str, Properties] | list[set[Properties]] | set[Properties]:
        """Query the store for annotation properties.

        Acts similarly to `AnnotationStore.query` but returns only the
        value defined by `select`.

        Args:
            select (str or bytes or Callable):
                A statement defining the value to look up from the
                annotation properties. If `select = "*"`, all properties
                are returned for each annotation (`unique` must be
                False).
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon). If a geometry is provided, the bounds of the
                geometry will be used for the query. Full geometry
                intersection is not used for the query method.
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query. It is
                important to note that untrusted user input should never
                be accepted to this argument as arbitrary code can be
                run via pickle or the parsing of the string statement.
            unique (bool):
                If True, only unique values for each selected property
                will be returned as a list of sets. If False, all values
                will be returned as a dictionary mapping keys values.
                Defaults to True.
            squeeze (bool):
                If True, when querying for a single value with
                `unique=True`, the result will be a single set instead
                of a list of sets.

        Examples:
            >>> from tiatoolbox.annotation.storage import DictionaryStore
            >>> from shapely.geometry import Point
            >>> store = DictionaryStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.append(annotation, "foo")
            >>> store.pquery("*", unique=False)
            ... {'foo': {'class': 42}}

            >>> from tiatoolbox.annotation.storage import DictionaryStore
            >>> from shapely.geometry import Point
            >>> store = DictionaryStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.append(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42}
            >>> annotation =  Annotation(Point(1, 1), {"class": 123})
            >>> store.append(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42, 123}

        """
        if where is not None and type(select) is not type(where):
            msg = "select and where must be of the same type"
            raise TypeError(msg)
        if not isinstance(select, (str, bytes)) and not callable(select):
            msg = f"select must be str, bytes, or Callable, not {type(select)}"
            raise TypeError(
                msg,
            )
        # Are we scanning through all annotations?
        is_scan = not any((geometry, where))
        items = self.items() if is_scan else self.query(geometry, where).items()

        def select_values(
            select: Select,
            annotation: Annotation,
        ) -> Properties | object | tuple[object, ...]:
            """Get the value(s) to return from an annotation via a select.

            Args:
                select (str or bytes or Callable):
                    A statement defining the value to look up from the
                    annotation properties. If `select = "*"`, all properties
                    are returned for each annotation (`unique` must be
                    False).
                annotation (Annotation):
                    The annotation to get the value(s) from.

            Raises:
                ValueError:
                    If arguments have incompatible values.

            Returns:
                Union[Properties, object, Tuple[object, ...]]:
                    The value(s) to return from the annotation. This
                    will be a dictionary if unique is False. Otherwise,
                    it will be a list of sets. If squeeze and unique are
                    True in addition to there only being one set in the
                    results list, the result will be a single set.

            """  # Q440, Q441
            if select == "*" and unique:
                msg = "unique=True cannot be used with select='*'"
                raise ValueError(msg)

            if select == "*":  # Special case for all properties
                return annotation.properties

            if isinstance(select, str):
                py_locals = {"props": annotation.properties}
                return eval(  # skipcq: PYL-W0123,  # noqa: S307
                    select,
                    PY_GLOBALS,
                    py_locals,
                )
            if isinstance(select, bytes):
                return pickle.loads(select)(  # skipcq: BAN-B301  # noqa: S301
                    annotation.properties,
                )

            return select(annotation.properties)

        return self._handle_pquery_results(
            select=select,
            items=items,
            get_values=select_values,
            unique=unique,
            squeeze=squeeze,
        )

    def nquery(
        self: AnnotationStore,
        geometry: Geometry | None = None,
        where: Predicate | None = None,
        n_where: Predicate | None = None,
        distance: float = 5.0,
        geometry_predicate: str = "intersects",
        mode: tuple[str, str] | str = "poly-poly",
    ) -> dict[str, dict[str, Annotation]]:
        """Query for annotations within a distance of another annotation.

        Args:
            geometry (Geometry):
                A geometry to use to query for the initial set of
                annotations to perform a neighbourhood search around. If
                None, all annotations in the store are considered.
                Defaults to None.
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will be
                returned. Defaults to None (assume always true). This may
                be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned the
                annotation store backend in python before being returned
                to the user. A pickle object is, where possible, hooked
                into the backend as a user defined function to filter
                results during the backend query. Strings are expected to
                be in a domain specific language and are converted to SQL
                on a best-effort basis. For supported operators of the DSL
                see :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query. It is
                important to note that untrusted user input should never
                be accepted to this argument as arbitrary code can be
                run via pickle or the parsing of the string statement.
            n_where (str or bytes or Callable):
                Predicate to filter the nearest annotations by. Defaults
                to None (assume always true). See `where` for more
                details.
            distance (float):
                The distance to search for annotations within. Defaults to
                5.0.
            geometry_predicate (str):
                The predicate to use when comparing geometries. Defaults
                to "intersects". Other options include "within" and
                "contains". Ignored if `mode` is "boxpoint-boxpoint" or
                "box-box".
            mode (tuple[str, str] or str):
                The method to use for determining distance during the
                query. Defaults to "box-box". This may significantly
                change performance depending on the backend. Possible
                options are:
                  - "poly-poly": Polygon boundary to polygon boundary.
                  - "boxpoint-boxpoint": Bounding box centre point to
                    bounding box centre point.
                  - "box-box": Bounding box to bounding box.
                May be specified as a dash separated string or a tuple
                of two strings. The first string is the mode for the
                query geometry and the second string is the mode for
                the nearest annotation geometry.

        Returns:
            Dict[str, Dict[str, Annotation]]:
                A dictionary mapping annotation keys to another
                dictionary which represents an annotation key and all
                annotations within `distance` of it.

        The `mode` argument is used to determine how to calculate the
        distance between annotations. The default mode is "box-box".

        The "box-box" mode uses the bounding boxes of stored annotations
        and the query geometry when determining if annotations are
        within the neighbourhood.

        .. figure:: ../images/nquery-box-box.png
            :width: 512
            :alt: "box-box" mode

        The "poly-poly" performs full polygon-polygon intersection with
        the polygon boundary of stored annotations and the query
        geometry to determine if annotations are within the
        neighbourhood.

        .. figure:: ../images/nquery-poly-poly.png
            :width: 512
            :alt: "poly-poly" mode


        The "boxpoint-boxpoint" mode uses the centre point of the
        bounding box of stored annotations and the query geometry when
        determining if annotations are within the neighbourhood.

        .. figure:: ../images/nquery-boxpoint-boxpoint.png
            :width: 512
            :alt: "boxpoint-boxpoint" mode


        Examples:
            Example bounding box query with one neighbour within a
            distance of 2.0.

            >>> from shapely.geometry import Point, Polygon
            >>> from tiatoolbox.annotation.storage import Annotation, SQLiteStore
            >>> store = SQLiteStore()
            >>> annotation = Annotation(Point(0, 0), {"class": 42})
            >>> store.append(annotation, "foo")
            >>> neighbour = Annotation(Point(1, 1), {"class": 123})
            >>> store.add(neighbour, "bar")
            >>> store.nquery((-.5, -.5, .5, .5), distance=2.0)
            {
              "foo": {
                Annotation(POINT (0 0), {'class': 42}): {
                  "bar": Annotation(POINT (1 1), {'class': 123}),
                }
              },
            }

            Example bounding box query with no neighbours within a
            distance of 1.0.

            >>> from shapely.geometry import Point
            >>> from tiatoolbox.annotation.storage import Annotation, SQLiteStore
            >>> store = SQLiteStore()
            >>> annotation = Annotation(Point(0, 0), {"class": 42})
            >>> store.add(annotation, "foo")
            >>> store.nquery((-.5, -.5, .5, .5), distance=1.0)
            {"foo": {Annotation(POINT (0 0), {'class': 42}): {}}}

            Example of querying for TILs - lympocytes within 3 units
            of tumour cells.

            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> store = SQLiteStore("hovernet-pannuke-output.db")
            >>> tils = store.nquery(
            ...     where="props['class'] == 1",   # Tumour cells
            ...     n_where="props['class'] == 0",  # Lymphocytes
            ...     distance=32.0,  # n_where within 32 units of where
            ...     mode="point-point",  # Use point to point distance
            ... )

        """
        # This is a naive generic implementation which can be overridden
        # by back ends which can do this more efficiently.
        if not isinstance(mode, (str, tuple)):
            msg = "mode must be a string or tuple of strings"
            raise TypeError(msg)

        mode_tuple = tuple(mode.split("-")) if isinstance(mode, str) else mode
        if mode_tuple not in (
            ("box", "box"),
            ("boxpoint", "boxpoint"),
            ("poly", "poly"),
        ):
            msg = "mode must be one of 'box-box', 'boxpoint-boxpoint', or 'poly-poly'"
            raise ValueError(
                msg,
            )
        from_mode, _ = mode_tuple

        # Initial selection of annotations to query around
        selection = self.query(
            geometry=geometry,
            where=where,
        )

        # Query for others within the distance of initial selection
        result = {}
        for key, ann in selection.items():
            geometry = ann.geometry
            if from_mode == "box":
                geometry_predicate = "bbox_intersects"
                min_x, min_y, max_x, max_y = ann.geometry.bounds
                geometry = Polygon.from_bounds(
                    min_x - distance,
                    min_y - distance,
                    max_x + distance,
                    max_y + distance,
                )
            elif from_mode == "boxpoint":
                geometry_predicate = "centers_within_k"
            elif from_mode == "poly":  # pragma: no branch
                geometry = ann.geometry
                geometry = cast(Geometry, geometry)
                geometry = geometry.buffer(distance)
            subquery_result = self.query(
                geometry=geometry,
                where=n_where,
                geometry_predicate=geometry_predicate,
                distance=distance,
            )
            if subquery_result:
                result[key] = subquery_result

        return result

    @staticmethod
    def _handle_pquery_results(
        select: Select,
        items: ItemsView[str, Annotation],
        get_values: Callable[
            [Select, Annotation],
            Properties | object | tuple[object, ...],
        ],
        *,
        unique: bool,
        squeeze: bool,
    ) -> dict[str, Properties] | list[set[Properties]] | set[Properties]:
        """Package the results of a pquery into the right output format.

        Args:
            select (str or bytes or Callable):
                A statement defining the value to look up from the
                annotation properties. If `select = "*"`, all properties
                are returned for each annotation (`unique` must be
                False).
            unique (bool):
                If True, only unique values for each selected property
                will be returned as a list of sets. If False, all values
                will be returned as a dictionary mapping keys values.
                Defaults to True.
            squeeze (bool):
                If True, when querying for a single value with
                `unique=True`, the result will be a single set instead
                of a list of sets.
            items (Dict[str, Properties]):
                A dictionary mapping annotation keys/IDs to annotation
                properties.
            get_values (Callable):
                A function to get the values to return from an
                annotation via a select.

        Returns:
            results (dict[str, object] | list[set] | set):
                results

        """  # Q440, Q441
        result_set: defaultdict[str, set] = defaultdict(set)
        result_dict: dict = {}
        for key, annotation in items:
            values = get_values(select, annotation)
            if unique:
                # Wrap scalar values in a tuple
                if not isinstance(values, tuple):
                    values = (values,)
                # Add each value to the result set
                for i, value in enumerate(values):
                    result_set[str(i)].add(value)
            else:
                result_dict[key] = values
        if unique:
            results = list(result_set.values())
            if squeeze and len(results) == 1:
                return results[0]
            return results

        return result_dict

    def features(self: AnnotationStore) -> Generator[dict[str, object], None, None]:
        """Return annotations as a list of geoJSON features.

        Returns:
            list:
                List of features as dictionaries.

        """
        for a in self.values():
            yield a.to_feature()

    def to_geodict(self: AnnotationStore) -> dict[str, object]:
        """Return annotations as a dictionary in geoJSON format.

        Returns:
            dict:
                Dictionary of annotations in geoJSON format.

        """
        return {
            "type": "FeatureCollection",
            "features": list(self.features()),
        }

    @staticmethod
    def _dump_cases(
        fp: IO | str | Path | None,
        file_fn: Callable[[IO], None],
        none_fn: Callable[[], str | bytes],
    ) -> str | bytes | None:
        """Helper function to handle cases for dumping.

        Args:
            fp:
                The file path or handle to dump to.
            file_fn(Callable):
                The function to call when fp is a file handle.
            none_fn(Callable):
                The function to call when fp is None.

        Returns:
            str | bytes | None:
                The result of dump. Depends on the provided functions.

        """
        if fp is not None:
            # It is a file-like object, write to it
            if hasattr(fp, "write"):
                file_handle = cast(IO, fp)
                return file_fn(file_handle)  # type: ignore[func-returns-value]
            # Turn a path into a file handle, then write to it
            with Path(fp).open("w", encoding="utf-8") as file_handle:
                return file_fn(file_handle)  # type: ignore[func-returns-value]
        # Return as str or bytes if no handle/path is given
        return none_fn()

    @staticmethod
    def _load_cases(
        fp: IO | str | Path,
        string_fn: Callable[[str | bytes], object],
        file_fn: Callable[[IO], object],
    ) -> object:
        """Loads cases for an input file handle or path."""
        with contextlib.suppress(OSError):
            if isinstance(fp, (Path, str)) and Path(fp).exists():
                with Path(fp).open() as file_handle:
                    return file_fn(file_handle)
        if isinstance(fp, (str, bytes)):
            return string_fn(fp)
        if hasattr(fp, "read"):
            file_io = cast(IO, fp)
            return file_fn(file_io)
        msg = "Invalid file handle or path."
        raise OSError(msg)

    @classmethod
    def from_geojson(
        cls: type[AnnotationStore],
        fp: IO | str,
        scale_factor: tuple[float, float] = (1, 1),
        origin: tuple[float, float] = (0, 0),
        transform: Callable[[Annotation], Annotation] | None = None,
    ) -> AnnotationStore:
        """Create a new database with annotations loaded from a geoJSON file.

        Args:
            fp (Union[IO, str, Path]):
                The file path or handle to load from.
            scale_factor (Tuple[float, float]):
                The scale factor in each dimension to use when loading the annotations.
                All coordinates will be multiplied by this factor to allow import of
                annotations saved at non-baseline resolution.
            origin (Tuple[float, float]):
                The x and y coordinates to use as the origin for the annotations.
            transform (Callable):
                A function to apply to each annotation after loading. Should take an
                annotation as input and return an annotation. Defaults to None.
                Intended to facilitate modifying the way annotations are loaded to
                accomodate the specifics of different annotation formats.

        Returns:
            AnnotationStore:
                A new annotation store with the annotations loaded from the file.

        Example:
            To load annotations from a GeoJSON exported by QuPath, with measurements
            stored in a 'measurements' property as a list of name-value pairs, and
            unpack those measurements into a flat dictionary of properties of
            each annotation:
            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> def unpack_qupath(ann: Annotation) -> Annotation:
            >>>    #Helper function to unpack QuPath measurements.
            >>>    props = ann.properties
            >>>    measurements = props.pop("measurements")
            >>>    for m in measurements:
            >>>        props[m["name"]] = m["value"]
            >>>    return ann
            >>> store = SQLiteStore.from_geojson(
            ...     "exported_file.geojson",
            ...     transform=unpack_qupath,
            ... )

        """
        store = cls()
        if transform is None:

            def transform(annotation: Annotation) -> Annotation:
                """Default import transform. Does Nothing."""
                return annotation

        store.add_from_geojson(
            fp,
            scale_factor,
            origin=origin,
            transform=transform,
        )
        return store

    def add_from_geojson(
        self: AnnotationStore,
        fp: IO | str,
        scale_factor: tuple[float, float] = (1, 1),
        origin: tuple[float, float] = (0, 0),
        transform: Callable[[Annotation], Annotation] = lambda x: x,
    ) -> None:
        """Add annotations from a .geojson file to an existing store.

        Make the best effort to create valid shapely geometries from provided contours.

        Args:
            fp (Union[IO, str, Path]):
                The file path or handle to load from.
            scale_factor (float):
                The scale factor to use when loading the annotations. All coordinates
                will be multiplied by this factor to allow import of annotations saved
                at non-baseline resolution.
            origin (Tuple[float, float]):
                The x and y coordinates to use as the origin for the annotations.
            transform (Callable):
                A function to apply to each annotation after loading. Should take an
                annotation as input and return an annotation. Defaults to None.
                Intended to facilitate modifying the way annotations are loaded to
                accommodate the specifics of different annotation formats.

        """

        def transform_geometry(geom: Geometry) -> Geometry:
            """Helper function to transform a geometry if needed."""
            if origin != (0, 0):
                # transform coords to be relative to given origin.
                geom = translate(geom, -origin[0], -origin[1])
            if scale_factor != (1, 1):
                geom = scale(
                    geom,
                    xfact=scale_factor[0],
                    yfact=scale_factor[1],
                    origin=(0, 0, 0),
                )
            return geom

        geojson = self._load_cases(
            fp=fp,
            string_fn=json.loads,
            file_fn=json.load,
        )
        geojson = cast(dict, geojson)

        annotations = [
            transform(
                Annotation(
                    transform_geometry(
                        feature2geometry(feature["geometry"]),
                    ),
                    feature["properties"],
                ),
            )
            for feature in geojson["features"]
        ]

        logger.info("Adding %d annotations.", len(annotations))
        self.append_many(annotations)

    def to_geojson(
        self: AnnotationStore,
        fp: IO | str | Path | None = None,
    ) -> str | None:
        """Serialise the store to geoJSON.

        For more information on the geoJSON format see:
        - https://geojson.org/
        - https://tools.ietf.org/html/rfc7946

        Args:
             fp (IO):
                A file-like object supporting `.read`. Defaults to None
                which returns geoJSON as a string.

        Returns:
            Optional[str]:
                None if writing to file or the geoJSON string if `fp` is
                None.

        """

        def write_geojson_to_file_handle(file_handle: IO) -> None:
            """Write the store to a GeoJson file give a handle.

            This replaces the naive method which uses a lot of memory::
                json.dump(self.to_geodict(), file_handle)

            """
            # Write head
            file_handle.write('{"type": "FeatureCollection", "features": [')
            # Write each feature
            for feature in self.features():
                file_handle.write(json.dumps(feature))  # skipcq: PY-W0079
                tell = file_handle.tell()
                # Comma separate features
                file_handle.write(",")
            # Seek to before last comma
            file_handle.seek(tell, os.SEEK_SET)
            # Write tail
            file_handle.write("]}")

        result = self._dump_cases(
            fp=fp,
            file_fn=write_geojson_to_file_handle,
            none_fn=lambda: json.dumps(self.to_geodict()),
        )
        if result is not None:
            return cast(str, result)
        return result

    @overload
    def to_ndjson(
        self: AnnotationStore, fp: None = None
    ) -> str: ...  # pragma: no cover

    @overload
    def to_ndjson(
        self: AnnotationStore, fp: IO | str | Path
    ) -> None: ...  # pragma: no cover

    def to_ndjson(
        self: AnnotationStore, fp: IO | str | Path | None = None
    ) -> str | None:
        """Serialise to New Line Delimited JSON.

        Each line contains a JSON object with the following format:

        .. code-block:: json

           {
                "key": "...",
                "geometry": {
                    "type": "...",
                    "coordinates": [...]
                },
                "properties": {
                    "...": "..."
                }
           }

        That is a geoJSON object with an additional key field.

        For more information on the NDJSON format see:
        - ndjson Specification: http://ndjson.org
        - JSON Lines Documentation: https://jsonlines.org
        - Streaming JSON: https://w.wiki/4Qan
        - GeoJSON RFC: https://tools.ietf.org/html/rfc7946
        - JSON RFC: https://tools.ietf.org/html/rfc7159


        Args:
            fp (IO): A file-like object supporting `.read`. Defaults to
                None which returns geoJSON as a string.

        Returns:
            Optional[str]:
                None if writing to file or the geoJSON string if`fp` is
                None.

        """
        string_lines_generator = (
            json.dumps({"key": key, **annotation.to_feature()}, separators=(",", ":"))
            + "\n"
            for key, annotation in self.items()
        )
        result = self._dump_cases(
            fp=fp,
            file_fn=lambda fp: fp.writelines(string_lines_generator),
            none_fn=lambda: "".join(string_lines_generator),
        )
        if result is not None:
            return cast(str, result)
        return result

    @classmethod
    def from_ndjson(cls: type[AnnotationStore], fp: Path | IO | str) -> AnnotationStore:
        """Load annotations from NDJSON.

        Expects each line to be a JSON object with the following format:

        .. code-block:: json

           {
                "key": "...",
                "geometry": {
                    "type": "...",
                    "coordinates": [...]
                },
                "properties": {
                    "...": "..."
                }
           }

        That is a geoJSON object with an additional key field. If this key
        field is missing, then a new UUID4 key will be generated for this
        annotation.

        Args:
            fp (IO): A file-like object supporting `.read`.

        Returns:
            AnnotationStore:
                The loaded annotations.

        """
        store = cls()
        cases = cls._load_cases(
            fp=fp,
            string_fn=lambda fp: fp.splitlines(),
            file_fn=lambda fp: fp.readlines(),
        )
        cases = cast(list, cases)
        for line in cases:
            dictionary = json.loads(line)
            key = dictionary.get("key", uuid.uuid4().hex)
            geometry = feature2geometry(dictionary["geometry"])
            properties = dictionary["properties"]
            store.append(Annotation(geometry, properties), key)
        return store

    @classmethod
    def from_dataframe(cls: type[AnnotationStore], df: pd.DataFrame) -> AnnotationStore:
        """Converts to AnnotationStore from :class:`pandas.DataFrame`."""
        store = cls()
        for key, row in df.iterrows():
            geometry = row["geometry"]
            properties = dict(row.filter(regex="^(?!geometry|key).*$"))
            store.append(Annotation(geometry, properties), str(key))
        return store

    def to_dataframe(self: AnnotationStore) -> pd.DataFrame:
        """Converts AnnotationStore to :class:`pandas.DataFrame`."""
        features: list[dict] = []
        for key, annotation in self.items():
            feature_dict = {
                "key": key,
                "geometry": annotation.geometry,
                "properties": annotation.properties,
            }
            features.append(feature_dict)

        return pd.json_normalize(features).set_index("key")

    def transform(
        self: AnnotationStore,
        transform: Callable[[Geometry], Geometry],
    ) -> None:
        """Transform all annotations in the store using provided function.

        Useful for transforming coordinates from slide space into
        patch/tile/core space, or to a different resolution, for example.

        Args:
            transform (Callable[Geometry, Geometry]):
                A function that takes a geometry and returns a new
                transformed geometry.

        """
        transformed_geoms = {
            key: transform(annotation.geometry) for key, annotation in self.items()
        }
        _keys = transformed_geoms.keys()
        _values = transformed_geoms.values()
        self.patch_many(_keys, _values)

    def __del__(self: AnnotationStore) -> None:
        """Implements destructor method.

        This should be called when all references to the object have been deleted
        i.e., when an object is garbage collected.

        """
        self.close()

    @abstractmethod
    def close(self: AnnotationStore) -> None:
        """Closes :class:`AnnotationStore` from file pointer or path."""

    def clear(self: AnnotationStore) -> None:
        """Remove all annotations from the store.

        This is a naive implementation, it simply iterates over all annotations
        and removes them. Faster implementations may be possible in specific
        cases and may be implemented by subclasses.

        """
        for key in list(self.keys()):
            del self[key]


class SQLiteMetadata(MutableMapping):
    """Metadata storage for an SQLiteStore.

    Attributes:
        con (sqlite3.Connection):
            The sqlite3 database connection.

    """

    def __init__(self: SQLiteMetadata, con: sqlite3.Connection) -> None:
        """Initialize :class:`SQLiteMetadata`."""
        self.con = con
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT UNIQUE, value TEXT)",
        )
        self.con.commit()

    def __contains__(self: SQLiteMetadata, key: object) -> bool:
        """Test whether the object contains the specified object or not."""
        cursor = self.con.execute("SELECT 1 FROM metadata WHERE [key] = ?", (key,))
        return cursor.fetchone() is not None

    def __setitem__(
        self: SQLiteMetadata,
        key: str,
        value: dict | list | float | str,
    ) -> None:
        """Set a metadata value."""
        value = json.dumps(value)
        self.con.execute(
            "REPLACE INTO metadata (key, value) VALUES (?,?)",
            (key, value),
        )
        self.con.commit()

    def __getitem__(self: SQLiteMetadata, key: str) -> dict | list | int | float | str:
        """Get a metadata value."""
        cursor = self.con.execute("SELECT value FROM metadata WHERE [key] = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError(key)
        return json.loads(result[0])

    def __delitem__(self: SQLiteMetadata, key: str) -> None:
        """Delete a metadata value."""
        if key not in self:
            raise KeyError(key)
        self.con.execute("DELETE FROM metadata WHERE [key] = ?", (key,))

    def __iter__(self: SQLiteMetadata) -> Iterator[str]:
        """Iterate over all keys."""
        cursor = self.con.execute("SELECT [key] FROM metadata")
        for row in cursor:
            yield row[0]

    def __len__(self: SQLiteMetadata) -> int:
        """Return the number of metadata entries."""
        cursor = self.con.execute("SELECT COUNT(*) FROM metadata")
        return cursor.fetchone()[0]


class SQLiteStore(AnnotationStore):
    """SQLite backed annotation store.

    Uses and rtree index for fast spatial queries.

    Version History:
        1.0.0:
            Initial version.
        1.0.1 (07/10/2022):
            Added optional "area" column and queries sorted/filtered by area.

    """

    @classmethod
    def open(cls: type[SQLiteStore], fp: Path | str | IO) -> SQLiteStore:
        """Opens :class:`SQLiteStore` from file pointer or path."""
        return SQLiteStore(fp)

    def __init__(
        self: SQLiteStore,
        connection: Path | str | IO = ":memory:",
        compression: str = "zlib",
        compression_level: int = 9,
        *,
        auto_commit: bool = True,
    ) -> None:
        """Initialize :class:`SQLiteStore`."""
        super().__init__()
        # Check that JSON and RTree support is enabled
        compile_options = self.compile_options()
        if sqlite3.sqlite_version_info >= (3, 38, 0):
            if not all(
                ["OMIT_JSON" not in compile_options, "ENABLE_RTREE" in compile_options],
            ):
                msg = (
                    "RTREE sqlite3 compile option is required, and\n"
                    "JSON must not be disabled with OMIT_JSON compile option"
                )
                raise OSError(
                    msg,
                )
        elif not all(
            ["ENABLE_JSON1" in compile_options, "ENABLE_RTREE" in compile_options],
        ):
            msg = "RTREE and JSON1 sqlite3 compile options are required."
            raise OSError(msg)

        # Check that math functions are enabled
        if "ENABLE_MATH_FUNCTIONS" not in compile_options:
            logger.warning(
                "SQLite math functions are not enabled."
                " This may cause problems with some queries."
                " For example, floor division (//) will not work."
                " For a full list see https://tia-toolbox.readthedocs.io/"
                "en/v%s/_autosummary/tiatoolbox.annotation.dsl.html",
                tiatoolbox.__version__,
            )

        # Set up database connection and cursor
        self.connection = connection
        self.path = self._connection_to_path(self.connection)
        self.auto_commit = auto_commit

        # Check if the path is a non-empty file
        exists = (
            # Use 'and' to short-circuit
            self.path.is_file() and self.path.stat().st_size > 0
        )
        self.cons: dict = {}
        self.con.execute("BEGIN")

        # Set up metadata
        self.metadata = SQLiteMetadata(self.con)
        if not exists:
            self.metadata["version"] = "1.0.1"
            self.metadata["compression"] = compression
            self.metadata["compression_level"] = compression_level

        # store locally as constantly fetching from db in (de)serialization is slow
        self.compression = self.metadata["compression"]
        self.compression_level: int
        self.compression_level = (
            self.metadata["compression_level"]
            if isinstance(self.metadata["compression_level"], int)
            else compression_level
        )

        if exists:
            self.table_columns = self._get_table_columns()
            return

        # Create tables for geometry and RTree index
        self.con.execute(
            """
            CREATE VIRTUAL TABLE rtree USING rtree(
                id,                      -- Integer primary key
                min_x, max_x,            -- 1st dimension min, max
                min_y, max_y             -- 2nd dimension min, max
            )
            """,
        )
        self.con.execute(
            """
            CREATE TABLE annotations(
                id INTEGER PRIMARY KEY,  -- Integer primary key
                key TEXT UNIQUE,         -- Unique identifier (UUID)
                objtype TEXT,            -- Object type
                cx FLOAT NOT NULL,       -- X of centroid/representative point
                cy FLOAT NOT NULL,       -- Y of centroid/representative point
                geometry BLOB,           -- Detailed geometry
                properties TEXT,         -- JSON properties
                area FLOAT NOT NULL      -- Area (for ordering)
            )

            """,
        )
        if self.auto_commit:
            self.con.commit()
        self.table_columns = self._get_table_columns()

    def __getattribute__(self: SQLiteStore, name: str) -> Any:  # noqa: ANN401
        """If attr is con, return thread-local connection."""
        if name == "con":
            return self.get_connection(threading.get_ident())
        return super().__getattribute__(name)

    def get_connection(self: SQLiteStore, thread_id: int) -> sqlite3.Connection:
        """Get a connection to the database."""
        if thread_id not in self.cons:
            con = sqlite3.connect(str(self.path), isolation_level="DEFERRED", uri=True)

            # Register predicate functions as custom SQLite functions
            def wkb_predicate(
                name: str,
                wkb_a: bytes,
                b: bytes,
                cx: float,
                cy: float,
            ) -> bool:
                """Wrapper function to allow WKB as inputs to binary predicates."""
                a = shapely_wkb.loads(wkb_a)
                b = self._unpack_geometry(b, cx, cy)
                return self._geometry_predicate(name, a, b)

            def pickle_expression(pickle_bytes: bytes, properties: str) -> bool:
                """Function to load and execute pickle bytes with "properties" dict."""
                fn = pickle.loads(pickle_bytes)  # skipcq: BAN-B301  # noqa: S301
                properties = json.loads(properties)
                return fn(properties)

            def get_area(wkb_bytes: bytes, cx: float, cy: float) -> float:
                """Function to get the area of a geometry."""
                return self._unpack_geometry(
                    wkb_bytes,
                    cx,
                    cy,
                ).area

            # Register custom functions
            def register_custom_function(
                name: str,
                nargs: int,
                fn: Callable,
                *,
                deterministic: bool = False,
            ) -> None:
                """Register a custom SQLite function.

                Only Python >= 3.8 supports deterministic functions,
                fallback to without this argument if not available.

                Args:
                    name:
                        The name of the function.
                    nargs:
                        The number of arguments the function takes.
                    fn:
                        The function to register.
                    deterministic:
                        Whether the function is deterministic.

                """
                con.create_function(
                    name,
                    nargs,
                    fn,
                    deterministic=deterministic,
                )

            register_custom_function(
                "geometry_predicate",
                5,
                wkb_predicate,
                deterministic=True,
            )
            register_custom_function(
                "pickle_expression",
                2,
                pickle_expression,
                deterministic=True,
            )
            register_custom_function("REGEXP", 2, py_regexp)
            register_custom_function("REGEXP", 3, py_regexp)
            register_custom_function("LISTSUM", 1, json_list_sum)
            register_custom_function("CONTAINS", 1, json_contains)
            register_custom_function("get_area", 3, get_area)
            self.cons[thread_id] = con
            return con
        return self.cons[thread_id]

    def serialise_geometry(  # type: ignore[override]  # skipcq: PYL-W0221
        self: SQLiteStore,
        geometry: Geometry,
    ) -> str | bytes:
        """Serialise a geometry to WKB with optional compression.

        Converts shapely geometry objects to well-known binary (WKB) and
        applies optional compression.

        Args:
            geometry(Geometry):
                The Shapely geometry to be serialised.

        Returns:
            bytes or str:
                The serialised geometry.

        """
        data = geometry.wkb
        if self.compression is None:
            return data
        if self.compression == "zlib":
            return zlib.compress(data, level=self.compression_level)
        msg = "Unsupported compression method."
        raise ValueError(msg)

    def _unpack_geometry(
        self: SQLiteStore,
        data: str | bytes,
        cx: float,
        cy: float,
    ) -> Geometry:
        """Return the geometry using WKB data and rtree bounds index.

        For space optimisation, points are stored as centroids and all
        other geometry types are stored as WKB. This function unpacks
        the WKB data and uses the rtree index to find the centroid for
        points where the data is null.

        Args:
            data (bytes or str):
                The WKB/WKT data to be unpacked.
            cx (int):
                The X coordinate of the centroid/representative point.
            cy (float):
                The Y coordinate of the centroid/representative point.

        Returns:
            Geometry:
                The Shapely geometry.

        """
        return Point(cx, cy) if data is None else self.deserialize_geometry(data)

    def _unpack_wkb(
        self: SQLiteStore,
        data: bytes,
        cx: float,
        cy: float,
    ) -> bytes:
        """Return the geometry as bytes using WKB.

        Args:
            data (bytes or str):
                The WKB/WKT data to be unpacked.
            cx (int):
                The X coordinate of the centroid/representative point.
            cy (float):
                The Y coordinate of the centroid/representative point.

        Returns:
            bytes:
                The geometry as bytes.

        """
        return (
            self._decompress_data(data)
            if data
            else WKB_POINT_STRUCT.pack(1, GeometryType.POINT, cx, cy)
        )

    def deserialize_geometry(  # type: ignore[override]  # skipcq: PYL-W0221
        self: SQLiteStore, data: bytes | str
    ) -> Geometry:
        """Deserialize a geometry from a string or bytes.

        Args:
            data(bytes or str):
                The serialised representation of a Shapely geometry.

        Returns:
            Geometry:
                The deserialized Shapely geometry.

        """
        data = self._decompress_data(data)
        if isinstance(data, str):
            return shapely_wkt.loads(data)
        return shapely_wkb.loads(data)

    def _decompress_data(self: SQLiteStore, data: bytes | str) -> bytes:
        """Decompresses geometry data.

        Args:
            data (bytes):
                The data to be decompressed.

        Returns:
            bytes:
                The decompressed data.

        Raises:
            ValueError:
                If the compression method is unsupported.

        """
        if self.compression == "zlib":
            # No type annotation avaliable for Buffer until Python12
            data = zlib.decompress(data)  # type: ignore[arg-type]
        elif self.compression is not None:
            msg = "Unsupported compression method."
            raise ValueError(msg)
        return data

    @staticmethod
    def compile_options() -> list[str]:
        """Get the list of options that sqlite3 was compiled with.

        Example:
            >>> for opt in SQLiteRTreeStore.compile_options():
            >>>     print(opt)
            COMPILER=gcc-7.5.0
            ENABLE_COLUMN_METADATA
            ENABLE_DBSTAT_VTAB
            ENABLE_FTS3
            ENABLE_FTS3_PARENTHESIS
            ENABLE_FTS3_TOKENIZER
            ENABLE_FTS4
            ENABLE_FTS5
            ENABLE_JSON1
            ENABLE_LOAD_EXTENSION
            ENABLE_PREUPDATE_HOOK
            ENABLE_RTREE
            ENABLE_SESSION
            ENABLE_STMTVTAB
            ENABLE_UNLOCK_NOTIFY
            ENABLE_UPDATE_DELETE_LIMIT
            HAVE_ISNAN
            LIKE_DOESNT_MATCH_BLOBS
            MAX_SCHEMA_RETRY=25
            MAX_VARIABLE_NUMBER=250000
            OMIT_LOOKASIDE
            SECURE_DELETE
            SOUNDEX
            TEMP_STORE=1
            THREADSAFE=1

        """
        with sqlite3.connect(":memory:") as conn:
            conn.enable_load_extension(True)  # noqa: FBT003
            options = conn.execute("pragma compile_options").fetchall()
        return [opt for (opt,) in options]

    def close(self: SQLiteStore) -> None:
        """Closes :class:`SQLiteStore` from file pointer or path."""
        if self.auto_commit:
            self.con.commit()
        self.optimize(vacuum=False, limit=1000)
        for con in self.cons.values():
            con.close()
        self.cons = {}

    def _make_token(self: SQLiteStore, annotation: Annotation, key: str | None) -> dict:
        """Create token data dict for tokenized SQL transaction."""
        key = key or str(uuid.uuid4())
        geometry = annotation.geometry
        if geometry.geom_type == "Point":
            serialised_geometry = None
        else:
            serialised_geometry = self.serialise_geometry(geometry)
        return {
            "key": key,
            "geometry": serialised_geometry,
            "cx": geometry.centroid.x,
            "cy": geometry.centroid.y,
            "min_x": geometry.bounds[0],
            "min_y": geometry.bounds[1],
            "max_x": geometry.bounds[2],
            "max_y": geometry.bounds[3],
            "geom_type": geometry.geom_type,
            "properties": json.dumps(annotation.properties, separators=(",", ":")),
            "area": geometry.area,
        }

    def append_many(
        self: SQLiteStore,
        annotations: Iterable[Annotation],
        keys: Iterable[str] | None = None,
    ) -> list[str]:
        """Appends new annotations to specified keys."""
        annotations = list(annotations)
        keys = list(keys) if keys else [str(uuid.uuid4()) for _ in annotations]
        self._validate_equal_lengths(keys, annotations)
        cur = self.con.cursor()
        if self.auto_commit:
            cur.execute("BEGIN")
        result = []
        for annotation, key in zip(annotations, keys):
            self._append(key, annotation, cur)
            result.append(key)
        if self.auto_commit:
            self.con.commit()
        return result

    def _append(
        self: SQLiteStore,
        key: str,
        annotation: Annotation,
        cur: sqlite3.Cursor,
    ) -> None:
        """Append without starting a transaction.

        Args:
            key(str):
                The unique identifier (UUID) for the annotation.
            annotation(Annotation):
                The annotation to be appended.
            cur(sqlite3.Cursor):
                The cursor to use for the transaction.

        """
        token = self._make_token(
            annotation=annotation,
            key=key,
        )
        cur.execute(
            """
                INSERT INTO annotations VALUES(
                    NULL, :key, :geom_type,
                    :cx, :cy, :geometry, :properties, :area
                )

                """,
            token,
        )
        row_id = cur.lastrowid
        token.update({"row_id": row_id})
        cur.execute(
            """
                INSERT INTO rtree VALUES(
                    :row_id, :min_x, :max_x, :min_y, :max_y
                )
                """,
            token,
        )

    @staticmethod
    def _initialize_query_string_parameters(
        query_geometry: Geometry | None,
        query_parameters: dict[str, object],
        geometry_predicate: str | None,
        columns: str,
        where: bytes | str | CallablePredicate | None,
        distance: float = 0,
    ) -> tuple[str, dict[str, object]]:
        """Initialises the query string and parameters."""
        query_string = (
            "SELECT "  # skipcq: BAN-B608  # noqa: S608
            + columns  # skipcq: BAN-B608
            + """
                 FROM annotations, rtree
                WHERE annotations.id == rtree.id
                """
        )

        # There is query geometry, add a simple rtree bounds check to
        # rapidly narrow candidates down.
        if query_geometry is not None:
            # Add rtree index checks to the query.
            # For special case of centers_within_k, Check for
            # center of the annotation bounds within query geometry
            # centroid + k.
            if geometry_predicate == "centers_within_k":
                # Use rtree index to check distance between points
                query_string += (
                    "AND (((:min_x + :max_x)/2 - (min_x + max_x)/2)*"
                    "((:min_x + :max_x)/2 - (min_x + max_x)/2) + "
                    " ((:min_y + :max_y)/2 - (min_y + max_y)/2)*"
                    "((:min_y + :max_y)/2 - (min_y+ max_y)/2)) < :distance2 "
                )
                query_parameters["distance2"] = distance**2
            # Otherwise, perform a regular bounding box intersection
            else:
                query_string += (
                    "AND max_x >= :min_x "
                    "AND min_x <= :max_x "
                    "AND max_y >= :min_y "
                    "AND min_y <= :max_y "
                )

            # Find the bounds of the geometry for the rtree index
            min_x, min_y, max_x, max_y = query_geometry.bounds

            # Update query parameters
            query_parameters.update(
                {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "geometry_predicate": geometry_predicate,
                    "query_geometry": query_geometry.wkb,
                },
            )

            # The query is a full intersection check, not a simple bounds
            # check only.
            if geometry_predicate is not None and geometry_predicate not in (
                "bbox_intersects",
                "centers_within_k",
            ):
                query_string += (
                    "\nAND geometry_predicate("
                    ":geometry_predicate, :query_geometry, geometry, cx, cy"
                    ") "
                )
                query_parameters["geometry_predicate"] = geometry_predicate
                query_parameters["query_geometry"] = query_geometry.wkb

        # Predicate is pickled function
        if isinstance(where, bytes):
            query_string += "\nAND pickle_expression(:where, properties)"
            query_parameters["where"] = where
        # Predicate is a string
        if isinstance(where, str):
            sql_predicate = eval(  # skipcq: PYL-W0123,  # noqa: S307
                where,
                SQL_GLOBALS,
                {},
            )
            query_string += f" AND {sql_predicate}"

        return query_string, query_parameters

    def _query(
        self: SQLiteStore,
        columns: str,
        geometry: Geometry | None = None,
        callable_columns: str | None = None,
        geometry_predicate: str = "intersects",
        where: Predicate | None = None,
        min_area: float | None = None,
        distance: float = 0,
        *,
        unique: bool = False,
        no_constraints_ok: bool = False,
        index_warning: bool = False,
    ) -> sqlite3.Cursor:
        """Common query construction logic for `query` and `iquery`.

        Args:
            columns(str):
                The columns to select.
            geometry(tuple or Geometry):
                The geometry being queried against.
            callable_columns(str):
                The columns to select when a Callable is given to
                `where`.
            geometry_predicate(str):
                A string which define which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.
            where (str or bytes or Callable):
                The predicate to evaluate against candidate properties
                during the query.
            unique(bool):
                Whether to return only unique results. Defaults to
                False.
            no_constraints_ok(bool):
                Whether to allow the query to return results without
                constraints (e.g. when the geometry or where predicate
                is not provided). Defaults to False.
            index_warning(bool):
                Whether to warn if the query is not using an index.
                Defaults to False.
            min_area (float or None):
                Minimum area of the annotations to be returned.
                Defaults to None.
            distance (float):
                Distance used when performing a distance based query.
                E.g. "centers_within_k" geometry predicate.

        Returns:
            sqlite3.Cursor:
                A database cursor for the current query.

        """
        if not no_constraints_ok and all(x is None for x in (geometry, where)):
            msg = "At least one of `geometry` or `where` must be specified."
            raise ValueError(msg)
        query_geometry = geometry
        if callable_columns is None:
            callable_columns = columns
        if geometry_predicate not in self._geometry_predicate_names:
            msg = (
                "Invalid geometry predicate. Allowed values are: "
                f"{', '.join(self._geometry_predicate_names)}."
            )
            raise ValueError(
                msg,
            )

        cur = self.con.cursor()

        # Normalise query geometry and determine if it is a rectangle
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)

        if callable(where):
            columns = callable_columns

        query_parameters: dict[str, object] = {}

        query_string, query_parameters = self._initialize_query_string_parameters(
            query_geometry,
            query_parameters,
            geometry_predicate,
            columns,
            where,
            distance=distance,
        )

        # Add area column constraint to query if min_area is specified
        if min_area is not None and "area" in self.table_columns:
            query_string += f"\nAND area > {min_area}"
        elif min_area is not None:
            msg = (
                "Cannot use `min_area` without an area column.\n"
                "SQLiteStore.add_area_column() can be used to add an area column."
            )
            raise ValueError(
                msg,
            )

        if unique:
            query_string = query_string.replace("SELECT", "SELECT DISTINCT")

        # Warn if the query is not using an index
        if index_warning:
            query_plan = cur.execute(
                "EXPLAIN QUERY PLAN " + query_string,
                query_parameters,
            ).fetchone()
            if "USING INDEX" not in query_plan[-1]:
                logger.warning(
                    "Query is not using an index. "
                    "Consider adding an index to improve performance.",
                    stacklevel=2,
                )
        # if area column exists, sort annotations by area
        if "area" in self.table_columns:
            query_string += "\nORDER BY area DESC"
        cur.execute(query_string, query_parameters)
        return cur

    def iquery(
        self: SQLiteStore,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        distance: float = 0,
    ) -> list[str]:
        """Query the store for annotation keys.

        Acts the same as `AnnotationStore.query` except returns keys
        instead of annotations.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon).
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore `and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.
            geometry_predicate (str):
                A string which define which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.
            distance (float):
                Distance used when performing a distance based query.
                E.g. "centers_within_k" geometry predicate.

        Returns:
            list:
                A list of keys for each Annotation.

        """
        query_geometry = geometry
        cur = self._query(
            "[key]",
            geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            where=where,
            callable_columns="[key], properties",
            distance=distance,
        )
        if callable(where):
            return [
                key
                for key, properties in cur.fetchall()
                if where(json.loads(properties))
            ]
        return [key for (key,) in cur.fetchall()]

    def query(
        self: SQLiteStore,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        min_area: float | None = None,
        distance: float = 0,
    ) -> dict[str, Annotation]:
        """Runs Query."""
        query_geometry = geometry
        cur = self._query(
            columns="[key], properties, cx, cy, geometry",
            geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            where=where,
            min_area=min_area,
            distance=distance,
        )
        if callable(where):
            return {
                key: Annotation(
                    properties=json.loads(properties),
                    wkb=self._unpack_wkb(blob, cx, cy),
                )
                for key, properties, cx, cy, blob in cur.fetchall()
                if where(json.loads(properties))
            }
        return {
            key: Annotation(
                properties=json.loads(properties),
                wkb=self._unpack_wkb(blob, cx, cy),
            )
            for key, properties, cx, cy, blob in cur.fetchall()
        }

    def bquery(
        self: SQLiteStore,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Query the store for annotation bounding boxes.

        Acts similarly to `AnnotationStore.query` except it checks for
        intersection between stored and query geometry bounding boxes.
        This may be faster than a regular query in some cases, e.g. for
        SQliteStore with a large number of annotations.

        Note that this method only checks for bounding box intersection
        and therefore may give a different result to using
        `AnnotationStore.query` with a box polygon and the "intersects"
        geometry predicate. Also note that geometry predicates are not
        supported for this method.

        Args:
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon). If a geometry is provided, the bounds of the
                geometry will be used for the query. Full geometry
                intersection is not used for the query method.
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query.
                Additionally, the same string can be used across
                different backends (e.g. the previous example predicate
                string is valid for both `DictionaryStore` and a
                `SQliteStore`). On the other hand it has many more
                limitations. It is important to note that untrusted user
                input should never be accepted to this argument as
                arbitrary code can be run via pickle or the parsing of
                the string statement.

        Returns:
            list:
                A list of bounding boxes for each Annotation.

        Example:
            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> from shapely.geometry import Polygon
            >>> store = SQLiteStore()
            >>> store.append(
            ...     Annotation(
            ...         geometry=Polygon.from_bounds(0, 0, 1, 1),
            ...         properties={"class": 42},
            ...     ),
            ...     key="foo",
            ... )
            >>> store.bquery(where="props['class'] == 42")
            {'foo': (0.0, 0.0, 1.0, 1.0)}

        """
        cur = self._query(
            columns="[key], min_x, min_y, max_x, max_y",
            geometry=geometry,
            geometry_predicate="bbox_intersects",
            where=where,
            callable_columns="[key], properties, min_x, min_y, max_x, max_y",
        )
        if callable(where):
            return {
                key: bounds
                for key, properties, *bounds in cur.fetchall()
                if where(json.loads(properties))
            }
        return {key: bounds for key, *bounds in cur.fetchall()}

    @staticmethod
    def _handle_pickle_callable_pquery(
        select: CallableSelect,
        where: CallablePredicate | None,
        cur: sqlite3.Cursor,
        *,
        unique: bool,
    ) -> list[set[Properties]] | dict[str, Properties]:
        """Package the results of a pquery into the right output format.

        This variant is used when select and where are Callable or
        pickle objects.

        Args:
            select (Union[str, bytes, Callable]):
                A Callable to select the properties to return.
            where (CallablePredicate):
                A Callable predicate to filter the rows with. Maybe
                None for no-op (no filtering).
            cur (sqlite3.Cursor):
                The cursor for the query.
            unique (bool):
                Whether to return only unique results.

        Returns:
            dict:
                If unique, a list of sets is returned. Otherwise,
                a dictionary mapping annotation keys to JSON-like
                property dictionaries is returned.

        """

        def add_props_to_result(
            result: defaultdict[str, set],
            properties: Properties,
        ) -> None:
            """Add the properties to the appropriate set in result.

            Args:
                result (DefaultDict[str, set]):
                    The result dictionary to add the properties to.
                properties (Dict[str, object]):
                    The properties to add to the result.

            """
            # Get the selected values
            selection = select(properties)
            # Wrap scalar values into a tuple
            if not isinstance(selection, tuple):
                selection_tuple = (selection,)
            else:
                selection_tuple = selection
            # Add the properties to the appropriate set
            for i, value in enumerate(selection_tuple):
                result[str(i)].add(value)

        # Load a pickled select function
        if isinstance(select, bytes):
            select = pickle.loads(select)  # skipcq: BAN-B301  # noqa: S301
        if unique:
            # Create a dictionary of sets to store the unique properties
            # for each property key / name.
            result: defaultdict[str, set] = defaultdict(set)
            for (properties_string,) in cur.fetchall():
                properties = json.loads(properties_string)
                # Apply where filter and skip if False
                if where and not where(properties):
                    continue
                add_props_to_result(result, properties)
            return list(result.values())
        if not where:
            return {
                key: select(json.loads(properties))
                for key, properties in cur.fetchall()
            }
        return {
            key: select(json.loads(properties))
            for key, properties in cur.fetchall()
            if where(json.loads(properties))
        }

    @staticmethod
    def _handle_str_pquery(
        cur: sqlite3.Cursor,
        *,
        unique: bool,
        star_query: bool,
    ) -> list[set[Properties]] | dict[str, Properties]:
        """Package the results of a pquery into the right output format.

        This variant is used when select and where are DSL strings.

        Args:
            cur (sqlite3.Cursor):
                The cursor for the query.
            unique (bool):
                Whether to return only unique results.
            star_query (bool):
                True if the query is a star query, i.e. select == "*".

        Returns:
            dict:
                If unique, a dictionary of sets is returned. Otherwise,
                a dictionary mapping annotation keys to JSON-like
                property dictionaries is returned.

        """
        if unique:
            result = defaultdict(set)
            for values in cur.fetchall():
                for i, value in enumerate(values):
                    result[i].add(value)
            return list(result.values())
        return {key: json.loads(x) if star_query else x for key, x in cur.fetchall()}

    @staticmethod
    def _kind_of_pquery(
        select: Select,
        where: Predicate | None,
    ) -> tuple[bool, bool, bool]:
        """Determine boolean flags for the kind of pquery this is.

        If either one of `select` or `where` is a str, bytes, or
        Callable, then is_callable_query, is_pickle_query, and
        is_str_query respectively will be set to True.

        Returns:
            tuple:
                A tuple of bools:
                - True if select or where are Callable (functions).
                - True if select or where are bytes (pickle expressions).
                - True if select or where are str (SQL expressions).

        """
        is_callable_query = any(callable(x) for x in (select, where) if x)
        is_pickle_query = any(isinstance(x, bytes) for x in (select, where) if x)
        is_str_query = any(isinstance(x, str) for x in (select, where) if x)

        return is_callable_query, is_pickle_query, is_str_query

    @staticmethod
    def _validate_select_where_type(
        select: Select,
        where: Predicate | None,
    ) -> None:
        """Validate that select and where are valid types.

        1. Check that select and where are the same type if where is given.
        2. Check that select is in (str, bytes, Callable).

        Raises:
            TypeError:
                If select and where are not the same type or not in
                (str, bytes, Callable).

        """
        if where is not None and type(select) is not type(where):
            msg = "select and where must be of the same type"
            raise TypeError(msg)
        if not isinstance(select, (str, bytes)) and not callable(select):
            msg = f"select must be str, bytes, or Callable, not {type(select)}"
            raise TypeError(
                msg,
            )

    def pquery(
        self: SQLiteStore,
        select: Select,
        geometry: QueryGeometry | None = None,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        *,
        unique: bool = True,
        squeeze: bool = True,
    ) -> dict[str, Properties] | list[set[Properties]] | set[Properties]:
        """Query the store for annotation properties.

        Acts similarly to `AnnotationStore.query` but returns only the
        value defined by `select`.

        Args:
            select (str or bytes or Callable):
                A statement defining the value to look up from the
                annotation properties. If `select = "*"`, all properties
                are returned for each annotation (`unique` must be
                False).
            geometry (Geometry or Iterable):
                Geometry to use when querying. This can be a bounds
                (iterable of length 4) or a Shapely geometry (e.g.
                Polygon). If a geometry is provided, the bounds of the
                geometry will be used for the query. Full geometry
                intersection is not used for the query method.
            where (str or bytes or Callable):
                A statement which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). This
                may be a string, Callable, or pickled function as bytes.
                Callables are called to filter each result returned
                from the annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are expected to be in a domain specific language
                and are converted to SQL on a best-effort basis. For
                supported operators of the DSL see
                :mod:`tiatoolbox.annotation.dsl`. E.g. a simple python
                expression `props["class"] == 42` will be converted to a
                valid SQLite predicate when using `SQLiteStore` and
                inserted into the SQL query. This should be faster than
                filtering in python after or during the query. It is
                important to note that untrusted user input should never
                be accepted to this argument as arbitrary code can be
                run via pickle or the parsing of the string statement.
            geometry_predicate (str):
                A string defining which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                "intersects". For more information see the `shapely
                documentation on binary predicates <https://shapely.
                readthedocs.io/en/stable/manual.html#binary-predicates>`_.
            unique (bool):
                If True, only unique values for each selected property
                will be returned as a list of sets. If False, all values
                will be returned as a dictionary mapping keys values.
                Defaults to True.
            squeeze (bool):
                If True, when querying for a single value with
                `unique=True`, the result will be a single set instead
                of a list of sets.

        Examples:
            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> from shapely.geometry import Point
            >>> store = SQLiteStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.append(annotation, "foo")
            >>> store.pquery("*", unique=False)
            ... {'foo': {'class': 42}}

            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> from shapely.geometry import Point
            >>> store = SQLiteStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.append(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42}
            >>> annotation =  Annotation(Point(1, 1), {"class": 123})
            >>> store.append(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42, 123}

        """
        self._validate_select_where_type(select, where)

        is_callable_query, is_pickle_query, is_str_query = self._kind_of_pquery(
            select,
            where,
        )

        is_star_query = select == "*"  # Get all properties, special case
        query_geometry = geometry  # Rename arg
        return_columns = []  # Initialise return rows list of column names

        if is_star_query and unique:
            msg = "unique=True cannot be used with select='*'"
            raise ValueError(msg)

        if not unique:
            return_columns.append("[key]")
        if is_str_query and not is_star_query:
            select = cast(str, select)
            select_names = eval(  # skipcq: PYL-W0123,  # noqa: S307
                select,
                SQL_GLOBALS,
                {},
            )
            return_columns += [str(select_names)]
        if is_callable_query or is_star_query or is_pickle_query:
            return_columns.append("properties")
        columns = ", ".join(return_columns)

        cur = self._query(
            columns=columns,
            geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            where=where,
            unique=unique,
            no_constraints_ok=True,
            index_warning=True,
        )

        if is_pickle_query or is_callable_query:
            # Where to apply after database query
            # only done for Callable where.
            post_where = cast(CallablePredicate, where) if is_callable_query else None
            select = cast(CallableSelect, select)
            result = self._handle_pickle_callable_pquery(
                select,
                post_where,
                cur,
                unique=unique,
            )
        else:
            result = self._handle_str_pquery(
                cur,
                unique=unique,
                star_query=is_star_query,
            )

        if unique and squeeze and len(result) == 1:
            result = cast(list[set], result)
            return result[0]
        return result

    def __len__(self: SQLiteStore) -> int:
        """Return number of annotations in the store."""
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM annotations")
        (count,) = cur.fetchone()
        return count

    def __contains__(self: SQLiteStore, key: object) -> bool:
        """Test whether the object contains the specified object or not."""
        cur = self.con.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM annotations WHERE [key] = ?)", (key,))
        return cur.fetchone()[0] == 1

    def __getitem__(self: SQLiteStore, key: str) -> Annotation:
        """Get an item from the store."""
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT geometry, properties, cx, cy
              FROM annotations
             WHERE [key] = :key
            """,
            {"key": key},
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(key)
        serialised_geometry, serialised_properties, cx, cy = row
        properties = json.loads(serialised_properties or "{}")
        return Annotation(
            properties=properties,
            wkb=self._unpack_wkb(serialised_geometry, cx, cy),
        )

    def keys(self: SQLiteStore) -> KeysView[str]:
        """Return an iterable (usually generator) of all keys in the store.

        Returns:
            Iterable[str]:
                An iterable of keys.

        """
        keys_dict: dict[str, None] = {}
        for key, _ in self.items():  # noqa: PERF102
            keys_dict[key] = None
        return keys_dict.keys()
        # yield from self

    def __iter__(self: SQLiteStore) -> Iterator[str]:
        """Return an iterator for the given object."""
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key]
              FROM annotations
            """,
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            yield row[0]  # The key

    def values(self: SQLiteStore) -> ValuesView[Annotation]:
        """Return an iterable of all annotation in the store.

        Returns:
            Iterable[Annotation]:
                An iterable of annotations.

        """
        values_dict: dict[int, Annotation] = {}

        for i, (_, annotation) in enumerate(self.items()):
            values_dict[i] = annotation
        return values_dict.values()

    def items(self: SQLiteStore) -> ItemsView[str, Annotation]:
        """Return iterable (generator) over key and annotations."""
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key], cx, cy, geometry, properties
              FROM annotations
            """,
        )
        items_dict: dict[str, Annotation] = {}
        while True:
            row = cur.fetchone()
            if row is None:
                break
            key, cx, cy, serialised_geometry, serialised_properties = row
            if serialised_geometry is not None:
                geometry = self._unpack_geometry(serialised_geometry, cx, cy)
            else:
                geometry = Point(cx, cy)
            properties = json.loads(serialised_properties)
            items_dict[key] = Annotation(geometry, properties)
        return items_dict.items()

    def patch_many(
        self: SQLiteStore,
        keys: Iterable[str],
        geometries: Iterable[Geometry] | None = None,
        properties_iter: Iterable[Properties] | None = None,
    ) -> None:
        """Bulk patch of annotations.

        This may be more efficient than calling `patch` repeatedly
        in a loop.

        Args:
            geometries (iter(Geometry)):
                An iterable of geometries to update.
            properties_iter (iter(dict)):
                An iterable of properties to update.
            keys (iter(str)):
                An iterable of keys for each annotation to be updated.

        """
        # Validate inputs
        if not any([geometries, properties_iter]):
            msg = "At least one of geometries or properties_iter must be given"
            raise ValueError(
                msg,
            )
        keys = list(keys)
        geometries = list(geometries) if geometries else None
        properties_iter = list(properties_iter) if properties_iter else None
        self._validate_equal_lengths(keys, geometries, properties_iter)
        properties_iter = properties_iter or ({} for _ in keys)  # pragma: no branch
        geometries = geometries or (None for _ in keys)  # pragma: no branch
        # Update the database
        cur = self.con.cursor()
        # Begin a transaction
        if self.auto_commit:
            cur.execute("BEGIN")
        for key, geometry, properties in zip(keys, geometries, properties_iter):
            # Annotation is not in DB:
            if key not in self:
                self._append(str(key), Annotation(geometry, properties), cur)
                continue
            # Annotation is in DB:
            if geometry:
                self._patch_geometry(str(key), geometry, cur)
            if properties:
                cur.execute(
                    """
                    UPDATE annotations
                       SET properties = json_patch(properties, :properties)
                     WHERE [key] = :key
                    """,
                    {
                        "key": key,
                        "properties": json.dumps(properties, separators=(",", ":")),
                    },
                )
        if self.auto_commit:
            self.con.commit()

    def _patch_geometry(
        self: SQLiteStore,
        key: str,
        geometry: Geometry,
        cur: sqlite3.Cursor,
    ) -> None:
        """Patch a geometry in the database.

        Update the geometry of the annotation with the given key but
        leave the properties untouched.

        Args:
            key (str): The key of the annotation to patch.
            geometry (Geometry): The new geometry.
            cur (sqlite3.Cursor): The cursor to use.

        """
        bounds = dict(zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds))
        xy = dict(zip("xy", np.array(geometry.centroid.coords[0])))
        query_parameters = dict(
            **bounds,
            **xy,
            key=key,
            geometry=self.serialise_geometry(geometry),
        )
        cur.execute(
            """
            UPDATE rtree
               SET min_x = :min_x, min_y = :min_y,
                   max_x = :max_x, max_y = :max_y
             WHERE EXISTS
                   (SELECT 1
                      FROM annotations
                     WHERE rtree.id = annotations.id
                       AND annotations.key == :key);
            """,
            query_parameters,
        )
        cur.execute(
            """
            UPDATE annotations
               SET cx = :x, cy = :y, geometry = :geometry
             WHERE [key] = :key
            """,
            query_parameters,
        )

    def remove_many(self: SQLiteStore, keys: Iterable[str]) -> None:
        """Bulk removal of annotations by keys.

        Args:
            keys (iter(str)):
                An iterable of keys for the annotation to be removed.

        """
        cur = self.con.cursor()
        if self.auto_commit:
            cur.execute("BEGIN")
        for key in keys:
            cur.execute(
                """
                DELETE
                  FROM rtree
                 WHERE EXISTS
                       (SELECT 1
                          FROM annotations
                         WHERE rtree.id = annotations.id
                           AND annotations.key == ?);
                """,
                (key,),
            )
            cur.execute(
                "DELETE FROM annotations WHERE [key] = ?",
                (key,),
            )
        if self.auto_commit:
            self.con.commit()

    def __setitem__(self: SQLiteStore, key: str, annotation: Annotation) -> None:
        """Implements a method to assign a value to an item."""
        if key in self:
            self.patch(key, annotation.geometry, annotation.properties)
            return
        self.append(annotation, key)

    def _get_table_columns(self: SQLiteStore) -> list[str]:
        """Get a list of columns in the annotations table."""
        cur = self.con.execute("PRAGMA table_info(annotations)")
        return [row[1] for row in cur.fetchall()]

    def add_area_column(self: SQLiteStore, *, mk_index: bool = True) -> None:
        """Add a column to store the area of the geometry."""
        cur = self.con.cursor()
        cur.execute(
            """
            ALTER TABLE annotations
            ADD COLUMN area INTEGER NOT NULL DEFAULT 0
            """,
        )
        cur.execute(
            """
            UPDATE annotations
            SET area = get_area(geometry, cx, cy)
            """,
        )
        if mk_index:
            self.create_index("area", '"area"')
        self.con.commit()
        self.table_columns.append("area")

    def remove_area_column(self: SQLiteStore) -> None:
        """Remove the area column from the store."""
        if "area" in self.indexes():
            self.drop_index("area")
        cur = self.con.cursor()
        cur.execute(
            """
            ALTER TABLE annotations
            DROP COLUMN area
            """,
        )
        self.con.commit()
        self.table_columns.remove("area")

    def to_dataframe(self: SQLiteStore) -> pd.DataFrame:
        """Converts AnnotationStore to :class:`pandas.DataFrame`."""
        store_to_df = pd.DataFrame()
        df_rows: list[dict] = []
        for key, annotation in self.items():
            row = {
                "key": key,
                "geometry": annotation.geometry,
                "properties": annotation.properties,
            }
            df_rows.append(row)
        store_to_df = pd.concat([store_to_df, pd.json_normalize(df_rows)])
        return store_to_df.set_index("key")

    def features(self: SQLiteStore) -> Generator[dict[str, object], None, None]:
        """Return annotations as a list of geoJSON features.

        Returns:
            list:
                List of features as dictionaries.

        """
        return (
            {
                "type": "Feature",
                "geometry": geometry2feature(annotation.geometry),
                "properties": annotation.properties,
            }
            for annotation in self.values()
        )

    def commit(self: SQLiteStore) -> None:
        """Commit any in-memory changes to disk."""
        self.con.commit()

    def dump(self: SQLiteStore, fp: Path | str | IO) -> None:
        """Serialise a copy of the whole store to a file-like object.

        Args:
            fp(Path or str or IO):
                A file path or file handle object for output to disk.

        """
        if hasattr(fp, "write"):
            fp = cast(IO, fp)
            fp = fp.name
        target = sqlite3.connect(fp)
        self.con.backup(target)

    def dumps(self: SQLiteStore) -> str:
        """Serialise and return a copy of store as a string or bytes.

        Returns:
            str or bytes:
                The serialised store.

        """
        return "\n".join(self.con.iterdump())

    def clear(self: SQLiteStore) -> None:
        """Remove all annotations from the store."""
        cur = self.con.cursor()
        cur.execute("DELETE FROM rtree")
        cur.execute("DELETE FROM annotations")
        if self.auto_commit:
            self.con.commit()

    def create_index(
        self: SQLiteStore,
        name: str,
        where: str | bytes,
        *,
        analyze: bool = True,
    ) -> None:
        """Create an SQLite expression index based on the provided predicate.

        Note that an expression index will only be used if the query expression
        (in the WHERE clause) exactly matches the expression used when creating
        the index (excluding minor inconsequential changes such as
        whitespace).

        An SQLite expression indexes require SQLite version 3.9.0 or higher.

        Args:
            name (str):
                Name of the index to create.
            where:
                The predicate used to create the index.
            analyze (bool):
                Whether to run the "ANALYZE" command after creating the
                index.

        """
        _, minor, _ = sqlite3.sqlite_version_info
        if minor < 9:  # noqa: PLR2004
            msg = "Requires sqlite version 3.9.0 or higher."
            raise OSError(msg)
        cur = self.con.cursor()
        if not isinstance(where, str):
            msg = f"Invalid type for `where` ({type(where)})."
            raise TypeError(msg)
        sql_predicate = eval(  # skipcq: PYL-W0123,  # noqa: S307
            where,
            SQL_GLOBALS,
        )
        cur.execute(f"CREATE INDEX {name} ON annotations({sql_predicate})")
        if analyze:
            cur.execute(f"ANALYZE {name}")

    def indexes(self: SQLiteStore) -> list[str]:
        """Return a list of the names of all indexes in the store.

        Returns:
            List[str]:
                The list of index names.

        """
        cur = self.con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE TYPE = 'index'")
        return [row[0] for row in cur.fetchall()]

    def drop_index(self: SQLiteStore, name: str) -> None:
        """Drop an index from the store.

        Args:
            name (str):
                The name of the index to drop.

        """
        cur = self.con.cursor()
        cur.execute(f"DROP INDEX {name}")

    def optimize(self: SQLiteStore, limit: int = 1000, *, vacuum: bool = True) -> None:
        """Optimize the database with VACUUM and ANALYZE.

        Args:
            vacuum (bool):
                Whether to run VACUUM.
            limit (int):
                The approximate maximum number of rows to examine when
                running ANALYZE. If zero or negative, not limit will be
                used. For more information see
                https://www.sqlite.org/pragma.html#pragma_analysis_limit.

        """
        if vacuum:
            self.con.execute("VACUUM")
        # Cannot use parameterized statements with PRAGMA!
        self.con.execute(f"PRAGMA analysis_limit = {int(limit)}")
        self.con.execute("PRAGMA optimize")


class DictionaryStore(AnnotationStore):
    """Pure python dictionary backed annotation store."""

    def __init__(
        self: DictionaryStore,
        connection: Path | str | IO = ":memory:",
    ) -> None:
        """Initialize :class:`DictionaryStore`."""
        super().__init__()
        self._rows: dict = {}
        self.connection = connection
        self.path = self._connection_to_path(connection)
        if self.connection not in [None, ":memory:"] and self.path.exists():
            cases = self._load_cases(
                fp=self.connection,
                string_fn=lambda fp: fp.splitlines(),
                file_fn=lambda fp: fp.readlines(),
            )
            cases = cast(list, cases)
            for line in cases:
                dictionary = json.loads(line)
                key = dictionary.get("key", uuid.uuid4().hex)
                geometry = feature2geometry(dictionary["geometry"])
                properties = dictionary["properties"]
                self.append(Annotation(geometry, properties), key=key)

    def append(
        self: DictionaryStore,
        annotation: Annotation,
        key: str | None = None,
    ) -> str:
        """Insert a new annotation, returning the key.

        Args:
            annotation (Annotation):
                The shapely annotation to insert.
            key (str):
                Optional. The unique key used to identify the annotation in the
                store. If not given a new UUID4 will be generated and returned
                instead.

        Returns:
            str:
                The unique key of the newly inserted annotation.

        """
        if not isinstance(annotation.geometry, (Polygon, Point, LineString)):
            msg = "Invalid geometry type. Must be one of Point, LineString, Polygon."
            raise TypeError(msg)
        key = key or str(uuid.uuid4())
        self._rows[key] = {"annotation": annotation}
        return key

    def patch(
        self: DictionaryStore,
        key: str,
        geometry: Geometry | None = None,
        properties: dict[str, Properties] | None = None,
    ) -> None:
        """Patch an annotation at given key.

        Partial update of an annotation. Providing only a geometry will update
        the geometry and leave properties unchanged. Providing a properties
        dictionary applies a patch operation to the properties. Only updating
        the properties which are given and leaving the rest unchanged. To
        completely replace an annotation use `__setitem__`.

        Args:
            key(str):
                The key of the annotation to update.
            geometry(Geometry):
                The new geometry. If None, the geometry is not updated.
            properties(dict):
                A dictionary of properties to patch and their new values.
                If None, the existing properties are not altered.

        """
        if key not in self:
            self.append(Annotation(geometry, properties), key)
            return
        existing = self[key]
        geometry = geometry or existing.geometry
        properties = properties or {}
        new_properties = copy.deepcopy(existing.properties)
        new_properties.update(properties)
        self[key] = Annotation(geometry, new_properties)

    def remove(self: DictionaryStore, key: str) -> None:
        """Remove annotation from the store with its unique key.

        Args:
            key (str):
                The key of the annotation to be removed.

        """
        del self._rows[key]

    def __getitem__(self: DictionaryStore, key: str) -> Annotation:
        """Get an item from the store."""
        return self._rows[key]["annotation"]

    def __setitem__(self: DictionaryStore, key: str, annotation: Annotation) -> None:
        """Implements a method to assign a value to an item."""
        if key in self._rows:
            self._rows[key]["annotation"] = annotation
        self._rows[key] = {"annotation": annotation}

    def __contains__(self: DictionaryStore, key: object) -> bool:
        """Test whether the object contains the specified object or not."""
        return key in self._rows

    def items(self: DictionaryStore) -> ItemsView[str, Annotation]:
        """Return iterable (generator) over key and annotations."""
        items_dict: dict[str, Annotation] = {}
        for key, row in self._rows.items():
            items_dict[key] = row["annotation"]
        return items_dict.items()

    def __len__(self: DictionaryStore) -> int:
        """Return the length of the instance attributes."""
        return len(self._rows)

    # flake8: noqa: A003
    @classmethod
    def open(cls: type[AnnotationStore], fp: Path | str | IO) -> AnnotationStore:
        """Opens :class:`DictionaryStore` from file pointer or path."""
        return cls.from_ndjson(fp)

    def commit(self: DictionaryStore) -> None:
        """Commit any in-memory changes to disk."""
        if str(self.connection) == ":memory:":
            logger.warning("In-memory store. Nothing to commit.", stacklevel=2)
            return
        if not self.path.exists():
            self.path.touch()
        self.dump(self.connection)

    def dump(self: DictionaryStore, fp: Path | str | IO) -> None:
        """Serialise a copy of the whole store to a file-like object.

        Args:
            fp(Path or str or IO):
                A file path or file handle object for output to disk.

        """
        self.to_ndjson(fp)

    def dumps(self: DictionaryStore) -> str:
        """Serialise and return a copy of store as a string or bytes.

        Returns:
            str or bytes:
                The serialised store.

        """
        return self.to_ndjson()

    def close(self: DictionaryStore) -> None:
        """Closes :class:`DictionaryStore` from file pointer or path."""
        duplicate_filter = DuplicateFilter()
        logger.addFilter(duplicate_filter)
        # Try to commit any changes if the file is still open.
        with contextlib.suppress(ValueError):
            self.commit()
        logger.removeFilter(duplicate_filter)
