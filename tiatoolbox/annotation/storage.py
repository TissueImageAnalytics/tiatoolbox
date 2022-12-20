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
import contextlib
import copy
import io
import json
import os
import pickle
import sqlite3
import sys
import tempfile
import uuid
import warnings
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Number
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from shapely import speedups, wkb, wkt
from shapely.affinity import scale, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import mapping as geometry2feature
from shapely.geometry import shape as feature2geometry

import tiatoolbox
from tiatoolbox import logger
from tiatoolbox.annotation.dsl import (
    PY_GLOBALS,
    SQL_GLOBALS,
    json_contains,
    json_list_sum,
    py_regexp,
)

sqlite3.enable_callback_tracebacks(True)

if speedups.available:  # pragma: no branch
    speedups.enable()

Geometry = Union[Point, Polygon, LineString]
Properties = Dict[str, Union[Dict, List, Number, str]]
BBox = Tuple[Number, Number, Number, Number]
QueryGeometry = Union[BBox, Geometry]
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = Union[str, bytes, CallablePredicate]
Select = Union[str, bytes, CallableSelect]

ASCII_FILE_SEP = "\x1c"
ASCII_GROUP_SEP = "\x1d"
ASCII_RECORD_SEP = "\x1e"
ASCII_UNIT_SEP = "\x1f"
ASCII_NULL = "\0"
ISO_8601_DATE_FORMAT = r"%Y-%m-%dT%H:%M:%S.%f%z"

# Only Python 3.10+ supports using slots for dataclasses
# https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass
# therefore we use the following workaround to only use them when available.
# Using slots gives a performance boost at object creation time.
_DATACLASS_KWARGS = {"frozen": True}
if sys.version_info >= (3, 10):  # pragma: no cover
    _DATACLASS_KWARGS["slots"] = True


@dataclass(**_DATACLASS_KWARGS)
class Annotation:
    """An annotation: a geometry and associated properties.

    Attributes:
        geometry (Geometry):
            The geometry of the annotation.
        properties (dict):
            The properties of the annotation.

    """

    geometry: Geometry
    properties: Properties = field(default_factory=dict)

    def to_feature(self) -> Dict:
        """
        Return a feature representation of this annotation.

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

    def to_geojson(self) -> str:
        """
        Return a GeoJSON string representation of this annotation.

        Returns:
            str:
                A GeoJSON representation of this annotation.

        """
        return json.dumps(self.to_feature())


class AnnotationStore(ABC, MutableMapping):
    """Annotation store abstract base class."""

    @staticmethod
    def _is_right_angle(a, b, c) -> bool:
        """Returns True if three points make a right angle.

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
    def _is_rectangle(a, b, c, d, *args) -> bool:
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
    def _connection_to_path(connection: Union[str, Path, IO]) -> Path:
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
                tempfile._TemporaryFileWrapper,  # skipcq: PYL-W0212
            ),
        ):
            raise TypeError(
                "Connection must be a string, Path, or an IO object, "
                f"not {type(connection)}"
            )
        if isinstance(
            connection,
            (
                io.IOBase,
                io.TextIOBase,
                tempfile._TemporaryFileWrapper,  # skipcq: PYL-W0212
            ),
        ):
            connection = connection.name
        return Path(connection)

    @staticmethod
    def _validate_equal_lengths(*args):
        """Validate that all given args are either None or have the same length."""
        lengths = [len(v) for v in args if v is not None]
        if lengths and any(length != lengths[0] for length in lengths):
            raise ValueError("All arguments must be None or of equal length.")

    @staticmethod
    def _geometry_predicate(name: str, a: Geometry, b: Geometry) -> Callable:
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
    _geometry_predicate_names = [
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
        "bbox_intersects",  # Special non-shapely case, bounding-boxes intersect
    ]

    @classmethod  # noqa: A003
    @abstractmethod
    def open(cls, fp: Union[Path, str, IO]) -> "AnnotationStore":  # noqa: A003
        """Load a store object from a path or file-like object.

        Args:
            fp(Path or str or IO): The file path or file handle.

        Returns:
            AnnotationStoreABC:
                An instance of an annotation store.

        """

    @staticmethod
    def serialise_geometry(geometry: Geometry) -> Union[str, bytes]:
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
    def deserialize_geometry(data: Union[str, bytes]) -> Geometry:
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
        return wkt.loads(data) if isinstance(data, str) else wkb.loads(data)

    @abstractmethod
    def commit(self) -> None:
        """Commit any in-memory changes to disk."""

    @abstractmethod
    def dump(self, fp: Union[Path, str, IO]) -> None:
        """Serialise a copy of the whole store to a file-like object.

        Args:
            fp(Path or str or IO):
                A file path or file handle object for output to disk.

        """

    @abstractmethod
    def dumps(self) -> Union[str, bytes]:
        """Serialise and return a copy of store as a string or bytes.

        Returns:
            str or bytes:
                The serialised store.

        """

    def append(
        self,
        annotation: Annotation,
        key: Optional[str] = None,
    ) -> int:
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
        keys = key if key is None else [key]
        return self.append_many([annotation], keys)[0]

    def append_many(
        self,
        annotations: Iterable[Annotation],
        keys: Optional[Iterable[str]] = None,
    ) -> List[str]:
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
        result = []
        if keys:
            result.extend(
                self.append(annotation, key)
                for key, annotation in zip(keys, annotations)
            )
            return result
        result.extend(self.append(annotation) for annotation in annotations)
        return result

    def patch(
        self,
        key: str,
        geometry: Optional[Geometry] = None,
        properties: Optional[Dict[str, Any]] = None,
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
        properties = properties if properties is None else [properties]
        self.patch_many([key], geometry, properties)

    def patch_many(
        self,
        keys: Iterable[int],
        geometries: Optional[Iterable[Geometry]] = None,
        properties_iter: Optional[Iterable[Properties]] = None,
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
            raise ValueError(
                "At least one of geometries or properties_iter must be given"
            )
        keys = list(keys)
        geometries = list(geometries) if geometries else None
        properties_iter = list(properties_iter) if properties_iter else None
        self._validate_equal_lengths(keys, geometries, properties_iter)
        properties_iter = properties_iter or ({} for _ in keys)  # pragma: no branch
        geometries = geometries or (None for _ in keys)  # pragma: no branch
        # Update the store
        for key, geometry, properties in zip(keys, geometries, properties_iter):
            properties = copy.deepcopy(properties)
            self.patch(key, geometry, properties)

    def remove(self, key: str) -> None:
        """Remove annotation from the store with its unique key.

        Args:
            key (str):
                The key of the annotation to be removed.

        """
        self.remove_many([key])

    def remove_many(self, keys: Iterable[str]) -> None:
        """Bulk removal of annotations by keys.

        Args:
            keys (iter(str)):
                An iterable of keys for the annotation to be removed.

        """
        for key in keys:
            self.remove(key)

    def setdefault(self, key: str, default: Annotation = None) -> Annotation:
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
            raise TypeError("default value must be an Annotation instance.")
        return super().setdefault(key, default)

    def __delitem__(self, key: str) -> None:
        """Delete an annotation by key.

        An alias of `remove`.

        Args:
            key (str):
                The key of the annotation to be removed.

        """
        self.remove(key)

    def keys(self) -> Iterable[str]:
        """Return an iterable (usually generator) of all keys in the store.

        Returns:
            Iterable[str]:
                An iterable of keys.

        """
        for key, _ in self.items():
            yield key

    def values(self) -> Iterable[Annotation]:
        """Return an iterable of all annotation in the store.

        Returns:
            Iterable[Annotation]:
                An iterable of annotations.

        """
        for _, annotation in self.items():
            yield annotation

    def __iter__(self) -> Iterable[str]:
        """Return an iterable of keys in the store.

        An alias of `keys`.

        Returns:
            Iterable[str]:
                An iterable of keys.

        """
        yield from self.keys()

    @staticmethod
    def _eval_where(
        predicate: Optional[Predicate],
        properties: Dict[str, Any],
    ) -> bool:
        """Evaluate properties predicate against properties.

        Args:
            predicate (str or bytes or Callable):
                The predicate to evaluate on properties. The predicate may be a
                string, pickled bytes, or a callable (e.g. a function).
            properties (dict):
                A dictionary of JSON serializable
                properties on which to evaluate the predicate.

        Returns:
            bool:
                Returns True if the predicate holds.

        """
        if predicate is None:
            return True
        if isinstance(predicate, str):
            return bool(
                eval(predicate, PY_GLOBALS, {"props": properties})  # skipcq: PYL-W0123
            )
        if isinstance(predicate, bytes):
            predicate = pickle.loads(predicate)  # skipcq: BAN-B301
        return bool(predicate(properties))

    def query(
        self,
        geometry: Optional[QueryGeometry] = None,
        where: Optional[Predicate] = None,
        geometry_predicate: str = "intersects",
    ) -> Dict[str, Annotation]:
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
                may be a string, callable, or pickled function as bytes.
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
                list:
                    A list of Annotation objects.

        """
        if all(x is None for x in (geometry, where)):
            raise ValueError("At least one of geometry or where must be set.")
        if geometry_predicate not in self._geometry_predicate_names:
            raise ValueError(
                "Invalid geometry predicate."
                f"Allowed values are: {', '.join(self._geometry_predicate_names)}."
            )
        query_geometry = geometry
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)

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
            return (  # Geometry is None or the geometry predicate matches
                query_geometry is None
                or self._geometry_predicate(
                    geometry_predicate, query_geometry, annotation.geometry
                )
            ) and self._eval_where(where, annotation.properties)

        return {
            key: annotation
            for key, annotation in self.items()
            if filter_function(annotation)
        }

    def iquery(
        self,
        geometry: QueryGeometry,
        where: Optional[Predicate] = None,
        geometry_predicate: str = "intersects",
    ) -> List[int]:
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
                may be a string, callable, or pickled function as bytes.
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
            raise ValueError(
                "Invalid geometry predicate."
                f"Allowed values are: {', '.join(self._geometry_predicate_names)}."
            )
        query_geometry = geometry
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            key
            for key, annotation in self.items()
            if (
                self._geometry_predicate(
                    geometry_predicate, query_geometry, annotation.geometry
                )
                and self._eval_where(where, annotation.properties)
            )
        ]

    def bquery(
        self,
        geometry: Optional[QueryGeometry] = None,
        where: Predicate = None,
    ) -> Dict[str, Tuple[float, float, float, float]]:
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
                may be a string, callable, or pickled function as bytes.
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
                >>> from tiatoolbox.annotation.storage import AnnotationStore
                >>> from shapely.geometry import Polygon
                >>> store = AnnotationStore()
                >>> store.add(
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
                Polygon.from_bounds(*annotation.geometry.bounds).intersects(
                    Polygon.from_bounds(*query_geometry.bounds)
                )
                and self._eval_where(where, annotation.properties)
            )
        }

    def pquery(
        self,
        select: Select,
        geometry: Optional[QueryGeometry] = None,
        where: Optional[Predicate] = None,
        unique: bool = True,
        squeeze: bool = True,
    ) -> Union[Dict[str, Any], Set[Any]]:
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
                may be a string, callable, or pickled function as bytes.
                Callables are called to filter each result returned the
                from annotation store backend in python before being
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

            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> from shapely.geometry import Point
            >>> store = AnnotationStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.add(annotation, "foo")
            >>> store.pquery("*", unique=False)
            ... {'foo': {'class': 42}}

            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> from shapely.geometry import Point
            >>> store = AnnotationStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.add(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42}
            >>> annotation =  Annotation(Point(1, 1), {"class": 123})
            >>> store.add(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42, 123}

        """  # noqa
        if where is not None and type(select) is not type(where):
            raise TypeError("select and where must be of the same type")
        if not isinstance(select, (str, bytes)) and not callable(select):
            raise TypeError(
                f"select must be str, bytes, or callable, not {type(select)}"
            )
        # Are we scanning through all annotations?
        is_scan = not any((geometry, where))
        items = self.items() if is_scan else self.query(geometry, where).items()

        def select_values(
            select: Select, annotation: Annotation
        ) -> Union[Properties, Any, Tuple[Any, ...]]:
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
                Union[Properties, Any, Tuple[Any, ...]]:
                    The value(s) to return from the annotation. This
                    will be a dictionary if unique is False. Otherwise,
                    it will be a list of sets. If squeeze and unique are
                    True in addtion to there only being one set in the
                    results list, the result will be a single set.

            """  # noqa Q440, Q441
            if select == "*" and unique:
                raise ValueError("unique=True cannot be used with select='*'")

            if select == "*":  # Special case for all properties
                return annotation.properties

            if isinstance(select, str):
                py_locals = {"props": annotation.properties}
                return eval(select, PY_GLOBALS, py_locals)  # skipcq: PYL-W0123
            if isinstance(select, bytes):
                return pickle.loads(select)(annotation.properties)  # skipcq: BAN-B301

            return select(annotation.properties)

        return self._handle_pquery_results(
            select, unique, squeeze, items, select_values
        )

    @staticmethod
    def _handle_pquery_results(
        select: Select,
        unique: bool,
        squeeze: bool,
        items: Generator[Tuple[str, Properties], None, None],
        get_values: Callable[
            [Select, Annotation], Union[Properties, Any, Tuple[Any, ...]]
        ],
    ):
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

        """  # noqa Q440, Q441
        result = defaultdict(set) if unique else {}
        for key, annotation in items:
            values = get_values(select, annotation)
            if unique:
                # Wrap scalar values in a tuple
                if not isinstance(values, tuple):
                    values = (values,)
                # Add each value to the result set
                for i, value in enumerate(values):
                    result[i].add(value)
            else:
                result[key] = values
        if unique:
            result = list(result.values())
        if unique and squeeze and len(result) == 1:
            result = result[0]
        return result  # noqa CCR001

    def features(self) -> Generator[Dict[str, Any], None, None]:
        """Return annotations as a list of geoJSON features.

        Returns:
            list:
                List of features as dictionaries.

        """
        for a in self.values():
            yield a.to_feature()

    def to_geodict(self) -> Dict[str, Any]:
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
        fp: Union[IO, str, Path, None],
        file_fn: Callable[[IO], None],
        none_fn: Callable[[], Union[str, bytes]],
    ) -> Optional[Union[str, bytes]]:
        """Helper function to handle cases for dumping.

        Args:
            fp:
                The file path or handle to dump to.
            file_fn(Callable):
                The function to call when fp is a file handle.
            none_fn(Callable):
                The function to call when fp is None.

        Returns:
            Any:
                The result of dump. Depends on the provided functions.

        """
        if fp is not None:
            # It is a file-like object, write to it
            if hasattr(fp, "write"):
                return file_fn(fp)
            # Turn a path into a file handle, then write to it
            with open(fp, "w", encoding="utf-8") as file_handle:
                return file_fn(file_handle)
        # Return as str or bytes if no handle/path is given
        return none_fn()

    @staticmethod
    def _load_cases(
        fp: Union[IO, str, Path],
        string_fn: Callable[[Union[str, bytes]], Any],
        file_fn: Callable[[IO], Any],
    ) -> Any:
        """Loads cases for an input file handle or path."""
        with contextlib.suppress(OSError):
            if isinstance(fp, (Path, str)) and Path(fp).exists():
                with open(fp) as file_handle:
                    return file_fn(file_handle)
        if isinstance(fp, (str, bytes)):
            return string_fn(fp)
        if hasattr(fp, "read"):
            return file_fn(fp)
        raise IOError("Invalid file handle or path.")

    @classmethod
    def from_geojson(
        cls,
        fp: Union[IO, str],
        scale_factor: Tuple[float, float] = (1, 1),
        origin: Tuple[float, float] = (0, 0),
    ) -> "AnnotationStore":
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

        Returns:
            AnnotationStore:
                A new annotation store with the annotations loaded from the file.

        """
        store = cls()
        store.add_from_geojson(fp, scale_factor, origin=origin)
        return store

    def add_from_geojson(
        self,
        fp: Union[IO, str],
        scale_factor: Tuple[float, float] = (1, 1),
        origin: Tuple[float, float] = (0, 0),
    ) -> None:
        """Add annotations from a .geojson file to an existing store. Make
        the best effort to create valid shapely geometries from provided contours.

        Args:
            fp (Union[IO, str, Path]):
                The file path or handle to load from.
            scale_factor (float):
                The scale factor to use when loading the annotations. All coordinates
                will be multiplied by this factor to allow import of annotations saved
                at non-baseline resolution.
            origin [float, float]:
                The x and y coordinates to use as the origin for the annotations.

        """

        def transform_geometry(geom):
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

        annotations = [
            Annotation(
                transform_geometry(
                    feature2geometry(feature["geometry"]),
                ),
                feature["properties"],
            )
            for feature in geojson["features"]
        ]

        print(f"added {len(annotations)} annotations")
        self.append_many(annotations)

    def to_geojson(self, fp: Optional[Union[IO, str, Path]] = None) -> Optional[str]:
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

        def write_geojson_to_file_handle(file_handle: IO):
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

        return self._dump_cases(
            fp=fp,
            file_fn=write_geojson_to_file_handle,
            none_fn=lambda: json.dumps(self.to_geodict()),
        )

    def to_ndjson(self, fp: Optional[IO] = None) -> Optional[str]:
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
        return self._dump_cases(
            fp=fp,
            file_fn=lambda fp: fp.writelines(string_lines_generator),
            none_fn=lambda: "".join(string_lines_generator),
        )

    @classmethod
    def from_ndjson(cls, fp: Union[IO, str]) -> "AnnotationStore":
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
        for line in cls._load_cases(
            fp=fp,
            string_fn=lambda fp: fp.splitlines(),
            file_fn=lambda fp: fp.readlines(),
        ):
            dictionary = json.loads(line)
            key = dictionary.get("key", uuid.uuid4().hex)
            geometry = feature2geometry(dictionary["geometry"])
            properties = dictionary["properties"]
            store.append(Annotation(geometry, properties), key)
        return store

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "AnnotationStore":
        """Converts to AnnotationStore from :class:`pandas.DataFrame`."""
        store = cls()
        for key, row in df.iterrows():
            geometry = row["geometry"]
            properties = dict(row.filter(regex="^(?!geometry|key).*$"))
            store.append(Annotation(geometry, properties), str(key))
        return store

    def to_dataframe(self) -> pd.DataFrame:
        """Converts AnnotationStore to :class:`pandas.DataFrame`."""
        features = (
            {
                "geometry": annotation.geometry,
                "properties": annotation.properties,
                "key": key,
            }
            for key, annotation in self.items()
        )
        return pd.json_normalize(features).set_index("key")

    def transform(self, transform: Callable[[Geometry], Geometry]) -> None:
        """Transform all annotations in the store using provided function.

        Useful for transforming coordinates from slide space into
        patch/tile/core space, or to a different resolution, for example.

        Args:
            transform (callable[Geometry, Geometry]):
                A function that takes a geometry and returns a new
                transformed geometry.

        """
        transformed_geoms = {
            key: transform(annotation.geometry) for key, annotation in self.items()
        }
        self.patch_many(transformed_geoms.keys(), transformed_geoms.values())

    def __del__(self) -> None:
        self.close()

    def clear(self) -> None:
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
        connection (Union[str, Path, IO]):
            A reference to where the data is stored. It maybe a string (
            e.g. ":memory:" or "./data.db"), a pathlib Path, or a file
            handle.
        path (Path):
            The path to the annotation store data. This will be
            ":memory:" if the annotation store is in-memory. This is
            derived from `connection` and normalised to be a pathlib
            Path object.
        con (sqlite3.Connection):
            The sqlite3 database connection.

    """

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT UNIQUE, value TEXT)"
        )
        self.con.commit()

    def __contains__(self, key: str) -> bool:
        cursor = self.con.execute("SELECT 1 FROM metadata WHERE [key] = ?", (key,))
        return cursor.fetchone() is not None

    def __setitem__(self, key: str, value: Union[dict, list, int, float, str]) -> None:
        """Set a metadata value."""
        value = json.dumps(value)
        self.con.execute(
            "REPLACE INTO metadata (key, value) VALUES (?,?)", (key, value)
        )
        self.con.commit()

    def __getitem__(self, key: str) -> Union[dict, list, int, float, str]:
        """Get a metadata value."""
        cursor = self.con.execute("SELECT value FROM metadata WHERE [key] = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError(key)
        return json.loads(result[0])

    def __delitem__(self, key: str) -> None:
        """Delete a metadata value."""
        if key not in self:
            raise KeyError(key)
        self.con.execute("DELETE FROM metadata WHERE [key] = ?", (key,))

    def __iter__(self) -> Iterator[str]:
        """Iterate over all keys."""
        cursor = self.con.execute("SELECT [key] FROM metadata")
        for row in cursor:
            yield row[0]

    def __len__(self) -> int:
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

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str]) -> "SQLiteStore":  # noqa: A003
        """Opens :class:`SQLiteStore` from file pointer or path."""
        return SQLiteStore(fp)

    def __init__(
        self,
        connection: Union[Path, str, IO] = ":memory:",
        compression: str = "zlib",
        compression_level: int = 9,
        auto_commit: bool = True,
    ) -> None:
        super().__init__()
        # Check that JSON and RTree support is enabled
        compile_options = self.compile_options()
        if sqlite3.sqlite_version_info >= (3, 38, 0):
            if not all(
                ["OMIT_JSON" not in compile_options, "ENABLE_RTREE" in compile_options]
            ):
                raise Exception(
                    """RTREE sqlite3 compile option is required, and
                    JSON must not be disabled with OMIT_JSON compile option"""
                )
        else:
            if not all(
                ["ENABLE_JSON1" in compile_options, "ENABLE_RTREE" in compile_options]
            ):
                raise Exception("RTREE and JSON1 sqlite3 compile options are required.")

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
            self.path.is_file()
            and self.path.stat().st_size > 0
        )
        self.con = sqlite3.connect(str(self.path), isolation_level="DEFERRED")
        self.con.execute("BEGIN")

        # Set up metadata
        self.metadata = SQLiteMetadata(self.con)
        if not exists:
            self.metadata["version"] = "1.0.1"
            self.metadata["compression"] = compression
            self.metadata["compression_level"] = compression_level

        # store locally as constantly fetching from db in (de)serialization is slow
        self.compression = self.metadata["compression"]
        self.compression_level = self.metadata["compression_level"]

        # Register predicate functions as custom SQLite functions
        def wkb_predicate(name: str, wkb_a: bytes, b: bytes, cx: int, cy: int) -> bool:
            """Wrapper function to allow WKB as inputs to binary predicates."""
            a = wkb.loads(wkb_a)
            b = self._unpack_geometry(b, cx, cy)
            return self._geometry_predicate(name, a, b)

        def pickle_expression(pickle_bytes: bytes, properties: str) -> bool:
            """Function to load and execute pickle bytes with a "properties" dict."""
            fn = pickle.loads(pickle_bytes)  # skipcq: BAN-B301
            properties = json.loads(properties)
            return fn(properties)

        def get_area(wkb_bytes: bytes, cx: int, cy: int) -> float:
            """Function to get the area of a geometry."""
            return self._unpack_geometry(
                wkb_bytes,
                cx,
                cy,
            ).area

        # Register custom functions
        def register_custom_function(
            name: str, nargs: int, fn: Callable, deterministic: bool = False
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
            try:
                self.con.create_function(name, nargs, fn, deterministic=deterministic)
            except TypeError:
                self.con.create_function(name, nargs, fn)

        register_custom_function(
            "geometry_predicate", 5, wkb_predicate, deterministic=True
        )
        register_custom_function(
            "pickle_expression", 2, pickle_expression, deterministic=True
        )
        register_custom_function("REGEXP", 2, py_regexp)
        register_custom_function("REGEXP", 3, py_regexp)
        register_custom_function("LISTSUM", 1, json_list_sum)
        register_custom_function("CONTAINS", 1, json_contains)
        register_custom_function("get_area", 3, get_area)

        if exists:
            self.table_columns = self._get_table_columns()
            return

        # Create tables for geometry and RTree index
        self.con.execute(
            """
            CREATE VIRTUAL TABLE rtree USING rtree_i32(
                id,                      -- Integer primary key
                min_x, max_x,            -- 1st dimension min, max
                min_y, max_y             -- 2nd dimension min, max
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE annotations(
                id INTEGER PRIMARY KEY,  -- Integer primary key
                key TEXT UNIQUE,         -- Unique identifier (UUID)
                objtype TEXT,            -- Object type
                cx INTEGER NOT NULL,     -- X of centroid/representative point
                cy INTEGER NOT NULL,     -- Y of centroid/representative point
                geometry BLOB,           -- Detailed geometry
                properties TEXT,         -- JSON properties
                area INTEGER NOT NULL    -- Area (for ordering)
            )

            """
        )
        if self.auto_commit:
            self.con.commit()
        self.table_columns = self._get_table_columns()

    def serialise_geometry(  # skipcq: PYL-W0221
        self, geometry: Geometry
    ) -> Union[str, bytes]:
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
        raise ValueError("Unsupported compression method.")

    def _unpack_geometry(self, data: Union[str, bytes], cx: int, cy: int) -> Geometry:
        """Return the geometry using WKB data and rtree bounds index.

        For space optimisation, points are stored as centroids and all
        other geometry types are stored as WKB. This function unpacks
        the WKB data and uses the rtree index to find the centroid for
        points where the data is null.

        Args:
            data(bytes or str):
                The WKB/WKT data to be unpacked.
            cx(int):
                The X coordinate of the centroid/representative point.
            cy(int):
                The Y coordinate of the centroid/representative point.

        Returns:
            Geometry:
                The Shapely geometry.

        """
        return Point(cx, cy) if data is None else self.deserialize_geometry(data)

    def deserialize_geometry(  # skipcq: PYL-W0221
        self,
        data: Union[str, bytes],
    ) -> Geometry:
        """Deserialize a geometry from a string or bytes.

        Args:
            data(bytes or str):
                The serialised representation of a Shapely geometry.

        Returns:
            Geometry:
                The deserialized Shapely geometry.

        """
        if self.compression == "zlib":
            data = zlib.decompress(data)
        elif self.compression is not None:
            raise ValueError("Unsupported compression method.")
        if isinstance(data, str):
            return wkt.loads(data)
        return wkb.loads(data)

    @staticmethod
    def compile_options() -> List[str]:
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
            conn.enable_load_extension(True)
            options = conn.execute("pragma compile_options").fetchall()
        return [opt for opt, in options]

    def close(self) -> None:
        """Closes :class:`SQLiteStore` from file pointer or path."""
        if self.auto_commit:
            self.con.commit()
        self.optimize(vacuum=False, limit=1000)
        self.con.close()

    def _make_token(self, annotation: Annotation, key: Optional[str]) -> Dict:
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
            "cx": int(geometry.centroid.x),
            "cy": int(geometry.centroid.y),
            "min_x": geometry.bounds[0],
            "min_y": geometry.bounds[1],
            "max_x": geometry.bounds[2],
            "max_y": geometry.bounds[3],
            "geom_type": geometry.geom_type,
            "properties": json.dumps(annotation.properties, separators=(",", ":")),
            "area": int(geometry.area),
        }

    def append_many(
        self,
        annotations: Iterable[Annotation],
        keys: Optional[Iterable[str]] = None,
    ) -> List[str]:
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

    def _append(self, key: str, annotation: Annotation, cur: sqlite3.Cursor) -> None:
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
        query_geometry, query_parameters, geometry_predicate, columns, where
    ):
        """Initialises the query string and parameters."""
        query_string = (
            "SELECT "  # skipcq: BAN-B608
            + columns  # skipcq: BAN-B608
            + """
                 FROM annotations, rtree
                WHERE annotations.id == rtree.id
                """
        )

        # There is query geometry, add a simple rtree bounds check to
        # rapidly narrow candidates down.
        if query_geometry is not None:
            # Add rtree index checks to the query
            query_string += """
                    AND max_x >= :min_x
                    AND min_x <= :max_x
                    AND max_y >= :min_y
                    AND min_y <= :max_y
                    """

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
                }
            )

            # The query is a full intersection check, not a simple bounds
            # check only.
            if (
                geometry_predicate is not None
                and geometry_predicate != "bbox_intersects"
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
            sql_predicate = eval(where, SQL_GLOBALS, {})  # skipcq: PYL-W0123
            query_string += f" AND {sql_predicate}"

        return query_string, query_parameters

    def _query(
        self,
        columns: str,
        geometry: Optional[Geometry] = None,
        callable_columns: Optional[str] = None,
        geometry_predicate="intersects",
        where: Optional[Predicate] = None,
        unique: bool = False,
        no_constraints_ok: bool = False,
        index_warning: bool = False,
        min_area=None,
    ) -> sqlite3.Cursor:
        """Common query construction logic for `query` and `iquery`.

        Args:
            columns(str):
                The columns to select.
            geometry(tuple or Geometry):
                The geometry being queried against.
            select_callable(str):
                The rows to select when a callable is given to `where`.
            callable_columns(str):
                The columns to select when a callable is given to
                `where`.
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

        Returns:
            sqlite3.Cursor:
                A database cursor for the current query.

        """
        if not no_constraints_ok and all(x is None for x in (geometry, where)):
            raise ValueError("At least one of `geometry` or `where` must be specified.")
        query_geometry = geometry
        if callable_columns is None:
            callable_columns = columns
        if geometry_predicate not in self._geometry_predicate_names:
            raise ValueError(
                "Invalid geometry predicate."
                f"Allowed values are: {', '.join(self._geometry_predicate_names)}."
            )
        cur = self.con.cursor()

        # Normalise query geometry and determine if it is a rectangle
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)

        if isinstance(where, Callable):
            columns = callable_columns

        query_parameters = {}

        query_string, query_parameters = self._initialize_query_string_parameters(
            query_geometry, query_parameters, geometry_predicate, columns, where
        )

        if min_area is not None and "area" in self.table_columns:
            query_string += f"\nAND area > {min_area}"
        elif min_area is not None:
            raise ValueError(
                """Cannot use `min_area` without an area column.
            SQLiteStore.add_area_column() can be used to add an area column."""
            )

        if unique:
            query_string = query_string.replace("SELECT", "SELECT DISTINCT")

        # Warn if the query is not using an index
        if index_warning:
            query_plan = cur.execute(
                "EXPLAIN QUERY PLAN " + query_string, query_parameters
            ).fetchone()
            if "USING INDEX" not in query_plan[-1]:
                warnings.warn(
                    "Query is not using an index. "
                    "Consider adding an index to improve performance."
                )
        # if area column exists, sort annotations by area
        if "area" in self.table_columns:
            query_string += "\nORDER BY area DESC"
        cur.execute(query_string, query_parameters)
        return cur

    def iquery(
        self,
        geometry: Optional[QueryGeometry] = None,
        where: Optional[Predicate] = None,
        geometry_predicate="intersects",
        min_area=None,
    ) -> List[str]:
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
                may be a string, callable, or pickled function as bytes.
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
        query_geometry = geometry
        cur = self._query(
            "[key]",
            geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            where=where,
            callable_columns="[key], properties",
            min_area=min_area,
        )
        if isinstance(where, Callable):
            return [
                key
                for key, properties in cur.fetchall()
                if where(json.loads(properties))
            ]
        return [key for key, in cur.fetchall()]

    def query(
        self,
        geometry: Optional[QueryGeometry] = None,
        where: Optional[Predicate] = None,
        geometry_predicate: str = "intersects",
        min_area=None,
    ) -> Dict[str, Annotation]:
        """Runs Query."""
        query_geometry = geometry
        cur = self._query(
            columns="[key], properties, cx, cy, geometry",
            geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            where=where,
            min_area=min_area,
        )
        if isinstance(where, Callable):
            return {
                key: Annotation(
                    geometry=self._unpack_geometry(blob, cx, cy),
                    properties=json.loads(properties),
                )
                for key, properties, cx, cy, blob in cur.fetchall()
                if where(json.loads(properties))
            }
        return {
            key: Annotation(
                geometry=self._unpack_geometry(blob, cx, cy),
                properties=json.loads(properties),
            )
            for key, properties, cx, cy, blob in cur.fetchall()
        }

    def bquery(
        self,
        geometry: Optional[QueryGeometry] = None,
        where: Union[str, bytes, Callable[[Geometry, Dict[str, Any]], bool]] = None,
        min_area=None,
    ) -> Dict[str, Tuple[float, float, float, float]]:
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
                may be a string, callable, or pickled function as bytes.
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
            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> from shapely.geometry import Polygon
            >>> store = AnnotationStore()
            >>> store.add(
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
            min_area=min_area,
        )
        if isinstance(where, Callable):
            return {
                key: bounds
                for key, properties, *bounds in cur.fetchall()
                if where(json.loads(properties))
            }
        return {key: bounds for key, *bounds in cur.fetchall()}

    @staticmethod
    def _handle_pickle_callable_pquery(
        select: CallableSelect,
        where: Optional[CallablePredicate],
        cur: sqlite3.Cursor,
        unique: bool,
    ) -> Union[Dict[str, Set[Properties]], Dict[str, Properties]]:
        """Package the results of a pquery into the right output format.

        This variant is used when select and where are callable or
        pickle objects.

        Args:
            select (Union[str, bytes, Callable]):
                A callable to select the properties to return.
            where (CallablePredicate):
                A callable predicate to filter the rows with. Maybe
                None for no-op (no filtering).
            cur (sqlite3.Cursor):
                The cursor for the query.
            unique (bool):
                Whether to return only unique results.

        Returns:
            dict:
                If unique, a dictionary of sets is returned. Otherwise,
                a dictionary mapping annotation keys to JSON-like
                property dictionaries is returned.

        """

        def add_props_to_result(
            result: DefaultDict[str, set], properties: Dict[str, Any]
        ) -> None:
            """Add the properties to the appropriate set in result.

            Args:
                result (DefaultDict[str, set]):
                    The result dictionary to add the properties to.
                properties (Dict[str, Any]):
                    The properties to add to the result.

            """
            # Get the selected values
            selection = select(properties)
            # Wrap scalar values into a tuple
            if not isinstance(selection, tuple):
                selection = (selection,)
            # Add the properties to the appropriate set
            for i, value in enumerate(selection):
                result[i].add(value)

        # Load a pickled select function
        if isinstance(select, bytes):
            select = pickle.loads(select)  # skipcq: BAN-B301
        if unique:
            # Create a dictionary of sets to store the unique properties
            # for each property key / name.
            result = defaultdict(set)
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
        unique: bool,
        star_query: bool,
    ) -> Union[Dict[str, Set[Properties]], Dict[str, Properties]]:
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
        select: Union[str, bytes, Callable],
        where: Union[str, bytes, Callable],
    ) -> Tuple[bool, bool, bool]:
        """Determine boolean flags for the kind of pquery this is.

        If either one of `select` or `where` is a str, bytes, or
        callable, then is_callable_query, is_pickle_query, and
        is_str_query respectively will be set to True.

        Returns:
            tuple:
                A tuple of bools:
                - True if select or where are callable (functions).
                - True if select or where are bytes (pickle expressions).
                - True if select or where are str (SQL expressions).

        """
        is_callable_query = any(isinstance(x, Callable) for x in (select, where) if x)
        is_pickle_query = any(isinstance(x, bytes) for x in (select, where) if x)
        is_str_query = any(isinstance(x, str) for x in (select, where) if x)

        return is_callable_query, is_pickle_query, is_str_query

    @staticmethod
    def _validate_select_where_type(
        select: Union[str, bytes, Callable],
        where: Union[str, bytes, Callable],
    ) -> None:
        """Validate that select and where are valid types.

        1. Check that select and where are the same type if where is given.
        2. Check that select is in (str, bytes, callable).

        Raises:
            TypeError:
                If select and where are not the same type or not in
                (str, bytes, callable).

        """
        if where is not None and type(select) is not type(where):
            raise TypeError("select and where must be of the same type")
        if not isinstance(select, (str, bytes)) and not callable(select):
            raise TypeError(
                f"select must be str, bytes, or callable, not {type(select)}"
            )

    def pquery(
        self,
        select: Select,
        geometry: Optional[QueryGeometry] = None,
        where: Optional[Predicate] = None,
        geometry_predicate: str = "intersects",
        unique: bool = True,
        squeeze: bool = True,
    ) -> Union[Dict[str, Any], Set[Any]]:
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
                may be a string, callable, or pickled function as bytes.
                Callables are called to filter each result returned the
                from annotation store backend in python before being
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

            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> from shapely.geometry import Point
            >>> store = AnnotationStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.add(annotation, "foo")
            >>> store.pquery("*", unique=False)
            ... {'foo': {'class': 42}}

            >>> from tiatoolbox.annotation.storage import AnnotationStore
            >>> from shapely.geometry import Point
            >>> store = AnnotationStore()
            >>> annotation =  Annotation(
            ...     geometry=Point(0, 0),
            ...     properties={"class": 42},
            ... )
            >>> store.add(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42}
            >>> annotation =  Annotation(Point(1, 1), {"class": 123})
            >>> store.add(annotation, "foo")
            >>> store.pquery("props['class']")
            ... {42, 123}

        """  # noqa

        self._validate_select_where_type(select, where)

        is_callable_query, is_pickle_query, is_str_query = self._kind_of_pquery(
            select, where
        )

        is_star_query = select == "*"  # Get all properties, special case
        query_geometry = geometry  # Rename arg
        return_columns = []  # Initialise return rows list of column names

        if is_star_query and unique:
            raise ValueError("unique=True cannot be used with select='*'")

        if not unique:
            return_columns.append("[key]")
        if is_str_query and not is_star_query:
            select_names = eval(select, SQL_GLOBALS, {})  # skipcq: PYL-W0123
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
            # only done for callable where.
            post_where = where if is_callable_query else None
            result = self._handle_pickle_callable_pquery(
                select, post_where, cur, unique
            )
        else:
            result = self._handle_str_pquery(cur, unique, is_star_query)

        if unique and squeeze and len(result) == 1:
            return result[0]
        return result

    def __len__(self) -> int:
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM annotations")
        (count,) = cur.fetchone()
        return count

    def __contains__(self, key: str) -> bool:
        cur = self.con.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM annotations WHERE [key] = ?)", (key,))
        return cur.fetchone()[0] == 1

    def __getitem__(self, key: str) -> Annotation:
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
        geometry = self._unpack_geometry(
            serialised_geometry,
            cx,
            cy,
        )
        return Annotation(geometry, properties)

    def keys(self) -> Iterable[int]:
        """Return an iterable (usually generator) of all keys in the store.

        Returns:
            Iterable[str]:
                An iterable of keys.

        """
        yield from self

    def __iter__(self) -> Iterable[int]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key]
              FROM annotations
            """
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            yield row[0]  # The key

    def values(self) -> Iterable[Tuple[int, Annotation]]:
        """Return an iterable of all annotation in the store.

        Returns:
            Iterable[Annotation]:
                An iterable of annotations.

        """
        for _, value in self.items():
            yield value

    def items(self) -> Iterable[Tuple[int, Annotation]]:
        """Returns iterable (generator) over key and annotations."""
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key], cx, cy, geometry, properties
              FROM annotations
            """
        )
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
            yield key, Annotation(geometry, properties)

    def patch_many(
        self,
        keys: Iterable[int],
        geometries: Optional[Iterable[Geometry]] = None,
        properties_iter: Optional[Iterable[Properties]] = None,
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
            raise ValueError(
                "At least one of geometries or properties_iter must be given"
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
                self._append(key, Annotation(geometry, properties), cur)
                continue
            # Annotation is in DB:
            if geometry:
                self._patch_geometry(key, geometry, cur)
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
        self, key: str, geometry: Geometry, cur: sqlite3.Cursor
    ) -> None:
        """Patch a geometry in the database.

        Update the geometry of the annotation with the given key but
        leave the properties untouched.

        Args:
            key: The key of the annotation to patch.
            geometry: The new geometry.
            cur: The cursor to use.

        """
        bounds = dict(zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds))
        xy = dict(zip("xy", np.array(geometry.centroid)))
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

    def remove_many(self, keys: Iterable[str]) -> None:
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

    def __setitem__(self, key: str, annotation: Annotation) -> None:
        if key in self:
            self.patch(key, annotation.geometry, annotation.properties)
            return
        self.append(annotation, key)

    def _get_table_columns(self):
        """Get a list of columns in the annotations table."""
        cur = self.con.execute("PRAGMA table_info(annotations)")
        return [row[1] for row in cur.fetchall()]

    def add_area_column(self, mk_index=True):
        """Add a column to store the area of the geometry."""
        cur = self.con.cursor()
        cur.execute(
            """
            ALTER TABLE annotations
            ADD COLUMN area INTEGER NOT NULL DEFAULT 0
            """
        )
        cur.execute(
            """
            UPDATE annotations
            SET area = get_area(geometry, cx, cy)
            """
        )
        if mk_index:
            self.create_index("area", '"area"')
        self.con.commit()
        self.table_columns.append("area")

    def remove_area_column(self):
        """Remove the area column from the store."""
        if "area" in self.indexes():
            self.drop_index("area")
        cur = self.con.cursor()
        cur.execute(
            """
            ALTER TABLE annotations
            DROP COLUMN area
            """
        )
        self.con.commit()
        self.table_columns.remove("area")

    def to_dataframe(self) -> pd.DataFrame:
        """Converts AnnotationStore to :class:`pandas.DataFrame`."""
        df = pd.DataFrame()
        df_rows = (
            {
                "key": key,
                "geometry": annotation.geometry,
                "properties": annotation.properties,
            }
            for key, annotation in self.items()
        )
        df = df.append(pd.json_normalize(df_rows))
        return df.set_index("key")

    def features(self) -> Generator[Dict[str, Any], None, None]:
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

    def commit(self) -> None:
        self.con.commit()

    def dump(self, fp: Union[Path, str, IO]) -> None:
        if hasattr(fp, "write"):
            fp = fp.name
        target = sqlite3.connect(fp)
        self.con.backup(target)

    def dumps(self) -> str:
        return "\n".join(self.con.iterdump())

    def clear(self) -> None:
        """Remove all annotations from the store."""
        cur = self.con.cursor()
        cur.execute("DELETE FROM rtree")
        cur.execute("DELETE FROM annotations")
        if self.auto_commit:
            self.con.commit()

    def create_index(
        self, name: str, where: Union[str, bytes], analyze: bool = True
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
        if minor < 9:
            raise Exception("Requires sqlite version 3.9.0 or higher.")
        cur = self.con.cursor()
        if not isinstance(where, str):
            raise TypeError(f"Invalid type for `where` ({type(where)}).")
        sql_predicate = eval(where, SQL_GLOBALS)  # skipcq: PYL-W0123
        cur.execute(f"CREATE INDEX {name} ON annotations({sql_predicate})")
        if analyze:
            cur.execute(f"ANALYZE {name}")

    def indexes(self) -> List[str]:
        """Returns a list of the names of all indexes in the store.

        Returns:
            List[str]:
                The list of index names.

        """
        cur = self.con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE TYPE = 'index'")
        return [row[0] for row in cur.fetchall()]

    def drop_index(self, name: str) -> None:
        """Drop an index from the store.

        Args:
            name (str):
                The name of the index to drop.

        """
        cur = self.con.cursor()
        cur.execute(f"DROP INDEX {name}")

    def optimize(self, vacuum: bool = True, limit: int = 1000) -> None:
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

    def __init__(self, connection: Union[Path, str, IO] = ":memory:") -> None:
        super().__init__()
        self._rows = {}
        self.connection = connection
        self.path = self._connection_to_path(connection)
        if self.connection not in [None, ":memory:"] and self.path.exists():
            for line in self._load_cases(
                fp=self.connection,
                string_fn=lambda fp: fp.splitlines(),
                file_fn=lambda fp: fp.readlines(),
            ):
                dictionary = json.loads(line)
                key = dictionary.get("key", uuid.uuid4().hex)
                geometry = feature2geometry(dictionary["geometry"])
                properties = dictionary["properties"]
                self.append(Annotation(geometry, properties), key=key)

    def append(
        self,
        annotation: Annotation,
        key: Optional[str] = None,
    ) -> int:
        if not isinstance(annotation.geometry, (Polygon, Point, LineString)):
            raise TypeError("Invalid geometry type.")
        key = key or str(uuid.uuid4())
        self._rows[key] = {"annotation": annotation}
        return key

    def patch(
        self,
        key: str,
        geometry: Optional[Geometry] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        if key not in self:
            self.append(Annotation(geometry, properties), key)
            return
        existing = self[key]
        geometry = geometry or existing.geometry
        properties = properties or {}
        new_properties = copy.deepcopy(existing.properties)
        new_properties.update(properties)
        self[key] = Annotation(geometry, new_properties)

    def remove(self, key: str) -> None:
        del self._rows[key]

    def __getitem__(self, key: str) -> Annotation:
        return self._rows[key]["annotation"]

    def __setitem__(self, key: str, annotation: Annotation) -> None:
        if key in self._rows:
            self._rows[key]["annotation"] = annotation
        self._rows[key] = {"annotation": annotation}

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def items(self) -> Generator[Tuple[str, Annotation], None, None]:
        """Returns iterable (generator) over key and annotations."""
        for key, row in self._rows.items():
            yield key, row["annotation"]

    def __len__(self) -> int:
        return len(self._rows)

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO]) -> "DictionaryStore":  # noqa: A003
        return cls.from_ndjson(fp)

    def commit(self) -> None:
        if str(self.connection) == ":memory:":
            warnings.warn("In-memory store. Nothing to commit.")
            return
        if not self.path.exists():
            self.path.touch()
        self.dump(self.connection)

    def dump(self, fp: Union[Path, str, IO]) -> None:
        return self.to_ndjson(fp)

    def dumps(self) -> str:
        return self.to_ndjson()

    def close(self) -> None:
        warnings.simplefilter("ignore")
        # Try to commit any changes if the file is still open.
        with contextlib.suppress(ValueError):
            self.commit()
        warnings.resetwarnings()
