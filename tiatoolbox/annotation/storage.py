# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""
This module contains a collection of classes for handling storage of
annotations in memeory in addition to serialisation/deserialisaion
to/from disk.

Definitions
-----------

For the sake of clarity it is helpful to define a few terms used
throught this documentation.

Annotation
    A geometry and associated properties.
Geometry
    One of: a point, a polygon, or a line string.
    .. image: images/geometries.png
Properties
    Key-value pairs associated with a geometry.

"""
import copy
import itertools
import pickle
import sqlite3
import uuid
import warnings
from abc import ABC
from functools import lru_cache
from numbers import Number
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from shapely import speedups, wkb, wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import mapping as geometry2feature
from shapely.geometry import shape as feature2geometry

try:
    import ujson as json  # pyright: reportMissingModuleSource=false
except ImportError:
    import json

from tiatoolbox.annotation.predicate import PY_GLOBALS, SQL_GLOBALS

sqlite3.enable_callback_tracebacks(True)

if speedups.available:
    speedups.enable()

Geometry = Union[Point, Polygon, LineString]
BBox = Tuple[Number, Number, Number, Number]
QueryGeometry = Union[BBox, Geometry]

ASCII_FILE_SEP = "\x1c"
ASCII_GROUP_SEP = "\x1d"
ASCII_RECORD_SEP = "\x1e"
ASCII_UNIT_SEP = "\x1f"
ASCII_NULL = "\0"
ISO_8601_DATE_FORMAT = r"%Y-%m-%dT%H:%M:%S.%f%z"


class AnnotationStoreABC(ABC):
    """Annotation storage abstract base class."""

    @staticmethod
    def _geometry_predicate(name: str, a: Geometry, b: Geometry) -> Callable:
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
    ]

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO]) -> "AnnotationStoreABC":
        """Load a store object from a path or file-like object."""
        raise NotImplementedError()

    @staticmethod
    def serialise_geometry(geometry: Geometry) -> Union[str, bytes]:
        """Serialise a geometry to a string or bytes."""
        return geometry.wkt

    @staticmethod
    @lru_cache(32)
    def deserialise_geometry(data: Union[str, bytes]):
        """Deserialise a geometry from a string or bytes."""
        if isinstance(data, str):
            return wkt.loads(data)
        return wkb.loads(data)

    def commit(self) -> None:
        """Commit any in-memory changes to disk."""
        raise NotImplementedError()

    def dump(self, fp: Union[Path, str, IO]) -> None:
        """Serialise a copy of the store to a file-like object."""
        raise NotImplementedError()

    def dumps(self) -> str:
        """Serialise and return a copy of store as a string."""
        raise NotImplementedError()

    def append(
        self,
        geometry: Geometry,
        properties: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> int:
        """Insert a new annotation, returning the key."""
        if properties is not None:
            properties_iter = [properties]
        if key is not None:
            keys = [key]
        return self.append_many([geometry], properties_iter, keys)[0]

    def append_many(
        self,
        geometries: Iterable[Geometry],
        properties_iter: Optional[Iterable[Dict[str, Any]]] = None,
        keys: Optional[Iterable[str]] = None,
    ) -> List[int]:
        if properties_iter is None:
            properties_iter = ({} for _ in geometries)
        if keys is None:
            keys = (str(uuid.uuid4()) for _ in geometries)
        for geometry, properties in zip(geometries, properties_iter):
            key = self.append(geometry, properties)
            keys.append(key)
        return keys

    def update(self, key: str, update: Dict[str, Any]) -> None:
        """Update an annotation at given key."""
        return self.update_many([key], [update])

    def update_many(
        self, keys: Iterable[int], updates: Iterable[Dict[str, Any]]
    ) -> None:
        for key, update in zip(keys, updates):
            self.update(key, update)

    def remove(self, key: str) -> None:
        """Remove annotation by key."""
        self.remove_many([key])

    def remove_many(self, keys: Iterable[int]) -> None:
        """Bulk removal of annotations by key."""
        for key in keys:
            self.remove(key)

    def __len__(self) -> int:
        """Return the number of annotations in the store."""
        raise NotImplementedError()

    def __getitem__(self, key: str) -> Tuple[Geometry, Dict[str, Any]]:
        raise NotImplementedError()

    def __setitem__(
        self, key: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        raise NotImplementedError()

    def __delitem__(self, key: str) -> None:
        self.remove(key)

    def keys(self) -> Iterable[int]:
        for key, _ in self.items():
            yield key

    def values(self) -> Iterable[Tuple[Geometry, Dict[str, Any]]]:
        for _, value in self.items():
            yield value

    def items(self) -> Iterable[Tuple[int, Tuple[Geometry, Dict[str, Any]]]]:
        raise NotImplementedError()

    def __iter__(self) -> Iterable[int]:
        raise NotImplementedError()

    @staticmethod
    def _eval_properties_predicate(
        properties_predicate: Optional[
            Union[str, bytes, Callable[[Geometry, Dict[str, Any]], bool]]
        ],
        properties: Dict[str, Any],
    ) -> bool:
        """Evaluate properties predicate against properties."""
        if properties_predicate is None:
            return True
        if isinstance(properties_predicate, str):
            return bool(eval(properties_predicate, PY_GLOBALS, {"props": properties}))
        if isinstance(properties_predicate, bytes):
            properties_predicate = pickle.loads(
                properties_predicate
            )  # noqa: SC100 skipcq: BAN-B301
        return bool(properties_predicate(properties))

    def query(
        self,
        query_geometry: QueryGeometry,
        geometry_predicate: str = "intersects",
        properties_predicate: Union[
            str, bytes, Callable[[Geometry, Dict[str, Any]], bool]
        ] = None,
    ) -> List[Tuple[Geometry, Dict[str, Any]]]:
        """Query the store for annotations.

        Args:
            query_geometry:
                Geometry to use when querying. This can be a bounds or
                a Shapely geometry (e.g. Polygon).
            geometry_predicate:
                A string which define which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                intersects.
                For more information see the `shapely documentation on
                binary predicates`__.
            properties_predicate:
                A statment which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). May
                be a string, callable, or pickled function as bytes.
                Callables are called to filter each result returned
                the from annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are converted to a domain specific predicate
                representation on a best-effort basis. E.g. a simple
                python expression `props["class"] == 42` will be
                converted to a valid SQLite predicate when using
                `SQLiteStore` and inserted into the SQL query. This
                should be faster than filtering in python after or
                during the query. Additionally, the same string can be
                used across different backends (e.g. the previous
                example predicate string is valid for both
                `DictionaryStore `and a `SQliteStore`). On the
                other hand it has many more limitations. It is important
                to note that untrusted user input should never be
                accepted to this argument as arbitrary code can be run
                via pickle or the parsing of the string statement.

            Returns:
                list: A list of 2-tuples containing:
                - geometry: The annotation Shapely geometry object.
                - properties: The properties JSON as a dictionary.

            .. _BP:
                | https://shapely.readthedocs.io/en/stable/
                | manual.html#binary-predicates

            __ BP_

        """
        if isinstance(query_geometry, tuple):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            (geometry, properties)
            for geometry, properties in self.values()
            if (
                self._geometry_predicate(geometry_predicate, query_geometry, geometry)
                and self._eval_properties_predicate(properties_predicate, properties)
            )
        ]

    def iquery(
        self,
        query_geometry: QueryGeometry,
        geometry_predicate: str = "intersects",
        properties_predicate: Union[
            str, bytes, Callable[[Geometry, Dict[str, Any]], bool]
        ] = None,
    ) -> List[int]:
        """Query the store for annotation keys.

        Acts the same as `AnnotationStore.query` except returns keys
        insteaf the annotations (geometry, properties).

        Args:
            query_geometry:
                Geometry to use when querying. This can be a bounds or
                a Shapely geometry (e.g. Polygon).
            geometry_predicate:
                A string which define which binary geometry predicate to
                use when comparing the query geometry and a geometry in
                the store. Only annotations for which this binary
                predicate is true will be returned. Defaults to
                intersects.
                For more information see the `shapely documentation on
                binary predicates`__.
            properties_predicate:
                A statment which should evaluate to a boolean value.
                Only annotations for which this predicate is true will
                be returned. Defaults to None (assume always true). May
                be a string, callable, or pickled function as bytes.
                Callables are called to filter each result returned
                the from annotation store backend in python before being
                returned to the user. A pickle object is, where
                possible, hooked into the backend as a user defined
                function to filter results during the backend query.
                Strings are converted to a domain specific predicate
                representation on a best-effort basis. E.g. a simple
                python expression `props["class"] == 42` will be
                converted to a valid SQLite predicate when using
                `SQLiteStore` and inserted into the SQL query. This
                should be faster than filtering in python after or
                during the query. Additionally, the same string can be
                used across different backends (e.g. the previous
                example predicate string is valid for both a
                 `DictionaryStore `and a `SQliteStore`). On the
                other hand it has many more limitations. It is important
                to note that untrusted user input should never be
                accepted to this argument as arbitrary code can be run
                via pickle or the parsing of the string statement.

            Returns:
                list: A list of 2-tuples containing:
                - geometry: The annotation Shapely geometry object.
                - properties: The properties JSON as a dictionary.

            .. _BP:
                | https://shapely.readthedocs.io/en/stable/
                | manual.html#binary-predicates

            __ BP_

        """
        if isinstance(query_geometry, tuple):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            key
            for key, (geometry, properties) in self.items()
            if (
                self._geometry_predicate(geometry_predicate, query_geometry, geometry)
                and self._eval_properties_predicate(properties_predicate, properties)
            )
        ]

    def features(self) -> Generator[Dict[str, Any], None, None]:
        """Return anotations as a list of geoJSON features.

        Returns:
            list: List of features as dictionaries.

        """
        raise NotImplementedError()

    def to_geodict(self) -> Dict[str, Any]:
        """Return annotations as a dictionary in geoJSON format.

        Args:
            int_coords(bool): Make coordinates intergers. Defaults to
                True.
            drop_na(bool): Don't include keys for None/NaN values.
                Defaults to True.

        Returns:
            dict: Dictionary of annotations in geoJSON format.

        """
        return {
            "type": "FeatureCollection",
            "features": list(self.features()),
        }

    @staticmethod
    def _dump_cases(
        fp: Union[IO, str, Path, None],
        file_fn: Callable[[IO], None],
        none_fn: Callable[[], str],
    ) -> Union[None, Any]:
        if fp is not None:
            # It is a file-like object, write to it
            if hasattr(fp, "write"):
                return file_fn(fp)
            # Turn a path into a file handle, then write to it
            with open(fp, "w", encoding="utf-8") as file_handle:
                return file_fn(file_handle)
        # Return as an object (str or bytes) if no handle/path is given
        return none_fn()

    @staticmethod
    def _load_cases(
        fp: Union[IO, str, Path],
        string_fn: Callable[[Union[str, bytes]], Any],
        file_fn: Callable[[IO], Any],
    ) -> Any:
        try:
            return string_fn(fp)
        except TypeError:
            if hasattr(fp, "read"):
                return file_fn(fp)
            with open(fp) as file_handle:
                return file_fn(file_handle)

    @classmethod
    def from_geojson(cls, fp: Union[IO, str]) -> "AnnotationStoreABC":
        geojson = cls._load_cases(
            fp=fp,
            string_fn=json.loads,
            file_fn=json.load,
        )
        store = cls()
        for feature in geojson["features"]:
            geometry = feature2geometry(feature["geometry"])
            properties = feature["properties"]
            store.append(geometry, properties)
        return store

    def to_geojson(self, fp: Optional[IO] = None) -> Union[str, None]:
        """Serialise the store to geoJSON.

        Args:
             fp (IO): A file-like object supporting `.read`. Defaults to
                None which returns geoJSON as a string.

        Returns:
            None or str: None if writing to file or the geoJSON string.

        """
        return self._dump_cases(
            fp=fp,
            file_fn=lambda fp: json.dump(self.to_geodict(), fp),
            none_fn=lambda: json.dumps(self.to_geodict()),
        )

    def to_ldjson(self, fp: Optional[IO] = None) -> Union[str, None]:
        """Serialise to Line-Delimited (Geo)JSON."""
        string_lines_generator = (
            json.dumps(line, separators=(",", ":")) + "\n" for line in self.features()
        )
        return self._dump_cases(
            fp=fp,
            file_fn=lambda fp: fp.writelines(string_lines_generator),
            none_fn=lambda: "".join(string_lines_generator),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "AnnotationStoreABC":
        store = cls()
        for _, row in df.iterrows():
            geometry = row["geometry"]
            properties = dict(row.filter(regex="^(?!geometry|key).*$"))
            store.append(geometry, properties)
        return store

    def to_dataframe(self) -> pd.DataFrame:
        features = (
            {"key": key, "geometry": geometry, "properties": properties}
            for key, (geometry, properties) in self.items()
        )
        return pd.json_normalize(features).set_index("key")

    def __del__(self) -> None:
        self.close()


class SQLiteStore(AnnotationStoreABC):
    """SQLite backed annotation store."""

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str]) -> "SQLiteStore":
        return SQLiteStore(fp)

    def __init__(self, connection: Union[Path, str] = ":memory:") -> None:
        super().__init__()
        # Check that JSON and RTree support is enabled
        compile_options = self.compile_options()
        if not all(
            ["ENABLE_JSON1" in compile_options, "ENABLE_RTREE" in compile_options]
        ):
            raise Exception("RTREE and JSON1 sqlite3 compile options are required.")

        # Set up database connection and cursor
        path = Path(connection)
        exists = path.exists() and path.is_file()
        self.con = sqlite3.connect(connection, isolation_level="DEFERRED")

        # Register predicate functions as custom SQLite functions
        def wkb_predicate(name: str, wkb_a: bytes, b: bytes) -> bool:
            """Wrapper function to allow WKB as inputs to binary predicates."""
            a = wkb.loads(wkb_a)
            b = self.deserialise_geometry(b)
            return self._geometry_predicate(name, a, b)

        def pickle_properties_predicate(pickle_bytes: bytes, properties: str) -> bool:
            fn = pickle.loads(pickle_bytes)  # noqa: SC100 skipcq: BAN-B301
            properties = json.loads(properties)
            return fn(properties)

        try:
            self.con.create_function(
                "geometry_predicate", 3, wkb_predicate, deterministic=True
            )
            self.con.create_function(
                "pickle_properties_predicate",
                2,
                pickle_properties_predicate,
                deterministic=True,
            )
        # Only Python >= 3.8 supports deterministic, fallback
        # to without this argument.
        except TypeError:
            self.con.create_function("geometry_predicate", 3, wkb_predicate)
            self.con.create_function(
                "pickle_properties_predicate", 2, pickle_properties_predicate
            )

        if exists:
            return
        cur = self.con.cursor()

        # Create tables for geometry and RTree index
        cur.execute(
            """
            CREATE VIRTUAL TABLE main
            USING rtree_i32(
                id,                      -- Integer primary key
                min_x, max_x,            -- 1st dimension min, max
                min_y, max_y,            -- 2nd dimension min, max
                +key TEXT UNIQUE,        -- Unique identifier
                +objtype TEXT,           -- Object type
                +cx INTEGER NOT NULL,    -- X of centroid/representative point
                +cy INTEGER NOT NULL,    -- Y of centroid/representative point
                +boundary BLOB NOT NULL, -- Detailed boundary
                +properties TEXT,        -- JSON properties
            )
            """
        )
        self.con.commit()

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
        self.con.close()

    def _make_token(
        self, geometry: Geometry, properties: Dict, key: Optional[str]
    ) -> Dict:
        """Create token data dict for tokenised SQL transaction."""
        key = key or str(uuid.uuid4())
        if geometry.geom_type == "Point":
            boundary = None
        else:
            boundary = self.serialise_geometry(geometry)
        return {
            "key": key,
            "boundary": boundary,
            "cx": int(geometry.centroid.x),
            "cy": int(geometry.centroid.y),
            "min_x": geometry.bounds[0],
            "min_y": geometry.bounds[1],
            "max_x": geometry.bounds[2],
            "max_y": geometry.bounds[3],
            "geom_type": geometry.geom_type,
            "properties": json.dumps(properties, separators=(",", ":")),
        }

    def append_many(
        self,
        geometries: Iterable[Geometry],
        properties_iter: Optional[Iterable[Dict[str, Any]]] = None,
        keys: Optional[Iterable[str]] = None,
    ) -> List[int]:
        if properties_iter is None:
            properties_iter = itertools.repeat({})
        if keys is None:
            keys = (str(uuid.uuid4()) for _ in geometries)
        tokens = [
            self._make_token(
                geometry=geometry,
                properties=properties,
                key=key,
            )
            for geometry, properties, key in zip(geometries, properties_iter, keys)
        ]
        cur = self.con.cursor()
        cur.executemany(
            """
            INSERT INTO main VALUES(
                NULL, :min_x, :max_x, :min_y, :max_y, :key, :geom_type,
                :cx, :cy, :boundary, :properties
            )
            """,
            tokens,
        )
        self.con.commit()
        return [token["key"] for token in tokens]

    def _query(
        self,
        query_select: str,
        query_geometry: QueryGeometry,
        query_select_callable: Optional[str] = None,
        geometry_predicate="intersects",
        properties_predicate: Union[
            str, bytes, Callable[[Geometry, Dict[str, Any]], bool]
        ] = None,
    ) -> sqlite3.Cursor:
        """Common query construction logic for `query` and `iquery`."""
        if geometry_predicate not in self._geometry_predicate_names:
            raise ValueError(
                "Invalid geometry predicate."
                f"Allowed values are: {', '.join(self._geometry_predicate_names)}."
            )
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        min_x, min_y, max_x, max_y = query_geometry.bounds

        if isinstance(properties_predicate, Callable):
            query_select = query_select_callable

        query_string = (
            "SELECT "
            + query_select
            + """
         FROM main
        WHERE max_x >= :min_x
          AND min_x <= :max_x
          AND max_y >= :min_y
          AND min_y <= :max_y
          AND geometry_predicate(:geometry_predicate, :query_geometry, boundary)
        """
        )
        query_parameters = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "geometry_predicate": geometry_predicate,
            "query_geometry": query_geometry.wkb,
        }
        if isinstance(properties_predicate, str):
            sql_predicate = eval(
                properties_predicate, SQL_GLOBALS, {}
            )  # noqa: SC100 skipcq: PYL-W0123
            query_string += f"AND {sql_predicate}"
        if isinstance(properties_predicate, bytes):
            query_string += (
                "AND pickle_properties_predicate(:properties_predicate, properties)"
            )
            query_parameters["properties_predicate"] = properties_predicate
        cur.execute(query_string, query_parameters)
        return cur

    def iquery(
        self,
        query_geometry: QueryGeometry,
        geometry_predicate="intersects",
        properties_predicate: Union[
            str, bytes, Callable[[Geometry, Dict[str, Any]], bool]
        ] = None,
    ) -> List[int]:
        cur = self._query(
            "[key]",
            query_geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            properties_predicate=properties_predicate,
            query_select_callable="[key], boundary, properties",
        )
        if isinstance(properties_predicate, Callable):
            return [
                key
                for key, boundary, properties in cur.fetchall()
                if properties_predicate(json.loads(properties))
            ]
        return [key for key, in cur.fetchall()]

    def query(
        self,
        query_geometry: QueryGeometry,
        geometry_predicate: str = "intersects",
        properties_predicate: Union[
            str, bytes, Callable[[Geometry, Dict[str, Any]], bool]
        ] = None,
    ) -> List[Tuple[Geometry, Dict[str, Any]]]:
        cur = self._query(
            "boundary, properties",
            query_geometry=query_geometry,
            geometry_predicate=geometry_predicate,
            properties_predicate=properties_predicate,
        )
        if isinstance(properties_predicate, Callable):
            return [
                (boundary, properties)
                for boundary, properties in cur.fetchall()
                if properties_predicate(json.loads(properties))
            ]
        return [
            (
                self.deserialise_geometry(blob),
                json.loads(properties),
            )
            for blob, properties in cur.fetchall()
        ]

    def __len__(self) -> int:
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM main")
        (count,) = cur.fetchone()
        return count

    def __contains__(self, key: int) -> bool:
        cur = self.con.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM main WHERE [key] = ?)", (key,))
        return cur.fetchone()[0] == 1

    def __getitem__(self, key: str) -> Tuple[Geometry, Dict[str, Any]]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT boundary, properties
              FROM main
             WHERE [key] = :key
            """,
            {"key": key},
        )
        boundary, properties = cur.fetchone()
        if properties is None:
            properties = {}
        else:
            properties = json.loads(properties)
        geometry = self.deserialise_geometry(boundary)
        return geometry, properties

    def keys(self) -> Iterable[int]:
        yield from self

    def __iter__(self) -> Iterable[int]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key]
              FROM main
            """
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            key = row[0]
            yield key

    def values(self) -> Iterable[Tuple[int, Tuple[Geometry, Dict[str, Any]]]]:
        for _, value in self.items():
            yield value

    def items(self) -> Iterable[Tuple[int, Tuple[Geometry, Dict[str, Any]]]]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key], cx, cy, boundary, properties
              FROM main
            """
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            key, cx, cy, boundary, properties = row
            if boundary is not None:
                geometry = self.deserialise_geometry(boundary)
            else:
                geometry = Point(cx, cy)
            properties = json.loads(properties)
            yield key, (geometry, properties)

    def update_many(
        self, keys: Iterable[int], updates: Iterable[Dict[str, Any]]
    ) -> None:
        cur = self.con.cursor()
        cur.execute("BEGIN")
        for key, update in zip(keys, updates):
            update = copy.copy(update)
            geometry = update.pop("geometry", None)
            if geometry is not None:
                bounds = dict(
                    zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds)
                )
                xy = dict(zip("xy", np.array(geometry.centroid)))
                cur.execute(
                    """
                    UPDATE main
                       SET cx = :x, cy = :y, boundary = :boundary,
                           min_x = :min_x, min_y = :min_y,
                           max_x = :max_x, max_y = :max_y
                     WHERE [key] = :key
                    """,
                    dict(
                        **bounds,
                        **xy,
                        key=key,
                        boundary=self.serialise_geometry(geometry),
                    ),
                )
            if len(update) > 0:
                cur.execute(
                    """
                    UPDATE main
                       SET properties = json_patch(properties, :properties)
                     WHERE [key] = :key
                    """,
                    {
                        "key": key,
                        "properties": json.dumps(update, separators=(",", ":")),
                    },
                )
        self.con.commit()

    def remove_many(self, keys: Iterable[str]) -> None:
        cur = self.con.cursor()
        cur.executemany(
            "DELETE FROM main WHERE [key] = ?",
            ((i,) for i in keys),
        )
        self.con.commit()

    def __setitem__(
        self, key: str, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        properties = copy.deepcopy(properties)
        cur = self.con.cursor()
        cur.execute("BEGIN")
        bounds = dict(zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds))
        xy = dict(zip("xy", geometry.centroid.xy))
        cur.execute(
            """
            UPDATE main
               SET cx = :x, cy = :y, boundary = :boundary,
                   min_x = :min_x, min_y = :min_y, max_x = :max_x, max_y = :max_y
             WHERE [key] = :key
            """,
            dict(
                **xy,
                **bounds,
                key=key,
                boundary=self.serialise_geometry(geometry),
            ),
        )
        if len(properties) > 0:
            cur.execute(
                """
                UPDATE main
                   SET properties = json_patch(properties, :properties)
                 WHERE [key] = :key
                """,
                {
                    "key": key,
                    "properties": json.dumps(properties, separators=(",", ":")),
                },
            )
        self.con.commit()

    def __delitem__(self, key: str) -> None:
        cur = self.con.cursor()
        cur.execute(
            "DELETE FROM main WHERE [key] = ?",
            (key,),
        )
        self.con.commit()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        cur = self.con.cursor()
        cur.execute("SELECT id, boundary, properties FROM main")
        while True:
            rows = cur.fetchmany(100)
            if len(rows) == 0:
                break
            rows = (
                {
                    "key": key,
                    "geometry": geometry,
                    "properties": properties,
                }
                for key, (geometry, properties) in self.items()
            )
            df = df.append(pd.json_normalize(rows))
        return df.set_index("key")

    def features(self) -> Generator[Dict[str, Any], None, None]:
        return (
            {
                "type": "Feature",
                "geometry": geometry2feature(geometry),
                "properties": properties,
            }
            for geometry, properties in self.values()
        )

    def commit(self) -> None:
        return self.con.commit()

    def dump(self, fp: Union[Path, str, IO]) -> None:
        if hasattr(fp, "write"):
            fp = fp.name
        target = sqlite3.connect(fp)
        self.con.backup(target)

    def dumps(self) -> str:
        return "\n".join(self.con.iterdump())


class DictionaryStore(AnnotationStoreABC):
    """Pure python dictionary backed annotation store."""

    def __init__(self, connection: Union[Path, str] = ":memory:") -> None:
        super().__init__()
        self._features = {}
        self.connection = connection

    def append(
        self,
        geometry: Union[Geometry, Iterable[Geometry]],
        properties: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> int:
        if properties is None:
            properties = {}
        key = key or str(uuid.uuid4())
        self._features[key] = {
            "geometry": geometry,
            "properties": properties,
        }
        return key

    def update(self, key: str, update: Dict[str, Any]) -> None:
        geometry, properties = self[key]
        update = copy.copy(update)
        geometry = update.pop("geometry", geometry)
        properties.update(update)
        self[key] = (geometry, properties)

    def remove(self, key: str) -> None:
        del self._features[key]

    def features(self) -> Generator[Dict[str, Any], None, None]:
        return (
            {
                "type": "Feature",
                "geometry": geometry2feature(feature["geometry"]),
                "properties": feature["properties"],
            }
            for feature in self._features.values()
        )

    def __getitem__(self, key: str) -> Tuple[Geometry, Dict[str, Any]]:
        feature = self._features[key]
        return feature["geometry"], feature["properties"]

    def __setitem__(
        self, key: str, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        properties = dict(properties)
        self._features[key] = {"geometry": geometry, "properties": properties}

    def __contains__(self, key: int) -> bool:
        return key in self._features

    def __iter__(self) -> Iterable[int]:
        yield from self.keys()

    def items(self):
        for key, value in self._features.items():
            yield key, (value["geometry"], value["properties"])

    def __len__(self) -> int:
        return len(self._features)

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO]) -> "DictionaryStore":
        return cls.from_geojson(fp)

    def commit(self) -> None:
        if self.connection == ":memory:":
            warnings.warn("In-memory store. Nothing to commit.")
            return
        self.dump(self.connection)

    def dump(self, fp: Union[Path, str, IO]) -> None:
        return self.to_geojson(fp)

    def dumps(self) -> str:
        return self.to_geojson()

    def close(self) -> None:
        warnings.simplefilter("ignore")
        self.commit()
        warnings.resetwarnings()
