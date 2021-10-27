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
import sqlite3
import uuid
import warnings
from abc import ABC
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

if speedups.available:
    speedups.enable()

Geometry = Union[Point, Polygon, LineString]
BBox = Tuple[Number, Number, Number, Number]
QueryGeometry = Union[BBox, Geometry]

RESERVED_PROPERTIES = {  # Currently unused
    "class": pd.Int8Dtype(),
    "class_": pd.Int8Dtype(),
    "properties": str,
    "boundary": bytes,
    "index": int,
    "x": int,
    "y": int,
    "min_x": int,
    "min_y": int,
    "max_x": int,
    "max_y": int,
    "type": int,
    "type_": int,
}

ASCII_FILE_SEP = "\x1c"
ASCII_GROUP_SEP = "\x1d"
ASCII_RECORD_SEP = "\x1e"
ASCII_UNIT_SEP = "\x1f"
ASCII_NULL = "\0"
ISO_8601_DATE_FORMAT = r"%Y-%m-%dT%H:%M:%S.%f%z"


class AnnotationStoreABC(ABC):
    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO]) -> "AnnotationStoreABC":
        """Load a store object from a path or file-like object."""
        raise NotImplementedError()

    @staticmethod
    def serialise_geometry(geometry: Geometry) -> Union[str, bytes]:
        """Serialise a geometry to a string or bytes."""
        return geometry.wkt

    @staticmethod
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
    ) -> int:
        """Insert a new annotation, returning the index."""
        if properties is None:
            properties = {}
        return self.append_many([geometry], [properties])[0]

    def append_many(
        self,
        geometries: Iterable[Geometry],
        properties_iter: Optional[Iterable[Dict[str, Any]]] = None,
        keys: Optional[Iterable[str]] = None,
    ) -> List[int]:
        indexes = []
        if properties_iter is None:
            properties_iter = itertools.repeat({})
        if keys is None:
            keys = (str(uuid.uuid4()) for _ in geometries)
        for geometry, properties in zip(geometries, properties_iter):
            key = self.append(geometry, properties)
            indexes.append(key)
        return indexes

    def update(self, index: int, update: Dict[str, Any]) -> None:
        """Update an annotation at given index."""
        return self.update_many([index], [update])

    def update_many(
        self, indexes: Iterable[int], updates: Iterable[Dict[str, Any]]
    ) -> None:
        for index, update in zip(indexes, updates):
            self.update(index, update)

    def remove(self, index: int) -> None:
        """Remove annotation by index."""
        self.remove_many([index])

    def remove_many(self, indexes: Iterable[int]) -> None:
        for index in indexes:
            self.remove(index)

    def __len__(self) -> int:
        """Return the number of annotations in the store."""
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        raise NotImplementedError()

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        raise NotImplementedError()

    def __delitem__(self, index: int) -> None:
        self.remove(index)

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

    def query(
        self, query_geometry: QueryGeometry
    ) -> List[Tuple[Geometry, Dict[str, Any]]]:
        if isinstance(query_geometry, tuple):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            (geometry, properties)
            for geometry, properties in self.values()
            if geometry.intersects(query_geometry)
        ]

    def iquery(self, query_geometry: QueryGeometry) -> List[int]:
        if isinstance(query_geometry, tuple):
            query_geometry = Polygon.from_bounds(*query_geometry)
        return [
            index
            for index, (geometry, _) in self.items()
            if geometry.intersects(query_geometry)
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
            properties = dict(row.filter(regex="^(?!geometry|index).*$"))
            store.append(geometry, properties)
        return store

    def to_dataframe(self) -> pd.DataFrame:
        features = (
            {"index": index, "geometry": geometry, "properties": properties}
            for index, (geometry, properties) in self.items()
        )
        return pd.json_normalize(features).set_index("index")

    def __del__(self) -> None:
        self.close()


class SQLiteStore(AnnotationStoreABC):
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

        # Register custom functions
        def wkb_intersects(candidate: Union[bytes, str], geometry_wkb: bytes) -> bool:
            """Check if one WKB/WKT intersects a geometry."""
            candidate = self.deserialise_geometry(candidate)
            return wkb.loads(geometry_wkb).intersects(candidate)

        try:
            self.con.create_function(
                "intersects", 2, wkb_intersects, deterministic=True
            )
        except TypeError:  # only Python >= 3.8 supports deterministic
            self.con.create_function("intersects", 2, wkb_intersects)

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
                +class INTEGER,          -- Class ID
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
        class_ = properties.get("class")
        if "class" in properties:
            properties = copy.copy(properties)
            del properties["class"]
        return {
            "key": key,
            "boundary": boundary,
            "cx": int(geometry.centroid.x),
            "cy": int(geometry.centroid.y),
            "min_x": geometry.bounds[0],
            "min_y": geometry.bounds[1],
            "max_x": geometry.bounds[2],
            "max_y": geometry.bounds[3],
            "class": class_,
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
                :class, :cx, :cy, :boundary, :properties
            )
            """,
            tokens,
        )
        self.con.commit()
        return [token["key"] for token in tokens]

    def iquery(self, query_geometry: QueryGeometry) -> List[int]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        min_x, min_y, max_x, max_y = query_geometry.bounds
        cur.execute(
            """
            SELECT [key]
              FROM main
             WHERE max_x >= :min_x
               AND min_x <= :max_x
               AND max_y >= :min_y
               AND min_y <= :max_y
               AND intersects(boundary, :intersector)
            """,
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "intersector": query_geometry.wkb,
            },
        )
        keys = cur.fetchall()
        return [key for key, in keys]

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        min_x, min_y, max_x, max_y = query_geometry.bounds
        cur.execute(
            """
            SELECT boundary, [class], properties
              FROM main
             WHERE max_x >= :min_x
               AND min_x <= :max_x
               AND max_y >= :min_y
               AND min_y <= :max_y
               AND intersects(boundary, :intersector)
            """,
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "intersector": query_geometry.wkb,
            },
        )
        rows = cur.fetchall()

        return [
            (
                self.deserialise_geometry(blob),
                {"class": class_, **json.loads(properties)},
            )
            for blob, class_, properties in rows
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
            SELECT boundary, [class], properties
              FROM main
             WHERE [key] = :key
            """,
            {"key": key},
        )
        boundary, class_, properties = cur.fetchone()
        if properties is None:
            properties = {}
        else:
            properties = json.loads(properties)
        if class_ is not None:
            properties.update({"class": class_})
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
            index = row[0]
            yield index

    def values(self) -> Iterable[Tuple[int, Tuple[Geometry, Dict[str, Any]]]]:
        for _, value in self.items():
            yield value

    def items(self) -> Iterable[Tuple[int, Tuple[Geometry, Dict[str, Any]]]]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT [key], [class], cx, cy, boundary, properties
              FROM main
            """
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            key, class_, cx, cy, boundary, properties = row
            if boundary is not None:
                geometry = self.deserialise_geometry(boundary)
            else:
                geometry = Point(cx, cy)
            properties = json.loads(properties)
            properties.update({"class": class_})
            yield key, (geometry, properties)

    def update_many(
        self, indexes: Iterable[int], updates: Iterable[Dict[str, Any]]
    ) -> None:
        cur = self.con.cursor()
        cur.execute("BEGIN")
        for key, update in zip(indexes, updates):
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

    def remove_many(self, indexes: Iterable[str]) -> None:
        cur = self.con.cursor()
        cur.executemany(
            "DELETE FROM main WHERE [key] = ?",
            ((i,) for i in indexes),
        )
        self.con.commit()

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        properties = copy.deepcopy(properties)
        cur = self.con.cursor()
        cur.execute("BEGIN")
        class_ = properties.pop("class", None)
        bounds = dict(zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds))
        xy = dict(zip("xy", geometry.centroid.xy))
        cur.execute(
            """
            UPDATE main
               SET cx = :x, cy = :y, boundary = :boundary, [class] = :class_,
                   min_x = :min_x, min_y = :min_y, max_x = :max_x, max_y = :max_y
             WHERE [key] = :key
            """,
            dict(
                **xy,
                **bounds,
                key=index,
                boundary=self.serialise_geometry(geometry),
                class_=class_,
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
                    "key": index,
                    "properties": json.dumps(properties, separators=(",", ":")),
                },
            )
        self.con.commit()

    def __delitem__(self, index: str) -> None:
        cur = self.con.cursor()
        cur.execute(
            "DELETE FROM main WHERE [key] = ?",
            (index,),
        )
        self.con.commit()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        cur = self.con.cursor()
        cur.execute("SELECT id, boundary, [class], properties FROM main")
        while True:
            rows = cur.fetchmany(100)
            if len(rows) == 0:
                break
            rows = (
                {
                    "key": index,
                    "geometry": geometry,
                    "properties": properties,
                }
                for index, (geometry, properties) in self.items()
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

    def update(self, index: int, update: Dict[str, Any]) -> None:
        feature = self[index]
        update = copy.copy(update)
        geometry = update.pop("geometry", feature[0])
        feature[1].update(update)
        self[index] = (geometry, feature[1])

    def remove(self, index: int) -> None:
        del self._features[index]

    def features(self) -> Generator[Dict[str, Any], None, None]:
        return (
            {
                "type": "Feature",
                "geometry": geometry2feature(feature["geometry"]),
                "properties": feature["properties"],
            }
            for feature in self._features.values()
        )

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        feature = self._features[index]
        return feature["geometry"], feature["properties"]

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        properties = dict(properties)
        self._features[index] = {"geometry": geometry, "properties": properties}

    def __contains__(self, key: int) -> bool:
        return key in self._features

    def __iter__(self) -> Iterable[int]:
        yield from self.keys()

    def items(self):
        for index, value in self._features.items():
            yield index, (value["geometry"], value["properties"])

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
