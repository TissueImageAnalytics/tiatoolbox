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
import hashlib
import itertools
import sqlite3
from abc import ABC
from io import StringIO
from numbers import Number
from pathlib import Path
from typing import IO, Any, Dict, Iterable, List, Optional, Tuple, Union
import copy

import numpy as np
import pandas as pd
from shapely import speedups, wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import mapping as geometry2feature

try:
    import ujson as json  # pyright: reportMissingModuleSource=false
except ImportError:
    import json

import tables

if speedups.available:
    speedups.enable()

Geometry = Union[Point, Polygon, LineString]
BBox = Tuple[Number, Number, Number, Number]
QueryGeometry = Union[BBox, Geometry]

RESERVED_PROPERTIES = {
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
    @staticmethod
    def geometry_hash(geometry: Geometry) -> int:
        """Create a 64 bit integer hash of a geometry object.

        Args:
            geometry (Geometry): Shapely geometry object to hash.

        Returns:
            int: 64 bit hash
        """
        return int.from_bytes(
            hashlib.md5(geometry.wkb).digest()[:8], "big", signed=True
        )

    @staticmethod
    def _int_feature(feature: Dict) -> Dict:
        """Convert feature coordinates to integers.

        Args:
            feature (dict): GeoJSON style feature dictionary with
                keys: 'type', 'coordinates', an optionally 'properties'.

        Returns:
            dict: Feature dictionary with coordinates as integers.
        """
        feature["coordinates"] = np.array(feature["coordinates"]).astype(int).tolist()
        return feature

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO]) -> "AnnotationStoreABC":
        """Load a store object from a path or file-like object."""
        raise NotImplementedError()

    @staticmethod
    def _iterfy(obj: Union[Any, Iterable[Any]]) -> Iterable[Any]:
        """Make passed objects into an iterable.

        Expected behaviour:
        - Passing None returns a zero length iterable.
        - Passing an interable returns the passed iterable unchanged.
        - Passing a non-iterable (e.g. and int) returns a single element iterator.

        """
        if obj is None:
            return iter(())
        if isinstance(obj, Iterable):
            return obj
        return iter((obj,))

    @staticmethod
    def serialise_geometry(geometry: Geometry) -> Union[str, bytes]:
        """Serialise a geometry to a string or bytes."""
        return geometry.wkt

    @staticmethod
    def deserialise_geometry(data: Union[str, bytes]):
        """Deserialise a geometry from a string or bytes."""
        if isinstance(data, str):
            return wkt.loads(data)
        return wkt.load(data)

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
        properties: Dict[str, Any] = None,
    ) -> int:
        """Insert a new annotation, returning the index."""
        if properties is None:
            properties = {}
        return self.append_many([geometry], [properties])[0]

    def append_many(
        self,
        geometries: Iterable[Geometry],
        properties_iter: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> List[int]:
        indexes = []
        if properties_iter is None:
            properties_iter = itertools.repeat({})
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

    def __iter__(self) -> Iterable:
        raise NotImplementedError()

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        """Query with a geometry and return a list of annotation indexes."""
        raise NotImplementedError()

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        """Query with a geometry and return a list of annotation geometries."""
        raise NotImplementedError()

    def to_features(self, int_coords: bool = False, drop_na: bool = True) -> List[Dict]:
        """Return anotations as a list of geoJSON features.

        Args:
            int_coords(bool): Make coordinates intergers. Defaults to
                True.
            drop_na(bool): Don't include keys for None/NaN values.
                Defaults to True.

        Returns:
            list: List of features as dictionaries.

        """
        raise NotImplementedError()

    def to_geodict(self, int_coords: bool = False, drop_na: bool = True) -> Dict:
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
            "features": self.to_features(int_coords=int_coords, drop_na=drop_na),
        }

    def to_geojson(self, fp: Optional[IO] = None) -> Union[str, None]:
        """Serialise the store to geoJSON.

        Args:
             fp (IO): A file-like object supporting `.read`. Defaults to
                None which returns geoJSON as a string.

        Returns:
            None or str: None if writing to file or the geoJSON string.

        """
        if fp is not None:
            json.dump(self.to_geodict(), fp)
            return
        return json.dumps(self.to_geodict())


class SQLite3RTreeStore(AnnotationStoreABC):
    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str]) -> "SQLite3RTreeStore":
        return SQLite3RTreeStore(fp)

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
        if exists:
            return
        cur = self.con.cursor()

        # Create tables for geometry and RTree index
        cur.execute(
            """
            CREATE VIRTUAL TABLE rtree
            USING rtree_i32(
                id,      -- Integer primary key
                min_x, max_x, -- 1st dimension min, max
                min_y, max_y  -- 2nd dimension min, max
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE geometry(
                id INTEGER PRIMARY KEY, -- 64 bit truncated MD5 hash of boundary WKB
                objtype TEXT,              -- Object type
                class INTEGER,             -- Class ID
                x INTEGER NOT NULL,        -- X centroid
                y INTEGER NOT NULL,        -- Y centroid
                boundary BLOB,             -- Detailed boundary
                properties TEXT,           -- JSON properties
                FOREIGN KEY(id) REFERENCES rtree(id) ON DELETE CASCADE
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

    def __del__(self) -> None:
        self.con.close()

    def _make_token(self, geometry: Geometry, properties: Dict) -> Dict:
        """Create token data dict for tokenised SQL transaction."""
        key = self.geometry_hash(geometry)
        if geometry.geom_type == "Point":
            boundary = None
        else:
            boundary = self.serialise_geometry(geometry)
        class_ = properties.get("class")
        if "class" in properties:
            properties = copy.copy(properties)
            del properties["class"]
        return {
            "index": key,
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
    ) -> List[int]:
        if properties_iter is None:
            properties_iter = itertools.repeat({})
        tokens = [
            self._make_token(
                geometry=geometry,
                properties=properties,
            )
            for geometry, properties in zip(geometries, properties_iter)
        ]
        cur = self.con.cursor()
        cur.executemany(
            """
            INSERT INTO geometry VALUES(
                :index, :geom_type, :class, :cx, :cy, :boundary, :properties
            )
            """,
            tokens,
        )
        cur.executemany(
            """
            INSERT INTO rtree VALUES(
                :index, :min_x, :max_x, :min_y, :max_y
            )
            """,
            tokens,
        )
        self.con.commit()
        return [token["index"] for token in tokens]

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            min_x, min_y, max_x, max_y = query_geometry
        else:
            min_x, min_y, max_x, max_y = query_geometry.bounds
        cur.execute(
            """
            SELECT geometry.id
              FROM geometry, rtree
             WHERE rtree.id = geometry.id
               AND rtree.min_x >= :min_x
               AND rtree.max_x <= :max_x
               AND rtree.min_y >= :min_y
               AND rtree.max_y <= :max_y
            """,
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            },
        )
        boundaries = cur.fetchall()
        return [index for index, in boundaries]

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            min_x, min_y, max_x, max_y = query_geometry
        else:
            min_x, min_y, max_x, max_y = query_geometry.bounds
        cur.execute(
            """
            SELECT geometry.boundary, [class], properties
              FROM geometry, rtree
             WHERE rtree.id = geometry.id
               AND rtree.min_x >= :min_x
               AND rtree.max_x <= :max_x
               AND rtree.min_y >= :min_y
               AND rtree.max_y <= :max_y
            """,
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            },
        )
        boundaries = cur.fetchall()
        return [self.deserialise_geometry(blob) for blob, in boundaries]

    def __len__(self) -> int:
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM geometry")
        (count,) = cur.fetchone()
        return count

    def __contains__(self, key: int) -> bool:
        cur = self.con.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM geometry WHERE id = ?)", (key,))
        return cur.fetchone()[0] == 1

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT boundary, [class], properties
              FROM geometry
             WHERE geometry.id = :index
            """,
            {"index": index},
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

    def __iter__(self) -> Iterable:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT id, [class], x, y, boundary, properties
              FROM geometry
            """
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            index, class_, x, y, boundary, properties = row
            if boundary is not None:
                geometry = self.deserialise_geometry(boundary)
            else:
                geometry = Point(x, y)
            properties = json.loads(properties)
            properties.update({"class": class_, "index": index})
            yield geometry, properties

    def update_many(
        self, indexes: Iterable[int], updates: Iterable[Dict[str, Any]]
    ) -> None:
        cur = self.con.cursor()
        cur.execute("BEGIN")
        for index, update in zip(indexes, updates):
            update = copy.copy(update)
            geometry = update.pop("geometry", None)
            if geometry is not None:
                bounds = dict(
                    zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds)
                )
                xy = dict(zip("xy", np.array(geometry.centroid)))
                cur.execute(
                    """
                    UPDATE geometry
                       SET x = :x, y = :y, boundary = :boundary
                     WHERE id = :index
                    """,
                    dict(
                        xy,
                        index=index,
                        boundary=self.serialise_geometry(geometry),
                    ),
                )
                cur.execute(
                    """
                    UPDATE rtree
                       SET min_x = :min_x, min_y = :min_y,
                           max_x = :max_x, max_y = :max_y
                     WHERE id = :index
                    """,
                    dict(
                        bounds,
                        index=index,
                    ),
                )
            if len(update) > 0:
                cur.execute(
                    """
                    UPDATE geometry
                       SET properties = json_patch(properties, :properties)
                     WHERE id = :index
                    """,
                    {
                        "index": index,
                        "properties": json.dumps(update, separators=(",", ":")),
                    },
                )
        self.con.commit()

    def remove_many(self, indexes: Iterable[int]) -> None:
        cur = self.con.cursor()
        cur.executemany(
            "DELETE FROM geometry WHERE id = ?",
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
        class_ = properties.get("class")
        if "class" in properties:
            del properties["class"]
        bounds = dict(zip(("min_x", "min_y", "max_x", "max_y"), geometry.bounds))
        xy = dict(zip("xy", geometry.centroid.xy))
        cur.execute(
            """
            UPDATE geometry
               SET x = :x, y = :y, boundary = :boundary, [class] = :class_
             WHERE id = :index
            """,
            dict(
                xy,
                index=index,
                boundary=self.serialise_geometry(geometry),
                class_=class_,
            ),
        )
        cur.execute(
            """
            UPDATE rtree
               SET min_x = :min_x, min_y = :min_y, max_x = :max_x, max_y = :max_y
             WHERE id = :index
            """,
            dict(
                bounds,
                index=index,
            ),
        )
        if len(properties) > 0:
            cur.execute(
                """
                UPDATE geometry
                   SET properties = json_patch(properties, :properties)
                 WHERE id = :index
                """,
                {
                    "index": index,
                    "properties": json.dumps(properties, separators=(",", ":")),
                },
            )
        self.con.commit()

    def __delitem__(self, index: int) -> None:
        cur = self.con.cursor()
        cur.execute(
            "DELETE FROM geometry WHERE id = ?",
            (index,),
        )
        self.con.commit()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        cur = self.con.cursor()
        cur.execute("SELECT id, boundary, [class], properties FROM geometry")
        while True:
            rows = cur.fetchmany(1000)
            if len(rows) == 0:
                break
            rows = [
                dict(
                    geometry=geometry,
                    **properties,
                )
                for geometry, properties in iter(self)
            ]
            df = df.append(rows)
        return df.set_index("index")

    def to_features(self, int_coords: bool = True) -> List:
        return [
            {
                "type": "Feature",
                "geometry": self._int_feature(geometry2feature(geometry))
                if int_coords
                else geometry2feature(geometry),
                "properties": properties,
            }
            for geometry, properties in iter(self)
        ]

    def to_geodict(self, int_coords: bool = True) -> Dict:
        features = self.to_features(int_coords=int_coords)
        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def commit(self) -> None:
        return self.con.commit()

    def dump(self, fp: Union[Path, str, IO]) -> None:
        if isinstance(fp, IO):
            fp = fp.name
        target = sqlite3.connect(fp)
        self.con.backup(target)

    def dumps(self) -> str:
        raise NotImplementedError()


class DictionaryStore(AnnotationStoreABC):
    def __init__(self) -> None:
        super().__init__()
        self.features = {}

    def append(
        self, geometry: Union[Geometry, Iterable[Geometry]], properties: Dict[str, Any]
    ) -> int:
        key = self.geometry_hash(geometry)
        self.features[key] = {
            "geometry": geometry,
            "properties": properties,
        }
        return key

    def update(self, index: int, update: Dict[str, Any]) -> None:
        feature = self[index]
        update = copy.copy(update)
        if "geometry" in update:
            feature["geometry"]
        feature["properties"] = update

    def remove(self, index: int) -> None:
        del self.features[index]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, no_copy=False) -> "DictionaryStore":
        store = cls()
        for index, row in df.iterrows():
            geometry = row["geometry"]
            properties = dict(row.loc[:, row.columns != "geometry"])
            feature = {
                "geometry": geometry,
                "properties": properties,
            }
            store.features[index] = feature
        return store

    def to_features(self, int_coords: bool = False, drop_na: bool = True) -> List[Dict]:
        return [{"type": "Feature", **feature} for feature in self.features]

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        feature = self.features[index]
        return feature["geometry"], feature["properties"]

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        properties = dict(properties)
        self.features[index] = {"geometry": geometry, "properties": properties}

    def __contains__(self, key: int) -> bool:
        return key in self.features

    def __iter__(self):
        for value in self.features.values():
            yield value["geometry"], value["propeties"]


class DataFrameStore(AnnotationStoreABC):
    """DataFrame backed annotation store.

    Geometries are assumed to be unique and a hash of the
    well-known binary (WKB) representation is used as the index.

    A DataFrameStore holds an internal dataframe of annotations
    and associated properties. This can be accesse with the `dataframe`
    attribute.

    Attributes:
        data (DataFrame): Internal dataframe containing geometries and
            associated properties. Note that this is a reference to the
            underlying data which may mutate. Use `to_dataframe` to get
            a copy.
        dtypes (dict): A mapping of property (column) names to pandas
            data types. Defaults to {"class": "Int8"}. Setting this will
            update the internal dataframe columns. See pandas types for
            more information:
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html

    """

    def __init__(
        self,
        dtypes: dict = None,
    ):
        super().__init__()
        self.dataframe = pd.DataFrame()
        if dtypes is None:
            dtypes = {"class": "Int8"}
        self.dataframe = pd.DataFrame(
            columns=["geometry", *dtypes.keys()],
        )
        self.dtypes = dtypes
        self.rtree = None

    @property
    def dtypes(self) -> Optional[dict]:
        """A dictionary mapping columns to data types."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value: dict) -> None:
        self._dtypes = value
        self.dataframe = self.dataframe.astype(self.dtypes)

    @classmethod  # noqa: A003
    def open(cls, fp: Union[Path, str, IO], dtypes: Dict = None) -> "DataFrameStore":
        df = cls._load(fp, dtypes=dtypes)
        return cls().from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, no_copy=False) -> "DataFrameStore":
        store = cls()
        if no_copy:
            store.dataframe = df
        else:
            store.dataframe = df.copy()
        store.dtypes = dict(df.dtypes)
        return store

    def append(
        self,
        geometry: Union[Geometry, Iterable[Geometry]],
        class_: Optional[Union[int, Iterable[int]]] = None,
        **extra_properties: Optional[Dict[str, Union[Any, Iterable[Any]]]],
    ) -> int:
        geometrys_iter = self._iterfy(geometry)
        clses_iter = self._iterfy(class_)
        properties_iters = {(k, self._iterfy(v)) for k, v in extra_properties.items()}
        indexes = []

        for geom in geometrys_iter:
            key = self.geometry_hash(geom)
            properties = {k: next(it, {}) for k, it in properties_iters}
            if "class" in properties or "class_" in properties:
                raise Exception("Class may only be specified once.")
            cls = next(clses_iter, None)
            if cls is not None:
                properties.update({"class": cls})
            row = pd.DataFrame(dict(geometry=geom, **properties), index=[key])
            self.dataframe = self.dataframe.append(row, verify_integrity=True)
            indexes.append(key)

        if not isinstance(geometry, Iterable) and len(indexes) == 1:
            return indexes[0]
        return indexes

    def __getitem__(self, index: int) -> Tuple[Geometry, Optional[dict]]:
        columns = self.dataframe.loc[index]
        geometry = columns[0]
        properties = columns[1:]
        return geometry, properties

    def __delitem__(self, index: int) -> None:
        del self.dataframe[index]

    def to_features(self, int_coords: bool = True, drop_na: bool = True) -> List[Dict]:
        return [
            dict(
                self._int_feature(geometry2feature(columns[0]))
                if int_coords
                else geometry2feature(columns[0]),
                properties=dict(
                    columns[1:].dropna()
                    if drop_na
                    else columns[1:].where(pd.notna(columns[1:]), None)
                ),
            )
            for _, columns in self.dataframe.iterrows()
        ]

    def to_dataframe(self) -> pd.DataFrame:
        return self.dataframe.copy()

    @classmethod
    def from_csv(cls, fp: Union[IO, str]) -> "DataFrameStore":
        if isinstance(fp, str):
            fp = StringIO(fp)
        store = cls().from_dataframe(pd.read_csv(fp))
        return store

    def to_csv(self, fp: Optional[IO] = None) -> Union[str, None]:
        return self.dataframe.to_csv(fp)

    @classmethod
    def from_adt(cls, fp: Union[IO, str]) -> "DataFrameStore":
        if isinstance(fp, str):
            fp = StringIO(fp)
        store = cls().from_dataframe(
            pd.read_csv(
                fp,
                sep=ASCII_UNIT_SEP,
                na_values=[ASCII_NULL],
                encoding="utf-8",
                compression="infer",
                lineterminator=ASCII_RECORD_SEP,
                decimal=".",
            )
        )
        return store

    def to_adt(self, fp: Optional[IO] = None) -> Union[str, None]:
        """Serialise to ASCII Delimited Text (ADT)."""
        return self.dataframe.to_csv(
            fp,
            sep=ASCII_UNIT_SEP,
            na_rep=ASCII_NULL,
            header=True,
            index=True,
            encoding="utf-8",
            compression="infer",
            line_terminator=ASCII_RECORD_SEP,
            date_format=ISO_8601_DATE_FORMAT,
            decimal=".",
        )

    def __iter__(self):
        for index, row in self.dataframe.iterrows():
            yield index, dict(row.dropna())

    def dumps(self, int_coords: bool = True, drop_na: bool = True, **kwargs) -> str:
        return self._dumps(
            self.to_geodict(int_coords=int_coords, drop_na=drop_na), **kwargs
        )

    def dump(
        self, fp: IO, int_coords: bool = True, drop_na: bool = True, **kwargs
    ) -> None:
        self._dump(
            self.to_geodict(int_coords=int_coords, drop_na=drop_na), fp, **kwargs
        )


class PyTablesStore(AnnotationStoreABC):
    max_boundary_len = 1024
    max_properties_len = 1024

    def __init__(self, file_path: Union[Path, str]) -> None:
        self.file_handle = tables.open_file(file_path, mode="w")
        self.root = self.file_handle.root

        class TablesGeometry(tables.IsDescription):
            index = tables.Int64Col()
            boundary = tables.StringCol(self.max_boundary_len)
            class_ = tables.Int8Col()
            x = tables.Int32Col()
            y = tables.Int32Col()
            min_x = tables.Int32Col()
            min_y = tables.Int32Col()
            max_x = tables.Int32Col()
            max_y = tables.Int32Col()
            properties = tables.StringCol(self.max_boundary_len)

        self.geometry_table = self.file_handle.create_table(
            self.root, "geometry", TablesGeometry
        )

    def append_many(
        self,
        geometries: Iterable[Geometry],
        properties_iter: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> List[int]:
        indexes = []
        if properties_iter is None:
            properties_iter = itertools.repeat({})
        for geometry, properties in zip(geometries, properties_iter):
            row = self.geometry_table.row
            row["index"] = self.geometry_hash(geometry)
            boundary = self.serialise_geometry(geometry)
            if len(boundary) > self.max_boundary_len:
                raise ValueError("Boundary > TablesStore.max_boundary_len")
            row["boundary"] = boundary
            row["min_x"], row["min_y"], row["max_x"], row["max_y"] = geometry.bounds
            row["x"], row["y"] = np.array(geometry.centroid)
            row["class_"] = properties.pop("class", -1)
            properties = json.dumps(properties, separators=(",", ":"))
            if len(properties) > self.max_properties_len:
                raise ValueError("Properties > TablesStore.max_properties_len")
            row["properties"] = properties
            indexes.append(row["index"])
            row.append()

        self.geometry_table.flush()
        return indexes

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        min_x, min_y, max_x, max_y = query_geometry.bounds
        query = self.geometry_table.where(
            f"  (min_x <= {max_x})"
            f"& (min_y <= {max_y})"
            f"& (max_x >= {min_x})"
            f"& (max_y >= {min_y})"
        )
        results = []
        for row in query:
            geometry = self.deserialise_geometry(bytes.decode(row["boundary"]))
            if not query_geometry.intersects(geometry):
                continue
            results.append(geometry)
        return results

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        min_x, min_y, max_x, max_y = query_geometry.bounds
        query = self.geometry_table.where(
            f"  (min_x <= {max_x})"
            f"& (min_y <= {max_y})"
            f"& (max_x >= {min_x})"
            f"& (max_y >= {min_y})"
        )
        results = []
        for row in query:
            geometry = self.deserialise_geometry(bytes.decode(row["boundary"]))
            if not query_geometry.intersects(geometry):
                continue
            index = row["index"]
            results.append(index)
        return results

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        rows = list(self.geometry_table.where(f"index == {index}"))
        if len(rows) < 1:
            raise IndexError()
        if len(rows) > 1:
            raise Exception("Index collision. Multiple rows with matching index.")
        row = rows[0]
        geometry = self.deserialise_geometry(bytes.decode(row["boundary"]))
        properties_str = row["properties"]
        if len(properties_str) == 0:
            properties_str = "{}"
        properties = json.loads(properties_str)
        properties.update({"class": row["class_"]})
        return geometry, properties

    def __len__(self) -> int:
        return len(self.geometry_table)

    def __iter__(self) -> Iterable:
        for row in self.geometry_table.iterrows():
            geometry = self.deserialise_geometry(bytes.decode(row["boundary"]))
            properties = json.loads(row["properties"] or "{}")
            class_ = row["class_"]
            properties.update({"class": class_})
            yield geometry, properties

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.geometry_table.read())
        dtypes = {
            name: type_
            for name, type_ in RESERVED_PROPERTIES.items()
            if name in df.columns
        }
        df = df.astype(dtypes)
        df.set_index("index", inplace=True)
        df.rename(columns={"class_": "class", "boundary": "geometry"}, inplace=True)
        df.loc[:, "geometry"] = df.geometry.str.decode("utf-8")
        df.loc[:, "properties"] = df.properties.str.decode("utf-8")
        df.loc[:, "geometry"] = df.geometry.apply(self.deserialise_geometry)
        return df
