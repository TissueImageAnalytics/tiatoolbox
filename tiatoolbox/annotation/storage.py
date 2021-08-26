from abc import ABC
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Iterable,
    Optional,
    Any,
    IO,
    Callable,
)
from numbers import Number
import pickle
import io
import sqlite3
from itertools import zip_longest
from pathlib import Path
import hashlib


from shapely import speedups
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import mapping as geometry2feature
from shapely.geometry import shape as feature2geometry

from shapely import wkt
import numpy as np
import pandas as pd

try:
    import ujson as json  # pyright: reportMissingModuleSource=false
except ImportError:
    import json
import msgpack
import yaml
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
    "minX": int,
    "minY": int,
    "maxX": int,
    "maxY": int,
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
        """Create a 64 bit integer hash of a geometry object."""
        return int.from_bytes(
            hashlib.md5(geometry.wkb).digest()[:8], "big", signed=True
        )

    @staticmethod
    def _int_feature(feature: dict) -> Dict:
        """Convert feature coordinates to integers."""
        feature["coordinates"] = np.array(feature["coordinates"]).astype(int).tolist()
        return feature

    @classmethod
    def open(cls, io: Union[Path, str, IO]) -> "AnnotationStoreABC":
        """Load a store object from a path or file-like object."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the number of annotations in the store."""
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
        """Commit in memory changes to disk."""
        raise NotImplementedError()

    def dump(self, io: Union[Path, str, IO]) -> None:
        """Serialise a copy of the store to a file-like object."""
        raise NotImplementedError()

    def dumps(self) -> str:
        """Serialise and return a copy of store as a string."""
        raise NotImplementedError()

    def append(
        self,
        geometry: Union[Geometry, Iterable[Geometry]],
        class_: Optional[Union[int, Iterable[int]]] = None,
        **extra_properties: Dict[str, Union[Any, Iterable[Any]]],
    ) -> int:
        """Insert a new annotation, returning the index."""
        raise NotImplementedError()

    def update(self, index: int, **keys_and_values: Dict[str, Any]) -> None:
        """Update an annotation at given index.

        Extra key-word arguments are used to update properties.
        """
        raise NotImplementedError()

    def remove(self, indexes: Union[int, Iterable[int]]) -> None:
        """Remove annotation(s) by index."""
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        raise NotImplementedError()

    def __iter__(self) -> Iterable:
        raise NotImplementedError()

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        raise NotImplementedError()

    def __delitem__(self, index: int) -> None:
        raise NotImplementedError()

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        """Query with a geometry and return a list of annotation indexes."""
        raise NotImplementedError()

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        """Query with a geometry and return a list of annotation geometries."""
        raise NotImplementedError()


class SQLite3RTreeStore(AnnotationStoreABC):
    @classmethod
    def open(cls, io: Union[Path, str]) -> "SQLite3RTreeStore":
        store = SQLite3RTreeStore(io)
        return store

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
                minX, maxX, -- 1st dimension min, max
                minY, maxY  -- 2nd dimension min, max
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

    def _make_token(
        self, geometry, class_: int, properties: Dict, save_boundary: bool = True
    ) -> Dict:
        """Create token data dict for tokenised SQL transaction."""
        key = self.geometry_hash(geometry)
        if not save_boundary or geometry.geom_type == "Point":
            boundary = None
        else:
            boundary = self.serialise_geometry(geometry)
        datum = {
            "index": key,
            "boundary": boundary,
            "cx": int(geometry.centroid.x),
            "cy": int(geometry.centroid.y),
            "minX": geometry.bounds[0],
            "minY": geometry.bounds[1],
            "maxX": geometry.bounds[2],
            "maxY": geometry.bounds[3],
            "class": class_,
            "geom_type": geometry.geom_type,
            "properties": json.dumps(properties, separators=(",", ":")),
        }
        return datum

    def append(
        self,
        geometry: Union[Geometry, Iterable[Geometry]],
        class_: Optional[Union[int, Iterable[int]]] = None,
        **extra_properties: Dict[str, Union[Any, Iterable[Any]]],
    ) -> int:
        geometries_iter = self._iterfy(geometry)
        classes_iter = self._iterfy(class_)
        property_values_iter = iter(
            zip_longest(*(self._iterfy(x) for x in extra_properties.values()))
        )
        tokens = [
            self._make_token(
                geometry=geometry,
                class_=class_,
                properties={
                    k: v for k, v in zip(extra_properties.keys(), self._iterfy(props))
                },
            )
            for geometry, class_, props in zip_longest(
                geometries_iter, classes_iter, property_values_iter
            )
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
                :index, :minX, :maxX, :minY, :maxY
            )
            """,
            tokens,
        )
        self.con.commit()
        indexes = [token["index"] for token in tokens]
        if not isinstance(geometry, Iterable):
            return indexes[0]
        return indexes

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            minX, minY, maxX, maxY = query_geometry
        else:
            minX, minY, maxX, maxY = query_geometry.bounds
        cur.execute(
            """
        SELECT geometry.id FROM geometry, rtree
        WHERE rtree.id=geometry.id
        AND rtree.minX>=:minX AND rtree.maxX<=:maxX
        AND rtree.minY>=:minY AND rtree.maxY<=:maxY
        -- AND contained_in(geometry.boundary, :boundary)
        -- uncomment to do full polygon queries
        """,
            dict(
                # boundary=query_geometry,
                minX=minX,
                maxX=maxX,
                minY=minY,
                maxY=maxY,
            ),
        )
        boundaries = cur.fetchall()
        return [index for index, in boundaries]

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        cur = self.con.cursor()
        if isinstance(query_geometry, Iterable):
            minX, minY, maxX, maxY = query_geometry
        else:
            minX, minY, maxX, maxY = query_geometry.bounds
        cur.execute(
            """
        SELECT geometry.boundary, class, properties FROM geometry, rtree
        WHERE rtree.id=geometry.id
        AND rtree.minX>=:minX AND rtree.maxX<=:maxX
        AND rtree.minY>=:minY AND rtree.maxY<=:maxY
        -- AND contained_in(geometry.boundary, :boundary)
        -- uncomment to do full polygon queries
        """,
            dict(
                # boundary=query_geometry.wkb,
                minX=minX,
                maxX=maxX,
                minY=minY,
                maxY=maxY,
            ),
        )
        boundaries = cur.fetchall()
        return [self.deserialise_geometry(blob) for blob, in boundaries]

    def __len__(self) -> int:
        cur = self.con.cursor()
        cur.execute("SELECT Count(*) FROM geometry")
        (count,) = cur.fetchone()
        return count

    def __getitem__(self, index: int) -> Tuple[Geometry, Dict[str, Any]]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT boundary, class, properties FROM geometry
            WHERE geometry.id = :index
            """,
            dict(index=index),
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
            SELECT id, class, x, y, boundary, properties FROM geometry
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

    def update(
        self,
        index: Union[int, List[int]],
        **kwargs: Union[pd.DataFrame, Dict[str, Union[Any, List[Any]]]],
    ) -> None:
        index_iter = self._iterfy(index)
        kwarg_values_iter = iter(
            zip_longest(*(self._iterfy(x) for x in kwargs.values()), fillvalue=...)
        )
        cur = self.con.cursor()
        cur.execute("BEGIN")
        for i, kwarg_values in zip_longest(index_iter, kwarg_values_iter):
            properties = {
                k: v for k, v in zip(kwargs.keys(), kwarg_values) if v is not ...
            }
            geometry = properties.get("geometry")
            if "geometry" in properties:
                del properties["geometry"]
            if geometry is not None:
                bounds = dict(zip(("minX", "minY", "maxX", "maxY"), geometry.bounds))
                xy = dict(zip("xy", np.array(geometry.centroid)))
                cur.execute(
                    """
                    UPDATE geometry
                    SET x = :x, y = :y, boundary = :boundary
                    WHERE id = :index
                    """,
                    dict(
                        xy,
                        index=i,
                        boundary=self.serialise_geometry(geometry),
                    ),
                )
                cur.execute(
                    """
                    UPDATE rtree
                    SET minX = :minX, minY = :minY, maxX = :maxX, maxY = :maxY
                    WHERE id = :index
                    """,
                    dict(
                        bounds,
                        index=i,
                    ),
                )
            if len(properties) > 0:
                cur.execute(
                    """
                    UPDATE geometry
                    SET properties = json_patch(properties, :properties)
                    WHERE id = :index
                    """,
                    dict(
                        index=i,
                        properties=json.dumps(properties, separators=(",", ":")),
                    ),
                )
        self.con.commit()

    def __setitem__(
        self, index: int, record: Tuple[Geometry, Union[Dict[str, Any], pd.Series]]
    ) -> None:
        geometry, properties = record
        cur = self.con.cursor()
        cur.execute("BEGIN")
        class_ = properties.get("class")
        if "class" in properties:
            del properties["class"]
        bounds = dict(zip(("minX", "minY", "maxX", "maxY"), geometry.bounds))
        xy = dict(zip("xy", geometry.centroid.xy))
        cur.execute(
            """
            UPDATE geometry
            SET x = :x, y = :y, boundary = :boundary, class = :class_
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
            SET minX = :minX, minY = :minY, maxX = :maxX, maxY = :maxY
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
                dict(
                    index=index,
                    properties=json.dumps(properties, separators=(",", ":")),
                ),
            )
        self.con.commit()

    def __delitem__(self, index: int) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            DELETE FROM geometry WHERE id = ?
            """,
            (index,),
        )
        self.con.commit()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        cur = self.con.cursor()
        cur.execute("SELECT id, boundary, class, properties FROM geometry")
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
        df = df.set_index("index")
        return df

    def to_features(self, int_coords: bool = True) -> List:
        features = [
            dict(
                type="Feature",
                geometry=self._int_feature(geometry2feature(geometry))
                if int_coords
                else geometry2feature(geometry),
                properties=properties,
            )
            for geometry, properties in iter(self)
        ]
        return features

    def to_geodict(self, int_coords: bool = True) -> Dict:
        features = self.to_features(int_coords=int_coords)
        geodict = {
            "type": "FeatureCollection",
            "features": features,
        }
        return geodict


class DataFrameStore(AnnotationStoreABC):
    pass


class DictionaryStore(AnnotationStoreABC):
    """Dictionary based annotation store.

    A dictionary store serialises to and from a dictionary based file
    format such as JSON, YAML, and MessagePack. The GeoJSON layout for
    data is used with a top level "FeatureCollection" element
    containing and array of features, and each feature containing one
    geometry along with associated properties.

    Geometries are assumed to be unique and a hash of the
    well-known binary (WKB) representation is used as an index.

    All dictionary like stores hold an internal dataframe of annotations
    and associated properties. This can be accesse with the `data`
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
        self.data = pd.DataFrame()
        if dtypes is None:
            dtypes = {"class": "Int8"}
        self.data = pd.DataFrame(
            columns=["geometry", *dtypes.keys()],
        )
        self.dtypes = dtypes
        self.rtree = None

    @property
    def dtypes(self) -> Optional[dict]:
        """ "A dictionary mapping columns to data types."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value: dict) -> None:
        self._dtypes = value
        self.data = self.data.astype(self.dtypes)

    @classmethod
    def open(cls, io: Union[Path, str, IO], dtypes: Dict = None) -> "DictionaryStore":
        store = cls(dtypes=dtypes)
        feature_collection = cls._load(io)
        features = feature_collection["features"]
        for feat in features:
            geometry = feature2geometry(feat)
            properties = feat.get("properties", {})
            key = cls.geometry_hash(geometry)
            row = pd.DataFrame(properties, index=[key])
            row["geometry"] = geometry
            store.data = store.data.append(row)
        return store

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, no_copy=False) -> "DictionaryStore":
        store = cls()
        if no_copy:
            store.data = df
        else:
            store.data = df.copy()
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
            properties = {k: next(it) for k, it in properties_iters}
            if "class" in properties or "class_" in properties:
                raise Exception("Class may only be specified once.")
            cls = next(clses_iter, None)
            if cls is not None:
                properties.update({"class": cls})
            row = pd.DataFrame(dict(geometry=geom, **properties), index=[key])
            self.data = self.data.append(row)
            indexes.append(key)

        if not isinstance(geometry, Iterable) and len(indexes) == 1:
            return indexes[0]
        return indexes

    def __getitem__(self, index: int) -> Tuple[Geometry, Optional[dict]]:
        columns = self.data.loc[index]
        geometry = columns[0]
        properties = columns[1:]
        return geometry, properties

    def __delitem__(self, index: int) -> None:
        del self.data[index]

    def to_geodict(self, int_coords: bool = True, drop_na: bool = True) -> Dict:
        feature_collection = {
            "type": "FeatureCollection",
            "features": self.to_features(int_coords=int_coords, drop_na=drop_na),
        }
        return feature_collection

    def to_features(self, int_coords: bool = True, drop_na: bool = True) -> List[Dict]:
        return [
            dict(
                self._int_feature(geometry2feature(columns[0]))
                if int_coords
                else geometry2feature(columns[0]),
                properties=dict(
                    columns[1:].dropna()
                    if drop_na
                    else columns[1:].where(pd.notnull(columns[1:]), None)
                ),
            )
            for _, columns in self.data.iterrows()
        ]

    def to_dataframe(self) -> pd.DataFrame:
        return self.data.copy()

    def __iter__(self):
        for index, row in self.data.iterrows():
            yield index, dict(row.dropna())

    def dumps(self, int_coords: bool = True, drop_na: bool = True, **kwargs) -> str:
        return self._dumps(
            self.to_geodict(int_coords=int_coords, drop_na=drop_na), **kwargs
        )

    def dump(
        self, io: IO, int_coords: bool = True, drop_na: bool = True, **kwargs
    ) -> None:
        self._dump(
            self.to_geodict(int_coords=int_coords, drop_na=drop_na), io, **kwargs
        )


class GeoJSONStore(DictionaryStore):
    _load: Callable = staticmethod(json.load)
    _loads: Callable = staticmethod(json.loads)
    _dump: Callable = staticmethod(json.dump)
    _dumps: Callable = staticmethod(json.dumps)


class YAMLStore(DictionaryStore):
    _load: Callable = staticmethod(yaml.safe_load)
    _loads: Callable = staticmethod(yaml.safe_load)

    @staticmethod
    def _dump(dictionary: dict, io: IO):
        return yaml.safe_dump(dictionary, io, default_flow_style=None)

    @staticmethod
    def _dumps(dictionary: dict):
        return yaml.safe_dump(dictionary, default_flow_style=None)


class MsgPackStore(DictionaryStore):
    _load: Callable = staticmethod(msgpack.load)
    _loads: Callable = staticmethod(msgpack.loads)
    _dump: Callable = staticmethod(msgpack.dump)
    _dumps: Callable = staticmethod(msgpack.dumps)


class PickleDictStore(DictionaryStore):
    _load: Callable = staticmethod(pickle.load)
    _loads: Callable = staticmethod(pickle.loads)
    _dump: Callable = staticmethod(pickle.dump)
    _dumps: Callable = staticmethod(pickle.dumps)


class TableStore(DictionaryStore):
    @classmethod
    def open(cls, io: Union[Path, str, IO], dtypes: Dict = None) -> "TableStore":
        df = cls._load(io, dtypes=dtypes)
        store = cls().from_dataframe(df)
        return store

    def dump(self, io: IO) -> None:
        data = self.data.copy()
        data.geometry = data["geometry"].apply(self.serialise_geometry)
        self._dump(data, io)

    def dumps(self) -> str:
        data = self.data.copy()
        data.geometry = data["geometry"].apply(self.serialise_geometry)
        return self._dumps(data)

    @staticmethod
    def _dump(df: pd.DataFrame, io: IO):
        raise NotImplementedError()


class CSVStore(TableStore):
    _load: Callable = staticmethod(pd.read_csv)

    @staticmethod
    def _loads(string: str) -> pd.DataFrame:
        return pd.read_csv(string)

    @staticmethod
    def _dump(df: pd.DataFrame, io: IO):
        df.to_csv(io)

    @staticmethod
    def _dumps(df: pd.DataFrame) -> str:
        string_io = io.StringIO()
        df.to_csv(string_io, index_label=df.index.name)
        string_io.seek(0)
        return string_io.read()


class ADTStore(TableStore):
    @staticmethod
    def _load(io: IO) -> pd.DataFrame:
        return pd.read_table(
            io,
            sep=ASCII_UNIT_SEP,
            na_values=[ASCII_NULL],
            encoding="utf-8",
            compression="infer",
            lineterminator=ASCII_RECORD_SEP,
            decimal=".",
        )

    @staticmethod
    def _loads(string: str) -> pd.DataFrame:
        return pd.read_table(
            string,
            sep=ASCII_UNIT_SEP,
            na_values=[ASCII_NULL],
            encoding="utf-8",
            compression="infer",
            lineterminator=ASCII_RECORD_SEP,
            decimal=".",
        )

    @staticmethod
    def _dump(df: pd.DataFrame, io: IO):
        df.to_csv(
            io,
            sep=ASCII_UNIT_SEP,
            na_rep=ASCII_NULL,
            header=True,
            index=True,
            index_label=True,
            encoding="utf-8",
            compression="infer",
            line_terminator=ASCII_RECORD_SEP,
            date_format=ISO_8601_DATE_FORMAT,
            decimal=".",
        )

    @staticmethod
    def _dumps(df: pd.DataFrame) -> str:
        string_io = io.StringIO()
        df.to_csv(string_io, index_label=df.index.name)
        string_io.seek(0)
        return string_io.read()


class FeatherStore(TableStore):
    @staticmethod
    def _load(io: IO) -> pd.DataFrame:
        df = pd.read_feather(io)
        df = df.set_index("index")
        return df

    @staticmethod
    def _dump(df: pd.DataFrame, io: IO):
        df = df.reset_index()
        df.to_feather(io)


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
            minX = tables.Int32Col()
            minY = tables.Int32Col()
            maxX = tables.Int32Col()
            maxY = tables.Int32Col()
            properties = tables.StringCol(self.max_boundary_len)

        self.geometry_table = self.file_handle.create_table(
            self.root, "geometry", TablesGeometry
        )

    def append(
        self,
        geometry: Union[Geometry, Iterable[Geometry]],
        class_: Optional[Union[int, Iterable[int]]] = None,
        **extra_properties: Dict[str, Union[Any, Iterable[Any]]],
    ) -> int:
        geometry_iter = self._iterfy(geometry)
        class_iter = self._iterfy(class_)
        property_values_iter = iter(
            zip_longest(
                *(self._iterfy(x) for x in extra_properties.values()), fillvalue=...
            )
        )

        indexes = []
        for geom, cls, property_values in zip_longest(
            geometry_iter, class_iter, property_values_iter
        ):
            row = self.geometry_table.row
            row["index"] = self.geometry_hash(geom)
            boundary = self.serialise_geometry(geom)
            if len(boundary) > self.max_boundary_len:
                raise ValueError("Boundary > TablesStore.max_boundary_len")
            row["boundary"] = boundary
            row["minX"], row["minY"], row["maxX"], row["maxY"] = geom.bounds
            row["x"], row["y"] = np.array(geom.centroid)
            if cls is not None:
                row["class_"] = cls
            if property_values is not None:
                properties = json.dumps(
                    {
                        (k, v)
                        for k, v in zip(extra_properties.keys(), property_values)
                        if v is not ...
                    }
                )
                if len(properties) > self.max_properties_len:
                    raise ValueError("Properties > TablesStore.max_properties_len")
                row["properties"] = properties
            indexes.append(row["index"])
            row.append()

        self.geometry_table.flush()

        if isinstance(geometry, Iterable):
            return indexes
        return indexes[0]

    def query(self, query_geometry: QueryGeometry) -> List[Geometry]:
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        minX, minY, maxX, maxY = query_geometry.bounds
        query = self.geometry_table.where(
            f"(minX <= {maxX}) & (minY <= {maxY}) & (maxX >= {minX}) & (maxY >= {minY})"
        )
        results = []
        for row in query:
            geometry = self.deserialise_geometry(bytes.decode(row["boundary"]))
            if not query_geometry.intersects(geometry):
                continue
            # properties = json.loads(row["properties"])
            # properties.update({"class": row["class_"], "index": row["index"]})
            results.append(geometry)
        return results

    def query_index(self, query_geometry: QueryGeometry) -> List[int]:
        if isinstance(query_geometry, Iterable):
            query_geometry = Polygon.from_bounds(*query_geometry)
        minX, minY, maxX, maxY = query_geometry.bounds
        query = self.geometry_table.where(
            f"(minX <= {maxX}) & (minY <= {maxY}) & (maxX >= {minX}) & (maxY >= {minY})"
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
