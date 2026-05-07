"""Module initialisation."""

from tiatoolbox.annotation import dsl, storage
from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    DictionaryStore,
    SQLiteStore,
)
from tiatoolbox.annotation.utils import combine_annotation_stores

__all__ = [
    "Annotation",
    "AnnotationStore",
    "DictionaryStore",
    "SQLiteStore",
    "combine_annotation_stores",
]
