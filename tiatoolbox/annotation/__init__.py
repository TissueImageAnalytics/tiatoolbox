"""Module initialisation."""
from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    DictionaryStore,
    SQLiteStore,
)

__all__ = ["AnnotationStore", "SQLiteStore", "DictionaryStore", "Annotation"]
