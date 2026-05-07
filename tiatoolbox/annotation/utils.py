"""Utilities for working with annotation stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tiatoolbox.annotation.storage import Annotation, SQLiteStore

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence


def combine_annotation_stores(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    labels: Mapping[str | Path, str] | None = None,
    *,
    label_property: str = "source",
    overwrite: bool = False,
) -> Path:
    """Combine multiple SQLite annotation stores into one store.

    Args:
        input_paths:
            Paths to SQLite-backed ``.db`` annotation stores.
        output_path:
            Path to write the combined ``.db`` annotation store.
        labels:
            Optional mapping from input path to a label to write into each
            annotation's properties under ``label_property``. If omitted, each
            source store's filename stem is used.
        label_property:
            Name of the property used to record the source label.
        overwrite:
            Whether to replace an existing output store.

    Returns:
        Path:
            Path to the combined annotation store.

    """
    input_paths = [Path(path) for path in input_paths]
    if len(input_paths) == 0:
        msg = "At least one input annotation store path is required."
        raise ValueError(msg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            msg = f"Output annotation store already exists: {output_path}"
            raise FileExistsError(msg)
        output_path.unlink()

    labels_ = _normalise_labels(input_paths, labels)
    combined_store = SQLiteStore(auto_commit=False)

    for source_path in input_paths:
        source_store = SQLiteStore.open(source_path)
        source_label = labels_[source_path]
        annotations = []
        keys = []
        for key, annotation in source_store.items():
            properties = dict(annotation.properties)
            properties[label_property] = source_label
            annotations.append(Annotation(annotation.geometry, properties))
            keys.append(f"{source_label}:{key}")
        if annotations:
            combined_store.append_many(annotations, keys)

    combined_store.commit()
    combined_store.dump(output_path)
    return output_path


def _normalise_labels(
    input_paths: Iterable[Path],
    labels: Mapping[str | Path, str] | None,
) -> dict[Path, str]:
    """Normalise optional path labels to resolved ``Path`` keys."""
    input_paths = list(input_paths)
    if labels is None:
        return {path: path.stem for path in input_paths}

    labels_by_path = {Path(path): label for path, label in labels.items()}
    labels_by_resolved_path = {
        Path(path).resolve(): label for path, label in labels.items()
    }
    normalised = {}
    for path in input_paths:
        normalised[path] = labels_by_path.get(
            path,
            labels_by_resolved_path.get(path.resolve(), path.stem),
        )
    return normalised
