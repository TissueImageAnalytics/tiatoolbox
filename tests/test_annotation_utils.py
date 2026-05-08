"""Tests for annotation utility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from shapely.geometry import Point

from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.annotation.utils import combine_annotation_stores

if TYPE_CHECKING:
    from pathlib import Path


def _write_store(
    path: Path,
    annotation: Annotation,
    key: str,
) -> None:
    """Write a one-annotation SQLite store."""
    store = SQLiteStore(path)
    store.append_many([annotation], keys=[key])
    store.close()


def test_combine_annotation_stores_preserves_annotations_and_labels(
    track_tmp_path: Path,
) -> None:
    """Test combining SQLite stores with explicit source labels."""
    store_a_path = track_tmp_path / "store-a.db"
    store_b_path = track_tmp_path / "store-b.db"
    output_path = track_tmp_path / "combined.db"
    _write_store(
        store_a_path,
        Annotation(Point(1, 2), {"class": 1}),
        "ann-a",
    )
    _write_store(
        store_b_path,
        Annotation(Point(3, 4), {"class": 2}),
        "ann-b",
    )

    result_path = combine_annotation_stores(
        [store_a_path, store_b_path],
        output_path,
        labels={store_a_path: "alpha", store_b_path.resolve(): "beta"},
        label_property="dataset",
    )

    assert result_path == output_path
    combined_store = SQLiteStore(output_path)
    assert set(combined_store.keys()) == {"alpha:ann-a", "beta:ann-b"}
    assert combined_store["alpha:ann-a"].geometry == Point(1, 2)
    assert combined_store["alpha:ann-a"].properties == {
        "class": 1,
        "dataset": "alpha",
    }
    assert combined_store["beta:ann-b"].geometry == Point(3, 4)
    assert combined_store["beta:ann-b"].properties == {
        "class": 2,
        "dataset": "beta",
    }
    combined_store.close()


def test_combine_annotation_stores_defaults_to_stems_and_checks_output(
    track_tmp_path: Path,
) -> None:
    """Test default labels, overwrite protection, and empty input validation."""
    source_path = track_tmp_path / "source.db"
    output_path = track_tmp_path / "combined.db"
    _write_store(source_path, Annotation(Point(5, 6), {"score": 0.5}), "ann")

    combine_annotation_stores([source_path], output_path)
    combined_store = SQLiteStore(output_path)
    assert set(combined_store.keys()) == {"source:ann"}
    assert combined_store["source:ann"].properties == {
        "score": 0.5,
        "source": "source",
    }
    combined_store.close()

    with pytest.raises(FileExistsError, match="already exists"):
        combine_annotation_stores([source_path], output_path)

    combine_annotation_stores([source_path], output_path, overwrite=True)

    with pytest.raises(ValueError, match="At least one"):
        combine_annotation_stores([], output_path, overwrite=True)
