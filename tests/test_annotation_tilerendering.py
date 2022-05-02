"""tests for annotation rendering using
AnnotationRenderer and AnnotationTileGenerator
"""
import pytest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from pathlib import Path
from typing import List, Union
from shapely.geometry import Polygon, LineString
from shapely.geometry.point import Point

from tiatoolbox.tools.pyramid import AnnotationTileGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import wsireader
from tiatoolbox.utils.env_detection import running_on_travis
from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    SQLiteStore,
)
from tests.test_annotation_stores import cell_polygon


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a grid of fake cell boundary polygon annotations."""
    np.random.seed(0)
    return [cell_polygon(((i + 1) * 100, (j + 1) * 100)) for i, j in np.ndindex(5, 5)]


@pytest.fixture(scope="session")
def points_grid(spacing=60) -> List[Point]:
    """Generate a grid of fake point annotations."""
    np.random.seed(0)
    return [Point((600 + i * spacing, 600 + j * spacing)) for i, j in np.ndindex(7, 7)]


@pytest.fixture()
def fill_store(cell_grid, points_grid, spacing=60):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, Path],
    ):
        """fills store with random variety of annotations"""
        store = store_class(path)
        annotations = (
            [
                Annotation(cell, {"type": "cell", "prob": np.random.rand(1)[0]})
                for cell in cell_grid
            ]
            + [
                Annotation(point, {"type": "pt", "prob": np.random.rand(1)[0]})
                for point in points_grid
            ]
            + [
                Annotation(
                    LineString(((x, x + 500) for x in range(100, 400, 10))),
                    {"type": "line", "prob": np.random.rand(1)[0]},
                )
            ]
        )
        keys = store.append_many(annotations)
        return keys, store

    return _fill_store


def test_tile_generator_len(fill_store, tmp_path):
    """Test __len__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    assert len(tg) == (4 * 4) + (2 * 2) + 1


def test_tile_generator_iter(fill_store, tmp_path):
    """Test __iter__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    for tile in tg:
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)


@pytest.mark.skipif(running_on_travis(), reason="no display on travis.")
def test_show_generator_iter(fill_store, tmp_path):
    """Show tiles with example annotations (if not travis)"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer("prob")
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    for i, tile in enumerate(tg):
        if i > 5:
            break
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)
        plt.imshow(tile)
        plt.show()


def test_correct_number_rendered(fill_store, tmp_path):
    """test that the expected number of annotations are rendered"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)

    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 75  # expect 75 rendered objects


def test_correct_colour_rendered(fill_store, tmp_path):
    """test colour mapping"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(
        "type",
        {"cell": (255, 0, 0, 255), "pt": (0, 255, 0, 255), "line": (0, 0, 255, 255)},
    )
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 49  # expect 49 green objects
    _, num = label(np.array(thumb)[:, :, 0])
    assert num == 25  # expect 25 red objects
    _, num = label(np.array(thumb)[:, :, 2])
    assert num == 1  # expect 1 blue objects


def test_filter_by_expression(fill_store, tmp_path):
    """test filtering using a where expression"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(where='props["type"] == "cell"')
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 25  # expect 25 cell objects


def test_zoomed_out_rendering(fill_store, tmp_path):
    """test that the expected number of annotations are rendered"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_tile(1, 0, 0)
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 25  # expect 25 boxes in top left quadrant


def test_decimation(fill_store, tmp_path):
    """test decimation"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_tile(1, 1, 1)
    plt.imshow(thumb)
    plt.show()
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 16  # expect 16 pts in bottom right quadrant
