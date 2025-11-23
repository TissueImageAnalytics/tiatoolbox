"""Tests for NucleusDetector."""

import pathlib
import shutil
from collections.abc import Callable

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _rm_dir(path: pathlib.Path) -> None:
    """Helper func to remove directory."""
    if pathlib.Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


def check_output(path: pathlib.Path) -> None:
    """Check NucleusDetector output."""
    store = SQLiteStore.open(path)
    assert len(store.values()) == 281


def test_nucleus_detector_wsi(remote_sample: Callable, tmp_path: pathlib.Path) -> None:
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    pretrained_model = "mapde-conic"

    save_dir = tmp_path 

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="annotationstore",
        auto_get_mask=True,
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
    )

    check_output(save_dir / "wsi4_512_512.db")

    _rm_dir(save_dir)