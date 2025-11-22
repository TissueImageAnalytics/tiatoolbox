"""Tests for NucleusDetector."""

import pathlib
import shutil

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _rm_dir(path):
    """Helper func to remove directory."""
    if pathlib.Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


def check_output(path):
    """Check NucleusDetector output."""
    store = SQLiteStore.open(path)
    # coordinates = store.to_dataframe()
    for item in store.values():
        geometry = item.geometry
        print(geometry.centroid)
        break
    # assert coordinates.x[0] == pytest.approx(53, abs=2)
    # assert coordinates.x[1] == pytest.approx(55, abs=2)
    # assert coordinates.y[0] == pytest.approx(107, abs=2)
    # assert coordinates.y[1] == pytest.approx(127, abs=2)


def test_nucleus_detector_wsi(remote_sample, tmp_path):
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    pretrained_model = "mapde-conic"

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="annotationstore",
        auto_get_mask=True,
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=tmp_path / "output",
    )

    check_output(tmp_path / "output" / "wsi4_512_512_svs.db")

    _rm_dir(tmp_path / "output")

    # nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
    # _ = nucleus_detector.predict(
    #     [mini_wsi_svs],
    #     mode="wsi",
    #     save_dir=tmp_path / "output",
    #     on_gpu=ON_GPU,
    #     ioconfig=ioconfig,
    # )

    # check_output(tmp_path / "output" / "0.locations.0.csv")
