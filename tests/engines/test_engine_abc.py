"""Test tiatoolbox.models.engine.engine_abc."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest

from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.engine_abc import EngineABC, prepare_engines_save_dir

if TYPE_CHECKING:
    import torch.nn


class TestEngineABC(EngineABC):
    """Test EngineABC."""

    def __init__(
        self: TestEngineABC,
        model: str | torch.nn.Module,
        verbose: bool | None = None,
    ) -> NoReturn:
        """Test EngineABC init."""
        super().__init__(model=model, verbose=verbose)

    def infer_wsi(self: EngineABC) -> NoReturn:
        """Test infer_wsi."""
        ...  # dummy function for tests.

    def post_process_wsi(self: EngineABC) -> NoReturn:
        """Test post_process_wsi."""
        ...  # dummy function for tests.

    def pre_process_wsi(self: EngineABC) -> NoReturn:
        """Test pre_process_wsi."""
        ...  # dummy function for tests.


def test_engine_abc() -> NoReturn:
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC with abstract methods*",
    ):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq


def test_engine_abc_incorrect_model_type() -> NoReturn:
    """Test EngineABC initialization with incorrect model type."""
    with pytest.raises(
        TypeError,
        match=r".*missing 1 required positional argument: 'model'",
    ):
        TestEngineABC()  # skipcq

    with pytest.raises(
        TypeError,
        match="Input model must be a string or 'torch.nn.Module'.",
    ):
        # Can't instantiate abstract class with abstract methods
        TestEngineABC(model=1)


def test_incorrect_ioconfig() -> NoReturn:
    """Test EngineABC initialization with incorrect ioconfig."""
    import torchvision.models as torch_models

    model = torch_models.resnet18()
    engine = TestEngineABC(model=model)
    with pytest.raises(
        ValueError,
        match=r".*provide a valid ModelIOConfigABC.*",
    ):
        engine.run(images=[], masks=[], ioconfig=None)


def test_pretrained_ioconfig() -> NoReturn:

    """Test EngineABC initialization with ioconfig from
    the pretrained model in the toolbox.
    """

    # pre-trained model as a string
    pretrained_model = "alexnet-kather100k"

    """Test engine run without ioconfig"""
    eng = TestEngineABC(model=pretrained_model)
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        patch_mode=True,
        ioconfig=None,
    )
    assert "predictions" in out
    assert "labels" not in out


def test_prepare_engines_save_dir(
    tmp_path: pytest.TempPathFactory,
    caplog: pytest.LogCaptureFixture,
) -> NoReturn:
    """Test prepare save directory for engines."""
    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output",
        patch_mode=True,
        len_images=1,
        overwrite=False,
    )

    assert out_dir == tmp_path / "patch_output"
    assert out_dir.exists()

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output",
        patch_mode=True,
        len_images=1,
        overwrite=True,
    )

    assert out_dir == tmp_path / "patch_output"
    assert out_dir.exists()

    out_dir = prepare_engines_save_dir(
        save_dir=None,
        patch_mode=True,
        len_images=1,
        overwrite=False,
    )
    assert out_dir is None

    with pytest.raises(
        OSError,
        match=r".*More than 1 WSIs detected but there is no save directory provided.*",
    ):
        _ = prepare_engines_save_dir(
            save_dir=None,
            patch_mode=False,
            len_images=2,
            overwrite=False,
        )

    out_dir = prepare_engines_save_dir(
        save_dir=None,
        patch_mode=False,
        len_images=1,
        overwrite=False,
    )

    assert out_dir == Path.cwd()

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_single_output",
        patch_mode=False,
        len_images=1,
        overwrite=False,
    )

    assert out_dir == tmp_path / "wsi_single_output"
    assert out_dir.exists()
    assert r"When providing multiple whole-slide images / tiles" not in caplog.text

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_multiple_output",
        patch_mode=False,
        len_images=2,
        overwrite=False,
    )

    assert out_dir == tmp_path / "wsi_multiple_output"
    assert out_dir.exists()
    assert r"When providing multiple whole slide images" in caplog.text


def test_engine_initalization() -> NoReturn:
    """Test engine initialization."""
    with pytest.raises(
        TypeError,
        match="Input model must be a string or 'torch.nn.Module'.",
    ):
        _ = TestEngineABC(model=0)

    eng = TestEngineABC(model="alexnet-kather100k")
    assert isinstance(eng, EngineABC)
    model = CNNModel("alexnet", num_classes=1)
    eng = TestEngineABC(model=model)
    assert isinstance(eng, EngineABC)


def test_engine_run() -> NoReturn:
    """Test engine run."""
    eng = TestEngineABC(model="alexnet-kather100k")
    assert isinstance(eng, EngineABC)

    eng = TestEngineABC(model="alexnet-kather100k")
    with pytest.raises(
        ValueError,
        match=r".*The input numpy array should be four dimensional.*",
    ):
        eng.run(images=np.zeros((10, 10)))

    eng = TestEngineABC(model="alexnet-kather100k")
    with pytest.raises(
        TypeError,
        match=r"Input must be a list of file paths or a numpy array.",
    ):
        eng.run(images=1)

    eng = TestEngineABC(model="alexnet-kather100k")
    with pytest.raises(
        ValueError,
        match=r".*len\(labels\) is not equal to len(images)*",
    ):
        eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            labels=list(range(1)),
            on_gpu=False,
        )

    with pytest.raises(
        ValueError,
        match=r".*len\(masks\) is not equal to len(images)*",
    ):
        eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            masks=np.zeros((1, 224, 224, 3)),
            on_gpu=False,
        )

    with pytest.raises(
        ValueError,
        match=r".*The shape of the numpy array should be NHWC*",
    ):
        eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            masks=np.zeros((10, 3)),
            on_gpu=False,
        )

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        patch_mode=True,
    )
    assert "predictions" in out
    assert "labels" not in out

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        verbose=False,
    )
    assert "predictions" in out
    assert "labels" not in out

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        labels=list(range(10)),
        on_gpu=False,
    )
    assert "predictions" in out
    assert "labels" in out


def test_engine_run_with_verbose() -> NoReturn:
    """Test engine run with verbose."""
    """Run pytest with `-rP` option to view progress bar on the captured stderr call"""

    eng = TestEngineABC(model="alexnet-kather100k", verbose=True)
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        labels=list(range(10)),
        on_gpu=False,
    )

    assert "predictions" in out
    assert "labels" in out


def test_patch_pred_zarr_store(tmp_path: pytest.TempPathFactory) -> NoReturn:
    """Test the engine run and patch pred store."""
    save_dir = tmp_path / "patch_output"

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        save_dir=save_dir,
        overwrite=True,
    )
    assert Path.exists(out), "Zarr output file does not exist"

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        verbose=False,
        save_dir=save_dir,
        overwrite=True,
    )
    assert Path.exists(out), "Zarr output file does not exist"

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        labels=list(range(10)),
        on_gpu=False,
        save_dir=save_dir,
        overwrite=True,
    )
    assert Path.exists(out), "Zarr output file does not exist"

    """ test custom zarr output file name"""
    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        labels=list(range(10)),
        on_gpu=False,
        save_dir=save_dir,
        overwrite=True,
        output_file="patch_pred_output",
    )
    assert Path.exists(out), "Zarr output file does not exist"
