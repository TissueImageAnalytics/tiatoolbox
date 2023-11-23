"""Test tiatoolbox.models.engine.engine_abc."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
import torchvision.models as torch_models

from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.engine_abc import EngineABC, prepare_engines_save_dir
from tiatoolbox.models.engine.io_config import ModelIOConfigABC

if TYPE_CHECKING:
    import torch.nn


class TestEngineABC(EngineABC):
    """Test EngineABC."""

    def __init__(
        self: TestEngineABC,
        model: str | torch.nn.Module,
        weights: str | Path | None = None,
        verbose: bool | None = None,
    ) -> NoReturn:
        """Test EngineABC init."""
        super().__init__(model=model, weights=weights, verbose=verbose)

    def set_dataloader(
        self: EngineABC,
        images: Path,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
    ) -> torch.utils.data.DataLoader:
        """Test pre process images."""
        return super().set_dataloader(images, masks, labels, ioconfig)

    def post_process_wsi(
        self: EngineABC,
        raw_output: dict,
        save_dir: Path,
        **kwargs: dict,
    ) -> Path:
        """Test post_process_wsi."""
        return super().post_process_wsi(
            raw_output,
            save_dir,
            **kwargs,
        )

    def infer_wsi(
        self: EngineABC,
        dataloader: torch.utils.data.DataLoader,
        img_path: Path,
        img_label: str,
        highest_input_resolution: list[dict],
        *,
        merge_predictions: bool,
        **kwargs: dict,
    ) -> dict | np.ndarray:
        """Test infer_wsi."""
        return super().infer_wsi(
            dataloader,
            img_path,
            img_label,
            highest_input_resolution,
            merge_predictions=merge_predictions,
            **kwargs,
        )


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
        TestEngineABC(model=1)


def test_incorrect_ioconfig() -> NoReturn:
    """Test EngineABC initialization with incorrect ioconfig."""
    model = torch_models.resnet18()
    engine = TestEngineABC(model=model)
    with pytest.raises(
        ValueError,
        match=r".*provide a valid ModelIOConfigABC.*",
    ):
        engine.run(images=[], masks=[], ioconfig=None)


def test_pretrained_ioconfig() -> NoReturn:
    """Test EngineABC initialization with pretrained model name in the toolbox."""
    pretrained_model = "alexnet-kather100k"

    # Test engine run without ioconfig
    eng = TestEngineABC(model=pretrained_model)
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        on_gpu=False,
        patch_mode=True,
        ioconfig=None,
    )
    assert "predictions" in out
    assert "labels" not in out


def test_ioconfig() -> NoReturn:
    """Test EngineABC initialization with valid ioconfig."""
    ioconfig = ModelIOConfigABC(
        input_resolutions=[
            {"units": "baseline", "resolution": 1.0},
        ],
        patch_input_shape=(224, 224),
    )

    eng = TestEngineABC(model="alexnet-kather100k")
    out = eng.run(
        images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
        ioconfig=ioconfig,
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
        overwrite=False,
    )

    assert out_dir == tmp_path / "patch_output"
    assert out_dir.exists()

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output",
        patch_mode=True,
        overwrite=True,
    )

    assert out_dir == tmp_path / "patch_output"
    assert out_dir.exists()

    out_dir = prepare_engines_save_dir(
        save_dir=None,
        patch_mode=True,
        overwrite=False,
    )
    assert out_dir is None

    with pytest.raises(
        OSError,
        match=r".*Input WSIs detected but there is no save directory provided.*",
    ):
        _ = prepare_engines_save_dir(
            save_dir=None,
            patch_mode=False,
            overwrite=False,
        )

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_single_output",
        patch_mode=False,
        overwrite=False,
    )

    assert out_dir == tmp_path / "wsi_single_output"
    assert out_dir.exists()
    assert r"When providing multiple whole-slide images / tiles" not in caplog.text

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_multiple_output",
        patch_mode=False,
        overwrite=False,
    )

    assert out_dir == tmp_path / "wsi_multiple_output"
    assert out_dir.exists()
    assert r"When providing multiple whole slide images" in caplog.text

    # test for file overwrite with Path.mkdirs() method
    out_path = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output" / "output.zarr",
        patch_mode=True,
        overwrite=True,
    )
    assert out_path.exists()

    out_path = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output" / "output.zarr",
        patch_mode=True,
        overwrite=True,
    )
    assert out_path.exists()

    with pytest.raises(FileExistsError):
        out_path = prepare_engines_save_dir(
            save_dir=tmp_path / "patch_output" / "output.zarr",
            patch_mode=True,
            overwrite=False,
        )


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

    model = get_pretrained_model("alexnet-kather100k")[0]
    weights_path = fetch_pretrained_weights("alexnet-kather100k")
    eng = TestEngineABC(model=model, weights=weights_path)
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

    eng = TestEngineABC(model="alexnet-kather100k")
    with pytest.raises(
        ValueError,
        match=r".*Patch output must contain coordinates.",
    ):
        out = eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            labels=list(range(10)),
            on_gpu=False,
            save_dir=save_dir,
            overwrite=True,
            output_type="AnnotationStore",
        )

    with pytest.raises(
        ValueError,
        match=r".*Patch output must contain coordinates.",
    ):
        out = eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            labels=list(range(10)),
            on_gpu=False,
            save_dir=save_dir,
            overwrite=True,
            output_type="AnnotationStore",
            class_dict={0: "class0", 1: "class1"},
        )

    with pytest.raises(
        ValueError,
        match=r".*Patch output must contain coordinates.",
    ):
        out = eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            labels=list(range(10)),
            on_gpu=False,
            save_dir=save_dir,
            overwrite=True,
            output_type="AnnotationStore",
            scale_factor=(2.0, 2.0),
        )


# to be implemented
def test_engine_run_wsi(
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> NoReturn:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = TestEngineABC(model="alexnet-kather100k")

    patch_size = np.array([224, 224])
    save_dir = f"{tmp_path}/model_wsi_output"

    kwargs = {
        "return_labels": True,
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 0.5,
        "save_dir": save_dir,
        "units": "mpp",
    }

    out = eng.run(
        images=[mini_wsi_svs],
        masks=[mini_wsi_msk],
        patch_mode=False,
        **kwargs,
    )

    for output_info in out.values():
        assert Path(output_info["raw"]).exists()
        assert "merged" not in output_info
    shutil.rmtree(save_dir)

    _kwargs = copy.deepcopy(kwargs)
    _kwargs["merge_predictions"] = False
    # test reading of multiple whole-slide images
    out = eng.run(
        images=[mini_wsi_svs, mini_wsi_svs],
        masks=[mini_wsi_msk, mini_wsi_msk],
        patch_mode=False,
        **_kwargs,
    )

    for output_info in out.values():
        assert Path(output_info["raw"]).exists()
        assert "merged" not in output_info
    shutil.rmtree(save_dir)

    _kwargs["merge_predictions"] = True
    # test reading of multiple whole-slide images
    out = eng.run(
        images=[mini_wsi_svs, mini_wsi_svs],
        masks=[mini_wsi_msk, mini_wsi_msk],
        patch_mode=False,
        **_kwargs,
    )

    for output_info in out.values():
        assert Path(output_info["raw"]).exists()
        assert "merged" in output_info
    shutil.rmtree(save_dir)
