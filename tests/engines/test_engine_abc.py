"""Test tiatoolbox.models.engine.engine_abc."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn

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

    def get_dataloader(
        self: EngineABC,
        images: Path,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Test pre process images."""
        return super().get_dataloader(
            images,
            masks,
            labels,
            ioconfig,
            patch_mode=patch_mode,
        )

    def save_output(
        self: EngineABC,
        raw_output: dict,
        save_dir: Path,
        **kwargs: dict,
    ) -> Path:
        """Test post_process_wsi."""
        return super().save_output(
            raw_output,
            save_dir=save_dir,
            **kwargs,
        )

    def infer_wsi(
        self: EngineABC,
        dataloader: torch.utils.data.DataLoader,
        img_label: str,
        highest_input_resolution: list[dict],
        save_dir: Path,
        **kwargs: dict,
    ) -> dict | np.ndarray:
        """Test infer_wsi."""
        return super().infer_wsi(
            dataloader,
            img_label,
            highest_input_resolution,
            save_dir,
            **kwargs,
        )


def test_engine_abc() -> NoReturn:
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC*",
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
        match=r".*Input WSIs detected but no save directory provided.*",
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


def test_engine_run(tmp_path: Path) -> NoReturn:
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

    eng = TestEngineABC(model="alexnet-kather100k")

    with pytest.raises(NotImplementedError):
        eng.run(
            images=np.zeros(shape=(10, 224, 224, 3)),
            save_dir=tmp_path / "output",
            patch_mode=False,
        )


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
        _ = eng.run(
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
        _ = eng.run(
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
        _ = eng.run(
            images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            labels=list(range(10)),
            on_gpu=False,
            save_dir=save_dir,
            overwrite=True,
            output_type="AnnotationStore",
            scale_factor=(2.0, 2.0),
        )


def test_io_config_delegation(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for delegating args to io config."""
    mini_wsi_svs = Path(remote_sample("wsi2_4k_4k_svs"))

    # test not providing config / full input info for not pretrained models
    model = CNNModel("resnet50")
    eng = TestEngineABC(model=model)
    with pytest.raises(ValueError, match=r".*Please provide a valid ModelIOConfigABC*"):
        eng.run([mini_wsi_svs], patch_mode=False, save_dir=tmp_path / "dump")
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    kwargs = {
        "patch_input_shape": [512, 512],
        "resolution": 1.75,
        "units": "mpp",
    }
    for key in kwargs:
        _kwargs = copy.deepcopy(kwargs)
        _kwargs.pop(key)
        with pytest.raises(ValueError, match=r".*Please provide.*.ModelIOConfigABC*"):
            eng.run(
                [mini_wsi_svs],
                patch_mode=False,
                save_dir=f"{tmp_path}/dump",
                **_kwargs,
            )
        shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    # test providing config / full input info for non pretrained models
    ioconfig = ModelIOConfigABC(
        patch_input_shape=(512, 512),
        stride_shape=(256, 256),
        input_resolutions=[{"resolution": 1.35, "units": "mpp"}],
    )
    eng.run(
        [mini_wsi_svs],
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
        ioconfig=ioconfig,
    )
    assert eng._ioconfig.patch_input_shape == (512, 512)
    assert eng._ioconfig.stride_shape == (256, 256)
    assert eng._ioconfig.input_resolutions == [{"resolution": 1.35, "units": "mpp"}]
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    eng.run(
        [mini_wsi_svs],
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
        **kwargs,
    )
    assert eng._ioconfig.patch_input_shape == [512, 512]
    assert eng._ioconfig.stride_shape == [512, 512]
    assert eng._ioconfig.input_resolutions == [{"resolution": 1.75, "units": "mpp"}]
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    # test overwriting pretrained ioconfig
    eng = TestEngineABC(model="alexnet-kather100k")
    eng.run(
        [mini_wsi_svs],
        patch_input_shape=(300, 300),
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert eng._ioconfig.patch_input_shape == (300, 300)
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    eng.run(
        [mini_wsi_svs],
        stride_shape=(300, 300),
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert eng._ioconfig.stride_shape == (300, 300)
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    eng.run(
        [mini_wsi_svs],
        resolution=1.99,
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert eng._ioconfig.input_resolutions[0]["resolution"] == 1.99
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    eng = TestEngineABC(model="alexnet-kather100k")
    eng.run(
        [mini_wsi_svs],
        units="baseline",
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert eng._ioconfig.input_resolutions[0]["units"] == "baseline"
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)


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

    assert isinstance(out, dict)
