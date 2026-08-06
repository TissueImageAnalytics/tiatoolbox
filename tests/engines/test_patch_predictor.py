"""Test PatchPredictor."""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pytest
import torch
import yaml
import zarr
from click.testing import CliRunner

from tests.conftest import timed
from tiatoolbox import cli, logger, rcParam
from tiatoolbox.cli.common import tiatoolbox_cli
from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.engine_abc import EngineABC
from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.env_detection import running_on_ci
from tiatoolbox.utils.misc import download_data, get_zarr_array, imwrite

if TYPE_CHECKING:
    from collections.abc import Callable

device = "cuda" if toolbox_env.has_gpu() else "cpu"
_RUNNING_ON_CI = running_on_ci()


class FakeMiniModel(ModelABC):
    """Lightweight model double with a realistic TIAToolbox-style interface."""

    def __init__(self) -> None:
        """Initialize FakeMiniModel."""
        super().__init__()
        self.class_dict = {0: "background", 1: "tumour"}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the model."""
        batch = np.asarray(x)
        batch_size = batch.shape[0]
        logits = np.zeros((batch_size, 2), dtype=np.float32)
        logits[:, 1] = np.linspace(0.2, 0.8, batch_size, dtype=np.float32)
        logits[:, 0] = 1.0 - logits[:, 1]
        return logits

    @staticmethod
    def infer_batch(
        model: ModelABC,
        batch_image: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Model inference."""
        _ = device
        return model.forward(batch_image)

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Model postprocessing function."""
        return np.argmax(image, axis=1)


class DummyIOConfig:
    """Dummy IOConfig."""

    patch_input_shape = (224, 224)
    stride_shape = (224, 224)
    input_resolutions = [{"units": "mpp", "resolution": 0.5}]  # noqa: RUF012


@pytest.fixture
def predictor() -> PatchPredictor:
    """Fake PatchPredictor."""
    return PatchPredictor(model=FakeMiniModel(), batch_size=2, verbose=False)


@pytest.fixture
def fake_patch_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the dataloader so tests do not touch file IO or image decoding."""

    def _fake_get_dataloader(
        self: EngineABC,
        images: object,
        masks: object | None = None,
        labels: list | None = None,
        ioconfig: object | None = None,
        *,
        patch_mode: bool = True,
        auto_get_mask: bool = True,
        wsireader_kwargs: object | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Fake get dataloader function."""
        _ = (
            self,
            images,
            masks,
            labels,
            ioconfig,
            patch_mode,
            auto_get_mask,
            wsireader_kwargs,
        )
        batch_1 = {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}
        batch_2 = {"image": np.ones((1, 4, 4, 3), dtype=np.uint8)}
        return [batch_1, batch_2]

    monkeypatch.setattr(EngineABC, "get_dataloader", _fake_get_dataloader)


def test_string_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure model-name initialization can be tested without any download."""

    def _fake_get_pretrained_model(
        model: str,
        weights: str | Path | None,
    ) -> tuple[FakeMiniModel, DummyIOConfig]:
        """Fake get pretrained model function."""
        _ = weights
        assert model == "resnet18-kather100k"
        return FakeMiniModel(), DummyIOConfig()

    monkeypatch.setattr(
        "tiatoolbox.models.engine.engine_abc.get_pretrained_model",
        _fake_get_pretrained_model,
    )

    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=1, verbose=False)

    assert isinstance(predictor.model, FakeMiniModel)
    assert predictor.ioconfig is not None
    assert predictor.ioconfig.patch_input_shape == (224, 224)


def test_patch_run_fast_in_memory(
    predictor: PatchPredictor,
    fake_patch_batches: None,
) -> None:
    """Run the patch pipeline entirely in memory using fake batches."""
    _ = fake_patch_batches
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    ioconfig = IOPatchPredictorConfig(
        patch_input_shape=(224, 224),
        stride_shape=(224, 224),
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    )

    output = predictor.run(
        images=patches,
        patch_mode=True,
        output_type="dict",
        return_probabilities=True,
        return_labels=False,
        device="cpu",
        ioconfig=ioconfig,
    )

    assert sorted(output.keys()) == ["predictions", "probabilities"]
    assert output["probabilities"].shape == (2, 2)
    assert output["predictions"].shape == (2,)
    assert np.all(output["probabilities"] >= 0.0)
    assert np.all(output["probabilities"] <= 1.0)


class FakePatchPredictor:
    """Minimal stand-in for PatchPredictor that records run arguments."""

    def __init__(
        self,
        model: str,
        weights: str | None,
        batch_size: int,
        num_workers: int,
        *,
        verbose: bool,
    ) -> None:
        """Initialize FakePatchPredictor."""
        _ = model, weights, batch_size, num_workers, verbose
        self.run_calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Run function."""
        self.run_calls.append(kwargs)


def test_patch_predictor_cli_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the patch-predictor CLI forwards parsed values correctly.

    This test covers the CLI wiring only. It verifies that:
    - input and output paths are prepared,
    - the predictor is instantiated with CLI arguments,
    - the run method receives the expected keyword arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest fixture used to replace expensive collaborators.
        tmp_path (Path):
            Temporary directory used to create input and output test data.

    Returns:
        None:
            Assertions verify the expected CLI behavior.

    """
    img_input = tmp_path / "input"
    img_input.mkdir()
    sample_image = img_input / "sample.png"
    sample_image.write_bytes(b"fake image data")

    output_dir = tmp_path / "output"

    fake_predictor = FakePatchPredictor(
        model="resnet18-kather100k",
        weights=None,
        batch_size=64,
        num_workers=0,
        verbose=True,
    )

    def _fake_prepare_model_cli(
        img_input: str | Path,
        output_path: str | Path,
        masks: str | Path | None,
        file_types: str,
    ) -> tuple[list[Path], list[Path] | None, Path]:
        """Fake prepare_model_cli method."""
        _ = img_input, masks, file_types
        return [sample_image], None, Path(output_path)

    def _fake_prepare_ioconfig(
        config_class: type[Any],
        pretrained_weights: str | Path | None,
        yaml_config_path: str | Path,
    ) -> object | None:
        """Fake prepare_ioconfig method."""
        _ = config_class, pretrained_weights, yaml_config_path
        return None

    def _fake_patch_predictor(
        model: str,
        weights: str | None,
        batch_size: int,
        num_workers: int,
        *,
        verbose: bool,
    ) -> FakePatchPredictor:
        """Fake patch_predictor method."""
        assert model == "resnet18-kather100k"
        assert weights is None
        assert batch_size == 64
        assert num_workers == 0
        assert verbose is True
        return fake_predictor

    monkeypatch.setattr(
        "tiatoolbox.cli.common.prepare_model_cli",
        _fake_prepare_model_cli,
    )
    monkeypatch.setattr(
        "tiatoolbox.cli.common.prepare_ioconfig",
        _fake_prepare_ioconfig,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.engine.patch_predictor.PatchPredictor",
        _fake_patch_predictor,
    )

    runner = CliRunner()
    result = runner.invoke(
        tiatoolbox_cli,
        [
            "patch-predictor",
            "--img-input",
            str(img_input),
            "--output-path",
            str(output_dir),
            "--patch-mode",
            "False",
            "--model",
            "resnet18-kather100k",
            "--device",
            "cpu",
            "--batch-size",
            "64",
            "--return-probabilities",
            "True",
            "--auto-get-mask",
            "True",
            "--overwrite",
            "False",
            "--verbose",
            "True",
        ],
    )

    assert result.exit_code == 0
    assert len(fake_predictor.run_calls) == 1

    run_kwargs = fake_predictor.run_calls[0]
    assert run_kwargs["images"] == [sample_image]
    assert run_kwargs["masks"] is None
    assert run_kwargs["patch_mode"] is False
    assert run_kwargs["device"] == "cpu"
    assert run_kwargs["save_dir"] == output_dir
    assert run_kwargs["output_type"] == "AnnotationStore"
    assert run_kwargs["return_probabilities"] is True
    assert run_kwargs["auto_get_mask"] is True
    assert run_kwargs["overwrite"] is False
    assert run_kwargs["verbose"] is True


def test_patch_predictor_cli_uses_yaml_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the patch-predictor CLI forwards YAML config information."""
    img_input = tmp_path / "input"
    img_input.mkdir()
    (img_input / "sample.png").write_bytes(b"fake image data")

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "patch_input_shape: [224, 224]\n"
        "stride_shape: [224, 224]\n"
        "input_resolutions:\n"
        "  - units: mpp\n"
        "    resolution: 0.5\n",
        encoding="utf-8",
    )

    fake_predictor = FakePatchPredictor(
        model="resnet18-kather100k",
        weights=None,
        batch_size=64,
        num_workers=0,
        verbose=True,
    )

    def _fake_prepare_model_cli(
        img_input: str | Path,
        output_path: str | Path,
        masks: str | Path | None,
        file_types: str,
    ) -> tuple[list[Path], list[Path] | None, Path]:
        """Fake prepare_model_cli method."""
        _ = masks, file_types
        return [Path(img_input) / "sample.png"], None, Path(output_path)

    def _fake_prepare_ioconfig(
        config_class: type[Any],
        pretrained_weights: str | Path | None,
        yaml_config_path: str | Path,
    ) -> object | None:
        """Fake prepare_ioconfig method."""
        _ = config_class, pretrained_weights
        assert Path(yaml_config_path) == yaml_path
        return object()

    def _fake_patch_predictor(
        model: str,
        weights: str | None,
        batch_size: int,
        num_workers: int,
        *,
        verbose: bool,
    ) -> FakePatchPredictor:
        """Fake patch_predictor method."""
        assert model == "resnet18-kather100k"
        assert weights is None
        assert batch_size == 64
        assert num_workers == 0
        assert verbose is True
        return fake_predictor

    monkeypatch.setattr(
        "tiatoolbox.cli.common.prepare_model_cli",
        _fake_prepare_model_cli,
    )
    monkeypatch.setattr(
        "tiatoolbox.cli.common.prepare_ioconfig",
        _fake_prepare_ioconfig,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.engine.patch_predictor.PatchPredictor",
        _fake_patch_predictor,
    )

    runner = CliRunner()
    result = runner.invoke(
        tiatoolbox_cli,
        [
            "patch-predictor",
            "--img-input",
            str(img_input),
            "--output-path",
            str(tmp_path / "output"),
            "--yaml-config-path",
            str(yaml_path),
            "--patch-mode",
            "False",
        ],
    )

    assert result.exit_code == 0
    assert len(fake_predictor.run_calls) == 1


@pytest.mark.parametrize("output_type", ["zarr", "annotationstore", "qupath"])
def test_save_predictions_formats(
    predictor: PatchPredictor,
    tmp_path: Path,
    output_type: str,
) -> None:
    """Make sure the output writers work with small synthetic predictions."""
    processed_predictions = {
        "probabilities": da.from_array(
            np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        ),
        "predictions": da.from_array(np.asarray([1, 0], dtype=np.uint8)),
        "coordinates": da.from_array(
            np.asarray([[0, 0, 224, 224], [224, 224, 448, 448]], dtype=np.int64)
        ),
    }

    if output_type == "zarr":
        save_path = tmp_path / "out.zarr"
        result = predictor.save_predictions(
            processed_predictions=processed_predictions,
            output_type=output_type,
            save_path=save_path,
        )
        assert isinstance(result, Path)
        assert result.exists()
        return

    if output_type == "annotationstore":
        save_path = tmp_path / "out.db"
        result = predictor.save_predictions(
            processed_predictions=processed_predictions,
            output_type=output_type,
            save_path=save_path,
        )
        assert isinstance(result, Path)
        assert result.exists()

        with sqlite3.connect(result) as con:
            cursor = con.cursor()
            rows = list(cursor.execute("SELECT properties FROM annotations"))
        assert len(rows) == 2
        return

    save_path = tmp_path / "out.json"
    result = predictor.save_predictions(
        processed_predictions=processed_predictions,
        output_type=output_type,
        save_path=save_path,
    )
    assert isinstance(result, Path)
    assert result.exists()

    with result.open("r", encoding="utf-8") as fptr:
        payload = json.load(fptr)
    assert isinstance(payload, dict)
    assert payload.get("features")


def test_infer_wsi_calls_infer_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests ``infer_wsi``.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Fixture used to patch ``infer_patches``.

    Returns:
        None:
            Assertions verify expected behavior.

    """
    predictor = PatchPredictor(
        model=FakeMiniModel(),
        batch_size=1,
        verbose=False,
    )

    expected_output = {
        "predictions": "dummy_predictions",
        "coordinates": "dummy_coordinates",
    }

    called: dict[str, Any] = {}

    def _fake_infer_patches(
        self: PatchPredictor,
        dataloader: object,
        *,
        return_coordinates: bool,
    ) -> dict[str, object]:
        _ = self
        called["dataloader"] = dataloader
        called["return_coordinates"] = return_coordinates
        return expected_output

    monkeypatch.setattr(
        PatchPredictor,
        "infer_patches",
        _fake_infer_patches,
    )

    dataloader = object()

    result = predictor.infer_wsi(
        dataloader=dataloader,
        save_path=Path("unused.zarr"),
    )

    assert result is expected_output
    assert called["dataloader"] is dataloader
    assert called["return_coordinates"] is True


def test_update_run_params_does_not_drop_label_when_return_labels_true(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover the branch where ``if not self.return_labels`` is False.

    This test verifies that when ``return_labels=True`` is supplied via kwargs,
    the engine does not append ``"label"`` to ``drop_keys`` during
    ``_update_run_params``.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest fixture used to replace downstream validation helpers.
        tmp_path (Path):
            Temporary directory used to create a dummy image fixture.

    Returns:
        None:
            Assertions validate the expected branch behavior.

    """
    predictor = PatchPredictor(
        model=FakeMiniModel(),
        batch_size=1,
        verbose=False,
    )

    # Minimal valid image input so the method can proceed far enough.
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(b"fake image")

    def _fake_validate_input_numbers(
        *,
        images: object,
        masks: object,
        labels: object,
    ) -> None:
        _ = images, masks, labels

    def _fake_validate_images_masks(
        *,
        images: object,
    ) -> list:
        return [Path(img_path)] if images is not None else []

    def _fake_load_ioconfig(
        *,
        ioconfig: object,
    ) -> object:
        return ioconfig

    def _fake_update_ioconfig(
        ioconfig: object,
        patch_input_shape: object,
        stride_shape: object,
        input_resolutions: object,
    ) -> object:
        _ = patch_input_shape, stride_shape, input_resolutions
        return ioconfig

    def _fake_prepare_engines_save_dir(
        save_dir: Path | None,
        *,
        patch_mode: bool,
        overwrite: bool = False,
    ) -> Path | None:
        _ = patch_mode, overwrite
        return save_dir

    monkeypatch.setattr(
        predictor,
        "_validate_input_numbers",
        _fake_validate_input_numbers,
    )
    monkeypatch.setattr(
        predictor,
        "_validate_images_masks",
        _fake_validate_images_masks,
    )
    monkeypatch.setattr(
        predictor,
        "_load_ioconfig",
        _fake_load_ioconfig,
    )
    monkeypatch.setattr(
        predictor,
        "_update_ioconfig",
        _fake_update_ioconfig,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.engine.engine_abc.prepare_engines_save_dir",
        _fake_prepare_engines_save_dir,
    )

    predictor.drop_keys = []

    predictor._update_run_params(
        images=[img_path],
        patch_mode=True,
        output_type="dict",
        save_dir=None,
        return_labels=True,
    )

    assert "label" not in predictor.drop_keys
    assert predictor.return_labels is True


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_io_config_delegation(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for delegating args to io config."""
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    model = CNNModel("resnet50")
    predictor = PatchPredictor(model=model, weights=None)
    kwargs = {
        "patch_input_shape": [512, 512],
        "input_resolutions": [{"units": "mpp", "resolution": 1.75}],
    }

    # test providing config / full input info for default models without weights
    ioconfig = IOPatchPredictorConfig(
        patch_input_shape=(512, 512),
        stride_shape=(256, 256),
        input_resolutions=[{"resolution": 1.35, "units": "mpp"}],
    )
    predictor.run(
        images=[mini_wsi_svs],
        ioconfig=ioconfig,
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
        **kwargs,
    )
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    # test overwriting pretrained ioconfig
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=1)
    predictor.run(
        images=[mini_wsi_svs],
        patch_input_shape=(300, 300),
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.patch_input_shape == (300, 300)
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        stride_shape=(300, 300),
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.stride_shape == (300, 300)
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        input_resolutions=[{"units": "mpp", "resolution": 1.99}],
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 1.99
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "baseline"
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        input_resolutions=[{"units": "level", "resolution": 0}],
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "level"
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 0
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        input_resolutions=[{"units": "power", "resolution": 20}],
        patch_mode=False,
        save_dir=Path(f"{track_tmp_path}/dump"),
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "power"
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 20
    shutil.rmtree(track_tmp_path / "dump", ignore_errors=True)


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_patch_predictor_api(
    sample_patch1: Path,
    sample_patch2: Path,
    track_tmp_path: Path,
) -> None:
    """Test PatchPredictor API."""
    save_dir_path = track_tmp_path

    # Test both Path and str
    inputs = [Path(sample_patch1), str(sample_patch2)]
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=1)
    # don't run test on GPU
    # Default run
    output = predictor.run(
        inputs,
        device="cpu",
        return_probabilities=True,
    )
    assert sorted(output.keys()) == ["predictions", "probabilities"]
    assert len(output["probabilities"]) == 2
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # whether to return labels
    output = predictor.run(
        inputs,
        labels=["1", "a"],
        return_labels=True,
        return_probabilities=True,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["probabilities"]) == len(output["labels"])
    assert list(output["labels"]) == ["1", "a"]
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # test loading user weight
    pretrained_weights_url = "https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/main/resnet18-kather100k.pth"

    # remove prev generated data
    shutil.rmtree(save_dir_path, ignore_errors=True)
    save_dir_path.mkdir(parents=True)
    pretrained_weights = (
        save_dir_path / "tmp_pretrained_weigths" / "resnet18-kather100k.pth"
    )

    download_data(pretrained_weights_url, pretrained_weights)

    predictor = PatchPredictor(
        model="resnet18-kather100k",
        weights=pretrained_weights,
        batch_size=1,
    )
    ioconfig = predictor.ioconfig

    # --- test different using user model
    model = CNNModel(backbone="resnet18", num_classes=9)
    # test prediction
    predictor = PatchPredictor(model=model, batch_size=1, verbose=False)
    output = predictor.run(
        inputs,
        labels=[1, 2],
        return_labels=True,
        ioconfig=ioconfig,
        return_probabilities=True,
        num_workers=1,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["probabilities"]) == len(output["labels"])
    assert list(output["labels"]) == [1, 2]

    processed_predictions = {
        k: da.from_array(v) for k, v in output.items() if k != "labels"
    }
    processed_predictions["coordinates"] = np.asarray(
        [[0, 0, 224, 224], [0, 0, 224, 224]]
    )
    (track_tmp_path / "patch_out_check").mkdir()
    output_ = predictor.save_predictions(
        processed_predictions=processed_predictions,
        output_type="annotationstore",
        save_path=track_tmp_path / "patch_out_check" / "output.db",
    )

    assert output_.exists()
    output_ann = _extract_probabilities_from_annotation_store(output_)
    assert np.all(np.array(output_ann["probabilities"]) <= 1)
    assert np.all(np.array(output_ann["probabilities"]) >= 0)


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_wsi_predictor_api(
    sample_wsi_dict: dict,
    track_tmp_path: Path,
) -> None:
    """Test normal run of wsi predictor."""
    save_dir_path = track_tmp_path

    # Test both Path and str input
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_jpg = sample_wsi_dict["wsi2_4k_4k_jpg"]
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    patch_size = np.array([224, 224])
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=32)

    save_dir = f"{save_dir_path}/model_wsi_output"

    # wrapper to make this more clean
    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "save_dir": save_dir,
    }

    _kwargs = copy.deepcopy(kwargs)
    # test reading of multiple whole-slide images
    output = predictor.run(
        images=[mini_wsi_svs, str(mini_wsi_jpg)],
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        masks=[mini_wsi_msk, mini_wsi_msk],
        patch_mode=False,
        return_probabilities=True,
        **_kwargs,
    )

    wsi_out = zarr.open(str(output[mini_wsi_svs]), mode="r")
    tile_out = zarr.open(str(output[mini_wsi_jpg]), mode="r")
    diff = tile_out["probabilities"][:] == wsi_out["probabilities"][:]
    accuracy = np.sum(diff) / np.size(wsi_out["probabilities"][:])
    assert accuracy > 0.99, np.nonzero(~diff)

    diff = tile_out["predictions"][:] == wsi_out["predictions"][:]
    accuracy = np.sum(diff) / np.size(wsi_out["predictions"][:])
    assert accuracy > 0.99, np.nonzero(~diff)

    shutil.rmtree(_kwargs["save_dir"], ignore_errors=True)


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_patch_predictor_kather100k_output(
    sample_patch1: Path,
    sample_patch2: Path,
    track_tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]
    pretrained_info = {
        "alexnet-kather100k": [1.0, 0.9999735355377197],
        "resnet18-kather100k": [1.0, 0.9999911785125732],
        "mobilenet_v3_small-kather100k": [0.9999998807907104, 0.9999997615814209],
    }
    for model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            model,
            probabilities_check=expected_prob,
            classification_check=[6, 3],
        )

    for model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            model,
            probabilities_check=expected_prob,
            classification_check=[6, 3],
            track_tmp_path=track_tmp_path,
        )


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_wsi_predictor_zarr(
    sample_wsi_dict: dict, track_tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test normal run of patch predictor for WSIs."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        images=[mini_wsi_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_check",
    )

    assert output[mini_wsi_svs].exists()

    output_ = zarr.open(output[mini_wsi_svs])

    assert output_["probabilities"].shape == (70, 9)  # number of patches x classes
    assert output_["probabilities"].ndim == 2
    # number of patches x [start_x, start_y, end_x, end_y]
    assert output_["coordinates"].shape == (70, 4)
    assert output_["coordinates"].ndim == 2
    # prediction for each patch
    assert output_["predictions"].shape == (70,)
    assert output_["predictions"].ndim == 1
    assert _validate_probabilities(output=output_)
    assert "Output file saved at " in caplog.text

    output = predictor.run(
        images=[mini_wsi_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_check_no_probabilities",
    )

    assert output[mini_wsi_svs].exists()

    output_ = zarr.open(output[mini_wsi_svs])

    assert "probabilities" not in output_
    # number of patches x [start_x, start_y, end_x, end_y]
    assert output_["coordinates"].shape == (70, 4)
    assert output_["coordinates"].ndim == 2
    # prediction for each patch
    assert output_["predictions"].shape == (70,)
    assert output_["predictions"].ndim == 1
    assert _validate_probabilities(output=output_)
    assert "Output file saved at " in caplog.text


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_patch_predictor_patch_mode_annotation_store(
    sample_patch1: Path,
    sample_patch2: Path,
    track_tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "patch_out_check",
        output_type="annotationstore",
    )

    assert output.exists()
    output = _extract_probabilities_from_annotation_store(output)
    assert np.all(output["predictions"] == [6, 3])
    assert np.all(np.array(output["probabilities"]) <= 1)
    assert np.all(np.array(output["probabilities"]) >= 0)


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_patch_predictor_patch_mode_no_probabilities(
    sample_patch1: Path,
    sample_patch2: Path,
    track_tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )

    output = predictor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert "probabilities" not in output

    processed_predictions = {k: v for k, v in output.items() if k != "labels"}
    processed_predictions["coordinates"] = np.asarray(
        [[0, 0, 224, 224], [0, 0, 224, 224]]
    )

    (track_tmp_path / "patch_out_check").mkdir()

    output_ = predictor.save_predictions(
        processed_predictions=processed_predictions,
        output_type="annotationstore",
        save_path=track_tmp_path / "patch_out_check" / "output.db",
    )

    assert output_.exists()
    output_ann = _extract_probabilities_from_annotation_store(output_)
    assert np.all(output_ann["predictions"] == [6, 3])
    assert "probabilities" not in output

    # QuPath Output
    output_ = predictor.save_predictions(
        processed_predictions=processed_predictions,
        output_type="qupath",
        save_path=track_tmp_path / "patch_out_check" / "output.json",
    )

    assert output_.exists()
    output_ann = _extract_from_qupath_json(output_)
    assert np.all(output_ann["predictions"] == [6, 3])
    assert "probabilities" not in output


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_engine_run_wsi_annotation_store(
    sample_wsi_dict: dict,
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = PatchPredictor(model="alexnet-kather100k")

    patch_size = np.array([224, 224])
    save_dir = f"{track_tmp_path}/model_wsi_output"

    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 0.5,
        "save_dir": save_dir,
        "units": "mpp",
        "scale_factor": (2.0, 2.0),
    }

    output = eng.run(
        images=[mini_wsi_svs],
        masks=[mini_wsi_msk],
        patch_mode=False,
        output_type="AnnotationStore",
        batch_size=4,
        **kwargs,
    )

    output_ = output[mini_wsi_svs]

    assert output_.exists()
    assert output_.suffix == ".db"
    output_ = _extract_probabilities_from_annotation_store(output_)

    # prediction for each patch
    assert np.array(output_["predictions"]).shape == (69,)
    assert _validate_probabilities(output_)

    assert "Output file saved at " in caplog.text

    shutil.rmtree(save_dir)


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_engine_run_wsi_qupath(
    sample_wsi_dict: dict,
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = PatchPredictor(model="alexnet-kather100k")

    patch_size = np.array([224, 224])
    save_dir = f"{track_tmp_path}/model_wsi_output"

    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 0.5,
        "save_dir": save_dir,
        "units": "mpp",
        "scale_factor": (2.0, 2.0),
    }

    output = eng.run(
        images=[mini_wsi_svs],
        masks=[mini_wsi_msk],
        patch_mode=False,
        output_type="QuPath",
        batch_size=4,
        **kwargs,
    )

    output_ = output[mini_wsi_svs]

    assert output_.exists()
    assert output_.suffix == ".json"
    output_ = _extract_from_qupath_json(output_)

    # prediction for each patch
    assert np.array(output_["predictions"]).shape == (69,)
    assert _validate_probabilities(output_)

    assert "Output file saved at " in caplog.text

    shutil.rmtree(save_dir)


# --------------------------------------------------------------------------------------
# torch.compile
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_patch_predictor_torch_compile(
    sample_patch1: Path,
    sample_patch2: Path,
    track_tmp_path: Path,
) -> None:
    """Test PatchPredictor with torch.compile functionality.

    Args:
        sample_patch1 (Path): Path to sample patch 1.
        sample_patch2 (Path): Path to sample patch 2.
        track_tmp_path (Path): Path to temporary directory.

    """
    torch_compile_mode = rcParam["torch_compile_mode"]
    torch._dynamo.reset()
    rcParam["torch_compile_mode"] = "default"
    _, compile_time = timed(
        test_patch_predictor_api,
        sample_patch1,
        sample_patch2,
        track_tmp_path,
    )
    logger.info("torch.compile default mode: %s", compile_time)
    torch._dynamo.reset()
    rcParam["torch_compile_mode"] = "reduce-overhead"
    _, compile_time = timed(
        test_patch_predictor_api,
        sample_patch1,
        sample_patch2,
        track_tmp_path,
    )
    logger.info("torch.compile reduce-overhead mode: %s", compile_time)
    torch._dynamo.reset()
    rcParam["torch_compile_mode"] = "max-autotune"
    _, compile_time = timed(
        test_patch_predictor_api,
        sample_patch1,
        sample_patch2,
        track_tmp_path,
    )
    logger.info("torch.compile max-autotune mode: %s", compile_time)
    torch._dynamo.reset()
    rcParam["torch_compile_mode"] = torch_compile_mode


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_cli_model_single_file(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for models CLI single file."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    runner = CliRunner()
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(wsi4_512_512_svs),
            "--patch-mode",
            "False",
            "--output-path",
            str(track_tmp_path / "output"),
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert (track_tmp_path / "output" / (wsi4_512_512_svs.stem + ".db")).exists()


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_cli_model_multiple_file_mask(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test for models CLI multiple file with mask."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(Path(f"{track_tmp_path}/small_svs_tissue_mask.jpg"), sample_wsi_msk)
    mini_wsi_msk = track_tmp_path.joinpath("small_svs_tissue_mask.jpg")

    # Make multiple copies for test
    dir_path = track_tmp_path.joinpath("new_copies")
    dir_path.mkdir()

    dir_path_masks = track_tmp_path.joinpath("new_copies_masks")
    dir_path_masks.mkdir()

    config = {
        "input_resolutions": [{"units": "mpp", "resolution": 0.5}],
        "patch_input_shape": [224, 224],
    }

    with Path.open(track_tmp_path.joinpath("config.yaml"), "w") as fptr:
        yaml.dump(config, fptr)

    model = "alexnet-kather100k"
    weights = fetch_pretrained_weights(model)

    try:
        dir_path.joinpath("1_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
        dir_path.joinpath("2_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
        dir_path.joinpath("3_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
    except OSError:
        shutil.copy(mini_wsi_svs, dir_path / ("1_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path / ("2_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path / ("3_" + mini_wsi_svs.name))

    try:
        dir_path_masks.joinpath("1_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("2_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("3_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
    except OSError:
        shutil.copy(mini_wsi_msk, dir_path_masks / ("1_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks / ("2_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks / ("3_" + mini_wsi_msk.name))

    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(dir_path),
            "--patch-mode",
            str(False),
            "--masks",
            str(dir_path_masks),
            "--model",
            model,
            "--weights",
            str(weights),
            "--yaml-config-path",
            track_tmp_path / "config.yaml",
            "--output-path",
            str(track_tmp_path / "output"),
            "--output-type",
            "zarr",
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert (track_tmp_path / "output" / ("1_" + mini_wsi_svs.stem + ".zarr")).exists()
    assert (track_tmp_path / "output" / ("2_" + mini_wsi_svs.stem + ".zarr")).exists()
    assert (track_tmp_path / "output" / ("3_" + mini_wsi_svs.stem + ".zarr")).exists()


def _test_predictor_output(
    inputs: list,
    model: str,
    probabilities_check: list | None = None,
    classification_check: list | None = None,
    output_type: str = "dict",
    track_tmp_path: Path | None = None,
) -> None:
    """Test the predictions of multiple models included in tiatoolbox."""
    cache_mode = None if track_tmp_path is None else True
    save_dir = None if track_tmp_path is None else track_tmp_path / "output"
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        inputs,
        return_labels=False,
        device=device,
        cache_mode=cache_mode,
        save_dir=save_dir,
        output_type=output_type,
        return_probabilities=True,
    )

    if track_tmp_path is not None:
        output = zarr.open(output, mode="r")

    probabilities = output["probabilities"]
    classification = output["predictions"]
    for idx, probabilities_ in enumerate(probabilities):
        probabilities_max = max(probabilities_)
        assert np.abs(probabilities_max - probabilities_check[idx]) <= 1e-3, (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            classification_check[idx],
        )
        assert classification[idx] == classification_check[idx], (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            classification_check[idx],
        )
    if save_dir:
        shutil.rmtree(save_dir)


def _extract_probabilities_from_annotation_store(db_file: str | Path) -> dict:
    """Helper function to extract probabilities from Annotation Store."""
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    annotations_properties = list(cur.execute("SELECT properties FROM annotations"))

    output = {"probabilities": [], "predictions": []}

    for item in annotations_properties:
        for json_str in item:
            probs_dict = json.loads(json_str)
            if "proba_0" in probs_dict:
                output["probabilities"].append(probs_dict.pop("prob_0"))
            output["predictions"].append(probs_dict.pop("type"))

    return output


def _extract_from_qupath_json(json_file: str) -> dict:
    """Extract predictions (and optionally coordinates) from QuPath GeoJSON."""
    with Path.open(json_file, "r") as f:
        data = json.load(f)

    output = {"predictions": [], "coordinates": []}

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        cls = props.get("classification", {})

        # prediction - class name
        output["predictions"].append(cls.get("name"))

        # geometry - polygon
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0]  # first ring of polygon
        output["coordinates"].append(coords)

    return output


def _validate_probabilities(output: list | dict | zarr.group) -> bool:
    """Helper function to test if the probabilities value are valid."""
    probabilities = np.array([0.5])

    if "probabilities" in output:
        probabilities = output["probabilities"]

    predictions = output["predictions"]
    if isinstance(probabilities, dict):
        return all(0 <= probability <= 1 for _, probability in probabilities.items())

    predictions = np.array(get_zarr_array(predictions)).astype(int)
    probabilities = get_zarr_array(probabilities)

    if not np.all(np.array(probabilities) <= 1):
        return False

    if not np.all(np.array(probabilities) >= 0):
        return False

    return np.all(predictions[:][0:5] == [7, 3, 2, 3, 3])
