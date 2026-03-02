"""Test for feature extractor."""

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture.vanilla import CNNBackbone, TimmBackbone
from tiatoolbox.models.engines.deep_feature_extractor import DeepFeatureExtractor
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()

# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_feature_extractor_patches(
    remote_sample: Callable,
) -> None:
    """Tests DeepFeatureExtractor on image patches."""
    extractor = DeepFeatureExtractor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [sample_image, sample_image]

    assert not extractor.patch_mode
    output = extractor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert 0.48 < np.mean(output["features"][:]) < 0.52

    with pytest.raises(
        ValueError,
        match=r".*output_type: `annotationstore` is not supported "
        r"for `DeepFeatureExtractor` engine",
    ):
        _ = extractor.run(
            images=inputs,
            return_probabilities=True,
            return_labels=False,
            device=device,
            patch_mode=True,
            output_type="annotationstore",
        )


def test_feature_extractor_wsi(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test feature extraction with DeepFeatureExtractor engine."""
    save_dir = track_tmp_path / "output"

    # * test providing pretrained from torch vs pretrained_model.yaml
    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test

    extractor = DeepFeatureExtractor(batch_size=1, model="fcn-tissue_mask")
    output = extractor.run(
        images=[sample_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_check",
        batch_size=1,
        output_type="zarr",
        memory_threshold=0,
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert len(output_["coordinates"].shape) == 2
    assert len(output_["features"].shape) == 3


@pytest.mark.parametrize(
    "model",
    [
        CNNBackbone("resnet18"),
        TimmBackbone("efficientnet_b0", pretrained=True),
        "resnet18",
        "efficientnet_b0",
    ],
)
def test_full_inference(
    remote_sample: Callable, track_tmp_path: Path, model: ModelABC
) -> None:
    """Test full inference with CNNBackbone and TimmBackbone models."""
    save_dir = track_tmp_path / "output"
    # pre-emptive clean up
    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test

    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    ioconfig = IOPatchPredictorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        patch_input_shape=[512, 512],
        stride_shape=[256, 256],
    )

    extractor = DeepFeatureExtractor(batch_size=4, model=model)
    output = extractor.run(
        images=[mini_wsi_svs],
        device=device,
        save_dir=track_tmp_path / "wsi_out_check",
        batch_size=4,
        output_type="zarr",
        ioconfig=ioconfig,
        patch_mode=False,
    )

    output_ = zarr.open(output[mini_wsi_svs], mode="r")

    positions = output_["coordinates"]
    features = output_["features"]

    reader = WSIReader.open(mini_wsi_svs)
    patches = [
        reader.read_bounds(
            positions[patch_idx],
            resolution=0.25,
            units="mpp",
            pad_constant_values=255,
            coord_space="resolution",
        )
        for patch_idx in range(4)
    ]
    patches = np.array(patches)
    patches = torch.from_numpy(patches)  # NHWC
    patches = patches.permute(0, 3, 1, 2).contiguous()  # NCHW
    patches = patches.to(device).type(torch.float32)
    model = extractor.model
    # Inference mode
    model.eval()
    with torch.inference_mode():
        _features = model(patches).cpu().numpy()
    # ! must maintain same batch size and likely same ordering
    # ! else the output values will not exactly be the same (still < 1.0e-4
    # ! of epsilon though)
    assert np.mean(np.abs(features[:4] - _features)) < 1.0e-1


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_multi_gpu_feature_extraction(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Local functionality test for feature extraction using multiple GPUs."""
    save_dir = track_tmp_path / "output"
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    shutil.rmtree(save_dir, ignore_errors=True)

    wsi_ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        stride_shape=[224, 224],
    )

    extractor = DeepFeatureExtractor(
        model="UNI",
        batch_size=32,
        num_workers=4,
    )

    output = extractor.run(
        [mini_wsi_svs],
        patch_mode=False,
        device=device,
        ioconfig=wsi_ioconfig,
        save_dir=save_dir,
        auto_get_mask=True,
        output_type="zarr",
    )
    output_ = zarr.open(output[mini_wsi_svs], mode="r")

    positions = output_["coordinates"]
    features = output_["features"]
    assert len(positions.shape) == 2
    assert len(features.shape) == 2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_cli_model_single_file(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for feature extractor CLI single file."""
    runner = CliRunner()
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")

    models_wsi_result = runner.invoke(
        cli.main,
        [
            "deep-feature-extractor",
            "--img-input",
            str(wsi4_512_512_svs),
            "--model",
            "resnet18",
            "--patch-mode",
            "False",
            "--output-path",
            str(track_tmp_path / "output"),
            "--patch-input-shape",
            "224",
            "224",
            "--input-resolutions",
            '[{"units": "mpp", "resolution": 0.25}]',
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert (track_tmp_path / "output" / (wsi4_512_512_svs.stem + ".zarr")).exists()

    output = zarr.open(
        str(track_tmp_path / "output" / (wsi4_512_512_svs.stem + ".zarr")), mode="r"
    )

    # Output shape should be # of patches x feature size
    assert output["features"].shape == (9, 512)
