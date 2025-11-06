"""Test for feature extractor."""

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from tiatoolbox.models import IOSegmentorConfig
from tiatoolbox.models.architecture.vanilla import CNNBackbone, TimmBackbone
from tiatoolbox.models.engine.deep_feature_extractor import DeepFeatureExtractor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()

# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_engine(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test feature extraction with DeepFeatureExtractor engine."""
    save_dir = track_tmp_path / "output"
    # # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))

    # * test providing pretrained from torch vs pretrained_model.yaml
    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test

    extractor = DeepFeatureExtractor(batch_size=1, model="fcn-tissue_mask")
    output = extractor.run(
        images=[mini_wsi_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_check",
        batch_size=2,
        output_type="zarr",
    )

    output_ = zarr.open(output[mini_wsi_svs], mode="r")
    assert len(output_["coordinates"].shape) == 2
    assert len(output_["probabilities"].shape)


@pytest.mark.parametrize(
    "model", [CNNBackbone("resnet50"), TimmBackbone("efficientnet_b0", pretrained=True)]
)
def test_full_inference(
    remote_sample: Callable, track_tmp_path: Path, model: Callable
) -> None:
    """Test full inference with CNNBackbone and TimmBackbone models."""
    save_dir = track_tmp_path / "output"
    # pre-emptive clean up
    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test

    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0},
    )

    extractor = DeepFeatureExtractor(batch_size=4, model=model)
    output = extractor.run(
        images=[mini_wsi_svs],
        device=device,
        save_dir=track_tmp_path / "wsi_out_check",
        batch_size=2,
        output_type="zarr",
        ioconfig=ioconfig,
        patch_mode=False,
    )

    output_ = zarr.open(output[mini_wsi_svs], mode="r")

    positions = output_["coordinates"]
    features = output_["probabilities"]

    reader = WSIReader.open(mini_wsi_svs)
    patches = [
        reader.read_bounds(
            positions[patch_idx],
            resolution=0.25,
            units="mpp",
            pad_constant_values=0,
            coord_space="resolution",
        )
        for patch_idx in range(4)
    ]
    patches = np.array(patches)
    patches = torch.from_numpy(patches)  # NHWC
    patches = patches.permute(0, 3, 1, 2)  # NCHW
    patches = patches.type(torch.float32)
    model = model.to("cpu")
    # Inference mode
    model.eval()
    with torch.inference_mode():
        _features = model(patches).numpy()
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

    # Use multiple GPUs
    device = select_device(on_gpu=ON_GPU)

    wsi_ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        output_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_output_shape=[224, 224],
        stride_shape=[224, 224],
    )

    model = TimmBackbone(backbone="UNI", pretrained=True)
    extractor = DeepFeatureExtractor(
        model=model,
        auto_generate_mask=True,
        batch_size=32,
        num_loader_workers=4,
        num_postproc_workers=4,
    )

    output_list = extractor.predict(
        [mini_wsi_svs],
        mode="wsi",
        device=device,
        ioconfig=wsi_ioconfig,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    wsi_0_root_path = output_list[0][1]
    positions = np.load(f"{wsi_0_root_path}.position.npy")
    features = np.load(f"{wsi_0_root_path}.features.0.npy")
    assert len(positions.shape) == 2
    assert len(features.shape) == 2
