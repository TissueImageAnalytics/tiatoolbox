"""Test MultiTaskSegmentor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore import WSIReader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

OutputType = dict[str, Any] | Any
device = "cuda" if toolbox_env.has_gpu() else "cpu"


def assert_output_lengths(output: OutputType, expected_counts: Sequence[int]) -> None:
    """Assert lengths of output dict fields against expected counts."""
    output = output["info_dict"]
    for field in output:
        for i, expected in enumerate(expected_counts):
            assert len(output[field][i]) == expected, f"{field}[{i}] mismatch"


def assert_predictions_and_boxes(
    output: OutputType, expected_counts: Sequence[int], *, is_zarr: bool = False
) -> None:
    """Assert predictions maxima and box lengths against expected counts."""
    # predictions maxima
    for idx, expected in enumerate(expected_counts):
        if is_zarr and idx == 2:
            # zarr output doesn't store predictions for patch 2
            continue
        assert np.max(output["predictions"][idx][:]) == expected, (
            f"predictions[{idx}] mismatch"
        )


def test_mtsegmentor_init() -> None:
    """Tests SemanticSegmentor initialization."""
    segmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    assert isinstance(segmentor, MultiTaskSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_mtsegmentor_patches(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Tests MultiTaskSegmentor on image patches."""
    mtsegmentor = MultiTaskSegmentor(
        model="hovernetplus-oed", batch_size=32, verbose=False, device=device
    )

    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.25
    units: Final = "mpp"

    patch1 = mini_wsi.read_rect(
        location=(0, 0), size=size, resolution=resolution, units=units
    )
    patch2 = mini_wsi.read_rect(
        location=(512, 512), size=size, resolution=resolution, units=units
    )
    patch3 = np.zeros_like(patch1)
    patches = np.stack([patch1, patch2, patch3], axis=0)

    assert not mtsegmentor.patch_mode

    output_dict = mtsegmentor.run(
        images=patches,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    expected_counts = [50, 17, 0]
    assert_output_lengths(output_dict["nuclei_segmentation"], expected_counts)
    assert_predictions_and_boxes(
        output_dict["nuclei_segmentation"], expected_counts, is_zarr=False
    )
    expected_counts = [1, 1, 0]
    assert_output_lengths(output_dict["layer_segmentation"], expected_counts)
    assert_predictions_and_boxes(
        output_dict["layer_segmentation"], expected_counts, is_zarr=False
    )

    _ = track_tmp_path
