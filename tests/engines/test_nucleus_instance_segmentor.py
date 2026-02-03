"""Test NucleusInstanceSegmentor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch

from tiatoolbox.models.engine.nucleus_instance_segmentor import (
    NucleusInstanceSegmentor,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore import WSIReader

from .test_multi_task_segmentor import (
    assert_output_lengths,
    assert_predictions_and_boxes,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

device = "cuda" if toolbox_env.has_gpu() else "cpu"
OutputType = dict[str, Any] | Any


def test_mtsegmentor_init(caplog: pytest.LogCaptureFixture) -> None:
    """Tests NucleusInstanceSegmentor initialization."""
    segmentor = NucleusInstanceSegmentor(model="hovernetplus-oed", device=device)

    assert isinstance(segmentor, NucleusInstanceSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)
    assert (
        "NucleusInstanceSegmentor is deprecated and will be removed in "
        "a future release." in caplog.text
    )


def test_mtsegmentor_patches(remote_sample: Callable) -> None:
    """Tests NucleusInstanceSegmentor on image patches."""
    mtsegmentor = NucleusInstanceSegmentor(
        model="hovernet_fast-pannuke", batch_size=32, verbose=False, device=device
    )

    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.50
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

    expected_counts_nuclei = [62, 33, 0]
    assert_output_lengths(
        output_dict,
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )
    assert_predictions_and_boxes(output_dict, expected_counts_nuclei, is_zarr=False)
