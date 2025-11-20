"""Test tiatoolbox.models.engine.nucleus_instance_segmentor."""

from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import torch

from tiatoolbox.models import NucleusInstanceSegmentor
from tiatoolbox.wsicore import WSIReader

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_functionality_patch_mode(remote_sample: Callable) -> None:
    """Patch mode functionality test for nuclei instance segmentor."""
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.25
    units: Final = "mpp"
    patch1 = mini_wsi.read_rect(
        location=(0, 0),
        size=size,
        resolution=resolution,
        units=units,
    )
    patch2 = mini_wsi.read_rect(
        location=(512, 512),
        size=size,
        resolution=resolution,
        units=units,
    )

    patches = np.stack(arrays=[patch1, patch2], axis=0)

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_workers=0,
        model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="dict",
    )

    assert np.max(output["predictions"][0][:]) == 41
    assert np.max(output["predictions"][1][:]) == 17
    assert len(output["inst_dict"][0].columns) == 41
    assert len(output["inst_dict"][1].columns) == 17
