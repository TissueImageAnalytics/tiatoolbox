"""Test tiatoolbox.models.engine.nucleus_instance_segmentor."""

import gc
import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from tiatoolbox.models import IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.utils import imwrite
from tiatoolbox.wsicore import WSIReader

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_functionality_ci(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Functionality test for nuclei instance segmentor."""
    gc.collect()
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    resolution = 2.0

    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=resolution, units="mpp")
    mini_wsi_jpg = f"{track_tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)

    # * test run on wsi, test run with worker
    # resolution for travis testing, not the correct ones
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[1024, 1024],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )

    save_dir = track_tmp_path / "instance"
    shutil.rmtree(save_dir, ignore_errors=True)

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_loader_workers=0,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )
    inst_segmentor.run(
        [mini_wsi_svs],
        patch_mode=False,
        ioconfig=ioconfig,
        device=device,
        save_dir=save_dir,
    )
