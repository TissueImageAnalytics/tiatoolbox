"""Test tiatoolbox.models.engine.nucleus_instance_segmentor."""

import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from tiatoolbox.models import IOSegmentorConfig, NucleusInstanceSegmentor

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_functionality_tile(source_image: Path, track_tmp_path: Path) -> None:
    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_workers=0,
        model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.run(
        [source_image],
        patch_mode=True,
        device=device,
        save_dir=track_tmp_path / "hovernet_fast-pannuke",
    )


def test_functionality_wsi(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Local functionality test for nuclei instance segmentor."""
    root_save_dir = Path(track_tmp_path)
    save_dir = Path(f"{track_tmp_path}/output")
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))

    # * generate full output w/o parallel post-processing worker first
    shutil.rmtree(save_dir, ignore_errors=True)
    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=8,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.run(
        [mini_wsi_svs],
        patch_mode=False,
        device=device,
        save_dir=save_dir,
    )
