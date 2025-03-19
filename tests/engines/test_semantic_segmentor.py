"""Test SemanticSegmentor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from tiatoolbox.models.engine.semantic_segmentor_new import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_semantic_segmentor_init() -> None:
    """Tests SemanticSegmentor initialization."""
    segmentor = SemanticSegmentor(model="fcn-tissue_mask", device=device)

    assert isinstance(segmentor, SemanticSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_semantic_segmentor_patches(
    sample_patch1: Path, sample_patch2: Path, tmp_path: Path
) -> None:
    """Tests SemanticSegmentor on image patches."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    inputs = [Path(sample_patch1), Path(sample_patch2)]

    assert segmentor.cache_mode is False

    output = segmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert 0.24 < np.mean(output["predictions"][:]) < 0.25
    assert 0.495 < np.mean(output["probabilities"][:]) < 0.505

    assert (
        tuple(segmentor._ioconfig.patch_output_shape)
        == output["probabilities"][0].shape[:-1]
    )

    assert (
        tuple(segmentor._ioconfig.patch_output_shape) == output["predictions"][0].shape
    )

    output = segmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output0",
    )

    assert output == tmp_path / "output0" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.24 < np.mean(output["predictions"][:]) < 0.25
    assert 0.495 < np.mean(output["probabilities"][:]) < 0.505

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output1",
    )

    assert output == tmp_path / "output1" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.24 < np.mean(output["predictions"][:]) < 0.25
    assert "probabilities" not in output.keys()  # noqa: SIM118


def test_save_annotation_store(
    sample_patch1: Path, sample_patch2: Path, tmp_path: Path
):
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    inputs = [Path(sample_patch1), Path(sample_patch2)]
    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output == tmp_path / "output1" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.24 < np.mean(output["predictions"][:]) < 0.25
    assert "probabilities" not in output.keys()  # noqa: SIM118


def test_hovernet_dat() -> None:
    from pathlib import Path

    from tiatoolbox.utils.misc import store_from_dat

    path_to_file = Path.cwd().parent.parent / "output" / "0.dat"
    out = store_from_dat(path_to_file, scale_factor=(1.0, 1.0))
