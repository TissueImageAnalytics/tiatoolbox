"""Test SemanticSegmentor."""

from __future__ import annotations

from pathlib import Path

import torch

from tiatoolbox.models.engine.semantic_segmentor_new import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_semantic_segmentor_init() -> None:
    """Tests SemanticSegmentor initialization."""
    segmentor = SemanticSegmentor(model="fcn-tissue_mask", device=device)

    assert isinstance(segmentor, SemanticSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_semantic_segmentor_patches(sample_patch1: Path, sample_patch2: Path) -> None:
    """Tests SemanticSegmentor on image patches."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    inputs = [Path(sample_patch1), Path(sample_patch2)]

    output = segmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert (
        tuple(segmentor._ioconfig.patch_output_shape)
        == output["predictions"][0].shape[:-1]
    )
