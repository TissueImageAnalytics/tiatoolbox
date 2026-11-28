"""Sampling validation, reproducibility, and resource ownership tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tiatoolbox.models.training import (
    ClassBalancedIndexSampler,
    GroupedBlockSampler,
    generate_slide_patch_coordinates,
    samplers,
    split_indices,
)
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


@pytest.mark.parametrize("shape", [0, -1, (2,), (0, 2), (2, -1)])
@pytest.mark.parametrize("argument", ["patch_size", "stride"])
def test_invalid_patch_dimensions(shape: object, argument: str) -> None:
    """Reject invalid patch sizes and strides before opening a slide."""
    kwargs = {"patch_size": 4, argument: shape}
    with pytest.raises(ValueError, match=argument):
        generate_slide_patch_coordinates(np.zeros((8, 8, 3)), **kwargs)


def test_missing_annotation_mask(tmp_path: Path) -> None:
    """A missing store path must not silently create an empty database."""
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="Annotation mask store does not exist"):
        generate_slide_patch_coordinates(np.zeros((8, 8, 3)), 4, input_mask=path)
    assert not path.exists()


@pytest.mark.parametrize("coordinates", [[], [[0, 0, 2]]])
def test_malformed_extractor_coordinates(
    monkeypatch: pytest.MonkeyPatch, coordinates: list
) -> None:
    """Reject malformed coordinates while preserving borrowed readers."""
    reader = VirtualWSIReader(np.zeros((8, 8, 3), dtype=np.uint8))
    close = Mock(wraps=reader.close)
    monkeypatch.setattr(reader, "close", close)
    monkeypatch.setattr(
        samplers,
        "get_patch_extractor",
        Mock(return_value=SimpleNamespace(coordinate_list=coordinates)),
    )
    with pytest.raises(ValueError, match="Generated coordinates must have shape"):
        generate_slide_patch_coordinates(reader, 4)
    close.assert_not_called()


@pytest.mark.parametrize("labels", [[], [[0, 1]]])
def test_invalid_sampler_labels(labels: list) -> None:
    """Labels must be a nonempty one-dimensional sequence."""
    with pytest.raises(ValueError, match="`labels`"):
        ClassBalancedIndexSampler(labels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ignore_labels": {0, 1}}, "No samples remain"),
        ({"num_samples": 0}, "positive integer"),
        ({"num_samples": 4, "replacement": False}, "cannot exceed"),
    ],
)
def test_invalid_balanced_sample_counts(kwargs: dict, message: str) -> None:
    """Sampling counts respect ignored labels and replacement policy."""
    with pytest.raises(ValueError, match=message):
        ClassBalancedIndexSampler([0, 0, 1], **kwargs)


@pytest.mark.parametrize("container", [list, np.asarray, torch.tensor])
def test_balanced_weights_and_ignored_labels(container: object) -> None:
    """Each class receives equal total weight and ignored samples are never drawn."""
    sampler = ClassBalancedIndexSampler(
        container([0, 0, 1, -1]),
        ignore_labels={-1},
        replacement=False,
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(
        sampler.weights, torch.tensor([0.5, 0.5, 1, 0], dtype=torch.double)
    )
    assert len(sampler) == 3
    assert sorted(sampler) == [0, 1, 2]


def test_grouped_sampler_uses_global_seed() -> None:
    """Without a generator, torch's global seed still makes an epoch reproducible."""
    sampler = GroupedBlockSampler([0, 0, 1, 1, 1], block_size=2)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        first = list(sampler)
        torch.manual_seed(19)
        assert list(sampler) == first
    assert len(sampler) == 5
    assert sorted(first) == list(range(5))


@pytest.mark.parametrize("num_samples", [0, 1])
def test_split_requires_two_samples(num_samples: int) -> None:
    """Training and validation each need at least one sample."""
    with pytest.raises(ValueError, match="at least two"):
        split_indices(num_samples)
