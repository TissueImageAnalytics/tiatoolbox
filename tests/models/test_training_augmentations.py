"""Regression coverage for the torchvision training preset backend."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from tiatoolbox.models.training import (
    StackedSpatialTargetSpec,
    get_annotation_augmentation,
    get_classification_augmentation,
    get_segmentation_augmentation,
)
from tiatoolbox.models.training.augmentations import (
    _StainAugmentation,
    _TorchvisionImageTransform,
    _TorchvisionSpatialPairTransform,
    _TrainingAffine,
    _Uint8Transform,
)


@pytest.mark.parametrize("level", ["light", "medium", "heavy"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_presets_replay_without_mutating_images_or_targets(
    level: str, dtype: type[np.generic]
) -> None:
    """Seeds replay geometry and appearance, preserving layouts and label values."""
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8).astype(dtype)
    if dtype == np.float32:
        image /= 255
    elif dtype == np.uint16:
        image *= 257
    # Reversed arrays exercise negative strides at the NumPy/Torch boundary.
    image = image[:, ::-1]
    mask = rng.choice(np.array([0, 300, 1001], np.uint16), size=(16, 16))[:, ::-1]
    original_image, original_mask = image.copy(), mask.copy()
    preset = get_segmentation_augmentation(level)
    results = []
    for seed in range(8):
        preset.pair_transform.set_random_seed(seed)
        preset.image_transform.set_random_seed(seed)
        transformed, target = preset.pair_transform(image, mask)
        transformed = preset.image_transform(transformed)
        preset.pair_transform.set_random_seed(seed)
        preset.image_transform.set_random_seed(seed)
        replay_image, replay_target = preset.pair_transform(image, mask)
        replay_image = preset.image_transform(replay_image)
        np.testing.assert_array_equal(replay_image, transformed)
        np.testing.assert_array_equal(replay_target, target)
        assert transformed.shape == image.shape
        assert transformed.dtype == dtype
        assert np.isfinite(transformed).all()
        assert target.dtype == mask.dtype
        assert set(np.unique(target)) <= {0, 300, 1001}
        results.append(transformed)
    assert any(not np.array_equal(results[0], value) for value in results[1:])
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(mask, original_mask)


def test_stain_replay_is_independent_of_patch_history() -> None:
    """Revisited patches and fresh workers produce identical seeded stains."""
    rng = np.random.default_rng(9)
    image, other = rng.integers(0, 200, (2, 16, 16, 3), dtype=np.uint8)
    original = image.copy()

    def build() -> _TorchvisionImageTransform:
        return _TorchvisionImageTransform(
            v2.Compose([_StainAugmentation(sigma1=0.4, sigma2=0.2, p=1)])
        )

    pipeline = build()
    pipeline.set_random_seed(123)
    expected = pipeline(image)
    pipeline.set_random_seed(987)
    pipeline(other)
    pipeline.set_random_seed(123)
    np.testing.assert_array_equal(pipeline(image), expected)
    fresh = build()
    fresh.set_random_seed(123)
    np.testing.assert_array_equal(fresh(image), expected)
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(expected, image)


@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 1), (16, 16, 3)])
def test_stain_skips_grayscale_and_empty_tissue(shape: tuple[int, ...]) -> None:
    """Skipped stain operations preserve even non-quantized floating intensities."""
    pipeline = _TorchvisionImageTransform(
        v2.Compose([_Uint8Transform(_StainAugmentation(0.4, 0.2, p=1))])
    )
    image = np.full(shape, 0.98765, np.float32)
    pipeline.set_random_seed(7)
    np.testing.assert_array_equal(pipeline(image), image)


def test_seeded_presets_leave_training_random_streams_unchanged() -> None:
    """Augmenting in the main process must not reseed dropout or CUDA training."""
    image = np.random.default_rng(4).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    preset = get_segmentation_augmentation("heavy")
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()  # noqa: NPY002 - inspect the legacy worker RNG
    python_state = random.getstate()
    for seed in range(12):
        preset.pair_transform.set_random_seed(seed)
        preset.image_transform.set_random_seed(seed)
        transformed, _ = preset.pair_transform(image, np.zeros((16, 16), np.int64))
        preset.image_transform(transformed)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    restored_cuda_states = torch.cuda.get_rng_state_all() if cuda_states else []
    for actual, expected in zip(restored_cuda_states, cuda_states, strict=True):
        torch.testing.assert_close(actual, expected)
    restored_numpy_state = np.random.get_state()  # noqa: NPY002
    np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_failed_transform_restores_torch_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the pipeline still restores the caller's RNG state."""
    pipeline = get_classification_augmentation("heavy").image_transform

    def fail(*_inputs: object) -> None:
        torch.rand(10)
        msg = "transform failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(pipeline, "transform", fail)
    state = torch.get_rng_state()
    pipeline.set_random_seed(7)
    with pytest.raises(RuntimeError, match="transform failed"):
        pipeline(np.zeros((16, 16, 3), np.uint8))
    torch.testing.assert_close(torch.get_rng_state(), state)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_affine_preserves_large_integer_labels(
    monkeypatch: pytest.MonkeyPatch, dtype: type[np.generic]
) -> None:
    """Preserve large label IDs and white padding across supported image dtypes."""
    op = _TrainingAffine(degrees=0)
    monkeypatch.setattr(op, "get_params", lambda *_args: (0, (3, 0), 1, (0, 0)))
    pipeline = _TorchvisionSpatialPairTransform([op], "mask")
    mask = np.full((16, 16, 1), 2**24 + 1, np.int64)
    image, target = pipeline(np.zeros((16, 16, 3), dtype), mask)
    white = 1 if dtype == np.float32 else np.iinfo(dtype).max
    assert np.all(image[:, :3] == white)
    assert np.all(target[:, :3] == 0)
    np.testing.assert_array_equal(target[:, 3:], mask[:, :-3])
    assert target.dtype == mask.dtype


def test_uint16_image_round_trip_preserves_white() -> None:
    """Restoring image dtype must not wrap full-intensity uint16 pixels to zero."""
    pipeline = _TorchvisionImageTransform(v2.Compose([v2.ColorJitter()]))
    image = np.array([0, 1, 300, 32768, 65534, 65535], np.uint16).reshape(2, 3, 1)
    pipeline.set_random_seed(7)
    np.testing.assert_array_equal(pipeline(image), image)


def test_spatial_size_mismatch_is_rejected() -> None:
    """Targets with different source geometry must not silently diverge."""
    pair = get_annotation_augmentation(
        "heavy", spatial_target_spec="image"
    ).pair_transform
    with pytest.raises(ValueError, match="same height and width"):
        pair(np.zeros((16, 16, 3), np.uint8), np.zeros((8, 8), np.float32))


@pytest.mark.parametrize("nested", [False, True])
def test_stacked_reconstruction_preserves_unselected_values(*, nested: bool) -> None:
    """Rebuilding root/nested stacks preserves inputs and non-spatial leaves."""
    stack = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    stack.setflags(write=False)
    spec = StackedSpatialTargetSpec((("mask", "mask"), ("heat", "image")))
    labels = np.array([1, 0])
    target = {"outer": {"stack": stack}, "labels": labels} if nested else stack
    pair = _TorchvisionSpatialPairTransform(
        [v2.RandomHorizontalFlip(p=1)],
        {"outer": {"stack": spec}} if nested else spec,
    )
    _, result = pair(np.zeros((16, 16, 3), np.uint8), target)
    transformed = result["outer"]["stack"] if nested else result
    np.testing.assert_array_equal(transformed[..., :2], stack[:, ::-1, :2])
    np.testing.assert_array_equal(transformed[..., 2], stack[..., 2])
    assert not np.shares_memory(transformed, stack)
    assert not stack.flags.writeable
    if nested:
        assert result["labels"] is labels


def test_mask_leaves_take_target_fill_under_affine_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded mask pixels get ``target_fill``; continuous leaves keep zero fill."""
    op = _TrainingAffine(degrees=0, target_fill=-100)
    monkeypatch.setattr(op, "get_params", lambda *_args: (0, (3, 0), 1, (0, 0)))
    transform = _TorchvisionSpatialPairTransform(
        [op], {"heat": "image", "mask": "mask"}, target_fill=-100
    )
    image = np.zeros((16, 16, 3), np.uint8)
    heat = np.ones((16, 16), np.float32)
    transformed, targets = transform(
        image, {"heat": heat, "mask": np.ones((16, 16), np.int64)}
    )
    assert np.all(transformed[:, :3] == 255)
    assert np.all(targets["heat"][:, :3] == 0)
    assert np.all(targets["heat"][:, 3:] == 1)
    assert np.all(targets["mask"][:, :3] == -100)
    assert np.all(targets["mask"][:, 3:] == 1)
    with pytest.raises(ValueError, match="cannot be represented by the uint8 mask"):
        transform(image, {"heat": heat, "mask": np.ones((16, 16), np.uint8)})


@pytest.mark.parametrize("fill", [0, 255])
def test_presets_forward_target_fill_to_affine_padding(fill: int) -> None:
    """Segmentation and annotation presets configure the affine mask fill."""
    presets = (
        get_segmentation_augmentation("heavy", target_fill=fill),
        get_annotation_augmentation(
            "heavy", spatial_target_spec={"mask": "mask"}, target_fill=fill
        ),
    )
    for preset in presets:
        pair = preset.pair_transform
        assert pair.target_fill == fill
        affines = [
            op.transforms[0]
            for op in pair.transform.transforms
            if isinstance(op, v2.RandomApply)
            and isinstance(op.transforms[0], _TrainingAffine)
        ]
        assert [op.target_fill for op in affines] == [fill]
