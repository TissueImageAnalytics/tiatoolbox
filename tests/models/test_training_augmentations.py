"""Regression coverage for augmentation presets and crop and zoom helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from tiatoolbox.models.training import TrainingAugmentationPreset, get_augmentation
from tiatoolbox.models.training import augmentations as training_augmentations
from tiatoolbox.models.training.augmentations import (
    _apply_appearance,
    _map_spatial,
    _pad_array,
    _sample_zoom_window,
    _StainAugmentation,
    _StrongAugmentation,
    _Uint8Transform,
    _validate_zoom,
    _window_layout,
    _zoom_image,
    _zoom_mask,
)
from tiatoolbox.tools.strongaugment import StrongAugment


def _seeded_appearance(
    operations: list[torch.nn.Module], image: np.ndarray, seed: int
) -> np.ndarray:
    """Apply appearance operations from a seed, as a preset does."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return _apply_appearance(v2.Compose(operations), image)


@pytest.mark.parametrize("level", ["light", "medium", "heavy", "strong"])
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
    preset = get_augmentation(level)
    results = []
    for seed in range(8):
        preset.set_random_seed(seed)
        transformed, target = preset(image, mask)
        preset.set_random_seed(seed)
        replay_image, replay_target = preset(image, mask)
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
    stain = _StainAugmentation(sigma1=0.4, sigma2=0.2, p=1)
    expected = _seeded_appearance([stain], image, 123)
    _seeded_appearance([stain], other, 987)
    np.testing.assert_array_equal(_seeded_appearance([stain], image, 123), expected)
    fresh = _StainAugmentation(sigma1=0.4, sigma2=0.2, p=1)
    np.testing.assert_array_equal(_seeded_appearance([fresh], image, 123), expected)
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(expected, image)


@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 1), (16, 16, 3)])
def test_stain_skips_grayscale_and_empty_tissue(shape: tuple[int, ...]) -> None:
    """Skipped stain operations preserve even non-quantized floating intensities."""
    stain = _Uint8Transform(_StainAugmentation(0.4, 0.2, p=1))
    image = np.full(shape, 0.98765, np.float32)
    np.testing.assert_array_equal(_seeded_appearance([stain], image, 7), image)


def test_strong_preset_uses_strong_augment_for_appearance() -> None:
    """``strong`` has StrongAugment as its only appearance operation."""
    (operation,) = get_augmentation("strong").appearance.transforms
    assert isinstance(operation, _Uint8Transform)
    assert isinstance(operation.transform, _StrongAugmentation)
    assert repr(operation.transform.augmentor) == repr(StrongAugment())


def test_strong_augmentation_handles_channels_and_no_ops() -> None:
    """Greyscale changes, other channel counts pass, and no-ops keep exact floats."""
    strong = _Uint8Transform(
        _StrongAugmentation(StrongAugment(operations=(0, 3), probabilities=(0.5, 0.5)))
    )
    rgb = np.random.default_rng(3).random((16, 16, 3), dtype=np.float32)
    grey = np.random.default_rng(3).integers(0, 256, (16, 16), dtype=np.uint8)
    multiplexed = rgb[..., :2]
    outputs, grey_outputs = [], []
    for seed in range(16):
        outputs.append(_seeded_appearance([strong], rgb, seed))
        grey_outputs.append(_seeded_appearance([strong], grey, seed))
        np.testing.assert_array_equal(
            _seeded_appearance([strong], multiplexed, seed), multiplexed
        )
    assert any(np.array_equal(output, rgb) for output in outputs)
    assert any(not np.array_equal(output, rgb) for output in outputs)
    assert all(output.shape == grey.shape for output in grey_outputs)
    assert any(not np.array_equal(output, grey) for output in grey_outputs)


@pytest.mark.parametrize("level", ["heavy", "strong"])
def test_seeded_presets_leave_training_random_streams_unchanged(level: str) -> None:
    """Augmenting in the main process must not reseed dropout or CUDA training."""
    image = np.random.default_rng(4).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    preset = get_augmentation(level)
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()  # noqa: NPY002 - inspect the legacy worker RNG
    python_state = random.getstate()
    for seed in range(12):
        preset.set_random_seed(seed)
        preset(image, np.zeros((16, 16), np.int64))
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
    preset = get_augmentation("heavy")

    def fail(*_inputs: object) -> None:
        """Advance the RNG before simulating a transform failure."""
        torch.rand(10)
        msg = "transform failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(preset, "appearance", fail)
    state = torch.get_rng_state()
    preset.set_random_seed(7)
    with pytest.raises(RuntimeError, match="transform failed"):
        preset(np.zeros((16, 16, 3), np.uint8), 0)
    torch.testing.assert_close(torch.get_rng_state(), state)


def test_unseeded_presets_draw_from_the_worker_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker's NumPy stream supplies a reproducible fallback seed."""
    monkeypatch.setattr(np.random, "randint", lambda *_args, **_kwargs: 19)
    preset = get_augmentation("heavy")
    image = np.random.default_rng(2).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    unseeded, _ = preset(image, 0)
    preset.set_random_seed(19)
    np.testing.assert_array_equal(preset(image, 0)[0], unseeded)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int64, np.float32])
def test_moves_keep_pairs_aligned_and_values_exact(dtype: type[np.generic]) -> None:
    """Every flip and turn reaches image and target alike, for any target dtype."""
    positions = np.arange(16, dtype=dtype).reshape(4, 4)
    if dtype == np.int64:
        positions = positions + 2**53 + 1  # Beyond float64's exact integers.
    preset = TrainingAugmentationPreset(torch.nn.Identity())
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    outcomes = set()
    for seed in range(64):
        preset.set_random_seed(seed)
        moved_image, moved_target = preset(image, positions)
        assert moved_target.dtype == positions.dtype
        np.testing.assert_array_equal(
            moved_target, positions.ravel()[moved_image.ravel()].reshape(4, 4)
        )
        assert moved_image.flags.c_contiguous
        assert moved_target.flags.c_contiguous
        outcomes.add(moved_image.tobytes())
    assert len(outcomes) == 8  # The four turns of the patch and of its mirror image.


def test_rectangular_patches_keep_their_shape() -> None:
    """Quarter turns never swap the height and width of rectangular patches."""
    preset = get_augmentation("heavy")
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    mask = np.zeros((16, 24), dtype=np.int64)
    for seed in range(24):
        preset.set_random_seed(seed)
        out_image, out_mask = preset(image, mask)
        assert (out_image.shape, out_mask.shape) == (image.shape, mask.shape)


def test_uint16_image_round_trip_preserves_white() -> None:
    """Restoring image dtype must not wrap full-intensity uint16 pixels to zero."""
    image = np.array([0, 1, 300, 32768, 65534, 65535], np.uint16).reshape((2, 3, 1))
    np.testing.assert_array_equal(
        _seeded_appearance([v2.ColorJitter()], image, 7), image
    )


def test_spatial_size_mismatch_is_rejected() -> None:
    """Targets with different source geometry must not silently diverge."""
    preset = get_augmentation("heavy")
    image = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="height and width"):
        preset(image, np.zeros((8, 8), np.float32))
    with pytest.raises(ValueError, match="height and width"):
        preset(image, {"nested": [torch.zeros(16, 16)]})


@pytest.mark.parametrize("nested", [False, True])
def test_moving_targets_leaves_their_sources_alone(
    monkeypatch: pytest.MonkeyPatch, *, nested: bool
) -> None:
    """Moved stacks are new arrays; read-only sources and labels stay as they were."""
    monkeypatch.setattr(
        training_augmentations, "_draw_moves", lambda _shape: (True, False, 0)
    )
    stack = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    stack.setflags(write=False)
    labels = np.array([1, 0])
    target = {"outer": {"stack": stack}, "labels": labels} if nested else stack
    _, result = get_augmentation("light")(np.zeros((16, 16, 3), np.uint8), target)
    transformed = result["outer"]["stack"] if nested else result
    np.testing.assert_array_equal(transformed, stack[:, ::-1])
    assert not np.shares_memory(transformed, stack)
    assert not stack.flags.writeable
    if nested:
        assert result["labels"] is labels


def test_zoom_windows_are_unbiased_and_contain_or_fit_the_input() -> None:
    """Zoom factors span the whole range; larger windows contain the input."""
    rng = np.random.default_rng(0)
    shape, crop = (40, 60), (32, 48)
    windows, centred, crop_factors = set(), set(), []
    unzoomed = 0
    for _ in range(600):
        window = _sample_zoom_window(rng, shape, shape, 0.2)
        left, top, right, bottom = window
        factor = 40 / (bottom - top)
        assert 0.8 - 0.05 <= factor <= 1.2 + 0.05
        unzoomed += window == (0, 0, 60, 40)
        windows.add(window)
        crop_window = _sample_zoom_window(rng, shape, crop, 0.2)
        crop_factors.append(32 / (crop_window[3] - crop_window[1]))
        for left, top, right, bottom in (window, crop_window):
            # Along each axis, a window that fits lies inside the input and a
            # larger one contains it.
            for size, start, end in ((60, left, right), (40, top, bottom)):
                if end - start <= size:
                    assert 0 <= start < end <= size
                else:
                    assert start <= 0
                    assert end >= size
        window = _sample_zoom_window(rng, shape, shape, 0.2, centred=True)
        left, top, right, bottom = window
        assert 0.8 - 0.05 <= 40 / (bottom - top) <= 1.2 + 0.05
        assert abs((left + right) - 60) <= 1
        assert abs((top + bottom) - 40) <= 1
        centred.add(window)
    assert 240 < unzoomed < 360  # Half the patches keep their scale.
    assert len(windows) > 100
    factors = [40 / (bottom - top) for _, top, _, bottom in windows]
    assert min(factors) < 0.85
    assert max(factors) > 1.15
    assert sum(factor < 1 for factor in factors) > 100  # Not clipped at one.
    assert min(crop_factors) < 0.85
    assert max(crop_factors) > 1.15
    assert any(right - left > 60 for left, _, right, _ in centred)
    assert any(right - left < 60 for left, _, right, _ in centred)
    assert _sample_zoom_window(rng, shape, shape, 0.0) == (0, 0, 60, 40)
    with pytest.raises(ValueError, match="do not fit"):
        _sample_zoom_window(rng, shape, (48, 48), 0.0)


def test_window_layout_locates_the_input_in_the_output() -> None:
    """The clipped window maps to a box in the output; the rest is padding."""
    assert _window_layout((0, 0, 60, 40), (40, 60), (40, 60)) == (
        (0, 0, 60, 40),
        (0, 0, 60, 40),
    )
    assert _window_layout((10, 5, 40, 25), (40, 60), (20, 30)) == (
        (10, 5, 40, 25),
        (0, 0, 30, 20),
    )
    # A window twice the input, offset so the input sits off-centre.
    assert _window_layout((-20, -30, 100, 50), (40, 60), (40, 60)) == (
        (0, 0, 60, 40),
        (10, 15, 40, 35),
    )
    assert _window_layout((70, 0, 130, 40), (40, 60), (40, 60)) is None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_zooming_out_pads_images_white_and_masks_with_their_fill(
    dtype: type[np.generic],
) -> None:
    """A window larger than the patch shrinks it into a padded output."""
    rng = np.random.default_rng(8)
    image = rng.integers(10, 200, (40, 40, 3), dtype=np.uint8).astype(dtype)
    if dtype == np.float32:
        image /= 255
    labels = rng.integers(1, 5, (40, 40), dtype=np.int64)
    window = (-20, -10, 60, 70)
    zoomed = _zoom_image(image, window, (40, 40))
    white = 1 if dtype == np.float32 else np.iinfo(dtype).max
    assert zoomed.shape == image.shape
    assert zoomed.dtype == dtype
    # The 40-pixel patch is resized to 20 pixels at (10, 5) in the output.
    box = np.s_[5:25, 10:30]
    outside = np.ones((40, 40), bool)
    outside[box] = False
    assert np.all(zoomed[outside] == white)
    np.testing.assert_array_equal(
        zoomed[box], _zoom_image(image, (0, 0, 40, 40), (20, 20))
    )
    mask = _zoom_mask(labels, window, (40, 40), -1)
    assert mask.dtype == labels.dtype
    assert np.all(mask[outside] == -1)
    np.testing.assert_array_equal(mask[box], labels[1::2, 1::2])
    with pytest.raises(ValueError, match="cannot be represented"):
        _zoom_mask(labels.astype(np.uint8), window, (40, 40), -1)


def test_image_windows_of_the_output_size_are_exact_crops() -> None:
    """Crops copy pixels; zooms resample them within the image's value range."""
    image = np.random.default_rng(1).integers(10, 200, (16, 24, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        _zoom_image(image, (3, 2, 23, 14), (12, 20)), image[2:14, 3:23]
    )
    zoomed = _zoom_image(image, (3, 2, 13, 8), (12, 20))
    assert zoomed.shape == (12, 20, 3)
    assert image.min() <= zoomed.min() <= zoomed.max() <= image.max()
    grey = image[..., 0]
    assert _zoom_image(grey, (0, 0, 24, 16), (8, 12)).shape == (8, 12)


@pytest.mark.parametrize(
    "mask",
    [
        np.repeat(np.arange(32, dtype=np.int64)[None] * (2**40 + 1), 32, axis=0),
        np.repeat(np.linspace(0.123, 0.987, 32, dtype=np.float32)[None], 32, axis=0),
        np.repeat(np.arange(32, dtype=np.uint16)[None, :, None] * 1000, 32, axis=0),
    ],
)
def test_masks_zoom_by_nearest_neighbour_with_exact_values(mask: np.ndarray) -> None:
    """Nearest-neighbour zooms keep large labels, floating values and channels."""
    zoomed = _zoom_mask(mask, (5, 5, 26, 26), (32, 32), 0)
    assert zoomed.shape == mask.shape
    assert zoomed.dtype == mask.dtype
    # A 1.5x zoom repeats about every third column of a 21-pixel window.
    columns = zoomed[0].squeeze()
    assert set(columns.tolist()) <= set(mask[0].squeeze().tolist())
    assert 19 <= len(set(columns.tolist())) <= 22
    assert np.all(np.diff(columns.astype(np.float64)) >= 0)


def test_pad_array_places_the_array_and_checks_the_fill() -> None:
    """Padding keeps channels; arrays already at the output shape are unchanged."""
    array = np.ones((2, 3, 2), np.float32)
    padded = _pad_array(array, (1, 2, 4, 4), (5, 5), -100)
    assert padded.shape == (5, 5, 2)
    assert padded.dtype == np.float32
    assert padded[2:4, 1:4].min() == 1
    assert (padded == -100).sum() == (25 - 6) * 2
    assert _pad_array(array, (0, 0, 3, 2), (2, 3), -100) is array
    for dtype, fill in ((np.uint8, 256), (np.uint8, -1), (np.int8, 255)):
        with pytest.raises(ValueError, match="cannot be represented"):
            _pad_array(np.ones((2, 3), dtype), (0, 0, 3, 2), (4, 4), fill)
    assert _pad_array(np.ones((2, 3), np.uint8), (0, 0, 3, 2), (4, 4), 255).max() == 255


def test_map_spatial_follows_the_shape_convention() -> None:
    """Spatial arrays follow nested containers; scalars and vectors stay as they are."""
    stack = np.zeros((4, 6, 3))
    labels = np.array([1, 0])
    target = {"stack": stack, "nested": [stack[..., 0], (2, labels)], "tensor": 1}
    target["scalar"] = torch.tensor(3)
    mapped = _map_spatial(target, (4, 6), lambda value: value + 1)
    assert mapped["stack"].min() == 1
    assert mapped["nested"][0].min() == 1
    assert isinstance(mapped["nested"], list)
    assert isinstance(mapped["nested"][1], tuple)
    assert mapped["nested"][1][0] == 2
    assert mapped["nested"][1][1] is labels
    assert mapped["scalar"] is target["scalar"]
    for value in (np.zeros((4, 4)), np.zeros((3, 4, 6)), torch.zeros(4, 6)):
        with pytest.raises(ValueError, match="height and width"):
            _map_spatial({"bad": value}, (4, 6), lambda item: item)


@pytest.mark.parametrize(
    ("scale_jitter", "crop_size", "message"),
    [
        (-0.1, None, "scale_jitter"),
        (1.0, None, "scale_jitter"),
        (0.0, 0, "crop_size"),
        (0.0, (4, 0), "crop_size"),
        (0.0, (4, 4, 4), "crop_size"),
    ],
)
def test_invalid_zoom_settings(
    scale_jitter: float, crop_size: object, message: str
) -> None:
    """Zoom ranges and crop sizes are checked when a dataset is built."""
    with pytest.raises(ValueError, match=message):
        _validate_zoom(scale_jitter, crop_size)


def test_zoom_settings_normalize_crop_sizes() -> None:
    """A single crop size means a square."""
    assert _validate_zoom(0, 12) == (0.0, (12, 12))
    assert _validate_zoom(0.1, [12, 20]) == (0.1, (12, 20))
    assert _validate_zoom(0.1, None) == (0.1, None)


def test_stain_augmentation_skips_tiny_tissue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Too few tissue pixels leave the original image untouched."""
    tissue = np.zeros((8, 8), dtype=bool)
    tissue[0, 0] = True
    monkeypatch.setattr(
        "tiatoolbox.models.training.augmentations.get_luminosity_tissue_mask",
        lambda *_args, **_kwargs: tissue,
    )
    image = torch.full((3, 8, 8), 128, dtype=torch.uint8)
    assert _StainAugmentation(0.1, 0.1, 1.0)(image) is image


def test_invalid_augmentation_level() -> None:
    """Unknown preset levels fail with supported names in the error."""
    with pytest.raises(ValueError, match="Unsupported augmentation preset"):
        get_augmentation("invalid")
