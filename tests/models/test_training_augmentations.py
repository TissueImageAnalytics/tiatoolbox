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
    _RandomZoom,
    _sample_zoom_window,
    _ScaleJitter,
    _StainAugmentation,
    _StrongAugmentation,
    _TorchvisionImageTransform,
    _TorchvisionSpatialPairTransform,
    _TrainingAffine,
    _Uint8Transform,
    _window_layout,
    _zoom_image,
)
from tiatoolbox.tools.strongaugment import StrongAugment


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
    preset = get_segmentation_augmentation(level)
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


def test_strong_preset_pairs_heavy_geometry_with_strong_augment() -> None:
    """``strong`` swaps only the appearance operations of ``heavy``."""
    strong = get_segmentation_augmentation("strong")
    heavy = get_segmentation_augmentation("heavy")
    assert repr(strong.pair_transform.transform) == repr(heavy.pair_transform.transform)
    assert strong.scale_jitter is None  # Zooms are opt-in at every level.
    assert heavy.scale_jitter is None
    appearance = strong.image_transform
    (operation,) = appearance.transform.transforms
    assert isinstance(operation, _Uint8Transform)
    assert isinstance(operation.transform, _StrongAugmentation)
    assert repr(operation.transform.augmentor) == repr(StrongAugment())


def test_strong_augmentation_handles_channels_and_no_ops() -> None:
    """Greyscale changes, other channel counts pass, and no-ops keep exact floats."""
    pipeline = _TorchvisionImageTransform(
        v2.Compose(
            [
                _Uint8Transform(
                    _StrongAugmentation(
                        StrongAugment(operations=(0, 3), probabilities=(0.5, 0.5))
                    )
                )
            ]
        )
    )
    rgb = np.random.default_rng(3).random((16, 16, 3), dtype=np.float32)
    grey = np.random.default_rng(3).integers(0, 256, (16, 16), dtype=np.uint8)
    multiplexed = rgb[..., :2]
    outputs, grey_outputs = [], []
    for seed in range(16):
        pipeline.set_random_seed(seed)
        outputs.append(pipeline(rgb))
        pipeline.set_random_seed(seed)
        grey_outputs.append(pipeline(grey))
        pipeline.set_random_seed(seed)
        np.testing.assert_array_equal(pipeline(multiplexed), multiplexed)
    assert any(np.array_equal(output, rgb) for output in outputs)
    assert any(not np.array_equal(output, rgb) for output in outputs)
    assert all(output.shape == grey.shape for output in grey_outputs)
    assert any(not np.array_equal(output, grey) for output in grey_outputs)


@pytest.mark.parametrize("level", ["heavy", "strong"])
def test_seeded_presets_leave_training_random_streams_unchanged(level: str) -> None:
    """Augmenting in the main process must not reseed dropout or CUDA training."""
    image = np.random.default_rng(4).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    preset = get_segmentation_augmentation(level, scale_jitter=0.1, crop_size=12)
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()  # noqa: NPY002 - inspect the legacy worker RNG
    python_state = random.getstate()
    for seed in range(12):
        preset.set_random_seed(seed)
        preset(image, np.zeros((16, 16), np.int64))
        preset.set_random_seed(seed)
        preset.scale_jitter.sample_window((16, 16), centred=True)
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


def _affines(ops: list[torch.nn.Module]) -> list[_TrainingAffine]:
    """Return the free rotations among a pipeline's operations."""
    return [
        op.transforms[0]
        for op in ops
        if isinstance(op, v2.RandomApply)
        and isinstance(op.transforms[0], _TrainingAffine)
    ]


@pytest.mark.parametrize("fill", [0, 255])
def test_free_rotation_is_opt_in_and_forwards_target_fill(fill: int) -> None:
    """Presets pad only when asked to rotate, filling masks with ``target_fill``."""
    for level in ("light", "medium", "heavy", "strong"):
        assert not _affines(
            get_segmentation_augmentation(level).pair_transform.transform.transforms
        )
        assert not _affines(
            get_classification_augmentation(level).image_transform.transform.transforms
        )
    presets = (
        get_segmentation_augmentation("heavy", target_fill=fill, rotation=15),
        get_annotation_augmentation(
            "heavy",
            spatial_target_spec={"mask": "mask"},
            target_fill=fill,
            rotation=15,
        ),
    )
    for preset in presets:
        pair = preset.pair_transform
        assert pair.target_fill == fill
        (rotation,) = _affines(pair.transform.transforms)
        assert rotation.target_fill == fill
        assert rotation.degrees == [-15.0, 15.0]
        assert rotation.translate is None
        assert rotation.scale is None
    zoomed = get_segmentation_augmentation("heavy", target_fill=fill, scale_jitter=0.1)
    assert zoomed.scale_jitter.target_fill == fill
    assert zoomed.scale_jitter.zoom.target_fill == fill
    with pytest.raises(ValueError, match="rotation"):
        get_segmentation_augmentation(rotation=-1)


def test_zoom_windows_are_unbiased_and_contain_or_fit_the_input() -> None:
    """Zoom factors span the whole range; larger windows contain the input."""
    rng = np.random.default_rng(0)
    shape, crop = (40, 60), (32, 48)
    windows, centred, crop_factors = set(), set(), []
    for _ in range(300):
        window = _sample_zoom_window(rng, shape, shape, (0.8, 1.25), p=1)
        left, top, right, bottom = window
        factor = 40 / (bottom - top)
        assert 0.8 - 0.05 <= factor <= 1.25 + 0.05
        windows.add(window)
        crop_window = _sample_zoom_window(rng, shape, crop, (0.8, 1.25), p=1)
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
        window = _sample_zoom_window(rng, shape, shape, (0.8, 1.25), p=1, centred=True)
        left, top, right, bottom = window
        assert 0.8 - 0.05 <= 40 / (bottom - top) <= 1.25 + 0.05
        assert abs((left + right) - 60) <= 1
        assert abs((top + bottom) - 40) <= 1
        centred.add(window)
    assert len(windows) > 100
    factors = [40 / (bottom - top) for _, top, _, bottom in windows]
    assert min(factors) < 0.85
    assert max(factors) > 1.2
    assert sum(factor < 1 for factor in factors) > 100  # Not clipped at one.
    assert min(crop_factors) < 0.85
    assert max(crop_factors) > 1.2
    assert any(right - left > 60 for left, _, right, _ in centred)
    assert any(right - left < 60 for left, _, right, _ in centred)
    assert _sample_zoom_window(rng, shape, shape, (0.8, 1.25), p=0) == (0, 0, 60, 40)
    with pytest.raises(ValueError, match="do not fit"):
        _sample_zoom_window(rng, shape, (48, 48), (1, 1), p=0)


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
def test_zooming_out_pads_images_white_and_targets_with_their_fill(
    dtype: type[np.generic],
) -> None:
    """A window larger than the patch shrinks it into a padded output."""
    rng = np.random.default_rng(8)
    image = rng.integers(10, 200, (40, 40, 3), dtype=np.uint8).astype(dtype)
    if dtype == np.float32:
        image /= 255
    labels = rng.integers(1, 5, (40, 40), dtype=np.int64)
    heat = rng.random((40, 40), dtype=np.float32) / 2 + 0.25
    zoomed = _zoom_image(image, (-20, -10, 60, 70), (40, 40))
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
    jitter = _ScaleJitter(
        _RandomZoom((0.5, 0.5), 1.0, target_fill=-1),
        {"labels": "mask", "heat": "image"},
    )
    jitter.set_random_seed(3)
    zoomed, target = jitter(image, {"labels": labels, "heat": heat})
    inside = target["labels"] != -1
    assert 0.2 < inside.mean() < 0.3  # A quarter of the output is the patch.
    assert set(np.unique(target["labels"][inside])) <= set(np.unique(labels))
    assert np.all(target["heat"][~inside] == 0)
    assert np.all(target["heat"][inside] >= 0.25)
    assert np.all(zoomed[~inside] == white)
    assert np.all(zoomed[inside] != white)
    padded = jitter.pad_target(
        {"labels": labels[:20, :20], "heat": heat[:20, :20]}, (10, 5, 30, 25), (40, 40)
    )
    np.testing.assert_array_equal(padded["labels"][box], labels[:20, :20])
    assert np.all(padded["labels"][outside] == -1)
    assert np.all(padded["heat"][outside] == 0)
    with pytest.raises(ValueError, match="cannot be represented"):
        jitter.pad_target(
            {"labels": labels[:20, :20].astype(np.uint8), "heat": heat[:20, :20]},
            (10, 5, 30, 25),
            (40, 40),
        )


def test_crops_resize_every_spatial_target() -> None:
    """Crops change the shape of images, masks and each stacked channel alike."""
    spec = {
        "stack": StackedSpatialTargetSpec((("mask", "mask"), ("heat", "image"))),
        "mask": "mask",
    }
    preset = get_annotation_augmentation(
        "light", spatial_target_spec=spec, scale_jitter=0.1, crop_size=(12, 20)
    )
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, (16, 24, 3), dtype=np.uint8)
    target = {
        "stack": rng.random((16, 24, 2), dtype=np.float32),
        "mask": rng.integers(0, 3, (16, 24)),
        "label": 3,
    }
    for seed in range(6):
        preset.set_random_seed(seed)
        output, output_target = preset(image, target)
        assert output.shape == (12, 20, 3)
        assert output_target["stack"].shape == (12, 20, 2)
        assert output_target["mask"].shape == (12, 20)
        assert output_target["label"] == 3
    partial = _ScaleJitter(
        _RandomZoom((1, 1), 1, (12, 20)), StackedSpatialTargetSpec((("mask", "mask"),))
    )
    with pytest.raises(ValueError, match="returned shape"):
        partial(image, np.zeros((16, 24, 2), np.float32))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_scale_jitter_zooms_arrays_without_padding(
    dtype: type[np.generic],
) -> None:
    """Arrays zoom in: labels stay exact and no padding value appears."""
    rng = np.random.default_rng(5)
    image = rng.integers(10, 200, (32, 32, 3), dtype=np.uint8).astype(dtype)
    labels = np.repeat(np.arange(32, dtype=np.int64)[None] * (2**40 + 1), 32, axis=0)
    heat = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    jitter = _ScaleJitter(
        _RandomZoom((1.5, 1.5), 1.0), {"labels": "mask", "heat": "image"}
    )
    outputs = []
    for seed in range(2):
        jitter.set_random_seed(seed)
        zoomed, target = jitter(image, {"labels": labels, "heat": heat})
        jitter.set_random_seed(seed)
        replay, _ = jitter(image, {"labels": labels, "heat": heat})
        np.testing.assert_array_equal(replay, zoomed)
        assert zoomed.shape == image.shape
        assert zoomed.dtype == image.dtype
        assert zoomed.min() >= image.min()
        assert zoomed.max() <= image.max()
        # A 1.5x zoom repeats about every third column of a 21-pixel window.
        columns = target["labels"][0]
        assert set(columns) <= set(labels[0])
        assert 19 <= len(set(columns)) <= 22
        assert np.all(np.diff(columns) >= 0)
        assert target["heat"].dtype == np.float32
        assert 0 <= target["heat"].min() < target["heat"].max() <= 1
        outputs.append(zoomed)
    assert not np.array_equal(outputs[0], outputs[1])


def test_zooms_and_crops_are_opt_in() -> None:
    """Presets zoom or crop only when asked, and validate the settings."""
    default = get_classification_augmentation("heavy").image_transform
    assert not any(isinstance(op, _RandomZoom) for op in default.transform.transforms)
    pipeline = get_classification_augmentation(
        "heavy", scale_jitter=0.1, crop_size=(12, 20)
    ).image_transform
    (zoom,) = [
        op for op in pipeline.transform.transforms if isinstance(op, _RandomZoom)
    ]
    assert zoom.scale == pytest.approx((0.9, 1.1))
    image = np.random.default_rng(1).integers(10, 200, (16, 24, 3), dtype=np.uint8)
    for seed in range(8):
        pipeline.set_random_seed(seed)
        assert pipeline(image).shape == (12, 20, 3)
    with pytest.raises(ValueError, match="scale_jitter"):
        get_classification_augmentation(scale_jitter=1.0)
    with pytest.raises(ValueError, match="crop_size"):
        get_segmentation_augmentation(crop_size=0)
    with pytest.raises(ValueError, match="spatial targets"):
        get_annotation_augmentation(scale_jitter=0.1)
