"""Dataset input validation, target conversion, and worker resource ownership."""

from __future__ import annotations

import os
import pickle
from contextlib import closing
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from shapely.geometry import box

from tiatoolbox.annotation import Annotation, DictionaryStore, SQLiteStore
from tiatoolbox.models.training import (
    MaskTargetBuilder,
    PatchAnnotationDataset,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    PresenceTargetBuilder,
    SlideAnnotationPatchDataset,
)
from tiatoolbox.models.training.augmentations import (
    TrainingAugmentationPreset,
    _RandomZoom,
    _ScaleJitter,
    _window_layout,
)
from tiatoolbox.models.training.datasets import (
    _ensure_tensor_image,
    _ensure_tensor_mask,
    _ensure_tensor_target,
    _load_image,
    _load_mask,
)
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3, 4)])
def test_invalid_image_rank(shape: tuple) -> None:
    """Images must describe one two-dimensional patch."""
    with pytest.raises(ValueError, match="Images must have shape"):
        _ensure_tensor_image(np.zeros(shape))


def test_unreadable_image(tmp_path: Path) -> None:
    """Corrupt images fail at decoding with the offending path."""
    path = tmp_path / "corrupt.png"
    path.write_text("not an image")
    with pytest.raises(ValueError, match=f"Unable to read image: {path}"):
        _load_image(path)


@pytest.mark.parametrize("channels", [1, 3])
def test_repeated_mask_channels_are_labels(tmp_path: Path, channels: int) -> None:
    """Singleton or identical mask channels preserve the original class indices."""
    expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = tmp_path / "mask.npy"
    np.save(path, np.repeat(expected[..., None], channels, axis=-1))
    np.testing.assert_array_equal(_load_mask(path), expected)


@pytest.mark.parametrize("layout", ["hw", "chw", "hwc", "numpy_hwc"])
def test_tensor_masks_preserve_labels(layout: str) -> None:
    """Mask conversion accepts singleton tensor axes without changing labels."""
    expected = torch.arange(12).reshape(3, 4)
    masks = {
        "hw": expected,
        "chw": expected[None],
        "hwc": expected[..., None],
        "numpy_hwc": expected.numpy()[..., None],
    }
    torch.testing.assert_close(_ensure_tensor_mask(masks[layout]), expected)


def test_nested_target_conversion() -> None:
    """Nested targets retain their containers, tensor identity, shapes, and values."""
    tensor = torch.tensor([0.25])
    converted = _ensure_tensor_target(
        {"heads": [tensor, (np.array(2), np.ones((3, 4, 2), np.float32))]}
    )
    assert isinstance(converted["heads"], list)
    assert converted["heads"][0] is tensor
    scalar, dense = converted["heads"][1]
    assert isinstance(converted["heads"][1], tuple)
    assert scalar.shape == ()
    assert scalar.item() == 2
    torch.testing.assert_close(dense, torch.ones(2, 3, 4))


@pytest.mark.parametrize("root_kind", ["missing", "file", "empty"])
def test_invalid_classification_root(tmp_path: Path, root_kind: str) -> None:
    """Classification discovery requires a directory containing class folders."""
    root = tmp_path / root_kind
    if root_kind == "file":
        root.touch()
    elif root_kind == "empty":
        root.mkdir()
    message = "class directories" if root_kind == "empty" else "existing directory"
    with pytest.raises(ValueError, match=message):
        PatchFolderClassificationDataset(root)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Provide either"),
        ({"images": []}, "provided together"),
        ({"masks": []}, "provided together"),
        ({"images": [np.zeros((4, 4, 3))], "masks": []}, "same length"),
    ],
)
def test_invalid_mask_pair_sources(kwargs: dict, message: str) -> None:
    """Image and mask sources must both exist and have equal lengths."""
    with pytest.raises(ValueError, match=message):
        PatchMaskPairDataset(**kwargs)


@pytest.mark.parametrize("missing", ["image_dir", "mask_dir"])
def test_pair_directories_must_exist(tmp_path: Path, missing: str) -> None:
    """Both sides of directory pairing are validated before discovery."""
    kwargs = {
        "image_dir": tmp_path,
        "mask_dir": tmp_path,
        missing: tmp_path / "missing",
    }
    with pytest.raises(ValueError, match=missing):
        PatchMaskPairDataset(**kwargs)


def test_masks_without_images_are_reported(tmp_path: Path) -> None:
    """Extra mask files cannot be silently omitted from the training set."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (4, 4)).save(images / "paired.png")
    Image.new("L", (4, 4)).save(masks / "paired.png")
    Image.new("L", (4, 4)).save(masks / "orphan.png")
    with pytest.raises(ValueError, match="missing images for orphan"):
        PatchMaskPairDataset(images, masks)


@pytest.mark.parametrize("patch_inputs", [[], np.zeros((4, 4, 3)), "patch.png"])
def test_invalid_annotation_patch_inputs(patch_inputs: object) -> None:
    """Annotation datasets require a nonempty list or a supported array batch."""
    with pytest.raises(ValueError, match="`patch_inputs`"):
        PatchAnnotationDataset(patch_inputs, DictionaryStore(), MaskTargetBuilder())


@pytest.mark.parametrize("bounds", [[(0, 0, 2)], [(0, 0, 0, 2)], [(0, 2, 2, 1)]])
def test_invalid_patch_bounds(bounds: list) -> None:
    """Patch bounds must have four values and positive width and height."""
    with pytest.raises(ValueError, match=r"[Pp]atch.*bounds"):
        PatchAnnotationDataset(
            [np.zeros((4, 4, 3))],
            DictionaryStore(),
            MaskTargetBuilder(),
            patch_bounds=bounds,
        )


@pytest.mark.parametrize("as_paths", [False, True])
def test_annotation_patch_array_inputs(tmp_path: Path, *, as_paths: bool) -> None:
    """Object arrays of paths and stacked patches yield the same aligned sample."""
    image = np.full((4, 4, 3), 17, dtype=np.uint8)
    inputs = image[None]
    if as_paths:
        path = tmp_path / "image.npy"
        np.save(path, image)
        inputs = np.array([path], dtype=object)
    with PatchAnnotationDataset(
        inputs,
        DictionaryStore(),
        MaskTargetBuilder(),
        patch_bounds=np.array([[0, 0, 4, 4]]),
    ) as dataset:
        assert len(dataset) == 1
        torch.testing.assert_close(dataset[0]["image"], torch.full((3, 4, 4), 17.0))
        torch.testing.assert_close(
            dataset[0]["target"], torch.zeros(4, 4, dtype=torch.int64)
        )


@pytest.mark.parametrize("slides", [[], "slide.png"])
def test_invalid_slide_inputs(slides: object) -> None:
    """Slide datasets require a nonempty list."""
    with pytest.raises(ValueError, match="non-empty list"):
        SlideAnnotationPatchDataset(slides, DictionaryStore(), MaskTargetBuilder(), 4)


def test_unseeded_transforms_keep_their_own_random_state() -> None:
    """An unseeded dataset does not overwrite a caller's transform seed."""
    transform = Mock(side_effect=lambda image: image)
    dataset = PatchMaskPairDataset(
        images=[np.zeros((4, 4, 3))],
        masks=[np.zeros((4, 4))],
        image_transform=transform,
    )
    assert dataset[0]["image"].shape == (3, 4, 4)
    transform.set_random_seed.assert_not_called()
    dataset.set_epoch(0, 19)
    _ = dataset[0]
    transform.set_random_seed.assert_called_once_with(dataset._sample_seed(0, 1))


def test_file_store_serialization_requires_commit(tmp_path: Path) -> None:
    """Workers reopen committed file stores and cannot lose uncommitted annotations."""
    path = tmp_path / "annotations.db"
    with closing(SQLiteStore(path, auto_commit=False)) as store:
        store.append(Annotation(box(0, 0, 4, 4)))
        with PatchAnnotationDataset(
            [np.zeros((4, 4, 3))], store, PresenceTargetBuilder()
        ) as dataset:
            with pytest.raises(
                ValueError, match="Commit file-backed annotation stores"
            ):
                pickle.dumps(dataset)
            store.commit()
            with pickle.loads(pickle.dumps(dataset)) as restored:  # noqa: S301 - own snapshot
                assert restored.annotation_store_specs == [path]
                assert restored[0]["target"].item() == 1
            assert len(store) == 1


@pytest.mark.parametrize("dataset_kind", ["patch", "slide"])
def test_inherited_dataset_reopens_owned_handles(
    tmp_path: Path, dataset_kind: str
) -> None:
    """A process change replaces cached handles while preserving borrowed data."""
    path = tmp_path / "annotations.db"
    with closing(SQLiteStore(path)) as store:
        store.append(Annotation(box(0, 0, 4, 4)))
    cls = (
        PatchAnnotationDataset
        if dataset_kind == "patch"
        else SlideAnnotationPatchDataset
    )
    kwargs = {} if dataset_kind == "patch" else {"patch_size": 4}
    image = np.zeros((8, 8, 3), np.uint8)
    source = image if dataset_kind == "patch" else VirtualWSIReader(image)
    with cls([source], path, PresenceTargetBuilder(), **kwargs) as dataset:
        expected = dataset[0]
        previous = dataset._get_store(0)
        dataset._resource_pid = -1  # Simulate inheriting the cache through fork.
        actual = dataset[0]
        assert dataset._resource_pid == os.getpid()
        assert dataset._get_store(0) is not previous
        assert previous.cons == {}
        torch.testing.assert_close(actual["image"], expected["image"])
        torch.testing.assert_close(actual["target"], expected["target"])


@pytest.mark.parametrize("scale", [1.0, 0.0001])
def test_annotation_zoom_identity_and_subpixel_fallback(scale: float) -> None:
    """Identity or subpixel zoom windows retain the original image and annotations."""
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    builder = MaskTargetBuilder()
    zoom = _ScaleJitter(_RandomZoom((scale, scale), 1), "mask")
    preset = TrainingAugmentationPreset(scale_jitter=zoom)
    with PatchAnnotationDataset(
        [image], DictionaryStore(), builder, pair_transform=preset
    ) as dataset:
        dataset.set_epoch(0, 19)
        zoom.set_random_seed(dataset._sample_seed(0, 0))
        window, shape = zoom.sample_window((4, 4))
        if scale < 1:
            assert _window_layout(window, (4, 4), shape) is None
        sample = dataset[0]
        torch.testing.assert_close(
            sample["image"], torch.from_numpy(image).permute(2, 0, 1).float()
        )
        torch.testing.assert_close(
            sample["target"], torch.zeros(4, 4, dtype=torch.int64)
        )


def test_slide_zoom_can_extend_past_slide() -> None:
    """Disabling within_bound preserves a centred zoom even outside slide edges."""
    preset = TrainingAugmentationPreset(
        scale_jitter=_ScaleJitter(_RandomZoom((0.5, 0.5), 1), "mask")
    )
    reader = VirtualWSIReader(np.zeros((16, 16, 3), np.uint8))
    with SlideAnnotationPatchDataset(
        [reader],
        DictionaryStore(),
        MaskTargetBuilder(),
        8,
        within_bound=False,
        pair_transform=preset,
    ) as dataset:
        sample = dataset[0]
        assert sample["bounds"].tolist() == [-4, -4, 12, 12]
        assert sample["image"].shape == (3, 8, 8)
        assert sample["target"].shape == (8, 8)


def test_slide_jitter_can_extend_past_slide(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unbounded jitter retains the sampled offset at the edge of a slide."""
    reader = VirtualWSIReader(np.zeros((16, 16, 3), np.uint8))
    with SlideAnnotationPatchDataset(
        [reader],
        DictionaryStore(),
        MaskTargetBuilder(),
        8,
        within_bound=False,
        jitter=6,
    ) as dataset:
        monkeypatch.setattr(dataset, "_sample_seed", Mock(return_value=0))
        sample = dataset[-1]
        # Seed zero draws the offset (5, 2), which is not clipped to the slide.
        assert sample["bounds"].tolist() == [13, 10, 21, 18]
        assert sample["image"].shape == (3, 8, 8)
