"""Preconfigured augmentation presets for TIAToolbox training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import albumentations as A
import cv2
import numpy as np

from tiatoolbox.models.training.targets import MaskTargetBuilder, TargetBuilderABC
from tiatoolbox.tools.stainaugment import StainAugmentor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

AugmentationLevel = Literal["light", "medium", "heavy"]


@dataclass(frozen=True)
class TrainingAugmentationPreset:
    """Bundle of dataset-ready augmentation callables."""

    transform: Callable | None = None
    pair_transform: Callable | None = None
    image_transform: Callable | None = None
    target_transform: Callable | None = None


class _AlbumentationsImageTransform:
    """Wrap an Albumentations image pipeline as a plain callable."""

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the wrapped Albumentations pipeline."""
        return self.transform(image=image)["image"]


class _AlbumentationsPairTransform:
    """Wrap an Albumentations image+mask pipeline as a plain callable."""

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(
        self,
        image: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the wrapped Albumentations pipeline to an image and target."""
        transformed = self.transform(image=image, mask=target)
        return transformed["image"], transformed["mask"]


def _validate_level(level: AugmentationLevel) -> AugmentationLevel:
    """Validate supported preset names."""
    if level not in {"light", "medium", "heavy"}:
        msg = (
            f"Unsupported augmentation preset `{level}`. "
            "Choose from `light`, `medium`, or `heavy`."
        )
        raise ValueError(msg)
    return level


def _wrap_image_ops(
    ops: list[A.BasicTransform],
) -> Callable | None:
    """Convert image-only Albumentations transforms into a dataset callable."""
    if not ops:
        return None
    return _AlbumentationsImageTransform(A.Compose(ops))


def _wrap_pair_ops(
    ops: list[A.BasicTransform],
) -> Callable | None:
    """Convert paired Albumentations transforms into a dataset callable."""
    if not ops:
        return None
    return _AlbumentationsPairTransform(A.Compose(ops))


def _light_geometric_ops() -> list[A.BasicTransform]:
    """Return the light paired geometric transforms."""
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]


def _medium_affine_op() -> A.BasicTransform:
    """Return a moderate affine perturbation."""
    return A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(-0.02, 0.02),
        rotate=(-10, 10),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        fill_mask=0,
        p=0.3,
    )


def _heavy_affine_op() -> A.BasicTransform:
    """Return a stronger affine perturbation."""
    return A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        rotate=(-20, 20),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        fill_mask=0,
        p=0.5,
    )


def _light_image_ops() -> list[A.BasicTransform]:
    """Return the light image-only augmentations."""
    return [
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.08,
                    contrast_limit=0.08,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            ],
            p=0.3,
        ),
    ]


def _medium_image_ops() -> list[A.BasicTransform]:
    """Return medium-strength image-only augmentations."""
    return [
        *_light_image_ops(),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ImageCompression(quality_range=(85, 100), p=1.0),
            ],
            p=0.2,
        ),
        StainAugmentor(method='macenko', sigma1=0.2, sigma2=0.05, p=0.2),
    ]


def _heavy_image_ops() -> list[A.BasicTransform]:
    """Return heavy image-only augmentations."""
    return [
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.ImageCompression(quality_range=(70, 95), p=1.0),
            ],
            p=0.35,
        ),
        StainAugmentor(method='macenko', sigma1=0.35, sigma2=0.1, p=0.4),
    ]


def _geometric_ops(level: AugmentationLevel) -> list[A.BasicTransform]:
    """Build paired geometric operations for a preset level."""
    ops = _light_geometric_ops()
    if level == "medium":
        ops.append(_medium_affine_op())
    if level == "heavy":
        ops.append(_heavy_affine_op())
    return ops


def _image_ops(level: AugmentationLevel) -> list[A.BasicTransform]:
    """Build image-only operations for a preset level."""
    if level == "light":
        return _light_image_ops()
    if level == "medium":
        return _medium_image_ops()
    return _heavy_image_ops()


def get_classification_augmentation(
    level: AugmentationLevel = "medium",
) -> TrainingAugmentationPreset:
    """Return a preset for patch classification datasets."""
    normalized_level = _validate_level(level)
    ops = [*_geometric_ops(normalized_level), *_image_ops(normalized_level)]
    return TrainingAugmentationPreset(transform=_wrap_image_ops(ops))


def get_segmentation_augmentation(
    level: AugmentationLevel = "medium",
) -> TrainingAugmentationPreset:
    """Return a preset for paired patch/mask segmentation datasets."""
    normalized_level = _validate_level(level)
    return TrainingAugmentationPreset(
        pair_transform=_wrap_pair_ops(_geometric_ops(normalized_level)),
        image_transform=_wrap_image_ops(_image_ops(normalized_level)),
    )


def get_annotation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    target_builder: TargetBuilderABC | None = None,
    spatial_targets: bool | None = None,
) -> TrainingAugmentationPreset:
    """Return a preset for annotation-backed training datasets.

    Spatial targets can safely share geometric transforms with the image,
    while scalar or vector targets should use image-only perturbations.
    """

    normalized_level = _validate_level(level)
    if spatial_targets is None:
        spatial_targets = isinstance(target_builder, MaskTargetBuilder)

    if spatial_targets:
        return TrainingAugmentationPreset(
            pair_transform=_wrap_pair_ops(_geometric_ops(normalized_level)),
            image_transform=_wrap_image_ops(_image_ops(normalized_level)),
        )

    return TrainingAugmentationPreset(
        image_transform=_wrap_image_ops(_image_ops(normalized_level)),
    )


__all__ = [
    "AugmentationLevel",
    "TrainingAugmentationPreset",
    "get_annotation_augmentation",
    "get_classification_augmentation",
    "get_segmentation_augmentation",
]
