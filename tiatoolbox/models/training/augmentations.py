"""Preconfigured augmentation presets for TIAToolbox training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import albumentations as A
import cv2
import numpy as np

from tiatoolbox.models.training.targets import (
    SpatialTargetSpec,
    TargetBuilderABC,
)
from tiatoolbox.tools.stainaugment import StainAugmentor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

AugmentationLevel = Literal["light", "medium", "heavy"]
SpatialTargetKind = Literal["mask", "image"]
_TargetPath = tuple[str, ...]


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


def _normalize_spatial_target_kind(kind: str) -> SpatialTargetKind:
    """Validate a spatial target interpolation kind."""
    if kind not in {"mask", "image"}:
        msg = (
            f"Unsupported spatial target kind `{kind}`. "
            "Choose from `mask` or `image`."
        )
        raise ValueError(msg)
    return kind


def _flatten_spatial_target_spec(
    spec: SpatialTargetSpec,
    *,
    path: _TargetPath = (),
) -> dict[_TargetPath, SpatialTargetKind]:
    """Flatten a nested spatial target spec into leaf paths."""
    if isinstance(spec, str):
        return {path: _normalize_spatial_target_kind(spec)}

    flat_spec: dict[_TargetPath, SpatialTargetKind] = {}
    for key, child_spec in spec.items():
        if not isinstance(key, str):
            msg = "Structured spatial target spec keys must be strings."
            raise ValueError(msg)
        flat_spec.update(
            _flatten_spatial_target_spec(child_spec, path=(*path, key)),
        )
    return flat_spec


def _select_spatial_target(target: object, path: _TargetPath) -> np.ndarray:
    """Select one spatial target leaf from a possibly nested target object."""
    value = target
    for key in path:
        if not isinstance(value, dict) or key not in value:
            joined_path = ".".join(path) or "<root>"
            msg = f"Structured target is missing spatial key `{joined_path}`."
            raise ValueError(msg)
        value = value[key]

    if not isinstance(value, np.ndarray):
        joined_path = ".".join(path) or "<root>"
        msg = (
            f"Spatial target `{joined_path}` must be a NumPy array before "
            "dataset tensor conversion."
        )
        raise ValueError(msg)
    return value


def _replace_spatial_target(
    target: object,
    path: _TargetPath,
    replacement: np.ndarray,
) -> object:
    """Return a target object with one spatial leaf replaced."""
    if not path:
        return replacement

    if not isinstance(target, dict):
        joined_path = ".".join(path)
        msg = (
            f"Structured target path `{joined_path}` could not be updated because "
            "an intermediate value is not a dictionary."
        )
        raise ValueError(msg)

    key = path[0]
    if key not in target:
        joined_path = ".".join(path)
        msg = f"Structured target is missing spatial key `{joined_path}`."
        raise ValueError(msg)

    updated_target = dict(target)
    updated_target[key] = _replace_spatial_target(
        target[key],
        path[1:],
        replacement,
    )
    return updated_target


class _AlbumentationsSpatialPairTransform:
    """Wrap an Albumentations image+target pipeline for structured targets."""

    def __init__(
        self,
        ops: list[A.BasicTransform],
        target_spec: SpatialTargetSpec,
    ) -> None:
        flat_target_spec = _flatten_spatial_target_spec(target_spec)
        if not flat_target_spec:
            msg = "Structured spatial target specs must contain at least one leaf."
            raise ValueError(msg)

        self.path_to_name = {
            path: f"target_{index}"
            for index, path in enumerate(sorted(flat_target_spec))
        }
        self.transform = A.Compose(
            ops,
            additional_targets={
                name: flat_target_spec[path]
                for path, name in self.path_to_name.items()
            },
        )

    def __call__(
        self,
        image: np.ndarray,
        target: object,
    ) -> tuple[np.ndarray, object]:
        """Apply paired geometric transforms to an image and spatial targets."""
        payload = {"image": image}
        for path, name in self.path_to_name.items():
            payload[name] = _select_spatial_target(target, path)

        transformed = self.transform(**payload)
        updated_target = target
        for path, name in self.path_to_name.items():
            updated_target = _replace_spatial_target(
                updated_target,
                path,
                transformed[name],
            )

        return transformed["image"], updated_target


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
    target_spec: SpatialTargetSpec,
) -> Callable | None:
    """Convert paired Albumentations transforms into a dataset callable."""
    if not ops:
        return None
    return _AlbumentationsSpatialPairTransform(ops, target_spec)


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
        pair_transform=_wrap_pair_ops(_geometric_ops(normalized_level), "mask"),
        image_transform=_wrap_image_ops(_image_ops(normalized_level)),
    )


def get_annotation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    target_builder: TargetBuilderABC | None = None,
    spatial_target_spec: SpatialTargetSpec | None = None,
) -> TrainingAugmentationPreset:
    """Return a preset for annotation-backed training datasets.

    Spatial targets share geometric transforms with the image using the
    interpolation modes declared by the target builder. Non-spatial target
    leaves, such as scalar or vector labels, are left untouched.
    """

    normalized_level = _validate_level(level)
    resolved_spatial_target_spec = spatial_target_spec
    if resolved_spatial_target_spec is None and target_builder is not None:
        resolved_spatial_target_spec = target_builder.spatial_target_spec

    if resolved_spatial_target_spec is not None:
        return TrainingAugmentationPreset(
            pair_transform=_wrap_pair_ops(
                _geometric_ops(normalized_level),
                resolved_spatial_target_spec,
            ),
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
