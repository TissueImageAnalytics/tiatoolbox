"""Preconfigured augmentation presets for TIAToolbox training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional

from tiatoolbox.models.training.targets import (
    SpatialTargetKind,
    SpatialTargetSpec,
    StackedSpatialTargetSpec,
    TargetBuilderABC,
)
from tiatoolbox.tools.stainaugment import StainAugmentor
from tiatoolbox.utils.misc import get_luminosity_tissue_mask

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

_IMAGE_DIMS = 3
_MIN_STAIN_PIXELS = 2

AugmentationLevel = Literal["light", "medium", "heavy"]
_TargetPath = tuple[str, ...]


@dataclass(frozen=True)
class _SpatialTargetLeaf:
    """Resolved spatial target leaf, optionally selecting one stack channel."""

    path: _TargetPath
    channel_index: int | None = None
    channel_name: str | None = None


@dataclass(frozen=True)
class TrainingAugmentationPreset:
    """Augmentation applied as one pair transform: geometry, then appearance.

    Pass a preset as a dataset's ``pair_transform``; it takes ``(image,
    target)`` and returns the augmented pair, and forwards per-sample seeds to
    its parts. Presets augment raw HWC images and preserve their dtype, so the
    model's ``preproc`` (a separate dataset argument) runs afterwards. The
    parts remain available for custom pipelines.
    """

    pair_transform: Callable | None = None
    image_transform: Callable | None = None
    target_transform: Callable | None = None

    def __call__(self, image: np.ndarray, target: object) -> tuple[object, object]:
        """Apply paired geometry, image appearance, then target transforms."""
        if self.pair_transform is not None:
            image, target = self.pair_transform(image, target)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def set_random_seed(self, seed: int) -> None:
        """Derive independent seeds for each seeded part from one sample seed."""
        for offset, transform in enumerate(
            (self.pair_transform, self.image_transform, self.target_transform)
        ):
            if hasattr(transform, "set_random_seed"):
                state = np.random.SeedSequence([seed, offset]).generate_state(1)
                transform.set_random_seed(int(state[0]))


class _SeededTransform:
    """Run v2 with a sample seed without advancing the model's torch RNG."""

    def set_random_seed(self, seed: int) -> None:
        self._next_seed = seed

    def _run_pipeline(self, *inputs: torch.Tensor) -> object:
        seed = getattr(self, "_next_seed", None)
        if seed is None:
            # DataLoader seeds this global stream independently in each worker.
            seed = int(np.random.randint(0, 2**32, dtype=np.uint32))  # noqa: NPY002
        self._next_seed = None
        # Seed only the CPU generator: torch.manual_seed also changes CUDA RNGs.
        # fork_rng restores the caller's CPU state even if a transform fails.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return self.transform(*inputs)


def _to_tv_tensor(
    array: np.ndarray, kind: SpatialTargetKind = "image"
) -> tv_tensors.Image | tv_tensors.Mask:
    """Convert HWC/HW arrays without scaling target values."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    tensor = tensor.movedim(-1, 0) if array.ndim == _IMAGE_DIMS else tensor.unsqueeze(0)
    return tv_tensors.Mask(tensor) if kind == "mask" else tv_tensors.Image(tensor)


def _to_numpy(
    tensor: torch.Tensor, ndim: int, *, image_dtype: torch.dtype | None = None
) -> np.ndarray:
    """Restore array layout, optionally rescaling image intensities to their dtype."""
    if image_dtype == torch.uint16 and tensor.is_floating_point():
        # v2's float32 scaling can round its multiplier to 65536, wrapping white
        # to zero on conversion to uint16. Round within the valid range instead.
        maximum = torch.iinfo(image_dtype).max
        tensor = tensor.mul(maximum).round().clamp(0, maximum).to(image_dtype)
    elif image_dtype is not None:
        tensor = functional.to_dtype(tensor, image_dtype, scale=True)
    tensor = tensor.movedim(0, -1) if ndim == _IMAGE_DIMS else tensor.squeeze(0)
    return tensor.contiguous().numpy()


class _TorchvisionImageTransform(_SeededTransform):
    """Adapt a v2 image pipeline to the dataset's NumPy interface."""

    def __init__(self, transform: v2.Compose) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply appearance transforms in uint8 or floating [0, 1] space."""
        tensor = _to_tv_tensor(image)
        dtype = tensor.dtype
        if dtype != torch.uint8 and not tensor.is_floating_point():
            tensor = functional.to_dtype(tensor, torch.float32, scale=True)
        result = self._run_pipeline(tensor)
        return _to_numpy(result, image.ndim, image_dtype=dtype)


class _StainAugmentation(nn.Module):
    """Apply the standalone stain augmentor with per-call state and seeded RNG."""

    def __init__(self, sigma1: float, sigma2: float, p: float) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.p = p

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract stains afresh so results do not depend on previous patches."""
        if torch.rand(()) >= self.p or image.shape[0] != _IMAGE_DIMS:
            return image
        # Stain concentration extraction replaces zero pixels in-place.
        array = _to_numpy(image, _IMAGE_DIMS).copy()
        try:
            tissue = get_luminosity_tissue_mask(array, threshold=0.8)
        except ValueError:  # Empty background patches have no stains to estimate.
            return image
        if np.count_nonzero(tissue) < _MIN_STAIN_PIXELS:
            return image
        augmentor = StainAugmentor(
            method="macenko", sigma1=self.sigma1, sigma2=self.sigma2
        )
        augmentor.rng = np.random.default_rng(int(torch.randint(0, 2**32, ()).item()))
        return _to_tv_tensor(augmentor(array))


class _Uint8Transform(nn.Module):
    """Adapt uint8-only JPEG and stain transforms to other image dtypes."""

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Quantize only while applying the wrapped uint8 operation."""
        quantized = functional.to_dtype(image, torch.uint8, scale=True)
        result = self.transform(quantized)
        if result is quantized:
            return image
        return functional.to_dtype(result, image.dtype, scale=True)


class _RandomQuarterTurn(nn.Module):
    """Rotate all spatial inputs by the same exact multiple of 90 degrees."""

    def forward(self, *inputs: torch.Tensor) -> object:
        """Permute pixels without resampling, keeping each input's shape.

        Square inputs turn by any multiple of 90 degrees. Rectangular inputs
        only turn by 0 or 180 degrees, as odd turns would swap height and width
        and break batching with unrotated patches.
        """
        turns = int(torch.randint(4, ()).item())
        height, width = inputs[0].shape[-2:]
        if height != width:
            turns = 2 * (turns % 2)
        outputs = tuple(
            tv_tensors.wrap(torch.rot90(value, turns, (-2, -1)), like=value)
            for value in inputs
        )
        return outputs[0] if len(outputs) == 1 else outputs


class _RandomGamma(nn.Module):
    """Sample a gamma factor for torchvision's gamma adjustment."""

    def __init__(self, limit: float) -> None:
        super().__init__()
        self.limit = limit

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction without changing image dtype."""
        gamma = float(torch.empty(()).uniform_(1 - self.limit, 1 + self.limit))
        return functional.adjust_gamma(image, gamma)


class _RandomNoise(nn.Module):
    """Sample Gaussian noise strength in normalized intensity units."""

    def __init__(self, sigma: tuple[float, float]) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply noise in floating point and restore the original dtype."""
        sigma = float(torch.empty(()).uniform_(*self.sigma))
        result = functional.gaussian_noise(
            functional.to_dtype(image, torch.float32, scale=True), sigma=sigma
        )
        return functional.to_dtype(result, image.dtype, scale=True)


def _normalize_spatial_target_kind(kind: str) -> SpatialTargetKind:
    """Validate a spatial target interpolation kind."""
    if kind not in {"mask", "image"}:
        msg = (
            f"Unsupported spatial target kind `{kind}`. Choose from `mask` or `image`."
        )
        raise ValueError(msg)
    return kind


def _flatten_spatial_target_spec(
    spec: SpatialTargetSpec,
    *,
    path: _TargetPath = (),
) -> dict[_SpatialTargetLeaf, SpatialTargetKind]:
    """Flatten a nested spatial target spec into transformable leaves."""
    if isinstance(spec, str):
        return {_SpatialTargetLeaf(path=path): _normalize_spatial_target_kind(spec)}

    if isinstance(spec, StackedSpatialTargetSpec):
        return {
            _SpatialTargetLeaf(
                path=path,
                channel_index=index,
                channel_name=channel_name,
            ): _normalize_spatial_target_kind(kind)
            for index, (channel_name, kind) in enumerate(spec.channels)
        }

    flat_spec: dict[_SpatialTargetLeaf, SpatialTargetKind] = {}
    for key, child_spec in spec.items():
        if not isinstance(key, str):
            msg = "Structured spatial target spec keys must be strings."
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        flat_spec.update(
            _flatten_spatial_target_spec(child_spec, path=(*path, key)),
        )
    return flat_spec


def _spatial_leaf_sort_key(
    leaf: _SpatialTargetLeaf,
) -> tuple[_TargetPath, int, str]:
    """Return a stable ordering key for spatial target leaves."""
    channel_index = -1 if leaf.channel_index is None else leaf.channel_index
    return (leaf.path, channel_index, leaf.channel_name or "")


def _format_spatial_leaf_path(leaf: _SpatialTargetLeaf) -> str:
    """Format a spatial target leaf path for user-facing errors."""
    path = (
        (*leaf.path, leaf.channel_name) if leaf.channel_name is not None else leaf.path
    )
    return ".".join(path) or "<root>"


def _select_structured_target_path(
    target: object,
    path: _TargetPath,
) -> object:
    """Select a possibly non-array target value from a nested target object."""
    value = target
    for key in path:
        if not isinstance(value, dict) or key not in value:
            joined_path = ".".join(path) or "<root>"
            msg = f"Structured target is missing spatial key `{joined_path}`."
            raise ValueError(msg)
        value = value[key]
    return value


def _select_spatial_target(
    target: object,
    leaf: _SpatialTargetLeaf,
) -> np.ndarray:
    """Select one spatial target leaf from a possibly nested target object."""
    value = _select_structured_target_path(target, leaf.path)
    joined_path = _format_spatial_leaf_path(leaf)

    if not isinstance(value, np.ndarray):
        msg = (
            f"Spatial target `{joined_path}` must be a NumPy array before "
            "dataset tensor conversion."
        )
        raise ValueError(msg)  # noqa: TRY004 - preserve validation API

    if leaf.channel_index is None:
        return value

    if value.ndim != _IMAGE_DIMS:
        msg = f"Stacked spatial target `{joined_path}` must have shape HxWxC."
        raise ValueError(msg)
    if leaf.channel_index >= value.shape[-1]:
        msg = (
            f"Stacked spatial target `{joined_path}` requested channel index "
            f"{leaf.channel_index}, but target has {value.shape[-1]} channels."
        )
        raise ValueError(msg)
    return value[..., leaf.channel_index]


def _replace_spatial_targets(
    target: object,
    replacements: dict[_TargetPath, dict[int | None, np.ndarray]],
    path: _TargetPath = (),
) -> object:
    """Rebuild the target tree, copying each selected stack only once."""
    if path in replacements:
        channels = replacements[path]
        if None in channels:
            return channels[None]
        updated_target = np.array(target, copy=True)
        for channel_index, replacement in channels.items():
            if replacement.shape != updated_target.shape[:2]:
                joined_path = ".".join(path) or "<root>"
                msg = (
                    f"Transformed stacked spatial target `{joined_path}` returned "
                    f"shape `{replacement.shape}`, expected "
                    f"`{updated_target.shape[:2]}`."
                )
                raise ValueError(msg)
            updated_target[..., channel_index] = replacement
        return updated_target
    if isinstance(target, dict):
        return {
            key: _replace_spatial_targets(value, replacements, (*path, key))
            for key, value in target.items()
        }
    return target


class _TrainingAffine(v2.RandomAffine):
    """Share affine parameters, with white image fill and configurable mask fill.

    Padded pixels are white in the image, zero in continuous ``image`` target
    leaves and ``target_fill`` in ``mask`` leaves, so a task's ``ignore_index``
    can exclude them from supervision.
    """

    def __init__(self, *args: object, target_fill: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.target_fill = target_fill

    def forward(self, image: torch.Tensor, *targets: torch.Tensor) -> object:
        """Use bilinear images/heatmaps and nearest-neighbour class masks."""
        height, width = image.shape[-2:]
        angle, translate, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, [width, height]
        )
        outputs = []
        for index, value in enumerate((image, *targets)):
            if index == 0:
                fill = 1 if value.is_floating_point() else torch.iinfo(value.dtype).max
            elif isinstance(value, tv_tensors.Mask):
                fill = self.target_fill
            else:
                fill = 0
            # Float32 resampling would round large integer class/instance IDs.
            working = value
            if isinstance(value, tv_tensors.Mask) and value.dtype in {
                torch.int32,
                torch.int64,
            }:
                working = value.to(torch.float64)
            result = functional.affine(
                working,
                angle=angle,
                translate=list(translate),
                scale=scale,
                shear=list(shear),
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            )
            outputs.append(result.to(value.dtype))
        return tuple(outputs) if targets else outputs[0]


class _TorchvisionSpatialPairTransform(_SeededTransform):
    """Adapt v2 geometry to nested and mixed-channel NumPy targets."""

    def __init__(
        self,
        ops: list[nn.Module],
        target_spec: SpatialTargetSpec,
        *,
        target_fill: int = 0,
    ) -> None:
        self.target_spec = _flatten_spatial_target_spec(target_spec)
        if not self.target_spec:
            msg = "Structured spatial target specs must contain at least one leaf."
            raise ValueError(msg)
        self.leaves = sorted(self.target_spec, key=_spatial_leaf_sort_key)
        self.target_fill = target_fill
        self.transform = v2.Compose(ops)

    def _check_mask_fill(self, leaf: _SpatialTargetLeaf, array: np.ndarray) -> None:
        """Reject fills that would wrap around in the mask's integer dtype."""
        if self.target_spec[leaf] != "mask" or not np.issubdtype(
            array.dtype, np.integer
        ):
            return
        limits = np.iinfo(array.dtype)
        if not limits.min <= self.target_fill <= limits.max:
            msg = (
                f"`target_fill` {self.target_fill} cannot be represented by the "
                f"{array.dtype} mask `{_format_spatial_leaf_path(leaf)}`. Use an "
                "ignore value within the mask dtype range or a wider mask dtype."
            )
            raise ValueError(msg)

    def __call__(
        self,
        image: np.ndarray,
        target: object,
    ) -> tuple[np.ndarray, object]:
        """Transform selected spatial leaves while leaving other targets untouched."""
        arrays = [_select_spatial_target(target, leaf) for leaf in self.leaves]
        if any(array.shape[:2] != image.shape[:2] for array in arrays):
            msg = "Spatial targets must have the same height and width as the image."
            raise ValueError(msg)
        for leaf, array in zip(self.leaves, arrays, strict=True):
            self._check_mask_fill(leaf, array)
        tensors = [
            _to_tv_tensor(array, self.target_spec[leaf])
            for leaf, array in zip(self.leaves, arrays, strict=True)
        ]
        # Torch flips do not support these unsigned dtypes; widen without scaling.
        tensors = [
            value.to(torch.int64)
            if value.dtype in {torch.uint16, torch.uint32}
            else value
            for value in tensors
        ]
        image_tensor = _to_tv_tensor(image)
        image_dtype = image_tensor.dtype
        if image_dtype != torch.uint8 and not image_tensor.is_floating_point():
            image_tensor = functional.to_dtype(image_tensor, torch.float32, scale=True)
        transformed_image, *transformed_targets = self._run_pipeline(
            image_tensor, *tensors
        )
        replacements: dict[_TargetPath, dict[int | None, np.ndarray]] = {}
        for leaf, array, transformed in zip(
            self.leaves, arrays, transformed_targets, strict=True
        ):
            replacements.setdefault(leaf.path, {})[leaf.channel_index] = _to_numpy(
                transformed, array.ndim
            ).astype(array.dtype, copy=False)
        return (
            _to_numpy(transformed_image, image.ndim, image_dtype=image_dtype),
            _replace_spatial_targets(target, replacements),
        )


def _validate_level(level: AugmentationLevel) -> AugmentationLevel:
    """Validate supported preset names."""
    if level not in {"light", "medium", "heavy"}:
        msg = (
            f"Unsupported augmentation preset `{level}`. "
            "Choose from `light`, `medium`, or `heavy`."
        )
        raise ValueError(msg)
    return level


def _wrap_image_ops(ops: list[nn.Module]) -> Callable | None:
    """Convert image-only v2 transforms into a dataset callable."""
    return _TorchvisionImageTransform(v2.Compose(ops)) if ops else None


def _wrap_pair_ops(
    ops: list[nn.Module],
    target_spec: SpatialTargetSpec,
    *,
    target_fill: int = 0,
) -> Callable | None:
    """Convert paired v2 transforms into a dataset callable."""
    if not ops:
        return None
    return _TorchvisionSpatialPairTransform(ops, target_spec, target_fill=target_fill)


def _geometric_ops(
    level: AugmentationLevel, *, target_fill: int = 0
) -> list[nn.Module]:
    """Build flips, quarter turns and optional affine geometry."""
    ops = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([_RandomQuarterTurn()], p=0.5),
    ]
    if level != "light":
        heavy = level == "heavy"
        ops.append(
            v2.RandomApply(
                [
                    _TrainingAffine(
                        degrees=20 if heavy else 10,
                        translate=(0.05, 0.05) if heavy else (0.02, 0.02),
                        scale=(0.9, 1.1) if heavy else (0.95, 1.05),
                        target_fill=target_fill,
                    )
                ],
                p=0.5 if heavy else 0.3,
            )
        )
    return ops


def _image_ops(level: AugmentationLevel) -> list[nn.Module]:
    """Build appearance operations without transforming any target values."""
    heavy = level == "heavy"
    limit = 0.15 if heavy else 0.08
    ops = [
        v2.RandomApply(
            [
                v2.RandomChoice(
                    [
                        v2.ColorJitter(brightness=limit, contrast=limit),
                        _RandomGamma(0.15 if heavy else 0.1),
                    ]
                )
            ],
            p=0.5 if heavy else 0.3,
        )
    ]
    if level != "light":
        ops.extend(
            [
                v2.RandomApply(
                    [
                        v2.RandomChoice(
                            [
                                _RandomNoise((0.02, 0.05) if heavy else (0.01, 0.03)),
                                v2.RandomChoice(
                                    [
                                        v2.GaussianBlur(kernel_size=kernel)
                                        for kernel in ((3, 5, 7) if heavy else (3, 5))
                                    ]
                                ),
                                _Uint8Transform(
                                    v2.JPEG((70, 95) if heavy else (85, 100))
                                ),
                            ]
                        )
                    ],
                    p=0.35 if heavy else 0.2,
                ),
                _Uint8Transform(
                    _StainAugmentation(
                        sigma1=0.35 if heavy else 0.2,
                        sigma2=0.1 if heavy else 0.05,
                        p=0.4 if heavy else 0.2,
                    )
                ),
            ]
        )
    return ops


def get_classification_augmentation(
    level: AugmentationLevel = "medium",
) -> TrainingAugmentationPreset:
    """Return a preset for patch classification datasets."""
    normalized_level = _validate_level(level)
    ops = [*_geometric_ops(normalized_level), *_image_ops(normalized_level)]
    return TrainingAugmentationPreset(image_transform=_wrap_image_ops(ops))


def get_segmentation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    target_fill: int = 0,
) -> TrainingAugmentationPreset:
    """Return a preset for paired patch/mask segmentation datasets.

    ``target_fill`` is written to mask pixels exposed by affine padding; pass
    the task's ``ignore_index`` to exclude them from supervision. It must be
    representable in the mask dtype (for example 255 for uint8 masks).
    """
    normalized_level = _validate_level(level)
    return TrainingAugmentationPreset(
        pair_transform=_wrap_pair_ops(
            _geometric_ops(normalized_level, target_fill=target_fill),
            "mask",
            target_fill=target_fill,
        ),
        image_transform=_wrap_image_ops(_image_ops(normalized_level)),
    )


def get_annotation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    target_builder: TargetBuilderABC | None = None,
    spatial_target_spec: SpatialTargetSpec | None = None,
    target_fill: int = 0,
) -> TrainingAugmentationPreset:
    """Return a preset for annotation-backed training datasets.

    Spatial targets share geometric transforms with the image using the
    interpolation modes declared by the target builder. Non-spatial target
    leaves, such as scalar or vector labels, are left untouched. Affine padding
    writes ``target_fill`` to ``"mask"`` leaves and zero to ``"image"`` leaves.
    """
    normalized_level = _validate_level(level)
    resolved_spatial_target_spec = spatial_target_spec
    if resolved_spatial_target_spec is None and target_builder is not None:
        resolved_spatial_target_spec = target_builder.spatial_target_spec

    if resolved_spatial_target_spec is not None:
        return TrainingAugmentationPreset(
            pair_transform=_wrap_pair_ops(
                _geometric_ops(normalized_level, target_fill=target_fill),
                resolved_spatial_target_spec,
                target_fill=target_fill,
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
