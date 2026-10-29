"""Preconfigured augmentation presets for TIAToolbox training datasets.

Every level flips patches and turns them by multiples of 90 degrees, which
moves pixels exactly. ``light``, ``medium`` and ``heavy`` add increasingly
strong appearance changes: colour, noise, blur, JPEG and stain augmentation.
``strong`` replaces the appearance changes with
:class:`~tiatoolbox.tools.strongaugment.StrongAugment`, which helps models learn
morphology independently of stain appearance. Zooms, crops and free rotation
are opt-in.
"""

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
from tiatoolbox.tools.strongaugment import StrongAugment
from tiatoolbox.utils.misc import get_luminosity_tissue_mask

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

_IMAGE_DIMS = 3
_MIN_STAIN_PIXELS = 2
_ZOOM_PROBABILITY = 0.5

AugmentationLevel = Literal["light", "medium", "heavy", "strong"]
_LEVELS = ("light", "medium", "heavy", "strong")
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

    ``scale_jitter`` crops or zooms a sample before the other geometry,
    padding where it zooms out. TIAToolbox's annotation datasets apply it
    themselves when they read the sample, building targets for the window,
    and then call the preset with ``apply_scale_jitter=False``.
    """

    pair_transform: Callable | None = None
    image_transform: Callable | None = None
    target_transform: Callable | None = None
    scale_jitter: _ScaleJitter | None = None

    def __call__(
        self, image: np.ndarray, target: object, *, apply_scale_jitter: bool = True
    ) -> tuple[object, object]:
        """Apply zoom and paired geometry, image appearance, then target transforms."""
        if apply_scale_jitter and self.scale_jitter is not None:
            image, target = self.scale_jitter(image, target)
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
            (
                self.pair_transform,
                self.image_transform,
                self.target_transform,
                self.scale_jitter,
            )
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


class _StrongAugmentation(nn.Module):
    """Apply StrongAugment with a generator drawn from the sample's seeded stream."""

    def __init__(self, augmentor: StrongAugment) -> None:
        super().__init__()
        self.augmentor = augmentor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Augment RGB and greyscale images; return others, and no-ops, unchanged."""
        if image.shape[0] not in {1, _IMAGE_DIMS}:
            return image
        rng = np.random.default_rng(int(torch.randint(0, 2**32, ()).item()))
        operations = self.augmentor.sample_operations(rng)
        if not operations:
            return image
        ndim = _IMAGE_DIMS if image.shape[0] == _IMAGE_DIMS else 2
        return _to_tv_tensor(self.augmentor.apply(_to_numpy(image, ndim), operations))


class _Uint8Transform(nn.Module):
    """Adapt uint8-only JPEG, stain and StrongAugment transforms to other dtypes."""

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


def _sample_zoom_window(
    rng: np.random.Generator,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scale: tuple[float, float],
    p: float,
    *,
    centred: bool = False,
) -> tuple[int, int, int, int]:
    """Sample the ``(x_min, y_min, x_max, y_max)`` window to resize to ``output_shape``.

    With probability ``p`` the window is zoomed by a factor drawn from
    ``scale``, where factors above one zoom in; otherwise it is a crop of the
    output size. The factor is never limited by the input size, so the zoom
    is unbiased: a window larger than the input contains it whole and is
    padded on resizing. By default the window lies at a random position; a
    ``centred`` window is centred on the input, for callers that can read
    beyond its edges.
    """
    in_height, in_width = input_shape
    out_height, out_width = output_shape
    if not centred and (out_height > in_height or out_width > in_width):
        msg = f"Crops of {output_shape} do not fit inside inputs of {input_shape}."
        raise ValueError(msg)
    low, high = scale
    factor = 1.0
    if rng.random() < p and low <= high:
        factor = rng.uniform(low, high)
    height = max(1, round(out_height / factor))
    width = max(1, round(out_width / factor))
    if centred:
        top, left = (in_height - height) // 2, (in_width - width) // 2
    else:
        # A window larger than the input contains it, at a random offset.
        top = int(
            rng.integers(min(0, in_height - height), max(0, in_height - height) + 1)
        )
        left = int(rng.integers(min(0, in_width - width), max(0, in_width - width) + 1))
    return (left, top, left + width, top + height)


def _window_layout(
    window: tuple[int, int, int, int],
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Clip a window to the input and locate the clipped part in the output.

    Returns the clipped ``(x_min, y_min, x_max, y_max)`` input window and the
    output box it is resized into, or ``None`` when they do not overlap. The
    rest of the output is padding.
    """
    left, top, right, bottom = window
    in_height, in_width = input_shape
    height, width = output_shape
    inner = (max(left, 0), max(top, 0), min(right, in_width), min(bottom, in_height))
    scale_x, scale_y = width / (right - left), height / (bottom - top)
    box = (
        round((inner[0] - left) * scale_x),
        round((inner[1] - top) * scale_y),
        round((inner[2] - left) * scale_x),
        round((inner[3] - top) * scale_y),
    )
    if (
        inner[0] >= inner[2]
        or inner[1] >= inner[3]
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        return None
    return inner, box


def _resized_crop(
    value: torch.Tensor,
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
    fill: float = 0,
) -> torch.Tensor:
    """Resize a window of a CHW input to ``shape``, padding with ``fill``.

    Masks are resized by nearest neighbour, other inputs bilinearly. The part
    of the window outside the input becomes ``fill``, without resampling.
    """
    input_shape = (int(value.shape[-2]), int(value.shape[-1]))
    height, width = shape
    layout = _window_layout(window, input_shape, shape)
    if layout is None:
        return tv_tensors.wrap(
            torch.full((*value.shape[:-2], height, width), fill, dtype=value.dtype),
            like=value,
        )
    (left, top, right, bottom), box = layout
    box_height, box_width = box[3] - box[1], box[2] - box[0]
    if (bottom - top, right - left) == (box_height, box_width):
        # A window of the output size is a crop and needs no resampling.
        if window == (0, 0, width, height) and input_shape == shape:
            return value
        result = value[..., top:bottom, left:right]
    elif isinstance(value, tv_tensors.Mask):
        # Index arithmetic keeps labels of any dtype exact.
        rows = top + ((torch.arange(box_height) + 0.5) * (bottom - top) / box_height)
        cols = left + ((torch.arange(box_width) + 0.5) * (right - left) / box_width)
        result = value[..., rows.long()[:, None], cols.long()]
    else:
        working = value
        if not (value.is_floating_point() or value.dtype == torch.uint8):
            working = value.to(torch.float64)
        result = functional.resized_crop(
            working,
            top,
            left,
            bottom - top,
            right - left,
            [box_height, box_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if result.dtype != value.dtype:
            result = result.round().to(value.dtype)
    if box == (0, 0, width, height):
        return tv_tensors.wrap(result, like=value)
    output = torch.full((*value.shape[:-2], height, width), fill, dtype=value.dtype)
    output[..., box[1] : box[3], box[0] : box[2]] = result
    return tv_tensors.wrap(output, like=value)


def _white(tensor: torch.Tensor) -> float:
    """Return the padding value of images: white in the tensor's dtype."""
    return 1.0 if tensor.is_floating_point() else torch.iinfo(tensor.dtype).max


def _zoom_image(
    image: np.ndarray,
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Resize a window of an HWC or HW image to ``shape``, padding with white."""
    tensor = _to_tv_tensor(image)
    dtype = tensor.dtype
    if dtype != torch.uint8 and not tensor.is_floating_point():
        tensor = functional.to_dtype(tensor, torch.float32, scale=True)
    return _to_numpy(
        _resized_crop(tensor, window, shape, _white(tensor)),
        image.ndim,
        image_dtype=dtype,
    )


class _RandomZoom(nn.Module):
    """Crop or zoom all inputs to the same random window, resized to the output.

    The window is resized to ``crop_size``, or to the input size by default.
    Where it extends past the inputs, the first input (the image) is padded
    white, masks with ``target_fill`` and other inputs with zero.
    """

    def __init__(
        self,
        scale: tuple[float, float],
        p: float,
        crop_size: tuple[int, int] | None = None,
        *,
        target_fill: int = 0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.p = p
        self.crop_size = crop_size
        self.target_fill = target_fill

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        """Return the training shape for inputs of ``input_shape``."""
        return input_shape if self.crop_size is None else self.crop_size

    def fill(self, value: torch.Tensor, *, image: bool) -> float:
        """Return the padding value of one input."""
        if image:
            return _white(value)
        return self.target_fill if isinstance(value, tv_tensors.Mask) else 0

    def forward(self, *inputs: torch.Tensor) -> object:
        """Resize one sampled window of every input."""
        input_shape = (int(inputs[0].shape[-2]), int(inputs[0].shape[-1]))
        output_shape = self.output_shape(input_shape)
        rng = np.random.default_rng(int(torch.randint(0, 2**32, ()).item()))
        window = _sample_zoom_window(rng, input_shape, output_shape, self.scale, self.p)
        outputs = tuple(
            _resized_crop(
                value, window, output_shape, self.fill(value, image=not index)
            )
            for index, value in enumerate(inputs)
        )
        return outputs[0] if len(outputs) == 1 else outputs


def _random_zoom(
    scale_jitter: float,
    crop_size: int | tuple[int, int] | None,
    *,
    target_fill: int = 0,
) -> _RandomZoom | None:
    """Build the opt-in zoom and crop, or return ``None`` when neither is set."""
    if not 0 <= scale_jitter < 1:
        msg = "`scale_jitter` must be at least 0 and less than 1."
        raise ValueError(msg)
    if crop_size is not None:
        crop_size = (
            (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        )
        if len(crop_size) != 2 or min(crop_size) <= 0:  # noqa: PLR2004
            msg = "`crop_size` must be a positive size or (height, width) pair."
            raise ValueError(msg)
    if not scale_jitter and crop_size is None:
        return None
    return _RandomZoom(
        (1 - scale_jitter, 1 + scale_jitter),
        _ZOOM_PROBABILITY,
        crop_size,
        target_fill=target_fill,
    )


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
        shape = next(iter(channels.values())).shape
        if shape != target.shape[:2] and len(channels) == target.shape[-1]:
            # Crops change every channel's shape, so restack them.
            return np.stack(
                [channels[index] for index in range(target.shape[-1])], axis=-1
            ).astype(target.dtype, copy=False)
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


class _ScaleJitter(_TorchvisionSpatialPairTransform):
    """Crop or zoom a sample by a random factor, padding where it zooms out.

    TIAToolbox's annotation datasets call :meth:`sample_window` instead and
    build targets for that window, so thin targets such as one-pixel
    boundaries are drawn at the training resolution rather than resampled.
    Called on arrays, it resizes masks by nearest neighbour.
    """

    def __init__(self, zoom: _RandomZoom, target_spec: SpatialTargetSpec) -> None:
        self.zoom = zoom
        super().__init__([zoom], target_spec, target_fill=zoom.target_fill)

    def pad_target(
        self,
        target: object,
        box: tuple[int, int, int, int],
        output_shape: tuple[int, int],
    ) -> object:
        """Place spatial leaves built for ``box`` in padded arrays of ``output_shape``.

        The padding is ``target_fill`` in ``mask`` leaves and zero in ``image``
        leaves, as under the image's white padding.
        """
        left, top, right, bottom = box
        replacements: dict[_TargetPath, dict[int | None, np.ndarray]] = {}
        for leaf in self.leaves:
            array = _select_spatial_target(target, leaf)
            self._check_mask_fill(leaf, array)
            fill = self.target_fill if self.target_spec[leaf] == "mask" else 0
            padded = np.full((*output_shape, *array.shape[2:]), fill, dtype=array.dtype)
            padded[top:bottom, left:right] = array
            replacements.setdefault(leaf.path, {})[leaf.channel_index] = padded
        return _replace_spatial_targets(target, replacements)

    def sample_window(
        self, input_shape: tuple[int, int], *, centred: bool = False
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        """Sample an input window and the shape to resize it to, from the sample seed.

        The window may extend past the input; see :func:`_sample_zoom_window`
        and :func:`_window_layout`.
        """
        seed = getattr(self, "_next_seed", None)
        self._next_seed = None
        if seed is None:
            # DataLoader seeds this global stream independently in each worker.
            seed = int(np.random.randint(0, 2**32, dtype=np.uint32))  # noqa: NPY002
        output_shape = self.zoom.output_shape(input_shape)
        window = _sample_zoom_window(
            np.random.default_rng(seed),
            input_shape,
            output_shape,
            self.zoom.scale,
            self.zoom.p,
            centred=centred,
        )
        return window, output_shape


def _validate_level(level: AugmentationLevel) -> AugmentationLevel:
    """Validate supported preset names."""
    if level not in _LEVELS:
        msg = (
            f"Unsupported augmentation preset `{level}`. "
            "Choose from `light`, `medium`, `heavy` or `strong`."
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


def _geometric_ops(*, target_fill: int = 0, rotation: float = 0.0) -> list[nn.Module]:
    """Build exact flips and quarter turns, with optional free rotation."""
    if rotation < 0:
        msg = "`rotation` must be a non-negative angle in degrees."
        raise ValueError(msg)
    ops = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([_RandomQuarterTurn()], p=0.5),
    ]
    if rotation:
        ops.append(
            v2.RandomApply(
                [_TrainingAffine(degrees=rotation, target_fill=target_fill)], p=0.5
            )
        )
    return ops


def _image_ops(level: AugmentationLevel) -> list[nn.Module]:
    """Build appearance operations without transforming any target values."""
    if level == "strong":
        # StrongAugment replaces every appearance operation of the other levels.
        return [_Uint8Transform(_StrongAugmentation(StrongAugment()))]
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
    *,
    scale_jitter: float = 0.0,
    crop_size: int | tuple[int, int] | None = None,
    rotation: float = 0.0,
) -> TrainingAugmentationPreset:
    """Return a preset for patch classification datasets.

    Patches are flipped and turned by multiples of 90 degrees. Other geometry
    is opt-in. ``scale_jitter=0.1`` zooms half the patches by a factor between
    0.9 and 1.1, centred on one so that the training scale stays unbiased;
    zooming out pads the patch with white. ``crop_size`` trains on random
    crops of that size, which also give the zoom room, so a crop of 224 in a
    256-pixel patch zooms by up to 10% without padding. ``rotation`` rotates
    half the patches by up to that many degrees, exposing white padding in the
    corners.
    """
    normalized_level = _validate_level(level)
    zoom = _random_zoom(scale_jitter, crop_size)
    ops = [
        *([] if zoom is None else [zoom]),
        *_geometric_ops(rotation=rotation),
        *_image_ops(normalized_level),
    ]
    return TrainingAugmentationPreset(image_transform=_wrap_image_ops(ops))


def get_segmentation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    scale_jitter: float = 0.0,
    crop_size: int | tuple[int, int] | None = None,
    rotation: float = 0.0,
    target_fill: int = 0,
) -> TrainingAugmentationPreset:
    """Return a preset for paired patch/mask segmentation datasets.

    Patches and masks are flipped and turned by multiples of 90 degrees.
    ``scale_jitter`` and ``crop_size`` opt in to zooms and crops as for
    :func:`get_classification_augmentation`, resizing masks by nearest
    neighbour. ``rotation`` rotates half the patches by up to that many
    degrees. Mask pixels exposed by zoom or rotation padding get
    ``target_fill``; pass the task's ``ignore_index`` to exclude them from
    supervision. It must be representable in the mask dtype (for example 255
    for uint8 masks).
    """
    normalized_level = _validate_level(level)
    zoom = _random_zoom(scale_jitter, crop_size, target_fill=target_fill)
    return TrainingAugmentationPreset(
        pair_transform=_wrap_pair_ops(
            _geometric_ops(target_fill=target_fill, rotation=rotation),
            "mask",
            target_fill=target_fill,
        ),
        image_transform=_wrap_image_ops(_image_ops(normalized_level)),
        scale_jitter=None if zoom is None else _ScaleJitter(zoom, "mask"),
    )


def get_annotation_augmentation(
    level: AugmentationLevel = "medium",
    *,
    target_builder: TargetBuilderABC | None = None,
    spatial_target_spec: SpatialTargetSpec | None = None,
    scale_jitter: float = 0.0,
    crop_size: int | tuple[int, int] | None = None,
    rotation: float = 0.0,
    target_fill: int = 0,
) -> TrainingAugmentationPreset:
    """Return a preset for annotation-backed training datasets.

    Spatial targets share geometric transforms with the image using the
    interpolation modes declared by the target builder. Non-spatial target
    leaves, such as scalar or vector labels, are left untouched, and presets
    for them change only appearance.

    ``scale_jitter`` and ``crop_size`` opt in to zooms and crops as for
    :func:`get_classification_augmentation`.
    :class:`~tiatoolbox.models.training.PatchAnnotationDataset` and
    :class:`~tiatoolbox.models.training.SlideAnnotationPatchDataset` apply
    them while reading and build targets for the window, so masks, boundaries
    and centroids stay exact; slide datasets zoom out by reading a larger
    window instead of padding. ``rotation`` rotates half the patches by up to
    that many degrees and resamples the targets. Zoom and rotation padding
    writes ``target_fill`` to ``"mask"`` leaves and zero to ``"image"``
    leaves.
    """
    normalized_level = _validate_level(level)
    resolved_spatial_target_spec = spatial_target_spec
    if resolved_spatial_target_spec is None and target_builder is not None:
        resolved_spatial_target_spec = target_builder.spatial_target_spec
    zoom = _random_zoom(scale_jitter, crop_size, target_fill=target_fill)

    if resolved_spatial_target_spec is not None:
        return TrainingAugmentationPreset(
            pair_transform=_wrap_pair_ops(
                _geometric_ops(target_fill=target_fill, rotation=rotation),
                resolved_spatial_target_spec,
                target_fill=target_fill,
            ),
            image_transform=_wrap_image_ops(_image_ops(normalized_level)),
            scale_jitter=(
                None
                if zoom is None
                else _ScaleJitter(zoom, resolved_spatial_target_spec)
            ),
        )

    if zoom is not None or rotation:
        msg = (
            "Zooms, crops and rotation need spatial targets: pass a target "
            "builder with spatial targets or `spatial_target_spec`."
        )
        raise ValueError(msg)
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
