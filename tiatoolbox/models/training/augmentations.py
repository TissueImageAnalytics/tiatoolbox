"""Augmentation presets for TIAToolbox training datasets.

Every level flips patches and turns them by multiples of 90 degrees, which
moves pixels exactly. ``light``, ``medium`` and ``heavy`` add increasingly
strong appearance changes: colour, noise, blur, JPEG and stain augmentation.
``strong`` replaces the appearance changes with
:class:`~tiatoolbox.tools.strongaugment.StrongAugment`, which helps models learn
morphology independently of stain appearance. Crops and zooms are dataset
arguments, as each dataset follows them with its targets in its own way; the
helpers they share live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional

from tiatoolbox.tools.stainaugment import StainAugmentor
from tiatoolbox.tools.strongaugment import StrongAugment
from tiatoolbox.utils.misc import get_luminosity_tissue_mask

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

_IMAGE_DIMS = 3
_SPATIAL_DIMS = 2
_MIN_STAIN_PIXELS = 2
_MOVE_PROBABILITY = 0.5
_ZOOM_PROBABILITY = 0.5

AugmentationLevel = Literal["light", "medium", "heavy", "strong"]
_LEVELS = ("light", "medium", "heavy", "strong")


class TrainingAugmentationPreset:
    """Flip and turn an image with its targets, then change the image's appearance.

    Pass a preset as a dataset's ``pair_transform``; it takes ``(image,
    target)`` and returns the augmented pair. It augments raw HWC images and
    preserves their dtype, so the model's ``preproc`` (a separate dataset
    argument) runs afterwards.

    Flips and turns move whole arrays, so targets of any dtype stay exact.
    Spatial targets are NumPy arrays that start with the image's height and
    width, at any depth of dictionaries, lists and tuples. Scalars and label
    vectors are unchanged; other arrays and tensors raise, because they cannot
    follow the image.

    Datasets seed each sample through :meth:`set_random_seed`. Unseeded calls
    draw from the global NumPy stream, which DataLoader seeds in each worker.
    Calls leave the caller's random state alone.
    """

    def __init__(self, appearance: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the appearance operations, which take a CHW image tensor."""
        self.appearance = appearance
        self._next_seed: int | None = None

    def set_random_seed(self, seed: int) -> None:
        """Set the seed for the next call."""
        self._next_seed = seed

    def __call__(self, image: np.ndarray, target: object) -> tuple[np.ndarray, object]:
        """Return the augmented image and target."""
        seed = self._next_seed
        self._next_seed = None
        if seed is None:
            seed = int(np.random.randint(0, 2**32, dtype=np.uint32))  # noqa: NPY002
        geometry_seed, appearance_seed = (
            int(np.random.SeedSequence([seed, offset]).generate_state(1)[0])
            for offset in (0, 1)
        )
        generator = torch.random.default_generator
        shape = image.shape[:2]
        # Seed only the CPU generator: torch.manual_seed also changes CUDA RNGs.
        # fork_rng restores the caller's CPU state even if a transform fails.
        with torch.random.fork_rng(devices=[]):
            generator.manual_seed(geometry_seed)
            moves = _draw_moves(shape)
            target = _map_spatial(target, shape, lambda value: _move(value, *moves))
            generator.manual_seed(appearance_seed)
            return _apply_appearance(self.appearance, _move(image, *moves)), target


def _draw_moves(shape: tuple[int, int]) -> tuple[bool, bool, int]:
    """Draw a sample's horizontal flip, vertical flip and quarter turns.

    Square inputs turn by any multiple of 90 degrees. Rectangular inputs only
    turn by 0 or 180 degrees, as odd turns would swap height and width and
    break batching with unrotated patches.
    """
    flip_x = bool(torch.rand(1) < _MOVE_PROBABILITY)
    flip_y = bool(torch.rand(1) < _MOVE_PROBABILITY)
    turns = int(torch.randint(4, ())) if torch.rand(1) < _MOVE_PROBABILITY else 0
    height, width = shape
    return flip_x, flip_y, turns if height == width else 2 * (turns % 2)


def _move(array: np.ndarray, flip_x: bool, flip_y: bool, turns: int) -> np.ndarray:  # noqa: FBT001
    """Flip and turn an array whose first two axes are height and width."""
    if flip_x:
        array = np.flip(array, axis=1)
    if flip_y:
        array = np.flip(array, axis=0)
    if turns:
        array = np.rot90(array, turns, axes=(0, 1))
    return np.ascontiguousarray(array)


def _to_tv_tensor(array: np.ndarray) -> tv_tensors.Image:
    """Convert HWC/HW arrays without scaling their values."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    tensor = tensor.movedim(-1, 0) if array.ndim == _IMAGE_DIMS else tensor.unsqueeze(0)
    return tv_tensors.Image(tensor)


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


def _apply_appearance(
    pipeline: Callable[[torch.Tensor], torch.Tensor], image: np.ndarray
) -> np.ndarray:
    """Apply appearance operations in uint8 or floating [0, 1] space."""
    tensor = _to_tv_tensor(image)
    dtype = tensor.dtype
    if dtype != torch.uint8 and not tensor.is_floating_point():
        tensor = functional.to_dtype(tensor, torch.float32, scale=True)
    return _to_numpy(pipeline(tensor), image.ndim, image_dtype=dtype)


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


def _validate_zoom(
    scale_jitter: float, crop_size: int | tuple[int, int] | None
) -> tuple[float, tuple[int, int] | None]:
    """Validate a dataset's zoom range and crop, as a (height, width) pair."""
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
    return float(scale_jitter), crop_size


def _sample_zoom_window(
    rng: np.random.Generator,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scale_jitter: float,
    *,
    centred: bool = False,
) -> tuple[int, int, int, int]:
    """Sample the ``(x_min, y_min, x_max, y_max)`` window to resize to ``output_shape``.

    Half the windows are zoomed by a factor within ``scale_jitter`` of one,
    where factors above one zoom in; the others are crops of the output size.
    The factor is never limited by the input size, so the zoom is unbiased: a
    window larger than the input contains it whole and is padded on resizing.
    By default the window lies at a random position; a ``centred`` window is
    centred on the input, for callers that can read beyond its edges.
    """
    in_height, in_width = input_shape
    out_height, out_width = output_shape
    if not centred and (out_height > in_height or out_width > in_width):
        msg = f"Crops of {output_shape} do not fit inside inputs of {input_shape}."
        raise ValueError(msg)
    factor = 1.0
    if rng.random() < _ZOOM_PROBABILITY:
        factor = rng.uniform(1 - scale_jitter, 1 + scale_jitter)
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


def _zoom_image(
    image: np.ndarray,
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Resize a window of an HWC or HW image to ``shape``, padding with white.

    A window of the output size is a crop, which needs no resampling.
    """
    (left, top, right, bottom), box = _window_layout(window, image.shape[:2], shape)
    tensor = _to_tv_tensor(image)
    dtype = tensor.dtype
    if dtype != torch.uint8 and not tensor.is_floating_point():
        tensor = functional.to_dtype(tensor, torch.float32, scale=True)
    white = 1.0 if tensor.is_floating_point() else torch.iinfo(tensor.dtype).max
    output = torch.full((*tensor.shape[:-2], *shape), white, dtype=tensor.dtype)
    size = [box[3] - box[1], box[2] - box[0]]
    if [bottom - top, right - left] == size:
        part = tensor[..., top:bottom, left:right]
    else:
        part = functional.resized_crop(
            tensor,
            top,
            left,
            bottom - top,
            right - left,
            size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    output[..., box[1] : box[3], box[0] : box[2]] = part
    return _to_numpy(output, image.ndim, image_dtype=dtype)


def _pad_array(
    array: np.ndarray,
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    fill: int,
) -> np.ndarray:
    """Place an array built for ``box`` in a ``shape`` array of ``fill``."""
    if array.shape[:2] == shape:
        return array
    if np.asarray(fill).astype(array.dtype) != fill:
        msg = (
            f"`target_fill` {fill} cannot be represented by a {array.dtype} target. "
            "Use an ignore value within the target's dtype range or a wider dtype."
        )
        raise ValueError(msg)
    left, top, right, bottom = box
    padded = np.full((*shape, *array.shape[2:]), fill, dtype=array.dtype)
    padded[top:bottom, left:right] = array
    return padded


def _zoom_mask(
    mask: np.ndarray,
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
    fill: int,
) -> np.ndarray:
    """Resize a window of a mask to ``shape``, padding with ``fill``.

    Nearest-neighbour indexing keeps values of any dtype exact, including
    floating masks.
    """
    (left, top, right, bottom), box = _window_layout(window, mask.shape[:2], shape)
    height, width = box[3] - box[1], box[2] - box[0]
    rows = top + ((np.arange(height) + 0.5) * (bottom - top) / height).astype(np.int64)
    cols = left + ((np.arange(width) + 0.5) * (right - left) / width).astype(np.int64)
    return _pad_array(mask[rows[:, None], cols], box, shape, fill)


def _map_spatial(
    target: object, shape: tuple[int, int], function: Callable[[np.ndarray], np.ndarray]
) -> object:
    """Apply ``function`` to the spatial arrays of a nested target.

    Spatial arrays start with the image's height and width. Scalars and label
    vectors are returned unchanged. Other arrays or tensors raise, because they
    cannot follow the image.
    """
    if isinstance(target, dict):
        return {
            key: _map_spatial(value, shape, function) for key, value in target.items()
        }
    if isinstance(target, (list, tuple)):
        return type(target)(_map_spatial(value, shape, function) for value in target)
    if getattr(target, "ndim", 0) < _SPATIAL_DIMS:
        return target
    if not isinstance(target, np.ndarray) or target.shape[:2] != shape:
        msg = (
            "Spatial targets must be NumPy arrays that start with the image's "
            f"height and width {shape}, but got a {type(target).__name__} of shape "
            f"{tuple(target.shape)}. Use a custom `pair_transform` for other targets."
        )
        raise ValueError(msg)
    return function(target)


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


def get_augmentation(level: AugmentationLevel = "medium") -> TrainingAugmentationPreset:
    """Return a preset to pass as a dataset's ``pair_transform``.

    Every level flips patches and turns them by multiples of 90 degrees, with
    their spatial targets. ``level`` selects the appearance changes: ``light``,
    ``medium`` and ``heavy`` change colour, noise, blur, JPEG and stain
    increasingly strongly, and ``strong`` uses
    :class:`~tiatoolbox.tools.strongaugment.StrongAugment` instead.
    """
    if level not in _LEVELS:
        msg = (
            f"Unsupported augmentation preset `{level}`. "
            "Choose from `light`, `medium`, `heavy` or `strong`."
        )
        raise ValueError(msg)
    return TrainingAugmentationPreset(v2.Compose(_image_ops(level)))


__all__ = [
    "AugmentationLevel",
    "TrainingAugmentationPreset",
    "get_augmentation",
]
