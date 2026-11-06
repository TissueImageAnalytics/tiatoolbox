# This module is derived from StrongAugment (https://github.com/jopo666/StrongAugment),
# which is distributed under the MIT License:
#
# Copyright (c) 2022 Joona Pohjonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Strong pixel-level augmentation for histology images.

``StrongAugment`` applies a random number of randomly chosen colour, intensity,
filter and compression operations to an image, each with a random magnitude.
Training with it helps models learn morphology independently of stain and
scanner appearance.

This is TIAToolbox's port of `StrongAugment
<https://github.com/jopo666/StrongAugment>`_ by Joona Pohjonen, released under
the MIT License (reproduced at the top of this module's source). If you use it,
please cite:

    Pohjonen, J. et al. "Augment like there's no tomorrow: Consistently
    performing neural networks for medical imaging." arXiv:2206.15274 (2022).

Changes from the original:

- One NumPy generator draws the number, choice and magnitude of operations and
  the noise, so a seed or a per-call generator reproduces a result. The
  original also drew from NumPy's global random state.
- The defaults are those of TIAToolbox's ``"strong"`` training augmentation:
  zero, two, three or four operations with probabilities 0.3, 0.3, 0.3 and
  0.1, and no ``solarize`` or ``posterize``, which can make images
  unrealistically extreme. Both remain available.
- Operations with continuous magnitudes sample them continuously even from
  integer bounds, ``posterize`` works with NumPy 2, a gamma of 1.0 leaves
  images unchanged, and JPEG compression encodes in OpenCV's BGR channel order.
- Bounds must be ordered, and the tools for creating distribution-shifted
  datasets are not included.

"""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping, Sequence

type Bounds = tuple[float, float] | tuple[bool, bool]
type Operation = tuple[str, dict[str, float]]

_GRAYSCALE_DIMS = 2
_RGB_CHANNELS = 3
_PAIR = 2
_IDENTITY_KERNEL = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
_SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
_EMBOSS_KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)


def get_augment_space() -> dict[str, Bounds]:
    """Return the default operations and the bounds of their magnitudes.

    Each key names an operation and each value gives the ``(low, high)``
    bounds its magnitude is sampled from. Operations without a magnitude
    (``autocontrast``, ``equalize`` and ``grayscale``) take ``(True, True)``.
    Two more operations are supported but left out, as they can make images
    unrealistically extreme: ``solarize``, with an integer threshold from 0 to
    256, and ``posterize``, with an integer number of bits from 1 to 8.

    Returns:
        dict[str, tuple]:
            A new dictionary that can be edited and passed to
            :class:`StrongAugment` as ``augment_space``.

    """
    return {
        "red": (0.0, 2.0),
        "green": (0.0, 2.0),
        "blue": (0.0, 2.0),
        "hue": (-0.5, 0.5),
        "saturation": (0.0, 2.0),
        "brightness": (0.1, 2.0),
        "contrast": (0.1, 2.0),
        "gamma": (0.1, 2.0),
        "sharpen": (0.0, 1.0),
        "emboss": (0.0, 1.0),
        "blur": (0.0, 3.0),
        "noise": (0.0, 0.2),
        "jpeg": (0, 100),
        "tone": (0.0, 1.0),
        "autocontrast": (True, True),
        "equalize": (True, True),
        "grayscale": (True, True),
    }


class StrongAugment:
    """Apply a random number of random pixel-level operations to images.

    Each call draws how many operations to apply from ``operations`` with the
    matching ``probabilities``, picks that many distinct operations from
    ``augment_space`` and samples a magnitude for each within its bounds.
    Operations change intensities only, so masks and other targets need no
    matching transform.

    Args:
        operations (Sequence[int]):
            Possible numbers of operations per image. Defaults to
            ``(0, 2, 3, 4)``.
        probabilities (Sequence[float]):
            Probability of each entry in ``operations``. Defaults to
            ``(0.3, 0.3, 0.3, 0.1)``.
        augment_space (Mapping[str, tuple] | None):
            Operations to choose from and the ``(low, high)`` bounds of their
            magnitudes. Defaults to :func:`get_augment_space`.
        seed (int | None):
            Seed for the generator used when a call passes no ``rng``.

    Example:
        >>> from tiatoolbox.tools.strongaugment import StrongAugment
        >>> augmentor = StrongAugment(seed=0)
        >>> augmented = augmentor(image)  # An RGB or greyscale uint8 image.

    """

    def __init__(
        self: StrongAugment,
        operations: Sequence[int] = (0, 2, 3, 4),
        probabilities: Sequence[float] = (0.3, 0.3, 0.3, 0.1),
        augment_space: Mapping[str, Bounds] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize :class:`StrongAugment`."""
        space = get_augment_space() if augment_space is None else dict(augment_space)
        _check_augment_space(space)
        if len(operations) != len(probabilities):
            msg = "`operations` and `probabilities` must have the same length."
            raise ValueError(msg)
        if any(
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not 0 <= count <= len(space)
            for count in operations
        ):
            msg = (
                "`operations` must hold whole numbers from 0 to the number of "
                f"operations in the augment space ({len(space)})."
            )
            raise ValueError(msg)
        if any(probability < 0 for probability in probabilities) or not np.isclose(
            sum(probabilities), 1
        ):
            msg = "`probabilities` must be non-negative and sum to 1."
            raise ValueError(msg)
        self.operations = tuple(int(count) for count in operations)
        self.probabilities = tuple(float(value) for value in probabilities)
        self.augment_space = space
        self.rng = np.random.default_rng(seed)

    def __call__(
        self: StrongAugment,
        image: np.ndarray | Image.Image,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | Image.Image:
        """Augment an image with newly sampled operations.

        Args:
            image (numpy.ndarray | PIL.Image.Image):
                RGB or greyscale uint8 image. It is not modified.
            rng (numpy.random.Generator | None):
                Generator for this call, such as one seeded per sample.
                Defaults to the augmentor's own generator.

        Returns:
            numpy.ndarray | PIL.Image.Image:
                The augmented image, of the same type and shape as ``image``.

        """
        return self.apply(image, self.sample_operations(rng))

    def sample_operations(
        self: StrongAugment,
        rng: np.random.Generator | None = None,
    ) -> list[Operation]:
        """Draw the operations for one image, with their parameters.

        Args:
            rng (numpy.random.Generator | None):
                Generator to draw from. Defaults to the augmentor's own.

        Returns:
            list[tuple[str, dict[str, float]]]:
                ``(name, parameters)`` pairs, in the order :meth:`apply` applies
                them. The list is empty when zero operations are drawn.

        """
        rng = self.rng if rng is None else rng
        count = int(rng.choice(self.operations, p=self.probabilities))
        names = rng.choice(list(self.augment_space), size=count, replace=False)
        return [(str(name), self._sample_parameters(str(name), rng)) for name in names]

    def _sample_parameters(
        self: StrongAugment, name: str, rng: np.random.Generator
    ) -> dict[str, float]:
        """Sample the magnitudes, and any seed, that operation ``name`` needs."""
        low, high = self.augment_space[name]
        if name in _FLAG_OPERATIONS:
            return {}
        if name in _INTEGER_OPERATIONS:
            return {"magnitude": int(rng.integers(low, high, endpoint=True))}
        if name == "tone":
            return {
                "magnitude_0": float(rng.uniform(low, high)),
                "magnitude_1": float(rng.uniform(low, high)),
            }
        parameters = {"magnitude": float(rng.uniform(low, high))}
        if name == "noise":
            parameters["seed"] = int(rng.integers(2**32))
        return parameters

    @staticmethod
    def apply(
        image: np.ndarray | Image.Image, operations: Sequence[Operation]
    ) -> np.ndarray | Image.Image:
        """Apply operations from :meth:`sample_operations` to an image.

        Args:
            image (numpy.ndarray | PIL.Image.Image):
                RGB or greyscale uint8 image. It is not modified.
            operations (Sequence[tuple[str, dict[str, float]]]):
                ``(name, parameters)`` pairs to apply in order.

        Returns:
            numpy.ndarray | PIL.Image.Image:
                The augmented image, of the same type and shape as ``image``.

        """
        if isinstance(image, Image.Image) and image.mode not in {"RGB", "L"}:
            msg = f"StrongAugment expects an RGB or L mode image, not {image.mode}."
            raise ValueError(msg)
        array = np.array(image)
        if array.dtype != np.uint8 or not (
            array.ndim == _GRAYSCALE_DIMS
            or (array.ndim == _GRAYSCALE_DIMS + 1 and array.shape[-1] == _RGB_CHANNELS)
        ):
            msg = (
                "StrongAugment expects an RGB or greyscale uint8 image, not an "
                f"array of shape {array.shape} and dtype {array.dtype}."
            )
            raise ValueError(msg)
        grayscale = array.ndim == _GRAYSCALE_DIMS
        if grayscale:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        for name, parameters in operations:
            if name not in _OPERATIONS:
                msg = _unsupported_message(name)
                raise ValueError(msg)
            array = _OPERATIONS[name](array, **parameters)
        if grayscale:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(array) if isinstance(image, Image.Image) else array

    def __repr__(self: StrongAugment) -> str:
        """Return the augmentor's settings."""
        return (
            f"{type(self).__name__}(operations={self.operations}, "
            f"probabilities={self.probabilities}, "
            f"augment_space={self.augment_space})"
        )


def _blend(image: np.ndarray, other: np.ndarray, magnitude: float) -> np.ndarray:
    """Return ``magnitude * image + (1 - magnitude) * other`` as uint8."""
    return cv2.addWeighted(image, magnitude, other, 1 - magnitude, 0)


def _scale_channel(image: np.ndarray, magnitude: float, channel: int) -> np.ndarray:
    """Scale one colour channel in place."""
    plane = np.ascontiguousarray(image[..., channel])
    image[..., channel] = _blend(plane, np.zeros_like(plane), magnitude)
    return image


def _shift_hue(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Shift hue with wraparound in OpenCV hue space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # OpenCV stores 8-bit hue as 0-179.
    lut = np.mod(np.arange(256) + 180 * magnitude, 180).astype(np.uint8)
    hsv[..., 0] = cv2.LUT(hsv[..., 0], lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to greyscale while retaining three channels."""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def _adjust_saturation(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Blend with greyscale to adjust colour saturation."""
    return _blend(image, _grayscale(image), magnitude)


def _adjust_brightness(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Scale image brightness by blending with black."""
    return _blend(image, np.zeros_like(image), magnitude)


def _adjust_contrast(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Adjust contrast around the mean greyscale intensity."""
    mean = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()
    return _blend(image, np.full_like(image, mean), magnitude)


def _adjust_gamma(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply gamma correction through an intensity lookup table."""
    lut = np.rint((np.arange(256) / 255) ** magnitude * 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def _solarize(image: np.ndarray, magnitude: int) -> np.ndarray:
    """Invert intensities at or above the solarization threshold."""
    values = np.arange(256)
    lut = np.where(values < magnitude, values, 255 - values).astype(np.uint8)
    return cv2.LUT(image, lut)


def _posterize(image: np.ndarray, magnitude: int) -> np.ndarray:
    # Keep the top `magnitude` bits of each value.
    """Retain the requested number of high bits per channel."""
    return image & np.uint8(256 - 2 ** (8 - magnitude))


def _autocontrast(image: np.ndarray) -> np.ndarray:
    """Stretch each channel to its full intensity range."""
    return np.array(ImageOps.autocontrast(Image.fromarray(image)))


def _equalize(image: np.ndarray) -> np.ndarray:
    """Equalize each colour channel independently."""
    channels = [cv2.equalizeHist(np.ascontiguousarray(image[..., c])) for c in range(3)]
    return np.stack(channels, axis=-1)


def _gaussian_blur(image: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply Gaussian blur with the requested standard deviation."""
    if magnitude <= 0:
        return image
    size = max(3, round(magnitude * 3.5) // 2 * 2 + 1)
    return cv2.GaussianBlur(image, (size, size), sigmaX=magnitude)


def _filter(image: np.ndarray, kernel: np.ndarray, magnitude: float) -> np.ndarray:
    """Blend a filter kernel with the identity before convolution."""
    blended = (1 - magnitude) * _IDENTITY_KERNEL + magnitude * kernel
    return cv2.filter2D(image, -1, blended)


def _jpeg(image: np.ndarray, magnitude: int) -> np.ndarray:
    """Round-trip the image through JPEG at the requested quality."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, magnitude])
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _tone_curve(
    image: np.ndarray, magnitude_0: float, magnitude_1: float
) -> np.ndarray:
    # A cubic Bezier curve from (0, 0) to (1, 1) with two control values.
    """Map intensities through a cubic Bezier tone curve."""
    t = np.linspace(0.0, 1.0, 256)
    curve = 3 * (1 - t) ** 2 * t * magnitude_0 + 3 * (1 - t) * t**2 * magnitude_1 + t**3
    return cv2.LUT(image, np.rint(curve * 255).astype(np.uint8))


def _add_noise(image: np.ndarray, magnitude: float, seed: int) -> np.ndarray:
    """Blend with seeded greyscale noise shared across channels."""
    noise = np.random.default_rng(seed).integers(
        0, 256, image.shape[:2], dtype=np.uint8
    )
    return _blend(image, np.stack([noise] * _RGB_CHANNELS, axis=-1), 1 - magnitude)


_OPERATIONS: dict[str, Callable[..., np.ndarray]] = {
    "red": lambda image, magnitude: _scale_channel(image, magnitude, 0),
    "green": lambda image, magnitude: _scale_channel(image, magnitude, 1),
    "blue": lambda image, magnitude: _scale_channel(image, magnitude, 2),
    "hue": _shift_hue,
    "saturation": _adjust_saturation,
    "brightness": _adjust_brightness,
    "contrast": _adjust_contrast,
    "gamma": _adjust_gamma,
    "solarize": _solarize,
    "posterize": _posterize,
    "sharpen": lambda image, magnitude: _filter(image, _SHARPEN_KERNEL, magnitude),
    "emboss": lambda image, magnitude: _filter(image, _EMBOSS_KERNEL, magnitude),
    "blur": _gaussian_blur,
    "noise": _add_noise,
    "jpeg": _jpeg,
    "tone": _tone_curve,
    "autocontrast": _autocontrast,
    "equalize": _equalize,
    "grayscale": _grayscale,
}
_INTEGER_OPERATIONS = frozenset({"solarize", "posterize", "jpeg"})
_FLAG_OPERATIONS = frozenset({"autocontrast", "equalize", "grayscale"})
# Inclusive limits for magnitude bounds; None leaves the upper bound open.
_BOUND_LIMITS: dict[str, tuple[float, float | None]] = {
    "hue": (-0.5, 0.5),
    "solarize": (0, 256),
    "posterize": (1, 8),
    "jpeg": (0, 100),
    "tone": (0.0, 1.0),
}


def _unsupported_message(name: str) -> str:
    """Describe an unsupported operation and list the valid choices."""
    return (
        f"Unsupported StrongAugment operation `{name}`. "
        f"Choose from {', '.join(_OPERATIONS)}."
    )


def _check_magnitude_bounds(name: str, bounds: Bounds) -> None:
    """Validate numeric magnitude types and operation-specific limits."""
    number = Integral if name in _INTEGER_OPERATIONS else Real
    if any(
        isinstance(value, bool) or not isinstance(value, number) for value in bounds
    ):
        kind = "integers" if number is Integral else "numbers"
        msg = f"Bounds for `{name}` must be {kind}."
        raise TypeError(msg)
    minimum, maximum = _BOUND_LIMITS.get(name, (0, None))
    low, high = bounds
    if not minimum <= low <= high or (maximum is not None and high > maximum):
        upper = f" <= {maximum}" if maximum is not None else ""
        msg = f"Bounds for `{name}` must satisfy {minimum} <= low <= high{upper}."
        raise ValueError(msg)


def _check_augment_space(space: dict[str, Bounds]) -> None:
    """Check operation names, and the types and ranges of their bounds."""
    for name, bounds in space.items():
        if name not in _OPERATIONS:
            raise ValueError(_unsupported_message(name))
        if not isinstance(bounds, (tuple, list)) or len(bounds) != _PAIR:
            msg = f"Bounds for `{name}` must be a `(low, high)` pair."
            raise TypeError(msg)
        if name in _FLAG_OPERATIONS:
            if not all(isinstance(value, bool) for value in bounds):
                msg = f"`{name}` has no magnitude; give its bounds as `(True, True)`."
                raise TypeError(msg)
            continue
        _check_magnitude_bounds(name, bounds)
