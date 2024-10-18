"""Module to provide postprocessing classes."""

from __future__ import annotations

import colorsys
import warnings

import numpy as np


class MultichannelToRGB:
    """Class to convert multi-channel images to RGB images."""

    def __init__(
        self: MultichannelToRGB,
        color_dict: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialize the MultichannelToRGB converter.

        Args:
            color_dict: Dict of channel names with RGB colors for each channel. If not
            provided, a set of distinct colors will be auto-generated.

        """
        self.colors = None
        self.color_dict = color_dict
        self.is_validated = False
        self.channels = None
        self.enhance = 1.0

    def validate(self: MultichannelToRGB, n: int) -> None:
        """Validate the input color_dict on first read from image.

        Checks that n is either equal to the number of colors provided, or is
        one less. In the latter case it is assumed that the last channel is background
        autofluorescence and is not in the tiff and we will drop it from
        the color_dict with a warning.

        Args:
            n (int): Number of channels

        """
        n_colors = len(self.colors)
        if n_colors == n:
            self.is_validated = True
            return

        if n_colors - 1 == n:
            self.colors = self.colors[:n]
            self.channels = [c for c in self.channels if c < n]
            self.is_validated = True
            warnings.warn(
                """Number of channels in image is one less than number of channels in
                dict. Assuming last channel is background autofluorescence and ignoring
                it. If this is not the case please provide a manual color_dict.""",
                stacklevel=2,
            )
            return

        msg = f"Number of colors: {n_colors} does not match channels in image: {n}."
        raise ValueError(msg)

    def generate_colors(self: MultichannelToRGB, n_channels: int) -> np.ndarray:
        """Generate a set of visually distinct colors.

        Args:
            n_channels (int): Number of channels/colors to generate

        Returns:
            np.ndarray: Array of RGB colors

        """
        self.color_dict = {
            f"channel_{i}": colorsys.hsv_to_rgb(i / n_channels, 1, 1)
            for i in range(n_channels)
        }

    def __call__(self: MultichannelToRGB, image: np.ndarray) -> np.ndarray:
        """Convert a multi-channel image to an RGB image.

        Args:
            image (np.ndarray): Input image of shape (H, W, N)

        Returns:
            np.ndarray: RGB image of shape (H, W, 3)

        """
        n = image.shape[2]

        if n < 5:  # noqa: PLR2004
            # assume already rgb(a) so just return image
            return image

        if self.colors is None:
            self.generate_colors(n)

        if not self.is_validated:
            self.validate(n)

        # Convert to RGB image
        rgb_image = (
            np.einsum(
                "hwn,nc->hwc",
                image[:, :, self.channels],
                self.colors[self.channels, :],
                optimize=True,
            )
            * self.enhance
        )

        # Clip  to ensure in valid range and return
        return np.clip(rgb_image, 0, 255).astype(np.uint8)

    def __setattr__(self: MultichannelToRGB, name: str, value: np.Any) -> None:
        """Ensure that colors is updated if color_dict is updated."""
        if name == "color_dict" and value is not None:
            self.colors = np.array(list(value.values()), dtype=np.float32)
            if self.channels is None:
                self.channels = list(range(len(value)))

        super().__setattr__(name, value)
