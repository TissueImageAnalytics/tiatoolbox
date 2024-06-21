"""Module to provide postprocessing classes."""

from __future__ import annotations

import colorsys

import numpy as np


class MultichannelToRGB:
    """Class to convert multi-channel images to RGB images."""

    def __init__(
        self: MultichannelToRGB,
        default_colors: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialize the MultichannelToRGB converter.

        Args:
            default_colors: List of default RGB colors for each channel. If not
            provided, colors will be auto-generated using the HSV color space.

        """
        self.default_colors = default_colors
        if self.default_colors is not None:
            self.default_colors = np.array(self.default_colors, dtype=np.float32)

    def generate_colors(self: MultichannelToRGB, n_channels: int) -> np.ndarray:
        """Generate a set of visually distinct colors.

        Args:
            n_channels (int): Number of channels/colors to generate

        Returns:
            np.ndarray: Array of RGB colors

        """
        return np.array(
            [colorsys.hsv_to_rgb(i / n_channels, 1, 1) for i in range(n_channels)],
            dtype=np.float32,
        )

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

        if self.default_colors is None:
            colors = self.generate_colors(n)
        else:
            colors = self.default_colors

        # Convert to RGB image
        rgb_image = np.einsum("hwn,nc->hwc", image, colors, optimize=True)

        # Clip  to ensure in valid range and return
        return np.clip(rgb_image, 0, 255).astype(np.uint8)
