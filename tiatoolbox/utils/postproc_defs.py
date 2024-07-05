"""Module to provide postprocessing classes."""

from __future__ import annotations

import colorsys

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
        print(n)
        print(self.color_dict)

        if n < 5:  # noqa: PLR2004
            # assume already rgb(a) so just return image
            return image

        if self.colors is None:
            self.generate_colors(n)

        # Convert to RGB image
        rgb_image = np.einsum("hwn,nc->hwc", image, self.colors[:, :], optimize=True)

        # Clip  to ensure in valid range and return
        return np.clip(rgb_image, 0, 255).astype(np.uint8)

    def __setattr__(self: MultichannelToRGB, name: str, value: np.Any) -> None:
        """Ensure that colors is updated if color_dict is updated."""
        if name == "color_dict" and value is not None:
            self.colors = np.array(list(value.values()), dtype=np.float32)

        super().__setattr__(name, value)
