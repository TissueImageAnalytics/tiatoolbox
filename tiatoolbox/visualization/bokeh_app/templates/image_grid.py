"""Contains a class for creating a Bokeh grid layout of images from a folder."""

from pathlib import Path

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from PIL import Image

from tiatoolbox.visualization.ui_utils import UIPlugin


class ImageGrid(UIPlugin):
    """Class for creating a grid of images from a folder."""

    def create_extra_layout(
        self: UIPlugin,
        slide_path: Path,
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Creates a Bokeh grid layout of images from a specified folder.

        Args:
            slide_path (str): Path to slide for which the extra layout is being created
            UI: contains the main UI elements of the bokeh app
            old_children: contains the previous children of the layout

        Returns:
            list: A list containing the new children of the extra layout
        """
        image_folder_path = slide_path.with_name(slide_path.stem + "_files")
        if not image_folder_path.is_dir():
            return []
        # Find supported images in the folder
        search_patterns = ["*.jpg", "*.png"]
        image_paths = []
        for pattern in search_patterns:
            image_paths.extend(image_folder_path.glob(pattern))

        # Create figures for each image
        figures = []
        for image_path in image_paths:
            img_pil = Image.open(image_path)
            img_array = np.array(img_pil)
            if img_array.shape[2] == 3:  # noqa: PLR2004
                img_array = np.dstack(
                    [
                        img_array,
                        np.full(
                            (img_array.shape[0], img_array.shape[1]),
                            255,
                            dtype=np.uint8,
                        ),
                    ],
                )
            # bokeh needs as 2D uint32
            img_rgba = img_array.view(dtype=np.uint32).reshape(
                (img_array.shape[0], img_array.shape[1]),
            )

            p = figure(
                x_range=(0, 100),
                y_range=(0, 100),
                width=400,
                height=400,
                title=image_path.stem,
            )
            p.axis.visible = False
            p.image_rgba(image=[img_rgba], x=0, y=0, dw=100, dh=100)
            figures.append(p)

        # Arrange figures in a grid layout
        grid = gridplot(figures, ncols=3)

        return [grid]

    def create_extra_layout_once(
        self: UIPlugin,
        slide_path: str,  # noqa: ARG002
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Create extra layout elements on widow initialization."""
        return []
