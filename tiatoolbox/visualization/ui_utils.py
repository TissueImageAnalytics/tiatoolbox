"""Some utilities for bokeh ui."""

from __future__ import annotations

from abc import ABC, abstractmethod
from cmath import pi
from typing import Any

import numpy as np
from bokeh.models import Dialog, Model, OpenDialog

scale_factor = 2
init_res = 40211.5 * scale_factor * (2 / (100 * pi))
min_zoom = 0
max_zoom = 10
resolutions = [init_res / 2**lev for lev in range(min_zoom, max_zoom + 1)]


def get_level_by_extent(extent: tuple[float, float, float, float]) -> int:
    """Replicate the Bokeh tile renderer `get_level_by_extent` function."""
    x_rs = (extent[2] - extent[0]) / 1700
    y_rs = (extent[3] - extent[1]) / 1000
    resolution = np.maximum(x_rs, y_rs)

    i = 0
    for r in resolutions:
        if resolution > r:
            if i == 0:
                return 0
            return i - 1
        i += 1

    # Otherwise return the highest available resolution
    return i - 1


class UIWrapper:
    """Wrapper class to access ui elements."""

    def __init__(self: UIWrapper, win_dicts: list) -> None:
        """Initialize the class."""
        self.active = 0
        self.win_dicts = win_dicts

    def __getitem__(self: UIWrapper, key: str) -> Any:  # noqa: ANN401
        """Gets ui element for the active window."""
        return self.win_dicts[self.active][key]


class UIPlugin(ABC):
    """Abstract class for UI plugins.

    To add a new UI plugin, create a new class
    in its own file that inherits from this class and implements the abstract
    methods. The class must be named, in camel case, the same as the file that
    defines it, for example:
    image_grid.py -> ImageGrid
    bar_plot_grid.py -> BarPlotGrid
    To launch the UI with specified plugin(s), add --plugin options to the cli command
    specifiying the path to each plugin .py file. For example:
    tiatoolbox visualize --slides path/to/slides --overlays path/to/overlays
        --plugin path/to/image_grid.py --plugin path/to/bar_plot_grid.py

    We provide a few pre-built plugins in tiatoolbox.visualization.bokeh_app.templates
    folder. These can be used as a starting point for creating new plugins, or as-is if
    they happen to meet your needs. Provided plugins are:
    - ImageGrid: Displays a grid of images below the main view window from a specified
    folder for each slide.
    - BarPlotGrid: Displays a grid of bar plots below the main view window for each csv
    file found within a folder for each slide.
    - StatsPlot: Displays a histogram of the distributions of properties of annotations
    contained within the selected box-shaped region of the slide.
    """

    def __init__(self: UIPlugin, ui: UIWrapper) -> None:
        """Initialize the class."""
        self.UI = ui

    @abstractmethod
    def create_extra_layout(
        self: UIPlugin,
        slide_path: str,
        old_children: list,
    ) -> list:
        """Create extra layout that will be placed below the main view window.

        Will be run every time the slide is changed.
        """
        raise NotImplementedError

    def add_to_ui(self: UIPlugin) -> None:  # noqa: B027
        """Insert a UI element into the existing main UI panel.

        Will be run every time the slide is changed.
        """

    @abstractmethod
    def create_extra_layout_once(
        self: UIPlugin,
        slide_path: str,
        old_children: list,
    ) -> list:
        """Create extra layout that will be placed below the main view window.

        Will be run only once on loading UI.
        """
        raise NotImplementedError

    def add_to_ui_once(self: UIPlugin) -> None:  # noqa: B027
        """Insert a UI element into the existing main UI panel.

        Will be run only once on loading UI.
        """


def make_into_popup(layout: Model, trigger: callable, title: str = "popup") -> Dialog:
    """Make a layout into a popup, that will open on trigger callback."""
    dialog = Dialog(
        title=title,
        content=layout,
    )
    trigger(OpenDialog(dialog=dialog))

    return dialog
