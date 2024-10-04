"""This module contains a class for creating bar plots for CSV files in a folder."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import dodge

from tiatoolbox.utils.visualization import random_colors
from tiatoolbox.visualization.ui_utils import UIPlugin


class BarPlotGrid(UIPlugin):
    """Class for creating a grid of bar plots for CSV files in a folder."""

    def create_extra_layout(
        self: UIPlugin,
        slide_path: Path,
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Creates a grid-like layout of bar charts for each CSV file within a folder.

        Args:
            slide_path (str): The path to the slide for which the extra layout is
            being created
            UI: contains the main UI elements of the bokeh app
            old_children: contains the previous children of the layout

        Returns:
            list: A list containing the new children of the extra layout
        """
        folder_path = slide_path.with_name(slide_path.stem + "_files")
        if not folder_path.is_dir():
            return []
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        plots = []

        for file in csv_files:
            filepath = folder_path / file
            csv_data = pd.read_csv(filepath, header=0, index_col=0)

            # Get tags and labels for the x-axis
            tags = csv_data.index.tolist()
            x_labels = csv_data.columns.tolist()
            colors = random_colors(len(tags), bright=True)
            colors = [
                f"{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in 255 * colors
            ]

            # Prepare data for Bokeh
            data = {"x": x_labels}
            for tag in tags:
                data[tag] = csv_data.loc[tag].tolist()

            source = ColumnDataSource(data=data)

            # Create the figure (customize as needed)
            p = figure(
                x_range=x_labels,
                title=f"Data from {file}",
                height=700,
                width=900,
                toolbar_location=None,
                tools="",
            )

            # Add bars for each tag
            r = np.array(range(len(tags)))
            dodge_vals = (r - np.mean(r)) / ((len(r) - 1) * 2)
            for i, tag in enumerate(tags):
                p.vbar(
                    x=dodge("x", dodge_vals[i], range=p.x_range),
                    top=tag,
                    width=0.6 / len(tags),
                    source=source,
                    legend_label=tag,
                    color=colors[i],
                )

            # Customize axes and appearance
            p.x_range.range_padding = 0.1
            p.xgrid.grid_line_color = None
            p.axis.minor_tick_line_color = None
            p.outline_line_color = None

            # Add a hover tool for more information
            hover = HoverTool()
            hover.tooltips = [(tag, f"@{tag}") for tag in tags]
            p.add_tools(hover)

            plots.append(p)

        # Arrange plots in a grid
        grid = gridplot(plots, ncols=2)

        return [grid]

    def create_extra_layout_once(
        self: UIPlugin,
        slide_path: str,  # noqa: ARG002
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Create extra layout elements on window initialization."""
        return []
