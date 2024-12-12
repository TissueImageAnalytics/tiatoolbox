"""Module for creating a histogram of distributions of properties of annotations."""

import json
from pathlib import Path

import numpy as np
from bokeh.models import Button, Column, ColumnDataSource, Select
from bokeh.plotting import figure
from scipy.stats import gaussian_kde

from tiatoolbox import logger
from tiatoolbox.visualization.ui_utils import UIPlugin, make_into_popup

popup_list = []


def make_histogram(consolidate_props: dict, popup_layout: Column) -> None:
    """Create a histogram and probability density function for the given data."""
    property_select = Select(title="Property", options=[])
    p = figure(
        width=1000,
        height=600,
        toolbar_location=None,
        title="Property statistics within selected region",
    )
    popup_layout.children = [property_select, p]
    property_select.options = list(consolidate_props.keys())
    property_select.value = property_select.options[0]
    prop = property_select.value

    def property_select_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
        """Update the histogram and pdf line when new property is selected."""
        prop = property_select.value
        update_histogram(consolidate_props[prop], p)

    property_select.on_change("value", property_select_cb)

    x = consolidate_props[prop]
    bins = np.linspace(np.min(x), np.max(x), 40)
    hist, edges = np.histogram(x, density=True, bins=bins)
    source = ColumnDataSource(
        data={
            "top": hist.tolist(),
            "bottom": np.zeros((39,)).tolist(),
            "left": edges[:-1].tolist(),
            "right": edges[1:].tolist(),
        },
    )
    p.quad(
        top="top",
        bottom="bottom",
        left="left",
        right="right",
        source=source,
        fill_color="skyblue",
        line_color="white",
        legend_label="counts",
    )

    # Estimate probability density function by kernel smoothing
    eval_pts = np.linspace(np.min(x), np.max(x), 100)
    pdf = gaussian_kde(x).evaluate(eval_pts)

    source = ColumnDataSource(data={"x": eval_pts, "pdf": pdf})
    p.line(
        "x",
        "pdf",
        source=source,
        line_color="#FF8888",
        line_width=4,
        alpha=0.7,
        legend_label="Probability Density Function",
    )

    p.y_range.start = 0
    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "PDF(x)"


def update_histogram(x: list, p: figure) -> None:
    """Update histrogram and pdf line in the plot."""
    bins = np.linspace(np.min(x), np.max(x), 40)
    hist, edges = np.histogram(x, density=True, bins=bins)
    if len(p.renderers) == 0:
        return
    p.renderers[0].data_source.data = {
        "top": hist.tolist(),
        "bottom": np.zeros((39,)).tolist(),
        "left": edges[:-1].tolist(),
        "right": edges[1:].tolist(),
    }

    # Probability density function
    eval_pts = np.linspace(np.min(x), np.max(x), 100)
    pdf = gaussian_kde(x).evaluate(eval_pts)
    p.renderers[1].data_source.data = {"x": eval_pts, "pdf": pdf}


class StatsPlot(UIPlugin):
    """Class for creating a histogram of distributions of properties of annotations."""

    def create_extra_layout(
        self: UIPlugin,
        slide_path: Path,  # noqa: ARG002
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Create extra layout that will be placed below the main view window."""
        return []

    def create_extra_layout_once(
        self: UIPlugin,
        slide_path: str,  # noqa: ARG002
        old_children: list,  # noqa: ARG002
    ) -> list:
        """Create extra layout elements on widow initialization."""
        return []

    def add_to_ui_once(self: UIPlugin) -> None:
        """Creates a UI element to calculate annotation stats in a region.

        Calculates stats of properties of the annotations contained
        within the selected box, and displays the results in a histogram plot.

        Args:
            slide_path (str): The path to the slide for which the extra layout is being
            created
            UI: contains the main UI elements of the bokeh app
            old_children: contains the previous children of the layout

        Returns:
            list: A list containing the new children of the extra layout
        """
        get_stats_btn = Button(
            label="Get Stats",
            button_type="success",
            height=47,
            width=160,
        )
        popup_layout = Column(children=[], sizing_mode="fixed")

        def get_stats_cb(attr: str) -> None:  # noqa: ARG001
            """Get the stats of the annotations in the selected region."""
            box = self.UI["box_source"]
            if len(box.data["x"]) > 0:
                x = round(
                    box.data["x"][0] - 0.5 * box.data["width"][0],
                )
                y = -round(
                    box.data["y"][0] + 0.5 * box.data["height"][0],
                )
                width = round(box.data["width"][0])
                height = round(box.data["height"][0])
            else:
                logger.info("No box selected")
                return

            # Get the selected regions annotations
            resp = self.UI["s"].get(
                "http://127.0.0.1:5000/tileserver/annotations",
                data={
                    "bounds": json.dumps([x, y, x + width, y + height]),
                    "where": json.dumps(None),
                },
            )
            anns = json.loads(resp.text)
            consolidate_props = {}
            for ann in anns:
                for prop, v in ann["properties"].items():
                    if prop == "":
                        continue
                    if prop not in consolidate_props:
                        consolidate_props[prop] = []
                    consolidate_props[prop].append(v)

            self.UI["vstate"].consolidate_props = consolidate_props
            # Create a histogram of the selected property
            make_histogram(consolidate_props, popup_layout)

        # associate callbaks
        get_stats_btn.on_click(get_stats_cb)
        make_into_popup(
            popup_layout,
            get_stats_btn.js_on_click,
            "Stats",
        )

        # add to extra_options column children
        self.UI["extra_options"].children.append(get_stats_btn)
