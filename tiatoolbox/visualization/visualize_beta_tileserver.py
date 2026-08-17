"""Tile server for the experimental OpenLayers viewer."""

from __future__ import annotations

from flask.templating import render_template

from tiatoolbox.visualization.tileserver import TileServer


class VisualizeBetaTileServer(TileServer):
    """TileServer entry point for the experimental OpenLayers viewer."""

    def index(self: VisualizeBetaTileServer) -> str:
        """Serve the experimental OpenLayers viewer."""
        return render_template(
            "visualize_beta.html",
            title=self.title,
            layers="[]",
        )
