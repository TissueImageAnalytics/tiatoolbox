"""Command line interface for the experimental OpenLayers viewer."""

from tiatoolbox.cli.common import tiatoolbox_cli
from tiatoolbox.visualization.visualize_beta_tileserver import VisualizeBetaTileServer


@tiatoolbox_cli.command(name="visualize-beta")
def visualize_beta() -> None:  # pragma: no cover
    """Launch the experimental dynamic OpenLayers viewer."""
    app = VisualizeBetaTileServer(
        title="TIAToolbox OpenLayers beta",
        layers={},
    )

    app.run(
        host="127.0.0.1",
        port=5000,
    )
