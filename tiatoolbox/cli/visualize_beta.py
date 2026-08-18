"""Command line interface for the experimental OpenLayers viewer."""

from tiatoolbox.cli.common import tiatoolbox_cli
from tiatoolbox.visualization.tileserver import TileServer


@tiatoolbox_cli.command(name="visualize-beta")
def visualize_beta() -> None:  # pragma: no cover
    """Launch the experimental dynamic OpenLayers viewer."""
    app = TileServer(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
    )

    app.run(
        host="127.0.0.1",
        port=5000,
    )
