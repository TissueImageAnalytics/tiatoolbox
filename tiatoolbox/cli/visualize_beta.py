"""Command line interface for the experimental OpenLayers viewer."""

from __future__ import annotations

import click

from tiatoolbox.cli.common import tiatoolbox_cli


@tiatoolbox_cli.command(name="visualize-beta")
@click.option(
    "--port",
    type=int,
    help="Port to launch the visualization tool on.",
    default=5000,
)
def visualize_beta(
    port: int,
) -> None:  # pragma: no cover
    """Launches the experimental TIAToolbox visualization tool.

    Args:
        port (int): Port to launch the visualization tool on.

    """
    from tiatoolbox.visualization.tileserver import TileServer  # noqa: PLC0415

    app = TileServer(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
    )

    app.run(
        host="127.0.0.1",
        port=port,
    )
