"""Command line interface for the experimental OpenLayers viewer."""

from __future__ import annotations

from pathlib import Path

import click

from tiatoolbox.cli.common import tiatoolbox_cli


@tiatoolbox_cli.command(name="visualize-beta")
@click.option(
    "--base-path",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help=("Base directory containing 'slides' and 'overlays' subdirectories."),
)
@click.option(
    "--slides",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help="Directory containing slides.",
)
@click.option(
    "--overlays",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help="Directory containing overlays.",
)
@click.option(
    "--port",
    type=int,
    help="Port to launch the visualization tool on.",
    default=5000,
)
def visualize_beta(
    base_path: Path | None,
    slides: Path | None,
    overlays: Path | None,
    port: int,
) -> None:  # pragma: no cover
    """Launch the experimental TIAToolbox visualization tool."""
    from tiatoolbox.visualization.tileserver import TileServer  # noqa: PLC0415

    if base_path is not None and (slides is not None or overlays is not None):
        msg = "--base-path cannot be used together with --slides or --overlays."
        raise click.UsageError(msg)

    if base_path is not None:
        slides = base_path / "slides"
        overlays = base_path / "overlays"

        if not slides.is_dir():
            msg = f"Slides directory does not exist: {slides}"
            raise click.UsageError(msg)

        if not overlays.is_dir():
            msg = f"Overlays directory does not exist: {overlays}"
            raise click.UsageError(msg)

    elif (slides is None) != (overlays is None):
        msg = "--slides and --overlays must be provided together."
        raise click.UsageError(msg)

    app = TileServer(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
        slide_directory=slides,
        overlay_directory=overlays,
    )

    app.run(
        host="127.0.0.1",
        port=port,
    )
