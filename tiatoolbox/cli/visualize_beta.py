"""Command line interface for the experimental OpenLayers viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from tiatoolbox.cli.common import (
    cli_base_path,
    cli_overlays,
    cli_port,
    cli_slides,
    tiatoolbox_cli,
)

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


@tiatoolbox_cli.command(name="visualize-beta")
@cli_base_path(
    usage_help="""Path to base directory containing images to be displayed.
    Slides and overlays to be visualized are expected in subdirectories of the
    base directory named slides and overlays, respectively. It is also possible
    to provide a slide and overlay path separately
    (use --slides and --overlays).""",
)
@cli_slides(
    usage_help="""Path to directory containing slides to be displayed.
    This option must be used in conjunction with --overlays.
    The --base-path option should not be used in this case.""",
)
@cli_overlays(
    usage_help="""Path to directory containing overlays to be displayed.
    This option must be used in conjunction with --slides.
    The --base-path option should not be used in this case.""",
)
@cli_port(default=5000)
def visualize_beta(
    base_path: Path | None,
    slides: Path | None,
    overlays: Path | None,
    port: int,
) -> None:
    """Launch the experimental TIAToolbox visualization tool for given directory(s).

    If only base-path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays, respectively.

    Args:
        base_path (Path or None):
            Path to base directory containing images to be displayed.
        slides (Path or None):
            Path to directory containing slides to be displayed.
        overlays (Path or None):
            Path to directory containing overlays to be displayed.
        port (int):
            Port to launch the visualization tool on.

    """
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
