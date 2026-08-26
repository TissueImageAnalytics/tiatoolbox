"""Tests for the experimental OpenLayers viewer CLI."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from tiatoolbox.cli.visualize_beta import visualize_beta


def test_visualize_beta_no_directories() -> None:
    """Test launching without configured directories."""
    runner = CliRunner()

    with patch("tiatoolbox.visualization.tileserver.TileServer") as mock_server:
        result = runner.invoke(visualize_beta)

    assert result.exit_code == 0

    mock_server.assert_called_once_with(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
        slide_directory=None,
        overlay_directory=None,
    )

    mock_server.return_value.run.assert_called_once_with(
        host="127.0.0.1",
        port=5000,
    )


def test_visualize_beta_base_path(tmp_path: Path) -> None:
    """Test launching with a base directory."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()

    runner = CliRunner()

    with patch("tiatoolbox.visualization.tileserver.TileServer") as mock_server:
        result = runner.invoke(
            visualize_beta,
            ["--base-path", str(tmp_path)],
        )

    assert result.exit_code == 0

    mock_server.assert_called_once_with(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
        slide_directory=slides,
        overlay_directory=overlays,
    )


def test_visualize_beta_explicit_directories(
    tmp_path: Path,
) -> None:
    """Test launching with separate slide and overlay directories."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()

    runner = CliRunner()

    with patch("tiatoolbox.visualization.tileserver.TileServer") as mock_server:
        result = runner.invoke(
            visualize_beta,
            [
                "--slides",
                str(slides),
                "--overlays",
                str(overlays),
            ],
        )

    assert result.exit_code == 0

    mock_server.assert_called_once_with(
        title="TIAToolbox OpenLayers beta",
        layers={},
        legacy=False,
        slide_directory=slides,
        overlay_directory=overlays,
    )


def test_visualize_beta_base_path_with_explicit_directory(
    tmp_path: Path,
) -> None:
    """Test that base path cannot be combined with explicit directories."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()

    runner = CliRunner()

    result = runner.invoke(
        visualize_beta,
        [
            "--base-path",
            str(tmp_path),
            "--slides",
            str(slides),
        ],
    )

    assert result.exit_code != 0
    assert (
        "--base-path cannot be used together with --slides or --overlays."
        in result.output
    )


def test_visualize_beta_requires_both_directories(
    tmp_path: Path,
) -> None:
    """Test that slides and overlays must be provided together."""
    slides = tmp_path / "slides"
    slides.mkdir()

    runner = CliRunner()

    result = runner.invoke(
        visualize_beta,
        ["--slides", str(slides)],
    )

    assert result.exit_code != 0
    assert "--slides and --overlays must be provided together." in result.output


def test_visualize_beta_missing_slides_directory(
    tmp_path: Path,
) -> None:
    """Test a base path without a slides directory."""
    (tmp_path / "overlays").mkdir()

    runner = CliRunner()

    result = runner.invoke(
        visualize_beta,
        ["--base-path", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "Slides directory does not exist:" in result.output


def test_visualize_beta_missing_overlays_directory(
    tmp_path: Path,
) -> None:
    """Test a base path without an overlays directory."""
    (tmp_path / "slides").mkdir()

    runner = CliRunner()

    result = runner.invoke(
        visualize_beta,
        ["--base-path", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "Overlays directory does not exist:" in result.output


def test_visualize_beta_port() -> None:
    """Test launching on a custom port."""
    runner = CliRunner()

    with patch("tiatoolbox.visualization.tileserver.TileServer") as mock_server:
        result = runner.invoke(
            visualize_beta,
            ["--port", "5001"],
        )

    assert result.exit_code == 0

    mock_server.return_value.run.assert_called_once_with(
        host="127.0.0.1",
        port=5001,
    )
