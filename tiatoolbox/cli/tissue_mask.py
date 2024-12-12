"""Command line interface for tissue_mask."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_method,
    cli_mode,
    cli_output_path,
    cli_resolution,
    cli_units,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.tools.tissuemask import TissueMasker


def get_masker(
    method: str,
    kernel_size: tuple[int, int],
    units: str,
    resolution: float,
) -> TissueMasker:
    """Get Tissue Masker."""
    from tiatoolbox.tools import tissuemask

    if method == "Otsu":
        return tissuemask.OtsuTissueMasker()

    if kernel_size:
        return tissuemask.MorphologicalMasker(kernel_size=kernel_size)

    if units == "mpp":
        return tissuemask.MorphologicalMasker(mpp=resolution)

    return tissuemask.MorphologicalMasker(power=resolution)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(default="tissue_mask")
@cli_method(default="Otsu")
@cli_resolution(default=1.25)
@cli_units(
    default="power",
    input_type=click.Choice(["mpp", "power"], case_sensitive=False),
)
@cli_mode(default="show")
@cli_file_type(default="*.svs, *.ndpi, *.jp2, *.png, *.jpg, *.tif, *.tiff")
# inputs specific to this function
@click.option(
    "--kernel-size",
    type=int,
    nargs=2,
    help="kernel size for morphological dilation, default=1, 1",
)
def tissue_mask(
    img_input: str,
    output_path: str,
    method: str,
    resolution: float,
    units: str,
    kernel_size: [int, int],
    mode: str,
    file_types: str,
) -> None:
    """Generate tissue mask for a WSI."""
    import numpy as np
    from PIL import Image

    from tiatoolbox.utils import imwrite
    from tiatoolbox.wsicore.wsireader import WSIReader

    files_all, output_path = prepare_file_dir_cli(
        img_input,
        output_path,
        file_types,
        mode,
        "meta-data",
    )

    masker = get_masker(method, kernel_size, units, resolution)

    for curr_file in files_all:
        wsi = WSIReader.open(input_img=curr_file)
        wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
        mask = masker.fit_transform(wsi_thumb[np.newaxis, :])

        if mode == "show":  # pragma: no cover
            # Skipped on CI, and unless SHOW_TESTS is set
            im_region = Image.fromarray(mask[0])
            im_region.show()
            continue

        # Else, save (the only other option for mode)
        imwrite(
            output_path.joinpath(Path(curr_file).stem + ".png"),
            mask[0].astype(np.uint8) * 255,
        )
