"""Command line interface for slide_thumbnail."""

from pathlib import Path

from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output directory to save the output. "
    "default=img_input/../slide-thumbnail",
)
@cli_file_type(default="*.ndpi, *.svs, *.mrxs, *.jp2")
@cli_mode(default="save")
def slide_thumbnail(
    img_input: str,
    output_path: str,
    file_types: str,
    mode: str,
) -> None:
    """Reads whole slide image thumbnail and shows or saves based on mode argument.

    The default inputs are:

    img-input='', output-path=img-input-path/../meta-data,  mode="save",
    file-types="*.ndpi, *.svs, *.mrxs, *.jp2".

    """
    from PIL import Image

    from tiatoolbox.utils import imwrite
    from tiatoolbox.wsicore.wsireader import WSIReader

    files_all, output_path = prepare_file_dir_cli(
        img_input,
        output_path,
        file_types,
        mode,
        "slide-thumbnail",
    )

    for curr_file in files_all:
        wsi = WSIReader.open(input_img=curr_file)

        slide_thumb = wsi.slide_thumbnail()
        if mode == "show":  # pragma: no cover
            # Skipped on CI, and unless SHOW_TESTS is set
            im_region = Image.fromarray(slide_thumb)
            im_region.show()

        # the only other option left for mode is "save".
        imwrite(output_path / (Path(curr_file).stem + ".jpg"), slide_thumb)
