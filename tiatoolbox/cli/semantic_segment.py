"""Command line interface for semantic segmentation."""

from __future__ import annotations

import click

from tiatoolbox.cli.common import (
    cli_batch_size,
    cli_device,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_mode,
    cli_num_loader_workers,
    cli_output_path,
    cli_pretrained_model,
    cli_pretrained_weights,
    cli_verbose,
    cli_yaml_config_path,
    prepare_ioconfig_seg,
    prepare_model_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Output directory where model predictions will be saved.",
    default="semantic_segmentation",
)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@cli_mode(
    usage_help="Type of input file to process.",
    default="wsi",
    input_type=click.Choice(["patch", "wsi", "tile"], case_sensitive=False),
)
@cli_pretrained_model(default="fcn-tissue_mask")
@cli_pretrained_weights(default=None)
@cli_device()
@cli_batch_size()
@cli_masks(default=None)
@cli_yaml_config_path()
@cli_num_loader_workers()
@cli_verbose()
def semantic_segment(
    pretrained_model: str,
    pretrained_weights: str,
    img_input: str,
    file_types: str,
    masks: str | None,
    mode: str,
    output_path: str,
    batch_size: int,
    yaml_config_path: str,
    num_loader_workers: int,
    device: str,
    *,
    verbose: bool,
) -> None:
    """Process an image/directory of input images with a patch classification CNN."""
    from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
    from tiatoolbox.utils import save_as_json

    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
    )

    ioconfig = prepare_ioconfig_seg(
        IOSegmentorConfig,
        pretrained_weights,
        yaml_config_path,
    )

    predictor = SemanticSegmentor(
        pretrained_model=pretrained_model,
        pretrained_weights=pretrained_weights,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        verbose=verbose,
    )

    output = predictor.predict(
        imgs=files_all,
        masks=masks_all,
        mode=mode,
        device=device,
        save_dir=output_path,
        ioconfig=ioconfig,
    )

    save_as_json(output, str(output_path.joinpath("results.json")))
