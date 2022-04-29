"""Command line interface for nucleus instance segmentation."""
import click

from tiatoolbox.cli.common import (
    cli_auto_generate_mask,
    cli_batch_size,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_mode,
    cli_num_loader_workers,
    cli_num_postproc_workers,
    cli_on_gpu,
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
    default="nucleus_instance_segmentation",
)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs"
)
@cli_mode(
    usage_help="Type of input file to process.",
    default="wsi",
    input_type=click.Choice(["patch", "wsi", "tile"], case_sensitive=False),
)
@cli_pretrained_model(default="hovernet_fast-pannuke")
@cli_pretrained_weights(default=None)
@cli_on_gpu()
@cli_batch_size()
@cli_masks(default=None)
@cli_yaml_config_path(default=None)
@cli_num_loader_workers()
@cli_verbose()
@cli_num_postproc_workers(default=0)
@cli_auto_generate_mask(default=False)
def nucleus_instance_segment(
    pretrained_model,
    pretrained_weights,
    img_input,
    file_types,
    masks,
    mode,
    output_path,
    batch_size,
    yaml_config_path,
    num_loader_workers,
    num_postproc_workers,
    auto_generate_mask,
    on_gpu,
    verbose,
):
    """Process an image/directory of input images with a patch classification CNN."""
    from tiatoolbox.models.engine.nucleus_instance_segmentor import (
        IOSegmentorConfig,
        NucleusInstanceSegmentor,
    )
    from tiatoolbox.utils.misc import save_as_json

    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
    )

    ioconfig = prepare_ioconfig_seg(
        IOSegmentorConfig, pretrained_weights, yaml_config_path
    )

    predictor = NucleusInstanceSegmentor(
        pretrained_model=pretrained_model,
        pretrained_weights=pretrained_weights,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_postproc_workers=num_postproc_workers,
        auto_generate_mask=auto_generate_mask,
        verbose=verbose,
    )

    output = predictor.predict(
        imgs=files_all,
        masks=masks_all,
        mode=mode,
        on_gpu=on_gpu,
        save_dir=output_path,
        ioconfig=ioconfig,
    )

    save_as_json(output, str(output_path.joinpath("results.json")))
