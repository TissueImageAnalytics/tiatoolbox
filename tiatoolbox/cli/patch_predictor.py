"""Command line interface for patch_predictor."""
import click

from tiatoolbox.cli.common import (
    cli_batch_size,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_merge_predictions,
    cli_mode,
    cli_num_loader_workers,
    cli_on_gpu,
    cli_output_path,
    cli_pretrained_model,
    cli_pretrained_weights,
    cli_resolution,
    cli_return_labels,
    cli_return_probabilities,
    cli_units,
    cli_verbose,
    prepare_model_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Output directory where model predictions will be saved.",
    default="patch_prediction",
)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs"
)
@cli_mode(
    usage_help="Type of input file to process.",
    default="wsi",
    input_type=click.Choice(["patch", "wsi", "tile"], case_sensitive=False),
)
@cli_pretrained_model(default="resnet18-kather100k")
@cli_pretrained_weights()
@cli_return_probabilities(default=False)
@cli_merge_predictions(default=True)
@cli_return_labels(default=True)
@cli_on_gpu(default=False)
@cli_batch_size(default=1)
@cli_resolution(default=0.5)
@cli_units(default="mpp")
@cli_masks(default=None)
@cli_num_loader_workers(default=0)
@cli_verbose()
def patch_predictor(
    pretrained_model,
    pretrained_weights,
    img_input,
    file_types,
    masks,
    mode,
    output_path,
    batch_size,
    resolution,
    units,
    return_probabilities,
    return_labels,
    merge_predictions,
    num_loader_workers,
    on_gpu,
    verbose,
):
    """Process an image/directory of input images with a patch classification CNN."""
    from tiatoolbox.models.engine.patch_predictor import PatchPredictor
    from tiatoolbox.utils.misc import save_as_json

    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
    )

    predictor = PatchPredictor(
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
        return_probabilities=return_probabilities,
        merge_predictions=merge_predictions,
        labels=None,
        return_labels=return_labels,
        resolution=resolution,
        units=units,
        on_gpu=on_gpu,
        save_dir=output_path,
        save_output=True,
    )

    save_as_json(output, str(output_path.joinpath("results.json")))
