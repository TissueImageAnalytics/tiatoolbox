"""Command line interface for patch_predictor."""

from __future__ import annotations

from tiatoolbox.cli.common import (
    cli_batch_size,
    cli_device,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_model,
    cli_num_loader_workers,
    cli_output_path,
    cli_patch_mode,
    cli_resolution,
    cli_units,
    cli_verbose,
    cli_weights,
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
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@cli_patch_mode(default=False)
@cli_model(default="resnet18-kather100k")
@cli_weights()
@cli_device(default="cpu")
@cli_batch_size(default=1)
@cli_resolution(default=0.5)
@cli_units(default="mpp")
@cli_masks(default=None)
@cli_num_loader_workers(default=0)
@cli_verbose(default=True)
def patch_predictor(
    model: str,
    weights: str,
    img_input: str,
    file_types: str,
    masks: str | None,
    output_path: str,
    batch_size: int,
    resolution: float,
    units: str,
    num_loader_workers: int,
    device: str,
    *,
    patch_mode: bool,
    verbose: bool,
) -> None:
    """Process an image/directory of input images with a patch classification CNN."""
    from tiatoolbox.models.engine.patch_predictor_new import PatchPredictor

    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
    )

    predictor = PatchPredictor(
        model=model,
        weights=weights,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        verbose=verbose,
    )

    _ = predictor.run(
        images=files_all,
        masks=masks_all,
        patch_mode=patch_mode,
        resolution=resolution,
        units=units,
        device=device,
        save_dir=output_path,
        save_output=True,
    )
