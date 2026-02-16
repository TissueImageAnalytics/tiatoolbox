"""Command line interface for multitask segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiatoolbox.cli.common import (
    cli_auto_get_mask,
    cli_batch_size,
    cli_device,
    cli_file_type,
    cli_img_input,
    cli_input_resolutions,
    cli_masks,
    cli_memory_threshold,
    cli_model,
    cli_num_workers,
    cli_output_file,
    cli_output_path,
    cli_output_resolutions,
    cli_output_type,
    cli_overwrite,
    cli_patch_input_shape,
    cli_patch_mode,
    cli_patch_output_shape,
    cli_return_predictions,
    cli_return_probabilities,
    cli_scale_factor,
    cli_stride_shape,
    cli_verbose,
    cli_weights,
    cli_yaml_config_path,
    prepare_ioconfig,
    prepare_model_cli,
    tiatoolbox_cli,
)

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntPair


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Output directory where model output will be saved.",
    default="multitask_segmentor",
)
@cli_output_file(default=None)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@cli_input_resolutions(default=None)
@cli_output_resolutions(default=None)
@cli_model(default="hovernetplus-oed")
@cli_weights()
@cli_device(default="cpu")
@cli_batch_size(default=64)
@cli_yaml_config_path()
@cli_masks(default=None)
@cli_num_workers(default=0)
@cli_output_type(
    default="AnnotationStore",
)
@cli_memory_threshold(default=80)
@cli_patch_input_shape(default=None)
@cli_patch_output_shape(default=None)
@cli_stride_shape(default=None)
@cli_scale_factor(default=None)
@cli_patch_mode(default=False)
@cli_return_predictions(default=None)
@cli_return_probabilities(default=True)
@cli_auto_get_mask(default=True)
@cli_overwrite(default=False)
@cli_verbose(default=True)
def multitask_segmentor(
    model: str,
    weights: str,
    img_input: str,
    file_types: str,
    input_resolutions: list[dict],
    output_resolutions: list[dict],
    masks: str | None,
    output_path: str,
    patch_input_shape: IntPair | None,
    patch_output_shape: tuple[int, int] | None,
    stride_shape: IntPair | None,
    scale_factor: tuple[float, float] | None,
    batch_size: int,
    yaml_config_path: str,
    num_workers: int,
    device: str,
    output_type: str,
    memory_threshold: int,
    output_file: str | None,
    *,
    patch_mode: bool,
    return_predictions: tuple[bool, ...] | None,
    return_probabilities: bool,
    auto_get_mask: bool,
    verbose: bool,
    overwrite: bool,
) -> None:
    """Process a set of input images with a multitask segmentation engine."""
    from tiatoolbox.models import IOSegmentorConfig, MultiTaskSegmentor  # noqa: PLC0415

    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
    )

    ioconfig = prepare_ioconfig(
        IOSegmentorConfig,
        pretrained_weights=weights,
        yaml_config_path=yaml_config_path,
    )

    mtsegmentor = MultiTaskSegmentor(
        model=model,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )

    _ = mtsegmentor.run(
        images=files_all,
        masks=masks_all,
        patch_mode=patch_mode,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_output_shape,
        input_resolutions=input_resolutions,
        output_resolutions=output_resolutions,
        ioconfig=ioconfig,
        device=device,
        save_dir=output_path,
        output_type=output_type,
        return_predictions=return_predictions,
        return_probabilities=return_probabilities,
        auto_get_mask=auto_get_mask,
        memory_threshold=memory_threshold,
        output_file=output_file,
        scale_factor=scale_factor,
        stride_shape=stride_shape,
        overwrite=overwrite,
    )
