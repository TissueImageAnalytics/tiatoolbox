"""Command line interface for nucleus detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiatoolbox.cli.common import (
    cli_auto_get_mask,
    cli_batch_size,
    cli_class_dict,
    cli_device,
    cli_file_type,
    cli_img_input,
    cli_input_resolutions,
    cli_masks,
    cli_memory_threshold,
    cli_min_distance,
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
    cli_postproc_tile_shape,
    cli_return_probabilities,
    cli_scale_factor,
    cli_stride_shape,
    cli_threshold_abs,
    cli_threshold_rel,
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
    usage_help="Output directory where model prediction will be saved.",
    default="nucleus_detection",
)
@cli_output_file(default=None)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@cli_input_resolutions(default=None)
@cli_output_resolutions(default=None)
@cli_class_dict(default=None)
@cli_model(default="mapde-conic")
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
@cli_min_distance(default=None)
@cli_threshold_abs(default=None)
@cli_threshold_rel(default=None)
@cli_postproc_tile_shape(default=None)
@cli_stride_shape(default=None)
@cli_scale_factor(default=None)
@cli_patch_mode(default=False)
@cli_return_probabilities(default=True)
@cli_auto_get_mask(default=True)
@cli_overwrite(default=False)
@cli_verbose(default=True)
def nucleus_detector(
    model: str,
    weights: str,
    img_input: str,
    file_types: str,
    class_dict: list[tuple[int, str]],
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
    min_distance: int | None,
    threshold_abs: float | None,
    threshold_rel: float | None,
    postproc_tile_shape: IntPair | None,
    *,
    patch_mode: bool,
    return_probabilities: bool,
    auto_get_mask: bool,
    verbose: bool,
    overwrite: bool,
) -> None:
    """Process a set of input images with a nucleus detection engine."""
    from tiatoolbox.models import IOSegmentorConfig, NucleusDetector  # noqa: PLC0415

    class_dict = dict(class_dict) if class_dict else None

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

    detector = NucleusDetector(
        model=model,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )

    _ = detector.run(
        images=files_all,
        masks=masks_all,
        class_dict=class_dict,
        patch_mode=patch_mode,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_output_shape,
        input_resolutions=input_resolutions,
        output_resolutions=output_resolutions,
        batch_size=batch_size,
        ioconfig=ioconfig,
        device=device,
        save_dir=output_path,
        output_type=output_type,
        return_probabilities=return_probabilities,
        auto_get_mask=auto_get_mask,
        memory_threshold=memory_threshold,
        num_workers=num_workers,
        output_file=output_file,
        scale_factor=scale_factor,
        stride_shape=stride_shape,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        postproc_tile_shape=postproc_tile_shape,
        overwrite=overwrite,
        verbose=verbose,
    )
