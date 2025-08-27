"""Command line interface for semantic segmentation."""

from __future__ import annotations

from tiatoolbox.cli.common import (
    cli_auto_get_mask,
    cli_batch_size,
    cli_device,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_memory_threshold,
    cli_model,
    cli_num_workers,
    cli_output_path,
    cli_output_type,
    cli_patch_mode,
    cli_return_labels,
    cli_return_probabilities,
    cli_verbose,
    cli_weights,
    cli_yaml_config_path,
    prepare_ioconfig,
    prepare_model_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Output directory where model segmentation will be saved.",
    default="semantic_segmentation",
)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@cli_model(default="fcn-tissue_mask")
@cli_weights()
@cli_device(default="cpu")
@cli_batch_size(default=1)
@cli_yaml_config_path()
@cli_masks(default=None)
@cli_num_workers(default=0)
@cli_output_type(
    default="AnnotationStore",
)
@cli_memory_threshold(default=80)
@cli_patch_mode(default=False)
@cli_return_probabilities(default=True)
@cli_return_labels(default=False)
@cli_auto_get_mask(default=True)
@cli_verbose(default=True)
def semantic_segmentor(
    model: str,
    weights: str,
    img_input: str,
    file_types: str,
    masks: str | None,
    output_path: str,
    batch_size: int,
    yaml_config_path: str,
    num_workers: int,
    device: str,
    output_type: str,
    memory_threshold: int,
    *,
    patch_mode: bool,
    return_probabilities: bool,
    return_labels: bool,
    auto_get_mask: bool,
    verbose: bool,
) -> None:
    """Process a set of input images with a semantic segmentation engine."""
    from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor  # noqa: PLC0415

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

    segmentor = SemanticSegmentor(
        model=model,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )

    _ = segmentor.run(
        images=files_all,
        masks=masks_all,
        patch_mode=patch_mode,
        ioconfig=ioconfig,
        device=device,
        save_dir=output_path,
        output_type=output_type,
        return_probabilities=return_probabilities,
        return_labels=return_labels,
        auto_get_mask=auto_get_mask,
        memory_threshold=memory_threshold,
    )
