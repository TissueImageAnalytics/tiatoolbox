"""Command line interface for semantic segmentation."""
import click
import yaml

from tiatoolbox import utils
from tiatoolbox.models.controller.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)


@click.group()
def main():  # pragma: no cover
    """Define slide_info click group."""
    return 0


@main.command()
@click.option(
    "--pretrained_model",
    help="Predefined model used to process the data. the format is "
    "<model_name>_<dataset_trained_on>. For example, `resnet18-kather100K` is a "
    "resnet18 model trained on the kather dataset. For a detailed list of "
    "available pretrained models please see "
    "https://tia-toolbox.readthedocs.io/en/latest/usage.html"
    "#tiatoolbox.models.classification.patch_predictor.get_pretrained_model",
    default="fcn-tissue_mask",
)
@click.option(
    "--pretrained_weights",
    help="Path to the model weight file. If not supplied, the default "
    "pretrained weight will be used.",
    default=None,
)
@click.option(
    "--img_input",
    help="Path to the input directory containing images to process or an "
    "individual file.",
)
@click.option(
    "--file_types",
    help="File types to capture from directory. "
    "default='*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'",
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs",
)
@click.option(
    "--masks",
    help="Path to the input directory containing masks to process corresponding to "
    "image tiles and whole-slide images. Patches are only processed if they are "
    "within a masked area. If masks are not provided, then a tissue mask will be "
    "automatically generated for whole-slide images or the entire image is "
    "processed for image tiles. Supported file types are jpg, png.",
    default=None,
)
@click.option(
    "--mode",
    help="Type of input to process. Choose from either patch, tile or wsi. Default=wsi",
    default="wsi",
)
@click.option(
    "--output_path",
    help="Output directory where model predictions will be saved.",
    default="semantic_segmentation",
)
@click.option(
    "--batch_size",
    help="Number of images to feed into the model each time.",
    default=1,
)
@click.option(
    "--yaml_config_path",
    help="Path to ioconfig file. Sample yaml file can be viewed in "
    "tiatoolbox.data.pretrained_model.yaml. "
    "if pretrained_model is used the ioconfig is automatically set."
    "default=None",
    default="None",
)
@click.option(
    "--num_loader_workers",
    help="Number of workers to load the data. Please note that they will "
    "also perform preprocessing.",
    type=int,
    default=0,
)
@click.option(
    "--on_gpu",
    type=bool,
    default=False,
    help="Run the model on GPU, default=False",
)
@click.option(
    "--verbose",
    type=bool,
    default=True,
    help="Print output, default=True",
)
def semantic_segment(
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
    on_gpu,
    verbose,
):
    """Process an image/directory of input images with a patch classification CNN."""
    files_all, masks_all, output_path = utils.misc.prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
        mode=mode,
    )

    ioconfig = None

    if pretrained_weights is not None:
        with open(yaml_config_path) as registry_handle:
            ioconfig = yaml.safe_load(registry_handle)

        ioconfig = IOSegmentorConfig(**ioconfig)

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
        on_gpu=on_gpu,
        save_dir=output_path,
        ioconfig=ioconfig,
    )

    utils.misc.save_as_json(output, str(output_path.joinpath("results.json")))
