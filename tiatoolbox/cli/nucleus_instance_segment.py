# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Command line interface for nucleus instance segmentation."""
import click
import yaml

from tiatoolbox import utils
from tiatoolbox.models.engine.nucleus_instance_segmentor import (
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
)


@click.group()
def main():  # pragma: no cover
    """Define semantic_segment click group."""
    return 0


@main.command()
@click.option(
    "--pretrained-model",
    help="Name of the existing models support by tiatoolbox"
    "for processing the data. Refer to [URL] for details."
    "By default, the corresponding pretrained weights will also be"
    "downloaded. However, you can override with your own set of weights"
    "via the `pretrained_weights` argument. Argument is case insensitive. "
    "default = 'hovernet_fast-pannuke'",
    default="hovernet_fast-pannuke",
)
@click.option(
    "--pretrained-weights",
    help="Path to the model weight file. If not supplied, the default "
    "pretrained weight will be used.",
    default=None,
)
@click.option(
    "--img-input",
    help="Path to the input directory containing images to process or an "
    "individual file.",
)
@click.option(
    "--file-types",
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
    "--output-path",
    help="Output directory where model predictions will be saved.",
    default="semantic_segmentation",
)
@click.option(
    "--batch-size",
    help="Number of images to feed into the model each time.",
    default=1,
)
@click.option(
    "--yaml-config-path",
    help="Path to ioconfig file. Sample yaml file can be viewed in "
    "tiatoolbox.data.pretrained_model.yaml. "
    "if pretrained_model is used the ioconfig is automatically set."
    "default=None",
    default="None",
)
@click.option(
    "--num-loader-workers",
    help="Number of workers to load the data. Please note that they will "
    "also perform preprocessing.",
    type=int,
    default=0,
)
@click.option(
    "--num-postproc-workers",
    help="Number of workers to post-process the network output.",
    type=int,
    default=0,
)
@click.option(
    "--auto-generate-mask",
    help="If set to True, it automatically generates tile/WSI tissue mask. "
    "default=False",
    type=bool,
    default=False,
)
@click.option(
    "--on-gpu",
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

    utils.misc.save_as_json(output, str(output_path.joinpath("results.json")))
