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

"""Command line interface for semantic segmentation."""
import click
import yaml

from tiatoolbox import utils
from tiatoolbox.cli.common import (
    cli_batch_size,
    cli_file_type,
    cli_img_input,
    cli_masks,
    cli_mode,
    cli_num_loader_workers,
    cli_on_gpu,
    cli_output_path,
    cli_pretrained_model,
    cli_pretrained_weights,
    cli_verbose,
    cli_yaml_config_path,
    prepare_model_cli,
    tiatoolbox_cli,
)
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Output directory where model predictions will be saved.",
    default="semantic_segmentation",
)
@cli_file_type(
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff, *.svs, *.ndpi, *.jp2, *.mrxs"
)
@cli_mode(
    usage_help="Type of input file to process.",
    default="wsi",
    input_type=click.Choice(["patch", "wsi", "tile"], case_sensitive=False),
)
@cli_pretrained_model(default="fcn-tissue_mask")
@cli_pretrained_weights(default=None)
@cli_on_gpu()
@cli_batch_size()
@cli_masks(default=None)
@cli_yaml_config_path()
@cli_num_loader_workers()
@cli_verbose()
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
    files_all, masks_all, output_path = prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
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
