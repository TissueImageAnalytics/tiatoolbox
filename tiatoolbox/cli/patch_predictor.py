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

"""Command line interface for patch_predictor."""
import click

from tiatoolbox import utils
from tiatoolbox.models.engine.patch_predictor import PatchPredictor


@click.group()
def main():  # pragma: no cover
    """Define patch_predictor click group."""
    return 0


@main.command()
@click.option(
    "--pretrained-model",
    help="Predefined model used to process the data. the format is "
    "<model_name>_<dataset_trained_on>. For example, `resnet18-kather100K` is a "
    "resnet18 model trained on the kather dataset. For a detailed list of "
    "available pretrained models please see "
    "https://tia-toolbox.readthedocs.io/en/latest/usage.html"
    "#tiatoolbox.models.classification.patch_predictor.get_pretrained_model",
    default="resnet18-kather100k",
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
    "processed for image tiles. Supported file types are jpg, png and npy.",
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
    default="patch_prediction",
)
@click.option(
    "--batch-size",
    help="Number of images to feed into the model each time.",
    default=1,
)
@click.option(
    "--resolution",
    type=float,
    default=0.5,
    help="resolution to read the image at, default=0",
)
@click.option(
    "--units",
    default="mpp",
    type=click.Choice(["mpp", "power", "level", "baseline"], case_sensitive=False),
    help="resolution units, default=level",
)
@click.option(
    "--return-probabilities",
    type=bool,
    help="Whether to return raw model probabilities. default=False",
    default=False,
)
@click.option(
    "--return-labels",
    type=bool,
    help="Whether to return raw model output as labels. default=True",
    default=True,
)
@click.option(
    "--merge-predictions",
    type=bool,
    default=True,
    help="Whether to merge the predictions to form a 2-dimensional map. "
    "default=False",
)
@click.option(
    "--num-loader-workers",
    help="Number of workers to load the data. Please note that they will "
    "also perform preprocessing.",
    type=int,
    default=0,
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
    files_all, masks_all, output_path = utils.misc.prepare_model_cli(
        img_input=img_input,
        output_path=output_path,
        masks=masks,
        file_types=file_types,
        mode=mode,
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

    utils.misc.save_as_json(output, str(output_path.joinpath("results.json")))
