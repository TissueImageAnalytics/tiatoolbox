"""Defines common code required for cli."""
import os
import pathlib

import click


def add_default_to_usage_help(
    usage_help: str, default: str or int or float or bool
) -> str:
    """Adds default value to usage help string.

    Args:
        usage_help (str):
            usage help for click option.
        default (str or int or float):
            default value as string for click option.

    Returns:
        str:
            New usage_help value.

    """
    if default is not None:
        return f"{usage_help} default={default}"

    return usage_help


def cli_img_input(
    usage_help: str = "Path to WSI or directory containing WSIs.",
    multiple: bool = False,
) -> callable:
    """Enables --img-input option for cli."""
    if multiple:
        usage_help = usage_help + " Multiple instances may be provided."
    return click.option("--img-input", help=usage_help, type=str, multiple=multiple)


def cli_name(
    usage_help: str = "User defined name to be used as an identifier.",
    multiple: bool = False,
) -> callable:
    """enables --name option for cli"""
    if multiple:
        usage_help = usage_help + " Multiple instances may be provided."
    return click.option("--name", help=usage_help, type=str, multiple=multiple)


def cli_output_path(
    usage_help: str = "Path to output directory to save the output.",
    default: str = None,
) -> callable:
    """Enables --output-path option for cli."""
    return click.option(
        "--output-path",
        help=add_default_to_usage_help(usage_help, default),
        type=str,
        default=default,
    )


def cli_file_type(
    usage_help: str = "File types to capture from directory.",
    default: str = "*.ndpi, *.svs, *.mrxs, *.jp2",
) -> callable:
    """Enables --file-types option for cli."""
    return click.option(
        "--file-types",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
        type=str,
    )


def cli_mode(
    usage_help: str = "Selected mode to show or save the required information.",
    default: str = "save",
    input_type: click.Choice = None,
) -> callable:
    """Enables --mode option for cli."""
    if input_type is None:
        input_type = click.Choice(["show", "save"], case_sensitive=False)
    return click.option(
        "--mode",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
        type=input_type,
    )


def cli_region(
    usage_help: str = "Image region in the whole slide image to read from. "
    "default=0 0 2000 2000",
) -> callable:
    """Enables --region option for cli."""
    return click.option(
        "--region",
        type=int,
        nargs=4,
        help=usage_help,
    )


def cli_units(
    usage_help: str = "Image resolution units to read the image.",
    default: str = "level",
    input_type: click.Choice = None,
) -> callable:
    """Enables --units option for cli."""
    if input_type is None:
        input_type = click.Choice(
            ["mpp", "power", "level", "baseline"], case_sensitive=False
        )
    return click.option(
        "--units",
        default=default,
        type=input_type,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_resolution(
    usage_help: str = "Image resolution to read the image.", default: float = 0
) -> callable:
    """Enables --resolution option for cli."""
    return click.option(
        "--resolution",
        type=float,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_tile_objective(
    usage_help: str = "Objective value for the saved tiles.", default: int = 20
) -> callable:
    """Enables --tile-objective-value option for cli."""
    return click.option(
        "--tile-objective-value",
        type=int,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_tile_read_size(
    usage_help: str = "Width and Height of saved tiles. default=5000 5000",
) -> callable:
    """Enables --tile-read-size option for cli."""
    return click.option(
        "--tile-read-size",
        type=int,
        nargs=2,
        default=[5000, 5000],
        help=usage_help,
    )


def cli_method(
    usage_help: str = "Select method of for tissue masking.",
    default: str = "Otsu",
    input_type: click.Choice = None,
) -> callable:
    """Enables --method option for cli."""
    if input_type is None:
        input_type = click.Choice(["Otsu", "Morphological"], case_sensitive=True)
    return click.option(
        "--method",
        type=input_type,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_pretrained_model(
    usage_help: str = "Name of the predefined model used to process the data. "
    "The format is <model_name>_<dataset_trained_on>. For example, "
    "`resnet18-kather100K` is a resnet18 model trained on the Kather dataset. "
    "Please see "
    "https://tia-toolbox.readthedocs.io/en/latest/usage.html#deep-learning-models "
    "for a detailed list of available pretrained models."
    "By default, the corresponding pretrained weights will also be"
    "downloaded. However, you can override with your own set of weights"
    "via the `pretrained_weights` argument. Argument is case insensitive.",
    default: str = "resnet18-kather100k",
) -> callable:
    """Enables --pretrained-model option for cli."""
    return click.option(
        "--pretrained-model",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_pretrained_weights(
    usage_help: str = "Path to the model weight file. If not supplied, the default "
    "pretrained weight will be used.",
    default: str = None,
) -> callable:
    """Enables --pretrained-weights option for cli."""
    return click.option(
        "--pretrained-weights",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_return_probabilities(
    usage_help: str = "Whether to return raw model probabilities.",
    default: bool = False,
) -> callable:
    """Enables --return-probabilities option for cli."""
    return click.option(
        "--return-probabilities",
        type=bool,
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_merge_predictions(
    usage_help: str = "Whether to merge the predictions to form a 2-dimensional map.",
    default: bool = True,
) -> callable:
    """Enables --merge-predictions option for cli."""
    return click.option(
        "--merge-predictions",
        type=bool,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_return_labels(
    usage_help: str = "Whether to return raw model output as labels.",
    default: bool = True,
) -> callable:
    """Enables --return-labels option for cli."""
    return click.option(
        "--return-labels",
        type=bool,
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_batch_size(
    usage_help: str = "Number of image patches to feed into the model each time.",
    default: int = 1,
) -> callable:
    """Enables --batch-size option for cli."""
    return click.option(
        "--batch-size",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_masks(
    usage_help: str = "Path to the input directory containing masks to process "
    "corresponding to image tiles and whole-slide images. "
    "Patches are only processed if they are within a masked area. "
    "If masks are not provided, then a tissue mask will be "
    "automatically generated for whole-slide images or the entire image is "
    "processed for image tiles. Supported file types are jpg, png and npy.",
    default: str = None,
) -> callable:
    """Enables --masks option for cli."""
    return click.option(
        "--masks",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_auto_generate_mask(
    usage_help: str = "Automatically generate tile/WSI tissue mask.",
    default: bool = False,
) -> callable:
    """Enables --auto-generate-mask option for cli."""
    return click.option(
        "--auto-generate-mask",
        help=add_default_to_usage_help(usage_help, default),
        type=bool,
        default=default,
    )


def cli_yaml_config_path(
    usage_help: str = "Path to ioconfig file. Sample yaml file can be viewed in "
    "tiatoolbox.data.pretrained_model.yaml. "
    "if pretrained_model is used the ioconfig is automatically set.",
    default: str = None,
) -> callable:
    """Enables --yaml-config-path option for cli."""
    return click.option(
        "--yaml-config-path",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
    )


def cli_on_gpu(
    usage_help: str = "Run the model on GPU.", default: bool = False
) -> callable:
    """Enables --on-gpu option for cli."""
    return click.option(
        "--on-gpu",
        type=bool,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_num_loader_workers(
    usage_help: str = "Number of workers to load the data. Please note that they will "
    "also perform preprocessing.",
    default: int = 0,
) -> callable:
    """Enables --num-loader-workers option for cli."""
    return click.option(
        "--num-loader-workers",
        help=add_default_to_usage_help(usage_help, default),
        type=int,
        default=default,
    )


def cli_num_postproc_workers(
    usage_help: str = "Number of workers to post-process the network output.",
    default: int = 0,
) -> callable:
    """Enables --num-postproc-workers option for cli."""
    return click.option(
        "--num-postproc-workers",
        help=add_default_to_usage_help(usage_help, default),
        type=int,
        default=default,
    )


def cli_verbose(
    usage_help: str = "Prints the console output.", default: bool = True
) -> callable:
    """Enables --verbose option for cli."""
    return click.option(
        "--verbose",
        type=bool,
        help=add_default_to_usage_help(usage_help, str(default)),
        default=default,
    )


class TIAToolboxCLI(click.Group):
    """Defines TIAToolbox Commandline Interface Click group."""

    def __init__(self, *args, **kwargs):
        super(TIAToolboxCLI, self).__init__(*args, **kwargs)
        self.help = "Computational pathology toolbox by TIA Centre."
        self.add_help_option = {"help_option_names": ["-h", "--help"]}


def no_input_message(
    input_file: str or pathlib.Path = None, message: str = "No image input provided.\n"
) -> None:
    """This function is called if no input is provided.

    Args:
        input_file (str or pathlib.Path): Path to input file.
        message (str): Error message to display.

    """
    if input_file is None:
        ctx = click.get_current_context()
        ctx.fail(message=message)


def prepare_file_dir_cli(
    img_input: str or pathlib.Path,
    output_path: str or pathlib.Path,
    file_types: str,
    mode: str,
    sub_dirname: str,
) -> [list, pathlib.Path]:
    """Prepares CLI for running code on multiple files or a directory.

    Checks for existing directories to run tests.
    Converts file path to list of file paths or
    creates list of file paths if input is a directory.

    Args:
        img_input (str or pathlib.Path): file path to images.
        output_path (str or pathlib.Path): output directory path.
        file_types (str): file types to process using cli.
        mode (str): wsi or tile mode.
        sub_dirname (str): name of subdirectory to save output.

    Returns:
        list: list of file paths to process.
        pathlib.Path: updated output path.

    """
    from tiatoolbox.utils.misc import grab_files_from_dir, string_to_tuple

    no_input_message(input_file=img_input)
    file_types = string_to_tuple(in_str=file_types)

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = grab_files_from_dir(input_path=img_input, file_types=file_types)

    if output_path is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_path = input_dir / sub_dirname

    if mode == "save":
        output_path.mkdir(parents=True, exist_ok=True)

    return [files_all, output_path]


def prepare_model_cli(
    img_input: str or pathlib.Path,
    output_path: str or pathlib.Path,
    masks: str or pathlib.Path,
    file_types: str,
) -> [list, list, pathlib.Path]:
    """Prepares cli for running models.

    Checks for existing directories to run tests.
    Converts file path to list of file paths or
    creates list of file paths if input is a directory.

    Args:
        img_input (str or pathlib.Path): file path to images.
        output_path (str or pathlib.Path): output directory path.
        masks (str or pathlib.Path): file path to masks.
        file_types (str): file types to process using cli.

    Returns:
        list: list of file paths to process.
        list: list of masks corresponding to input files.
        pathlib.Path: output path

    """
    from tiatoolbox.utils.misc import grab_files_from_dir, string_to_tuple

    no_input_message(input_file=img_input)
    output_path = pathlib.Path(output_path)
    file_types = string_to_tuple(in_str=file_types)

    if output_path.exists():
        raise FileExistsError("Path already exists.")

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if masks is None:
        masks_all = None
    else:
        masks_all = [
            masks,
        ]

    if os.path.isdir(img_input):
        files_all = grab_files_from_dir(input_path=img_input, file_types=file_types)

    if os.path.isdir(str(masks)):
        masks_all = grab_files_from_dir(input_path=masks, file_types=("*.jpg", "*.png"))

    return [files_all, masks_all, output_path]


tiatoolbox_cli = TIAToolboxCLI()


def prepare_ioconfig_seg(segment_config_class, pretrained_weights, yaml_config_path):
    """Prepare ioconfig for segmentation."""
    import yaml

    if pretrained_weights is not None:
        with open(yaml_config_path) as registry_handle:
            ioconfig = yaml.safe_load(registry_handle)

        return segment_config_class(**ioconfig)

    return None
