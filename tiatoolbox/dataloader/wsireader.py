"""WSIReader for WSI reading or extracting metadata information from WSIs"""

import pathlib
import numpy as np

# from PIL import Image

import openslide


class WSIReader:
    """
    WSI Reader class to read WSI images

    Attributes:
        input_dir (pathlib.Path): input path to WSI directory
        file_name (str): file name of the WSI
        output_dir (pathlib.Path): output directory to save the output
        openslide_obj (:obj:`openslide.OpenSlide`)
        tile_objective_value (int): objective value at which tile is generated
        tile_read_size (int): [tile width, tile height]
        objective_power (int): objective value at which whole slide image is scanned
        level_count (int): The number of pyramid levels in the slide
        level_dimensions (int): A list of `(width, height)` tuples, one for each level
            of the slide
        level_downsamples (int): A list of down sample factors for each level
            of the slide

    """

    def __init__(
        self,
        input_dir=".",
        file_name=None,
        output_dir="./output",
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):
        """
        Args:
            input_dir (str, pathlib.Path): input path to WSI directory
            file_name (str): file name of the WSI
            output_dir (str, pathlib.Path): output directory to save the output,
                default=./output
            tile_objective_value (int): objective value at which tile is generated,
                default=20
            tile_read_size_w (int): tile width, default=5000
            tile_read_size_h (int): tile height, default=5000

        """

        self.input_dir = pathlib.Path(input_dir)
        self.file_name = pathlib.Path(file_name).name
        if output_dir is not None:
            self.output_dir = pathlib.Path(output_dir, self.file_name)

        self.openslide_obj = openslide.OpenSlide(
            filename=str(pathlib.Path(self.input_dir, self.file_name))
        )
        self.tile_objective_value = np.int(tile_objective_value)
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])
        self.objective_power = np.int(
            self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        )
        self.level_count = self.openslide_obj.level_count
        self.level_dimensions = self.openslide_obj.level_dimensions
        self.level_downsamples = self.openslide_obj.level_downsamples

    def __exit__(self):
        self.openslide_obj.close()

    def slide_info(self):
        """
        WSI meta data reader
        Args:

        Returns:
            param (dict): dictionary containing meta information

        """
        input_dir = self.input_dir
        if self.objective_power == 0:
            self.objective_power = np.int(
                self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            )
        objective_power = self.objective_power
        slide_dimension = self.openslide_obj.level_dimensions[0]
        tile_objective_value = self.tile_objective_value
        rescale = np.int(objective_power / tile_objective_value)
        filename = self.file_name
        tile_read_size = self.tile_read_size
        level_count = self.level_count
        level_dimensions = self.level_dimensions
        level_downsamples = self.level_downsamples
        file_name = self.file_name

        param = {
            "input_dir": input_dir,
            "objective_power": objective_power,
            "slide_dimension": slide_dimension,
            "rescale": rescale,
            "tile_objective_value": tile_objective_value,
            "filename": filename,
            "tile_read_size": tile_read_size.tolist(),
            "level_count": level_count,
            "level_dimensions": level_dimensions,
            "level_downsamples": level_downsamples,
            "file_name": file_name,
        }

        return param
