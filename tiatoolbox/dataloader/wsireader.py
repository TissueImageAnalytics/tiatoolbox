"""
This file contains WSIReader class for WSI reading or extracting metadata information from WSIs
"""
import os
import numpy as np
import yaml
from PIL import Image

# For Windows Platforms to add path to openslide binaries
if os.name == "nt":
    os.environ["PATH"] = (
        "C:\\tools\\openslide\\openslide-win64-20171122\\bin" + ";" + os.environ["PATH"]
    )

import openslide


class WSIReader:
    def __init__(
        self,
        input_dir=os.getcwd(),
        file_name=None,
        output_dir=os.path.join(os.getcwd(), "output"),
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):
        """
        WSI Reader class to read WSI images
        Args:
            input_dir: input path to WSI directory
            file_name: file name of the WSI
            output_dir: output directory to save the output, default=os.getcwd()/output
            tile_objective_value: objective value at which tile is generated, default=20
            tile_read_size_w: tile width, default=5000
            tile_read_size_h: tile height, default=5000
        """

        self.input_dir = input_dir
        self.file_name = os.path.basename(file_name)
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, self.file_name)

        self.openslide_obj = openslide.OpenSlide(
            filename=os.path.join(self.input_dir, self.file_name)
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

    def slide_info(self, save_mode=True, output_dir=None, output_name=None):
        """
        WSI meta data reader
        Args:
            save_mode: save meta information as yaml file
            output_dir: output directory to save the meta information
            output_name: output file name

        Returns:
            displays or saves WSI meta information

        """
        input_dir = self.input_dir
        if output_dir is None:
            self.output_dir = output_dir
        if output_name is None:
            output_name = "param.yaml"
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

        param = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "objective_power": objective_power,
            "slide_dimension": slide_dimension,
            "rescale": rescale,
            "tile_objective_value": tile_objective_value,
            "filename": filename,
            "tile_read_size": tile_read_size.tolist(),
            "level_count": level_count,
            "level_dimensions": level_dimensions,
            "level_downsamples": level_downsamples,
        }
        if save_mode:
            with open(os.path.join(output_dir, output_name), "w") as yaml_file:
                yaml.dump(param, yaml_file)
        else:
            return param
