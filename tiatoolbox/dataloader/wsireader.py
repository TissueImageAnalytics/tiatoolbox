"""WSIReader for WSI reading or extracting metadata information from WSIs"""
from tiatoolbox.utils import misc
from tiatoolbox.utils.transforms import background_composite

import pathlib
import numpy as np
import openslide
import math
import pandas as pd


class WSIReader:
    """WSI Reader class to read WSI images

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
        self.tile_objective_value = np.int(tile_objective_value)  # Tile magnification
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])
        self.objective_power = np.int(
            self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        )  # magnification at which slide is scanned, this is magnification at level 0
        self.level_count = self.openslide_obj.level_count
        self.level_dimensions = self.openslide_obj.level_dimensions
        self.level_downsamples = self.openslide_obj.level_downsamples

    def slide_info(self):
        """WSI meta data reader

        Args:
            self (WSIReader):

        Returns:
            dict: dictionary containing meta information

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_param = wsi_obj.slide_info()

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

    def read_region(self, start_w, start_h, end_w, end_h, level=0):
        """Read a region in whole slide image

        Args:
            start_w (int): starting point in x-direction (along width)
            start_h (int): starting point in y-direction (along height)
            end_w (int): end point in x-direction (along width)
            end_h (int): end point in y-direction (along height)
            level (int): pyramid level to read the image

        Returns:
            img_array : ndarray of size MxNx3
            M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> from matplotlib import pyplot as plt
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> level = 0
            >>> region = [13000, 17000, 15000, 19000]
            >>> im_region = wsi_obj.read_region(
            ...     region[0], region[1], region[2], region[3], level)
            >>> plt.imshow(im_region)

        """
        openslide_obj = self.openslide_obj
        im_region = openslide_obj.read_region(
            [start_w, start_h], level, [end_w - start_w, end_h - start_h]
        )
        im_region = background_composite(image=im_region)
        return im_region

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.5x

        Args:
            self (WSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()

        """
        openslide_obj = self.openslide_obj
        tile_objective_value = 20

        if self.objective_power == 0:
            self.objective_power = np.int(
                openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            )

        rescale = np.int(self.objective_power / tile_objective_value)
        slide_dimension = openslide_obj.level_dimensions[0]
        slide_dimension_20x = np.array(slide_dimension) / rescale
        thumb = openslide_obj.get_thumbnail(
            (int(slide_dimension_20x[0] / 16), int(slide_dimension_20x[1] / 16))
        )
        thumb = np.asarray(thumb)

        return thumb

    def save_tiles(self, tile_format=".jpg"):
        """Generate JPEG tiles from whole slide images

        Args:
            self (WSIReader):
            tile_format (str): file format to save image tiles, default=".jpg"

        Returns:
            saves tiles in the output directory output_dir

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi",
            ...     output_dir='./dev_test',
            ...     tile_objective_value=10,
            ...     tile_read_size_h=2000,
            ...     tile_read_size_w=2000)
            >>> wsi_obj.save_tiles()

        """
        openslide_obj = self.openslide_obj
        tile_objective_value = self.tile_objective_value
        tile_read_size = self.tile_read_size

        if self.objective_power == 0:
            self.objective_power = np.int(
                openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            )

        rescale = np.int(self.objective_power / tile_objective_value)
        tile_read_size = np.multiply(tile_read_size, rescale)
        slide_dimension = openslide_obj.level_dimensions[0]
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        tile_h = tile_read_size[0]
        tile_w = tile_read_size[1]

        iter_tot = 0
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True)
        data = []

        for h in range(int(math.ceil((slide_h - tile_h) / tile_h + 1))):
            for w in range(int(math.ceil((slide_w - tile_w) / tile_w + 1))):
                start_h = h * tile_h
                end_h = (h * tile_h) + tile_h
                start_w = w * tile_w
                end_w = (w * tile_w) + tile_w
                if end_h > slide_h:
                    end_h = slide_h

                if end_w > slide_w:
                    end_w = slide_w

                # Read image region
                im = self.read_region(start_w, start_h, end_w, end_h)
                format_str = (
                    "Tile%d:  start_w:%d, end_w:%d, "
                    "start_h:%d, end_h:%d, "
                    "width:%d, height:%d"
                )

                print(
                    format_str
                    % (
                        iter_tot,
                        start_w,
                        end_w,
                        start_h,
                        end_h,
                        end_w - start_w,
                        end_h - start_h,
                    ),
                    flush=True,
                )

                # Rescale to the correct objective value
                if rescale != 1:
                    im = misc.imresize(im, rescale)

                img_save_name = (
                    "_".join(
                        [
                            "Tile",
                            str(tile_objective_value),
                            str(int(start_w / rescale)),
                            str(int(start_h / rescale)),
                        ]
                    )
                    + tile_format
                )

                misc.imwrite(image_path=output_dir.joinpath(img_save_name), img=im)

                data.append(
                    [
                        iter_tot,
                        img_save_name,
                        start_w,
                        end_w,
                        start_h,
                        end_h,
                        im.shape[0],
                        im.shape[1],
                    ]
                )
                iter_tot += 1

        # Save information on each slide to relate to the whole slide image
        df = pd.DataFrame(
            data,
            columns=[
                "iter",
                "Tile_Name",
                "start_w",
                "end_w",
                "start_h",
                "end_h",
                "size_w",
                "size_h",
            ],
        )
        df.to_csv(output_dir.joinpath("Output.csv"), index=False)

        # Save slide thumbnail
        slide_thumb = self.slide_thumbnail()
        misc.imwrite(
            output_dir.joinpath("slide_thumbnail" + tile_format), img=slide_thumb
        )
