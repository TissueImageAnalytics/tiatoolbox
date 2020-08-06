"""WSIReader for WSI reading or extracting metadata information from WSIs"""
from tiatoolbox.utils import misc, transforms
from tiatoolbox.dataloader.wsimeta import WSIMeta

import pathlib
import numpy as np
import openslide
import glymur
import math
import pandas as pd
import re


class WSIReader:
    """WSI Reader class to read WSI images

    Attributes:
        input_dir (pathlib.Path): input path to WSI directory
        file_name (str): file name of the WSI
        output_dir (pathlib.Path): output directory to save the output
        tile_objective_value (int): objective value at which tile is generated
        tile_read_size (int): [tile width, tile height]
        slide_info (dict): Whole slide image slide information

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

        self.tile_objective_value = np.int(tile_objective_value)  # Tile magnification
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])
        self.slide_info = WSIMeta(input_dir=self.input_dir, file_name=self.file_name)

    def __slide_info(self):
        """WSI meta data reader

        Args:
            self (WSIReader):

        Returns:
            dict: dictionary containing meta information

        """
        param = WSIMeta(input_dir=self.input_dir, file_name=self.file_name)
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

        """
        raise NotImplementedError

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (WSIReader):

        Returns:
            ndarray : image array

        """
        raise NotImplementedError

    def save_tiles(self, tile_format=".jpg", verbose=True):
        """Generate image tiles from whole slide images.

        Args:
            self (WSIReader):
            tile_format (str): file format to save image tiles, default=".jpg"
            verbose (bool): Print output, default=True

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
        tile_objective_value = self.tile_objective_value
        tile_read_size = self.tile_read_size

        rescale = self.slide_info.objective_power / tile_objective_value
        if rescale.is_integer():
            try:
                level = np.log2(rescale)
                if level.is_integer():
                    level = np.int(level)
                    slide_dimension = self.slide_info.level_dimensions[level]
                    rescale = 1
                else:
                    raise ValueError
            # Raise index error if desired pyramid level not embedded
            # in level_dimensions
            except (IndexError, ValueError):
                level = 0
                slide_dimension = self.slide_info.level_dimensions[level]
                rescale = np.int(rescale)
        else:
            raise ValueError("rescaling factor must be an integer.")

        tile_read_size = np.multiply(tile_read_size, rescale)
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
                im = self.read_region(start_w, start_h, end_w, end_h, level)

                if verbose:
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
                    im = transforms.imresize(im, rescale)

                img_save_name = (
                    "_".join(
                        [
                            "Tile",
                            str(tile_objective_value),
                            str(int(start_h / rescale)),
                            str(int(start_w / rescale)),
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


class OpenSlideWSIReader(WSIReader):
    """Class for reading OpenSlide supported whole-slide images.

    Attributes:
        openslide_obj (:obj:`openslide.OpenSlide`)

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
        super().__init__(
            input_dir=input_dir,
            file_name=file_name,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        self.openslide_obj = openslide.OpenSlide(
            filename=str(pathlib.Path(self.input_dir, self.file_name))
        )
        self.slide_info = self.__slide_info()

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
            >>> wsi_obj = wsireader.OpenSlideWSIReader(input_dir="./",
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
        im_region = transforms.background_composite(image=im_region)
        return im_region

    def __slide_info(self):
        """WSI meta data reader

        Args:
            self (OpenSlideWSIReader):

        Returns:
            WSIMeta: containing meta information

        """
        input_dir = self.input_dir
        objective_power = np.int(
            self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        )

        slide_dimension = self.openslide_obj.level_dimensions[0]
        tile_objective_value = self.tile_objective_value
        rescale = np.int(objective_power / tile_objective_value)
        tile_read_size = self.tile_read_size
        level_count = self.openslide_obj.level_count
        level_dimensions = self.openslide_obj.level_dimensions
        level_downsamples = self.openslide_obj.level_downsamples
        file_name = self.file_name
        vendor = (self.openslide_obj.properties[openslide.PROPERTY_NAME_VENDOR],)
        mpp_x = (self.openslide_obj.properties[openslide.PROPERTY_NAME_MPP_X],)
        mpp_y = (self.openslide_obj.properties[openslide.PROPERTY_NAME_MPP_Y],)
        magnification_levels = [objective_power / lv for lv in level_downsamples]

        param = WSIMeta(
            input_dir=input_dir,
            objective_power=objective_power,
            slide_dimension=slide_dimension,
            rescale=rescale,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size.tolist(),
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            file_name=file_name,
            magnification_levels=magnification_levels,
        )

        return param

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (OpenSlideWSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.OpenSlideWSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()

        """
        openslide_obj = self.openslide_obj
        tile_objective_value = 20

        rescale = np.int(self.slide_info.objective_power / tile_objective_value)
        slide_dimension = self.slide_info.level_dimensions[0]
        slide_dimension_20x = np.array(slide_dimension) / rescale
        thumb = openslide_obj.get_thumbnail(
            (int(slide_dimension_20x[0] / 16), int(slide_dimension_20x[1] / 16))
        )
        thumb = np.asarray(thumb)

        return thumb


class OmnyxJP2WSIReader(WSIReader):
    """Class for reading Omnyx JP2 images.

    Attributes:
        glymur_obj (:obj:`glymur.Jp2k`)

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
        super().__init__(
            input_dir=input_dir,
            file_name=file_name,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        self.glymur_obj = glymur.Jp2k(
            filename=str(pathlib.Path(self.input_dir, self.file_name))
        )
        self.slide_info = self.__slide_info()

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
            >>> wsi_obj = wsireader.OmnyxJP2WSIReader(input_dir="./",
            ...     file_name="test.jp2")
            >>> level = 0
            >>> region = [13000, 17000, 15000, 19000]
            >>> im_region = wsi_obj.read_region(
            ...     region[0], region[1], region[2], region[3], level)
            >>> plt.imshow(im_region)

        """
        factor = 2 ** level
        start_h = start_h*factor
        start_w = start_w * factor
        end_h = end_h * factor
        end_w = end_w * factor

        glymur_obj = self.glymur_obj
        im_region = glymur_obj.read(rlevel=level,
                                    area=(start_h, start_w, end_h, end_w)
                                    )
        im_region = transforms.background_composite(image=im_region)
        return im_region

    def __slide_info(self):
        """JP2 meta data reader

        Args:
            self (OmnyxJP2WSIReader):

        Returns:
            WSIMeta: containing meta information

        """
        input_dir = self.input_dir
        box = self.glymur_obj.box
        m = re.search(r"(?<=AppMag = )\d\d", str(box[3]))
        objective_power = np.int(m.group(0))
        image_header = box[2].box[0]
        slide_dimension = (image_header.width, image_header.height)
        downsample_level = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        magnification_levels = [objective_power / lv for lv in downsample_level]
        level_dimensions = [(slide_dimension[0], slide_dimension[1]),
                            (int(slide_dimension[0]/2), int(slide_dimension[1]/2)),
                            (int(slide_dimension[0]/4), int(slide_dimension[1]/4)),
                            (int(slide_dimension[0]/8), int(slide_dimension[1]/8)),
                            (int(slide_dimension[0]/16), int(slide_dimension[1]/16)),
                            (int(slide_dimension[0]/32), int(slide_dimension[1]/32))
                            ]
        tile_objective_value = self.tile_objective_value
        rescale = np.int(objective_power / tile_objective_value)
        tile_read_size = self.tile_read_size
        level_count = None
        level_downsamples = None
        file_name = self.file_name
        vendor = "Omnyx JP2"
        m = re.search(r"(?<=AppMag = )\d\d", str(box[3]))
        mpp_x = float(m.group(0))
        mpp_y = float(m.group(0))

        param = WSIMeta(
            input_dir=input_dir,
            objective_power=objective_power,
            slide_dimension=slide_dimension,
            rescale=rescale,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size.tolist(),
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            file_name=file_name,
            magnification_levels=magnification_levels
        )

        return param

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (OmnyxJP2WSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.OmnyxJP2WSIReader(input_dir="./",
            ...     file_name="test.jp2")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()

        """
        glymur_obj = self.glymur_obj
        read_level = np.int(np.log2(self.slide_info.objective_power/1.25))
        thumb = np.asarray(glymur_obj.read(rlevel=read_level))

        return thumb
