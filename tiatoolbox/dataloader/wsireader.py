"""WSIReader for WSI reading or extracting metadata information from WSIs"""
from tiatoolbox.utils import misc, transforms
from tiatoolbox.dataloader.wsimeta import WSIMeta

import pathlib
import numpy as np
import openslide
import math
import warnings
import pandas as pd
import cv2


class WSIReader:
    """WSI Reader class to read WSI images

    Attributes:
        input_dir (pathlib.Path): input path to WSI directory
        file_name (str): file name of the WSI
        slide_info (dict): Whole slide image slide information

    """

    def __init__(
        self, input_dir=".", file_name=None,
    ):
        """
        Args:
            input_dir (str, pathlib.Path): input path to WSI directory
            file_name (str): file name of the WSI

        """

        self.input_dir = pathlib.Path(input_dir)
        self.file_name = pathlib.Path(file_name).name

    @property
    def slide_info(self):
        """WSI meta data reader

        Args:
            self (WSIReader):

        Returns:
            WSIMeta: An object containing normalised slide metadata

        """
        raise NotImplementedError

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

    def tiles(self, size=(224, 224), scale=None, objective_power=None, mpp=None):
        """Generator which yeilds tiles at a given scale.

        Args:
            size (int): [tile width, tile height]
            scale (float): scale at which tiles are generated
            objective_power (float): objective power at which tiles are generated
            mpp (float, list, ndarray): mpp at which tiles are generated

        Returns:
            tuple : index, slice, image
        """
        level, scale = self.level_and_scale(scale, objective_power, mpp)
        tile_read_size = np.round(np.multiply(size, 1 / scale)).astype(int)
        slide_w, slide_h = self.slide_info.slide_dimensions
        tile_w, tile_h = tile_read_size

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
                img = self.read_region(start_w, start_h, end_w, end_h, level)

                # Rescale to the tile to the correct size
                img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

                yield (w, h), (start_w, start_h, end_w, end_h), img

    def save_tiles(
        self,
        output_dir,
        size=(5000, 5000),
        tile_format=".jpg",
        scale=None,
        objective_power=None,
        mpp=None,
        verbose=True,
    ):
        """Generate image tiles at a given scale and save to disk.

        Args:
            self (WSIReader):
            output_dir (pathlib.Path): output directory to save the output
            size (int): [tile width, tile height]
            tile_format (str): file format to save image tiles, default=".jpg"
            scale (float): scale at which tiles are generated
            objective_power (float): objective power at which tiles are generated
            mpp (float, list, ndarray): mpp at which tiles are generated
            verbose (bool): Print output, default=True

        Returns:
            None

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
        output_dir = pathlib.Path(output_dir) / self.file_name
        output_dir.mkdir(parents=True)
        data = []

        for i, (index, slice_, img) in enumerate(
            self.tiles(size, scale, objective_power, mpp)
        ):
            x, y = index
            start_w, start_h, end_w, end_h = slice_

            if verbose:
                format_str = (
                    "Tile%d:  start_w:%d, end_w:%d, "
                    "start_h:%d, end_h:%d, "
                    "width:%d, height:%d"
                )

                print(
                    format_str
                    % (
                        i,
                        start_w,
                        end_w,
                        start_h,
                        end_h,
                        end_w - start_w,
                        end_h - start_h,
                    ),
                    flush=True,
                )

            img_save_name = "_".join(["Tile", str(y), str(x)]) + tile_format

            misc.imwrite(image_path=output_dir.joinpath(img_save_name), img=img)

            data.append(
                [
                    i,
                    img_save_name,
                    start_w,
                    end_w,
                    start_h,
                    end_h,
                    img.shape[0],
                    img.shape[1],
                ]
            )

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

    def scale_for_mpp(self, mpp):
        """Find scale factor between the given microns-per-pixel (mpp) and the slide
        mpp (at level 0)

        Args:
            mpp (float, ndarray): Target microns per pixel (mpp)

        Returns:
            float : scale
        """
        target_mpp = mpp
        slide_mpp = self.slide_info.mpp

        if slide_mpp is None:
            raise ValueError("Cannot determine rescale. MPP of the slide is unknown.")
        if any(target_mpp < x for x in slide_mpp):
            warnings.warn(
                "Requested MPP is less than minimum slide MPP."
                "Output may be interpolated."
            )

        scale = slide_mpp / target_mpp
        return scale

    def scale_for_objective_power(self, power):
        """Find scale factor between the given objective power and the slide
        objective power (at level 0)

        Args:
            power (float): Target objective power

        Returns:
            float : rescale
        """
        target_power = power
        slide_power = self.slide_info.objective_power

        if slide_power is None:
            raise ValueError(
                "Cannot determine rescale. Objective power of the slide is unknown."
            )
        if target_power > slide_power:
            warnings.warn(
                "Requested objective power is greater than maximum slide objective power."
                "Output may be interpolated."
            )

        scale = target_power / slide_power
        return np.array([scale, scale])

    def level_and_scale(self, scale=None, objective_power=None, mpp=None, precision=4):
        """Determine the best level and rescaling to read data most efficiently.

        If a required rescaling relative to level 0 is greater than the
        downsample offered by a level, the level is returned with an adjusted
        rescale for that level.

        The precision argument is used to avoid floating point errors caused
        by C floats from reader libraries such as OpenSlide when finding
        the optimal slide level to read at.

        Args:
            scale (float, ndarray): Desired rescaling relative to level 0
            objective_power (float): Desired objetive power
            mpp (float): Desired microns-per-pixel
            precision (int): Decimal places to check level downsamples to

        Returns:
            tuple: A 2-tuple with the optimal level and scaling for that level
        """
        if len([x for x in [scale, objective_power, mpp] if x is not None]) > 1:
            params = {"scale": scale, "objective_power": objective_power, "mpp": mpp}
            raise ValueError(
                "Only one of: scale, objective_power, mpp can be given."
                " Received {}".format(params)
            )
        if scale is not None:
            l0_scale = scale
        elif objective_power is not None:
            l0_scale = self.scale_for_objective_power(objective_power)
        elif mpp is not None:
            l0_scale = self.scale_for_mpp(mpp)

        if isinstance(l0_scale, (int, float)):
            l0_scale = np.array([l0_scale] * 2, dtype=np.float64)
        elif isinstance(l0_scale, (list,)):
            l0_scale = np.array(l0_scale, dtype=np.float64)

        l0_downsample = float(max(1 / l0_scale))

        # Not relying on OpenSlide get_best_level_for_downsample etc. becuase
        # of C float comparison errors e.g. 4.0000000000001 > 4
        for level, downsample in enumerate(self.slide_info.level_downsamples):
            if round(downsample, precision) > round(l0_downsample, precision):
                level = max(level - 1, 0)
                break
        level_downsample = float(self.slide_info.level_downsamples[level])
        level_scale = (1 / level_downsample) / l0_scale

        return level, level_scale


class OpenSlideWSIReader(WSIReader):
    """Class for reading OpenSlide supported whole-slide images.

    Attributes:
        openslide_obj (:obj:`openslide.OpenSlide`)
        file_name (str): file name of the WSI
        slide_info (dict): Whole slide image slide information

    """

    def __init__(
        self, input_dir=".", file_name=None,
    ):
        super().__init__(
            input_dir=input_dir, file_name=file_name,
        )
        self.openslide_obj = openslide.OpenSlide(
            filename=str(pathlib.Path(self.input_dir, self.file_name))
        )

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

    @property
    def slide_info(self):
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

        slide_dimensions = self.openslide_obj.level_dimensions[0]
        level_count = self.openslide_obj.level_count
        level_dimensions = self.openslide_obj.level_dimensions
        level_downsamples = self.openslide_obj.level_downsamples
        file_name = self.file_name
        vendor = self.openslide_obj.properties[openslide.PROPERTY_NAME_VENDOR]
        mpp_x = self.openslide_obj.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = self.openslide_obj.properties[openslide.PROPERTY_NAME_MPP_Y]
        mpp = [mpp_x, mpp_y]

        param = WSIMeta(
            input_dir=input_dir,
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
            file_name=file_name,
            raw=self.openslide_obj.properties,
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
