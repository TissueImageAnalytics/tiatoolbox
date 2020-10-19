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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""WSIReader for WSI reading or extracting metadata information from WSIs"""
from tiatoolbox.utils import misc, transforms
from tiatoolbox.dataloader.wsimeta import WSIMeta

import pathlib
import warnings
import numpy as np
import openslide
import glymur
import math
import pandas as pd
import cv2
import re


class WSIReader:
    """WSI Reader class to read WSI images

    Attributes:
        input_path (pathlib.Path): input path to WSI directory
        output_dir (pathlib.Path): output directory to save the output
        tile_objective_value (int): objective value at which tile is generated
        tile_read_size (int): [tile width, tile height]
        slide_info (WSIMeta): Whole slide image slide information

    """

    def __init__(
        self,
        input_path=".",
        output_dir="./output",
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):
        """
        Args:
            input_path (str, pathlib.Path): input path to WSI
            output_dir (str, pathlib.Path): output directory to save the output,
                default=./output
            tile_objective_value (int): objective value at which tile is generated,
                default=20
            tile_read_size_w (int): tile width, default=5000
            tile_read_size_h (int): tile height, default=5000

        """

        self.input_path = pathlib.Path(input_path)
        if output_dir is not None:
            self.output_dir = pathlib.Path(output_dir, self.input_path.name)

        self.tile_objective_value = np.int(tile_objective_value)  # Tile magnification
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])

    @property
    def slide_info(self):
        """WSI meta data reader

        Args:
            self (WSIReader):

        Returns:
            WSIMeta: An object containing normalised slide metadata

        """
        raise NotImplementedError

    def relative_level_scales(self, target_scale, units):
        """
        Calculate scale of each image pyramid level relative to the
        given target scale and units.

        Values > 1 indicate that the level has a larger scale than the
        target and < 1 indicates that it is smaller.

        Args:
            target_scale (float): Scale to calculate relative to
            units (str): Units of the scale. Allowed values are: mpp,
                power, level, baseline. Baseline refers to the largest
                resolution in the WSI (level 0).

        Raises:
            ValueError: Missing MPP metadata
            ValueError: Missing objective power metadata
            ValueError: Invalid units

        Returns:
            list: Scale for each level relative to the given scale and
                units

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi = wsireader.WSIReader("CMU-1.ndpi")
            >>> print(wsi.relative_level_scales(0.5, "mpp"))
            [array([0.91282519, 0.91012514]), array([1.82565039, 1.82025028]) ...

            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi = wsireader.WSIReader("CMU-1.ndpi")
            >>> print(wsi.relative_level_scales(0.5, "baseline"))
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        """
        info = self.slide_info

        if units == "mpp":
            if info.mpp is not None:
                base_scale = info.mpp
            else:
                raise ValueError("MPP is None")
        elif units == "power":
            if info.objective_power is not None:
                base_scale = 1 / info.objective_power
                target_scale = 1 / target_scale
            else:
                raise ValueError("Objective power is None")
        elif units == "level":
            if target_scale >= len(info.level_downsamples):
                raise ValueError("Target scale level > number of levels in WSI")
            base_scale = 1
            target_scale = info.level_downsamples[target_scale]
        elif units == "baseline":
            base_scale = 1
            target_scale = 1 / target_scale
        else:
            raise ValueError("Invalid units")

        return [(base_scale * ds) / target_scale for ds in info.level_downsamples]

    def optimal_relative_level_scale(self, target_scale, units, precision=3):
        """
        Find the optimal level to read at for a desired scale and units.

        The optimal level is the most downscaled level of the image
        pyramid (or multi-resolution layer) which is larger than the
        desired target scale. The returned scale is the scale factor
        required, post read, to achieve the desired scale.

        Args:
            target_scale (float): Scale to calculate relative to
            units (str): Units of the scale. Allowed values are the same
                as for WSIReader.relative_level_scales
            precision (int, optional): Decimal places to use when
                finding optimal scale. This can be adjusted to avoid
                errors when an unecessary precision is used. E.g.
                1.1e-10 > 1 is insignificant in most cases.
                Defaults to 3.

        Returns:
            tuple: Optimal read level and scale factor between the optimal level
                and the target scale (usually <= 1).
        """
        level_scales = self.relative_level_scales(target_scale, units)
        # Note that np.argmax finds the index of the first True element.
        # Here it is used on a reversed list to find the first
        # element <=1, which is the same element as the last <=1
        # element when counting forward in the regular list.
        reverse_index = np.argmax(
            [np.all(np.round(x, decimals=precision) <= 1) for x in level_scales[::-1]]
        )
        # Convert the index from the reversed list to the regular index (level)
        level = (len(level_scales) - 1) - reverse_index
        scale = level_scales[level]
        if np.any(np.array(scale) > 1):
            warnings.warn(
                "Scale > 1."
                "This means that the desired scale is a higher"
                " resolution than the WSI can produce."
                "Interpolation of read regions may occur."
            )
        return level, scale

    def read_rect_params_for_scale(
        self, location, target_size, target_scale, units, scale_kwargs=None
    ):
        """
        Find the optimal parameters to use for reading a rect at a give
        scale.

        Args:
            target_size (float): Desired output size in pixels
            target_scale (float): Scale to calculate relative to
            units (str): Units of the scale. Allowed values are the same
                as for WSIReader.relative_level_scales

        Returns:
            tuple: Optimal level, size (width, height) of the region to
                read, downscaling factor to apply after reading to reach
                target_size and correct scale.
        """
        if scale_kwargs is None:
            scale_kwargs = {}
        level, scale = self.optimal_relative_level_scale(
            target_scale, units, **scale_kwargs
        )
        read_size = np.round(np.array(target_size) * (1 / scale)).astype(int)
        post_read_scale = scale
        level_location = np.round(np.array(location) / scale).astype(int)
        return level, level_location, read_size, post_read_scale

    def read_region(self, start_w, start_h, end_w, end_h, scale=0, units="level"):
        """Read a region in whole slide image

        Args:
            start_w (int): starting point in x-direction (along width)
            start_h (int): starting point in y-direction (along height)
            end_w (int): end point in x-direction (along width)
            end_h (int): end point in y-direction (along height)
            scale (float): scale at which to read the image, default = 0
            units (str): the units of scale, default = "level".
                supported units are "mpp", "power", "level", "baseline"

        Returns:
            img_array : ndarray of size MxNx3
            M=end_h-start_h, N=end_w-start_w

        """
        raise NotImplementedError

    def slide_thumbnail(self, scale=1.25, units="power"):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (WSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.OpenSlideWSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()

        """
        level, post_read_scale = self.optimal_relative_level_scale(scale, units)
        level_dimensions = self.slide_info.level_dimensions[level]
        thumb = self.read_region(0, 0, *level_dimensions)

        if scale != 1.0:
            new_size = np.round(np.array(level_dimensions) * post_read_scale)
            new_size = tuple(new_size.astype(int))
            thumb = cv2.resize(thumb, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite("/home/john/Downloads/thumbnail.png", thumb)

        thumb = np.array(thumb)

        return thumb

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
            >>> wsi = wsireader.WSIReader(input_path="./CMU-1.ndpi",
            ...     output_dir='./dev_test',
            ...     tile_objective_value=10,
            ...     tile_read_size_h=2000,
            ...     tile_read_size_w=2000)
            >>> wsi.save_tiles()

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_param = wsi_obj.slide_info()

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


class OpenSlideWSIReader(WSIReader):
    """Class for reading OpenSlide supported whole-slide images.

    Attributes:
        openslide_wsi (:obj:`openslide.OpenSlide`)

    """

    def __init__(
        self,
        input_path=".",
        output_dir="./output",
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):
        super().__init__(
            input_path=input_path,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        self.openslide_wsi = openslide.OpenSlide(filename=str(self.input_path))

    def read_region(self, start_w, start_h, end_w, end_h, scale=0, units="level"):
        """Read a region in whole slide image

        Args:
            start_w (int): starting point in x-direction (along width)
            start_h (int): starting point in y-direction (along height)
            end_w (int): end point in x-direction (along width)
            end_h (int): end point in y-direction (along height)
            scale (float): scale at which to read the image, default = 0
            units (str): the units of scale, default = "level",
                supported units are "mpp", "power", "level", "baseline"

        Returns:
            img_array : ndarray of size MxNx3
            M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> from matplotlib import pyplot as plt
            >>> wsi = wsireader.OpenSlideWSIReader(input_path='./CMU-1.ndpi')
            >>> level = 0
            >>> region = [1000, 2000, 2000, 3000]
            >>> im_region = wsi.read_region(*region, level)
            >>> plt.imshow(im_region)

        """
        target_size = [end_w - start_w, end_h - start_h]
        location = (start_w, start_h)
        # Find parameters for optimal read
        (
            read_level,
            level_location,
            read_size,
            post_read_scale,
        ) = self.read_rect_params_for_scale(
            location=location, target_size=target_size, target_scale=scale, units=units,
        )
        wsi = self.openslide_wsi
        # Read at optimal level and corrected read size
        im_region = wsi.read_region([start_w, start_h], read_level, read_size)
        # Resize to correct scale if required
        if post_read_scale != 1.0:
            interpolation = cv2.INTER_AREA
            if post_read_scale > 1.0:
                interpolation = cv2.INTER_CUBIC
            im_region = cv2.resize(im_region, target_size, interpolation=interpolation)

        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
        """Openslide WSI meta data reader

        Args:
            self (OpenSlideWSIReader):

        Returns:
            WSIMeta: containing meta information

        """
        objective_power = np.int(
            self.openslide_wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        )

        slide_dimensions = self.openslide_wsi.level_dimensions[0]
        level_count = self.openslide_wsi.level_count
        level_dimensions = self.openslide_wsi.level_dimensions
        level_downsamples = self.openslide_wsi.level_downsamples
        vendor = self.openslide_wsi.properties[openslide.PROPERTY_NAME_VENDOR]
        mpp_x = self.openslide_wsi.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = self.openslide_wsi.properties[openslide.PROPERTY_NAME_MPP_Y]
        mpp = [mpp_x, mpp_y]

        param = WSIMeta(
            file_path=self.input_path,
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
            raw=self.openslide_wsi.properties,
        )

        return param

    def slide_thumbnail(self, scale=1.25, units="power"):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (OpenSlideWSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi = wsireader.OpenSlideWSIReader(input_path="./CMU-1.ndpi")
            >>> slide_thumbnail = wsi.slide_thumbnail()

        """
        openslide_wsi = self.openslide_wsi
        tile_objective_value = 20

        rescale = np.int(self.slide_info.objective_power / tile_objective_value)
        slide_dimension = self.slide_info.level_dimensions[0]
        slide_dimension_20x = np.array(slide_dimension) / rescale
        thumb = openslide_wsi.get_thumbnail(
            (int(slide_dimension_20x[0] / 16), int(slide_dimension_20x[1] / 16))
        )
        thumb = np.asarray(thumb)

        return thumb


class OmnyxJP2WSIReader(WSIReader):
    """Class for reading Omnyx JP2 images.

    Attributes:
        glymur_wsi (:obj:`glymur.Jp2k`)

    """

    def __init__(
        self,
        input_path=".",
        output_dir="./output",
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):
        super().__init__(
            input_path=input_path,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        self.glymur_wsi = glymur.Jp2k(filename=str(self.input_path))

    def read_region(self, start_w, start_h, end_w, end_h, scale=0, units="level"):
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
            >>> wsi = wsireader.OmnyxJP2WSIReader(input_path="./test.jp2")
            >>> level = 0
            >>> region = [13000, 17000, 15000, 19000]
            >>> im_region = wsi.read_region(
            ...     region[0], region[1], region[2], region[3], level)
            >>> plt.imshow(im_region)

        """
        target_size = (end_w - start_w, end_h - start_h)
        location = (start_w, start_h)
        # Find parameters for optimal read
        (
            read_level,
            level_location,
            read_size,
            post_read_scale,
        ) = self.read_rect_params_for_scale(
            location=location, target_size=target_size, target_scale=scale, units=units,
        )
        # Read at optimal level and corrected read size
        area = (*level_location[::-1], *(level_location[::-1] + read_size))

        glymur_wsi = self.glymur_wsi
        im_region = glymur_wsi.read(rlevel=read_level, area=area)
        if post_read_scale != 1.0:
            interpolation = cv2.INTER_AREA
            if post_read_scale > 1.0:
                interpolation = cv2.INTER_CUBIC
            im_region = cv2.resize(im_region, target_size, interpolation=interpolation)

        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
        """JP2 meta data reader

        Args:
            self (OmnyxJP2WSIReader):

        Returns:
            WSIMeta: containing meta information

        """

        glymur_wsi = self.glymur_wsi
        box = glymur_wsi.box
        m = re.search(r"(?<=AppMag = )\d\d", str(box[3]))
        objective_power = np.int(m.group(0))
        image_header = box[2].box[0]
        slide_dimensions = (image_header.width, image_header.height)

        # Determine level_count
        cod = None
        for segment in glymur_wsi.codestream.segment:
            if isinstance(segment, glymur.codestream.CODsegment):
                cod = segment

        if cod is None:
            warnings.warn(
                "JP2 codestream missing COD segment! "
                "Cannot determine number of decompositions (levels)"
            )
            level_count = 1
        else:
            level_count = cod.num_res

        level_downsamples = [2 ** n for n in range(level_count)]

        level_dimensions = [
            (int(slide_dimensions[0] / 2 ** n), int(slide_dimensions[1] / 2 ** n))
            for n in range(level_count)
        ]

        vendor = "Omnyx JP2"
        m = re.search(r"(?<=AppMag = )\d\d", str(box[3]))
        mpp_x = float(m.group(0))
        mpp_y = float(m.group(0))
        mpp = [mpp_x, mpp_y]

        param = WSIMeta(
            file_path=self.input_path,
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
            raw=self.glymur_wsi.box,
        )

        return param

    def slide_thumbnail(self, scale=1.25, units="power"):
        """Read whole slide image thumbnail at 1.25x

        Args:
            self (OmnyxJP2WSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi = wsireader.OmnyxJP2WSIReader(input_path="./test.jp2")
            >>> slide_thumbnail = wsi.slide_thumbnail()

        """
        glymur_wsi = self.glymur_wsi
        read_level = np.int(np.log2(self.slide_info.objective_power / 1.25))
        thumb = np.asarray(glymur_wsi.read(rlevel=read_level))

        return thumb
