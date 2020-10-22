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

"""WSIReader for WSI reading or extracting metadata information from WSIs."""
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
import numbers


class WSIReader:
    """WSI Reader class to read WSI images.

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
            input_path (str or pathlib.Path): input path to WSI
            output_dir (str or pathlib.Path): output directory to save the output,
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

    def relative_level_scales(self, resolution, units):
        """
        Calculate scale of each image pyramid level relative to the
        given target scale and units.

        Values > 1 indicate that the level has a larger scale than the
        target and < 1 indicates that it is smaller.

        Args:
            resolution (float or tuple of float): Scale to calculate
                relative to
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

        def make_into_array(x):
            """Ensure input x is a numpy array of length 2."""
            if isinstance(resolution, numbers.Number):
                # If one number is given, the same value is used for x and y
                return np.array([resolution] * 2)
            return np.array(resolution)

        @np.vectorize
        def level_to_downsample(x):
            """Get the downsample for a level, interpolating non-integer levels."""
            if isinstance(x, int) or int(x) == x:
                # Return the downsample for the level
                return info.level_downsamples[int(x)]
            # Linearly interpolate between levels
            floor = int(np.floor(x))
            ceil = int(np.ceil(x))
            floor_downsample = info.level_downsamples[floor]
            ceil_downsample = info.level_downsamples[ceil]
            return np.interp(x, [floor, ceil], [floor_downsample, ceil_downsample])

        resolution = make_into_array(resolution)

        if units == "mpp":
            if info.mpp is None:
                raise ValueError("MPP is None")
            base_scale = info.mpp
        elif units == "power":
            if info.objective_power is None:
                raise ValueError("Objective power is None")
            base_scale = 1 / info.objective_power
            resolution = 1 / resolution
        elif units == "level":
            if any(resolution >= len(info.level_downsamples)):
                raise ValueError("Target scale level > number of levels in WSI")
            base_scale = 1
            resolution = level_to_downsample(resolution)
        elif units == "baseline":
            base_scale = 1
            resolution = 1 / resolution
        else:
            raise ValueError("Invalid units")

        return [(base_scale * ds) / resolution for ds in info.level_downsamples]

    def find_optimal_level_and_downsample(self, resolution, units, precision=3):
        """
        Find the optimal level to read at for a desired resolution and units.

        The optimal level is the most downscaled level of the image
        pyramid (or multi-resolution layer) which is larger than the
        desired target scale. The returned downsample is the scale factor
        required, post read, to achieve the desired resolution.

        Args:
            resolution (float or tuple of float): Resolution to
                find optimal read parameters for
            units (str): Units of the scale. Allowed values are the same
                as for `WSIReader.relative_level_scales`
            precision (int, optional): Decimal places to use when
                finding optimal scale. This can be adjusted to avoid
                errors when an unecessary precision is used. E.g.
                1.1e-10 > 1 is insignificant in most cases.
                Defaults to 3.

        Returns:
            (int, float): Optimal read level and scale factor between
                the optimal level and the target scale (usually <= 1).
        """
        level_scales = self.relative_level_scales(resolution, units)
        # Note that np.argmax finds the index of the first True element.
        # Here it is used on a reversed list to find the first
        # element <=1, which is the same element as the last <=1
        # element when counting forward in the regular list.
        reverse_index = np.argmax(
            [all(np.round(x, decimals=precision) <= 1) for x in level_scales[::-1]]
        )
        # Convert the index from the reversed list to the regular index (level)
        level = (len(level_scales) - 1) - reverse_index
        scale = level_scales[level]
        if any(np.array(scale) > 1):
            warnings.warn(
                "Scale > 1."
                "This means that the desired scale is a higher"
                " resolution than the WSI can produce."
                "Interpolation of read regions may occur."
            )
        return level, scale

    def find_read_rect_params(self, location, size, resolution, units, precision=3):
        """
        Find the optimal parameters to use for reading a rect at a given
        resolution.

        Args:
            size (float): Desired output size in pixels
            resolutions (float): Resolution to calculate relative to
            units (str): Units of the scale. Allowed values are the same
                as for WSIReader.relative_level_scales

        Returns:
            (int, tuple of int, tuple of int, float): Optimal level,
                size (width, height) of the region to read, downscaling
                factor to apply after reading to reach size and
                correct scale.
        """
        read_level, post_read_downsample = self.find_optimal_level_and_downsample(
            resolution, units, precision
        )
        read_size = np.round(np.array(size) * (1 / post_read_downsample)).astype(int)
        level_location = np.round(np.array(location) / post_read_downsample).astype(int)
        return read_level, level_location, read_size, post_read_downsample

    def read_rect(self, location, size, resolution=0, units="level"):
        """Read a region of the whole slide image at a location and size.

        Location is in terms of the baseline image (level 0  / maximum
        resolution), and size is the output reguion size.

        This method reads provides a fast method for performing partial
        reads (reading without loading the whole image into memory) of
        the WSI.

        Reads can also be performed at different resolutions. This is
        done by supplying a pair of arguments for the resolution and
        units of resolution. If the WSI does not have a resolution layer
        corresponding exactly to the requested resolution, a larger
        resolution is downscaled to achieve the correct requested output
        resolution. If the requested resolution is higher than the
        baseline (maximum resultion of the image), then bicubic
        interpolation is applied to the output image.

        Args:
            location (tuple of int): (x, y) tuple giving
                the top left pixel in the baseline (level 0)
                reference frame.
            size (tuple of int): (width, height) tuple
                giving the desired output image size.
            resolution (int or float or tuple of float): resolution at
                which to read the image, default = 0. Either a single
                number or a sequence of two numbers for x and y are
                valid. This scale value is in terms of the corresponding
                units. For example: resolution=0.5 and units="mpp" will
                read the slide at 0.5 microns per-pixel, and
                resolution=3, units="level" will read at level at
                pyramid level / resolution layer 3.
            units (str): the units of scale, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                pixels per baseline pixel (baseline).

        Returns:
            ndarray : array of size MxNx3
            M=size[0], N=size[1]

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> # Load a WSI image
            >>> wsi = wsireader.WSIReader("/path/to/a/wsi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> img = wsi.read_rect(location, size)
            >>> # Read a region at 0.5 microns per pixel (mpp)
            >>> img = wsi.read_rect(location, size, 0.5, "mpp")
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=[0.5, 0.5],
            ...     units="mpp",
            ... )
            >>> # The resolution can be different in x and y, e.g.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=[0.5, 0.75],
            ...     units="mpp",
            ... )
            >>> # Several units can be used including: objective power,
            >>> # microns per pixel, pyramid/resolution level, and
            >>> # fraction of baseline.
            >>> # E.g. Read a region at an objective power of 10x
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=10,
            ...     units="power",
            ... )
            >>> # Read a region at pyramid / resolution level 1
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1,
            ...     units="level",
            ... )
            >>> # Read at a fractional level, this will linearly
            >>> # interpolate the downsampling factor between levels.
            >>> # E.g. if levels 0 and 1 have a downsampling of 1x and
            >>> # 2x of baseline, then level 0.5 will correspond to a
            >>> # downsampling factor 1.5x of baseline.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="level",
            ... )
            >>> # Read a region at half of the full / baseline
            >>> # resolution.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="baseline",
            ... )
            >>> # Read at a higher resolution than the baseline
            >>> # (interpolation applied to output)
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1.25,
            ...     units="baseline",
            ... )
            >>> # Assuming the image has a native mpp of 0.5,
            >>> # interpolation will be applied here.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.25,
            ...     units="mpp",
            ... )

        """
        raise NotImplementedError

    def read_bounds(self, start_w, start_h, end_w, end_h, resolution=0, units="level"):
        """Read a region of the whole slide image within given bounds.

        Bounds are in terms of the baseline image (level 0  / maximum
        resolution).

        Internally this method uses :func:`read_rect`. See
        :func:`read_rect` for more.

        This method reads provides a fast method for performing partial
        reads (reading without loading the whole image into memory) of
        the WSI.

        Reads can also be performed at different resolutions. This is
        done by supplying a pair of arguments for the resolution and
        units of resolution. If the WSI does not have a resolution layer
        corresponding exactly to the requested resolution, a larger
        resolution is downscaled to achieve the correct requested output
        resolution. If the requested resolution is higher than the
        baseline (maximum resultion of the image), then bicubic
        interpolation is applied to the output image.

        Args:
            start_w (int): starting point in x-direction (along width).
            start_h (int): starting point in y-direction (along height).
            end_w (int): end point in x-direction (along width).
            end_h (int): end point in y-direction (along height).
            resolution (int or float or tuple of float): resolution at
                which to read the image, default = 0. Either a single
                number or a sequence of two numbers for x and y are
                valid. This scale value is in terms of the corresponding
                units. For example: resolution=0.5 and units="mpp" will
                read the slide at 0.5 microns per-pixel, and
                resolution=3, units="level" will read at level at
                pyramid level / resolution layer 3.
            units (str): the units of scale, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                pixels per baseline pixel (baseline).

        Returns:
            ndarray : array of size MxNx3
            M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> from matplotlib import pyplot as plt
            >>> wsi_obj = wsireader.WSIReader("/path/to/a/wsi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = wsi_obj.read_bounds(*bounds)
            >>> plt.imshow(img)
            >>> # This could also be written more verbosely as follows
            >>> img = wsi_obj.read_bounds(
            ...     start_w=bounds[0],
            ...     start_h=bounds[1],
            ...     end_w=bounds[2],
            ...     end_h=bounds[3],
            ...     resolution=0,
            ...     units="level",
            ... )
            >>> plt.imshow(img)
        """
        location = (start_w, start_h)
        size = (end_w - start_w, end_h - start_h)
        return self.read_rect(
            location=location, size=size, resolution=resolution, units=units
        )

    def read_region(self, location, level, size):
        """Read a region of the whole slide image (OpenSlide format args).

        This function is to help with writing code which is backwards
        compatible with OpenSlide. As such it has the same arguments.

        Other reader classes will inherit this function and therefore
        some WSI formats which are not supported by OpenSlide may also
        be readable with the same syntax.

        Args:
            location: (x, y) tuple giving the top left pixel in the
                level 0 reference frame.
            level: the level number.
            size: (width, height) tuple giving the region size.

        Returns:
            PIL.Image: Image containing the contents of the region.
        """
        return self.read_rect(
            location=location, size=size, resolution=level, units="level"
        )

    def slide_thumbnail(self, resolution=1.25, units="power"):
        """Read the whole slide image thumbnail at 1.25x.

        Args:
            self (WSIReader):

        Returns:
            ndarray : image array

        Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> wsi = wsireader.OpenSlideWSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi.slide_thumbnail()
        """
        level, post_read_scale = self.find_optimal_level_and_downsample(
            resolution, units
        )
        level_dimensions = self.slide_info.level_dimensions[level]
        thumb = self.read_bounds(0, 0, *level_dimensions)

        if np.any(post_read_scale != 1.0):
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
                end_h = min(end_h, slide_h)
                end_w = min(end_w, slide_w)

                # Read image region
                im = self.read_bounds(start_w, start_h, end_w, end_h, level)

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

    def read_rect(self, location, size, resolution=0, units="level"):
        target_size = size

        # Find parameters for optimal read
        (read_level, _, read_size, post_read_scale,) = self.find_read_rect_params(
            location=location, size=target_size, resolution=resolution, units=units,
        )
        wsi = self.openslide_wsi

        # Read at optimal level and corrected read size
        im_region = wsi.read_region(location, read_level, read_size)
        im_region = np.array(im_region)

        # Resize to correct scale if required
        if np.any(post_read_scale != 1.0):
            interpolation = cv2.INTER_AREA
            if np.any(post_read_scale > 1.0):
                interpolation = cv2.INTER_CUBIC
            im_region = cv2.resize(im_region, target_size, interpolation=interpolation)

        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
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

    def slide_thumbnail(self, resolution=1.25, units="power"):
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

    def read_rect(self, location, size, resolution=0, units="level"):
        target_size = size
        # Find parameters for optimal read
        (
            read_level,
            level_location,
            read_size,
            post_read_scale,
        ) = self.find_read_rect_params(
            location=location, size=target_size, resolution=resolution, units=units,
        )
        # Read at optimal level and corrected read size
        area = (*level_location[::-1], *(level_location[::-1] + read_size))

        glymur_wsi = self.glymur_wsi
        im_region = glymur_wsi.read(rlevel=read_level, area=area)
        if np.any(post_read_scale != 1.0):
            interpolation = cv2.INTER_AREA
            if np.any(post_read_scale > 1.0):
                interpolation = cv2.INTER_CUBIC
            im_region = cv2.resize(im_region, target_size, interpolation=interpolation)

        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
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

    def slide_thumbnail(self, resolution=1.25, units="power"):
        glymur_wsi = self.glymur_wsi
        read_level = np.int(np.log2(self.slide_info.objective_power / 1.25))
        thumb = np.asarray(glymur_wsi.read(rlevel=read_level))

        return thumb
