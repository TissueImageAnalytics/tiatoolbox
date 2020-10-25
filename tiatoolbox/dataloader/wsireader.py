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
import numpy as np
import openslide
import glymur
import math
import pandas as pd
import re
import warnings


class WSIReader:
    """WSI Reader class to read WSI images.

    Attributes:
        input_path (pathlib.Path): input path to WSI directory.
        output_dir (pathlib.Path): output directory to save the output.
        tile_objective_value (int): objective value at which tile is generated.
        tile_read_size (int): [tile width, tile height]
        slide_info (WSIMeta): Whole slide image slide information.

    Args:
        input_path (str, pathlib.Path): input path to WSI.
        output_dir (str, pathlib.Path): output directory to save the output,
         default=./output.
        tile_objective_value (int): objective value at which tile is generated,
         default=20.
        tile_read_size_w (int): tile width, default=5000.
        tile_read_size_h (int): tile height, default=5000.

    """

    def __init__(
        self,
        input_path=".",
        output_dir="./output",
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
    ):

        self.input_path = pathlib.Path(input_path)
        if output_dir is not None:
            self.output_dir = pathlib.Path(output_dir, self.input_path.name)

        self.tile_objective_value = np.int(tile_objective_value)  # Tile magnification
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])

    @property
    def slide_info(self):
        """WSI meta data reader.

        Args:
            self (WSIReader):

        Returns:
            WSIMeta: An object containing normalised slide metadata

        """
        raise NotImplementedError

    def read_region(self, start_w, start_h, end_w, end_h, level=0):
        """Read a region in whole slide image.
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
        """Read whole slide image thumbnail at 1.25x.

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

    def read_region(self, start_w, start_h, end_w, end_h, level=0):
        """Read a region in whole slide image.

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
            >>> wsi = wsireader.OpenSlideWSIReader(input_path='./CMU-1.ndpi')
            >>> level = 0
            >>> region = [13000, 17000, 15000, 19000]
            >>> im_region = wsi.read_region(
            ...     region[0], region[1], region[2], region[3], level)
            >>> plt.imshow(im_region)

        """
        openslide_obj = self.openslide_wsi
        im_region = openslide_obj.read_region(
            [start_w, start_h], level, [end_w - start_w, end_h - start_h]
        )
        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
        """Openslide WSI meta data reader.

        Args:
            self (OpenSlideWSIReader):

        Returns:
            WSIMeta: containing meta information.

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

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.25x.

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

    def read_region(self, start_w, start_h, end_w, end_h, level=0):
        """Read a region in whole slide image.

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
        factor = 2 ** level
        start_h = start_h * factor
        start_w = start_w * factor
        end_h = end_h * factor
        end_w = end_w * factor

        glymur_wsi = self.glymur_wsi
        im_region = glymur_wsi.read(rlevel=level, area=(start_h, start_w, end_h, end_w))
        im_region = transforms.background_composite(image=im_region)
        return im_region

    @property
    def slide_info(self):
        """JP2 meta data reader.
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

    def slide_thumbnail(self):
        """Read whole slide image thumbnail at 1.25x.

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
