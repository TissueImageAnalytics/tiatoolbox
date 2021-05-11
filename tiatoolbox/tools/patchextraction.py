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

"""This file defines patch extraction methods for deep learning models."""
from abc import ABC
import numpy as np
import math

from tiatoolbox.wsicore import wsireader
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.utils import misc


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_img(str, pathlib.Path, :class:`numpy.ndarray`): input image for
          patch extraction.
        patch_size(int or tuple(int)): patch size tuple (width, height).
        resolution (int or float or tuple of float): resolution at
          which to read the image, default = 0. Either a single
          number or a sequence of two numbers for x and y are
          valid. This value is in terms of the corresponding
          units. For example: resolution=0.5 and units="mpp" will
          read the slide at 0.5 microns per-pixel, and
          resolution=3, units="level" will read at level at
          pyramid level / resolution layer 3.
        units (str): the units of resolution, default = "level".
          Supported units are: microns per pixel (mpp), objective
          power (power), pyramid / resolution level (level),
          Only pyramid / resolution levels (level) embedded in
          the whole slide image are supported.
        pad_mode (str): Method for padding at edges of the WSI. Default
          to 'constant'. See :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)): Values to use with
          constant padding. Defaults to 0. See :func:`numpy.pad` for
          more.

    Attributes:
        wsi(WSIReader): input image for patch extraction of type :obj:`WSIReader`.
        patch_size(tuple(int)): patch size tuple (width, height).
        resolution(tuple(int)): resolution at which to read the image.
        units (str): the units of resolution.
        n(int): current state of the iterator.
        locations_df(pd.DataFrame): A table containing location and/or type of patch.
        pad_mode (str): Method for padding at edges of the WSI.
          See :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)): Values to use with
          constant padding. Defaults to 0. See :func:`numpy.pad` for
          more.
        stride (tuple(int)): stride in (x, y) direction for patch extraction. Not used
         for :obj:`PointsPatchExtractor`

    """

    def __init__(
        self,
        input_img,
        patch_size,
        resolution=0,
        units="level",
        pad_mode="constant",
        pad_constant_values=0,
    ):
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))
        self.resolution = resolution
        self.units = units
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.n = 0
        self.wsi = wsireader.get_wsireader(input_img=input_img)
        self.locations_df = None
        self.stride = None

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        n = self.n

        if n >= self.locations_df.shape[0]:
            raise StopIteration
        self.n = n + 1
        return self[n]

    def __getitem__(self, item):
        if type(item) is not int:
            raise TypeError("Index should be an integer.")

        if item >= self.locations_df.shape[0]:
            raise IndexError

        x = self.locations_df["x"][item]
        y = self.locations_df["y"][item]

        data = self.wsi.read_rect(
            location=(int(x), int(y)),
            size=self.patch_size,
            resolution=self.resolution,
            units=self.units,
            pad_mode=self.pad_mode,
            pad_constant_values=self.pad_constant_values,
        )

        return data

    def _generate_location_df(self):
        """Generate location list based on slide dimension.
        The slide dimension is calculated using units and resolution.

        """
        (read_level, _, _, _, baseline_read_size,) = self.wsi.find_read_rect_params(
            location=(0, 0),
            size=self.patch_size,
            resolution=self.resolution,
            units=self.units,
        )

        slide_dimension = self.wsi.info.level_dimensions[0]
        level_downsample = self.wsi.info.level_downsamples[read_level]

        img_w = slide_dimension[0]
        img_h = slide_dimension[1]
        img_patch_w = baseline_read_size[0]
        img_patch_h = baseline_read_size[1]
        stride_w = self.stride[0] * level_downsample
        stride_h = self.stride[1] * level_downsample

        data = []

        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                start_w = w * stride_w
                data.append([start_w, start_h, None])

        self.locations_df = misc.read_locations(input_table=np.array(data))

        return self

    def merge_patches(self, patches):
        """Merge the patch-level results to get the overall image-level prediction.

        Args:
            patches: patch-level predictions

        Returns:
            image: merged prediction

        """
        raise NotImplementedError


class FixedWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using fixed sized windows for images and labels.

    Args:
        stride(int or tuple(int)): stride in (x, y) direction for patch extraction,
         default = patch_size

    Attributes:
        stride(tuple(int)): stride in (x, y) direction for patch extraction.

    """

    def __init__(
        self,
        input_img,
        patch_size,
        resolution=0,
        units="level",
        stride=None,
        pad_mode="constant",
        pad_constant_values=0,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )
        if stride is None:
            self.stride = self.patch_size
        else:
            if isinstance(stride, (tuple, list)):
                self.stride = (int(stride[0]), int(stride[1]))
            else:
                self.stride = (int(stride), int(stride))

        self._generate_location_df()

    def merge_patches(self, patches):
        raise NotImplementedError


class VariableWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using variable sized windows for images and labels.

    Args:
        stride(tuple(int)): stride in (x, y) direction for patch extraction.
        label_patch_size(tuple(int)): network output label (width, height).

    Attributes:
        stride(tuple(int)): stride in (x, y) direction for patch extraction.
        label_patch_size(tuple(int)): network output label (width, height).

    """

    def __init__(
        self,
        input_img,
        patch_size,
        resolution=0,
        units="level",
        stride=(1, 1),
        pad_mode="constant",
        pad_constant_values=0,
        label_patch_size=None,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )
        self.stride = stride
        self.label_patch_size = label_patch_size

    def __next__(self):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class PointsPatchExtractor(PatchExtractor):
    """Extracting patches with specified points as a centre.

    Args:
        locations_list(ndarray, pd.DataFrame, str, pathlib.Path): contains location
         and/or type of patch. Input can be path to a csv or json file.
         classification, default=9 (centre of patch and all the eight neighbours as
         centre).

    """

    def __init__(
        self,
        input_img,
        locations_list,
        patch_size=(224, 224),
        resolution=0,
        units="level",
        pad_mode="constant",
        pad_constant_values=0,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        self.locations_df = misc.read_locations(input_table=locations_list)
        self.locations_df["x"] = self.locations_df["x"] - int(
            (self.patch_size[1] - 1) / 2
        )
        self.locations_df["y"] = self.locations_df["y"] - int(
            (self.patch_size[1] - 1) / 2
        )

    def merge_patches(self, patches=None):
        """Merge patch is not supported by :obj:`PointsPatchExtractor`.
        Calling this function for :obj:`PointsPatchExtractor` will raise an error. This
        overrides the merge_patches function in the base class :obj:`PatchExtractor`

        """
        raise MethodNotSupported(
            message="Merge patches not supported for PointsPatchExtractor"
        )


def get_patch_extractor(method_name, **kwargs):
    """Return a patch extractor object as requested.

    Args:
        method_name (str): name of patch extraction method, must be one of "point",
          "fixedwindow", "variablewindow".
        **kwargs: Keyword arguments passed to :obj:`PatchExtractor`.

    Returns:
        PatchExtractor: an object with base :obj:`PatchExtractor` as base class.

    Examples:
        >>> from tiatoolbox.tools.patchextraction import get_patch_extractor
        >>> # PointsPatchExtractor with default values
        >>> patch_extract = get_patch_extractor(
        ...  'point', img_patch_h=200, img_patch_w=200)

    """
    if method_name.lower() == "point":
        patch_extractor = PointsPatchExtractor(**kwargs)
    elif method_name.lower() == "fixedwindow":
        patch_extractor = FixedWindowPatchExtractor(**kwargs)
    elif method_name.lower() == "variablewindow":
        patch_extractor = VariableWindowPatchExtractor(**kwargs)
    else:
        raise MethodNotSupported

    return patch_extractor
