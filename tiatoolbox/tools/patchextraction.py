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

from tiatoolbox.wsicore.wsireader import get_wsireader
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.utils.misc import read_locations


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_img(str, pathlib.Path, ndarray): input image for patch extraction.
        patch_size(Tuple of int): patch size tuple (width, height).
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
          pixels per baseline pixel (baseline).

    Attributes:
        input_img(ndarray, WSIReader): input image for patch extraction.
          input_image type is ndarray for an image tile whereas :obj:`WSIReader`
          for an WSI.
        patch_size(tuple(int)): patch size tuple (width, height).
        resolution(tuple(int)): resolution at which to read the image.
        units (str): the units of resolution.
        n(int): current state of the iterator.
        num_examples_per_patch(int): Number of examples per patch for ensemble
         classification.
        locations_df(pd.DataFrame): A table containing location and/or type of patch.

    """

    def __init__(self, input_img, patch_size, resolution=0, units="level"):
        self.patch_size = patch_size
        self.resolution = resolution
        self.units = units
        self.n = 0
        self.wsi = get_wsireader(input_img=input_img)
        self.num_examples_per_patch = None
        self.locations_df = None

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

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
        stride(tuple(int)): stride in (x, y) direction for patch extraction,
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
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
        )
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride

        self.locations_df, self.num_examples_per_patch = self._generate_location_df()

    def _generate_location_df(self):
        """Generate location list based on slide dimension.
        The slide dimension is calculated using units and resolution.

        """
        level, _ = self.wsi.find_optimal_level_and_downsample(
            resolution=self.resolution, units=self.units
        )
        try:
            level = np.int(level)
            slide_dimension = self.wsi.info.level_dimensions[level]
        except IndexError:
            slide_dimension = self.wsi.info.level_dimensions[0]
            rescale = 2 ** level
            slide_dimension = tuple([int(x / rescale) for x in slide_dimension])

        img_w = slide_dimension[0]
        img_h = slide_dimension[1]
        img_patch_w = self.patch_size[0]
        img_patch_h = self.patch_size[1]
        stride_w = self.stride[0]
        stride_h = self.stride[1]

        num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h + 1)
        num_patches_img_w = math.ceil(((img_w - img_patch_w) / stride_w + 1))
        num_patches_img = num_patches_img_h * num_patches_img_w

        data = []

        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                start_w = w * stride_w
                data.append([start_w, start_h, None])

        locations_df = read_locations(input_table=np.array(data))

        return locations_df, num_patches_img

    def __next__(self):
        raise NotImplementedError

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
        current_location(tuple(int)): current starting point location in
         (x, y) direction.

    """

    def __init__(
        self,
        input_img,
        patch_size,
        resolution=0,
        units="level",
        stride=(1, 1),
        label_patch_size=None,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
        )
        self.stride = stride
        self.label_patch_size = label_patch_size
        self.current_location = (0, 0)

    def __next__(self):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class PointsPatchExtractor(PatchExtractor):
    """Extracting patches with specified points as a centre.

    Args:
        locations_list(ndarray, pd.DataFrame, str, pathlib.Path): contains location
         and/or type of patch. Input can be path to a csv or json file.
        num_examples_per_patch(int): Number of examples per patch for ensemble
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
        num_examples_per_patch=1,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
        )

        self.num_examples_per_patch = num_examples_per_patch
        self.locations_df = read_locations(input_table=locations_list)

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

        x, y, _ = self.locations_df.values[item, :]

        x = x - int((self.patch_size[1] - 1) / 2)
        y = y - int((self.patch_size[0] - 1) / 2)

        data = self.wsi.read_rect(
            location=(int(x), int(y)),
            size=self.patch_size,
            resolution=self.resolution,
            units=self.units,
        )

        return data

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
