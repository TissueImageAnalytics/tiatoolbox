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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This file defines patch extraction methods for deep learning models."""
from abc import ABC
import numpy as np

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
        input_mask(str, pathlib.Path, :class:`numpy.ndarray`, or :obj:`WSIReader`):
          input mask that is used for position filtering when extracting patches
          i.e., patches will only be extracted based on the highlighted regions in
          the input_mask. input_mask can be either path to the mask, a numpy
          array, :class:`VirtualWSIReader`, or one of 'otsu' and 'morphological'
          options. In case of 'otsu' or 'morphological', a tissue mask is generated
          for the input_image using tiatoolbox :class:`TissueMasker` functionality.
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
        within_bound (bool): whether to extract patches beyond the
          input_image size limits. If False, extracted patches at margins
          will be padded appropriately based on `pad_constant_values` and
          `pad_mode`. If False, patches at the margin that their bounds
          exceed the mother image dimensions would be neglected.
          Default is False.

    Attributes:
        wsi(WSIReader): input image for patch extraction of type :obj:`WSIReader`.
        patch_size(tuple(int)): patch size tuple (width, height).
        resolution(tuple(int)): resolution at which to read the image.
        units (str): the units of resolution.
        n(int): current state of the iterator.
        locations_df(pd.DataFrame): A table containing location and/or type of patces
          in `(x_start, y_start, class)` format.
        coord_list(:class:`numpy.ndarray`): An array containing coordinates of patches
          in `(x_start, y_start, x_end, y_end)` format to be used for `slidingwindow`
          patch extraction.
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
        input_mask=None,
        resolution=0,
        units="level",
        pad_mode="constant",
        pad_constant_values=0,
        within_bound=False,
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
        self.coord_list = None
        self.stride = None
        if input_mask is None:
            self.mask = None
        elif isinstance(input_mask, str) and input_mask in {"otsu", "morphological"}:
            if isinstance(self.wsi, wsireader.VirtualWSIReader):
                self.mask = None
            else:
                self.mask = self.wsi.tissue_mask(
                    method=input_mask, resolution=1.25, units="power"
                )
        else:
            self.mask = wsireader.VirtualWSIReader(
                input_mask, info=self.wsi.info, mode="bool"
            )
        self.within_bound = within_bound

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
        stride_w = int(self.stride[0] * level_downsample)
        stride_h = int(self.stride[1] * level_downsample)

        self.coord_list = self.get_coordinates(
            image_shape=(img_w, img_h),
            patch_input_shape=(img_patch_w, img_patch_h),
            stride_shape=(stride_w, stride_h),
            input_within_bound=self.within_bound,
        )

        if self.mask is not None:
            selected_coord_idxs = self.filter_coordinates(
                self.mask,
                self.coord_list,
                resolution=self.resolution,
                units=self.units,
            )
            self.coord_list = self.coord_list[selected_coord_idxs]

            if len(self.coord_list) == 0:
                raise ValueError(
                    "No candidate coordinates left after "
                    "filtering by input_mask positions."
                )

        data = self.coord_list[:, :2]  # only use the x_start and y_start

        self.locations_df = misc.read_locations(input_table=np.array(data))

        return self

    @staticmethod
    def filter_coordinates(
        mask_reader, coordinates_list, func=None, resolution=None, units=None
    ):
        """
        Indicates which coordinate is valid for mask-based patch extraction.
        Locations are being validated by a custom or build-in `func`.

        Args:
            mask_reader (:class:`.VirtualReader`): a virtual pyramidal reader of the
              mask related to the WSI from which we want to extract the patches.
            coordinates_list (ndarray and np.int32): Coordinates to be checked
              via the `func`. They must be in the same resolution as requested
              `resolution` and `units`. The shape of `coordinates_list` is (N, K)
              where N is the number of coordinate sets and K is either 2 for centroids
              or 4 for bounding boxes. When using the default `func=None`, K should be
              4, as we expect the `coordinates_list` to be refer to bounding boxes in
              `[start_x, start_y, end_x, end_y]` format.
            func: The coordinate validator function. A function that takes `reader` and
              `coordinate` as arguments and return True or False as indication of
              coordinate validity.

        Returns:
            ndarray: list of flags to indicate which coordinate is valid.
        """

        def default_sel_func(reader: wsireader.VirtualWSIReader, coord: np.ndarray):
            """Accept coord as long as its box contains bits of mask."""
            roi = reader.read_bounds(
                coord,
                resolution=reader.info.mpp if resolution is None else resolution,
                units="mpp" if units is None else units,
                interpolation="nearest",
            )
            return np.sum(roi > 0) > 0

        if not isinstance(mask_reader, wsireader.VirtualWSIReader):
            raise ValueError("`mask_reader` should be wsireader.VirtualWSIReader.")
        if not isinstance(coordinates_list, np.ndarray) or not np.issubdtype(
            coordinates_list.dtype, np.integer
        ):
            raise ValueError("`coordinates_list` should be ndarray of integer type.")
        if func is None and coordinates_list.shape[-1] != 4:
            raise ValueError(
                "Default `func` does not support "
                "`coordinates_list` of shape {}.".format(coordinates_list.shape)
            )
        func = default_sel_func if func is None else func
        flag_list = [func(mask_reader, coord) for coord in coordinates_list]
        return np.array(flag_list)

    @staticmethod
    def get_coordinates(
        image_shape=None,
        patch_input_shape=None,
        patch_output_shape=None,
        stride_shape=None,
        input_within_bound=False,
        output_within_bound=False,
    ):
        """Calculate patch tiling coordinates.

        Args:
            image_shape (a tuple (int, int) or :class:`numpy.ndarray` of shape (2,)):
              This argument specifies the shape of mother image (the image we want to)
              extract patches from) at requested `resolution` and `units` and it is
              expected to be in (width, height) format.
            patch_input_shape (a tuple (int, int) or
              :class:`numpy.ndarray` of shape (2,)): Specifies the input shape of
              requested patches to be extracted from mother image at desired
              `resolution` and `units`. This argument is also expected to be in
              (width, height) format.
            patch_output_shape (a tuple (int, int) or
              :class:`numpy.ndarray` of shape (2,)): Specifies the output shape of
              requested patches to be extracted from mother image at desired
              `resolution` and `units`. This argument is also expected to be in
              (width, height) format. If this is not provided, `patch_output_shape`
              will be the same as `patch_input_shape`.
            stride_shape (a tuple (int, int) or :class:`numpy.ndarray` of shape (2,)):
              The stride that is used to calcualte the patch location during the patch
              extraction. If `patch_output_shape` is provided, next stride location
              will base on the output rather than the input.
            input_within_bound (bool): Whether to include the patches where their
              `input` location exceed the margins of mother image. If `True`, the
              patches with input location exceeds the `image_shape` would be
              neglected. Otherwise, those patches would be extracted with `Reader`
              function and appropriate padding.
            output_within_bound (bool): Whether to include the patches where their
              `output` location exceed the margins of mother image. If `True`, the
              patches with output location exceeds the `image_shape` would be
              neglected. Otherwise, those patches would be extracted with `Reader`
              function and appropriate padding.

        Return:
            coord_list: a list of corrdinates in `[start_x, start_y, end_x, end_y]`
            format to be used for patch extraction.

        """
        return_output_bound = patch_output_shape is not None
        image_shape = np.array(image_shape)
        patch_input_shape = np.array(patch_input_shape)
        if patch_output_shape is None:
            output_within_bound = False
            patch_output_shape = patch_input_shape
        patch_output_shape = np.array(patch_output_shape)
        stride_shape = np.array(stride_shape)

        def validate_shape(shape):
            return (
                not np.issubdtype(shape.dtype, np.integer)
                or np.size(shape) > 2
                or np.any(shape < 0)
            )

        if validate_shape(image_shape):
            raise ValueError(f"Invalid `image_shape` value {image_shape}.")
        if validate_shape(patch_input_shape):
            raise ValueError(f"Invalid `patch_input_shape` value {patch_input_shape}.")
        if validate_shape(patch_output_shape):
            raise ValueError(
                f"Invalid `patch_output_shape` value {patch_output_shape}."
            )
        if validate_shape(stride_shape):
            raise ValueError(f"Invalid `stride_shape` value {stride_shape}.")
        if np.any(patch_input_shape < patch_output_shape):
            raise ValueError(
                (
                    f"`patch_input_shape` must larger than `patch_output_shape`"
                    f" {patch_input_shape} must > {patch_output_shape}."
                )
            )
        if np.any(stride_shape < 1):
            raise ValueError(f"`stride_shape` value {stride_shape} must > 1.")

        def flat_mesh_grid_coord(x, y):
            """Helper function to obtain coordinate grid."""
            x, y = np.meshgrid(x, y)
            return np.stack([x.flatten(), y.flatten()], axis=-1)

        output_x_end = (
            np.ceil(image_shape[0] / patch_output_shape[0]) * patch_output_shape[0]
        )
        output_x_list = np.arange(0, int(output_x_end), stride_shape[0])
        output_y_end = (
            np.ceil(image_shape[1] / patch_output_shape[1]) * patch_output_shape[1]
        )
        output_y_list = np.arange(0, int(output_y_end), stride_shape[1])
        output_tl_list = flat_mesh_grid_coord(output_x_list, output_y_list)
        output_br_list = output_tl_list + patch_output_shape[None]

        io_diff = patch_input_shape - patch_output_shape
        input_tl_list = output_tl_list - (io_diff // 2)[None]
        input_br_list = input_tl_list + patch_input_shape[None]

        sel = np.zeros(input_tl_list.shape[0], dtype=bool)
        if output_within_bound:
            sel |= np.any(output_br_list > image_shape[None], axis=1)
        if input_within_bound:
            sel |= np.any(input_br_list > image_shape[None], axis=1)
            sel |= np.any(input_tl_list < 0, axis=1)
        ####
        input_bound_list = np.concatenate(
            [input_tl_list[~sel], input_br_list[~sel]], axis=-1
        )
        output_bound_list = np.concatenate(
            [output_tl_list[~sel], output_br_list[~sel]], axis=-1
        )
        if return_output_bound:
            return input_bound_list, output_bound_list
        return input_bound_list


class SlidingWindowPatchExtractor(PatchExtractor):
    """Extract patches using sliding fixed sized window for images and labels.

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
        input_mask=None,
        resolution=0,
        units="level",
        stride=None,
        pad_mode="constant",
        pad_constant_values=0,
        within_bound=False,
    ):
        super().__init__(
            input_img=input_img,
            input_mask=input_mask,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            within_bound=within_bound,
        )
        if stride is None:
            self.stride = self.patch_size
        else:
            if isinstance(stride, (tuple, list)):
                self.stride = (int(stride[0]), int(stride[1]))
            else:
                self.stride = (int(stride), int(stride))

        self._generate_location_df()


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
        within_bound=False,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            within_bound=within_bound,
        )

        self.locations_df = misc.read_locations(input_table=locations_list)
        self.locations_df["x"] = self.locations_df["x"] - int(
            (self.patch_size[1] - 1) / 2
        )
        self.locations_df["y"] = self.locations_df["y"] - int(
            (self.patch_size[1] - 1) / 2
        )


def get_patch_extractor(method_name, **kwargs):
    """Return a patch extractor object as requested.

    Args:
        method_name (str): name of patch extraction method, must be one of "point" or
          "slidingwindow".
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
    elif method_name.lower() == "slidingwindow":
        patch_extractor = SlidingWindowPatchExtractor(**kwargs)
    else:
        raise MethodNotSupported

    return patch_extractor
