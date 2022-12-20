"""This file defines patch extraction methods for deep learning models."""
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
from pandas import DataFrame

from tiatoolbox.utils import misc
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.wsicore import wsireader


class PatchExtractorABC(ABC):
    """Abstract base class for Patch Extraction in tiatoolbox."""

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: int):
        raise NotImplementedError


class PatchExtractor(PatchExtractorABC):
    """Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_img(str, pathlib.Path, :class:`numpy.ndarray`):
            Input image for patch extraction.
        patch_size(int or tuple(int)):
            Patch size tuple (width, height).
        input_mask(str, pathlib.Path, :class:`numpy.ndarray`, or :obj:`WSIReader`):
            Input mask that is used for position filtering when
            extracting patches i.e., patches will only be extracted
            based on the highlighted regions in the input_mask.
            input_mask can be either path to the mask, a numpy array,
            :class:`VirtualWSIReader`, or one of 'otsu' and
            'morphological' options. In case of 'otsu' or
            'morphological', a tissue mask is generated for the
            input_image using tiatoolbox :class:`TissueMasker`
            functionality.
        resolution (int or float or tuple of float):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (str):
            Units of resolution, default = "level". Supported units are:
            microns per pixel (mpp), objective power (power), pyramid /
            resolution level (level), Only pyramid / resolution levels
            (level) embedded in the whole slide image are supported.
        pad_mode (str):
            Method for padding at edges of the WSI. Default to
            'constant'. See :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)):
            Values to use with constant padding. Defaults to 0. See
            :func:`numpy.pad` for more.
        within_bound (bool):
            Whether to extract patches beyond the input_image size
            limits. If False, extracted patches at margins will be
            padded appropriately based on `pad_constant_values` and
            `pad_mode`. If False, patches at the margin that their
            bounds exceed the mother image dimensions would be
            neglected. Default is False.
        min_mask_ratio (float):
            Area in percentage that a patch needs to contain of positive
            mask to be included. Defaults to 0.


    Attributes:
        wsi(WSIReader):
            Input image for patch extraction of type :obj:`WSIReader`.
        patch_size(tuple(int)):
            Patch size tuple (width, height).
        resolution(tuple(int)):
            Resolution at which to read the image.
        units (str):
            Units of resolution.
        n (int):
            Current state of the iterator.
        locations_df (pd.DataFrame):
            A table containing location and/or type of patches in
            `(x_start, y_start, class)` format.
        coordinate_list (:class:`numpy.ndarray`):
            An array containing coordinates of patches in `(x_start,
            y_start, x_end, y_end)` format to be used for
            `slidingwindow` patch extraction.
        pad_mode (str):
            Method for padding at edges of the WSI. See
            :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)):
            Values to use with constant padding. Defaults to 0. See
            :func:`numpy.pad` for more.
        stride (tuple(int)):
            Stride in (x, y) direction for patch extraction. Not used
            for :obj:`PointsPatchExtractor`
        min_mask_ratio (float):
            Only patches with positive area percentage above this value are included

    """

    def __init__(
        self,
        input_img: Union[str, Path, np.ndarray],
        patch_size: Union[int, Tuple[int, int]],
        input_mask: Union[str, Path, np.ndarray, wsireader.WSIReader] = None,
        resolution: Union[int, float, Tuple[float, float]] = 0,
        units: str = "level",
        pad_mode: str = "constant",
        pad_constant_values: Union[int, Tuple[int, int]] = 0,
        within_bound: bool = False,
        min_mask_ratio: float = 0,
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
        self.wsi = wsireader.WSIReader.open(input_img=input_img)
        self.locations_df = None
        self.coordinate_list = None
        self.stride = None

        self.min_mask_ratio = min_mask_ratio

        if input_mask is None:
            self.mask = None
        elif isinstance(input_mask, str) and input_mask in {"otsu", "morphological"}:
            if isinstance(self.wsi, wsireader.VirtualWSIReader):
                self.mask = None
            else:
                self.mask = self.wsi.tissue_mask(
                    method=input_mask, resolution=1.25, units="power"
                )
        elif isinstance(input_mask, wsireader.VirtualWSIReader):
            self.mask = input_mask
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

    def __getitem__(self, item: int):
        if not isinstance(item, int):
            raise TypeError("Index should be an integer.")

        if item >= self.locations_df.shape[0]:
            raise IndexError

        x = self.locations_df["x"][item]
        y = self.locations_df["y"][item]

        return self.wsi.read_rect(
            location=(int(x), int(y)),
            size=self.patch_size,
            resolution=self.resolution,
            units=self.units,
            pad_mode=self.pad_mode,
            pad_constant_values=self.pad_constant_values,
            coord_space="resolution",
        )

    def _generate_location_df(self):
        """Generate location list based on slide dimension.

        The slide dimension is calculated using units and resolution.

        """
        slide_dimension = self.wsi.slide_dimensions(self.resolution, self.units)

        self.coordinate_list = self.get_coordinates(
            image_shape=(slide_dimension[0], slide_dimension[1]),
            patch_input_shape=(self.patch_size[0], self.patch_size[1]),
            stride_shape=(self.stride[0], self.stride[1]),
            input_within_bound=self.within_bound,
        )

        if self.mask is not None:
            # convert the coordinate_list resolution unit to acceptable units
            converted_units = self.wsi.convert_resolution_units(
                input_res=self.resolution,
                input_unit=self.units,
            )
            # find the first unit which is not None
            converted_units = {
                k: v for k, v in converted_units.items() if v is not None
            }
            converted_units_keys = list(converted_units.keys())
            selected_coord_indices = self.filter_coordinates_fast(
                self.mask,
                self.coordinate_list,
                coordinate_resolution=converted_units[converted_units_keys[0]],
                coordinate_units=converted_units_keys[0],
                min_mask_ratio=self.min_mask_ratio,
            )
            self.coordinate_list = self.coordinate_list[selected_coord_indices]
            if len(self.coordinate_list) == 0:
                warnings.warn(
                    "No candidate coordinates left after "
                    "filtering by `input_mask` positions."
                )

        data = self.coordinate_list[:, :2]  # only use the x_start and y_start

        self.locations_df = misc.read_locations(input_table=np.array(data))

        return self

    @staticmethod
    def filter_coordinates_fast(
        mask_reader: wsireader.VirtualWSIReader,
        coordinates_list: np.ndarray,
        coordinate_resolution: float,
        coordinate_units: str,
        mask_resolution: float = None,
        min_mask_ratio: float = 0,
    ):
        """Validate patch extraction coordinates based on the input mask.

        This function indicates which coordinate is valid for mask-based
        patch extraction based on checks in low resolution.

        Args:
            mask_reader (:class:`.VirtualReader`):
                A virtual pyramidal reader of the mask related to the
                WSI from which we want to extract the patches.
            coordinates_list (ndarray and np.int32):
                Coordinates to be checked via the `func`. They must be
                at the same resolution as requested `resolution` and
                `units`. The shape of `coordinates_list` is (N, K) where
                N is the number of coordinate sets and K is either 2 for
                centroids or 4 for bounding boxes. When using the
                default `func=None`, K should be 4, as we expect the
                `coordinates_list` to be bounding boxes in `[start_x,
                start_y, end_x, end_y]` format.
            coordinate_resolution (float):
                Resolution value at which `coordinates_list` is
                generated.
            coordinate_units (str):
                Resolution unit at which `coordinates_list` is generated.
            mask_resolution (float):
                Resolution at which mask array is extracted. It is
                supposed to be in the same units as `coord_resolution`
                i.e., `coordinate_units`. If not provided, a default
                value will be selected based on `coordinate_units`.
            min_mask_ratio (float):
                Only patches with positive area percentage above this value are
                included. Defaults to 0.

        Returns:
            :class:`numpy.ndarray`:
                list of flags to indicate which coordinate is valid.

        """
        if not isinstance(mask_reader, wsireader.VirtualWSIReader):
            raise ValueError("`mask_reader` should be wsireader.VirtualWSIReader.")
        if not isinstance(coordinates_list, np.ndarray) or not np.issubdtype(
            coordinates_list.dtype, np.integer
        ):
            raise ValueError("`coordinates_list` should be ndarray of integer type.")
        if coordinates_list.shape[-1] != 4:
            raise ValueError("`coordinates_list` must be of shape [N, 4].")
        if isinstance(coordinate_resolution, (int, float)):
            coordinate_resolution = [coordinate_resolution, coordinate_resolution]

        if not 0 <= min_mask_ratio <= 1:
            raise ValueError("`min_mask_ratio` must be between 0 and 1.")

        # define default mask_resolution based on the input `coordinate_units`
        if mask_resolution is None:
            mask_res_dict = {"mpp": 8, "power": 1.25, "baseline": 0.03125}
            mask_resolution = mask_res_dict[coordinate_units]

        tissue_mask = mask_reader.slide_thumbnail(
            resolution=mask_resolution, units=coordinate_units
        )

        # Scaling the coordinates_list to the `tissue_mask` array resolution
        scaled_coords = coordinates_list.copy().astype(np.float32)
        scaled_coords[:, [0, 2]] *= coordinate_resolution[0] / mask_resolution
        scaled_coords[:, [0, 2]] = np.clip(
            scaled_coords[:, [0, 2]], 0, tissue_mask.shape[1]
        )
        scaled_coords[:, [1, 3]] *= coordinate_resolution[1] / mask_resolution
        scaled_coords[:, [1, 3]] = np.clip(
            scaled_coords[:, [1, 3]], 0, tissue_mask.shape[0]
        )
        scaled_coords = list(np.int32(scaled_coords))

        flag_list = []
        for coord in scaled_coords:
            this_part = tissue_mask[coord[1] : coord[3], coord[0] : coord[2]]
            patch_area = np.prod(this_part.shape)
            pos_area = np.count_nonzero(this_part)

            if (
                (pos_area == patch_area) or (pos_area > patch_area * min_mask_ratio)
            ) and (pos_area > 0 and patch_area > 0):
                flag_list.append(True)
            else:
                flag_list.append(False)
        return np.array(flag_list)

    @staticmethod
    def filter_coordinates(
        mask_reader: wsireader.VirtualWSIReader,
        coordinates_list: np.ndarray,
        func: Callable = None,
        resolution: float = None,
        units: str = None,
    ):
        """Indicates which coordinate is valid for mask-based patch extraction.

        Locations are validated by a custom or default filter `func`.

        Args:
            mask_reader (:class:`.VirtualReader`):
                A virtual pyramidal reader of the mask related to the
                WSI from which we want to extract the patches.
            coordinates_list (ndarray and np.int32):
                Coordinates to be checked via the `func`. They must be
                in the same resolution as requested `resolution` and
                `units`. The shape of `coordinates_list` is (N, K) where
                N is the number of coordinate sets and K is either 2 for
                centroids or 4 for bounding boxes. When using the
                default `func=None`, K should be 4, as we expect the
                `coordinates_list` to refer to bounding boxes in
                `[start_x, start_y, end_x, end_y]` format.
            func:
                The coordinate validator function. A function that takes
                `reader` and `coordinate` as arguments and return True
                or False as indication of coordinate validity.
            resolution (float):
                The resolution value at which coordinates_list are
                generated.
            units (str):
                The resolution unit at which coordinates_list are
                generated.

        Returns:
            :class:`numpy.ndarray`:
                List of flags to indicate which coordinates are valid.

        """

        def default_sel_func(reader: wsireader.VirtualWSIReader, coord: np.ndarray):
            """Accept coord as long as its box contains bits of mask."""
            roi = reader.read_bounds(
                coord,
                resolution=reader.info.mpp if resolution is None else resolution,
                units="mpp" if units is None else units,
                interpolation="nearest",
                coord_space="resolution",
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
                f"Default `func` does not support "
                f"`coordinates_list` of shape {coordinates_list.shape}."
            )
        func = default_sel_func if func is None else func
        flag_list = [func(mask_reader, coord) for coord in coordinates_list]
        return np.array(flag_list)

    @staticmethod
    def get_coordinates(
        image_shape: Union[Tuple[int, int], np.ndarray] = None,
        patch_input_shape: Union[Tuple[int, int], np.ndarray] = None,
        patch_output_shape: Union[Tuple[int, int], np.ndarray] = None,
        stride_shape: Union[Tuple[int, int], np.ndarray] = None,
        input_within_bound: bool = False,
        output_within_bound: bool = False,
    ):
        """Calculate patch tiling coordinates.

        Args:
            image_shape (tuple (int, int) or :class:`numpy.ndarray`):
                This argument specifies the shape of mother image (the
                image we want to extract patches from) at requested
                `resolution` and `units` and it is expected to be in
                (width, height) format.
            patch_input_shape (tuple (int, int) or :class:`numpy.ndarray`):
                Specifies the input shape of requested patches to be
                extracted from mother image at desired `resolution` and
                `units`. This argument is also expected to be in (width,
                height) format.
            patch_output_shape (tuple (int, int) or :class:`numpy.ndarray`):
                Specifies the output shape of requested patches to be
                extracted from mother image at desired `resolution` and
                `units`. This argument is also expected to be in (width,
                height) format. If this is not provided,
                `patch_output_shape` will be the same as
                `patch_input_shape`.
            stride_shape (tuple (int, int) or :class:`numpy.ndarray`):
                The stride that is used to calculate the patch location
                during the patch extraction. If `patch_output_shape` is
                provided, next stride location will base on the output
                rather than the input.
            input_within_bound (bool):
                Whether to include the patches where their `input`
                location exceed the margins of mother image. If `True`,
                the patches with input location exceeds the
                `image_shape` would be neglected. Otherwise, those
                patches would be extracted with `Reader` function and
                appropriate padding.
            output_within_bound (bool):
                Whether to include the patches where their `output`
                location exceed the margins of mother image. If `True`,
                the patches with output location exceeds the
                `image_shape` would be neglected. Otherwise, those
                patches would be extracted with `Reader` function and
                appropriate padding.

        Return:
            coord_list:
                A list of coordinates in `[start_x, start_y, end_x,
                end_y]` format to be used for patch extraction.

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
            """Tests if the shape is valid for an image."""
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
        input_img(str, pathlib.Path, :class:`numpy.ndarray`):
            Input image for patch extraction.
        patch_size(int or tuple(int)):
            Patch size tuple (width, height).
        input_mask(str, pathlib.Path, :class:`numpy.ndarray`, or :obj:`WSIReader`):
            Input mask that is used for position filtering when
            extracting patches i.e., patches will only be extracted
            based on the highlighted regions in the `input_mask`.
            `input_mask` can be either path to the mask, a numpy array,
            :class:`VirtualWSIReader`, or one of 'otsu' and
            'morphological' options. In case of 'otsu' or
            'morphological', a tissue mask is generated for the
            input_image using tiatoolbox :class:`TissueMasker`
            functionality.
        resolution (int or float or tuple of float):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (str):
            The units of resolution, default = "level". Supported units
            are: microns per pixel (mpp), objective power (power),
            pyramid / resolution level (level), Only pyramid /
            resolution levels (level) embedded in the whole slide image
            are supported.
        pad_mode (str):
            Method for padding at edges of the WSI. Default to
            'constant'. See :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)):
            Values to use with constant padding. Defaults to 0.
            See :func:`numpy.pad` for more information.
        within_bound (bool):
            Whether to extract patches beyond the input_image size
            limits. If False, extracted patches at margins will be
            padded appropriately based on `pad_constant_values` and
            `pad_mode`. If False, patches at the margin that their
            bounds exceed the mother image dimensions would be
            neglected. Default is False.
        stride(int or tuple(int)):
            Stride in (x, y) direction for patch extraction, default =
            `patch_size`.
        min_mask_ratio (float):
            Only patches with positive area percentage above this value are included.
            Defaults to 0.

    Attributes:
        stride(tuple(int)):
            Stride in (x, y) direction for patch extraction.

    """

    def __init__(
        self,
        input_img: Union[str, Path, np.ndarray],
        patch_size: Union[int, Tuple[int, int]],
        input_mask: Union[str, Path, np.ndarray, wsireader.WSIReader] = None,
        resolution: Union[int, float, Tuple[float, float]] = 0,
        units: str = "level",
        stride: Union[int, Tuple[int, int]] = None,
        pad_mode: str = "constant",
        pad_constant_values: Union[int, Tuple[int, int]] = 0,
        within_bound: bool = False,
        min_mask_ratio: float = 0,
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
            min_mask_ratio=min_mask_ratio,
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
        input_img(str, pathlib.Path, :class:`numpy.ndarray`):
            Input image for patch extraction.
        locations_list(ndarray, pd.DataFrame, str, pathlib.Path):
            Contains location and/or type of patch. This can be path to
            csv, npy or json files. Input can also be a
            :class:`numpy.ndarray` or :class:`pandas.DataFrame`. NOTE:
            value of location $(x,y)$ is expected to be based on the
            specified `resolution` and `units` (not the `'baseline'`
            resolution).
        patch_size(int or tuple(int)):
            Patch size tuple (width, height).
        resolution (int or float or tuple of float):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (str):
            The units of resolution, default = "level". Supported units
            are: microns per pixel (mpp), objective power (power),
            pyramid / resolution level (level), Only pyramid /
            resolution levels (level) embedded in the whole slide image
            are supported.
        pad_mode (str):
            Method for padding at edges of the WSI. Default to
            'constant'. See :func:`numpy.pad` for more information.
        pad_constant_values (int or tuple(int)): Values to use with
            constant padding. Defaults to 0. See :func:`numpy.pad` for
            more.
        within_bound (bool):
            Whether to extract patches beyond the input_image size
            limits. If False, extracted patches at margins will be
            padded appropriately based on `pad_constant_values` and
            `pad_mode`. If False, patches at the margin that their
            bounds exceed the mother image dimensions would be
            neglected. Default is False.

    """

    def __init__(
        self,
        input_img: Union[str, Path, np.ndarray],
        locations_list: Union[np.ndarray, DataFrame, str, Path],
        patch_size: Union[int, Tuple[int, int]] = (224, 224),
        resolution: Union[int, float, Tuple[float, float]] = 0,
        units: str = "level",
        pad_mode: str = "constant",
        pad_constant_values: Union[int, Tuple[int, int]] = 0,
        within_bound: bool = False,
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


def get_patch_extractor(method_name: str, **kwargs: str):
    """Return a patch extractor object as requested.

    Args:
        method_name (str):
            Name of patch extraction method, must be one of "point" or
            "slidingwindow". The method name is case-insensitive.
        **kwargs:
            Keyword arguments passed to :obj:`PatchExtractor`.

    Returns:
        PatchExtractor:
            An object with base :obj:`PatchExtractor` as base class.

    Examples:
        >>> from tiatoolbox.tools.patchextraction import get_patch_extractor
        >>> # PointsPatchExtractor with default values
        >>> patch_extract = get_patch_extractor(
        ...  'point', img_patch_h=200, img_patch_w=200)

    """
    if method_name.lower() not in ["point", "slidingwindow"]:
        raise MethodNotSupported(
            f"{method_name.lower()} method is not currently supported."
        )

    if method_name.lower() == "point":
        return PointsPatchExtractor(**kwargs)

    return SlidingWindowPatchExtractor(**kwargs)
