"""This file defines patch extraction methods for deep learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, TypedDict, overload

import numpy as np
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.utils import misc
from tiatoolbox.utils.exceptions import FileNotSupportedError, MethodNotSupportedError
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import wsireader

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from pandas import DataFrame

    from tiatoolbox.annotation.storage import AnnotationStore
    from tiatoolbox.typing import Resolution, Units


def validate_shape(shape: np.ndarray) -> bool:
    """Test if the shape is valid for an image."""
    return (
        not np.issubdtype(shape.dtype, np.integer)
        or np.size(shape) > 2  # noqa: PLR2004
        or bool(np.any(shape < 0))
    )


class ExtractorParams(TypedDict, total=False):
    """A subclass of TypedDict.

    Defines the types of the keyword arguments passed into 'get_patch_extractor'.

    """

    input_img: str | Path | np.ndarray | wsireader.WSIReader
    locations_list: np.ndarray | DataFrame | str | Path
    patch_size: int | tuple[int, int]
    resolution: Resolution
    units: Units
    pad_mode: str
    pad_constant_values: int | tuple[int, int]
    within_bound: bool
    input_mask: (
        str | Path | np.ndarray | wsireader.VirtualWSIReader | AnnotationStore | None
    )
    stride: int | tuple[int, int]
    min_mask_ratio: float
    store_filter: str | None


class PointsPatchExtractorParams(TypedDict):
    """A subclass of TypedDict.

    Defines the types of the keyword arguments passed to PointsPatchExtractor.

    """

    input_img: str | Path | np.ndarray | wsireader.WSIReader
    locations_list: np.ndarray | DataFrame | str | Path
    patch_size: int | tuple[int, int]
    resolution: Resolution
    units: Units
    pad_mode: str
    pad_constant_values: int | tuple[int, int]
    within_bound: bool


class SlidingWindowPatchExtractorParams(TypedDict):
    """A subclass of TypedDict.

    Defines the types of the keyword arguments passed to SlidingWindowPatchExtractor.

    """

    input_img: str | Path | np.ndarray | wsireader.WSIReader
    patch_size: int | tuple[int, int]
    resolution: Resolution
    units: Units
    pad_mode: str
    pad_constant_values: int | tuple[int, int]
    within_bound: bool
    input_mask: (
        str | Path | np.ndarray | wsireader.VirtualWSIReader | AnnotationStore | None
    )
    stride: int | tuple[int, int] | None
    min_mask_ratio: float
    store_filter: str | None


class PatchExtractorABC(ABC):
    """Abstract base class for Patch Extraction in tiatoolbox."""

    @abstractmethod
    def __iter__(self: PatchExtractorABC) -> PatchExtractor:
        """Return an iterator for the given object."""
        raise NotImplementedError

    @abstractmethod
    def __next__(self: PatchExtractorABC) -> np.ndarray:
        """Return the next item for the iteration."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self: PatchExtractorABC, item: int) -> np.ndarray:
        """Get an item from the dataset."""
        raise NotImplementedError


class PatchExtractor(PatchExtractorABC):
    """Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_img(str, Path, :class:`numpy.ndarray`, :class:`WSIReader`):
            Input image for patch extraction.
        patch_size(int or tuple(int)):
            Patch size tuple (width, height).
        input_mask
            (str, pathlib.Path, :class:`numpy.ndarray`, or :obj:`VirtualWSIReader`):
            Input mask that is used for position filtering when
            extracting patches i.e., patches will only be extracted
            based on the highlighted regions in the input_mask.
            input_mask can be either path to the mask, a numpy array,
            :class:`VirtualWSIReader`, or one of 'otsu' and
            'morphological' options. In case of 'otsu' or
            'morphological', a tissue mask is generated for the
            input_image using tiatoolbox :class:`TissueMasker`
            functionality. May also be an annotation store, in which case the
            mask is generated based on the annotations. All annotations are used by
            default; the 'store_filter' argument can be used to specify a filter for
            a subset of annotations to use to build the mask.
        resolution (Resolution):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (Units):
            Units of resolution, default = "level".
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
            `pad_mode`. If True, patches at the margins whose
            bounds would exceed the mother image dimensions would be
            neglected. Default is False.
        min_mask_ratio (float):
            Area in percentage that a patch needs to contain of positive
            mask to be included. Defaults to 0.
        store_filter (str):
            Filter to apply to the annotations when generating the mask. Default is
            None, which uses all annotations. Only used if the provided mask is an
            annotation store.


    Attributes:
        wsi (WSIReader):
            Input image for patch extraction of type :obj:`WSIReader`.
        patch_size (tuple(int)):
            Patch size tuple (width, height).
        resolution (Resolution):
            Resolution at which to read the image.
        units (Units):
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
        self: PatchExtractor,
        input_img: str | Path | np.ndarray | wsireader.WSIReader,
        patch_size: int | tuple[int, int],
        input_mask: str
        | Path
        | np.ndarray
        | wsireader.VirtualWSIReader
        | AnnotationStore
        | None = None,
        resolution: Resolution = 0,
        units: Units = "level",
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        min_mask_ratio: float = 0,
        store_filter: str | None = None,
        *,
        within_bound: bool = False,
    ) -> None:
        """Initialize :class:`PatchExtractor`."""
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
        self.locations_df: DataFrame
        self.coordinate_list: np.ndarray
        self.stride: tuple[int, int]

        self.min_mask_ratio = min_mask_ratio

        if input_mask is None:
            self.mask = None
        elif isinstance(input_mask, str) and input_mask.endswith(".db"):
            # input_mask is an annotation store
            renderer = AnnotationRenderer(
                max_scale=10000, edge_thickness=0, where=store_filter
            )
            rendered_mask = wsireader.AnnotationStoreReader(
                input_mask,
                renderer=renderer,
                info=self.wsi.info,
            ).slide_thumbnail()
            rendered_mask = rendered_mask[:, :, 0] == 0
            self.mask = wsireader.VirtualWSIReader(
                rendered_mask,
                info=self.wsi.info,
                mode="bool",
            )
        elif isinstance(input_mask, str) and input_mask in {"otsu", "morphological"}:
            if isinstance(self.wsi, wsireader.VirtualWSIReader):
                self.mask = None
            else:
                self.mask = self.wsi.tissue_mask(
                    method=input_mask,
                    resolution=1.25,
                    units="power",
                )
        elif isinstance(input_mask, wsireader.VirtualWSIReader):
            self.mask = input_mask
        else:
            self.mask = wsireader.VirtualWSIReader(
                input_mask,
                info=self.wsi.info,
                mode="bool",
            )
        self.within_bound = within_bound

    def __iter__(self: PatchExtractor) -> PatchExtractor:
        """Return an iterator for the given object."""
        self.n = 0
        return self

    def __len__(self: PatchExtractor) -> int:
        """Return the number of patches in the extractor."""
        return self.locations_df.shape[0] if self.locations_df is not None else 0

    def __next__(self: PatchExtractor) -> np.ndarray:
        """Return the next item for the iteration."""
        n = self.n

        if n >= self.locations_df.shape[0]:
            raise StopIteration
        self.n = n + 1
        return self[n]

    def __getitem__(self: PatchExtractor, item: int) -> np.ndarray:
        """Get an item from the dataset."""
        if not isinstance(item, int):
            msg = "Index should be an integer."
            raise TypeError(msg)

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

    def _generate_location_df(self: PatchExtractor) -> PatchExtractor:
        """Generate location list based on slide dimension.

        The slide dimension is calculated using units and resolution.

        """
        slide_dimension = self.wsi.slide_dimensions(self.resolution, self.units)

        self.coordinate_list = self.get_coordinates(
            patch_output_shape=None,
            image_shape=(slide_dimension[0], slide_dimension[1]),
            patch_input_shape=(self.patch_size[0], self.patch_size[1]),
            stride_shape=(self.stride[0], self.stride[1]),
            input_within_bound=self.within_bound,
        )

        if self.mask is not None:
            selected_coord_indices = self.filter_coordinates(
                self.mask,
                self.coordinate_list,
                wsi_shape=slide_dimension,
                min_mask_ratio=self.min_mask_ratio,
            )
            self.coordinate_list = self.coordinate_list[selected_coord_indices]
            if len(self.coordinate_list) == 0:
                logger.warning(
                    "No candidate coordinates left after "
                    "filtering by `input_mask` positions.",
                    stacklevel=2,
                )

        data = self.coordinate_list[:, :2]  # only use the x_start and y_start

        self.locations_df = misc.read_locations(input_table=np.array(data))

        return self

    @staticmethod
    def filter_coordinates(
        mask_reader: wsireader.VirtualWSIReader,
        coordinates_list: np.ndarray,
        wsi_shape: tuple[int, int],
        min_mask_ratio: float = 0,
        func: Callable | None = None,
    ) -> np.ndarray:
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
            wsi_shape (tuple(int, int)):
                Shape of the WSI in the requested `resolution` and `units`.
            min_mask_ratio (float):
                Only patches with positive area percentage above this value are
                included. Defaults to 0. Has no effect if `func` is not `None`.
            func (callable):
                Function to be used to validate the coordinates. The function
                must take a `numpy.ndarray` of the mask and a `numpy.ndarray`
                of the coordinates as input and return a bool indicating
                whether the coordinate is valid or not. If `None`, a default
                function that accepts patches with positive area proportion above
                `min_mask_ratio` is used.


        Returns:
            :class:`numpy.ndarray`:
                list of flags to indicate which coordinate is valid.

        """
        if not isinstance(mask_reader, wsireader.VirtualWSIReader):
            msg = "`mask_reader` should be wsireader.VirtualWSIReader."
            raise TypeError(msg)
        if not isinstance(coordinates_list, np.ndarray) or not np.issubdtype(
            coordinates_list.dtype,
            np.integer,
        ):
            msg = "`coordinates_list` should be ndarray of integer type."
            raise ValueError(msg)
        if coordinates_list.shape[-1] != 4:  # noqa: PLR2004
            msg = "`coordinates_list` must be of shape [N, 4]."
            raise ValueError(msg)

        if not 0 <= min_mask_ratio <= 1:
            msg = "`min_mask_ratio` must be between 0 and 1."
            raise ValueError(msg)

        # the tissue mask exists in the reader already, no need to generate it
        tissue_mask = mask_reader.img

        # Scaling the coordinates_list to the `tissue_mask` array resolution
        scale_factors = np.array(tissue_mask.shape[1::-1]) / np.array(wsi_shape)
        scaled_coords = coordinates_list.copy().astype(np.float32)
        scaled_coords[:, [0, 2]] *= scale_factors[0]
        scaled_coords[:, [0, 2]] = np.clip(
            scaled_coords[:, [0, 2]],
            0,
            tissue_mask.shape[1],
        )
        scaled_coords[:, [1, 3]] *= scale_factors[1]
        scaled_coords[:, [1, 3]] = np.clip(
            scaled_coords[:, [1, 3]],
            0,
            tissue_mask.shape[0],
        )
        scaled_coords_list = list((scaled_coords).astype(np.int32))

        def default_sel_func(
            tissue_mask: np.ndarray,
            coord: tuple[int, ...] | list[int],
        ) -> bool:
            """Default selection function to filter coordinates.

            This function selects a coordinate if the proportion of
            positive mask in the corresponding patch is greater than
            `min_mask_ratio`.

            """
            this_part = tissue_mask[coord[1] : coord[3], coord[0] : coord[2]]
            patch_area = int(np.prod(this_part.shape))
            pos_area = int(np.count_nonzero(this_part))
            return (
                (pos_area == patch_area) or (pos_area > patch_area * min_mask_ratio)
            ) and (pos_area > 0 and patch_area > 0)

        func = default_sel_func if func is None else func
        flag_list = [func(tissue_mask, coord) for coord in scaled_coords_list]

        return np.array(flag_list)

    @overload
    @staticmethod
    def get_coordinates(  # pragma: no cover
        patch_output_shape: None = None,
        image_shape: tuple[int, int] | np.ndarray | None = None,
        patch_input_shape: tuple[int, int] | np.ndarray | None = None,
        stride_shape: tuple[int, int] | np.ndarray | None = None,
        *,
        input_within_bound: bool = False,
        output_within_bound: bool = False,
    ) -> np.ndarray: ...

    @overload
    @staticmethod
    def get_coordinates(  # pragma: no cover
        patch_output_shape: tuple[int, int] | np.ndarray,
        image_shape: tuple[int, int] | np.ndarray | None = None,
        patch_input_shape: tuple[int, int] | np.ndarray | None = None,
        stride_shape: tuple[int, int] | np.ndarray | None = None,
        *,
        input_within_bound: bool = False,
        output_within_bound: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def get_coordinates(
        patch_output_shape: tuple[int, int] | np.ndarray | None = None,
        image_shape: tuple[int, int] | np.ndarray | None = None,
        patch_input_shape: tuple[int, int] | np.ndarray | None = None,
        stride_shape: tuple[int, int] | np.ndarray | None = None,
        *,
        input_within_bound: bool = False,
        output_within_bound: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        image_shape_arr = np.array(image_shape)
        patch_input_shape_arr = np.array(patch_input_shape)
        if patch_output_shape is None:
            output_within_bound = False
            patch_output_shape_arr = patch_input_shape_arr
        else:
            patch_output_shape_arr = np.array(patch_output_shape)
        stride_shape_arr = np.array(stride_shape)

        if validate_shape(image_shape_arr):
            msg = f"Invalid `image_shape` value {image_shape_arr}."
            raise ValueError(msg)
        if validate_shape(patch_input_shape_arr):
            msg = f"Invalid `patch_input_shape` value {patch_input_shape_arr}."
            raise ValueError(msg)
        if validate_shape(patch_output_shape_arr):
            msg = f"Invalid `patch_output_shape` value {patch_output_shape_arr}."
            raise ValueError(msg)
        if validate_shape(stride_shape_arr):
            msg = f"Invalid `stride_shape` value {stride_shape_arr}."
            raise ValueError(msg)

        if np.any(patch_input_shape_arr < patch_output_shape_arr):
            msg = (
                f"`patch_input_shape` must larger than `patch_output_shape` "
                f"{patch_input_shape_arr} must > {patch_output_shape_arr}."
            )
            raise ValueError(msg)
        if np.any(stride_shape_arr < 1):
            msg = f"`stride_shape` value {stride_shape_arr} must > 1."
            raise ValueError(msg)

        def flat_mesh_grid_coord(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Helper function to obtain coordinate grid."""
            xv, yv = np.meshgrid(x, y)
            return np.stack([xv.flatten(), yv.flatten()], axis=-1)

        output_x_end = (
            np.ceil(image_shape_arr[0] / stride_shape_arr[0]) * stride_shape_arr[0]
        )
        output_x_list = np.arange(0, int(output_x_end), stride_shape_arr[0])
        output_y_end = (
            np.ceil(image_shape_arr[1] / stride_shape_arr[1]) * stride_shape_arr[1]
        )
        output_y_list = np.arange(0, int(output_y_end), stride_shape_arr[1])
        output_tl_list = flat_mesh_grid_coord(output_x_list, output_y_list)
        output_br_list = output_tl_list + patch_output_shape_arr[None]

        io_diff = patch_input_shape_arr - patch_output_shape_arr
        input_tl_list = output_tl_list - (io_diff // 2)[None]
        input_br_list = input_tl_list + patch_input_shape_arr[None]

        sel = np.zeros(input_tl_list.shape[0], dtype=bool)
        if output_within_bound:
            sel |= np.any(output_br_list > image_shape_arr[None], axis=1)
        if input_within_bound:
            sel |= np.any(input_br_list > image_shape_arr[None], axis=1)
            sel |= np.any(input_tl_list < 0, axis=1)
        ####
        input_bound_list = np.concatenate(
            [input_tl_list[~sel], input_br_list[~sel]],
            axis=-1,
        )
        output_bound_list = np.concatenate(
            [output_tl_list[~sel], output_br_list[~sel]],
            axis=-1,
        )
        if return_output_bound:
            return input_bound_list, output_bound_list
        return input_bound_list


class SlidingWindowPatchExtractor(PatchExtractor):
    """Extract patches using sliding fixed sized window for images and labels.

    Args:
        input_img(str, pathlib.Path, :class:`numpy.ndarray`, :class:`WSIReader`):
            Input image for patch extraction.
        patch_size(int or tuple(int)):
            Patch size tuple (width, height).
        input_mask
            (str, pathlib.Path, :class:`numpy.ndarray`, or :obj:`VirtualWSIReader`):
            Input mask that is used for position filtering when
            extracting patches i.e., patches will only be extracted
            based on the highlighted regions in the `input_mask`.
            `input_mask` can be either path to the mask, a numpy array,
            :class:`VirtualWSIReader`, or one of 'otsu' and
            'morphological' options. In case of 'otsu' or
            'morphological', a tissue mask is generated for the
            input_image using tiatoolbox :class:`TissueMasker`
            functionality.
        resolution (Resolution):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (Units):
            The units of resolution, default = "level".
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
            `pad_mode`. If True, patches at the margins whose
            bounds would exceed the mother image dimensions would be
            neglected. Default is False.
        stride(int or tuple(int)):
            Stride in (x, y) direction for patch extraction, default =
            `patch_size`.
        min_mask_ratio (float):
            Only patches with positive area percentage above this value are included.
            Defaults to 0.
        store_filter (str):
            Filter to apply to the annotations when generating the mask. Default is
            None, which uses all annotations. Only used if the provided mask is an
            annotation store.

    Attributes:
        stride(tuple(int)):
            Stride in (x, y) direction for patch extraction.

    """

    def __init__(  # noqa: PLR0913
        self: SlidingWindowPatchExtractor,
        input_img: str | Path | np.ndarray | wsireader.WSIReader,
        patch_size: int | tuple[int, int],
        input_mask: str | Path | np.ndarray | wsireader.VirtualWSIReader | None = None,
        resolution: Resolution = 0,
        units: Units = "level",
        stride: int | tuple[int, int] | None = None,
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        min_mask_ratio: float = 0,
        store_filter: str | None = None,
        *,
        within_bound: bool = False,
    ) -> None:
        """Initialize :class:`SlidingWindowPatchExtractor`."""
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
            store_filter=store_filter,
        )
        if stride is None:
            self.stride = self.patch_size
        elif isinstance(stride, (tuple, list)):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            self.stride = (int(stride), int(stride))

        self._generate_location_df()


class PointsPatchExtractor(PatchExtractor):
    """Extracting patches with specified points as a centre.

    Args:
        input_img(str, pathlib.Path, :class:`numpy.ndarray`: class:`WSIReader`):
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
        resolution (Resolution):
            Resolution at which to read the image, default = 0. Either a
            single number or a sequence of two numbers for x and y are
            valid. This value is in terms of the corresponding units.
            For example: resolution=0.5 and units="mpp" will read the
            slide at 0.5 microns per-pixel, and resolution=3,
            units="level" will read at level at pyramid level /
            resolution layer 3.
        units (Units):
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
            `pad_mode`. If True, patches at the margins whose
            bounds would exceed the mother image dimensions would be
            neglected. Default is False.

    """

    def __init__(
        # pylint: disable=PLR0913
        self: PointsPatchExtractor,
        input_img: str | Path | np.ndarray | wsireader.WSIReader,
        locations_list: np.ndarray | DataFrame | str | Path,
        patch_size: int | tuple[int, int] = (224, 224),
        resolution: Resolution = 0,
        units: Units = "level",
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        *,
        within_bound: bool = False,
    ) -> None:
        """Initialize :class:`PointsPatchExtractor`."""
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            resolution=resolution,
            units=units,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            within_bound=within_bound,
        )
        try:
            self.locations_df = misc.read_locations(input_table=locations_list)
        except (TypeError, FileNotSupportedError) as exc:
            msg = "Please input correct locations_list. "
            msg += "Supported types: np.ndarray, DataFrame, str, Path."
            raise TypeError(msg) from exc
        self.locations_df["x"] = self.locations_df["x"] - int(
            (self.patch_size[1] - 1) / 2,
        )
        self.locations_df["y"] = self.locations_df["y"] - int(
            (self.patch_size[1] - 1) / 2,
        )


def get_patch_extractor(
    method_name: str,
    **kwargs: Unpack[ExtractorParams],
) -> PatchExtractor:
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
        msg = f"{method_name.lower()} method is not currently supported."
        raise MethodNotSupportedError(
            msg,
        )

    if method_name.lower() == "point":
        point_patch_extractor_args: PointsPatchExtractorParams
        point_patch_extractor_args = {
            "input_img": kwargs.get("input_img", ""),
            "locations_list": kwargs.get("locations_list", ""),
            "patch_size": kwargs.get("patch_size", (224, 224)),
            "resolution": kwargs.get("resolution", 0),
            "units": kwargs.get("units", "level"),
            "pad_mode": kwargs.get("pad_mode", "constant"),
            "pad_constant_values": kwargs.get("pad_constant_values", 0),
            "within_bound": kwargs.get("within_bound", False),
        }
        return PointsPatchExtractor(**point_patch_extractor_args)

    sliding_window_patch_extractor_args: SlidingWindowPatchExtractorParams
    sliding_window_patch_extractor_args = {
        "input_img": kwargs.get("input_img", ""),
        "patch_size": kwargs.get("patch_size", (224, 224)),
        "input_mask": kwargs.get("input_mask"),
        "resolution": kwargs.get("resolution", 0),
        "units": kwargs.get("units", "level"),
        "stride": kwargs.get("stride"),
        "pad_mode": kwargs.get("pad_mode", "constant"),
        "pad_constant_values": kwargs.get("pad_constant_values", 0),
        "min_mask_ratio": kwargs.get("min_mask_ratio", 0),
        "within_bound": kwargs.get("within_bound", False),
        "store_filter": kwargs.get("store_filter"),
    }
    return SlidingWindowPatchExtractor(**sliding_window_patch_extractor_args)
