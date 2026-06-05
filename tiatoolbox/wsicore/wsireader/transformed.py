"""Reader for transformed whole slide images."""

from __future__ import annotations

import itertools
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import SimpleITK as sitk  # noqa: N813
from numpy.linalg import inv

from tiatoolbox import utils
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader, VirtualWSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntPair, Resolution, Units


class TransformedWSIReader(WSIReader):
    """Resampling regions from a whole slide image.

    This class is used to resample tiles/patches from a whole slide image
    using transformation.

    Example:
        >>> from tiatoolbox.wsicore.wsireader import TransformedWSIReader
        >>> transform_level0 = np.eye(3)
        >>> transformed_wsi = TransformedWSIReader(
        ...     input_img=sample_ome_tiff, target_img=sample_ome_tiff,
        ...     transform=transform_level0
        ... )
        >>> output = transformed_wsi.read_rect(
        ...     location,
        ...     size,
        ...     resolution,
        ...     units="level"
        ... )

    """

    def __init__(
        self: TransformedWSIReader,
        input_img: str | Path | np.ndarray,
        target_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        transform: np.ndarray | str | Path = None,  # Default to None
        fixed_info: WSIMeta = None,
    ) -> None:
        """Initialize :class:`TransformedWSIReader`.

        Args:
            input_img (str | Path | np.ndarray):
                Path to the input image or the image array.
            target_img (str | Path | np.ndarray):
                Path to the input target image or the image array.
            mpp (tuple(Number, Number)):
                Microns per pixel in x and y directions.
            power (Number):
                Objective power of the image.
            transform (str | Path | np.ndarray):
                Transformation matrix or path to a transformation file (.npy or .mha).

                - .npy: A file format for storing numpy arrays.
                - .mha: MetaImage Header, a file format used for storing medical imaging
                        data. It is part of the MetaIO library. For more information,
                        visit: https://public.kitware.com/Wiki/ITK/MetaIO/Documentation
            fixed_info (WSIMeta):
                Fixed metadata to use for the transformed image.

        """
        super().__init__(input_img=input_img, mpp=mpp, power=power)
        self.wsi_reader = WSIReader.open(input_img=input_img, mpp=mpp, power=power)
        self.target_wsi_reader = WSIReader.open(
            input_img=target_img, mpp=mpp, power=power
        )
        if transform is None:
            error_message = (
                "Transform cannot be None. "
                "Please provide a valid transformation matrix or file."
            )
            raise ValueError(error_message)
        # we need to set the info to be the fixed image info
        if fixed_info is not None:
            self.wsi_reader.info = fixed_info
        self.transformed_info = self.target_wsi_reader.info
        self.transform_type = "affine"
        if isinstance(transform, np.ndarray):
            self.transform_level0 = transform
        elif transform.suffix == ".npy":
            self.transform_level0 = np.load(transform)
        elif transform.suffix == ".mha":
            # .mha (MetaImage Header) is a file format used for storing medical imaging
            # data. It is part of the MetaIO library. For more information, visit:
            # https://public.kitware.com/Wiki/ITK/MetaIO/Documentation
            displacement_field = sitk.ReadImage(transform, sitk.sitkVectorFloat64)
            disp_array = sitk.GetArrayFromImage(displacement_field)  # (2, H, W)
            displacement_field_channels = 2
            if disp_array.shape[-1] != displacement_field_channels:
                # maybe in torch format with channel first
                disp_array = np.moveaxis(disp_array, 0, -1)
            self.df_dims = np.array((disp_array.shape[1], disp_array.shape[0]))
            # scale factors are actually in relation to the largest dimension
            # from source and target image (so add offset and then scale)
            self.level_scale_factors = [
                np.asarray([s_dims, t_dims]).max(axis=0) / np.array(self.df_dims)
                for s_dims, t_dims in zip(
                    self.wsi_reader.info.level_dimensions,
                    self.target_wsi_reader.info.level_dimensions,
                    strict=False,
                )
            ]
            self.level_pads = [
                (((t_dims[0] - s_dims[0]) // 2), ((t_dims[1] - s_dims[1]) // 2))
                for s_dims, t_dims in zip(
                    self.wsi_reader.info.level_dimensions,
                    self.target_wsi_reader.info.level_dimensions,
                    strict=False,
                )
            ]
            self.get_location_array(disp_array)
            self.transform_type = "displacement"
        else:
            error_message = "Unsupported transformation file format"
            raise ValueError(error_message)

        if self.transform_type == "affine":
            wsimeta = self.wsi_reader.info
            baseline_size = self.wsi_reader.slide_dimensions(
                resolution=0, units="level"
            )
            _, transformed_shape = self.get_transformed_location(
                location=(0, 0),
                size=baseline_size,
                level=0,
            )

            # Calculate the new shape at each level based on downsampling
            wsimeta.level_dimensions = tuple(
                tuple(
                    np.asarray(transformed_shape)
                    // (baseline_size[0] // s_dims[0], baseline_size[1] // s_dims[1])
                )
                for s_dims in self.wsi_reader.info.level_dimensions
            )

            wsimeta.slide_dimensions = wsimeta.level_dimensions[0]
            self.transformed_info = wsimeta

    def get_location_array(self, disp_array: np.ndarray) -> None:
        """Transform an array of locations using the displacement field.

        Gives an inverse showing, for a given pixel in a transformed image, where it
        would come from in the original image.

        Args:
            disp_array (np.ndarray): A numpy array representing the displacement values.

        Returns:
            None
        """
        location_array = np.mgrid[
            0 : disp_array.shape[0], 0 : disp_array.shape[1]
        ].transpose(1, 2, 0)
        location_array = np.flip(location_array, 2)
        transformed_image = self.transform_using_disp_array(location_array, disp_array)

        # make a reader for convenient reading at desired locations/resolutions
        wsimeta = self.wsi_reader.info
        wsimeta.level_dimensions = tuple(
            tuple(np.asarray([s_dims, t_dims]).max(axis=0))
            for s_dims, t_dims in zip(
                self.wsi_reader.info.level_dimensions,
                self.target_wsi_reader.info.level_dimensions,
                strict=False,
            )
        )
        wsimeta.slide_dimensions = wsimeta.level_dimensions[0]
        self.inverse_loc_reader = VirtualWSIReader(
            transformed_image, info=wsimeta, mode="feature"
        )
        self.transformed_info = wsimeta

    @staticmethod
    def transform_using_disp_array(
        input_array: np.ndarray, disp_array: np.ndarray
    ) -> np.ndarray:
        """Transform an array of locations using the displacement field.

        Args:
            input_array (np.ndarray): A numpy array representing the input locations.
            disp_array (np.ndarray): A numpy array representing the displacement field.

        Returns:
            np.ndarray: The transformed array of locations.

        """
        input_image = sitk.GetImageFromArray(input_array, isVector=True)

        # Convert displacement field numpy array to SimpleITK image
        displacement_field = sitk.GetImageFromArray(disp_array, isVector=True)

        # Create a displacement field transform
        transform = sitk.DisplacementFieldTransform(displacement_field)

        # Set the interpolator
        interpolator = sitk.sitkLinear  # maybe others better?

        # Apply the transform to the input image
        transformed_image = sitk.Resample(
            input_image,
            input_image,
            transform,
            interpolator,
            outputPixelType=sitk.sitkVectorFloat32,
        )
        return sitk.GetArrayFromImage(transformed_image)

    def _info(self: TransformedWSIReader) -> WSIMeta:
        """Get the WSI metadata."""
        return self.transformed_info

    @staticmethod
    def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform points using the given transformation matrix.

        Args:
            points (:class:`numpy.ndarray`):
                A set of points of shape (N, 2).
            transform (:class:`numpy.ndarray`):
                Transformation matrix of shape (3, 3).

        Returns:
            :class:`numpy.ndarray`:
                Warped points  of shape (N, 2).

        """
        points = np.array(points)
        # Pad the data with ones, so that our transformation can do translations
        points_pad = np.hstack([points, np.ones((points.shape[0], 1))])
        points_warp = np.dot(points_pad, transform.T)
        return points_warp[:, :-1]

    def get_patch_dimensions(
        self: TransformedWSIReader,
        size: tuple[int, int],
        transform: np.ndarray,
    ) -> tuple[int, int]:
        """Compute patch size needed for transformation.

        Args:
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            transform (:class:`numpy.ndarray`):
                Transformation matrix of shape (3, 3).

        Returns:
            :py:obj:`tuple` - Maximum size of the patch needed for transformation.
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height

        """
        width, height = size[0], size[1]

        x_info = [
            np.linspace(1, width, width, endpoint=True),
            np.ones(height) * width,
            np.linspace(1, width, width, endpoint=True),
            np.ones(height),
        ]
        x = np.array(list(itertools.chain.from_iterable(x_info)))

        y_info = [
            np.ones(width),
            np.linspace(1, height, height, endpoint=True),
            np.ones(width) * height,
            np.linspace(1, height, height, endpoint=True),
        ]
        y = np.array(list(itertools.chain.from_iterable(y_info)))

        points = np.array([x, y]).transpose()
        transform = transform * [[1, 1, 0], [1, 1, 0], [1, 1, 1]]  # remove translation
        transform_points = self.transform_points(points, transform)

        width = (
            int(np.max(transform_points[:, 0]))
            - int(np.min(transform_points[:, 0]))
            + 1
        )
        height = (
            int(np.max(transform_points[:, 1]))
            - int(np.min(transform_points[:, 1]))
            + 1
        )

        return (width, height)

    def get_transformed_location(
        self: TransformedWSIReader,
        location: tuple[int, int],
        size: tuple[int, int],
        level: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get corresponding location on unregistered image and the required patch size.

        This function applies inverse transformation to the centre point of the region.
        The transformed centre point is used to obtain the transformed top left pixel
        of the region.

        Args:
            location (tuple(int)):
                (x, y) tuple giving the top left pixel in the baseline (level 0)
                reference frame.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            level (int):
                Pyramid level/resolution layer.

        Returns:
            tuple:
                - :py:obj:`tuple` - Transformed location (top left pixel).
                    - :py:obj:`int` - X coordinate
                    - :py:obj:`int` - Y coordinate
                - :py:obj:`tuple` - Maximum size suitable for transformation.
                    - :py:obj:`int` - Width
                    - :py:obj:`int` - Height

        """
        inv_transform = inv(self.transform_level0)
        size_level0 = [x * (2**level) for x in size]
        center_level0 = [x + size_level0[i] / 2 for i, x in enumerate(location)]
        center_level0_arr = np.expand_dims(np.array(center_level0), axis=0)
        center_level0_arr = self.transform_points(center_level0_arr, inv_transform)[0]

        transformed_size = self.get_patch_dimensions(size, inv_transform)
        transformed_location = (
            int(center_level0_arr[0] - (transformed_size[0] * (2**level)) / 2),
            int(center_level0_arr[1] - (transformed_size[1] * (2**level)) / 2),
        )
        return transformed_location, transformed_size

    @staticmethod
    def sample_image_opencv(
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Samples image a at positions specified by b using OpenCV's remap function.

        Parameters:
        - a: numpy array of shape (H, W, 3), the source image.
        - b: numpy array of shape (M, N, 2), the array of x, y positions.

        Returns:
        - output: numpy array of shape (M, N, 3), the sampled image.
        """
        # Convert b to float32 and split into x and y maps
        b_float = b.astype(np.float32)
        map_x = b_float[..., 0]
        map_y = b_float[..., 1]

        # Use cv2.remap to sample the image
        return cv2.remap(
            a,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def get_transformed_location_df(
        self: TransformedWSIReader,
        location: tuple[int, int],
        size: tuple[int, int],
        level: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get corresponding location on unregistered image and the required patch size.

        This function applies inverse transformation to the points in the region,
        in the case of a displacement field transform.
        The transformed points are used to obtain the transformed bounding box
        of the region.

        Args:
            location (tuple(int)):
                (x, y) tuple giving the top left pixel in the read resolution
                reference frame.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            level (int):
                Pyramid level/resolution layer.

        Returns:
            tuple:
                - :py:obj:`tuple` - Transformed location (top left pixel).
                    - :py:obj:`int` - X coordinate
                    - :py:obj:`int` - Y coordinate
                - :py:obj:`tuple` - Maximum size suitable for transformation.
                    - :py:obj:`int` - Width
                    - :py:obj:`int` - Height

        """
        # read relevant bit of inverse displacement field
        inv_locs = self.inverse_loc_reader.read_rect(
            location, size, level, coord_space="resolution", interpolation="lanczos"
        )
        # scale the field according to the level
        transformed_grid = inv_locs * self.level_scale_factors[level]

        # Find bounding box of transformed grid + padding
        pad = 2
        min_x = max(np.min(transformed_grid[:, :, 0]) - pad, 0)
        max_x = np.max(transformed_grid[:, :, 0]) + pad
        min_y = max(np.min(transformed_grid[:, :, 1]) - pad, 0)
        max_y = np.max(transformed_grid[:, :, 1]) + pad
        # shift the grid into this coordinate space
        transformed_grid = transformed_grid - np.array([min_x, min_y]) + pad
        location = (int(min_x), int(min_y))
        # Unpad
        location = (
            location[0] - self.level_pads[level][0],
            location[1] - self.level_pads[level][1],
        )
        size = (
            int(max_x - min_x),
            int(max_y - min_y),
        )
        return location, size, transformed_grid

    def transform_patch(
        self: TransformedWSIReader,
        patch: np.ndarray,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Apply transformation to the given patch.

        This function applies the transformation matrix after removing the translation.

        Args:
            patch (:class:`numpy.ndarray`):
                A region of whole slide image.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.

        Returns:
            :class:`numpy.ndarray`:
                A transformed region/patch.

        """
        transform = self.transform_level0 * [[1, 1, 0], [1, 1, 0], [1, 1, 1]]
        translation = (-size[0] / 2 + 0.5, -size[1] / 2 + 0.5)
        forward_translation = np.array(
            [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]],
        )
        inverse_translation = np.linalg.inv(forward_translation)
        transform = inverse_translation @ transform @ forward_translation
        return cv2.warpAffine(patch, transform[0:-1][:], patch.shape[:2][::-1])

    def read_rect(
        self: TransformedWSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a transformed region of the transformed whole slide image.

        Location is in terms of the baseline image (level 0 / maximum resolution),
        and size is the output image size.

        Args:
            location (IntPair):
                (x, y) tuple giving the top left pixel in the baseline
                (level 0) reference frame.
            size (IntPair):
                (width, height) tuple giving the desired output image
                size.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                The units of resolution, default = "level". Supported
                units are: microns per pixel (mpp), objective power
                (power), pyramid / resolution level (level), pixels per
                baseline pixel (baseline).
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int, tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `location` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs:
                Extra key-word arguments for reader specific parameters.
                Currently only used by :obj:`VirtualWSIReader`. See
                class docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                A transformed region/patch.

        Example:
            >>> from tiatoolbox.wsicore.wsireader import TransformedWSIReader
            >>> transform_level0 = np.eye(3)
            >>> tfm = TransformedWSIReader(
            ...     input_img=sample_ome_tiff, target_img=sample_ome_tiff,
            ...     transform=transform_level0
            ... )
            >>> output = tfm.read_rect(
            ...     location, size, resolution=resolution, units="level"
            ... )

        """
        if coord_space == "resolution":
            # In actuality, `read_rect` at resolution is synonymous with
            # calling `read_bound` at resolution because `size` has always
            # been within the resolution system.

            tl = np.array(location)
            br = location + np.array(size)
            bounds = np.concatenate([tl, br])

            return self.read_bounds(
                bounds,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
                coord_space="resolution",
                **kwargs,
            )

        pad = 2
        (
            read_level,
            _level_location,
            level_size,
            post_read_scale,
            _baseline_read_size,
        ) = self.wsi_reader.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        if self.transform_type == "displacement":
            transformed_location, max_size, transformed_grid = (
                self.get_transformed_location_df(
                    _level_location - pad,
                    level_size + pad * 2,
                    read_level,
                )
            )
            # Read at optimal level and corrected read size
            patch = self.wsi_reader.read_rect(
                transformed_location, max_size, read_level, coord_space="resolution"
            )
        else:
            transformed_location, max_size = self.get_transformed_location(
                location,
                size,
                read_level,
            )
            patch = self.wsi_reader.read_region(
                transformed_location, read_level, max_size
            )
            # convert location to read resolution
            transformed_location = (
                int(transformed_location[0] / (2**read_level)),
                int(transformed_location[1] / (2**read_level)),
            )
        patch = np.array(patch)

        # Apply padding outside the slide area
        patch = utils.image.crop_and_pad_edges(
            bounds=utils.transforms.locsize2bounds(transformed_location, max_size),
            max_dimensions=self.info.level_dimensions[read_level],
            region=patch,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Apply transformation
        if self.transform_type == "displacement":
            transformed_patch = self.sample_image_opencv(patch, transformed_grid)
            transformed_patch = transformed_patch[pad:-pad, pad:-pad, :]
        else:
            transformed_patch = self.transform_patch(patch, max_size)

        # Resize to desired size
        post_read_scale = float(post_read_scale[0]), float(post_read_scale[1])
        return utils.transforms.imresize(
            img=transformed_patch,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

    def read_bounds(
        self: TransformedWSIReader,
        bounds: Bounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,  # noqa: ARG002
    ) -> np.ndarray:
        """Read a transformed region of the transformed whole slide image within bounds.

        Bounds are in terms of the baseline image (level 0 / maximum resolution),
        and size is the output image size.

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The output image size may be different to the width and height
        of the bounds as the resolution will affect this. To read a
        region with a fixed output image size see :func:`read_rect`.

        Args:
            bounds (IntBounds):
                By default, this is a tuple of (start_x, start_y, end_x,
                end_y) i.e. (left, top, right, bottom) of the region in
                baseline reference frame. However, with
                `coord_space="resolution"`, the bound is expected to be
                at the requested resolution system.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                The units of resolution, default = "level". Supported
                units are: microns per pixel (mpp), objective power
                (power), pyramid / resolution level (level), pixels per
                baseline pixel (baseline).
            coord_space (str):
                Coordinate space of the bounds. By default, the bounds
                are in the baseline reference frame. If
                `coord_space="resolution"` then the bounds are expected
                to be at the requested resolution system.
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int | tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs:
                Extra key-word arguments for reader specific parameters.
                Currently only used by :obj:`VirtualWSIReader`. See
                class docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                A transformed region/patch.

        Example:
            >>> from tiatoolbox.wsicore.wsireader import TransformedWSIReader
            >>> wsi = TransformedWSIReader(
            ...    input_img="cmu-1.ndpi", target_img="cmu-1.ndpi",
            ...    transform="transform.mha"
            ... )
            >>> # read a region of size 1000x1000 at 1.25x scale
            >>> # from (10000, 10000) at level 0
            >>> img = wsi.read_bounds(25000,25000,27000,27000)

        """
        pad = 2
        # convert from requested to `baseline`
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self.bounds_at_resolution_to_baseline(
                bounds,
                resolution,
                units,
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            (
                read_level,
                bounds_at_read_level,
                _,
                post_read_scale,
            ) = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )
        else:  # duplicated portion with VirtualReader, factoring out ?
            # Find parameters for optimal read
            (
                read_level,
                bounds_at_read_level,
                size_at_requested,
                post_read_scale,
            ) = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        # Read at optimal level and corrected read size
        location_at_baseline = np.array(bounds_at_baseline[:2])
        _, size_at_read_level = utils.transforms.bounds2locsize(bounds_at_read_level)

        # Transform bounds and read untransformed image
        if self.transform_type == "displacement":
            transformed_location, max_size, transformed_grid = (
                self.get_transformed_location_df(
                    location=np.array(bounds_at_read_level[:2]) - pad,
                    size=size_at_read_level + pad * 2,
                    level=read_level,
                )
            )
            # Read at optimal level and corrected read size
            patch = self.wsi_reader.read_rect(
                location=transformed_location,
                size=max_size,
                resolution=read_level,
                coord_space="resolution",
            )
        else:
            transformed_location, max_size = self.get_transformed_location(
                location=location_at_baseline,
                size=size_at_read_level,
                level=read_level,
            )
            patch = self.wsi_reader.read_region(
                location=transformed_location, level=read_level, size=max_size
            )
        patch = np.array(patch)

        # Apply padding outside the slide area.
        patch = utils.image.crop_and_pad_edges(
            bounds=bounds_at_read_level,
            max_dimensions=self.info.level_dimensions[read_level],
            region=patch,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Apply transformation.
        if self.transform_type == "displacement":
            transformed_patch = self.sample_image_opencv(patch, transformed_grid)
            transformed_patch = transformed_patch[pad:-pad, pad:-pad, :]
        else:
            transformed_patch = self.transform_patch(patch, max_size)

        # Resize to desired size
        post_read_scale = float(post_read_scale[0]), float(post_read_scale[1])
        return utils.transforms.imresize(
            img=transformed_patch,
            scale_factor=post_read_scale,
            output_size=size_at_requested,
            interpolation=interpolation,
        )
