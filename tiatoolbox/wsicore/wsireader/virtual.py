"""Reader for non-pyramidal images."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

from tiatoolbox import logger, utils
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntPair, Resolution, Units


class VirtualWSIReader(WSIReader):
    """Class for reading non-pyramidal images e.g., visual fields.

    Supported formats:

    - .jpg
    - .png
    - :class:`numpy.ndarray`

    This reader uses :func:`tiatoolbox.utils.image.sub_pixel_read` to
    allow reading low resolution images as if they are larger i.e. with
    'virtual' pyramid resolutions. This is useful for reading low
    resolution masks as if they were stretched to overlay a higher
    resolution WSI.

    Extra key-word arguments given to :func:`~WSIReader.read_region` and
    :func:`~WSIReader.read_bounds` will be passed to
    :func:`~tiatoolbox.utils.image.sub_pixel_read`.

    Attributes:
        img (:class:`numpy.ndarray`):
            Input image as :class:`numpy.ndarray`.
        mode (str):
            Mode of the input image. Default is 'rgb'. Allowed values
            are: rgb, bool, feature. "rgb" mode supports bright-field color images.
            "bool" mode supports binary masks,
            interpolation in this case will be "nearest" instead of "bicubic".
            "feature" mode allows multichannel features.

    Args:
        input_img (str, :obj:`Path`, :class:`numpy.ndarray`):
            Input path to WSI.
        info (WSIMeta):
            Metadata for the virtual wsi.
        mode (str):
            Mode of the input image. Default is 'rgb'. Allowed values
            are: rgb, bool, feature. "rgb" mode supports bright-field color images.
            "bool" mode supports binary masks,
            interpolation in this case will be "nearest" instead of "bicubic".
            "feature" mode allows multichannel features.
        post_proc (str, callable):
            Post-processing function to apply to the output image.

    """

    def __init__(
        self: VirtualWSIReader,
        input_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        info: WSIMeta | None = None,
        mode: str = "rgb",
        post_proc: str | callable | None = "auto",
    ) -> None:
        """Initialize :class:`VirtualWSIReader`."""
        super().__init__(
            input_img=input_img,
            mpp=mpp,
            power=power,
            post_proc=post_proc,
        )
        if mode.lower() not in ["rgb", "bool", "feature"]:
            msg = "Invalid mode."
            raise ValueError(msg)

        if isinstance(input_img, np.ndarray):
            self.img = input_img
        else:
            self.img = utils.imread(self.input_path)

        _min_image_ndims = (
            2  # Minimum number of dimensions required for an image (H, W)
        )
        # Reject 1D (or otherwise <2D) inputs early with a clear message.
        if self.img.ndim < _min_image_ndims:
            msg = "Input image must be 2D (H, W) or 3D (H, W, C). Got a 1D array."
            raise ValueError(msg)

        if mode != "bool" and (
            self.img.ndim == 2 or self.img.shape[2] not in [3, 4]  # noqa: PLR2004
        ):
            logger.warning(
                "The image mode is set to 'feature' as the input"
                " dimensions do not match with binary mask or RGB/RGBA.",
            )
            mode = "feature"

        self.mode = mode.lower()

        if info is not None:
            self._m_info = info

    def _info(self: VirtualWSIReader) -> WSIMeta:
        """Visual Field metadata getter.

        This generates a WSIMeta object for the slide if none exists.
        There is 1 level with dimensions equal to the image and no mpp,
        objective power, or vendor data.


        Returns:
            WSIMeta:
                Metadata information.

        """
        param = WSIMeta(
            file_path=self.input_path,
            axes="YSX",
            objective_power=None,
            # align to XY to match with OpenSlide
            slide_dimensions=self.img.shape[:2][::-1],
            level_count=1,
            level_dimensions=(self.img.shape[:2][::-1],),
            level_downsamples=[1.0],
            vendor=None,
            mpp=None,
            raw=None,
        )
        if self._m_info is None:
            self._m_info = param
        return self._m_info

    def _find_params_from_baseline(
        self: VirtualWSIReader,
        location: IntPair,
        baseline_read_size: IntPair,
    ) -> tuple[IntPair, IntPair]:
        """Convert read parameters from (virtual) baseline coordinates.

        Args:
            location (IntPair):
                Location of the location to read in (virtual) baseline
                coordinates.
            baseline_read_size (IntPair):
                Size of the region to read in (virtual) baseline
                coordinates.

        Returns:
            tuple(IntPair, IntPair):
                Baseline image location and read size.

        """
        baseline_size = np.array(self.info.slide_dimensions)
        image_size = np.array(self.img.shape[:2][::-1])
        size_ratio = image_size / baseline_size
        image_location = np.array(location, dtype=np.float32) * size_ratio
        read_size = np.array(baseline_read_size) * size_ratio
        return image_location, read_size

    def read_rect(
        self: VirtualWSIReader,
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
        """Read a region of the whole slide image at a location and size.

        Location is in terms of the baseline image (level 0  / maximum
        resolution), and size is the output image size.

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The field of view varies with resolution. For a fixed field of
        view see :func:`read_bounds`.

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
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently, only used by VirtualWSIReader. See class
                docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=size[0], N=size[1]

        Example:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
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
            ...     resolution=(0.5, 0.5),
            ...     units="mpp",
            ... )

        Note: The field of view varies with resolution when using
        :func:`read_rect`.

        .. figure:: ../images/read_rect_tissue.png
            :width: 512
            :alt: Diagram illustrating read_rect

        As the location is in the baseline reference frame but the size
        (width and height) is the output image size, the field of view
        therefore changes as resolution changes.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        .. figure:: ../images/read_rect-interpolated-reads.png
            :width: 512
            :alt: Diagram illustrating read_rect interpolting between levels

        When reading between the levels stored in the WSI, the
        coordinates of the requested region are projected to the next
        highest resolution. This resolution is then decoded and
        downsampled to produce the desired output. This is a major
        source of variability in the time take to perform a read
        operation. Reads which require reading a large region before
        downsampling will be significantly slower than reading at a
        fixed level.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # The resolution can be different in x and y, e.g.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.75),
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
        if coord_space == "resolution":
            return self.read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (
            _,
            _,
            _,
            _,
            baseline_read_size,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        image_location, image_read_size = self._find_params_from_baseline(
            location,
            baseline_read_size,
        )

        bounds = utils.transforms.locsize2bounds(
            location=image_location,
            size=image_read_size,
        )

        output_size = None if interpolation in [None, "none"] else size
        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds,
            output_size=output_size,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
            pad_at_baseline=False,
        )

        if self.mode == "rgb":
            if self.post_proc is not None:
                im_region = self.post_proc(im_region)
            return utils.transforms.background_composite(image=im_region, alpha=False)
        return im_region

    def read_bounds(
        self: VirtualWSIReader,
        bounds: IntBounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region of the whole slide image within given bounds.

        Bounds are in terms of the baseline image (level 0  / maximum
        resolution).

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
                Units of resolution, default="level". Supported units
                are: microns per pixel (mpp), objective power (power),
                pyramid / resolution level (level), pixels per baseline
                pixel (baseline).
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
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently only used by :obj:`VirtualWSIReader`. See
                class docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> from matplotlib import pyplot as plt
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = wsi.read_bounds(bounds)
            >>> plt.imshow(img)
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_bounds(
            ...     bounds,
            ...     resolution=0,
            ...     units="level",
            ... )
            >>> plt.imshow(img)

        Note: The field of view remains the same as resolution is varied
        when using :func:`read_bounds`.

        .. figure:: ../images/read_bounds_tissue.png
            :width: 512
            :alt: Diagram illustrating read_bounds

        This is because the bounds are in the baseline (level 0)
        reference frame. Therefore, varying the resolution does not
        change what is visible within the output image.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        """
        # convert from requested to `baseline`
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self.bounds_at_resolution_to_baseline(
                bounds,
                resolution,
                units,
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # * Find parameters for optimal read
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            _, _, _, post_read_scale = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )
        else:
            # * Find parameters for optimal read
            _, _, size_at_requested, post_read_scale = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        location_at_read, size_at_read = self._find_params_from_baseline(
            *utils.transforms.bounds2locsize(bounds_at_baseline),
        )
        bounds_at_read = utils.transforms.locsize2bounds(location_at_read, size_at_read)

        if interpolation in [None, "none"]:
            interpolation = None

        if interpolation == "optimise" and self.mode == "bool":
            interpolation = "nearest"

        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds_at_read,
            output_size=size_at_requested,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
            pad_at_baseline=False,
        )

        if coord_space == "resolution":
            # do this to enforce output size is as defined by input bounds
            im_region = utils.transforms.imresize(
                img=im_region,
                output_size=size_at_requested,
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
            )

        if self.mode == "rgb":
            if self.post_proc is not None:
                im_region = self.post_proc(im_region)
            return utils.transforms.background_composite(image=im_region, alpha=False)
        return im_region


class ArrayView:
    """An object for viewing a zarr array with a different index ordering.

    Used to allow YXS index order reads for arrays with axes in other
    orders such as SYX. Currently supported axes are:
    - YXS
    - SYX

    """

    def __init__(self: ArrayView, array: zarr.Array, axes: str) -> None:
        """Initialise the view object.

        Args:
            array (zarr.Array):
                Zarr Array to read from.
            axes (str):
                Axes ordering string. Allowed values are YXS and SYX.

        """
        self.array = array
        self.axes = axes
        self._shape = dict(zip(self.axes, self.array.shape, strict=False))

    @property
    def shape(self: ArrayView) -> tuple:
        """Return array shape."""
        try:
            return tuple(self._shape[c] for c in "YXC")
        except KeyError:
            return tuple(self._shape[c] for c in "YXS")

    def __getitem__(self: ArrayView, index: int) -> np.ndarray:
        """Get an item from the dataset."""
        # Normalize to a tuple of length = len(self.axes)
        if not isinstance(index, tuple):
            index = (index,)
        while len(index) < len(self.axes):
            index = (*index, slice(None))

        if self.axes in ("YXS", "YXC"):
            return self.array[index]
        if self.axes in ("SYX", "CYX"):
            y, x, s = index
            index = (s, y, x)
            return np.rollaxis(self.array[index], 0, 3)
        msg = f"Unsupported axes `{self.axes}`."
        raise ValueError(msg)
