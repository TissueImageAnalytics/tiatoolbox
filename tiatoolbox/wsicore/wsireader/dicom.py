"""Reader for DICOM WSI files."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np

from tiatoolbox import logger, utils
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReaderParams


class DICOMWSIReader(WSIReader):
    """Define DICOM WSI Reader."""

    wsidicom = None

    def __init__(
        self: DICOMWSIReader,
        input_img: str | Path | np.ndarray,
        post_proc: str | callable | None = "auto",
        **kwargs: Unpack[WSIReaderParams],
    ) -> None:
        """Initialize :class:`DICOMWSIReader`."""
        from wsidicom import WsiDicom  # noqa: PLC0415

        super().__init__(input_img=input_img, post_proc=post_proc, **kwargs)
        self.wsi = WsiDicom.open(input_img)

    def _info(self: DICOMWSIReader) -> WSIMeta:
        """WSI metadata constructor.

        Returns:
            WSIMeta:
                Containing metadata.

        """
        level_dimensions = [
            (level.size.width, level.size.height) for level in self.wsi.levels
        ]
        level_downsamples = [
            np.mean(
                [
                    level_dimensions[0][0] / level.size.width,
                    level_dimensions[0][1] / level.size.height,
                ],
            )
            for level in self.wsi.levels
        ]
        dataset = self.wsi.levels.base_level.datasets[0]
        # Get pixel spacing in mm from DICOM file and convert to um/px (mpp)
        mm_per_pixel = dataset.pixel_spacing
        mpp = (mm_per_pixel.width * 1e3, mm_per_pixel.height * 1e3)

        objective_power = None
        ops_seq = getattr(dataset, "OpticalPathSequence", None)
        if ops_seq:
            ops = ops_seq[0]
            if hasattr(ops, "ObjectiveLensPower"):
                objective_power = ops.ObjectiveLensPower

        # Fallback to calculating objective power & mpp when metadata is missing.
        objective_power, mpp = self._estimate_mpp_objective_power(
            objective_power=objective_power,
            mpp=mpp,
        )

        return WSIMeta(
            slide_dimensions=level_dimensions[0],
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            axes="YXS",
            mpp=mpp,
            objective_power=objective_power,
            level_count=len(level_dimensions),
            vendor=dataset.Manufacturer,
            file_path=self.input_path,
        )

    def read_rect(
        self: DICOMWSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,  # noqa: ARG002
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
                Currently only used by VirtualWSIReader. See class
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
            read_level,
            level_location,
            level_read_size,
            post_read_scale,
            _,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        wsi = self.wsi

        # Read at optimal level and corrected read size
        level_size = self.info.level_dimensions[read_level]
        constrained_read_bounds = utils.image.find_overlap(
            read_location=level_location,
            read_size=level_read_size,
            image_size=level_size,
        )
        _, constrained_read_size = utils.transforms.bounds2locsize(
            constrained_read_bounds,
        )

        # if out of bounds, return empty image consistent with openslide
        if np.any(np.array(constrained_read_size) <= 0):
            return (
                np.ones(
                    shape=(int(size[1]), int(size[0]), 3),
                    dtype=np.uint8,
                )
                * 255
            )

        dicom_level = wsi.levels[read_level].level
        im_region = wsi.read_region(level_location, dicom_level, constrained_read_size)
        im_region = np.array(im_region)

        # Apply padding outside the slide area
        level_read_bounds = utils.transforms.locsize2bounds(
            level_location,
            level_read_size,
        )
        im_region = utils.image.crop_and_pad_edges(
            bounds=level_read_bounds,
            max_dimensions=level_size,
            region=im_region,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Resize to correct scale if required
        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=tuple(np.array(size).astype(int)),
            interpolation=interpolation,
        )

        if self.post_proc is not None:
            im_region = self.post_proc(im_region)
        return utils.transforms.background_composite(image=im_region, alpha=False)

    def read_bounds(
        self: DICOMWSIReader,
        bounds: IntBounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,  # noqa: ARG002
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

        wsi = self.wsi

        # Read at optimal level and corrected read size
        level_location, size_at_read_level = utils.transforms.bounds2locsize(
            bounds_at_read_level,
        )
        level_size = self.info.level_dimensions[read_level]
        read_bounds = utils.image.find_overlap(
            level_location,
            size_at_read_level,
            level_size,
        )
        _, read_size = utils.transforms.bounds2locsize(read_bounds)
        dicom_level = wsi.levels[read_level].level
        im_region = wsi.read_region(
            location=level_location,
            level=dicom_level,
            size=read_size,
        )
        im_region = np.array(im_region)

        # Apply padding outside the slide area
        im_region = utils.image.crop_and_pad_edges(
            bounds=bounds_at_read_level,
            max_dimensions=self.info.level_dimensions[read_level],
            region=im_region,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Resize to correct scale if required
        if coord_space == "resolution":
            im_region = utils.transforms.imresize(
                img=im_region,
                output_size=size_at_requested,
                interpolation=interpolation,
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
                interpolation=interpolation,
            )

        if self.post_proc is not None:
            return self.post_proc(im_region)
        return utils.transforms.background_composite(image=im_region, alpha=False)


