"""Reader for NGFF (OME-Zarr) WSI files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np
import zarr
from zarr.storage import FsspecStore

from tiatoolbox import logger, utils
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import Resolution, Units
    from tiatoolbox.wsicore import WSIReaderParams


class NGFFWSIReader(WSIReader):
    """Reader for NGFF WSI zarr(s).

    Support is currently experimental. This supports reading from
    NGFF version 0.4.

    """

    def __init__(
        self: NGFFWSIReader, path: str | Path, **kwargs: Unpack[WSIReaderParams]
    ) -> None:
        """Initialize :class:`NGFFWSIReader`."""
        super().__init__(path, **kwargs)
        from imagecodecs import numcodecs  # noqa: PLC0415

        from tiatoolbox.wsicore.metadata import ngff  # noqa: PLC0415

        numcodecs.register_codecs()
        storage_options = kwargs.get("storage_options", {})
        store = FsspecStore.from_url(path, storage_options=storage_options)
        self._zarr_group: zarr.Group = zarr.open(store, mode="r", zarr_format=2)
        attrs = self._zarr_group.attrs
        multiscales = attrs.get("multiscales")[0]
        axes = multiscales.get("axes")
        datasets = multiscales.get("datasets")
        omero = attrs.get("omero")
        self.zattrs = ngff.Zattrs(
            _creator=ngff.Creator(
                name=attrs.get("name"),
                version=attrs.get("version"),
            ),
            multiscales=ngff.Multiscales(
                version=multiscales.get("version"),
                axes=[ngff.Axis(**axis) for axis in axes],
                datasets=[
                    ngff.Dataset(
                        path=dataset["path"],
                        coordinateTransformations=dataset.get(
                            "coordinateTransformations",
                        ),
                    )
                    for dataset in datasets
                ],
            ),
            omero=ngff.Omero(
                name=omero.get("name"),
                id=omero.get("id"),
                channels=[ngff.Channel(**channel) for channel in omero["channels"]],
                rdefs=ngff.RDefs(**omero["rdefs"]),
                version=omero.get("version"),
            ),
        )
        self.level_arrays = {
            int(key): ArrayView(array, axes=self.info.axes)
            for key, array in self._zarr_group.arrays()
        }

    def _info(self: NGFFWSIReader) -> WSIMeta:
        """WSI metadata constructor.

        Returns:
            WSIMeta:
                Containing metadata.

        """
        multiscales = self.zattrs.multiscales
        mpp = self._get_mpp()
        # This needs to be replaced once an appropriate image is available for test.
        objective_power = None

        # Fallback to calculating objective power & mpp
        objective_power, mpp = self._estimate_mpp_objective_power(
            objective_power=objective_power,
            mpp=mpp,
        )
        # Get indices by matching the axis name
        if multiscales.axes:
            indices = [i for i, a in enumerate(multiscales.axes) if a.name == "x"]
            x_index = indices[0] if indices else None
            indices = [i for i, a in enumerate(multiscales.axes) if a.name == "y"]
            y_index = indices[0] if indices else None
        else:
            # Default to (y, x)
            x_index = 1
            y_index = 0

        return WSIMeta(
            axes="".join(axis.name.upper() for axis in multiscales.axes),
            level_dimensions=[
                (array.shape[x_index], array.shape[y_index])
                for _, array in sorted(self._zarr_group.arrays(), key=lambda x: x[0])
            ],
            slide_dimensions=(
                self._zarr_group["0"].shape[x_index],
                self._zarr_group["0"].shape[y_index],
            ),
            vendor=self.zattrs._creator.name,  # skipcq: PYL-W0212  # noqa: SLF001
            raw=self._zarr_group.attrs,
            mpp=mpp,
            objective_power=objective_power,
        )

    def _get_mpp(self: NGFFWSIReader) -> tuple[float, float] | None:
        """Get the microns-per-pixel (MPP) of the slide.

        Returns:
            Tuple[float, float]:
                The mpp of the slide an x,y tuple. None if not available.

        """
        # Check that the required axes are present
        multiscales = self.zattrs.multiscales
        axes_dict = {a.name.lower(): a for a in multiscales.axes}
        if "x" not in axes_dict or "y" not in axes_dict:
            return None
        x = axes_dict["x"]
        y = axes_dict["y"]

        # Check the units,
        # Currently only handle micrometer units
        if x.unit != y.unit != "micrometer":
            logger.warning(
                "Expected units of micrometer, got %s and %s",
                x.unit,
                y.unit,
            )
            return None

        # Check that datasets is non-empty and has at least one coordinateTransformation
        if (
            not multiscales.datasets
            or not multiscales.datasets[0].coordinateTransformations
        ):
            return None

        # Currently simply using the first scale transform
        transforms = multiscales.datasets[0].coordinateTransformations
        for t in transforms:
            if "scale" in t and t.get("type") == "scale":
                x_index = multiscales.axes.index(x)
                y_index = multiscales.axes.index(y)
                return (t["scale"][x_index], t["scale"][y_index])
        return None

    def read_rect(
        self: NGFFWSIReader,
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
            >>> wsi = WSIReader.open(input_img="./CMU-1.ome.zarr")
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
            >>> wsi = WSIReader.open(input_img="./CMU-1.ome.zarr")
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
            im_region = self.read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )
            return utils.transforms.background_composite(image=im_region, alpha=False)

        # Find parameters for optimal read
        (
            read_level,
            _,
            _,
            post_read_scale,
            baseline_read_size,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        bounds = utils.transforms.locsize2bounds(
            location=location,
            size=baseline_read_size,
        )
        im_region = utils.image.safe_padded_read(
            image=self.level_arrays[read_level],
            bounds=bounds,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

        if self.post_proc is not None:
            im_region = self.post_proc(im_region)
        return utils.transforms.background_composite(image=im_region, alpha=False)

    def read_bounds(
        self: NGFFWSIReader,
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
            pad_constant_values (int, IntPair):
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
            >>> wsi = WSIReader.open(input_img="./CMU-1.ome.zarr")
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
                _,
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
                _,
                size_at_requested,
                post_read_scale,
            ) = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        im_region = utils.image.sub_pixel_read(
            image=self.level_arrays[read_level],
            bounds=bounds_at_baseline,
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
        if self.post_proc is not None:
            im_region = self.post_proc(im_region)

        return im_region


