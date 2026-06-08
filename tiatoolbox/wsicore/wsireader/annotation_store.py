"""Reader for Annotation stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np
from PIL import Image

from tiatoolbox import utils
from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntBounds, IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReaderParams


class AnnotationStoreReader(WSIReader):
    """Reader for Annotation stores.

    This reader is used to read annotation store data as if it were a WSI,
    rendering the annotations in the specified region to be read. Can be used
    either to render annotations as a stand-alone mask, or to render annotations
    on top of its parent WSI as a virtual 'annotated slide'.
    Note: Currently only supports annotations stored at the same resolution as
    the parent WSI base resolution. Support for annotations stored at arbitrary
    resolutions will be added in the future.

    Args:
        store (AnnotationStore | str | Path):
            An AnnotationStore or a path to an annotation store .db file.
        info (WSIMeta):
            Metadata of the base WSI for the annotations in the store.
            If this is not provided, will attempt to read it read from
            the store metadata, or the base_wsi if provided.
            If no source of metadata is found, will raise an error.
        renderer (AnnotationRenderer):
            Renderer to use for rendering annotations. Providing a renderer
            allows for customisation of the rendering process. If not provided,
            a sensible default will be created.
        base_wsi (WSIReader | str):
            Base WSI reader or path to use for reading the base WSI. Annotations
            will be rendered on top of the base WSI. If not provided,
            will render annotation masks without a base image.
        alpha (float):
            Opacity of the overlaid annotations. Must be between 0 and 1.
            Has no effect if base_wsi is not provided.

    """

    def __init__(
        self: AnnotationStoreReader,
        store: AnnotationStore | str | Path,
        info: WSIMeta | None = None,
        renderer: AnnotationRenderer | None = None,
        base_wsi: WSIReader | str | None = None,
        alpha: float = 1.0,
        **kwargs: Unpack[WSIReaderParams],
    ) -> None:
        """Initialize :class:`AnnotationStoreReader`."""
        super().__init__(store, **kwargs)
        self.store = (
            SQLiteStore(Path(store)) if isinstance(store, (str, Path)) else store
        )
        self.base_wsi = base_wsi
        if isinstance(base_wsi, (str, Path)):
            self.base_wsi = WSIReader.open(base_wsi)
        if info is None:
            # try to get metadata from store
            try:
                info = WSIMeta(**json.loads(self.store.metadata["wsi_meta"]))
            except KeyError as exc:
                if self.base_wsi is not None:
                    # get the metadata from the base reader.
                    # assumes annotations saved at WSI baseline res
                    info = self.base_wsi.info
                else:
                    # we cant find any metadata
                    msg = (
                        "No metadata found in store. "
                        "Please provide either info or base slide."
                    )
                    raise ValueError(
                        msg,
                    ) from exc
        self.info = info
        if renderer is None:
            types = self.store.pquery("props['type']")
            if len(types) == 0:
                renderer = AnnotationRenderer(max_scale=1000)
            else:
                renderer = AnnotationRenderer("type", list(types), max_scale=1000)
        renderer.edge_thickness = 0
        self.renderer = renderer
        if self.base_wsi is not None:
            self.on_slide = True
        self.alpha = alpha

    def _info(self: AnnotationStoreReader) -> WSIMeta:
        """Get the metadata of the slide."""
        return self.info

    def read_rect(
        self: AnnotationStoreReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region using start location and size (width, height).

        Read a region of the annotation mask, or annotated whole slide
        image at a location and size.

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
            >>> # Load an annotation store and associated wsi to be
            >>> # overlaid upon.
            >>> annotated_wsi = WSIReader.open(input_img="./CMU-1.db",
            >>>                         base_wsi="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> img = annotated_wsi.read_rect(location, size)
            >>> # Read a region at 0.5 microns per pixel (mpp)
            >>> img = annotated_wsi.read_rect(location, size, 0.5, "mpp")
            >>> # This could also be written more verbosely as follows
            >>> img = annotated_wsi.read_rect(
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
            >>> # Load an annotation store and associated wsi to be
            >>> # overlaid upon.
            >>> annotated_wsi = WSIReader.open(input_img="./CMU-1.db",
            >>>                     base_wsi="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # The resolution can be different in x and y, e.g.
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.75),
            ...     units="mpp",
            ... )
            >>> # Several units can be used including: objective power,
            >>> # microns per pixel, pyramid/resolution level, and
            >>> # fraction of baseline.
            >>> # E.g. Read a region at an objective power of 10x
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=10,
            ...     units="power",
            ... )
            >>> # Read a region at pyramid / resolution level 1
            >>> img = annotated_wsi.read_rect(
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
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="level",
            ... )
            >>> # Read a region at half of the full / baseline
            >>> # resolution.
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="baseline",
            ... )
            >>> # Read at a higher resolution than the baseline
            >>> # (interpolation applied to output)
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1.25,
            ...     units="baseline",
            ... )
            >>> # Assuming the image has a native mpp of 0.5,
            >>> # interpolation will be applied here.
            >>> img = annotated_wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.25,
            ...     units="mpp",
            ... )

        Annotations can also be displayed as a stand-alone mask not
        overlaid on the WSI. In this case, the metadata of the store
        must contain the resolution at which the annotations were saved
        at, and the slide dimensions at that resolution.
        Alternatively, an instance of WSIMeta can be provided describing the
        slide the annotations are associated with (in which case annotations
        are assumed to be saved at the baseline resolution given in the metadata).

        Example:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # get metadata from the slide (could also manually create a
            >>> # WSIMeta object if you know the slide info but do not have the
            >>> # slide itself)
            >>> metadata = WSIReader.open("CMU-1.ndpi").info
            >>> # Load associated annotations
            >>> annotation_mask = WSIReader.open(input_img="./CMU-1.db", info=wsi_meta)
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region of the mask at level 0 (baseline / full resolution)
            >>> img = annotation_mask.read_rect(location, size)
            >>> # Read a region of the mask at 0.5 microns per pixel (mpp)
            >>> img = annotation_mask.read_rect(location, size, 0.5, "mpp")

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
        im_region = self.renderer.render_annotations(
            self.store,
            bounds,
            self.info.level_downsamples[read_level],
        )

        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

        if self.base_wsi is not None:
            # overlay image region on the base wsi
            base_region = self.base_wsi.read_rect(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
                coord_space=coord_space,
                **kwargs,
            )
            base_region = Image.fromarray(
                utils.transforms.background_composite(base_region, alpha=True),
            )
            im_region = Image.fromarray(im_region)
            if self.alpha < 1.0:
                im_region.putalpha(
                    im_region.getchannel("A").point(lambda i: i * self.alpha),
                )
            base_region = Image.alpha_composite(base_region, im_region)
            base_region = base_region.convert("RGB")
            base_region = np.array(base_region)
            if self.post_proc is not None:
                base_region = self.post_proc(base_region)
            return base_region
        if self.post_proc is not None:
            im_region = self.post_proc(im_region)
        return utils.transforms.background_composite(im_region, alpha=False)

    def read_bounds(
        self: AnnotationStoreReader,
        bounds: IntBounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region by defining boundary locations.

        Read a region of the annotation mask, or annotated whole slide
        image within given bounds.

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
            >>> annotated_wsi = WSIReader.open(input_img="./CMU-1.db",
            >>>                          base_wsi="./CMU-1.ndpi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = annotated_wsi.read_bounds(bounds)
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

        im_region = self.renderer.render_annotations(
            self.store,
            bounds_at_baseline,
            self.info.level_downsamples[read_level],
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
        if self.base_wsi is not None:
            # overlay image region on the base wsi
            base_region = self.base_wsi.read_bounds(
                bounds,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
                coord_space=coord_space,
                **kwargs,
            )
            base_region = Image.fromarray(
                utils.transforms.background_composite(base_region, alpha=True),
            )
            im_region = Image.fromarray(im_region)
            if self.alpha < 1.0:
                im_region.putalpha(
                    im_region.getchannel("A").point(lambda i: i * self.alpha),
                )
            base_region = Image.alpha_composite(base_region, im_region)
            base_region = base_region.convert("RGB")
            base_region = np.array(base_region)
            if self.post_proc is not None:
                base_region = self.post_proc(base_region)
            return base_region
        if self.post_proc is not None:
            im_region = self.post_proc(im_region)
        return utils.transforms.background_composite(im_region, alpha=False)
