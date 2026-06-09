"""Reader for Omnyx JP2 images."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import glymur

from tiatoolbox import logger, utils
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from numbers import Number
    from pathlib import Path

    import numpy as np

    from tiatoolbox.type_hints import IntBounds, IntPair, Resolution, Units


class JP2WSIReader(WSIReader):
    """Class for reading Omnyx JP2 images.

    Supported WSI formats:

    - Omnyx JPEG-2000 (.jp2)

    Attributes:
        glymur_wsi (:obj:`glymur.Jp2k`)

    """

    def __init__(
        self: JP2WSIReader,
        input_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        post_proc: str | callable | None = "auto",
    ) -> None:
        """Initialize :class:`OmnyxJP2WSIReader`."""
        super().__init__(input_img=input_img, mpp=mpp, power=power, post_proc=post_proc)

        glymur.set_option("lib.num_threads", os.cpu_count() or 1)
        self.glymur_jp2 = glymur.Jp2k(filename=str(self.input_path))

    def read_rect(
        self: JP2WSIReader,
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

        stride = 2**read_level
        glymur_wsi = self.glymur_jp2
        bounds = utils.transforms.locsize2bounds(
            location=location,
            size=baseline_read_size,
        )
        im_region = utils.image.safe_padded_read(
            image=glymur_wsi,
            bounds=bounds,
            stride=stride,
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
        self: JP2WSIReader,
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
                _,  # bounds_at_read_level,
                size_at_requested,
                post_read_scale,
            ) = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )
        glymur_wsi = self.glymur_jp2

        stride = 2**read_level

        im_region = utils.image.safe_padded_read(
            image=glymur_wsi,
            bounds=bounds_at_baseline,
            stride=stride,
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
            im_region = self.post_proc(im_region)
        return utils.transforms.background_composite(image=im_region, alpha=False)

    @staticmethod
    def _get_jp2_boxes(
        jp2: glymur.jp2.Jp2k,
    ) -> dict[str, glymur.jp2box.Jp2kBox]:
        """Get JP2 boxes.

        Args:
            jp2 (glymur.jp2.Jp2k):
                Glymur JP2 image object.

        Raises:
            ValueError:
                If the JP2 image header is missing.

        Returns:
            dict[str, glymur.jp2box.Jp2kBox]:
                Dictionary of JP2 boxes. Should contain the keys
                "xml " and "cres" for Omnyx JP2 images. For other JP2
                images this may contain only the "cres" key or neither.
                The image header "ihdr" box is always present.

        """

        def find_box(
            box: glymur.jp2box.Jp2kBox | None,
            box_id: str,
        ) -> glymur.jp2box.Jp2kBox | None:
            """Find a box by its ID.

            Args:
                box (glymur.jp2box.Jp2kBox):
                    A box to search within. If None, returns None.
                box_id (str):
                    Box ID to search for. Must be 4 characters.

            Returns:
                Optional[glymur.jp2box.Jp2kBox]:
                    JP2 box with the given ID. If no box is found, returns

            """
            expected_len_box_id = 4
            msg = f"Box ID must be {expected_len_box_id} characters."
            if not len(box_id) == expected_len_box_id:  # pragma: no cover
                raise ValueError(msg)
            if not box or not box.box:
                return None
            for sub_box in box.box:
                if sub_box.box_id == box_id:
                    return sub_box
            return None

        header_box = find_box(jp2, "jp2h")
        image_header = find_box(header_box, "ihdr")
        resolution_box = find_box(header_box, "res ")
        capture_resolution_box = find_box(resolution_box, "resc")
        xml_box = find_box(jp2, "xml ")
        if image_header is None:
            msg = "Metadata: JP2 image header missing!"
            raise ValueError(msg)
        result = {
            "ihdr": image_header,
        }
        if xml_box is not None:
            result["xml "] = xml_box
        if capture_resolution_box is not None:
            result["cres"] = capture_resolution_box
        return result

    def _info(self: JP2WSIReader) -> WSIMeta:
        """JP2 metadata reader.

        Returns:
            WSIMeta:
                Metadata information.

        """
        jp2 = self.glymur_jp2
        boxes = self._get_jp2_boxes(jp2)
        objective_power = None
        vendor = None
        mpp = None
        # Check capture resolution box
        if "cres" in boxes:
            # Get the resolution in pixels per meter
            ppm_x = boxes.get("cres").horizontal_resolution
            ppm_y = boxes.get("cres").vertical_resolution
            mpp_x = utils.misc.ppu2mpp(ppm_x, "meter")
            mpp_y = utils.misc.ppu2mpp(ppm_y, "meter")
            mpp = [mpp_x, mpp_y]
        # Check for Aperio style/Omnyx XML (overwrites capture
        # resolution). This XML contains pipe seperated key values e.g.
        # "AppMag = 40 | ..."" in a <description> tag.
        if "xml " in boxes:
            description = boxes.get("xml ").xml.find("description")
            if description is not None and description.text:
                matches = re.search(
                    r"AppMag\s*=\s*(\d+)",
                    description.text,
                    flags=re.IGNORECASE,
                )
                if matches is not None:
                    objective_power = int(matches[1])
                if "Omnyx" in description.text:
                    vendor = "Omnyx"
                if "Aperio" in description.text:
                    vendor = "Aperio"
                matches = re.search(
                    r"MPP\s*=\s*(\d*\.\d+)",
                    description.text,
                    flags=re.IGNORECASE,
                )
                if matches is not None:
                    mpp_x = float(matches[1])
                    mpp_y = float(matches[1])
                    mpp = [mpp_x, mpp_y]

        # Fallback to calculating objective power & mpp
        objective_power, mpp = self._estimate_mpp_objective_power(
            objective_power=objective_power,
            mpp=mpp,
        )

        # Get image dimensions
        image_header = boxes["ihdr"]
        slide_dimensions = (image_header.width, image_header.height)

        # Determine level_count
        cod = None
        for segment in jp2.codestream.segment:
            if isinstance(segment, glymur.codestream.CODsegment):
                cod = segment
        if cod is None:
            logger.warning(
                "Metadata: JP2 codestream missing COD segment! "
                "Cannot determine number of decompositions (levels)",
            )
            level_count = 1
        else:
            level_count = cod.num_res

        level_downsamples = [2**n for n in range(level_count)]
        level_dimensions = [
            (int(slide_dimensions[0] / 2**n), int(slide_dimensions[1] / 2**n))
            for n in range(level_count)
        ]

        return WSIMeta(
            file_path=self.input_path,
            axes="YXS",
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
        )
