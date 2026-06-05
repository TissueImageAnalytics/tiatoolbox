"""Readers for TIFF WSI files."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
import zarr
from zarr.storage import MemoryStore

from tiatoolbox import logger, utils
from tiatoolbox.utils.exceptions import FileNotSupportedError
from tiatoolbox.wsicore.wsimeta import WSIMeta

from .base import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntPair, Resolution, Units


class TIFFWSIReader(WSIReader):
    """Define Tiff WSI Reader."""

    def __init__(
        self: TIFFWSIReader,
        input_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        series: str = "auto",
        cache_size: int = 2**28,
        post_proc: str | callable | None = "auto",
    ) -> None:
        """Initialize :class:`TIFFWSIReader`."""
        super().__init__(input_img=input_img, mpp=mpp, power=power, post_proc=post_proc)
        self.tiff = tifffile.TiffFile(self.input_path)
        self._axes = self.tiff.series[0].axes
        # Flag which is True if the image is a simple single page tile TIFF
        is_single_page_tiled = all(
            [
                self.tiff.pages[0].is_tiled,
                # Not currently supporting multi-page images
                not self.tiff.is_multipage,
                # Currently only supporting single page generic tiled TIFF
                len(self.tiff.pages) == 1,
            ],
        )
        if not any(
            [
                self.tiff.is_svs,
                self.tiff.is_ome,
                is_single_page_tiled,
                self.tiff.is_bigtiff,
            ]
        ):
            msg = "Unsupported TIFF WSI format."
            raise ValueError(msg)

        self.series_n = series
        if self.tiff.series is None or len(self.tiff.series) == 0:  # pragma: no cover
            msg = "TIFF does not contain any valid series."
            raise FileNotSupportedError(msg)
        # Find the largest series if series="auto"
        if self.series_n == "auto":
            all_series = self.tiff.series or []

            def page_area(page: tifffile.TiffPage) -> float:
                """Calculate the area of a page."""
                return np.prod(
                    TIFFWSIReaderDelegate.canonical_shape(self._axes, page.shape)[:2],
                    dtype=float,
                )

            series_areas = [page_area(s.pages[0]) for s in all_series]  # skipcq
            self.series_n = np.argmax(series_areas)
        self._tiff_series = self.tiff.series[self.series_n]
        self._zarr_store = tifffile.imread(
            self.input_path,
            series=self.series_n,
            aszarr=True,
        )
        # Updated Zarr 3 logic for TIFFWSIReader
        cache_backend = MemoryStore()
        self._zarr_cache = CacheStore(
            store=self._zarr_store, cache_store=cache_backend, max_size=cache_size
        )
        self._zarr_group = zarr.open(self._zarr_cache)

        if isinstance(self._zarr_group, zarr.Group):  # pragma: no cover
            self.level_arrays = {
                int(key): ArrayView(array, axes=self._axes)
                for key, array in self._zarr_group.members()
            }
        else:  # pragma: no cover
            self.level_arrays = {0: ArrayView(self._zarr_group, axes=self._axes)}

        # ensure level arrays are sorted by descending area
        self.level_arrays = dict(
            sorted(
                self.level_arrays.items(),
                key=lambda x: (
                    -np.prod(
                        TIFFWSIReaderDelegate.canonical_shape(
                            self._axes, x[1].array.shape[:2]
                        ),
                        dtype=float,
                    )
                ),
            )
        )
        # maybe get colors if they exist in metadata
        self._get_colors_from_meta()

        self.tiff_reader_delegate = TIFFWSIReaderDelegate(self, self.level_arrays)

    def _get_colors_from_meta(self: TIFFWSIReader) -> None:
        """Get colors from metadata if they exist."""
        if not isinstance(self.post_proc, postproc_defs.MultichannelToRGB):
            return

        try:
            xml = self.info.raw["Description"]
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            return

        # Try multiple formats
        for parser in (
            TIFFWSIReader._parse_scancolortable,
            TIFFWSIReader._parse_filtercolor_metadata,
            TIFFWSIReader._parse_ome_metadata_mapping,
        ):
            color_dict = parser(root)
            if color_dict:
                self.post_proc.color_dict = color_dict
                return

    @staticmethod
    def _parse_scancolortable(
        root: ElementTree,
    ) -> dict[str, tuple[float, float, float]] | None:
        """Parse ScanColorTable metadata from XML and convert color values to RGB.

        Args:
            root (ElementTree): The root of the parsed XML tree.

        Returns:
            dict[str, tuple[float, float, float]] | None: A mapping of channel
            names to RGB tuples, or None if not found.

        """
        color_info = root.find(".//ScanColorTable")
        if color_info is None:
            return None

        color_dict = {
            k.text.split("_")[0]: v.text
            for k, v in zip(
                color_info.iterfind("ScanColorTable-k"),
                color_info.iterfind("ScanColorTable-v"),
                strict=False,
            )
        }
        # values will be either a string of 3 ints e.g 155, 128, 0, or
        # a color name e.g Lime. Convert them all to RGB tuples.
        for key, value in color_dict.items():
            if value is None:
                continue
            if "," in value:
                color_dict[key] = tuple(int(x) / 255 for x in value.split(","))
            else:
                color_dict[key] = mcolors.to_rgb(value)

        return color_dict

    @staticmethod
    def _parse_filtercolor_metadata(
        root: ElementTree,
    ) -> dict[str, tuple[float, float, float]] | None:
        """Parse FilterColors metadata from XML and convert color values to RGB.

        Args:
            root (ElementTree): The root of the parsed XML tree.

        Returns:
            dict[str, tuple[float, float, float]] | None: A mapping of channel
            names to RGB tuples, or None if not found.

        """
        # try alternate metadata format
        # Build a map from filter pair string -> color label or RGB string
        # from the <FilterColors> section
        filter_colors = {}
        filter_colors_section = root.find(".//FilterColors")
        if filter_colors_section is None:
            return None

        keys = filter_colors_section.findall(".//FilterColors-k")
        vals = filter_colors_section.findall(".//FilterColors-v")
        for k, v in zip(keys, vals, strict=False):
            filter_colors[k.text] = v.text

        # Helper function to convert color strings like "Lime" or
        # "255, 128, 0" into (R,G,B)
        def color_string_to_rgb(s: str) -> tuple[float, float, float]:
            """Convert a color string (e.g., 'Lime' or '255, 128, 0') to an RGB tuple.

            Args:
                s (str): The color string.

            Returns:
                tuple[float, float, float]: RGB values normalized to [0, 1].
            """
            if "," in s:
                return tuple(int(x.strip()) / 255 for x in s.split(","))
            return mcolors.to_rgb(s)

        # 2) For each <ScanBands-i>, find the channel's name and figure out
        #    which filter pair it uses, then match that to a color.
        channel_dict = {}
        for scan_band in root.findall(".//ScanBands-i"):
            # Inside a <ScanBands-i> there is a <Bands-i> with a <Name> tag
            bands_i = scan_band.find(".//Bands-i")
            if bands_i is not None:
                band_name_element = bands_i.find("Name")
                if band_name_element is not None:
                    channel_name = band_name_element.text.strip()

                    # Grab the filter pair manufacturer info
                    filter_pair = scan_band.find(".//FilterPair")
                    if filter_pair is not None:
                        emission_part = filter_pair.find(
                            ".//EmissionFilter/FixedFilter/PartNumber"
                        )
                        excitation_part = filter_pair.find(
                            ".//ExcitationFilter/FixedFilter/PartNumber"
                        )
                        if emission_part is not None and excitation_part is not None:
                            matching_rgb = (1.0, 1.0, 1.0)  # default white
                            for fc_key, fc_val in filter_colors.items():
                                # if both part numbers appear in the FilterColors-k
                                # string, assume it's the match
                                if (
                                    emission_part.text in fc_key
                                    and excitation_part.text in fc_key
                                ):
                                    matching_rgb = color_string_to_rgb(fc_val)
                                    break

                            channel_dict[channel_name] = matching_rgb

        return channel_dict if channel_dict else None

    @staticmethod
    def _get_namespace(root: ElementTree) -> dict:
        """Extract the XML namespace from the root element.

        Args:
            root (ElementTree): Root of the parsed XML tree.

        Returns:
            dict: Dictionary containing the namespace prefix and URI.

        """
        if root.tag.startswith("{"):
            ns_uri = root.tag.split("}")[0].strip("{")
            return {"ns": ns_uri}

        return {}

    @staticmethod
    def _extract_dye_mapping(root: ElementTree, ns: dict) -> dict:
        """Extract dye mapping from OME-XML annotations.

        Args:
            root (ElementTree): Root of the parsed XML tree.
            ns (dict): XML namespace dictionary.

        Returns:
            dict: Mapping of channel IDs to dye names.

        """
        dye_mapping = {}
        for annotation in root.findall(
            ".//ns:StructuredAnnotations/ns:XMLAnnotation", ns
        ):
            value_elem = annotation.find("ns:Value", ns)
            if value_elem is not None:
                for chan_priv in value_elem.findall(".//ns:ChannelPriv", ns):
                    chan_id = chan_priv.attrib.get("ID")
                    dye = chan_priv.attrib.get("FluorescenceChannel")
                    if chan_id and dye:
                        dye_mapping[chan_id] = dye
        return dye_mapping

    @staticmethod
    def _int_to_rgb(color_int: int) -> tuple[float, float, float]:
        """Convert an integer color value to an RGB tuple.

        Args:
            color_int (int): Integer representation of a color.

        Returns:
            tuple[float, float, float]: RGB values normalized to [0, 1].

        """
        if color_int < 0:
            color_int += 1 << 32
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF

        return (r / 255, g / 255, b / 255)

    @staticmethod
    def _parse_channel_data(
        root: ElementTree, ns: dict, dye_mapping: dict
    ) -> list[dict]:
        """Parse channel metadata from OME-XML.

        Extract RGB color and dye information for each channel defined in the metadata.

        Args:
            root (ElementTree): Root of the parsed XML tree.
            ns (dict): XML namespace dictionary.
            dye_mapping (dict): Mapping of channel IDs to dye names.

        Returns:
            list[dict]: List of dictionaries containing channel metadata.

        """
        channel_data = []
        for pixels in root.findall(".//ns:Pixels", ns):
            for channel in pixels.findall("ns:Channel", ns):
                chan_id = channel.attrib.get("ID")
                name = channel.attrib.get("Name")
                color = channel.attrib.get("Color")
                if chan_id and name and color:
                    try:
                        color_int = int(color)
                        rgb = TIFFWSIReader._int_to_rgb(color_int)
                    except ValueError:
                        rgb = None
                    dye = dye_mapping.get(chan_id, "Unknown")
                    label = f"{chan_id}: {name} ({dye})"
                    channel_data.append(
                        {
                            "id": chan_id,
                            "name": name,
                            "dye": dye,
                            "rgb": rgb,
                            "label": label,
                        }
                    )
        return channel_data

    @staticmethod
    def _build_color_dict(
        channel_data: list[dict], dye_mapping: dict
    ) -> dict[str, tuple[float, float, float]]:
        """Build a dictionary mapping channel names to RGB color tuples.

        Args:
            channel_data (list[dict]): List of channel metadata dictionaries.
            dye_mapping (dict): Mapping of channel IDs to dye names.

        Returns:
            dict[str, tuple[float, float, float]]: Dictionary mapping channel labels to
            RGB values.

        """
        color_dict = {}
        key_counts = defaultdict(int)
        for c_data in channel_data:
            chan_id = c_data["id"]
            name = c_data["name"]
            dye = dye_mapping.get(chan_id)
            rgb = c_data["rgb"]
            base_key = f"{name} ({dye})" if dye else name
            count = key_counts[base_key]
            key = base_key if count == 0 else f"{base_key} [{count + 1}]"
            color_dict[key] = rgb
            key_counts[base_key] += 1

        return color_dict

    @staticmethod
    def _parse_ome_metadata_mapping(
        root: ElementTree,
    ) -> dict[str, tuple[float, float, float]] | None:
        """Parse OME metadata from the given XML root element.

        Args:
            root (ElementTree): The root of the parsed XML tree.

        Returns:
            dict[str, tuple[float, float, float]] | None: A mapping
            of channel names to RGB tuples, or None if not found.

        """
        # 3) Try OME/Lunaphore format e.g. for COMET
        ns = TIFFWSIReader._get_namespace(root)
        dye_mapping = TIFFWSIReader._extract_dye_mapping(root, ns)
        channel_data = TIFFWSIReader._parse_channel_data(root, ns, dye_mapping)
        color_dict = TIFFWSIReader._build_color_dict(channel_data, dye_mapping)

        return color_dict if color_dict else None

    def _get_ome_xml(self: TIFFWSIReader) -> ElementTree.Element:
        """Parse OME-XML from the description of the first IFD (page).

        Returns:
            ElementTree.Element:
                OME-XML root element.

        """
        description = self.tiff.pages[0].description
        return ElementTree.fromstring(description)

    def _parse_ome_metadata(self: TIFFWSIReader) -> dict:
        """Extract OME specific metadata.

        Returns:
            dict:
                Dictionary of kwargs for WSIMeta.

        """
        # The OME-XML should be in each IFD but is optional. It must be
        # present in the first IFD. We simply get the description from
        # the first IFD.
        xml = self._get_ome_xml()
        objective_power = self._get_ome_objective_power(xml)
        mpp = self._get_ome_mpp(xml)

        return {
            "objective_power": objective_power,
            "vendor": None,
            "mpp": mpp,
            "raw": {
                "Description": self.tiff.pages[0].description,
                "OME-XML": xml,
            },
        }

    def _get_ome_objective_power(
        self: TIFFWSIReader,
        xml: ElementTree.Element | None = None,
    ) -> float | None:
        """Get the objective power from the OME-XML.

        Args:
            xml (ElementTree.Element, optional):
                OME-XML root element. Defaults to None. If None, the
                OME-XML will be parsed from the first IFD.

        Returns:
            float:
                Objective power.

        """
        xml = xml or self._get_ome_xml()
        namespaces = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

        try:
            xml_series = xml.findall("ome:Image", namespaces)[self.series_n]
            instrument_ref = xml_series.find("ome:InstrumentRef", namespaces)
            objective_settings = xml_series.find("ome:ObjectiveSettings", namespaces)
            if objective_settings is None:
                # try alternative tag
                objective_settings = xml_series.find("ome:Objective", namespaces)

            instrument_ref_id = instrument_ref.attrib.get("ID")
            objective_settings_id = (
                objective_settings.attrib.get("ID")
                if objective_settings is not None
                else "Objective:0"
            )

            instruments = {
                instrument.attrib.get("ID"): instrument
                for instrument in xml.findall("ome:Instrument", namespaces)
            }
            objectives = {
                (instrument_id, objective.attrib.get("ID")): objective
                for instrument_id, instrument in instruments.items()
                for objective in instrument.findall("ome:Objective", namespaces)
            }

            objective = objectives.get((instrument_ref_id, objective_settings_id))
            if objective is not None:
                return float(objective.attrib.get("NominalMagnification"))

        except (IndexError, AttributeError, ValueError, TypeError, KeyError) as e:
            logger.warning("OME objective power extraction failed: %s", e)

        # Fallback: try to infer from MPP
        mpp = self._get_ome_mpp(xml)
        if mpp is not None:
            try:
                return utils.misc.mpp2common_objective_power(float(np.mean(mpp)))
            except (TypeError, ValueError) as e:
                logger.warning("Failed to infer objective power from MPP: %s", e)

        logger.warning("Objective power could not be determined from OME-XML.")
        return None

    def _get_ome_mpp(
        self: TIFFWSIReader,
        xml: ElementTree.Element | None = None,
    ) -> list[float] | None:
        """Get the microns per pixel from the OME-XML.

        Args:
            xml (ElementTree.Element, optional):
                OME-XML root element. Defaults to None. If None, the
                OME-XML will be parsed from the first IFD.

        Returns:
            Optional[List[float]]:
                Microns per pixel.

        """
        xml = xml or self._get_ome_xml()
        namespaces = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        xml_series = xml.findall("ome:Image", namespaces)[self.series_n]
        xml_pixels = xml_series.find("ome:Pixels", namespaces)
        mppx = xml_pixels.attrib.get("PhysicalSizeX")
        mppy = xml_pixels.attrib.get("PhysicalSizeY")
        if mppx is not None and mppy is not None:
            return [mppx, mppy]
        if mppx is not None or mppy is not None:
            logger.warning("Only one MPP value found. Using it for both X  and Y.")
            return [mppx or mppy] * 2

        return None

    def _info(self: TIFFWSIReader) -> WSIMeta:
        """TIFF metadata constructor.

        Returns:
            WSIMeta:
                Containing metadata.

        """
        level_count = len(self.level_arrays)
        level_dimensions = [
            np.array(
                TIFFWSIReaderDelegate.canonical_shape(self._axes, p.array.shape)[:2][
                    ::-1
                ]
            )
            for p in self.level_arrays.values()
        ]
        slide_dimensions = level_dimensions[0]
        level_downsamples = [(level_dimensions[0] / x)[0] for x in level_dimensions]
        # The tags attribute object will not pickle or deepcopy,
        # so a copy with only python values or tifffile enums is made.
        tifffile_tags = self.tiff.pages[0].tags.items()
        tiff_tags = {
            code: {
                "code": code,
                "value": tag.value,
                "name": tag.name,
                "count": tag.count,
                "type": tag.dtype,
            }
            for code, tag in tifffile_tags
        }

        if self.tiff.is_svs:
            filetype_params = TIFFWSIReaderDelegate.parse_svs_metadata(self.tiff.pages)
        elif self.tiff.is_ome:
            filetype_params = self._parse_ome_metadata()
        else:
            filetype_params = TIFFWSIReaderDelegate.parse_generic_tiff_metadata(
                self.tiff.pages
            )
        filetype_params["raw"]["TIFF Tags"] = tiff_tags

        # Fallback to calculating objective power from mpp
        objective_power = filetype_params["objective_power"]
        mpp = (
            np.asarray(filetype_params["mpp"]).astype(float)
            if filetype_params["mpp"] is not None
            else None
        )
        # Fallback to calculating objective power & mpp
        objective_power, mpp = self._estimate_mpp_objective_power(
            objective_power=objective_power,
            mpp=mpp,
        )
        filetype_params["objective_power"] = objective_power
        filetype_params["mpp"] = mpp

        # Updating for mypy checks
        slide_dimensions = (slide_dimensions[0], slide_dimensions[1])
        level_dimensions = [
            (level_dimensions_[0], level_dimensions_[1])
            for level_dimensions_ in level_dimensions
        ]

        return WSIMeta(
            file_path=self.input_path,
            slide_dimensions=slide_dimensions,
            axes=self._axes,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            **filetype_params,
        )

    def read_rect(
        self: TIFFWSIReader,
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
        """See TIFFWSIReaderDelegate.read_rect docs for details."""
        return self.tiff_reader_delegate.read_rect(
            location,
            size,
            resolution,
            units,
            interpolation,
            pad_mode,
            pad_constant_values,
            coord_space,
            **kwargs,
        )

    def read_bounds(
        self: TIFFWSIReader,
        bounds: IntBounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """See TIFFWSIReaderDelegate.read_bounds docs for details."""
        return self.tiff_reader_delegate.read_bounds(
            bounds,
            resolution,
            units,
            interpolation,
            pad_mode,
            pad_constant_values,
            coord_space,
            **kwargs,
        )


class TIFFWSIReaderDelegate:
    """Delegate class to handle image reading operations.

    Currently used in FsspecJsonWSIReader and TIFFWSIReader.
    """

    def __init__(self, reader: WSIReader, level_arrays: dict[int, ArrayView]) -> None:
        """Initialize the delegate with a reader and level arrays.

        Args:
            reader (WSIReader): An instance of FsspecJsonWSIReader or TIFFWSIReader.
            level_arrays (dict[int, ArrayView]): Dictionary of level arrays.
        """
        self.reader = reader
        self.level_arrays = level_arrays

    @staticmethod
    def parse_svs_metadata(pages: TiffPages) -> dict:
        """Extract SVS specific metadata.

        Returns:
            dict:
                Dictionary of kwargs for WSIMeta.

        """
        raw = {}
        mpp = None
        objective_power = None
        vendor = "Aperio"

        description = pages[0].description
        raw["Description"] = description
        parts = description.split("|")
        description_headers, key_value_pairs = parts[0], parts[1:]
        description_headers = description_headers.split(";")

        software, photometric_info = description_headers[0].splitlines()
        raw["Software"] = software
        raw["Photometric Info"] = photometric_info

        def parse_svs_tag(string: str) -> tuple[str, Number | str]:
            """Parse SVS key-value string.

            Infers type(s) of data by trial and error with a fallback to
            the original string type.

            Args:
                string (str):
                    Key-value string in SVS format: "key=value".

            Returns:
                tuple:
                    Key-value pair.

            """
            pair = string.split("=")
            if len(pair) != 2:  # noqa: PLR2004
                msg = "Invalid metadata. Expected string of the format 'key=value'."
                raise ValueError(
                    msg,
                )
            key, value_string = pair
            key = key.strip()
            value_string = value_string.strip()

            def us_date(string: str) -> datetime:
                """Return datetime parsed according to US date format (UTC-aware)."""
                # and we immediately attach UTC.
                dt = datetime.strptime(string, r"%m/%d/%y")
                return dt.replace(tzinfo=UTC)

            def time(string: str) -> datetime:
                """Return datetime parsed according to HMS format (UTC-aware)."""
                # parse to time first; although .time() is tz-agnostic
                # DTZ007 is triggered by strptime
                t = datetime.strptime(string, r"%H:%M:%S").time()
                today_utc = datetime.now(UTC)
                return today_utc.replace(
                    hour=t.hour, minute=t.minute, second=t.second, microsecond=0
                )

            casting_precedence = [us_date, time, int, float]
            value = value_string
            for cast in casting_precedence:
                try:
                    value = cast(value_string)
                except ValueError:
                    continue
                else:
                    return key, value

            return key, value

        svs_tags = dict(parse_svs_tag(string) for string in key_value_pairs)
        raw["SVS Tags"] = svs_tags
        mpp = svs_tags.get("MPP")
        if mpp is not None:  # pragma: no cover
            mpp = [mpp] * 2
        objective_power = svs_tags.get("AppMag")

        return {
            "objective_power": objective_power,
            "vendor": vendor,
            "mpp": mpp,
            "raw": raw,
        }

    @staticmethod
    def canonical_shape(axes: str, shape: tuple[int, int]) -> tuple[int, int]:
        """Make a level shape tuple in YXS order.

        Args:
            axes (str): The axes format.
            shape (tuple[int, int]): Input shape tuple.

        Returns:
            tuple[int, int]: Shape in YXS order.
        """
        if axes in ("YXS", "YXC"):
            return shape
        if axes in ("SYX", "CYX"):
            return np.roll(shape, -1)
        msg = f"Unsupported axes `{axes}`."
        raise ValueError(msg)

    def read_rect(
        self,
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
            im_region = self.reader.read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )
            if self.reader.post_proc is not None:
                im_region = self.reader.post_proc(im_region)
            return im_region

        # Find parameters for optimal read
        (
            read_level,
            level_read_location,
            level_read_size,
            post_read_scale,
            _,
        ) = self.reader.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        bounds = utils.transforms.locsize2bounds(
            location=level_read_location,
            size=level_read_size,
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

        if self.reader.post_proc is not None:
            im_region = self.reader.post_proc(im_region)
        return im_region

    def read_bounds(
        self: TIFFWSIReaderDelegate,
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
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self.reader.bounds_at_resolution_to_baseline(
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
            ) = self.reader.find_read_bounds_params(
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
            ) = self.reader.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        im_region = utils.image.sub_pixel_read(
            image=self.level_arrays[read_level],
            bounds=bounds_at_read_level,
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

        if self.reader.post_proc is not None:
            return self.reader.post_proc(im_region)
        return im_region

    @staticmethod
    def parse_generic_tiff_metadata(pages: TiffPages) -> dict:
        """Extract generic tiled metadata.

        Returns:
            dict: Dictionary of kwargs for WSIMeta.

        """
        mpp = None
        objective_power = None
        vendor = "Generic"

        description = pages[0].description
        raw = {"Description": description}
        # Check for MPP in the tiff resolution tags
        # res_units: 1 = undefined, 2 = inch, 3 = centimeter
        res_units = pages[0].tags.get("ResolutionUnit")
        res_x = pages[0].tags.get("XResolution")
        res_y = pages[0].tags.get("YResolution")
        if (  # pragma: no cover
            all(x is not None for x in [res_units, res_x, res_y])
            and res_units.value != 1
        ):
            mpp = [
                utils.misc.ppu2mpp(res_x.value[0] / res_x.value[1], res_units.value),
                utils.misc.ppu2mpp(res_y.value[0] / res_y.value[1], res_units.value),
            ]

        return {
            "objective_power": objective_power,
            "vendor": vendor,
            "mpp": mpp,
            "raw": raw,
        }
