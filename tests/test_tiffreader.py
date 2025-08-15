"""Test TIFFWSIReader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from defusedxml import ElementTree
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from tiatoolbox.wsicore import wsireader


def test_ome_missing_instrument_ref(
    monkeypatch: pytest.MonkeyPatch,
    remote_sample: Callable,
) -> None:
    """Test that an OME-TIFF can be read without instrument reference."""
    sample = remote_sample("ome-brightfield-small-level-0")
    wsi = wsireader.TIFFWSIReader(sample)
    page = wsi.tiff.pages[0]
    description = page.description
    tree = ElementTree.fromstring(description)
    namespaces = {
        "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    }
    images = tree.findall("ome:Image", namespaces)
    for image in images:
        instruments = image.findall("ome:InstrumentRef", namespaces)
        for instrument in instruments:
            image.remove(instrument)
    new_description = ElementTree.tostring(tree, encoding="unicode")
    monkeypatch.setattr(page, "description", new_description)
    monkeypatch.setattr(wsi, "_m_info", None)
    assert wsi.info.objective_power is None


def test_ome_missing_physicalsize(
    monkeypatch: pytest.MonkeyPatch,
    remote_sample: Callable,
) -> None:
    """Test that an OME-TIFF can be read without physical size."""
    sample = remote_sample("ome-brightfield-small-level-0")
    wsi = wsireader.TIFFWSIReader(sample)
    page = wsi.tiff.pages[0]
    description = page.description
    tree = ElementTree.fromstring(description)
    namespaces = {
        "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    }
    images = tree.findall("ome:Image", namespaces)
    for image in images:
        pixels = image.find("ome:Pixels", namespaces)
        del pixels.attrib["PhysicalSizeX"]
        del pixels.attrib["PhysicalSizeY"]
    new_description = ElementTree.tostring(tree, encoding="unicode")
    monkeypatch.setattr(page, "description", new_description)
    monkeypatch.setattr(wsi, "_m_info", None)
    assert wsi.info.mpp is None


def test_ome_missing_physicalsizey(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test that an OME-TIFF can be read without physical size."""
    sample = remote_sample("ome-brightfield-small-level-0")
    wsi = wsireader.TIFFWSIReader(sample)
    page = wsi.tiff.pages[0]
    description = page.description
    tree = ElementTree.fromstring(description)
    namespaces = {
        "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    }
    images = tree.findall("ome:Image", namespaces)
    for image in images:
        pixels = image.find("ome:Pixels", namespaces)
        del pixels.attrib["PhysicalSizeY"]
    new_description = ElementTree.tostring(tree, encoding="unicode")
    monkeypatch.setattr(page, "description", new_description)
    monkeypatch.setattr(wsi, "_m_info", None)
    assert pytest.approx(wsi.info.mpp, abs=0.1) == 0.5
    assert "Only one MPP value found. Using it for both X  and Y" in caplog.text


def test_tiffreader_non_tiled_metadata(
    monkeypatch: pytest.MonkeyPatch,
    remote_sample: Callable,
) -> None:
    """Test that fetching metadata for non-tiled TIFF works."""
    sample = remote_sample("ome-brightfield-small-level-0")
    wsi = wsireader.TIFFWSIReader(sample)
    monkeypatch.setattr(wsi.tiff, "is_ome", False)
    monkeypatch.setattr(
        wsi.tiff.pages[0].__class__,
        "is_tiled",
        property(lambda _: False),  # skipcq: PYL-W0612
    )
    monkeypatch.setattr(wsi, "_m_info", None)
    assert pytest.approx(wsi.info.mpp, abs=0.1) == 0.5


def test_tiffreader_fallback_to_virtual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test fallback to VirtualWSIReader.

    Test fallback to VirtualWSIReader when TIFFWSIReader raises unsupported format.

    """

    class DummyTIFFWSIReader:
        def __init__(
            self,
            input_path: Path,
            mpp: tuple[float, float] | None = None,
            power: float | None = None,
            post_proc: str | None = None,
        ) -> None:
            _ = input_path
            _ = mpp
            _ = power
            _ = post_proc
            error_msg = "Unsupported TIFF WSI format"
            raise ValueError(error_msg)

    monkeypatch.setattr(wsireader, "TIFFWSIReader", DummyTIFFWSIReader)

    dummy_file = tmp_path / "dummy.tiff"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(dummy_file), dummy_img)

    reader = wsireader.WSIReader.try_tiff(dummy_file, ".tiff", None, None, None)
    assert isinstance(reader, wsireader.VirtualWSIReader)


def test_try_tiff_raises_other_valueerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test try_tiff raises ValueError if not an unsupported TIFF format."""
    tiff_path = tmp_path / "test.tiff"
    Image.new("RGB", (10, 10), color="white").save(tiff_path)

    # Patch TIFFWSIReader to raise a different ValueError
    def raise_other_valueerror(*args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs
        msg = "Some other TIFF error"
        raise ValueError(msg)

    monkeypatch.setattr(wsireader, "TIFFWSIReader", raise_other_valueerror)

    with pytest.raises(ValueError, match="Some other TIFF error"):
        wsireader.WSIReader.try_tiff(
            input_path=tiff_path,
            last_suffix=".tiff",
            mpp=(0.5, 0.5),
            power=20.0,
            post_proc=None,
        )


def test_parse_filtercolor_metadata_with_filter_pair() -> None:
    """Test full parsing including filter pair matching from XML metadata."""
    # We can't possibly test on all the different types of tiff files, so simulate them.
    xml_str = """
    <root>
        <FilterColors>
            <FilterColors-k>EM123_EX456</FilterColors-k>
            <FilterColors-v>255,128,0</FilterColors-v>
        </FilterColors>
        <ScanBands-i>
            <Bands-i>
                <Name>Channel1</Name>
            </Bands-i>
            <FilterPair>
                <EmissionFilter>
                    <FixedFilter>
                        <PartNumber>EM123</PartNumber>
                    </FixedFilter>
                </EmissionFilter>
                <ExcitationFilter>
                    <FixedFilter>
                        <PartNumber>EX456</PartNumber>
                    </FixedFilter>
                </ExcitationFilter>
            </FilterPair>
        </ScanBands-i>
    </root>
    """
    root = ElementTree.fromstring(xml_str)
    result = wsireader.TIFFWSIReader._parse_filtercolor_metadata(root)
    assert result is not None
    assert "Channel1" in result
    assert result["Channel1"] == (1.0, 128 / 255, 0.0)


def test_parse_scancolortable_rgb_and_named_colors() -> None:
    """Test parsing of ScanColorTable with RGB and named color values."""
    xml_str = """
    <root>
        <ScanColorTable>
            <ScanColorTable-k>FITC_Exc_Em</ScanColorTable-k>
            <ScanColorTable-v>0,255,0</ScanColorTable-v>
            <ScanColorTable-k>DAPI_Exc_Em</ScanColorTable-k>
            <ScanColorTable-v>Blue</ScanColorTable-v>
            <ScanColorTable-k>Cy3_Exc_Em</ScanColorTable-k>
            <ScanColorTable-v></ScanColorTable-v>
        </ScanColorTable>
    </root>
    """
    root = ElementTree.fromstring(xml_str)
    result = wsireader.TIFFWSIReader._parse_scancolortable(root)

    assert result is not None
    assert result["FITC"] == (0.0, 1.0, 0.0)
    assert result["DAPI"] == (0.0, 0.0, 1.0)
    assert result["Cy3"] is None  # Empty value is incluided but not converted


def test_get_namespace_extraction() -> None:
    """Test extraction of XML namespace from root tag."""
    # Case with namespace
    xml_with_ns = '<ome:OME xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06"></ome:OME>'
    root_with_ns = ElementTree.fromstring(xml_with_ns)
    result_with_ns = wsireader.TIFFWSIReader._get_namespace(root_with_ns)
    assert result_with_ns == {"ns": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    # Case without namespace
    xml_without_ns = "<OME></OME>"
    root_without_ns = ElementTree.fromstring(xml_without_ns)
    result_without_ns = wsireader.TIFFWSIReader._get_namespace(root_without_ns)
    assert result_without_ns == {}


def test_extract_dye_mapping() -> None:
    """Test extraction of dye mapping including missing and valid cases."""
    # Case with valid ChannelPriv entries
    xml_valid = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <StructuredAnnotations>
            <XMLAnnotation>
                <Value>
                    <ChannelPriv ID="Channel:0" FluorescenceChannel="FITC"/>
                    <ChannelPriv ID="Channel:1" FluorescenceChannel="DAPI"/>
                </Value>
            </XMLAnnotation>
        </StructuredAnnotations>
    </OME>
    """
    root_valid = ElementTree.fromstring(xml_valid)
    ns = {"ns": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    result_valid = wsireader.TIFFWSIReader._extract_dye_mapping(root_valid, ns)
    assert result_valid == {"Channel:0": "FITC", "Channel:1": "DAPI"}

    # Case with missing <Value>
    xml_missing_value = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <StructuredAnnotations>
            <XMLAnnotation>
            </XMLAnnotation>
        </StructuredAnnotations>
    </OME>
    """
    root_missing_value = ElementTree.fromstring(xml_missing_value)
    result_missing_value = wsireader.TIFFWSIReader._extract_dye_mapping(
        root_missing_value, ns
    )
    assert result_missing_value == {}

    # Case with ChannelPriv missing attributes
    xml_missing_attrs = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <StructuredAnnotations>
        <XMLAnnotation>
            <Value>
                <ChannelPriv FluorescenceChannel="FITC"/>
                <ChannelPriv ID="Channel:2"/>
            </Value>
        </XMLAnnotation>
        </StructuredAnnotations>
    </OME>
    """
    root_missing_attrs = ElementTree.fromstring(xml_missing_attrs)
    result_missing_attrs = wsireader.TIFFWSIReader._extract_dye_mapping(
        root_missing_attrs, ns
    )
    assert result_missing_attrs == {}


@pytest.mark.parametrize(
    ("color_int", "expected"),
    [
        (0xFF0000, (1.0, 0.0, 0.0)),  # Red
        (0x00FF00, (0.0, 1.0, 0.0)),  # Green
        (0x0000FF, (0.0, 0.0, 1.0)),  # Blue
        (-1, (1.0, 1.0, 1.0)),  # White (unsigned 32-bit)
    ],
)
def test_int_to_rgb(color_int: int, expected: tuple[float, float, float]) -> None:
    """Test conversion of integer color values to normalized RGB tuples."""
    result = wsireader.TIFFWSIReader._int_to_rgb(color_int)
    assert pytest.approx(result) == expected


def test_parse_channel_data() -> None:
    """Test parsing of channel metadata with valid color values."""
    xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image>
            <Pixels>
                <Channel ID="Channel:0" Name="DAPI" Color="16711680"/>
                <Channel ID="Channel:1" Name="FITC" Color="65280"/>
            </Pixels>
        </Image>
    </OME>
    """
    root = ElementTree.fromstring(xml)
    ns = {"ns": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    dye_mapping = {
        "Channel:0": "DAPI",
        "Channel:1": "FITC",
    }

    result = wsireader.TIFFWSIReader._parse_channel_data(root, ns, dye_mapping)
    assert result == [
        {
            "id": "Channel:0",
            "name": "DAPI",
            "rgb": (1.0, 0.0, 0.0),
            "dye": "DAPI",
            "label": "Channel:0: DAPI (DAPI)",
        },
        {
            "id": "Channel:1",
            "name": "FITC",
            "rgb": (0.0, 1.0, 0.0),
            "dye": "FITC",
            "label": "Channel:1: FITC (FITC)",
        },
    ]


def test_parse_channel_data_with_invalid_color() -> None:
    """Test parsing of channel metadata with an invalid color value."""
    xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image>
            <Pixels>
                <Channel ID="Channel:0" Name="DAPI" Color="16711680"/>
                <Channel ID="Channel:1" Name="FITC" Color="not_a_number"/>
            </Pixels>
        </Image>
    </OME>
    """
    root = ElementTree.fromstring(xml)
    ns = {"ns": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    dye_mapping = {
        "Channel:0": "DAPI",
        "Channel:1": "FITC",
    }

    result = wsireader.TIFFWSIReader._parse_channel_data(root, ns, dye_mapping)
    assert result == [
        {
            "id": "Channel:0",
            "name": "DAPI",
            "dye": "DAPI",
            "rgb": (1.0, 0.0, 0.0),
            "label": "Channel:0: DAPI (DAPI)",
        },
        {
            "id": "Channel:1",
            "name": "FITC",
            "dye": "FITC",
            "rgb": None,
            "label": "Channel:1: FITC (FITC)",
        },
    ]


def test_build_color_dict() -> None:
    """Test building of color dictionary with duplicate channel names."""
    channel_data = [
        {
            "id": "Channel:0",
            "name": "DAPI",
            "rgb": (1.0, 0.0, 0.0),
            "dye": "DAPI",
            "label": "Channel:0: DAPI (DAPI)",
        },
        {
            "id": "Channel:1",
            "name": "DAPI",
            "rgb": (0.0, 1.0, 0.0),
            "dye": "DAPI",
            "label": "Channel:1: DAPI (DAPI)",
        },
        {
            "id": "Channel:2",
            "name": "FITC",
            "rgb": (0.0, 0.0, 1.0),
            "dye": "FITC",
            "label": "Channel:2: FITC (FITC)",
        },
    ]

    dye_mapping = {
        "Channel:0": "DAPI",
        "Channel:1": "DAPI",
        "Channel:2": "FITC",
    }

    result = wsireader.TIFFWSIReader._build_color_dict(channel_data, dye_mapping)

    assert result == {
        "DAPI (DAPI)": (1.0, 0.0, 0.0),
        "DAPI (DAPI) [2]": (0.0, 1.0, 0.0),
        "FITC (FITC)": (0.0, 0.0, 1.0),
    }


def test_get_ome_objective_power_valid() -> None:
    """Test extraction of objective power from valid OME-XML."""
    xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Instrument ID="Instrument:0">
            <Objective ID="Objective:0" NominalMagnification="20.0"/>
        </Instrument>
        <Image>
            <InstrumentRef ID="Instrument:0"/>
            <ObjectiveSettings ID="Objective:0"/>
        </Image>
    </OME>
    """
    reader = wsireader.TIFFWSIReader.__new__(wsireader.TIFFWSIReader)
    reader.series_n = 0  # Required for _get_ome_mpp
    reader._get_ome_mpp = lambda _: [0.5, 0.5]  # Optional fallback mock
    result = reader._get_ome_objective_power(ElementTree.fromstring(xml))
    assert result == 20.0


def test_get_ome_objective_power_fallback_mpp() -> None:
    """Test fallback to MPP-based inference when objective power is missing."""
    xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image>
            <Pixels PhysicalSizeX="0.5" PhysicalSizeY="0.5"/>
        </Image>
    </OME>
    """
    reader = wsireader.TIFFWSIReader.__new__(wsireader.TIFFWSIReader)
    reader._get_ome_mpp = lambda _: [0.5, 0.5]  # Mock MPP extraction
    result = reader._get_ome_objective_power(ElementTree.fromstring(xml))
    assert result == 20.0  # Assuming mpp2common_objective_power(0.5) == 20.0


def test_get_ome_objective_power_none() -> None:
    """Test full fallback when both objective power and MPP are missing."""
    xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image>
            <Pixels/>
        </Image>
    </OME>
    """
    reader = wsireader.TIFFWSIReader.__new__(wsireader.TIFFWSIReader)
    reader._get_ome_mpp = lambda _: None  # Mock missing MPP
    result = reader._get_ome_objective_power(ElementTree.fromstring(xml))
    assert result is None


def test_handle_tiff_wsi_returns_tiff_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _handle_tiff_wsi returns TIFFWSIReader for valid TIFF image."""
    # Create a valid TIFF image using PIL
    tiff_path = tmp_path / "dummy.tiff"
    image = Image.new("RGB", (10, 10), color="white")
    image.save(tiff_path)

    # Patch is_tiled_tiff to return True
    monkeypatch.setattr(wsireader, "is_tiled_tiff", lambda _: True)

    # Patch TIFFWSIReader.__init__ to bypass internal checks
    with patch(
        "tiatoolbox.wsicore.wsireader.TIFFWSIReader.__init__", return_value=None
    ):
        reader = wsireader._handle_tiff_wsi(
            input_path=tiff_path,
            mpp=(0.5, 0.5),
            power=20.0,
            post_proc=None,
        )
        assert isinstance(reader, wsireader.TIFFWSIReader)


def raise_openslide_error(*args: object, **kwargs: object) -> None:
    """Simulate OpenSlideWSIReader raising an OpenSlideError."""
    _ = args
    _ = kwargs
    msg = "mock error"
    raise wsireader.openslide.OpenSlideError(msg)


def test_handle_tiff_wsi_openslide_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test _handle_tiff_wsi when OpenSlideWSIReader raises."""
    # Create a valid TIFF image
    tiff_path = tmp_path / "test.tiff"
    Image.new("RGB", (10, 10), color="white").save(tiff_path)

    # Patch detect_format to return a non-None value
    monkeypatch.setattr(wsireader.openslide.OpenSlide, "detect_format", lambda _: "SVS")

    # Patch OpenSlideWSIReader to raise OpenSlideError
    monkeypatch.setattr(wsireader, "OpenSlideWSIReader", raise_openslide_error)

    # Patch is_tiled_tiff to return True so fallback to TIFFWSIReader is triggered
    monkeypatch.setattr(wsireader, "is_tiled_tiff", lambda _: True)

    # Patch TIFFWSIReader.__init__ to bypass internal checks
    with patch(
        "tiatoolbox.wsicore.wsireader.TIFFWSIReader.__init__", return_value=None
    ):
        result = wsireader._handle_tiff_wsi(
            input_path=tiff_path,
            mpp=(0.5, 0.5),
            power=20.0,
            post_proc=None,
        )
        assert isinstance(result, wsireader.TIFFWSIReader)


def test_handle_tiff_wsi_openslide_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test _handle_tiff_wsi returns OpenSlideWSIReader when detect_format is valid."""
    # Create a valid TIFF image
    tiff_path = tmp_path / "test.tiff"
    Image.new("RGB", (10, 10), color="white").save(tiff_path)

    # Patch detect_format to return a valid format
    monkeypatch.setattr(wsireader.openslide.OpenSlide, "detect_format", lambda _: "SVS")

    # Patch OpenSlideWSIReader.__init__ to bypass actual init logic
    with patch.object(wsireader.OpenSlideWSIReader, "__init__", return_value=None):
        result = wsireader._handle_tiff_wsi(
            input_path=tiff_path,
            mpp=(0.5, 0.5),
            power=20.0,
            post_proc="auto",
        )
        assert isinstance(result, wsireader.OpenSlideWSIReader)
