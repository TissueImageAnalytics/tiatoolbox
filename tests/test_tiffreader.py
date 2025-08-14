"""Test TIFFWSIReader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
import pytest
from defusedxml import ElementTree

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
