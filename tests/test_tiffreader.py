"""Test TIFFWSIReader."""

from typing import Callable

import pytest
from defusedxml import ElementTree

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
