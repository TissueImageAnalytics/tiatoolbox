from defusedxml import ElementTree

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.wsicore import wsireader


def test_ome_missing_instrument_ref(monkeypatch):
    """Test that an OME-TIFF can be read without instrument reference."""
    sample = _fetch_remote_sample("ome-brightfield-pyramid-1-small")
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


def test_ome_missing_physicalsize(monkeypatch):
    """Test that an OME-TIFF can be read without physical size."""
    sample = _fetch_remote_sample("ome-brightfield-pyramid-1-small")
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
