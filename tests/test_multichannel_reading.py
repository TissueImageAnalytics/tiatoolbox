"""Tests for multichannel image reading and visualisation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from defusedxml import ElementTree

from tiatoolbox import utils
from tiatoolbox.utils import postproc_defs
from tiatoolbox.wsicore import wsireader


def test_multichannel_basic_read() -> None:
    """VirtualWSIReader: >4-channel input is reduced to RGB when using auto post-proc.

    Also verify that disabling post-proc preserves original channel count.

    """
    # Create a small synthetic 6-channel image
    rng = np.random.default_rng()
    img = rng.integers(0, 255, size=(64, 64, 6), dtype=np.uint8)
    meta = wsireader.WSIMeta(slide_dimensions=img.shape[:2][::-1], axes="YXS")

    # By default, the VirtualWSIReader will detect "feature" mode for
    # non-RGB(A) shapes and will not apply the auto post-processing.
    v_wsi = wsireader.VirtualWSIReader(img, info=meta)
    out = v_wsi.read_rect((0, 0), (16, 16))
    assert isinstance(out, np.ndarray)
    assert out.ndim == 3
    # Without explicitly setting post_proc/mode we should get the native channels
    assert out.shape[2] == 6

    # If we explicitly set a MultichannelToRGB post-proc and force rgb mode,
    # the output should be converted to 3 channels.
    v_wsi.post_proc = postproc_defs.MultichannelToRGB()
    v_wsi.mode = "rgb"
    out_rgb = v_wsi.read_rect((0, 0), (16, 16))
    assert out_rgb.shape[2] == 3


def test_ngff_multichannel_read(remote_sample: callable) -> None:
    """Sanity check for NGFF reader with possible multichannel data.

    This test is permissive: it asserts the reader returns an ndarray and,
    if the native dataset has many channels, that applying
    `MultichannelToRGB` produces a 3-channel output.

    """
    ngff_path = remote_sample("ngff-1")
    wsi = wsireader.NGFFWSIReader(ngff_path)

    # Read a tiny region to avoid heavy IO
    size = (8, 8)
    region = wsi.read_rect((0, 0), size)
    assert isinstance(region, np.ndarray)

    # If dataset has many channels, converting should yield 3 channels
    if region.ndim == 3 and region.shape[2] >= 5:
        wsi.post_proc = postproc_defs.MultichannelToRGB()
        region_rgb = wsi.read_rect((0, 0), size)
        assert region_rgb.ndim == 3
        assert region_rgb.shape[2] == 3


def test_read_multi_channel(source_image: Path) -> None:
    """Test reading image with more than three channels.

    Create a virtual WSI by concatenating the source_image.

    """
    img_array = utils.misc.imread(Path(source_image))
    new_img_array = np.concatenate((img_array, img_array), axis=-1)

    new_img_size = new_img_array.shape[:2][::-1]
    meta = wsireader.WSIMeta(slide_dimensions=new_img_size, axes="YXS", mpp=(0.5, 0.5))
    wsi = wsireader.VirtualWSIReader(new_img_array, info=meta)

    region = wsi.read_rect(
        location=(0, 0),
        size=(50, 100),
        pad_mode="reflect",
        units="mpp",
        resolution=0.25,
    )
    target = cv2.resize(
        new_img_array[:50, :25, :],
        (50, 100),
        interpolation=cv2.INTER_CUBIC,
    )

    assert region.shape == (100, 50, (new_img_array.shape[-1]))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_visualise_multi_channel(
    sample_qptiff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test visualising a multi-channel qptiff multiplex image."""
    calls = {"bg": 0}

    def fake_bg_composite(*, image: np.ndarray) -> np.ndarray:
        """Fake background_composite to record calls."""
        calls["bg"] += 1
        return image

    monkeypatch.setattr(utils.transforms, "background_composite", fake_bg_composite)

    wsi = wsireader.TIFFWSIReader(sample_qptiff, post_proc="auto")
    wsi2 = wsireader.TIFFWSIReader(sample_qptiff, post_proc=None)

    region = wsi.read_rect(location=(0, 0), size=(50, 100))
    region2 = wsi2.read_rect(location=(0, 0), size=(50, 100))

    assert region.shape == (100, 50, 3)
    assert region2.shape == (100, 50, 5)
    assert region2.shape[0:2] == (100, 50)
    assert region2.shape[-1] >= 4  # robust vs variation (was hard-coded 5)

    # In the TIFF delegate path, background_composite must NOT be called
    assert calls["bg"] == 0


def test_tiff_post_proc_auto_is_multichannel(sample_qptiff: Path) -> None:
    """Test that post_proc='auto' yields MultichannelToRGB for multiplex images."""
    r = wsireader.TIFFWSIReader(sample_qptiff, post_proc="auto")
    assert isinstance(r.post_proc, postproc_defs.MultichannelToRGB)


def test_get_post_proc_variants() -> None:
    """Test different branches of get_post_proc method."""
    reader = wsireader.VirtualWSIReader(np.zeros((10, 10, 3)))

    assert callable(reader.get_post_proc(lambda x: x))
    assert reader.get_post_proc(None) is None
    assert isinstance(reader.get_post_proc("auto"), postproc_defs.MultichannelToRGB)
    assert isinstance(
        reader.get_post_proc("MultichannelToRGB"), postproc_defs.MultichannelToRGB
    )

    with pytest.raises(ValueError, match="Invalid post-processing function"):
        reader.get_post_proc("invalid_proc")


OME_MINI = """
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="dummy">
    <Pixels SizeX="4" SizeY="4" SizeC="3" Type="uint8">
      <!-- OME Color is 24-bit: B + 256*G + 65536*R -->
      <Channel ID="Channel:0" Name="DAPI"      Color="255" />
      <Channel ID="Channel:1" Name="Alexa488"  Color="65280" />
      <Channel ID="Channel:2" Name="TRITC"     Color="16711680" />
    </Pixels>
  </Image>
</OME>
"""


def test_tiff_ome_channel_color_parsing_minimal_xml() -> None:
    """Test TIFFWSIReader OME channel color parsing with minimal OME XML."""
    root = ElementTree.fromstring(OME_MINI)

    # Private, but stable on the TIFF path:
    parse_channels = wsireader.TIFFWSIReader._parse_channel_data  # type: ignore[attr-defined]
    build_dict = wsireader.TIFFWSIReader._build_color_dict  # type: ignore[attr-defined]

    # Derive namespace from the root tag: "{uri}OME" -> uri
    if root.tag.startswith("{") and "}":
        uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")]
    else:
        uri = "http://www.openmicroscopy.org/Schemas/OME/2016-06"  # sensible default
    ns = {"ns": uri}

    channel_data = parse_channels(root, ns=ns, dye_mapping={})
    color_dict = build_dict(channel_data, {})

    # Helper to compare whether parser returns 0 to 1 floats or 0 to 255 ints
    def as_255(rgb: str) -> tuple[int, int, int]:
        """Convert comma-separated RGB string to 0-255 integer tuple."""
        vals = [float(c) for c in rgb]
        return tuple(round(c * 255) if max(vals) <= 1.0 else round(c) for c in vals)

    assert as_255(color_dict["DAPI"]) == (0, 0, 255)
    assert as_255(color_dict["Alexa488"]) == (0, 255, 0)
    assert as_255(color_dict["TRITC"]) == (255, 0, 0)
