"""Unit test package for architecture utilities."""

import dask.array as da
import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.utils import (
    UpSample2x,
    Attention,
    centre_crop,
    centre_crop_to_shape,
    peak_detection_map_overlap,
    nms_on_detection_maps,
)


def test_all() -> None:
    """Contains all tests for now."""
    layer = UpSample2x()
    sample = np.array([[1, 2], [3, 4]])[..., None]
    batch = torch.from_numpy(sample)[None]
    batch = batch.permute(0, 3, 1, 2).type(torch.float32)
    output = layer(batch).permute(0, 2, 3, 1)[0].numpy()
    _output = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
    )
    assert np.sum(_output - output) == 0

    with pytest.raises(ValueError, match=r".*Unknown.*format.*"):
        centre_crop(_output[None, :, :, None], [2, 2], "NHWCT")

    x = centre_crop(_output[None, :, :, None], [2, 2], "NHWC")
    assert np.sum(x[0, :, :, 0] - sample) == 0
    x = centre_crop(_output[None, None, :, :], [2, 2], "NCHW")
    assert np.sum(x[0, 0, :, :] - sample) == 0


def test_centre_crop_operators() -> None:
    """Test for crop et. al. ."""
    sample = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    output = centre_crop(sample, [3, 3], data_format="NCHW")
    assert torch.sum(output - sample[:, :, 1:13, 1:13]) == 0, f"{output.shape}"

    sample = torch.rand((1, 15, 15, 3), dtype=torch.float32)
    output = centre_crop(sample, [3, 3], data_format="NHWC")
    assert torch.sum(output - sample[:, 1:13, 1:13, :]) == 0, f"{output.shape}"

    # *
    x = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    y = x[:, :, 6:9, 6:9]
    output = centre_crop_to_shape(x, y, data_format="NCHW")
    assert torch.sum(output - y) == 0, f"{output.shape}"

    x = torch.rand((1, 15, 15, 3), dtype=torch.float32)
    y = x[:, 6:9, 6:9, :]
    output = centre_crop_to_shape(x, y, data_format="NHWC")
    assert torch.sum(output - y) == 0, f"{output.shape}"

    with pytest.raises(ValueError, match=r".*Unknown.*format.*"):
        centre_crop_to_shape(x, y, data_format="NHWCT")

    x = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    y = x[:, :, 6:9, 6:9]
    with pytest.raises(ValueError, match=r".*Height.*smaller than `y`*"):
        centre_crop_to_shape(y, x, data_format="NCHW")


def test_peak_detection() -> None:
    """Test for peak detection."""
    min_distance = 3
    threshold_abs = 0.5

    heatmap = np.zeros((7, 7, 1), dtype=np.float32)

    peak_map = peak_detection_map_overlap(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )
    assert np.sum(peak_map) == 0.0  # No peaks

    heatmap[0, 0, 0] = 0.9  # First peak
    heatmap[0, 1, 0] = 0.6  # Too close to first peak
    heatmap[1, 0, 0] = 0.6  # Too close to first peak
    heatmap[2, 2, 0] = 0.9  # Too close to first peak
    heatmap[3, 3, 0] = 0.9  # Second peak

    peak_map = peak_detection_map_overlap(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )
    assert peak_map[0, 0, 0] == 1.0
    assert peak_map[3, 3, 0] == 1.0
    assert np.sum(peak_map) == 2.0


def test_peak_detection_map_overlap() -> None:
    """Test for peak detection with da.map_overlap."""
    heatmap = np.zeros((7, 7, 1), dtype=np.float32)
    heatmap[0, 0, 0] = 0.9  # First peak
    heatmap[0, 1, 0] = 0.6  # Too close to first peak
    heatmap[1, 0, 0] = 0.6  # Too close to first peak
    heatmap[2, 2, 0] = 0.9  # Too close to first peak
    heatmap[3, 3, 0] = 0.9  # Second peak

    min_distance = 3
    threshold_abs = 0.5

    # Add halo (overlap) around each block for post-processing
    depth_h = min_distance
    depth_w = min_distance
    depth = {0: depth_h, 1: depth_w, 2: 0}

    # Test chunk is entire heatmap
    da_heatmap = da.from_array(heatmap, chunks=(7, 7, 1))

    da_peak_map = da.map_overlap(
        da_heatmap,
        peak_detection_map_overlap,
        depth=depth,
        boundary=0,
        dtype=np.float32,
        block_info=True,
        depth_h=depth_h,
        depth_w=depth_w,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
    )

    peak_map = da_peak_map.compute()

    assert peak_map[0, 0, 0] == 1.0
    assert peak_map[3, 3, 0] == 1.0
    assert np.sum(peak_map) == 2.0

    # Test small chunk with halo
    # using very small chunk sizes (1,1,1) to force multiple overlaps
    da_heatmap = da_heatmap.rechunk({0: 1, 1: 1, 2: 1})

    da_peak_map = da.map_overlap(
        da_heatmap,
        peak_detection_map_overlap,
        depth=depth,
        boundary=0,
        dtype=np.float32,
        block_info=True,
        depth_h=depth_h,
        depth_w=depth_w,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
    )

    peak_map = da_peak_map.compute()

    assert peak_map[0, 0, 0] == 1.0
    assert peak_map[3, 3, 0] == 1.0
    assert np.sum(peak_map) == 2.0


def test_nms_on_detection_maps() -> None:
    """Test for NMS on detection maps."""
    heatmap = np.zeros((7, 7, 3), dtype=np.float32)
    nms_map = nms_on_detection_maps(heatmap, min_distance=3)
    assert np.sum(nms_map) == 0.0  # No peaks

    heatmap[0, 0, 0] = 0.9  # Peak in channel 0 (valid)
    heatmap[0, 1, 0] = 0.6  # Peak in channel 0 (suppressed)
    heatmap[0, 0, 1] = 0.8  # Peak in channel 1 (suppressed)

    heatmap[5, 5, 2] = 0.9  # Peak in channel 2 (valid)
    heatmap[4, 4, 1] = 0.7  # Peak in channel 1 (suppressed)

    nms_map = nms_on_detection_maps(heatmap, min_distance=3)
    assert nms_map[0, 0, 0] == 0.9
    assert nms_map[5, 5, 2] == 0.9


def test_attention_module() -> None:
    """Test for Attention module."""

    test_input = torch.zeros((1, 16, 32, 32), dtype=torch.float32)
    
    # Default to identity
    attention = Attention(name=None, in_channels=16)
    output = attention(test_input)
    assert torch.sum(output - test_input) == 0

    attention = Attention(name="scse", in_channels=16, reduction=4)
    output = attention(test_input)
    assert output.shape == test_input.shape

    with pytest.raises(ValueError, match=r"Attention random_name is not implemented"):
        attention = Attention(name="random_name", in_channels=16)

