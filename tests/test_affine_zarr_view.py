"""Tests for affine zarr views."""
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.wsicore.wsireader import NGFFWSIReader
from tiatoolbox.wsicore.zarr_views import AffineZarrView


def test_affine_zarr_view():
    """Check that a transform does not error when read."""
    ngff = _fetch_remote_sample("ngff-1")
    wsi = NGFFWSIReader(ngff)
    # Rotation (22.5 degrees) and translation (+256 in x and y) transform
    translation = (256, 256)
    transform = np.array(
        [
            [+np.cos(np.pi / 8), -np.sin(np.pi / 8), translation[1]],
            [+np.sin(np.pi / 8), +np.cos(np.pi / 8), translation[0]],
            [0, 0, 1],
        ]
    )
    view = AffineZarrView(wsi.level_arrays[0], transform)

    t0 = perf_counter()
    region_1 = wsi.read_region(translation, 0, (1024, 1024))
    t1 = perf_counter()
    dt_region_1 = t1 - t0

    t0 = perf_counter()

    # Extract a region with the transform applied
    region_2 = view[:1024, :1024, :3]

    t1 = perf_counter()
    dt_region_2 = t1 - t0

    region_bounds = np.array(
        [
            [0, 0],
            [1024, 0],
            [1024, 1024],
            [0, 1024],
            [0, 0],
        ]
    )
    region_bounds_trans = np.stack(
        [*region_bounds.T, np.ones_like(region_bounds.T[0])]
    ).T
    region_bounds_trans = np.dot(transform, region_bounds_trans.T).T[:, :2]

    _, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs[0].imshow(wsi.level_arrays[0][:, :, :3])
    axs[0].plot(region_bounds_trans[:, 1], region_bounds_trans[:, 0], "r-", lw=1)
    axs[0].plot(
        region_bounds[:, 1] + translation[1],
        region_bounds[:, 0] + translation[0],
        "g-",
        lw=1,
    )
    axs[0].set_title("Original WSI with\n" "Region Bounds Overlaid")

    axs[1].imshow(region_2)
    axs[1].set_title(
        (
            "AffineZarrView Extracted Region\n(Affine Transform)\n"
            f"Time: {dt_region_2:.2f}s"
        )
    )
    for spine in ("top", "right", "bottom", "left"):
        axs[1].spines[spine].set_color("red")

    axs[2].imshow(region_1)
    axs[2].set_title("NGFFWSIReader\n(Translation Only)\n" f"Time: {dt_region_1:.2f}s")
    for spine in ("top", "right", "bottom", "left"):
        axs[2].spines[spine].set_color("green")

    plt.suptitle("test_affine_zarr_view")
    plt.savefig("test_affine_zarr_view.png")
