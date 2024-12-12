"""Simple benchmark to test the memory allocation of annotation stores.

This is to be run as a standalone script to avoid inteference from the
jupyter notebook environment. Tracking memory allocation in Python is
quick tricky and this is not perfect. Two methods are used here for
comparison. The first is simply to record the process memory usage
before and after the benchmark using psutil. The second is to use the
memray tool to track memory allocations.

psutil: https://psutil.readthedocs.io/en/latest/
memray: https://github.com/bloomberg/memray (Linux only)

Command Line Usage
==================

```
usage: annotation_store_alloc.py [-h] [-S SIZE SIZE] [-s {dict,sqlite}] [-m]

optional arguments:
  -h, --help            show this help message and exit
  -S SIZE SIZE, --size SIZE SIZE
                        The size of the grid of cells to generate. Defaults to
                        (100, 100).
  -s {dict,sqlite}, --store {dict,sqlite}
                        The type of annotation store to use. Defaults to
                        'dict'.
  -m, --in-memory       Use an in-memory store.
```


Example Outputs For 100x100 Grid (10,000 Annotations)
=====================================================

Peak Memory Usage In MiB (psutil/memray)
----------------------------------------

| store  | in-memory | on-disk   |
| ------ | --------- | --------- |
| dict   | 21.0/18.0 | 24.2/19.0 |
| sqlite | 16.8/6.4  | 6.8/2.7   |

````
  dict (in-memory): ##################
sqlite (in-memory): ######
  sqlite (on-disk): ###
````


File System Usage In MiB (on-disk)
----------------------------------

| store  | file-system |
| ------ | ----------- |
| dict   | 9.02        |
| sqlite | 5.34        |

```
  dict: #########
sqlite: #####
```

Example Outputs For 200x200 Grid (40,000 Annotations)
=====================================================

Peak Memory Usage In MiB (psutil/memray)
----------------------------------------

| store  | in-memory | on-disk   |
| ------ | --------- | --------- |
| dict   | 74.8/71.4 | 88.9/75.2 |
| sqlite | 33.2/23.2 | 9.64/2.7  |

```
  dict (mem): ########################################################################
sqlite (mem): ########################
sqlite (dsk): ###
```

File System Usage In MiB (on-disk)
----------------------------------

| store  | file-system |
| ------ | ----------- |
| dict   | 35.77       |
| sqlite | 21.09       |

```
  dict: ####################################
sqlite: #####################
```

"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

sys.path.append("../")

try:
    import memray
except ImportError:

    class memray:  # noqa: N801 No CapWords convention
        """Dummy memray module for when memray is not installed.

        A drop-in dummy replacement for the memray module on unsupported
        platforms.
        """

        dummy: bool = True

        class Tracker:
            """Dummy Tracker context manager."""

            def __init__(
                self: memray,
                *args: list[Any],  # noqa: ARG002
                **kwargs: dict[str, Any],  # noqa: ARG002
            ) -> None:
                """Initialize :class:`Tracker`."""
                warnings.warn("Memray not installed, skipping tracking.", stacklevel=2)

            def __enter__(self: memray) -> None:
                """Dummy enter method."""
                # Intentionally blank.

            def __exit__(self: memray, *args: object) -> None:
                """Dummy exit method."""
                # Intentionally blank.


import numpy as np
import psutil
from shapely.geometry import Polygon
from tqdm import tqdm

from tiatoolbox.annotation.storage import (
    Annotation,
    DictionaryStore,
    SQLiteStore,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator
    from numbers import Number


def cell_polygon(
    xy: tuple[Number, Number],
    n_points: int = 20,
    radius: Number = 8,
    noise: Number = 0.01,
    eccentricity: tuple[Number, Number] = (1, 3),
    direction: str = "CCW",
    seed: int = 0,
    *,
    repeat_first: bool = True,
    round_coords: bool = False,
) -> Polygon:
    """Generate a fake cell boundary polygon.

    Borrowed from tiatoolbox unit tests.

    Cell boundaries are generated an ellipsoids with randomised eccentricity,
    added noise, and a random rotation.

    Args:
        xy (tuple(int)): The x,y centre point to generate the cell boundary around.
        n_points (int): Number of points in the boundary. Defaults to 20.
        radius (float): Radius of the points from the centre. Defaults to 10.
        noise (float): Noise to add to the point locations. Defaults to 1.
        eccentricity (tuple(float)): Range of values (low, high) to use for
            randomised eccentricity. Defaults to (1, 3).
        repeat_first (bool): Enforce that the last point is equal to the first.
        direction (str): Ordering of the points. Defaults to "CCW". Valid options
            are: counter-clockwise "CCW", and clockwise "CW".
        seed (int): Seed for the random number generator. Defaults to 0.
        round_coords (bool): Round coordinates to integers. Defaults to False.

    """
    from shapely import affinity

    rand_state = np.random.default_rng().__getstate__()
    rng = np.random.default_rng(seed)
    if repeat_first:
        n_points -= 1

    # Generate points about an ellipse with random eccentricity
    x, y = xy
    alpha = np.linspace(0, 2 * np.pi - (2 * np.pi / n_points), n_points)
    rx = radius * (rng.random() + 0.5)
    ry = rng.uniform(*eccentricity) * radius - 0.5 * rx
    x = rx * np.cos(alpha) + x + (rng.random(n_points) - 0.5) * noise
    y = ry * np.sin(alpha) + y + (rng.random(n_points) - 0.5) * noise
    boundary_coords = np.stack([x, y], axis=1).astype(int).tolist()

    # Copy first coordinate to the end if required
    boundary_coords_0 = [boundary_coords[0]]
    if repeat_first:
        boundary_coords = boundary_coords + boundary_coords_0

    # Swap direction
    if direction.strip().lower() == "cw":
        boundary_coords = boundary_coords[::-1]

    polygon = Polygon(boundary_coords)

    # Add random rotation
    angle = rng.random() * 360
    polygon = affinity.rotate(polygon, angle, origin="centroid")

    # Round coordinates to integers
    if round_coords:
        polygon = Polygon(np.array(polygon.exterior.coords).round())

    # Restore the random state
    np.random.default_rng().__setstate__(rand_state)

    return polygon


def cell_grid(
    size: tuple[int, int] = (10, 10),
    spacing: Number = 25,
) -> Generator[Polygon, None, None]:
    """Generate a grid of cell boundaries."""
    return (
        cell_polygon(xy=np.multiply(ij, spacing), repeat_first=False, seed=n)
        for n, ij in enumerate(np.ndindex(size))
    )


STORES = {
    "dict": DictionaryStore,
    "sqlite": SQLiteStore,
}


def main(
    store: str,
    size: tuple[int, int],
    *,
    in_memory: bool,
) -> None:
    """Run the benchmark.

    Args:
        store (str): The store to use. Valid options are:
            - dict: In-memory dictionary store.
            - sqlite: SQLite store.
        in_memory (bool): Whether to use in-memory stores.
        size (tuple(int)): The size of the grid to generate.
    """
    process = psutil.Process(os.getpid())
    cls = STORES[store]
    tracker_filepath = Path(f"{store}-in-mem-{in_memory}.bin".lower())
    if tracker_filepath.is_file():
        tracker_filepath.unlink()

    with (
        NamedTemporaryFile(mode="w+") as temp_file,
        memray.Tracker(
            tracker_filepath,
            native_traces=True,
            follow_fork=True,
        ),
    ):
        io = ":memory:" if in_memory else temp_file  # Backing (memory/disk)
        print(f"Storing {size[0] * size[1]} cells")
        print(f"Using {cls.__name__}({io})")

        # Record memory usage before creating the store
        psutil_m0 = process.memory_info().rss

        store = cls(io)
        # Set up a polygon generator
        grid = cell_grid(size=tuple(size), spacing=35)
        # Store the grid of polygons
        for polygon in tqdm(grid, total=size[0] * size[1]):
            _ = store.append(copy.deepcopy(Annotation(Polygon(polygon))))
        # Ensure the store is flushed to disk if using a disk-based store
        if not in_memory:
            store.commit()

        # Print file size in MiB
        temp_file.seek(0, os.SEEK_END)
        file_size = temp_file.tell()
        print(f"File size: {file_size / (1024**2):.2f} MiB")

        # Print memory usage in MiB
        psutil_total_mem = process.memory_info().rss - psutil_m0
        print(f"Psutil Memory: {psutil_total_mem / (1024**2): .2f} MiB")

        # Print memory usage in MiB from memray
        if hasattr(memray, "dummy") and memray.dummy:
            # Skip memray if not installed
            return
        regex = re.compile(r"Total memory allocated:\s*([\d.]+)MB")
        pipe = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "memray",
                "stats",
                tracker_filepath.name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = pipe.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
            sys.exit(-1)
        memray_total_mem_str = regex.search(stdout.decode("utf-8")).group(1).strip()
        memray_total_mem = float(memray_total_mem_str)
        print(f"Memray Memory: {memray_total_mem: .2f} MiB")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description=(
            "Simple benchmark to test the memory allocation of annotation stores."
        ),
    )
    PARSER.add_argument(
        "-S",
        "--size",
        type=int,
        nargs=2,
        default=(100, 100),
        help="The size of the grid of cells to generate. Defaults to (100, 100).",
    )
    PARSER.add_argument(
        "-s",
        "--store",
        type=str,
        default="dict",
        help="The type of annotation store to use. Defaults to 'dict'.",
        choices=["dict", "sqlite"],
    )
    PARSER.add_argument(
        "-m",
        "--in-memory",
        help="Use an in-memory store.",
        action="store_true",
    )

    # Parsed CLI arguments
    ARGS = PARSER.parse_args()
    # Run the benchmark
    main(
        store=ARGS.store,
        in_memory=ARGS.in_memory,
        size=ARGS.size,
    )
