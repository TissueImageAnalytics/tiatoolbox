"""Benchmarking for annotation storage nquery."""
from numbers import Number
from time import perf_counter
from typing import Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    DictionaryStore,
    SQLiteStore,
)


# Copied from tests, refactor into a common location?
def cell_polygon(
    xy: Tuple[Number, Number],
    n_points: int = 20,
    radius: Number = 10,
    noise: Number = 0.01,
    eccentricity: Tuple[Number, Number] = (1, 3),
    repeat_first: bool = True,
    direction: str = "CCW",
) -> Polygon:
    """Generate a fake cell boundary polygon.

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

    """
    if repeat_first:
        n_points -= 1

    # Generate points about an ellipse with random eccentricity
    x, y = xy
    alpha = np.linspace(0, 2 * np.pi - (2 * np.pi / n_points), n_points)
    rx = radius * (np.random.rand() + 0.5)
    ry = np.random.uniform(*eccentricity) * radius - 0.5 * rx
    x = rx * np.cos(alpha) + x + (np.random.rand(n_points) - 0.5) * noise
    y = ry * np.sin(alpha) + y + (np.random.rand(n_points) - 0.5) * noise
    boundary_coords = np.stack([x, y], axis=1).tolist()

    # Copy first coordinate to the end if required
    if repeat_first:
        boundary_coords = boundary_coords + [boundary_coords[0]]

    # Swap direction
    if direction.strip().lower() == "cw":
        boundary_coords = boundary_coords[::-1]

    polygon = Polygon(boundary_coords)

    # Add random rotation
    angle = np.random.rand() * 360
    return affinity.rotate(polygon, angle, origin="centroid")


def main():  # noqa: CCR001
    """Run the benchmark."""
    spacing = 30
    radius = 5
    results = {}
    classes = (SQLiteStore, DictionaryStore)
    max_cls_name = max(len(cls.__name__) for cls in classes)
    modes = ("poly-poly", "box-box", "boxpoint-boxpoint")
    max_mode_name = max(len(mode) for mode in modes)

    with open("nquery-benchmark.csv", "w") as fh:
        header = " , ".join(["n", "time", "class", "mode"])
        print(header)
        fh.write(header + "\n")

        for grid_size in range(10, 60, 10):
            for cls in classes:
                grid = np.ndindex((grid_size, grid_size))
                store: AnnotationStore = cls()

                # Add annotations
                for x, y in grid:
                    cell_a = cell_polygon(
                        (x * spacing + radius, y * spacing + radius), radius=radius
                    )
                    ann_a = Annotation(cell_a, {"class": "A"})
                    cell_b = cell_polygon(
                        (x * spacing + radius, y * spacing + radius), radius=radius
                    )
                    ann_b = Annotation(cell_b, {"class": "B"})

                    store[f"A_{x}_{y}"] = ann_a
                    store[f"B_{x}_{y}"] = ann_b

                # Validate annotations were added
                if len(store) != grid_size**2 * 2:
                    raise ValueError("Not all annotations were added.")

                for mode in modes:
                    # Query
                    t0 = perf_counter()
                    result = store.nquery(
                        where="props['class'] == 'A'",
                        n_where="props['class'] == 'B'",
                        distance=2,
                        mode=mode,
                    )
                    t1 = perf_counter()
                    dt = t1 - t0

                    # Validate
                    if not isinstance(result, dict):
                        raise ValueError("Result is not a dictionary.")
                    if len(result) != grid_size**2:
                        raise ValueError("Result does not contain all annotations.")
                    for v in result.values():
                        if len(v) != 1:
                            raise ValueError(
                                "Result does not contain the correct number of "
                                "annotations."
                            )

                    # Store results
                    results[(grid_size, cls.__name__)] = dt
                    csv_line = " , ".join(
                        [
                            f"{grid_size:>3}",
                            f"{f'{dt:>3.2f}':>8}",
                            f"{cls.__name__: <{max_cls_name}}",
                            f"{mode: <{max_mode_name}}",
                        ]
                    )
                    print(csv_line)
                    fh.write(csv_line + "\n")


def plot_csv():
    """Plot the results from the benchmark."""
    import pandas as pd
    from matplotlib import pyplot as plt

    # Use latex for rendering
    plt.rc("text", usetex=True)
    # Use serif font
    plt.rc("font", family="serif")

    data = pd.read_csv(
        "nquery-benchmark.csv", sep=r"\s*,\s*", header=0, engine="python"
    )
    data = data.pivot(index=["n", "class"], columns="mode", values="time")
    mode_groups = data.groupby("class")
    # Plot each class (group) in a different colour and each mode
    # (series) in a different line style.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for n, (cls, group) in enumerate(mode_groups):
        for m, (mode, series) in enumerate(group.items()):
            for a, ax in enumerate(axs):
                ax.plot(
                    range(len(series.index)),
                    series,
                    label=f"{cls}\n({mode})" if a == 0 else None,
                    c=f"C{n}",
                    linestyle=["--", ":", "-"][m],
                )
                ax.grid("on")
                ax.set_xlabel("$n$")
                ax.set_ylabel("Time (s)")
                ax.set_xticks(
                    range(len(group.index)),
                    labels=group.index.get_level_values("n"),
                )
    fig.legend(loc="outside right")
    axs[1].set_yscale("log")
    axs[0].set_title("Linear Scale")
    axs[1].set_title("Log Scale")
    plt.suptitle(
        "Neighbourhood Query Performance For Two Overlaid"
        " $n \\times n$ Grids of Polygons"
    )
    plt.savefig("nquery-benchmark.png")


if __name__ == "__main__":
    main()
    plot_csv()
