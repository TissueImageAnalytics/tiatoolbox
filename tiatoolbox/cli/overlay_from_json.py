"""Command line interface for overlay_from_json."""
from pathlib import Path
import click

from tiatoolbox.cli.common import (
    cli_img_input,
    cli_output_path,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)


def numpy2vips(a):
    import pyvips

    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands, dtype_to_format[str(a.dtype)]
    )
    return vi


def process_one_slide(img_input, output_path, json_path, score_path):
    from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader
    import os
    import json
    import numpy as np
    import cv2
    import matplotlib.cm as cm
    import joblib

    # add path to slide for which you are building a visualisation here, to get slide dimensions
    slide_path = Path(img_input)
    wsi = OpenSlideWSIReader(img_input)
    meta = wsi.info
    save_path = "./_temp_overlay"  # temp path for the memmap, will be removed at end

    canvas_shape = (meta.slide_dimensions[1], meta.slide_dimensions[0])

    out_ch = 3
    cum_canvas = np.lib.format.open_memmap(  # open memmap to build the visualisation in
        save_path,
        mode="w+",
        shape=tuple(canvas_shape) + (out_ch,),
        dtype=np.uint8,
    )
    cum_canvas[:] = 255

    json_path = Path(json_path)  # path to whereever the segmentations are stored

    if json_path.suffix == ".geojson":
        with open(json_path, "rb") as gf:
            data = json.load(gf)
        hover_format = False
    else:
        data = joblib.load(json_path)
        hover_format = True

    contours = []
    c_map = cm.get_cmap("coolwarm")
    # im_rgb=(c_map(im)*255).astype(np.uint8)

    if hover_format:
        for id in data.keys():
            contours.append(data[id]["contour"])
            # vals.append(data[id][score_key])  #whatever score you are interested in visualising should go in vals list
    else:
        for feat in range(len(data["features"])):
            try:
                if data["features"][feat]["nucleusGeometry"]["type"] == "MultiPolygon":
                    continue
            except:
                print("not a cell")
            contours.append(
                (np.array(data["features"][feat]["geometry"]["coordinates"][0]))[
                    :, [0, 1]
                ].astype(np.int32)
            )

    # vals=np.array(vals)
    if score_path != None:
        score = np.load(score_path)
    else:
        score = np.ones((len(contours),))

    cs = (c_map(score) * 255).astype(np.uint8)  # map scores to colour using cmap

    for i, cnt in enumerate(contours):
        # fill in each contour according to its colour
        cv2.drawContours(
            cum_canvas, [cnt], 0, (int(cs[i, 0]), int(cs[i, 1]), int(cs[i, 2])), -1
        )
        cv2.drawContours(cum_canvas, [cnt], 0, (0, 0, 0), 1)

    vi = numpy2vips(cum_canvas)
    vi.tiffsave(
        str(output_path), tile=False, compression="deflate", bigtiff=False, pyramid=True
    )  # write memmap as a tiff
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(save_path)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output file/directory to save the output. "
    "default=./overlay.tiff"
)
@click.option(
    "--json-path",
    help="Path to file/directory containing objects from which to build overlay",
)
@click.option(
    "--score-path",
    help="Path to file/directory containing scores to colour overlay",
    default=None,
)
def overlay_from_json(img_input, output_path, json_path, score_path):
    """build overlay from a slide, json file containing contour info, and file containing scores
    inputs must either all be paths to a file, or all be paths to directories, in which case
    will build an overlay for each slide in img_input directory. For this usage, all files should
    be named the same as the slide it belongs too (eg slide1.svs, slide1.geojson, slide1.npy etc)

    objetcs stored as geojson, or in a hovernet-style .dat file are accepted at the moment.
    scores should be given as an array stored ina  .npy file (or can be ignored, in which case an
    uncoloured overlay will be created)

    example:
    tiatoolbox overlay-from-json --img-input /path/to/slide.svs --json-path /path/to/objects.geojson
     --score-path /path/to/scores.npy --output-path /path/to/overlay.tiff

    """
    from tiatoolbox.utils.misc import grab_files_from_dir

    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, "*.ndpi, *.svs, *.mrxs, *.jp2", " ", "overlays"
    )

    if len(files_all) == 1:
        process_one_slide(img_input, output_path, json_path, score_path)
    else:
        print("process directory")
        json_paths = grab_files_from_dir(json_path, ("*.dat", "*.geojson"))
        if score_path == None:
            score_paths = [None] * len(files_all)
        else:
            score_paths = grab_files_from_dir(score_path, ("*.npy", "*.npz"))
        for curr_file, curr_json, curr_score in zip(files_all, json_paths, score_paths):
            print(f"processing: {curr_file}")
            process_one_slide(
                curr_file,
                output_path.joinpath(curr_file.stem + "_overlay.tiff"),
                curr_json,
                curr_score,
            )
