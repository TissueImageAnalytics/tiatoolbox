"""Import modules required to run the Jupyter notebook."""

from __future__ import annotations

import gc

# Clear logger to use tiatoolbox.logger
import logging
import sys

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr

from tiatoolbox import logger
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
from tiatoolbox.utils.misc import download_data, imread

# We need this function to visualize the nuclear predictions
from tiatoolbox.utils.visualization import (
    overlay_prediction_contours,
    overlay_prediction_mask,
)
from tiatoolbox.wsicore.wsireader import WSIReader


multi_segmentor = MultiTaskSegmentor(
    model="hovernet_fast-pannuke",
    num_workers=2,
    batch_size=8,
)

tile_output = multi_segmentor.run(
    ["/media/u1910100/data/slides/wsi1_2k_2k.svs"],
    save_dir="/home/u1910100/Desktop/qupath-zarrv3",
    # TIAToolbox v2.0 and above use patch_mode=False to run models on Tiles and WSIs
    patch_mode=False,
    device='cuda',
    auto_get_mask=True,
    output_type='qupath',
    overwrite=True,
)