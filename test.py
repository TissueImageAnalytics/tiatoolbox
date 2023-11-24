"""Import modules required to run the Jupyter notebook."""
# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import numpy as np

from tiatoolbox.tools import patchextraction

input_img = np.zeros((1000, 1000, 6))

patch_extractor = patchextraction.get_patch_extractor(
    input_img=input_img,  # input image path, numpy array, or WSI object
    # path to list of points (csv, json), numpy list, panda DF
    method_name="slidingwindow",  # also supports "slidingwindow"
    patch_size=(
        32,
        32,
    ),  # size of the patch to extract around the centroids from centroids_list
    resolution=0,
    units="level",
)

i = 1
# show only first 16 patches
num_patches_to_show = 16

for patch in patch_extractor:
    print(i)
    i += 1
