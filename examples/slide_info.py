# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# In this example, we will load up a sample whole slide image (WSI), check out some key information, 
# then extract some image patches from it. From this, we will examine `wsireader`, `slide_info` modules of the library.
#
# We start by importing some related libraries. Then, we will download a small WSI which is 
# available from the web. Readers can switch to your WSI of choices by changing 
# `sample_file_name` variable. Readers should also direct `run_dir` location toward 
# your where you currently store the network. In case of downloading from the web, 
# the sample WSI will be saved under the name defined in `sample_file_name` within the 
# `run_dir` folder. Subsequently, we will load up the WSI of name `sample_file_name` 
# from the `run_dir` directory.

# %%
from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox.dataloader import wsireader
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300 # for hires figure in notebook

run_dir = '/home/tialab-dang/local/project/tiatoolbox/'
sample_file_name = 'sample_wsi_small.svs'


# %%
if not os.path.exists('%s/%s' % (run_dir, sample_file_name)):
    r = requests.get(
        "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
    )
    with open('%s/sample_wsi_small.svs' % run_dir, "wb") as f:
        f.write(r.content)

# %% [markdown]
# Now, we will create a `OpenSlideWSIReader` object to load the information from WSI. 
# For this class, the `input_dir` means the path to directory which contains the target, 
# while the `file_name` keyword points to the specific file within the folder. 

# %%
# create a file handler
wsi_reader = wsireader.OpenSlideWSIReader(
                input_dir=run_dir, 
                file_name=sample_file_name)

# %% [markdown]
# Let's first check the basic information about this WSI such as magnification, dimension, and etc.
# %%
wsi_info = wsi_reader.slide_info.as_dict()
# we will print out each info line by line
print(*list(wsi_info.items()), sep='\n')

# %% [markdown]
# Now, we will check the WSI thumbnail, which is loaded at x1.25 magnification.
# %%
wsi_thumb = wsi_reader.slide_thumbnail()
plt.imshow(wsi_thumb)

# %% [markdown]
# Now, we will see how to use the WSI object by a implementing a small task, 
# retrieving a set of location from the WSI thumbnail we got previously 
# then load the patches up for visualization. We prefer the patches containing
# only tissue. Hence, We begin with writing a `simple_get_mask` function to threshold 
# the WSI thumbnail intensity so that the tissue is seperated out from the background. 
# Some minor operations such as dilation are applied subsequently to prettify the output. 
# We often call the area highlight a specific region in the image as mask.

# %%
import cv2
import math
from skimage import morphology

# simple method to extract tissue regions using intensity thresholding and morphological operations
def simple_get_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = morphology.remove_small_objects(mask == 0, min_size=100, connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=100)
    mask = morphology.binary_dilation(mask, morphology.disk(5))
    return mask

wsi_thumb_mask = simple_get_mask(wsi_thumb)
plt.subplot(1,2,1)
plt.imshow(wsi_thumb)
plt.subplot(1,2,2)
plt.imshow(wsi_thumb_mask)

# %% [markdown]
# Now, we write a function to find the pick up some location to extract the patches within the tissue area. The location is defined as its top left location with repspect to the top left corner of the source image and its dimension (heigh and width). 
# For computational reason, the operation is done at the thumbnail level, so we will need to map the patches location and its size back to the targetted magnification we want to extract later.

# %%
from scipy.ndimage.measurements import center_of_mass
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

# get the super-pixels (a.k.a rois) of the tissue area and extract 
# patches centering those super-pixel (a.k.a rois)
lores_mag = 1.25 # the magnification of the thumbnail (lores = low resolution)
hires_mag = 20 # the magnification where the patch would be extracted (hires = high resolution)
hires_patch_size = 128 # expected output patch size at higher resolution
# map the expected patch size at hires to lores
lores_patch_size = int(hires_patch_size / (hires_mag / lores_mag))

# %% [markdown]
# For this example, we would like to create image patches such that 
# they can cover the entire WSI. We will use the super-pixel algorithm 
# SLIC from `scikit` to do this. It will split the tissue region into 
# regions of similar size at low resolution. Then we get the patches 
# which center these region.

# calculate the expected number of patches can be obtain with the given size
# however, note that the actual number of patches may not be the same as expected
nr_expected_rois = math.ceil(np.sum(wsi_thumb_mask) / ((lores_patch_size ** 2)))
wsi_rois_mask = slic(wsi_thumb, 
                    mask=wsi_thumb_mask,
                    n_segments=nr_expected_rois, 
                    compactness=1000, 
                    sigma=1)
print('#Actual Patches / #Expected Patches : %d/%d' % (np.unique(wsi_rois_mask).shape[0], nr_expected_rois))

lores_rois_center = center_of_mass(wsi_rois_mask, 
                labels=wsi_rois_mask, 
                index=np.unique(wsi_rois_mask)[1:])
lores_rois_center = np.array(lores_rois_center) # coordinates is Y, X
lores_rois_center = (lores_rois_center).astype(np.int32)
selected_indices = wsi_thumb_mask[lores_rois_center[:,0],lores_rois_center[:,1]]
lores_rois_center = lores_rois_center[selected_indices]

# show the patches region and their center of masses
plt.imshow(mark_boundaries(wsi_thumb, wsi_rois_mask))
plt.scatter(lores_rois_center[:,1], lores_rois_center[:,0], s=2)

# %% [markdown]
# We then convert the centers of each regions to the top-left position 
# of the patches at high resolution.

# %%
# convert to top left idx at hires_mag level
lores_rois_top_left = (lores_rois_center - (lores_patch_size // 2))
hires_rois_top_left = lores_rois_top_left * (hires_mag / lores_mag)
hires_rois_top_left = hires_rois_top_left.astype(np.int32) 

# %% [markdown]
# Now, we load up some patches for visualization
# %%
nr_viz_patches = 16

# for illustration purpose, only read a small amount
selected_indices = np.random.randint(0, hires_rois_top_left.shape[0], size=(4*nr_viz_patches,))
hires_rois_top_left = hires_rois_top_left[selected_indices] 

patch_list = []
for patch_coord in hires_rois_top_left:
    patch = wsi_reader.read_region(
                        patch_coord[1], patch_coord[0],
                        patch_coord[1]+hires_patch_size, 
                        patch_coord[0]+hires_patch_size,
                        level=0)
    patch_list.append(patch)

# plot the first 16
sub_patches = np.array(patch_list[:16])
sub_patches = np.reshape(sub_patches, (4, 4, hires_patch_size, hires_patch_size, 3))
sub_patches = np.transpose(sub_patches, (0, 2, 1, 3, 4))
sub_patches = np.reshape(sub_patches, (4 * hires_patch_size, 4 * hires_patch_size, 3))
plt.imshow(sub_patches)
plt.axis('off')

# %% [markdown]
# Conversely, if you want to extract the entire wsi (including the 
# background). You can use the built-in `save_tiles` functionality 
# of each `WSIReader` object.
# 
# We start by creating another `WSIReader` object with keyword 
# `tile_objective_value`, `tile_read_size_h`, `tile_read_size_w`, 
# and `output_dir`. They repsectively define the magnification 
# where the patches will be read, their expected height and 
# expected width, and where they will be saved. For terminology, 
# tile means image patches at sizeable dimension. For this, tiles are 
# read at x20, each of size 1000x1000 and will be saved at the `tmp` 
# folder within the `run_dir`.

# %%
# create a file handler
wsi_reader_v2 = wsireader.OpenSlideWSIReader(
                input_dir=run_dir,
                file_name=sample_file_name,
                output_dir=run_dir + '/tmp/',
                tile_objective_value=20,
                tile_read_size_h=1000,
                tile_read_size_w=1000)
wsi_reader_v2.save_tiles()

# %% [markdown]
# Now, we will check the content of the output folder and plot some 
# tiles for visualization. The extracted tiles would be saved under 
# a subfolder having the WSI name as its name, andThe output folder 
# would contain a `.csv` which summarize the extracted tiles.

# %%

import cv2
import pandas as pd

tile_summary = pd.read_csv('%s/tmp/%s/Output.csv' % (run_dir, sample_file_name))
print(tile_summary)

# %% [markdown]
# We will plot `Tile_20_1000_1000.jpg` as sample output.

# %%
sample_tile = cv2.imread('%s/tmp/%s/%s' % (run_dir, sample_file_name, tile_summary.iloc[4]['Tile_Name']))
sample_tile = cv2.cvtColor(sample_tile, cv2.COLOR_BGR2RGB)

plt.imshow(sample_tile)



# %%
