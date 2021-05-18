# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module enables patch-level prediction."""

import copy
import time
import tqdm
import numpy as np
import os
import pathlib
import warnings
import collections
import logging

from tiatoolbox import rcParam
from tiatoolbox.utils.misc import imwrite, save_json, imread
from tiatoolbox.utils.transforms import imresize
from tiatoolbox.wsicore.wsireader import get_wsireader, VirtualWSIReader, WSIMeta

import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data

####
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import cv2
import math
import random
import colorsys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def colorize(ch, vmin, vmax):
    """Will clamp value value outside the provided range to vmax and vmin."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


####
def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


####
def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.
    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


####
def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.
    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} , 
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay

####
class SerializeReader(torch_data.Dataset):
    """
    `mp_shared_space` must be from torch.multiprocessing, for example

    mp_manager = torch_mp.Manager()
    mp_shared_space = mp_manager.Namespace()
    mp_shared_space.image = torch.from_numpy(image)
    """
    def __init__(
            self,
            wsi_path_list,
            mp_shared_space,
            preproc=None,
            mode='wsi',
            scale_to_lv0=1.0,
            resolution=None,
            units=None):
        super().__init__()
        self.mode = mode
        self.mp_shared_space = mp_shared_space
        self.preproc = preproc
        self.wsi_path_list = wsi_path_list
        self.wsi_idx = None  # to be received externally via thread communication
        self.scale_to_lv0 = scale_to_lv0  # ! need to expose or sync read this
        self.resolution = resolution
        self.units = units
        return

    def _get_reader(self, wsi_path):
        wsi_path = pathlib.Path(wsi_path)
        if self.mode == "wsi":
            reader = get_wsireader(wsi_path)
        else:
            # overwriting for later read
            # units = 'mpp'
            # resolution = 1.0
            # units = "baseline"
            # resolution = 1.0
            img = imread(wsi_path)
            metadata = WSIMeta(
                # Assign blind default value as it has no impact later
                # but is required for sync mask read
                mpp=np.array([1.0, 1.0]),
                objective_power=10,
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
            # any value for mpp is fine, but the read
            # resolution for mask later must match
            # ? alignement, XY or YX ? Default to XY
            # ? to match OpenSlide for now
            reader = VirtualWSIReader(
                img,
                metadata,
            )
        return reader

    def __len__(self):
        return len(self.mp_shared_space.patch_info_list)

    def __getitem__(self, idx):
        # ! no need to lock as we dont modify source value in shared space
        if self.wsi_idx != self.mp_shared_space.wsi_idx:
            # TODO: enforcing strict typing
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.scale_to_lv0 = self.mp_shared_space.scale_to_lv0.item()
            self.reader = self._get_reader(self.wsi_path_list[self.wsi_idx])

        # this is in YX and at requested resolution not lv0
        patch_info = self.mp_shared_space.patch_info_list[idx]
        # convert from requested to lv0 coordinate and in XY
        tl, br = patch_info[0]  # retrieve input placement, [1] is output
        bound = np.array(tl.tolist()[::-1] + br.tolist()[::-1])
        bound = (bound * self.scale_to_lv0).astype(np.int32)
        patch_data = self.reader.read_bounds(
                        bound,
                        resolution=self.resolution,
                        units=self.units)
        # ! due to internal scaling, there will be rounding error and wont match
        # ! the requested size at requested read resolution
        # ! hence must apply rescale again
        output_size = (br - tl).numpy()[::-1]
        patch_data = imresize(img=patch_data, output_size=output_size)
        if self.preproc is not None:
            patch_data = patch_data.copy()
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info


def pbar_creator(x, y, pos=0, leave=False):
    return tqdm.tqdm(
        desc=x, leave=leave, total=y, ncols=80, ascii=True, position=pos
    )


# ! this is more elaborate compared to version in PatchExtractor
# ! TODO: merge and may deprecate the other version ?
def _get_patch_info(img_shape, input_size, output_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape
    """
    def flat_mesh_grid_coord(y, x):
        y, x = np.meshgrid(y, x)
        return np.stack([y.flatten(), x.flatten()], axis=-1)

    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    step = output_size[0]
    start = in_out_diff[0] // 2 - step
    end = last_output_coord[0] + step
    output_tl_y_list = np.arange(
        start, end, step, dtype=np.int32
    )
    step = output_size[1]
    start = in_out_diff[1] // 2 - step
    end = last_output_coord[1] + step
    output_tl_x_list = np.arange(
        start, end, step, dtype=np.int32,
    )
    output_tl = flat_mesh_grid_coord(output_tl_y_list, output_tl_x_list)
    output_br = output_tl + output_size

    input_tl = output_tl - in_out_diff // 2
    input_br = input_tl + input_size
    # exclude any patch where the input exceed the range of image,
    # can comment this out if do padding in reading
    # sel = np.any(input_br > img_shape, axis=-1)
    sel = np.zeros(input_br.shape[0], dtype=np.bool)

    info_list = np.stack(
        [
            np.stack([ input_tl[~sel],  input_br[~sel]], axis=1),  # noqa: E201
            np.stack([output_tl[~sel], output_br[~sel]], axis=1),
        ], axis=1)
    return info_list


def _get_tile_info(img_shape, input_size, output_size, margin_size, unit_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape
    """
    # ! ouput tile size must be multiple of unit
    assert np.sum(output_size % unit_size) == 0
    assert np.sum((margin_size*2) % unit_size) == 0

    def flat_mesh_grid_coord(y, x):
        y, x = np.meshgrid(y, x)
        return np.stack([y.flatten(), x.flatten()], axis=-1)

    in_out_diff = input_size - output_size
    nr_step = np.ceil((img_shape - in_out_diff) / output_size)
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size

    assert np.sum(output_size % unit_size) == 0
    nr_unit_step = np.floor((img_shape - in_out_diff) / unit_size)
    last_unit_output_coord = (in_out_diff // 2) + (nr_unit_step) * unit_size

    # generating subpatches index from orginal
    def get_top_left_1d(axis):
        step = output_size[axis]
        start = in_out_diff[axis] // 2 - unit_size[axis]
        end = last_output_coord[axis] + unit_size[axis]
        o_tl_list = np.arange(
                        start, end, step, dtype=np.int32
                    )
        o_br_list = o_tl_list + output_size[axis]
        o_br_list[-1] = last_unit_output_coord[axis] + step
        # in default behavior, last pos >= last multiple of unit
        # hence may cause duplication, do a check and remove if necessary
        if o_br_list[-1] == o_br_list[-2]:
            o_br_list = o_br_list[:-1]
            o_tl_list = o_tl_list[:-1]
        return o_tl_list, o_br_list

    output_tl_y_list, output_br_y_list = get_top_left_1d(axis=0)
    output_tl_x_list, output_br_x_list = get_top_left_1d(axis=1)

    output_tl = flat_mesh_grid_coord(output_tl_y_list, output_tl_x_list)
    output_br = flat_mesh_grid_coord(output_br_y_list, output_br_x_list)

    def get_info_stack(output_tl, output_br):
        input_tl = output_tl - (in_out_diff // 2)
        input_br = output_br + (in_out_diff // 2)

        info_list = np.stack(
            [
                np.stack([ input_tl,  input_br], axis=1),  # noqa: E201
                np.stack([output_tl, output_br], axis=1),
            ], axis=1)
        return info_list

    # * Full Tile Grid
    info_list = get_info_stack(output_tl, output_br).astype(np.int64)

    # flag surrounding ambiguous (left margin, right margin)
    # |----|------------|----|
    # |\\\\\\\\\\\\\\\\\\\\\\|  
    # |\\\\              \\\\|
    # |\\\\              \\\\|
    # |\\\\              \\\\|
    # |\\\\\\\\\\\\\\\\\\\\\\|  
    # |----|------------|----|
    removal_flag = np.full((info_list.shape[0], 4,), 1)  # left, right, top, bot
    # exclude those contain left most boundary
    removal_flag[(info_list[:,1,0,1] == np.min(output_tl[:,1])),0] = 0
    # exclude those contain right most boundary
    removal_flag[(info_list[:,1,1,1] == np.max(output_br[:,1])),1] = 0
    # exclude those contain top most boundary
    removal_flag[(info_list[:,1,0,0] == np.min(output_tl[:,0])),2] = 0
    # exclude those contain bot most boundary
    removal_flag[(info_list[:,1,1,0] == np.max(output_br[:,0])),3] = 0
    mode_list = np.full(info_list.shape[0], 0)
    all_info = [[info_list, removal_flag, mode_list]]

    br_most = np.max(output_br, axis=0)
    tl_most = np.min(output_tl, axis=0)
    # * Tile Boundary Redo with Margin
    # get the fix grid tile info
    y_fix_output_tl = output_tl - np.array([margin_size[0], 0])[None,:]
    y_fix_output_br = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    y_fix_output_br = y_fix_output_br + np.array([margin_size[0], 0])[None,:]
    # bound reassignment
    # ? do we need to do bound check for tl ?
    # ? (extreme case of 1 tile of size < margin size ?)
    y_fix_output_br[y_fix_output_br[:,0] > br_most[0], 0] = br_most[0]  
    y_fix_output_br[y_fix_output_br[:,1] > br_most[1], 1] = br_most[1]  
    # sel position not on the image boundary
    sel = (output_tl[:,0] == np.min(output_tl[:,0]))
    y_info_list = get_info_stack(
                    y_fix_output_tl[~sel],
                    y_fix_output_br[~sel]).astype(np.int64)

    # flag horizontal ambiguous region for y (left margin, right margin)
    # |----|------------|----|
    # |\\\\|            |\\\\|
    # |----|------------|----|
    # <----> ambiguous (margin size)
    removal_flag = np.zeros((y_info_list.shape[0], 4,))  # left, right, top, bot
    removal_flag[:,[0,1]] = 1
    # exclude the left most boundary
    removal_flag[(y_info_list[:,1,0,1] == np.min(output_tl[:,1])),0] = 0
    # exclude the right most boundary   
    removal_flag[(y_info_list[:,1,1,1] == np.max(output_br[:,1])),1] = 0
    mode_list = np.full(y_info_list.shape[0], 1)
    all_info.append([y_info_list, removal_flag, mode_list])

    x_fix_output_br = output_br + np.array([0, margin_size[1]])[None,:]
    x_fix_output_tl = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    x_fix_output_tl = x_fix_output_tl - np.array([0, margin_size[1]])[None,:]
    # bound reassignment
    x_fix_output_br[x_fix_output_br[:,0] > br_most[0], 0] = br_most[0]
    x_fix_output_br[x_fix_output_br[:,1] > br_most[1], 1] = br_most[1]
    # sel position not on the image boundary
    sel = (output_br[:,1] == np.max(output_br[:,1]))
    x_info_list = get_info_stack(
                    x_fix_output_tl[~sel],
                    x_fix_output_br[~sel]).astype(np.int64)
    # flag vertical ambiguous region for x (top margin, bottom margin)
    # |----| ^
    # |\\\\| | ambiguous
    # |----| V
    # |    |
    # |    |
    # |----|
    # |\\\\|
    # |----|
    removal_flag = np.zeros((x_info_list.shape[0], 4,))  # left, right, top, bot
    removal_flag[:,[2,3]] = 1
    # exclude the left most boundary
    removal_flag[(x_info_list[:,1,0,0] == np.min(output_tl[:,0])),2] = 0
    # exclude the right most boundary
    removal_flag[(x_info_list[:,1,1,0] == np.max(output_br[:,0])),3] = 0
    mode_list = np.full(x_info_list.shape[0], 2)
    all_info.append([x_info_list, removal_flag, mode_list])

    sel = np.any(output_br == br_most, axis=-1)
    xsect = output_br[~sel]
    xsect_tl = xsect - margin_size * 2
    xsect_br = xsect + margin_size * 2
    # do the bound check to ensure range stay within
    xsect_br[xsect_br[:,0] > br_most[0], 0] = br_most[0]  
    xsect_br[xsect_br[:,1] > br_most[1], 1] = br_most[1]  
    xsect_tl[xsect_tl[:,0] < tl_most[0], 0] = tl_most[0]  
    xsect_tl[xsect_tl[:,1] < tl_most[1], 1] = tl_most[1]  
    xsect_info_list = get_info_stack(xsect_tl, xsect_br).astype(np.int64)
    mode_list = np.full(xsect_info_list.shape[0], 3)
    removal_flag = np.full((xsect_info_list.shape[0], 4,), 0)  # left, right, top, bot
    all_info.append([xsect_info_list, removal_flag, mode_list])

    return all_info


def _get_valid_patch_idx(wsi_proc_shape, wsi_mask, patch_info_list):
    """Select valid patches from the list of input patch information.

    Args:
        patch_info_list: patch input coordinate information
        has_output_info: whether output information is given
    
    """
    def check_valid(info, wsi_mask):
        output_bbox = np.rint(info[1]).astype(np.int64)
        # contain negatice indexing and out of bound index
        # just treat them as within read only
        mask_shape = wsi_mask.shape[:2]
        tl, br = output_bbox
        tl[tl < 0] = 0
        br[br > mask_shape] = br[br > mask_shape]
        output_roi = wsi_mask[
            tl[0] : br[0],
            tl[1] : br[1],
        ]
        return (np.sum(output_roi) > 0)

    down_sample_ratio = wsi_mask.shape[0] / wsi_proc_shape[0]
    valid_indices = [check_valid(info * down_sample_ratio, wsi_mask)
                     for info in patch_info_list]
    # somehow multiproc is slower than single thread
    valid_indices = np.array(valid_indices)
    return valid_indices


def _get_valid_patch_idx_in_tile(tile_info, patch_info_list):
    # checking basing on the output alignment
    tile_tl, tile_br = tile_info[1]
    patch_tl_list = patch_info_list[:,1,0] 
    patch_br_list = patch_info_list[:,1,1]
    sel =  (patch_tl_list[:,0] >= tile_tl[0]) & (patch_tl_list[:,1] >= tile_tl[1])
    sel &= (patch_br_list[:,0] <= tile_br[0]) & (patch_br_list[:,1] <= tile_br[1])
    return sel


def _get_io_info(
    wsi_path,
    mask_path,
    patch_input_shape,
    patch_output_shape,
    tile_shape,
    ambiguous_size,
    resolution=None,
    units=None):

    def get_wsi_proc_shape(resolution=resolution, units=units):
        lv0_shape = wsi_reader.info.slide_dimensions
        lv0_bound = (0, 0) + lv0_shape
        _, _, read_shape, _ = wsi_reader.find_read_bounds_params(
            bounds=lv0_bound,
            resolution=resolution,
            units=units,
        )
        # ! output is in 
        print(lv0_shape, read_shape)
        return np.array(read_shape)

    wsi_reader = get_wsireader(wsi_path)
    wsi_proc_shape = get_wsi_proc_shape() # in XY
    scale_to_lv0 = np.array(wsi_reader.info.slide_dimensions) / wsi_proc_shape
    scale_to_lv0 = scale_to_lv0[0]
    wsi_proc_shape = wsi_proc_shape[::-1]  # in YX

    if mask_path is not None and os.path.exists(mask_path):
        wsi_mask = imread(mask_path)
        wsi_mask = cv2.cvtColor(wsi_mask, cv2.COLOR_RGB2GRAY)
    else:  # process everything if no mask is provided
        # ! HACK: inconsistent behavior between VirtualReader and WSI
        # ! how to reliably retrieve the shape of downscale version ?
        thumb_shape = get_wsi_proc_shape(1.0, units='baseline')
        wsi_mask = np.ones(thumb_shape[::-1], dtype=np.uint8)
    wsi_mask[wsi_mask > 0] = 1

    # * retrieve patch and tile placement
    patch_info_list = _get_patch_info(
        wsi_proc_shape, patch_input_shape, patch_output_shape,
    )
    # print(patch_input_shape)
    # print(patch_info_list)
    # exit()
    patch_diff_shape = patch_input_shape - patch_output_shape
    # derive tile output placement as consecutive tiling with step size of 0
    # and tile output will have shape being of multiple of patch_output_shape
    # (round down)
    tile_output_shape = np.floor(tile_shape / patch_output_shape) * patch_output_shape
    tile_input_shape = tile_output_shape + patch_diff_shape

    # [full_grid, vert/horiz, xsect]
    all_tile_info = _get_tile_info(
        wsi_proc_shape, tile_input_shape, tile_output_shape, 
        ambiguous_size, patch_output_shape
    )

    #
    sel_index = _get_valid_patch_idx(wsi_proc_shape, wsi_mask, patch_info_list)
    patch_info_list = patch_info_list[sel_index]    
    return wsi_mask, wsi_proc_shape, all_tile_info, patch_info_list, scale_to_lv0


####
# ! seem to be 1 pix off at cross section or sthg
def get_inst_in_margin(arr, margin_size, tile_pp_info):
    """
    include the margin line itself
    """
    assert margin_size > 0
    tile_pp_info = np.array(tile_pp_info)

    inst_in_margin = []
    # extract those lie within margin region
    if tile_pp_info[0] == 1: # left edge
        inst_in_margin.append(arr[:,:(margin_size+1)])
    if tile_pp_info[1] == 1: # right edge
        inst_in_margin.append(arr[:,-(margin_size+1):])
    if tile_pp_info[2] == 1: # top edge
        inst_in_margin.append(arr[:(margin_size+1),:])
    if tile_pp_info[3] == 1: # bottom edge
        inst_in_margin.append(arr[-(margin_size+1):,:])
    inst_in_margin = [v.flatten() for v in inst_in_margin]
    if len(inst_in_margin) > 0:
        inst_in_margin = np.concatenate(inst_in_margin, axis=0)
        inst_in_margin = np.unique(inst_in_margin)
    else:
        inst_in_margin = np.array([]) # empty array
    return inst_in_margin

####
def get_inst_on_margin(arr, margin_size, tile_pp_info):
    """
    """
    assert margin_size > 0
    # extract those lie on the margin line
    tile_pp_info = np.array(tile_pp_info)

    def line_intersection(line1, line2):
        ydiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        xdiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0: 
            return False # not intersect

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        # ! positive region only (due to indexing line)
        return int(abs(y)), int(abs(x))

    last_h, last_w = arr.shape[0]-1, arr.shape[1]-1
    line_list = [
        [[0, 0]     , [last_h, 0]     ], # left line
        [[0, last_w], [last_h, last_w]], # right line
        [[0, 0]     , [0, last_w]     ], # top line
        [[last_h, 0], [last_h, last_w]], # bottom line
    ]

    if tile_pp_info[0] == 1: 
        line_list[0] = [[0     , margin_size], 
                       [last_h, margin_size]] 
    if tile_pp_info[1] == 1: 
        line_list[1] = [[0     , last_w-margin_size], 
                        [last_h, last_w-margin_size]]
    if tile_pp_info[2] == 1:
        line_list[2] = [[margin_size, 0], 
                        [margin_size, last_w]] 
    if tile_pp_info[3] == 1:
        line_list[3] = [[last_h-margin_size, 0], 
                        [last_h-margin_size, last_w]] 
    # x1 x2
    # x3 x4
    # all pts need to be valid idx !
    pts_list = [
        line_intersection(line_list[2], line_list[0]), # x1 
        line_intersection(line_list[2], line_list[1]), # x2
        line_intersection(line_list[3], line_list[0]), # x3
        line_intersection(line_list[3], line_list[1]), # x4
    ]

    pt_index = lambda p1, p2: arr[p1[0]:p2[0]+1,
                                  p1[1]:p2[1]+1]                        
    line_pix_list = []
    if tile_pp_info[0] == 1:
        line_pix_list.append(pt_index(pts_list[0], pts_list[2])),
    if tile_pp_info[1] == 1:
        line_pix_list.append(pt_index(pts_list[1], pts_list[3])),
    if tile_pp_info[2] == 1:
        line_pix_list.append(pt_index(pts_list[0], pts_list[1])),
    if tile_pp_info[3] == 1:
        line_pix_list.append(pt_index(pts_list[2], pts_list[3])),

    inst_on_margin = [v.flatten() for v in line_pix_list]

    if len(inst_on_margin) > 0:
        inst_on_margin = np.concatenate(inst_on_margin, axis=0)
        inst_on_margin = np.unique(inst_on_margin)
    else:
        inst_on_margin = np.array([]) # empty array

    return inst_on_margin
####
def get_inst_on_edge(arr, tile_pp_info):
    inst_on_edge = []
    if tile_pp_info[0] == 1:
        inst_on_edge.append(arr[:,0])
    if tile_pp_info[1] == 1:
        inst_on_edge.append(arr[:,-1])
    if tile_pp_info[2] == 1:
        inst_on_edge.append(arr[0,:])
    if tile_pp_info[3] == 1:
        inst_on_edge.append(arr[-1,:])

    inst_on_edge = [v.flatten() for v in inst_on_edge]

    if len(inst_on_edge) > 0:
        inst_on_edge = np.concatenate(inst_on_edge, axis=0)
        inst_on_edge = np.unique(inst_on_edge)
    else:
        inst_on_edge = np.array([]) # empty array
    return inst_on_edge
####

def _postproc_tile(tile_io_info, tile_pp_info, tile_mode,
                margin_size, patch_info_list, postproc_func, 
                wsi_proc_shape,
                prev_wsi_inst_dict=None):
    # output pos of the tile within the source wsi
    tile_input_tl, tile_output_br = tile_io_info[0]
    tile_output_tl, tile_output_br = tile_io_info[1] # Y, X

    # ! shape may be uneven hence just detach all into a big list
    patch_pos_list  = [] 
    patch_feat_list = []
    split_inst = lambda x : np.split(x, x.shape[0], axis=0)
    for batch_pos, batch_feat in patch_info_list:
        patch_pos_list.extend(split_inst(batch_pos))
        patch_feat_list.extend(split_inst(batch_feat))

    # * assemble patch to tile
    nr_ch = patch_feat_list[-1].shape[-1]
    tile_shape = (tile_output_br - tile_output_tl).tolist()
    pred_map = np.zeros(tile_shape + [nr_ch], dtype=np.float32)
    for idx in range(len(patch_pos_list)):
        # zero idx to remove singleton, squeeze may kill h/w/c
        patch_pos = patch_pos_list[idx][0].copy()

        # # ! assume patch pos alrd aligned to be within wsi input system
        patch_pos = patch_pos - tile_output_tl # shift from wsi to tile output system

        pos_tl, pos_br = patch_pos[1] # retrieve ouput placement
        pred_map[
            pos_tl[0] : pos_br[0],
            pos_tl[1] : pos_br[1]
        ] = patch_feat_list[idx][0]
    del patch_pos_list, patch_feat_list

    # recalibrate the tile actual shape incase they crossing the boundary
    crop_tl = np.zeros_like(tile_output_tl)
    crop_tl[tile_output_tl < 0] = np.abs(tile_output_tl[tile_output_tl < 0])
    sel = tile_output_br < wsi_proc_shape
    crop_br= tile_output_br.copy()
    crop_br[sel] = wsi_proc_shape[sel]
    crop_br = crop_br - tile_output_tl # shift back to tile coordinate
    # now crop
    pred_map = pred_map[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1]]
    tile_output_tl[tile_output_tl < 0] = 0 # fixing the position

    # * retrieve actual output
    pred_inst, inst_info_dict = postproc_func(pred_map)
    del pred_map

    # * perform removal for ambiguous region

    # Consider each symbol as 1 pixel    
    # This is margin inner area //// 
    # ----------------------------- ^  ^
    # |///////////////////////////| |  | margin area 
    # |///////////////////////////| |  | (yes including the inner and outer edge)
    # |///|-------------------|///| |  V
    # |///|   ^margin_size    |///| |
    # |///|                   |///| |
    # |///| <-- margin line   |///| | Image area
    # |///|      |            |///| |
    # |///|      v            |///| |
    # |///|-------------------|///| |
    # |///////////////////////////| |
    # |///////////////////////////| |
    # ----------------------------| V

    # ! extreme slow down may happen because the aggregated results are huge
    def draw_prev_pred_inst():
        tile_output = tile_io_info[1]
        tile_canvas = np.zeros(tile_output[1] - tile_output[0], dtype=np.int32)
        if len(prev_wsi_inst_dict) == 0: return tile_canvas

        wsi_inst_uid_list = np.array(list(prev_wsi_inst_dict.keys()))
        wsi_inst_com_list = np.array([v['centroid'] for v in prev_wsi_inst_dict.values()])
        wsi_inst_com_list = wsi_inst_com_list[:,::-1] # XY to YX       
        sel =  (wsi_inst_com_list[:,0] > tile_output[0,0])
        sel &= (wsi_inst_com_list[:,0] < tile_output[1,0])
        sel &= (wsi_inst_com_list[:,1] > tile_output[0,1])
        sel &= (wsi_inst_com_list[:,1] < tile_output[1,1])
        sel_idx = np.nonzero(sel.flatten())[0]

        for inst_idx in sel_idx:
            inst_uid = wsi_inst_uid_list[inst_idx]
            # shift from wsi system to tile output system
            inst_info = prev_wsi_inst_dict[inst_uid]
            inst_cnt = np.array(inst_info['contour'])
            inst_cnt = inst_cnt - tile_output[0,[1,0]][None]
            tile_canvas = cv2.drawContours(tile_canvas, [inst_cnt], -1, int(inst_uid), -1)
        return tile_canvas

    output_remove_inst_set = None
    # tile_mode = -1 # ! no fix, debud mode
    if tile_mode == 0: 
        # for `full grid tile`
        # -- extend from the boundary by the margin size, remove 
        #    nuclei lie within the margin area but exclude those
        #    lie on the margin line
        # also contain those lying on the edges
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, tile_pp_info)
        # those lying on the margin line, check 2pix toward margin area for sanity
        inst_on_margin1 = get_inst_on_margin(pred_inst, margin_size-1, tile_pp_info) 
        inst_on_margin2 = get_inst_on_margin(pred_inst, margin_size  , tile_pp_info) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        inst_within_margin = np.setdiff1d(inst_in_margin, inst_on_margin, assume_unique=True)        
        remove_inst_set = inst_within_margin.tolist()
    elif tile_mode == 1 or tile_mode == 2:
        # for `horizontal/vertical strip tiles` for fixing artifacts
        # -- extend from the marked edges (top/bot or left/right) by the margin size, 
        #    remove all nuclei lie within the margin area (including on the margin line)
        # -- nuclei on all edges are removed (as these are alrd within `full grid tile`)
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, tile_pp_info) # also contain those lying on the edges
        if np.sum(tile_pp_info) == 1:
            holder_flag = tile_pp_info.copy()
            if tile_mode == 1: 
                holder_flag[[2, 3]] = 1
            else: 
                holder_flag[[0, 1]] = 1
        else:
            holder_flag = [1, 1, 1, 1]
        inst_on_edge = get_inst_on_edge(pred_inst, holder_flag)
        remove_inst_set = np.union1d(inst_in_margin, inst_on_edge)   
        remove_inst_set = remove_inst_set.tolist()
    elif tile_mode == 3:
        # inst within the tile after excluding margin area out
        # only for a tile at cross-section, which is designed such that 
        # their shape >= 3* margin size     

        # ! for removal of current pred, just like tile_mode = 0 but with all side   
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, [1, 1, 1, 1])
        # those lying on the margin line, check 2pix toward margin area for sanity
        inst_on_margin1 = get_inst_on_margin(pred_inst, margin_size-1, [1, 1, 1, 1]) 
        inst_on_margin2 = get_inst_on_margin(pred_inst, margin_size  , [1, 1, 1, 1]) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        inst_within_margin = np.setdiff1d(inst_in_margin, inst_on_margin, assume_unique=True)        
        remove_inst_set = inst_within_margin.tolist()
        # ! but we also need to remove prev inst exist in the global space of entire wsi
        # ! on the margin
        prev_pred_inst = draw_prev_pred_inst()
        inst_on_margin1 = get_inst_on_margin(prev_pred_inst, margin_size-1, [1, 1, 1, 1]) 
        inst_on_margin2 = get_inst_on_margin(prev_pred_inst, margin_size  , [1, 1, 1, 1]) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        output_remove_inst_set = inst_on_margin.tolist()    
    else:
        remove_inst_set = []

    remove_inst_set = set(remove_inst_set)
    # * move pos back to wsi position
    renew_id = 0
    new_inst_info_dict = {}
    for k, v in inst_info_dict.items():
        if k not in remove_inst_set:            
            v['bbox'] += tile_output_tl[::-1]
            v['centroid'] += tile_output_tl[::-1]
            v['contour'] += tile_output_tl[::-1]
            new_inst_info_dict[renew_id] = v
            renew_id += 1

    return new_inst_info_dict, output_remove_inst_set


class Predictor:
    """Pixel-wise segmentation predictor."""

    def __init__(
        self,
        batch_size=8,
        num_loader_worker=0,
        num_postproc_worker=0,
        model=None,
        pretrained_model=None,
        pretrained_weight=None,
        verbose=True,
    ):
        """Initialise the Patch Predictor."""
        super().__init__()

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        if model is not None:
            self._model = copy.deepcopy(model)
        else:
            raise NotImplementedError
            # self.model, self.patch_size, self.objective_value = get_pretrained_model(
            #     pretrained_model, pretrained_weight
            # )

        # to keepsake in case of being nested in DataParallel
        self.model = None # this is for runtime

        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.num_postproc_worker = num_postproc_worker
        self.verbose = verbose

    def _run_once(
            self,
            loader,
            patch_info_list,
            tile_io_info_list,
            tile_pp_info_list,
            tile_mode_list,
            wsi_proc_shape,
            prev_wsi_inst_dict=None):

        offset_id = 0 # offset id to increment overall saving dict
        wsi_inst_info = {} 
        if prev_wsi_inst_dict is not None:
            wsi_inst_info = copy.deepcopy(prev_wsi_inst_dict)
            if len(wsi_inst_info) > 0:
                offset_id = max(wsi_inst_info.keys()) + 1

        def postproc_callback(new_inst_dict, remove_uid_list, wsi_inst_info, offset_id):
            # * aggregate
            # ! the return id should be contiguous to maximuize 
            # ! counting range in int32
            inst_wsi_id = offset_id # barrier in case no output in tile!
            for inst_id, inst_info in new_inst_dict.items():
                inst_wsi_id = offset_id + inst_id + 1
                wsi_inst_info[inst_wsi_id] = inst_info
            offset_id = inst_wsi_id + 1

            if remove_uid_list is not None:
                for inst_uid in remove_uid_list:
                    if inst_uid in wsi_inst_info:
                        # faster than pop
                        del wsi_inst_info[inst_uid]
            return wsi_inst_info, offset_id

        forward_info_list = collections.deque()
        nr_tile = tile_io_info_list.shape[0]
        for tile_idx in range(nr_tile):
            tile_info = tile_io_info_list[tile_idx]
            sel_index = _get_valid_patch_idx_in_tile(tile_info, patch_info_list)
            patch_in_tile_info_list = patch_info_list[sel_index]
            forward_info_list.append([tile_info, patch_in_tile_info_list])

        # TODO: migrate this out to global persistent worker
        proc_pool = None
        if self.num_postproc_worker > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        pbar_t = pbar_creator('Fwrd-Tile', nr_tile, pos=1)
        pbar_p = pbar_creator('Post-Proc', nr_tile, pos=2)

        all_time = 0
        future_list = []
        for tile_idx, tile_info in enumerate(forward_info_list):

            tile_info, patch_info_list = tile_info

            start = time.perf_counter()
            # ! VERY MESSY AND HARD TO MANAGE
            self._mp_shared_space.wsi_idx = torch.Tensor([self.wsi_idx]).share_memory_()
            self._mp_shared_space.patch_info_list = torch.from_numpy(patch_info_list).share_memory_()
            self._mp_shared_space.scale_to_lv0 = torch.Tensor([self.scale_to_lv0]).share_memory_()
            end = time.perf_counter()
            all_time += (end - start)


            pbar_b = pbar_creator('Fwrd-Btch', patch_info_list.shape[0], pos=0)
            cum_output = []
            for batch_idx, batch_data in enumerate(loader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self._model.infer_batch(
                                self.model, sample_data_list, 'cuda'
                            )
                sample_info_list = sample_info_list.numpy()
                cum_output.append([sample_info_list, sample_output_list])
                pbar_b.update()
            pbar_b.close()
            pbar_t.update()

            tile_io_info = tile_io_info_list[tile_idx]
            tile_pp_info = tile_pp_info_list[tile_idx]
            tile_mode = tile_mode_list[tile_idx]
            args = [tile_io_info, tile_pp_info, tile_mode, 
                    self.ambiguous_size[0], cum_output, 
                    self._model.postproc_func,
                    wsi_proc_shape,
                    prev_wsi_inst_dict]

            # launch separate thread to deal with postproc
            if proc_pool is not None:
                future = proc_pool.submit(_postproc_tile, *args)
                future_list.append(future)
            else:
                new_inst_dict, remove_uid_list = _postproc_tile(*args)
                wsi_inst_info, offset_id = postproc_callback(
                                    new_inst_dict, 
                                    remove_uid_list, 
                                    wsi_inst_info,
                                    offset_id)
                pbar_b.update()
        pbar_t.close()

        while len(future_list) > 0:
            if not future_list[0].done(): 
                future_list.rotate()
                continue
            proc_future = future_list.popleft()
            if proc_future.exception() is not None:
                print(proc_future.exception())

            # * aggregate
            # ! the return id should be contiguous to maximize 
            # ! counting range in int32
            new_inst_dict, remove_uid_list = proc_future.result()
            wsi_inst_info, offset_id = postproc_callback(
                                new_inst_dict, 
                                remove_uid_list, 
                                wsi_inst_info,
                                offset_id)            
            pbar_b.update()
        if proc_pool is not None:
            proc_pool.shutdown()
        pbar_b.close()
        return wsi_inst_info
                
    def _predict_one_wsi(self, wsi_path, mask_path, loader):
        # ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # * Async Inference
        # * launch a seperate process to do forward and store the result in a queue.
        # * main thread will poll for forward result and launch separate process for
        # * doing the postproc for every newly predicted tiles
        #
        #         / forward \ (loop)
        # main ------------- main----------------main
        #                       \ postproc (loop)/
        # ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(mask_path)
        (wsi_mask, wsi_proc_shape, all_tile_info, 
            patch_info_list, scale_to_lv0) = _get_io_info(
                                                wsi_path, 
                                                mask_path,
                                                self.patch_input_shape,
                                                self.patch_output_shape,
                                                self.tile_shape,
                                                self.ambiguous_size,
                                                self.resolution,
                                                self.units)
        self.scale_to_lv0 = scale_to_lv0
        print(scale_to_lv0)

        ## ** process full grid and vert/horiz fixing at the same time
        start = time.perf_counter()
        # info_list = list(zip(*all_tile_info[:1])) # full grid only
        info_list = list(zip(*all_tile_info[:3])) # full grid, horiz, vert, only
        tile_io_info_list = np.concatenate(info_list[0], axis=0).astype(np.int32)
        tile_pp_info_list = np.concatenate(info_list[1], axis=0)
        tile_mode_list    = np.concatenate(info_list[2], axis=0)
        sel_index = _get_valid_patch_idx(wsi_proc_shape, wsi_mask, tile_io_info_list)
        tile_io_info_list = tile_io_info_list[sel_index]
        tile_pp_info_list = tile_pp_info_list[sel_index]
        tile_mode_list    = tile_mode_list[sel_index]

        wsi_inst_info = self._run_once(
                            loader,
                            patch_info_list,
                            tile_io_info_list, 
                            tile_pp_info_list, 
                            tile_mode_list,
                            wsi_proc_shape)
        end = time.perf_counter()
        logging.info("Proc Grid: {0}".format(end - start))
        
        # reader = get_wsireader(wsi_path)
        # thumb = reader.slide_thumbnail(resolution=self.resolution, units='mpp')
        # overlay = visualize_instances_dict(thumb, wsi_inst_info)
        # imwrite('dump.png', overlay)
        # exit()

        ## ** re-infer and redo postproc for xsect alone
        start = time.perf_counter()
        tile_io_info_list = all_tile_info[-1][0]
        tile_pp_info_list = all_tile_info[-1][1]
        tile_mode_list    = all_tile_info[-1][2]
        sel_index = _get_valid_patch_idx(wsi_proc_shape, wsi_mask, tile_io_info_list)
        tile_io_info_list = tile_io_info_list[sel_index]
        tile_pp_info_list = tile_pp_info_list[sel_index]
        tile_mode_list    = tile_mode_list[sel_index]
        
        wsi_inst_info = self._run_once(
                            loader,
                            patch_info_list,
                            tile_io_info_list, 
                            tile_pp_info_list, 
                            tile_mode_list,
                            wsi_proc_shape,
                            wsi_inst_info)        
        end = time.perf_counter()
        logging.info("Proc XSect: {0}".format(end - start))
        return wsi_inst_info

    def predict(
        self,
        img_list,
        mask_list=None,
        mode="patch",
        return_probabilities=False,
        return_labels=False,
        on_gpu=True,
        stride_shape=None,  # at requested read resolution, not wrt to lv0
        resolution=0.25,
        units="mpp",
        save_dir=None,
    ):
        """Make a prediction for a list of input data."""

        self.model = torch.nn.DataParallel(self._model)
        self.model = self.model.to('cuda')

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        mp_mutext = mp_manager.Lock()
        self._mp_shared_space = mp_shared_space

        # ! by default, torch will split idxing across worker
        # ! hence, for each batch, they coming entirely from different worker id
        ds = SerializeReader(img_list, mp_shared_space, 
                resolution=resolution, units=units, mode=mode)
        loader = torch_data.DataLoader(ds,
                                batch_size=8,
                                drop_last=False,
                                num_workers=0,
                                persistent_workers=False,
                                # num_workers=2,
                                # persistent_workers=True,
                            )
        self._loader = loader

        # ! TODO: refactor this into a state holder or sthg
        for wsi_idx in range(len(img_list)):
            self.wsi_idx = wsi_idx
            self.patch_input_shape  = np.array([256, 256])
            self.patch_output_shape = np.array([164, 164])
            self.tile_shape = 1000
            self.ambiguous_size = np.array([328, 328])
            # self.resolution = 0.25
            # self.units = 'mpp'
            self.resolution = resolution
            self.units = units
            output = self._predict_one_wsi(img_list[wsi_idx], mask_list[wsi_idx], loader)
            return output
        return output

