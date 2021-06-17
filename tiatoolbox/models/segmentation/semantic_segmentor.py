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

import collections
import colorsys
import copy
import itertools
import logging
####
import math
import os
import pathlib
import random
import time
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

from tiatoolbox import rcParam
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imread, imwrite, save_json
from tiatoolbox.utils.transforms import imresize
from tiatoolbox.wsicore.wsireader import (VirtualWSIReader, WSIMeta,
                                          get_wsireader)

from .pretrained_info import get_pretrained_model


####
class SerializeWSIReader(torch_data.Dataset):
    """Reading a wsi in parallel mode with persistent workers.

    To speed up the inference process for multiple WSIs. The
    `torch.utils.data.Dataloader` is set to run in persistent mode.
    Normally, it will prevent worker from altering their initial states
    (such as provided input etc.) . To sidestep this, we use a shared parallel
    workspace context manager to send data and signal from the main thread,
    thus allowing each worker to load new wsi as well as corresponding patch
    information.

    Args:

        `mp_shared_space`: must be from torch.multiprocessing, for example

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
            resolution=None,
            units=None):
        super().__init__()
        self.mode = mode
        self.preproc = preproc
        if mode == "tile":
            warnings.warn(
                (
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"` and `resolution=1.0`.'
                )
            )
            units = "baseline"
            resolution = 1.0
        self.resolution = resolution
        self.units = units

        self.mp_shared_space = mp_shared_space
        self.wsi_path_list = wsi_path_list
        self.wsi_idx = None  # to be received externally via thread communication
        return

    def _get_reader(self, img_path):
        img_path = pathlib.Path(img_path)
        if self.mode == "wsi":
            self.reader = get_wsireader(img_path)
        else:
            img = imread(img_path)
            # initialise metadata for VirtualWSIReader.
            # here, we simulate a whole-slide image, but with a single level.
            metadata = WSIMeta(
                mpp=np.array([1.0, 1.0]),
                objective_power=10,
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
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
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.reader = self._get_reader(self.wsi_path_list[self.wsi_idx])

        # this is in YX and at requested resolution not lv0
        tl, br = self.mp_shared_space.patch_input_list[idx]
        bounds = np.array(tl.tolist()[::-1] + br.tolist()[::-1])

        # be the same as bounds br-tl, unless bounds are of float
        patch_data = self.reader.read_bounds(
                        bounds.astype(np.int32),
                        resolution=self.resolution,
                        units=self.units,
                        location_at_requested=True)

        if self.preproc is not None:
            patch_data = patch_data.copy()
            patch_data = self.preproc(patch_data)

        tl, br = self.mp_shared_space.patch_output_list[idx]
        return patch_data, [tl, br]


class Segmentor:
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
            self.model = model
            iostate = None  # retrieve iostate from provided model ?
        else:
            model, iostate = get_pretrained_model(
                pretrained_model, pretrained_weight
            )

        self._iostate = iostate  # for storing original
        self.iostate = None  # for runtime
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.num_postproc_worker = num_postproc_worker
        self.verbose = verbose

    def _predict_one_wsi(self, wsi_path, mask_path, loader):

        wsi_reader = get_wsireader(wsi_path)
        # in XY
        wsi_proc_shape = wsi_reader.slide_dimensions(
                            self.iostate.resolution,
                            self.iostate.units)



        # * retrieve patch and tile placement
        patch_info_list = _get_patch_info(
            wsi_proc_shape,
            self.patch_input_shape,
            self.patch_output_shape,
            self.stride_shape
        )

        self._mp_shared_space.wsi_idx = torch.Tensor([self.wsi_idx]).share_memory_()
        self._mp_shared_space.patch_info_list = torch.from_numpy(patch_info_list).share_memory_()
        self._mp_shared_space.scale_to_lv0 = torch.Tensor([scale_to_lv0]).share_memory_()

        pbar_desc = 'Process Batch: '
        pbar = tqdm.tqdm(desc=pbar_desc, leave=True,
                    total=int(len(self._loader)), 
                    ncols=80, ascii=True, position=0)

        cum_output = []
        for batch_idx, batch_data in enumerate(self._loader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = self._model.infer_batch(
                            self.model, sample_data_list, 
                            self.on_gpu,
                        )
            curr_batch_size = sample_output_list.shape[0]
            sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0) 
            sample_output_list = list(zip(sample_info_list, sample_output_list))
            cum_output.extend(sample_output_list)
            pbar.update()
        pbar.close()

        nr_output_ch = cum_output[0][1].shape[-1]
        scale_output = 1.0 # ! TODO: protocol for this hard coded
        wsi_mask_shape = (wsi_proc_shape * scale_output).astype(np.int32)
        cum_canvas = np.zeros(list(wsi_mask_shape) + [nr_output_ch,], dtype=np.float32)
        counter_canvas = np.zeros(wsi_mask_shape, dtype=np.float32)
        for patch_idx, (patch_info, patch_pred) in enumerate(cum_output):
            patch_output_info = patch_info[1].numpy().copy()
            patch_output_info = (patch_output_info * scale_output).astype(np.int32)
            tl_in_wsi, br_in_wsi = patch_output_info

            # recalibrate the position in case of patch passing the edge
            crop_tl = np.zeros_like(tl_in_wsi)
            crop_tl[tl_in_wsi < 0] = np.abs(tl_in_wsi[tl_in_wsi < 0])

            sel = br_in_wsi > wsi_mask_shape
            crop_br = br_in_wsi.copy()
            crop_br[sel] = wsi_mask_shape[sel]
            crop_br = crop_br - tl_in_wsi  # shift back to tile coordinate
            # now crop
            patch_pred = patch_pred[0, crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1]]

            if tl_in_wsi[0] < 0: tl_in_wsi[0] = 0
            if tl_in_wsi[1] < 0: tl_in_wsi[1] = 0
            if tl_in_wsi[0] >= wsi_mask_shape[0]: tl_in_wsi[0] = wsi_mask_shape[0]
            if tl_in_wsi[1] >= wsi_mask_shape[1]: tl_in_wsi[1] = wsi_mask_shape[1]
            if br_in_wsi[0] >= wsi_mask_shape[0]: br_in_wsi[0] = wsi_mask_shape[0]
            if br_in_wsi[1] >= wsi_mask_shape[1]: br_in_wsi[1] = wsi_mask_shape[1]

            patch_in_wsi_shape = br_in_wsi - tl_in_wsi
            if np.any(patch_in_wsi_shape == 0): continue

            patch_count = np.ones(patch_pred.shape[:2]) 
            cum_canvas[tl_in_wsi[0] : br_in_wsi[0],
                       tl_in_wsi[1] : br_in_wsi[1]] += patch_pred
            counter_canvas[tl_in_wsi[0] : br_in_wsi[0],
                           tl_in_wsi[1] : br_in_wsi[1]] += patch_count
            cum_output[patch_idx] = None # delete out
        cum_canvas = cum_canvas / (counter_canvas[...,None] + 1.0e-6)
        return cum_canvas

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

        self.on_gpu = on_gpu
        if on_gpu:
            self.model = torch.nn.DataParallel(self._model)
            self.model = self.model.to('cuda')
        else:
            self.model = copy.deepcopy(self._model)
            self.model = self.model.to('cpu')

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        # ! by default, torch will split idxing across worker
        # ! hence, for each batch, they coming entirely from different worker id
        ds = _SerializeReader(img_list, mp_shared_space, 
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
            self.patch_input_shape  = np.array([1024, 1024])
            self.patch_output_shape = np.array([512 ,  512])
            self.stride_shape = np.array([256 , 256])
            self.tile_shape = 3000
            # self.resolution = 0.25
            # self.units = 'mpp'
            self.resolution = resolution
            self.units = units
            mask_path = mask_list[wsi_idx] if mask_list is not None else None
            output = self._predict_one_wsi(img_list[wsi_idx], mask_path, loader)
            break
        
        # ! TODO: retrieval mode
        from skimage import morphology
        wsi_mask = output[...,1]
        wsi_mask = np.array(wsi_mask > 0.8).astype(np.uint8)
        wsi_mask = morphology.remove_small_holes(wsi_mask, area_threshold=256*256)
        wsi_mask = morphology.remove_small_objects(wsi_mask, min_size=32*32, connectivity=2)

        return [wsi_mask]
