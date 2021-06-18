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

"""This module enables semantic segmentation."""

import collections
import colorsys
import copy
import itertools
import logging

import math
import os
import pathlib
from typing import Callable, Optional, Tuple, Union, List
import warnings

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm

from tiatoolbox.models.segmentation.abc import IOStateSegmentor
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsireader import (VirtualWSIReader, WSIMeta,
                                          get_wsireader)


class SerializeWSIReader(torch_data.Dataset):
    """Reading a wsi in parallel mode with persistent workers.

    To speed up the inference process for multiple WSIs. The
    `torch.utils.data.Dataloader` is set to run in persistent mode.
    Normally, this will prevent worker from altering their initial states
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
            iostate : IOStateSegmentor,
            wsi_path_list : List[Union[str, pathlib.Path]],
            mp_shared_space,
            preproc : Callable[[np.ndarray], np.ndarray] = None,
            mode='wsi',):
        super().__init__()
        self.mode = mode
        self.preproc = preproc
        self.iostate = copy.deepcopy(iostate)
        if mode == "tile":
            warnings.warn(
                (
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"`. Resolutios will be converted '
                    'to baseline value.'
                )
            )
            # migrate to IOState?
            self.iostate.convert_to_baseline()

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
        return len(self.mp_shared_space.patch_input_list)

    def __getitem__(self, idx):
        # ! no need to lock as we dont modify source value in shared space
        if self.wsi_idx != self.mp_shared_space.wsi_idx:
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.reader = self._get_reader(self.wsi_path_list[self.wsi_idx])

        # this is in XY and at requested resolution not baseline
        tl, br = self.mp_shared_space.patch_input_list[idx]
        bounds = np.array(tl.tolist() + br.tolist())

        # be the same as bounds br-tl, unless bounds are of float
        for resolution in self.iostate.input_resolutions:
            # ! conversion for other resolution !
            patch_data = self.reader.read_bounds(
                            bounds.astype(np.int32),
                            location_at_requested=True,
                            **resolution)

        if self.preproc is not None:
            patch_data = patch_data.copy()
            patch_data = self.preproc(patch_data)

        tl, br = self.mp_shared_space.patch_output_list[idx]
        return patch_data, torch.stack([tl, br])


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

        # TODO: add pretrained model

        # for runtime, such as after wrapping with nn.DataParallel
        self._model = None
        self._on_gpu = None
        self._mp_shared_space = None

        self.model = model  # original copy
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.num_postproc_worker = num_postproc_worker
        self.verbose = verbose

    # TODO: refactor this, duplicated functionalities wrt the patchpredictor
    @staticmethod
    def get_reader(img_path, mask_path, mode, auto_get_mask):
        img_path = pathlib.Path(img_path)
        reader = get_wsireader(img_path)

        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.attach_to_reader(reader.info)
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            # ? will this mess up ?
            mask_reader.attach_to_reader(reader.info)
        return reader, mask_reader

    def _predict_one_wsi(
            self,
            wsi_idx : int,
            iostate : IOStateSegmentor,
            loader,
            save_dir : str,
            mode : str):
        """
        """
        wsi_path = self.img_list[wsi_idx]
        base_name = pathlib.Path(wsi_path).stem
        mask_path = None if self.mask_list is None else self.msk_list[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(wsi_path, mask_path, mode, False)

        # take the highest input resolution, may need a find function ?
        resolution = iostate.input_resolutions[0]  # in XY
        if isinstance(wsi_reader, VirtualWSIReader):
            if resolution['units'] != 'baseline':
                raise ValueError("Inference on `tile` only use `units='mpp'`")
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch and tile placement
        patch_input_list, patch_output_list = PatchExtractor.get_coordinates(
            image_shape=wsi_proc_shape,
            patch_input_shape=iostate.patch_input_shape,
            patch_output_shape=iostate.patch_output_shape,
            stride_shape=iostate.stride_shape
        )
        if mask_reader is not None:
            sel = PatchExtractor.filter_coordinates(
                    mask_reader, patch_output_list, **resolution)
            patch_output_list = patch_output_list[sel]
            patch_input_list = patch_input_list[sel]

        # modify the shared space so that we can update worker info without
        # needing to re-create the worker. There should be no race-condition because
        # only the following enumerate loop triggers the parallelism, and this portion
        # is still in sequential execution order
        patch_input_list = torch.from_numpy(patch_input_list).share_memory_()
        patch_output_list = torch.from_numpy(patch_output_list).share_memory_()
        self._mp_shared_space.patch_input_list = patch_input_list
        self._mp_shared_space.patch_output_list = patch_output_list
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

        # ! TODO: need a protocol for pbar, or a decorator to make this less redundant
        pbar_desc = 'Process Batch: '
        pbar = tqdm.tqdm(
                desc=pbar_desc, leave=True,
                total=int(len(loader)),
                ncols=80, ascii=True, position=0)

        # refactor this out to explicit imply need to redefine when
        # changing model IO format ?
        cum_output = []
        for _, batch_data in enumerate(loader):
            sample_data_list, sample_info_list = batch_data
            batch_size = sample_info_list.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)
            sample_output_list = self.model.infer_batch(
                            self._model, sample_data_list,
                            self._on_gpu,
                        )
            # repackage so that its a N list, each contains
            # L x etc. output
            sample_output_list = [
                np.split(v, batch_size, axis=0)
                for v in sample_output_list]
            sample_output_list = list(zip(*sample_output_list))

            # tensor to numpy, costly?
            sample_info_list = sample_info_list.numpy()
            sample_info_list = np.split(sample_info_list, batch_size, axis=0)

            sample_output_list = list(zip(sample_info_list, sample_output_list))
            cum_output.extend(sample_output_list)
            pbar.update()
        pbar.close()

        # assume prediction_list is N, each item has L output element
        location_list, prediction_list = list(zip(*cum_output))
        # Nx2x2 (N x [tl, br]), denotes the location of output patch
        # this can exceed the image bound at the requested resolution
        # remove simpleton due to split
        location_list = [v[0] for v in location_list]
        for idx, output_resolution in enumerate(iostate.output_resolutions):
            # assume resolution idx to be in the same order as L
            merged_resolution = resolution
            merged_location_list = location_list
            # ! location is wrt highest resolution, hence still need conversion
            if iostate.save_resolution is not None:
                merged_resolution = iostate.save_resolution
                output_shape = wsi_reader.slide_dimensions(**output_resolution)
                merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                fx = output_shape[0] / merged_shape[0]
                # fy = output_shape[1] / merged_shape[1]
                merged_location_list = [np.ceil(v * fx).astype(np.int64)
                                        for v in location_list]
            merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
            # 0 idx is to remove singleton wihout removing other axes singleton
            to_merge_prediction_list = [v[idx][0] for v in prediction_list]
            save_path = os.path.join(save_dir, f'{base_name}.raw.{idx}.npy')
            self.merge_prediction(
                        merged_shape,
                        to_merge_prediction_list,
                        merged_location_list,
                        # save_path=save_path,
                        remove_prediction=True)
        return

    @staticmethod
    def merge_prediction(
            # canvas_shape : Union[Tuple[int, int], List[int, int], np.ndarray],
            canvas_shape,
            prediction_list : List[np.ndarray],
            location_list : List[np.ndarray],
            save_path : Union[str, pathlib.Path] = None,
            remove_prediction : bool = True,
            ):
        """"""
        out_ch = prediction_list[0].shape[-1]
        if save_path is not None:
            cum_canvas = np.lib.format.open_memmap(
                save_path,
                mode="w+",
                shape=tuple(canvas_shape) + (out_ch,),
                dtype=np.float32,
            )
        else:
            cum_canvas = np.zeros(tuple(canvas_shape) + (out_ch,), dtype=np.float32)
        # for pixel occurence counting
        cnt_canvas = np.zeros(canvas_shape, dtype=np.float32)

        patch_info_list = list(zip(location_list, prediction_list))
        for patch_idx, patch_info in enumerate(patch_info_list):
            ((tl_in_wsi, br_in_wsi), prediction) = patch_info
            tl_in_wsi = np.array(tl_in_wsi)
            br_in_wsi = np.array(br_in_wsi)
            old_tl_in_wsi = tl_in_wsi.copy()

            sel = tl_in_wsi < 0
            tl_in_wsi[sel] = 0

            if np.any(tl_in_wsi > canvas_shape):
                continue

            sel = br_in_wsi > canvas_shape
            br_in_wsi[sel] = canvas_shape[sel]

            # recalibrate the position in case patch passing the image bound
            br_in_patch = br_in_wsi - old_tl_in_wsi
            patch_actual_shape = br_in_wsi - tl_in_wsi
            tl_in_patch = br_in_patch - patch_actual_shape
            # internal error, switch to raise ?
            assert np.all(br_in_patch >= 0) and np.all(tl_in_patch >= 0)
            # now croping the prediction region
            patch_pred = prediction[tl_in_patch[0]:br_in_patch[0],
                                    tl_in_patch[1]:br_in_patch[1]]

            patch_count = np.ones(patch_pred.shape[:2])
            cum_canvas[tl_in_wsi[0] : br_in_wsi[0],
                       tl_in_wsi[1] : br_in_wsi[1]] += patch_pred
            cnt_canvas[tl_in_wsi[0] : br_in_wsi[0],
                       tl_in_wsi[1] : br_in_wsi[1]] += patch_count
            # remove prediction without altering list ordering or length
            if remove_prediction:
                patch_info_list[patch_idx] = None
        cum_canvas /= (cnt_canvas[..., None] + 1.0e-6)

        dump = cum_canvas[..., 0] / np.max(cum_canvas)
        # dump = cnt_canvas / np.max(cnt_canvas)
        dump = (dump * 255).astype(np.uint8)
        imwrite('dump.png', dump)
        exit()
        return cum_canvas

    def predict(
        self,
        img_list,
        mask_list=None,
        mode="tile",
        on_gpu=True,
        iostate=None,
        patch_input_shape=None,
        patch_output_shape=None,
        stride_shape=None,  # at requested read resolution, not wrt to lv0
        resolution=0.25,
        units="mpp",
        save_dir=None,
    ):
        """Make a prediction for a list of input data."""
        if mode not in ["wsi", "tile"]:
            raise ValueError(
                f"{mode} is not a valid mode. Use either `tile` or `wsi`."
            )
        if save_dir is None:
            warnings.warn(
                ' '.join(
                    "Segmentor will only output to directory.",
                    "All subsequent output will be saved to current runtime",
                    "location under folder 'output'. Overwriting may happen!",
                )
            )
            save_dir = os.path.join(os.getcwd(), "output")

        save_dir = os.path.abspath(save_dir)
        save_dir = pathlib.Path(save_dir)
        if not save_dir.is_dir():
            os.makedirs(save_dir)
        else:
            raise ValueError(f"`save_dir` already exists! {save_dir}")

        # use external for testing
        self._on_gpu = on_gpu
        self._model = misc.model_to(on_gpu, self.model)

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        if patch_output_shape is None:
            patch_output_shape = patch_input_shape
        if stride_shape is None:
            stride_shape = patch_output_shape

        if iostate is None:
            iostate = IOStateSegmentor(
                input_resolutions=[{'resolution': resolution, 'units': units}],
                output_resolutions=[{'resolution': resolution, 'units': units}],
                patch_input_shape=patch_input_shape,
                patch_output_shape=patch_output_shape,
                stride_shape=stride_shape
            )

        ds = SerializeWSIReader(
                iostate,
                img_list,
                mp_shared_space,
                mode=mode)

        loader = torch_data.DataLoader(
                            ds,
                            batch_size=8,
                            drop_last=False,
                            num_workers=0,
                            persistent_workers=False,
                            # num_workers=2,
                            # persistent_workers=True,
                        )

        self.img_list = img_list
        self.mask_list = mask_list
        for wsi_idx, _ in enumerate(img_list):
            self._predict_one_wsi(
                    wsi_idx, iostate, loader, save_dir, mode)

        # memory clean up
        self.img_list = None
        self.msk_list = None
        self._model = None
        self._on_gpu = None
        self._mp_shared_space = None
        return
