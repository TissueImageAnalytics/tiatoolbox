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


import copy
import logging
import os
import pathlib
import warnings
from typing import Callable, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm

from tiatoolbox.models.segmentation.abc import IOStateSegmentor
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, get_wsireader


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

        mp_shared_space: must be from torch.multiprocessing, for example

        >>> mp_manager = torch_mp.Manager()
        >>> mp_shared_space = mp_manager.Namespace()

        iostate: object which contains I/O placement for patches.

        wsi_path_list: List of paths pointing to a WSI or tiles.

        preproc: pre-processing function to be applied on a patch.

        mode: either `wsi` or `tile` to indicate which form the input in
            `wsi_path_list` is.
    """

    def __init__(
        self,
        iostate: IOStateSegmentor = None,
        wsi_path_list: List[Union[str, pathlib.Path]] = None,
        mp_shared_space=None,  # context variable, how to type hint this?
        preproc: Callable[[np.ndarray], np.ndarray] = None,
        mode="wsi",
    ):
        super().__init__()
        self.mode = mode
        self.preproc = preproc
        self.iostate = copy.deepcopy(iostate)
        if mode == "tile":
            warnings.warn(
                (
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"`. Resolutios will be converted '
                    "to baseline value."
                )
            )
            # migrate to IOState?
            self.iostate.convert_to_baseline()

        self.mp_shared_space = mp_shared_space
        self.wsi_path_list = wsi_path_list
        self.wsi_idx = None  # to be received externally via thread communication
        self.reader = None

    def _get_reader(self, img_path):
        """Get approriate reader for input path."""
        img_path = pathlib.Path(img_path)
        if self.mode == "wsi":
            reader = get_wsireader(img_path)
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

    @staticmethod
    def collate_fn(batch):
        """
        Proto to handle reading exception
        """
        batch = [v for v in batch if v is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, idx):
        # ! no need to lock as we dont modify source value in shared space
        if self.wsi_idx != self.mp_shared_space.wsi_idx:
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.reader = self._get_reader(self.wsi_path_list[self.wsi_idx])

        # this is in XY and at requested resolution not baseline
        bound = self.mp_shared_space.patch_input_list[idx]
        bound = bound.numpy()  # expected to be torch.Tensor

        # be the same as bounds br-tl, unless bounds are of float
        for resolution in self.iostate.input_resolutions:
            # ! conversion for other resolution !
            patch_data = self.reader.read_bounds(
                bound.astype(np.int32),
                coord_space="resolution",
                pad_constant_values=0,  # expose this ?
                **resolution,
            )

        if self.preproc is not None:
            patch_data = patch_data.copy()
            patch_data = self.preproc(patch_data)

        bound = self.mp_shared_space.patch_output_list[idx]
        return patch_data, bound


class SemanticSegmentor:
    """Pixel-wise segmentation predictor."""

    def __init__(
        self,
        batch_size=8,
        num_loader_worker=0,
        num_postproc_worker=0,  # has not effect
        model=None,
        pretrained_model=None,
        pretrained_weight=None,
        verbose=True,
        auto_generate_mask=False,
        dataset_class: Callable = SerializeWSIReader,
    ):
        """Initialise the Semantic Segmentor.

        Note, if model is supplied in the arguments, it will override the backbone.

        Args:
            model (nn.Module): Use externally defined PyTorch model for prediction with.
                weights already loaded. Default is `None`. If provided,
                `pretrained_model` argument is ignored.

            pretrained_model (str): Name of the existing models support by tiatoolbox
                for processing the data. Refer to
                `tiatoolbox.models.classification.get_pretrained_model` for details.

                By default, the corresponding pretrained weights will also be
                downloaded. However, you can override with your own set of weights
                via the `pretrained_weight` argument. Argument is case insensitive.

            pretrained_weight (str): Path to the weight of the corresponding
                `pretrained_model`.

            batch_size (int) : Number of images fed into the model each time.

            num_loader_worker (int) : Number of workers to load the data.
                Take note that they will also perform preprocessing.

            verbose (bool): Whether to output logging information.

            dataset_class (obj): Dataset class to be used instead of default.

            auto_generate_mask(bool): To automatically generate tile/WSI tissue mask
                if is not provided.
        """
        super().__init__()

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        # TODO: add pretrained model

        # for runtime, such as after wrapping with nn.DataParallel
        self._model = None
        self._on_gpu = None
        self._mp_shared_space = None
        self.img_list = None
        self.mask_list = None

        self.dataset_class: SerializeWSIReader = dataset_class
        self.model = model  # original copy
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.num_postproc_worker = num_postproc_worker
        self.verbose = verbose
        self.auto_generate_mask = auto_generate_mask

    # TODO: refactor this, duplicated functionalities wrt the patchpredictor
    @staticmethod
    def get_reader(img_path: str, mask_path: str, mode: str, auto_get_mask: bool):
        """Get reader for mask and source image."""
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
            mask_reader.info = reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            mask_reader.info = reader.info
        return reader, mask_reader

    def _predict_one_wsi(
        self, wsi_idx: int, iostate: IOStateSegmentor, loader, save_path: str, mode: str
    ):
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int): index of the tile/wsi to be processed within `self`.
            iostate (IOStateSegmentor): object which defines I/O placement during
                inference and when assembling back to full tile/wsi.
            loader (torch.Dataloader): loader object which return batch of data
                to be input to model.
            save_path (str): location to save output prediction as well as possible
                intermediat results.
            mode (str): `tile` or `wsi` to indicate run mode.
        """
        wsi_path = self.img_list[wsi_idx]
        mask_path = None if self.mask_list is None else self.mask_list[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path, mask_path, mode, self.auto_generate_mask
        )

        resolution = iostate.highest_input_resolution
        if (
            isinstance(wsi_reader, VirtualWSIReader)
            and resolution["units"] != "baseline"
        ):
            raise ValueError("Inference on `tile` only use `units='baseline'` !")
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch and tile placement
        # this is in XY
        (patch_input_list, patch_output_list) = PatchExtractor.get_coordinates(
            image_shape=wsi_proc_shape,
            patch_input_shape=iostate.patch_input_shape,
            patch_output_shape=iostate.patch_output_shape,
            stride_shape=iostate.stride_shape,
        )
        if mask_reader is not None:
            sel = PatchExtractor.filter_coordinates(
                mask_reader, patch_output_list, **resolution
            )
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
        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(loader)),
            ncols=80,
            ascii=True,
            position=0,
        )

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
                self._model,
                sample_data_list,
                self._on_gpu,
            )
            # repackage so that its a N list, each contains
            # L x etc. output
            sample_output_list = [
                np.split(v, batch_size, axis=0) for v in sample_output_list
            ]
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
        # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of output patch
        # this can exceed the image bound at the requested resolution
        # remove singleton due to split.
        location_list = np.array([v[0] for v in location_list])
        for idx, output_resolution in enumerate(iostate.output_resolutions):
            # assume resolution idx to be in the same order as L
            merged_resolution = resolution
            merged_location_list = location_list
            # ! location is wrt highest resolution, hence still need conversion
            if iostate.save_resolution is not None:
                merged_resolution = iostate.save_resolution
                output_shape = wsi_reader.slide_dimensions(**output_resolution)
                merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                fx = merged_shape[0] / output_shape[0]
                merged_location_list = np.ceil(location_list * fx).astype(np.int64)
            merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
            # 0 idx is to remove singleton wihout removing other axes singleton
            to_merge_prediction_list = [v[idx][0] for v in prediction_list]
            sub_save_path = f"{save_path}.raw.{idx}.npy"
            self.merge_prediction(
                merged_shape[::-1],  # XY to YX
                to_merge_prediction_list,
                merged_location_list,
                save_path=sub_save_path,
                free_prediction=True,
            )

    @staticmethod
    def merge_prediction(
        canvas_shape: Union[Tuple[int], List[int], np.ndarray],
        prediction_list: List[np.ndarray],
        location_list: Union[List, np.ndarray],
        save_path: Union[str, pathlib.Path] = None,
        free_prediction: bool = True,
    ):
        """Merge patch-level predictions to form a 2-dimensional prediction map.

        Args:
            canvas_shape (:class:`numpy.ndarray`): HW of the supposed assembled image.

            prediction_list (list): List of nd.array, each item is a prediction of
                a patch, assuming to be of shape HWC.

            location_list (list): List of nd.array, each item is the location of
                the patch at the same index within `prediction_list` in the to be
                assembled canvas.

            save_path (str): Location to save the assembled image.

            free_prediction (bool): If this is `True`, `prediction_list` will
                be modified in place and each patch will be replace with `None`
                once processed. This is to save memory when assembling.

        """
        out_ch = prediction_list[0].shape[-1]
        cum_canvas = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            shape=tuple(canvas_shape) + (out_ch,),
            dtype=np.float32,
        )

        # for pixel occurence counting
        cnt_canvas = np.zeros(canvas_shape, dtype=np.float32)

        patch_info_list = list(zip(location_list, prediction_list))
        for patch_idx, patch_info in enumerate(patch_info_list):
            # position is assumed to be in XY coordinate
            (bound_in_wsi, prediction) = patch_info
            # convert to XY to YX, and in tl, br
            tl_in_wsi = np.array(bound_in_wsi[:2][::-1])
            br_in_wsi = np.array(bound_in_wsi[2:][::-1])
            old_tl_in_wsi = tl_in_wsi.copy()

            # need to do conversion
            patch_shape_in_wsi = tuple(br_in_wsi - tl_in_wsi)
            prediction = cv2.resize(prediction, patch_shape_in_wsi[::-1])

            # insert singleton to align the shape when merging (which is HWC)
            if len(prediction.shape) == 2:
                prediction = prediction[..., None]

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

            #  internal error, switch to raise ?
            # if not (np.all(br_in_patch >= 0)
            #         and np.all(tl_in_patch >= 0)):
            #     raise RuntimeError(
            #             '[BUG] Locations should not be negative at this stage!')

            # now croping the prediction region
            patch_pred = prediction[
                tl_in_patch[0] : br_in_patch[0], tl_in_patch[1] : br_in_patch[1]
            ]

            patch_count = np.ones(patch_pred.shape[:2])
            cum_canvas[
                tl_in_wsi[0] : br_in_wsi[0], tl_in_wsi[1] : br_in_wsi[1]
            ] += patch_pred
            cnt_canvas[
                tl_in_wsi[0] : br_in_wsi[0], tl_in_wsi[1] : br_in_wsi[1]
            ] += patch_count
            # remove prediction without altering list ordering or length
            if free_prediction:
                patch_info_list[patch_idx] = None
        cum_canvas /= cnt_canvas[..., None] + 1.0e-6
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
        crash_on_exception=False,
    ):
        """Make a prediction for a list of input data.

        Args:
            img_list (list, ndarray): List of inputs to process. When using `patch`
            mode, the input must be either a list of images, a list of image file paths
            or a numpy array of an image list. When using `tile` or `wsi` mode, the
            input must be a list of file paths.

            mask_list (list): List of masks. Only utilised when processing image tiles
            and whole-slide images. Patches are only processed if they are witin a
            masked area. If not provided, then a tissue mask will be automatically
            generated for whole-slide images or the entire image is processed for
            image tiles.

            label_list: List of labels. If using `tile` or `wsi` mode, then only a
            single label per image tile or whole-slide image is supported.
            mode (str): Type of input to process. Choose from either `patch`, `tile` or
                `wsi`.

            return_probabilities (bool): Whether to return per-class probabilities.

            return_labels (bool): Whether to return the labels with the predictions.

            on_gpu (bool): whether to run model on the GPU.

            patch_input_shape (tuple): Size of patches input to the model. The value
                are at requested read resolution and must be positive.

            patch_output_shape (tuple): Size of patches output by the model. The value
                are at requested read resolution and must be positive.

            stride_shape (tuple): Stride using during tile and WSI processing. The value
                are at requested read resolution and must be positive.
                If not provided, `stride_shape=patch_input_shape`.

            resolution (float): Resolution used for reading the image.

            units (str): Units of resolution used for reading the image. Choose from
                either `level`, `power` or `mpp`.

            merge_predictions (bool): Whether to merge the predictions to form
            a 2-dimensional map. This is only applicable for `mode='wsi'` or
            `mode='tile'`.

            save_dir (str): Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output` where the
                running script is invoked.

        Returns:
            output (ndarray, dict): Model predictions of the input dataset.
                If multiple image tiles or whole-slide images are provided as input,
                then results are saved to `save_dir` and a dictionary indicating save
                location for each input is return.

                The dict has following format:
                - img_path: path of the input image.
                    - raw: path to save location for raw prediction.
                    - merged: path to .npy contain merged predictions.


        """
        if mode not in ["wsi", "tile"]:
            raise ValueError(f"{mode} is not a valid mode. Use either `tile` or `wsi`.")
        if save_dir is None:
            warnings.warn(
                " ".join(
                    [
                        "Segmentor will only output to directory.",
                        "All subsequent output will be saved to current runtime",
                        "location under folder 'output'. Overwriting may happen!",
                    ]
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
                input_resolutions=[{"resolution": resolution, "units": units}],
                output_resolutions=[{"resolution": resolution, "units": units}],
                patch_input_shape=patch_input_shape,
                patch_output_shape=patch_output_shape,
                stride_shape=stride_shape,
            )

        ds = self.dataset_class(
            iostate=iostate,
            preproc=self.model.preproc,
            wsi_path_list=img_list,
            mp_shared_space=mp_shared_space,
            mode=mode,
        )

        loader = torch_data.DataLoader(
            ds,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_loader_worker,
            persistent_workers=self.num_loader_worker > 0,
        )

        self.img_list = img_list
        self.mask_list = mask_list

        # contain input / ouput prediction mapping
        output_list = []
        # ? what will happen if this crash midway?
        # => may not be able to retrieve the result dict
        for wsi_idx, img_path in enumerate(img_list):
            try:
                wsi_save_path = os.path.join(save_dir, f"{wsi_idx}")
                self._predict_one_wsi(wsi_idx, iostate, loader, wsi_save_path, mode)

                # dont use dict as mapping, because can overwrite, if that is
                # user intention to provide same path twice
                output_list.append([img_path, wsi_save_path])

                # verbose mode, error by passing ?
                logging.info(f"Finish: {wsi_idx}/{len(img_list)}")
                logging.info(f"--Input: {img_path}")
                logging.info(f"--Ouput: {wsi_save_path}")
            # prevent deep source check because this is bypass and
            # delegating error message
            except Exception as err:  # noqa
                if not crash_on_exception:
                    logging.error(err)
                    continue
                raise err

        # memory clean up
        self.img_list = None
        self.mask_list = None
        self._model = None
        self._on_gpu = None
        self._mp_shared_space = None
        return output_list
