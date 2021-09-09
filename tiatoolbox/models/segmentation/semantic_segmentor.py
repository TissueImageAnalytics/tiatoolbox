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


import shutil
import joblib
import copy
import logging
import os
import pathlib
import warnings
from typing import Callable, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm

from tiatoolbox.models.abc import IOConfigABC
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, get_wsireader


class IOConfigSegmentor(IOConfigABC):
    """Define a class to hold IO information for patch predictor."""

    # We predefine to follow enforcement, actual initialization in init
    patch_size = None
    input_resolutions = None
    output_resolutions = None

    def __init__(
        self,
        input_resolutions: List[dict],
        output_resolutions: List[dict],
        patch_input_shape: Union[List[int], np.ndarray],
        patch_output_shape: Union[List[int], np.ndarray],
        save_resolution: dict = None,
        **kwargs,
    ):
        """Define IO placement for patch input and output.

        Args:
            input_resolutions: resolution of each input head of model
                inference, must be in the same order as target model.forward().
            output_resolutions: resolution of each output head from model
                inference, must be in the same order as target model.infer_batch().
            save_resolution: resolution to save all output.

        Examples:

            >>> # Defining io for a network having 1 input and 1 output at the
            >>> # same resolution
            >>> ioconfig = IOConfigSegmentor(
            ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
            ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
            ...     patch_input_shape=[2048, 2048],
            ...     patch_output_shape=[1024, 1024],
            ...     stride_shape=[512, 512],
            ... )

        Examples:

            >>> # Defining io for a network having 3 input and 2 output at the
            >>> # at the same resolution, the output is then merged at another
            >>> # different resolution.
            >>> ioconfig = IOConfigSegmentor(
            ...     input_resolutions=[
            ...         {"units": "mpp", "resolution": 0.25},
            ...         {"units": "mpp", "resolution": 0.50},
            ...         {"units": "mpp", "resolution": 0.75},
            ...     ],
            ...     output_resolutions=[
            ...         {"units": "mpp", "resolution": 0.25},
            ...         {"units": "mpp", "resolution": 0.50},
            ...     ],
            ...     patch_input_shape=[2048, 2048],
            ...     patch_output_shape=[1024, 1024],
            ...     stride_shape=[512, 512],
            ...     save_resolution={"units": "mpp", "resolution": 4.0},
            ... )

        """
        self.patch_input_shape = patch_input_shape
        self.patch_output_shape = patch_output_shape
        self.stride_shape = None
        self.input_resolutions = input_resolutions
        self.output_resolutions = output_resolutions

        self.resolution_unit = input_resolutions[0]["units"]
        self.save_resolution = save_resolution

        for variable, value in kwargs.items():
            self.__setattr__(variable, value)

        self._validate()

        if self.resolution_unit == "mpp":
            self.highest_input_resolution = min(
                self.input_resolutions, key=lambda x: x["resolution"]
            )
        else:
            self.highest_input_resolution = max(
                self.input_resolutions, key=lambda x: x["resolution"]
            )

    def _validate(self):
        """Validate the data format."""
        resolutions = self.input_resolutions + self.output_resolutions
        units = [v["units"] for v in resolutions]
        units = np.unique(units)
        if len(units) != 1 or units[0] not in [
            "power",
            "baseline",
            "mpp",
        ]:
            raise ValueError("Invalid resolution units.")

    @staticmethod
    def scale_to_highest(resolutions: List[dict], unit: str):
        """Get scaling factor from input resolutions.

        This will convert resolutions to scaling factor with repsect to
        highest resolutions found in the input list of resolutions.

        Args:
            resolutions (list): a list of resolutions where each defined
                as `{'resolution': value, 'unit': value}`
            unit (string): unit that the the resolutions are at.

        Return:
            (np.ndarray): an 1D array of scaling factor having the same
                length as `resolutions`

        """
        old_val = [v["resolution"] for v in resolutions]
        if unit == "baseline":
            new_val = old_val
        elif unit == "mpp":
            new_val = np.min(old_val) / np.array(old_val)
        else:
            # when being power
            new_val = np.array(old_val) / np.max(old_val)
        return new_val

    def to_baseline(self):
        """Convert IO to baseline form.

        This will return a new IO holder where resolutions have been converted
        to baseline form with highest possible resolution found in both input
        and output as reference.

        """
        _self = copy.deepcopy(self)
        resolutions = _self.input_resolutions + _self.output_resolutions
        scale_factors = _self.scale_to_highest(resolutions, _self.resolution_unit)
        _self.input_resolutions = [
            {"units": "baseline", "resolution": v}
            for v in scale_factors[: len(_self.input_resolutions)]
        ]
        _self.output_resolutions = [
            {"units": "baseline", "resolution": v}
            for v in scale_factors[len(_self.input_resolutions) :]
        ]
        return _self


class WSIStreamDataset(torch_data.Dataset):
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
        ioconfig: object which contains I/O placement for patches.
        wsi_paths: List of paths pointing to a WSI or tiles.
        preproc: pre-processing function to be applied on a patch.
        mode: either `wsi` or `tile` to indicate which form the input in
            `wsi_paths` is.

    Examples:

        >>> ioconfig = IOConfigSegmentor(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=[2048, 2048],
        ...     patch_output_shape=[1024, 1024],
        ...     stride_shape=[512, 512],
        ... )
        >>> mp_manager = torch_mp.Manager()
        >>> mp_shared_space = mp_manager.Namespace()
        >>> mp_shared_space.signal = 1  # adding variable to the shared space
        >>> wsi_paths = ['A.svs', 'B.svs']
        >>> ds = WSIStreamDataset(ioconfig, wsi_paths, mp_shared_space)

    """

    def __init__(
        self,
        ioconfig: IOConfigSegmentor,
        wsi_paths: List[Union[str, pathlib.Path]],
        mp_shared_space,  # context variable, how to type hint this?
        preproc: Callable[[np.ndarray], np.ndarray] = None,
        mode="wsi",
    ):
        super().__init__()
        self.mode = mode
        self.preproc = preproc
        self.ioconfig = copy.deepcopy(ioconfig)

        if mode == "tile":
            warnings.warn(
                " ".join(
                    [
                        "WSIPatchDataset only reads image tile at",
                        '`units="baseline"`. Resolutions will be converted',
                        "to baseline value.",
                    ]
                )
            )
            self.ioconfig = self.ioconfig.to_baseline()

        self.mp_shared_space = mp_shared_space
        self.wsi_paths = wsi_paths
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
        return len(self.mp_shared_space.patch_inputs)

    @staticmethod
    def collate_fn(batch):
        """Prototype to handle reading exception.

        This will exclude any sample with `None` out from the
        batch. As such, wrapping `__getitem__` with try-catch
        and return `None` upon exceptions will prevent crashing the
        entire programs. But as side effect, batch may not have size
        as defined.

        """
        batch = [v for v in batch if v is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, idx: int):
        # ! no need to lock as we dont modify source value in shared space
        if self.wsi_idx != self.mp_shared_space.wsi_idx:
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.reader = self._get_reader(self.wsi_paths[self.wsi_idx])

        # this is in XY and at requested resolution not baseline
        bound = self.mp_shared_space.patch_inputs[idx]
        bound = bound.numpy()  # expected to be torch.Tensor

        # be the same as bounds br-tl, unless bounds are of float
        patch_data_ = []
        scale_factors = self.ioconfig.scale_to_highest(
            self.ioconfig.input_resolutions, self.ioconfig.resolution_unit
        )
        for idy, resolution in enumerate(self.ioconfig.input_resolutions):
            resolution_bound = np.round(bound * scale_factors[idy])
            patch_data = self.reader.read_bounds(
                resolution_bound.astype(np.int32),
                coord_space="resolution",
                pad_constant_values=0,  # expose this ?
                **resolution,
            )

            if self.preproc is not None:
                patch_data = patch_data.copy()
                patch_data = self.preproc(patch_data)
            patch_data_.append(patch_data)
        if len(patch_data_) == 1:
            patch_data_ = patch_data_[0]

        bound = self.mp_shared_space.patch_outputs[idx]
        return patch_data_, bound


class SemanticSegmentor:
    """Pixel-wise segmentation predictor."""

    def __init__(
        self,
        batch_size: int = 8,
        num_loader_worker: int = 0,
        num_postproc_worker: int = 0,
        model: torch.nn.Module = None,
        pretrained_model: str = None,
        pretrained_weight: str = None,
        verbose: bool = True,
        auto_generate_mask: bool = False,
        dataset_class: Callable = WSIStreamDataset,
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
            num_postproc_worker (int) : This value is there to maintain input
                compatibility with `tiatoolbox.models.classification` and is
                not used.
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
        self.imgs = None
        self.masks = None

        self.dataset_class: WSIStreamDataset = dataset_class
        self.model = model  # original copy
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.num_postproc_worker = num_postproc_worker
        self.verbose = verbose
        self.auto_generate_mask = auto_generate_mask

    @staticmethod
    def get_coordinates(
        image_shape: Union[List[int], np.ndarray], ioconfig: IOConfigSegmentor
    ):
        """Calculate patch tiling coordinates.

        By default, internally, it will call the `PatchExtractor.get_coordinates`.
        To use your own approaches, either subclass to overwrite or directly
        assign your own function to this name. In either cases, the function must
        obey the API defined here.

        Args:
            image_shape (a tuple (int, int) or :class:`numpy.ndarray` of shape (2,)):
                This argument specifies the shape of mother image (the image we want to)
                extract patches from) at requested `resolution` and `units` and it is
                expected to be in (width, height) format.

            ioconfig (object): object that contains information about input and ouput
                placement of patches. Check `IOConfigSegmentor` for details about
                available attributes.

        Return:
            patch_inputs: a list of corrdinates in
                `[start_x, start_y, end_x, end_y]` format indicating the read location
                of the patch in the mother image.

            patch_outputs: a list of corrdinates in
                `[start_x, start_y, end_x, end_y]` format indicating the write location
                of the patch in the mother image.

        Examples:

            >>> # API of function expected to overwrite `get_coordinates`
            >>> def func(image_shape, ioconfig):
            ...   patch_inputs = np.array([[0, 0, 256, 256]])
            ...   patch_outputs = np.array([[0, 0, 256, 256]])
            ...   return patch_inputs, patch_outputs
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.get_coordinates = func

        """
        (patch_inputs, patch_outputs) = PatchExtractor.get_coordinates(
            image_shape=image_shape,
            patch_input_shape=ioconfig.patch_input_shape,
            patch_output_shape=ioconfig.patch_output_shape,
            stride_shape=ioconfig.stride_shape,
        )
        return patch_inputs, patch_outputs

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Union[float, int] = None,
        units: str = None,
    ):
        """
        Indicates which coordinate is valid basing on the mask.

        To use your own approaches, either subclass to overwrite or directly
        assign your own function to this name. In either cases, the function must
        obey the API defined here.

        Args:
            mask_reader (:class:`.VirtualReader`): a virtual pyramidal
                reader of the mask related to the WSI from which we want
                to extract the patches.

            bounds (ndarray and np.int32): Coordinates to be checked
                via the `func`. They must be in the same resolution as requested
                `resolution` and `units`. The shape of `coordinatess` is (N, K)
                where N is the number of coordinate sets and K is either 2 for centroids
                or 4 for bounding boxes. When using the default `func=None`, K should be
                4, as we expect the `coordinatess` to be refer to bounding boxes in
                `[start_x, start_y, end_x, end_y]` format.

        Returns:
            ndarray: list of flags to indicate which coordinate is valid.

        Examples:

            >>> # API of function expected to overwrite `filter_coordinates`
            >>> def func(reader, bounds, resolution, units):
            ...   # as example, only select first bound
            ...   return np.array([1, 0])
            >>> coords = [[0, 0, 256, 256], [128, 128, 384, 384]]
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.filter_coordinates = func

        """
        if not isinstance(mask_reader, VirtualWSIReader):
            raise ValueError("`mask_reader` should be VirtualWSIReader.")
        if not isinstance(bounds, np.ndarray) or not np.issubdtype(
            bounds.dtype, np.integer
        ):
            raise ValueError("`coordinatess` should be ndarray of integer type.")

        mask_real_shape = mask_reader.img.shape[:2]
        mask_resolution_shape = mask_reader.slide_dimensions(
            resolution=resolution, units=units
        )[::-1]
        mask_real_shape = np.array(mask_real_shape)
        mask_resolution_shape = np.array(mask_resolution_shape)
        scale_factor = mask_real_shape / mask_resolution_shape
        scale_factor = scale_factor[0]  # what if ratio x != y

        def sel_func(coord: np.ndarray):
            """Accept coord as long as its box contains bits of mask."""
            coord_in_real_mask = np.ceil(scale_factor * coord).astype(np.int32)
            tl_x, tl_y, br_x, br_y = coord_in_real_mask
            roi = mask_reader.img[tl_y:br_y, tl_x:br_x]
            return np.sum(roi > 0) > 0

        flags = [sel_func(bound) for bound in bounds]
        return np.array(flags)

    @staticmethod
    def get_reader(img_path: str, mask_path: str, mode: str, auto_get_mask: bool):
        """Define how to get reader for mask and source image."""
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
        self,
        wsi_idx: int,
        ioconfig: IOConfigSegmentor,
        save_path: str,
        mode: str,
    ):
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int): index of the tile/wsi to be processed within `self`.
            ioconfig (IOConfigSegmentor): object which defines I/O placement during
                inference and when assembling back to full tile/wsi.
            loader (torch.Dataloader): loader object which return batch of data
                to be input to model.
            save_path (str): location to save output prediction as well as possible
                intermediat results.
            mode (str): `tile` or `wsi` to indicate run mode.

        """
        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path, mask_path, mode, self.auto_generate_mask
        )

        resolution = ioconfig.highest_input_resolution
        if (
            isinstance(wsi_reader, VirtualWSIReader)
            and resolution["units"] != "baseline"
        ):
            raise ValueError("Inference on `tile` only use `units='baseline'` !")
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch and tile placement
        # this is in XY
        (patch_inputs, patch_outputs) = self.get_coordinates(wsi_proc_shape, ioconfig)
        if mask_reader is not None:
            sel = self.filter_coordinates(mask_reader, patch_outputs, **resolution)
            patch_outputs = patch_outputs[sel]
            patch_inputs = patch_inputs[sel]

        # modify the shared space so that we can update worker info without
        # needing to re-create the worker. There should be no race-condition because
        # only the following enumerate loop triggers the parallelism, and this portion
        # is still in sequential execution order
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

        # ! TODO: need a protocol for pbar, or a decorator to make this less redundant
        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(self._loader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for _, batch_data in enumerate(self._loader):
            sample_datas, sample_infos = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)
            sample_outputs = self.model.infer_batch(
                self._model,
                sample_datas,
                self._on_gpu,
            )
            # repackage so that its a N list, each contains
            # L x etc. output
            sample_outputs = [np.split(v, batch_size, axis=0) for v in sample_outputs]
            sample_outputs = list(zip(*sample_outputs))

            # tensor to numpy, costly?
            sample_infos = sample_infos.numpy()
            sample_infos = np.split(sample_infos, batch_size, axis=0)

            sample_outputs = list(zip(sample_infos, sample_outputs))
            cum_output.extend(sample_outputs)
            pbar.update()
        pbar.close()
        self._process_predictions(cum_output, wsi_reader, ioconfig, save_path)

    def _process_predictions(
        self, cum_batch_predictions, wsi_reader, ioconfig, save_path
    ):
        """Define how the aggregated predictions are processed.

        This includes merging the prediction if necessary and also saving afterward.

        """
        # assume predictions is N, each item has L output element
        locations, predictions = list(zip(*cum_batch_predictions))
        # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of output patch
        # this can exceed the image bound at the requested resolution
        # remove singleton due to split.
        locations = np.array([v[0] for v in locations])
        for idx, output_resolution in enumerate(ioconfig.output_resolutions):
            # assume resolution idx to be in the same order as L
            merged_resolution = ioconfig.highest_input_resolution
            merged_locations = locations
            # ! location is wrt highest resolution, hence still need conversion
            if ioconfig.save_resolution is not None:
                merged_resolution = ioconfig.save_resolution
                output_shape = wsi_reader.slide_dimensions(**output_resolution)
                merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                fx = merged_shape[0] / output_shape[0]
                merged_locations = np.ceil(locations * fx).astype(np.int64)
            merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
            # 0 idx is to remove singleton wihout removing other axes singleton
            to_merge_predictions = [v[idx][0] for v in predictions]
            sub_save_path = f"{save_path}.raw.{idx}.npy"
            self.merge_prediction(
                merged_shape[::-1],  # XY to YX
                to_merge_predictions,
                merged_locations,
                save_path=sub_save_path,
                free_prediction=True,
            )

    @staticmethod
    def merge_prediction(
        canvas_shape: Union[Tuple[int], List[int], np.ndarray],
        predictions: List[np.ndarray],
        locations: Union[List, np.ndarray],
        save_path: Union[str, pathlib.Path] = None,
        free_prediction: bool = True,
    ):
        """Merge patch-level predictions to form a 2-dimensional prediction map.

        Args:
            canvas_shape (:class:`numpy.ndarray`): HW of the supposed assembled image.
            predictions (list): List of nd.array, each item is a prediction of
                a patch, assuming to be of shape HWC.
            locations (list): List of nd.array, each item is the location of
                the patch at the same index within `predictions`. The location
                is in the to be assembled canvas and of the form
                (top_left_x, top_left_y, bottom_right_x, bottom_right_x).
            save_path (str): Location to save the assembled image.
            free_prediction (bool): If this is `True`, `predictions` will
                be modified in place and each patch will be replace with `None`
                once processed. This is to save memory when assembling.

        Examples:

        >>> SemanticSegmentor.merge_prediction(
        ...     canvas_shape=[4, 4],
        ...     predictions=[
        ...         np.full((2, 2), 1),
        ...         np.full((2, 2), 2)],
        ...     locations=[
        ...         [0, 0, 2, 2],
        ...         [2, 2, 4, 4]],
        ...     save_path=None,
        ...     free_prediction=False,
        ... )
        array([[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2]])

        """
        canvas_shape = np.array(canvas_shape)

        sample_prediction = predictions[0]

        num_output_ch = 0
        add_singleton = False
        canvas_count_shape_ = tuple(canvas_shape)
        canvas_cum_shape_ = tuple(canvas_shape)
        if len(sample_prediction.shape) == 3:
            num_output_ch = sample_prediction.shape[-1]
            canvas_cum_shape_ += (num_output_ch,)
            add_singleton = num_output_ch == 1

        if save_path is not None:
            cum_canvas = np.lib.format.open_memmap(
                save_path,
                mode="w+",
                shape=canvas_cum_shape_,
                dtype=np.float32,
            )
        else:
            cum_canvas = np.zeros(
                shape=canvas_cum_shape_,
                dtype=np.float32,
            )

        # for pixel occurence counting
        # ! this may be expensive
        count_canvas = np.zeros(canvas_count_shape_, dtype=np.float32)

        patch_infos = list(zip(locations, predictions))
        for patch_idx, patch_info in enumerate(patch_infos):
            # position is assumed to be in XY coordinate
            (bound_in_wsi, prediction) = patch_info
            # convert to XY to YX, and in tl, br
            tl_in_wsi = np.array(bound_in_wsi[:2][::-1])
            br_in_wsi = np.array(bound_in_wsi[2:][::-1])
            old_tl_in_wsi = tl_in_wsi.copy()

            # need to do conversion
            patch_shape_in_wsi = tuple(br_in_wsi - tl_in_wsi)
            # conversion to make cv2 happy
            prediction = prediction.astype(np.float32)
            prediction = cv2.resize(prediction, patch_shape_in_wsi[::-1])
            # ! cv2 resize will remove singleton !
            if add_singleton:
                prediction = prediction[..., None]

            sel = tl_in_wsi < 0
            tl_in_wsi[sel] = 0

            if np.any(tl_in_wsi >= canvas_shape):
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
            count_canvas[
                tl_in_wsi[0] : br_in_wsi[0], tl_in_wsi[1] : br_in_wsi[1]
            ] += patch_count
            # remove prediction without altering list ordering or length
            if free_prediction:
                patch_infos[patch_idx] = None
        if num_output_ch > 0:
            count_canvas = count_canvas[..., None]
        cum_canvas /= count_canvas + 1.0e-6
        return cum_canvas

    def predict(
        self,
        imgs,
        masks=None,
        mode="tile",
        on_gpu=True,
        ioconfig=None,
        patch_input_shape=None,
        patch_output_shape=None,
        stride_shape=None,  # at requested read resolution, not wrt to lv0
        resolution=1.0,
        units="baseline",
        save_dir=None,
        crash_on_exception=False,
    ):
        """Make a prediction for a list of input data.

        Args:
            imgs (list, ndarray): List of inputs to process. When using `patch`
                mode, the input must be either a list of images, a list of image file
                paths or a numpy array of an image list. When using `tile` or `wsi`
                mode, the input must be a list of file paths.
            masks (list): List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if they are witin a
                masked area. If not provided, then a tissue mask will be automatically
                generated for whole-slide images or the entire image is processed for
                image tiles.
            mode (str): Type of input to process. Choose from either `tile` or `wsi`.
            ioconfig (object): object that define information about input and ouput
                placement of patches. When provided, `patch_input_shape`,
                `patch_output_shape`, `stride_shape`, `resolution`, and `units`
                arguments are ignore. Otherwise, those arguments will be internally
                converted to an ioconfig object.
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
            save_dir (str): Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output` where the
                running script is invoked.
            crash_on_exception (bool): If `True`, the running loop will crash
                if there is any error during processing a WSI. Otherwise, the loop
                will move on to the next wsi for processing.

        Returns:
            output (list): A list of tuple(input_path, save_path) where
                `input_path` is the path of the input wsi while `save_path`
                corresponds to the output predictions.

        Examples:
            >>> # Sample output of a network
            >>> wsis = ['A/wsi1.svs', 'B/wsi2.svs']
            >>> predictor = SemanticSegmentor(model='unet')
            >>> output = predictor.predict(wsis, mode='wsi')
            >>> output.keys()
            [('A/wsi.svs', 'output/0.raw') , ('B/wsi.svs', 'output/1.raw')]
            >>> # if a network have 2 output heads, each head output of 'A/wsi.svs'
            >>> # will be respectively stored as 'output/0.raw.0', 'output/0.raw.1'

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

        if patch_output_shape is None:
            patch_output_shape = patch_input_shape
        if stride_shape is None:
            stride_shape = patch_output_shape

        if ioconfig is None:
            ioconfig = IOConfigSegmentor(
                input_resolutions=[{"resolution": resolution, "units": units}],
                output_resolutions=[{"resolution": resolution, "units": units}],
                patch_input_shape=patch_input_shape,
                patch_output_shape=patch_output_shape,
                stride_shape=stride_shape,
            )

        # use external for testing
        self._on_gpu = on_gpu
        self._model = misc.model_to(on_gpu, self.model)

        # workers should be > 0 else Value Error will be thrown
        self._postproc_workers = None
        if self.num_postproc_worker > 0:
            self._postproc_workers = ProcessPoolExecutor(
                max_workers=self.num_postproc_worker
            )

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        ds = self.dataset_class(
            ioconfig=ioconfig,
            preproc=self.model.preproc_func,
            wsi_paths=imgs,
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
        self._loader = loader

        self.imgs = imgs
        self.masks = masks

        # contain input / ouput prediction mapping
        outputs = []
        # ? what will happen if this crash midway?
        # => may not be able to retrieve the result dict
        for wsi_idx, img_path in enumerate(imgs):
            try:
                wsi_save_path = os.path.join(save_dir, f"{wsi_idx}")
                self._predict_one_wsi(wsi_idx, ioconfig, wsi_save_path, mode)

                # dont use dict as mapping, because can overwrite, if that is
                # user intention to provide same path twice
                outputs.append([img_path, wsi_save_path])

                # will this corrupt old version if ctrl-c midway?
                map_file_path = os.path.join(save_dir, "file_map.dat")
                # backup old version first
                if os.path.exists(map_file_path):
                    old_map_file_path = os.path.join(save_dir, "file_map_old.dat")
                    shutil.copy(map_file_path, old_map_file_path)
                joblib.dump(outputs, map_file_path)

                # verbose mode, error by passing ?
                logging.info(f"Finish: {wsi_idx}/{len(imgs)}")
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
        self.imgs = None
        self.masks = None
        self._model = None
        self._loader = None
        self._on_gpu = None
        self._futures = None
        self._mp_shared_space = None
        if self._postproc_workers is not None:
            self._postproc_workers.shutdown()
        self._postproc_workers = None
        return outputs
