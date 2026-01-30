"""This module enables multi-task segmentor."""

from __future__ import annotations

import gc
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd
import psutil
import torch
import zarr
from dask import compute
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as feature2geometry
from shapely.strtree import STRtree
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.annotation.storage import Annotation
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import create_smart_array, get_tqdm, make_valid_poly
from tiatoolbox.wsicore.wsireader import is_zarr

from .semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
    concatenate_none,
    merge_batch_to_canvas,
    store_probabilities,
)

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader
    from tqdm import tqdm, tqdm_notebook

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


class MultiTaskSegmentorRunParams(SemanticSegmentorRunParams, total=False):
    """Runtime parameters for configuring the `MultiTaskSegmentor.run()` method.

    This class extends `SemanticSegmentorRunParams`, and adds parameters specific
    to multitask segmentation workflows.

    Attributes:
        auto_get_mask (bool):
            Whether to automatically generate segmentation masks using
            `wsireader.tissue_mask()` during processing.
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        class_dict (dict):
            Optional dictionary mapping classification outputs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda").
        labels (list):
            Optional labels for input images. Only a single label per image
            is supported.
        memory_threshold (int):
            Memory usage threshold (in percentage) to trigger caching behavior.
        num_workers (int):
            Number of workers used in DataLoader.
        output_file (str):
            Output file name for saving results (e.g., .zarr or .db).
        output_resolutions (Resolution):
            Resolution used for writing output predictions.
        patch_output_shape (tuple[int, int]):
            Shape of output patches (height, width).
        return_labels (bool):
            Whether to return labels with predictions.
        return_probabilities (bool):
            Whether to return per-class probabilities.
        return_predictions (tuple(bool, ...):
            Whether to return array predictions for individual tasks.
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.

    """

    return_predictions: tuple[bool, ...]


class MultiTaskSegmentor(SemanticSegmentor):
    """A multitask segmentation engine for models like hovernet and hovernetplus."""

    def __init__(
        self: MultiTaskSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusInstanceSegmentor`."""
        self.tasks = set()
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def infer_patches(
        self: MultiTaskSegmentor,
        dataloader: DataLoader,
        *,
        return_coordinates: bool = False,
    ) -> dict[str, list[da.Array]]:
        """Run model inference on image patches and return predictions.

        This method performs batched inference using a PyTorch DataLoader,
        and accumulates predictions in Dask arrays. It supports optional inclusion
        of coordinates and labels in the output.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader containing image patches for inference.
            return_coordinates (bool):
                Whether to include coordinates in the output. Required when
                called by `infer_wsi` and `patch_mode` is False.

        Returns:
            dict[str, dask.array.Array]:
                Dictionary containing prediction results as Dask arrays.
                Keys include:
                    - "probabilities": Model output probabilities.
                    - "coordinates": Patch coordinates (if `return_coordinates` is
                      True).

        """
        keys = ["probabilities"]
        coordinates = []

        # Expected number of outputs from the model
        batch_output = self.model.infer_batch(
            self.model,
            torch.Tensor(dataloader.dataset[0]["image"][np.newaxis, ...]),
            device=self.device,
        )

        num_expected_output = len(batch_output)
        probabilities = [[] for _ in range(num_expected_output)]

        if return_coordinates:
            keys.append("coordinates")
            coordinates = []

        # Main output dictionary
        raw_predictions = {key: [] for key in keys}
        raw_predictions["probabilities"] = [[] for _ in range(num_expected_output)]

        # Inference loop
        tqdm_ = get_tqdm()
        tqdm_loop = (
            tqdm_(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else self.dataloader
        )

        for batch_data in tqdm_loop:
            batch_output = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            for i in range(num_expected_output):
                probabilities[i].append(
                    da.from_array(
                        batch_output[i],  # probabilities
                    )
                )

            if return_coordinates:
                coordinates.append(
                    da.from_array(
                        self._get_coordinates(batch_data),
                    )
                )

        for i in range(num_expected_output):
            raw_predictions["probabilities"][i] = da.concatenate(
                probabilities[i], axis=0
            )

        if return_coordinates:
            raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)

        return raw_predictions

    def infer_wsi(
        self: SemanticSegmentor,
        dataloader: DataLoader,
        save_path: Path,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict[str, da.Array]:
        """Perform model inference on a whole slide image (WSI)."""
        # Default Memory threshold percentage is 80.
        memory_threshold = kwargs.get("memory_threshold", 80)

        keys = ["probabilities", "coordinates"]
        coordinates = []

        # Main output dictionary
        raw_predictions = dict(
            zip(keys, [da.empty(shape=(0, 0))] * len(keys), strict=False)
        )

        # Inference loop
        tqdm_ = get_tqdm()
        tqdm_loop = (
            tqdm_(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else dataloader
        )

        # Expected number of outputs from the model
        batch_output = self.model.infer_batch(
            self.model,
            torch.Tensor(dataloader.dataset[0]["image"][np.newaxis, ...]),
            device=self.device,
        )

        num_expected_output = len(batch_output)
        canvas_np = [None for _ in range(num_expected_output)]
        canvas = [None for _ in range(num_expected_output)]
        count = [None for _ in range(num_expected_output)]
        canvas_zarr = [None for _ in range(num_expected_output)]
        count_zarr = [None for _ in range(num_expected_output)]

        output_locs_y_, output_locs = None, None

        full_output_locs = (
            dataloader.dataset.full_outputs
            if hasattr(dataloader.dataset, "full_outputs")
            else dataloader.dataset.outputs
        )

        infer_batch = self._get_model_attr("infer_batch")
        for batch_idx, batch_data in enumerate(tqdm_loop):
            batch_output = infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            batch_locs = batch_data["output_locs"].numpy()

            # Interpolate outputs for masked regions
            full_batch_output, full_output_locs, output_locs = (
                prepare_multitask_full_batch(
                    batch_output,
                    batch_locs,
                    full_output_locs,
                    output_locs,
                    is_last=(batch_idx == (len(dataloader) - 1)),
                )
            )

            for idx, full_batch_output_ in enumerate(full_batch_output):
                canvas_np[idx] = concatenate_none(
                    old_arr=canvas_np[idx], new_arr=full_batch_output_
                )

            # Determine if dataloader is moved to next row of patches
            change_indices = np.where(np.diff(output_locs[:, 1]) != 0)[0] + 1

            # If a row of patches has been processed.
            if change_indices.size > 0:
                canvas, count, canvas_np, output_locs, output_locs_y_ = (
                    merge_multitask_horizontal(
                        canvas,
                        count,
                        output_locs_y_,
                        canvas_np,
                        output_locs,
                        change_indices,
                    )
                )

                canvas, count, canvas_zarr, count_zarr, tqdm_loop = (
                    _check_and_update_for_memory_overload(
                        canvas=canvas,
                        count=count,
                        canvas_zarr=canvas_zarr,
                        count_zarr=count_zarr,
                        memory_threshold=memory_threshold,
                        tqdm_loop=tqdm_loop,
                        tqdm_=tqdm_,
                        save_path=save_path,
                        num_expected_output=num_expected_output,
                    )
                )

            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

        canvas, count, _, _, output_locs_y_ = merge_multitask_horizontal(
            canvas,
            count,
            output_locs_y_,
            canvas_np,
            output_locs,
            change_indices=[len(output_locs)],
        )

        raw_predictions["probabilities"] = _calculate_probabilities(
            canvas_zarr=canvas_zarr,
            count_zarr=count_zarr,
            canvas=canvas,
            count=count,
            output_locs_y_=output_locs_y_,
            save_path=save_path,
            memory_threshold=memory_threshold,
        )

        raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)

        return raw_predictions

    def post_process_patches(  # skipcq: PYL-R0201
        self: MultiTaskSegmentor,
        raw_predictions: dict,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],  # noqa: ARG002
    ) -> dict:
        """Post-process raw patch predictions from inference.

        This method applies a post-processing function (e.g., smoothing, filtering)
        to the raw model predictions. It supports delayed execution using Dask
        and returns a Dask array for efficient computation.

        Args:
            raw_predictions (dask.array.Array):
                Raw model predictions as a dask array.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters used for post-processing.

        Returns:
            dask.array.Array:
                Post-processed predictions as a Dask array.

        """
        probabilities = raw_predictions["probabilities"]
        post_process_predictions = [
            self.model.postproc_func(list(probs_for_idx))
            for probs_for_idx in zip(*probabilities, strict=False)
        ]

        raw_predictions = self.build_post_process_raw_predictions(
            post_process_predictions=post_process_predictions,
            raw_predictions=raw_predictions,
        )

        # Need to update info_dict
        _ = raw_predictions

        return raw_predictions

    def post_process_wsi(  # skipcq: PYL-R0201
        self: MultiTaskSegmentor,
        raw_predictions: dict,
        save_path: Path,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict:
        """Post-process raw patch predictions from inference."""
        probabilities = raw_predictions["probabilities"]

        probabilities_is_zarr = False
        for probabilities_ in probabilities:
            if any("from-zarr" in str(key) for key in probabilities_.dask.layers):
                probabilities_is_zarr = True
                break

        return_predictions = kwargs.get("return_predictions")
        # If dask array can fit in memory process without tiling.
        # This ignores post-processing tile size even if it is smaller.
        if not probabilities_is_zarr:
            post_process_predictions = self._process_full_wsi(
                probabilities=probabilities,
                return_predictions=return_predictions,
            )
        else:
            post_process_predictions = self._process_tile_mode(
                probabilities=probabilities,
                save_path=save_path.with_suffix(".zarr"),
                memory_threshold=kwargs.get("memory_threshold", 80),
                return_predictions=kwargs.get("return_predictions"),
            )

        tasks = set()
        for seg in post_process_predictions:
            task_name = seg["task_type"]
            tasks.add(task_name)
            raw_predictions[task_name] = {}

            for key, value in seg.items():
                if key == "task_type":
                    continue
                if isinstance(value, (np.ndarray, da.Array)):
                    raw_predictions[task_name][key] = da.array(value)

                if isinstance(value, dict):
                    for k, v in value.items():
                        raw_predictions[task_name][k] = v

        self.tasks = tasks

        return raw_predictions

    def _process_full_wsi(
        self: MultiTaskSegmentor,
        probabilities: list[da.Array | np.ndarray],
        *,
        return_predictions: tuple[bool, ...] | None = None,
    ) -> list[dict] | None:
        """Helper function to post process WSI when it can fit in memory."""
        post_process_predictions = self.model.postproc_func(probabilities)
        if return_predictions is None:
            return_predictions = [False for _ in post_process_predictions]
        for idx, return_predictions_ in enumerate(return_predictions):
            if not return_predictions_:
                del post_process_predictions[idx]["predictions"]

    def _process_tile_mode(
        self: MultiTaskSegmentor,
        probabilities: list[da.Array | np.ndarray],
        save_path: Path,
        memory_threshold: float = 80,
        *,
        return_predictions: tuple[bool, ...] | None = None,
    ) -> list[dict] | None:
        """Helper function to process WSI in tile mode."""
        highest_input_resolution = self.ioconfig.highest_input_resolution
        wsi_reader = self.dataloader.dataset.reader

        # assume ioconfig has already been converted to `baseline` for `tile` mode
        wsi_proc_shape = wsi_reader.slide_dimensions(**highest_input_resolution)

        # * retrieve tile placement and tile info flag
        # tile shape will always be corrected to be multiple of output
        tile_info_sets = self._get_tile_info(wsi_proc_shape, self.ioconfig)
        ioconfig = self.ioconfig.to_baseline()

        merged = []
        wsi_info_dict = None
        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            for tile_idx, tile_bounds in enumerate(set_bounds):
                tile_flag = set_flags[tile_idx]
                tile_tl = tile_bounds[:2]
                tile_br = tile_bounds[2:]
                tile_shape = tile_br - tile_tl  # in width height
                head_raws = [
                    probabilities_[
                        tile_bounds[1] : tile_bounds[3],
                        tile_bounds[0] : tile_bounds[2],
                        :,
                    ].compute()
                    for probabilities_ in probabilities
                ]
                post_process_output = self.model.postproc_func(head_raws)

                # create a list of info dict for each task
                wsi_info_dict = _create_wsi_info_dict(
                    post_process_output=post_process_output,
                    wsi_info_dict=wsi_info_dict,
                    wsi_proc_shape=wsi_proc_shape,
                    save_path=save_path,
                    memory_threshold=memory_threshold,
                    return_predictions=return_predictions,
                )

                wsi_info_dict = _update_tile_based_predictions_array(
                    post_process_output=post_process_output,
                    wsi_info_dict=wsi_info_dict,
                    bounds=tile_bounds,
                )

                inst_dicts = _get_inst_info_dicts(
                    post_process_output=post_process_output
                )

                tile_mode = set_idx
                new_inst_dicts, remove_insts_in_origs = [], []
                for inst_id, inst_dict in enumerate(inst_dicts):
                    new_inst_dict, remove_insts_in_orig = _process_instance_predictions(
                        inst_dict,
                        ioconfig,
                        tile_shape,
                        tile_flag,
                        tile_mode,
                        tile_tl,
                        wsi_info_dict[inst_id]["info_dict"],
                    )
                    new_inst_dicts.append(new_inst_dict)
                    remove_insts_in_origs.append(remove_insts_in_orig)

                merged.append((new_inst_dicts, remove_insts_in_origs))

            for new_inst_dicts, remove_uuid_lists in merged:
                for inst_id, new_inst_dict in enumerate(new_inst_dicts):
                    wsi_info_dict[inst_id]["info_dict"].update(new_inst_dict)
                    for inst_uuid in remove_uuid_lists[inst_id]:
                        wsi_info_dict[inst_id]["info_dict"].pop(inst_uuid, None)

        for idx, wsi_info_dict_ in enumerate(wsi_info_dict):
            info_df = pd.DataFrame(wsi_info_dict_["info_dict"]).transpose()
            dict_info_wsi = {}
            for key, col in info_df.items():
                col_np = col.to_numpy()
                dict_info_wsi[key] = da.from_array(
                    col_np,
                    chunks=(len(col),),
                )
            wsi_info_dict[idx]["info_dict"] = dict_info_wsi

        return wsi_info_dict

    @staticmethod
    def _get_tile_info(
        image_shape: list[int] | np.ndarray,
        ioconfig: IOSegmentorConfig,
    ) -> list[list, ...]:
        """Generating tile information.

        To avoid out of memory problem when processing WSI-scale in
        general, the predictor will perform the inference and assemble
        on a large image tiles (each may have size of 4000x4000 compared
        to patch output of 256x256) first before stitching every tiles
        by the end to complete the WSI output. For nuclei instance
        segmentation, the stitching process will require removal of
        predictions within some bounding areas. This function generates
        both the tile placement and the flag to indicate how the removal
        should be done to achieve the above goal.

        Args:
            image_shape (:class:`numpy.ndarray`, list(int)):
                The shape of WSI to extract the tile from, assumed to be
                in `[width, height]`.
            ioconfig (:obj:IOSegmentorConfig):
                The input and output configuration objects.

        Returns:
            list:
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Grid tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Vertical strip tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Horizontal strip tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Cross-section tiles
                    - :class:`numpy.ndarray` - Removal flags

        """
        margin = np.array(ioconfig.margin)
        tile_shape = np.array(ioconfig.tile_shape)
        tile_shape = (
            np.floor(tile_shape / ioconfig.patch_output_shape)
            * ioconfig.patch_output_shape
        ).astype(np.int32)
        image_shape = np.array(image_shape)
        tile_outputs = PatchExtractor.get_coordinates(
            image_shape=image_shape,
            patch_input_shape=tile_shape,
            patch_output_shape=tile_shape,
            stride_shape=tile_shape,
        )

        # * === Now generating the flags to indicate which side should
        # * === be removed in postproc callback
        boxes = tile_outputs[1]

        # This saves computation time if the image is smaller than the expected tile
        if np.all(image_shape <= tile_shape):
            flag = np.zeros([boxes.shape[0], 4], dtype=np.int32)
            return [[boxes, flag]]

        # * remove all sides for boxes
        # unset for those lie within the selection
        def unset_removal_flag(boxes: tuple, removal_flag: np.ndarray) -> np.ndarray:
            """Unset removal flags for tiles intersecting image boundaries."""
            sel_boxes = [
                shapely_box(0, 0, w, 0),  # top edge
                shapely_box(0, h, w, h),  # bottom edge
                shapely_box(0, 0, 0, h),  # left
                shapely_box(w, 0, w, h),  # right
            ]
            geometries = [shapely_box(*bounds) for bounds in boxes]
            spatial_indexer = STRtree(geometries)

            for idx, sel_box in enumerate(sel_boxes):
                sel_indices = list(spatial_indexer.query(sel_box))
                removal_flag[sel_indices, idx] = 0
            return removal_flag

        w, h = image_shape
        boxes = tile_outputs[1]
        #  expand to full four corners
        boxes_br = boxes[:, 2:]
        boxes_tr = np.dstack([boxes[:, 2], boxes[:, 1]])[0]
        boxes_bl = np.dstack([boxes[:, 0], boxes[:, 3]])[0]

        # * remove edges on all sides, excluding edges at on WSI boundary
        flag = np.ones([boxes.shape[0], 4], dtype=np.int32)
        flag = unset_removal_flag(boxes, flag)
        info = deque([[boxes, flag]])

        # * create vertical boxes at tile boundary and
        # * flag top and bottom removal, excluding those
        # * on the WSI boundary
        # -------------------
        # |    =|=   =|=    |
        # |    =|=   =|=    |
        # |   >=|=  >=|=    |
        # -------------------
        # |   >=|=  >=|=    |
        # |    =|=   =|=    |
        # |   >=|=  >=|=    |
        # -------------------
        # |   >=|=  >=|=    |
        # |    =|=   =|=    |
        # |    =|=   =|=    |
        # -------------------
        # only select boxes having right edges removed
        sel_indices = np.nonzero(flag[..., 3])
        _boxes = np.concatenate(
            [
                boxes_tr[sel_indices] - np.array([margin, 0])[None],
                boxes_br[sel_indices] + np.array([margin, 0])[None],
            ],
            axis=-1,
        )
        _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
        _flag[:, [0, 1]] = 1
        _flag = unset_removal_flag(_boxes, _flag)
        info.append([_boxes, _flag])

        # * create horizontal boxes at tile boundary and
        # * flag left and right removal, excluding those
        # * on the WSI boundary
        # -------------
        # |   |   |   |
        # |  v|v v|v  |
        # |===|===|===|
        # -------------
        # |===|===|===|
        # |   |   |   |
        # |   |   |   |
        # -------------
        # only select boxes having bottom edges removed
        sel_indices = np.nonzero(flag[..., 1])
        # top bottom left right
        _boxes = np.concatenate(
            [
                boxes_bl[sel_indices] - np.array([0, margin])[None],
                boxes_br[sel_indices] + np.array([0, margin])[None],
            ],
            axis=-1,
        )
        _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
        _flag[:, [2, 3]] = 1
        _flag = unset_removal_flag(_boxes, _flag)
        info.append([_boxes, _flag])

        # * create boxes at tile cross-section and all sides
        # ------------------------
        # |     |     |     |    |
        # |    v|     |     |    |
        # |  > =|=   =|=   =|=   |
        # -----=-=---=-=---=-=----
        # |    =|=   =|=   =|=   |
        # |     |     |     |    |
        # |    =|=   =|=   =|=   |
        # -----=-=---=-=---=-=----
        # |    =|=   =|=   =|=   |
        # |     |     |     |    |
        # |     |     |     |    |
        # ------------------------

        # only select boxes having both right and bottom edges removed
        sel_indices = np.nonzero(np.prod(flag[:, [1, 3]], axis=-1))
        _boxes = np.concatenate(
            [
                boxes_br[sel_indices] - np.array([2 * margin, 2 * margin])[None],
                boxes_br[sel_indices] + np.array([2 * margin, 2 * margin])[None],
            ],
            axis=-1,
        )
        flag = np.full([_boxes.shape[0], 4], 1, dtype=np.int32)
        info.append([_boxes, flag])

        return info

    def build_post_process_raw_predictions(
        self: MultiTaskSegmentor,
        post_process_predictions: list[tuple],
        raw_predictions: dict,
    ) -> dict:
        """Merge per-image outputs into a task-organized prediction structure.

        This function takes a list of outputs, where each element corresponds to one
        image and contains one or more segmentation dictionaries. Each segmentation
        dictionary must include a ``"task_type"`` key along with any number of
        additional fields (e.g., ``"predictions"``, ``"info_dict"``, or others).

        The function reorganizes these outputs into ``raw_predictions`` by grouping
        entries under their respective task types. For each task, all keys except
        ``"task_type"`` are stored in dictionaries indexed by ``img_id``. Existing
        content in ``raw_predictions`` is preserved and extended as needed.

        Args:
            post_process_predictions (list[tuple]):
                A list where each element represents one image. Each element is an
                iterable of segmentation dictionaries. Each segmentation dictionary
                must contain a ``"task_type"`` field and may contain any number of
                additional fields.
            raw_predictions (dict):
                A dictionary that will be updated in-place. It may already contain
                task entries or other unrelated keys. New tasks and new fields are
                added dynamically as they appear in ``outputs``.

        Returns:
            dict:
                The updated ``raw_predictions`` dictionary, containing all tasks and
                their associated per-image fields.

        """
        tasks = set()
        for seg_list in post_process_predictions:
            for seg in seg_list:
                task = seg["task_type"]
                tasks.add(task)

                # Initialize task entry if needed
                if task not in raw_predictions:
                    raw_predictions[task] = {}

                # For every key except task_type, store values by img_id
                for key, value in seg.items():
                    if key == "task_type":
                        continue

                    # Initialize list for this key
                    if key not in raw_predictions[task]:
                        raw_predictions[task][key] = []

                    raw_predictions[task][key].append(value)

        for task in tasks:
            task_dict = raw_predictions[task]
            for key in list(task_dict.keys()):
                values = task_dict[key]
                if all(isinstance(v, (np.ndarray, da.Array)) for v in values):
                    raw_predictions[task][key] = da.stack(values, axis=0)

                if all(isinstance(v, dict) for v in values):
                    first = values[0]

                    # Add new keys safely
                    for subkey in first:
                        raw_predictions[task][subkey] = [d[subkey] for d in values]

                    del raw_predictions[task][key]

        self.tasks = tasks
        return raw_predictions

    def _save_predictions_as_dict_zarr(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Helper function to save predictions as dictionary or zarr."""
        if output_type.lower() == "dict":
            # If there is a single task simplify the output.
            if len(self.tasks) == 1:
                task_output = processed_predictions.pop(next(iter(self.tasks)))
                processed_predictions.update(task_output)
            return super().save_predictions(
                processed_predictions, output_type, save_path=save_path, **kwargs
            )

        # Save to zarr
        if kwargs.get("return_probabilities", False):
            _ = self.save_predictions_as_zarr(
                processed_predictions={
                    "probabilities": processed_predictions.pop("probabilities")
                },
                save_path=save_path,
                keys_to_compute=["probabilities"],
                task_name=None,
            )

        for task_name in self.tasks:
            processed_predictions_ = processed_predictions.pop(task_name)
            # If there is a single task simplify the output.
            task_name_ = None if len(self.tasks) == 1 else task_name
            keys_to_compute = [
                k for k in processed_predictions_ if k not in self.drop_keys
            ]
            if "coordinates" in processed_predictions:
                processed_predictions_.update(
                    {"coordinates": processed_predictions["coordinates"]}
                )
                keys_to_compute.extend(["coordinates"])
            _ = self.save_predictions_as_zarr(
                processed_predictions=processed_predictions_,
                save_path=save_path,
                keys_to_compute=keys_to_compute,
                task_name=task_name_,
            )
        return save_path

    def _save_predictions_as_annotationstore(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        task_name: str | None = None,
        save_path: Path | None = None,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Helper function to save predictions as annotationstore."""
        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")

        # Need to add support for zarr conversion.
        save_paths = []

        logger.info("Saving predictions as AnnotationStore.")

        # predictions are not required when saving to AnnotationStore.
        for key in ("canvas", "count", "predictions"):
            processed_predictions.pop(key, None)

        keys_to_compute = list(processed_predictions.keys())
        if "probabilities" in keys_to_compute:
            keys_to_compute.remove("probabilities")
        if self.patch_mode:
            for idx, curr_image in enumerate(self.images):
                values = [processed_predictions[key][idx] for key in keys_to_compute]
                output_path = _save_annotation_store(
                    curr_image=curr_image,
                    keys_to_compute=keys_to_compute,
                    values=values,
                    task_name=task_name,
                    idx=idx,
                    save_path=save_path,
                    class_dict=class_dict,
                    scale_factor=scale_factor,
                )
                save_paths.append(output_path)

        else:
            for idx, curr_image in enumerate(self.images):
                values = [processed_predictions[key] for key in keys_to_compute]
                output_path = _save_annotation_store(
                    curr_image=curr_image,
                    keys_to_compute=keys_to_compute,
                    values=values,
                    task_name=task_name,
                    idx=idx,
                    save_path=save_path,
                    class_dict=class_dict,
                    scale_factor=scale_factor,
                )
                save_paths.append(output_path)

        for key in keys_to_compute:
            del processed_predictions[key]

        return_probabilities = kwargs.get("return_probabilities", False)
        if return_probabilities:
            msg = (
                f"Probability maps cannot be saved as AnnotationStore. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {save_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)

        return save_paths

    def save_predictions(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Save model predictions to disk or return them in memory.

        Depending on the output type, this method saves predictions as a zarr group,
        an AnnotationStore (SQLite database), or returns them as a dictionary.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
            output_type (str):
                Desired output format.
                Supported values are "dict", "zarr", and "annotationstore".
            save_path (Path | None):
                Path to save the output file.
                Required for "zarr" and "annotationstore" formats.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters to update engine attributes.
                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").
                        See :class:`torch.device` for more details.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., "zarr" or "annotationstore").
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates from non-baseline to baseline
                        resolution.
                    stride_shape (IntPair):
                        Stride used during WSI processing, at requested read resolution.
                        Must be positive. Defaults to `patch_input_shape` if not
                        provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            dict | AnnotationStore | Path | list [Path]:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved zarr file.
                - If output_type is "annotationstore": returns an AnnotationStore
                  or path to .db file.

        Raises:
            TypeError:
                If an unsupported output_type is provided.

        """
        if output_type in ["dict", "zarr"]:
            return self._save_predictions_as_dict_zarr(
                processed_predictions=processed_predictions,
                output_type=output_type,
                save_path=save_path,
                **kwargs,
            )

        # Save to AnnotationStore
        return_probabilities = kwargs.get("return_probabilities", False)
        output_type_ = (
            "zarr"
            if is_zarr(save_path.with_suffix(".zarr")) or return_probabilities
            else "dict"
        )

        # This runs dask.compute and returns numpy arrays
        # for saving annotationstore output.
        class_dict = kwargs.get("class_dict", self.model.class_dict)
        if len(self.tasks) == 1:
            kwargs["class_dict"] = class_dict[next(iter(self.tasks))]
        else:
            kwargs["class_dict"] = class_dict

        processed_predictions = self._save_predictions_as_dict_zarr(
            processed_predictions,
            output_type=output_type_,
            save_path=save_path.with_suffix(".zarr"),
            **kwargs,
        )

        save_paths = []
        if isinstance(processed_predictions, Path):
            if return_probabilities:
                save_paths.append(processed_predictions)
            processed_predictions = zarr.open(str(processed_predictions), mode="r+")

        # For single tasks there should be no overlap
        if self.tasks & processed_predictions.keys():
            for task_name in self.tasks:
                dict_for_store = processed_predictions[task_name]
                kwargs["class_dict"] = class_dict[task_name]
                if "coordinates" in processed_predictions:
                    dict_for_store = {
                        **processed_predictions[task_name],
                        "coordinates": processed_predictions["coordinates"],
                    }
                out_path = self._save_predictions_as_annotationstore(
                    processed_predictions=dict_for_store,
                    task_name=task_name,
                    save_path=save_path,
                    **kwargs,
                )
                save_paths += out_path
                del processed_predictions[task_name]

            return save_paths

        return self._save_predictions_as_annotationstore(
            processed_predictions=processed_predictions,
            task_name=None,
            save_path=save_path,
            **kwargs,
        )

    def run(
        self: MultiTaskSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        *,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the semantic segmentation engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both
        patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`). Default
                is True.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore". Default
                is "dict".
            **kwargs (MultiTaskSegmentorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").

                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    output_resolutions (Resolution):
                        Resolution used for writing output predictions.
                    patch_output_shape (tuple[int, int]):
                        Shape of output patches (height, width).
                    return_labels (bool):
                        Whether to return labels with predictions. Should be False.
                    return_probabilities (bool):
                        Whether to return per-class probabilities.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = segmentor.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... "/path/to/wsi1.db"

        """
        return_labels = kwargs.get("return_labels")

        # Passing multitask labels causes unnecessary memory overheads
        if return_labels:
            msg = "`return_labels` is not supported for MultiTaskSegmentor."
            raise ValueError(msg)

        kwargs["return_labels"] = False

        return super().run(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )


def dict_to_store(
    store: SQLiteStore,
    processed_predictions: dict,
    class_dict: dict | None = None,
    origin: tuple[float, float] = (0, 0),
    scale_factor: tuple[float, float] = (1, 1),
) -> AnnotationStore:
    """Helper function to convert dict to store."""
    contour = processed_predictions.pop("contours")

    ann = []
    for i, contour_ in enumerate(contour):
        ann_ = Annotation(
            make_valid_poly(
                feature2geometry(
                    {
                        "type": processed_predictions.get("geom_type", "Polygon"),
                        "coordinates": scale_factor * np.array([contour_]),
                    },
                ),
                tuple(origin),
            ),
            {
                prop: (
                    class_dict[processed_predictions[prop][i]]
                    if prop == "type" and class_dict is not None
                    # Intention is convert arrays to list
                    # There might be int or float values which need to be
                    # converted to arrays first and then apply tolist().
                    else np.array(processed_predictions[prop][i]).tolist()
                )
                for prop in processed_predictions
            },
        )
        ann.append(ann_)
    logger.info("Added %d annotations.", len(ann))
    store.append_many(ann)

    return store


def prepare_multitask_full_batch(
    batch_output: tuple[np.ndarray],
    batch_locs: np.ndarray,
    full_output_locs: np.ndarray,
    output_locs: np.ndarray,
    *,
    is_last: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Prepare full-sized output and count arrays for a batch of patch predictions.

    This function aligns patch-level predictions with global output locations when
    a mask (e.g., auto_get_mask) is applied. It initializes full-sized arrays and
    fills them using matched indices. If the batch is the last in the sequence,
    it pads the arrays to cover remaining locations.

    Args:
        batch_output (np.ndarray):
            Patch-level model predictions of shape (N, H, W, C).
        batch_locs (np.ndarray):
            Output locations corresponding to `batch_output`.
        full_output_locs (np.ndarray):
            Remaining global output locations to be matched.
        output_locs (np.ndarray):
            Accumulated output location array across batches.
        is_last (bool):
            Flag indicating whether this is the final batch.

    Returns:
        tuple[list[np.ndarray], np.ndarray, np.ndarray]:
            - full_batch_output: Full-sized output array with predictions placed.
            - full_output_locs: Updated remaining global output locations.
            - output_locs: Updated accumulated output locations.

    """
    # Use np.intersect1d once numpy version is upgraded to 2.0
    full_output_dict = {tuple(row): i for i, row in enumerate(full_output_locs)}
    matches = [full_output_dict[tuple(row)] for row in batch_locs]

    total_size = np.max(matches).astype(np.uint32) + 1

    full_batch_output = [np.empty(0) for _ in range(len(batch_output))]

    for idx, batch_output_ in enumerate(batch_output):
        # Initialize full output array
        full_batch_output[idx] = np.zeros(
            shape=(total_size, *batch_output_.shape[1:]),
            dtype=batch_output_.dtype,
        )

        # Place matching outputs using matching indices
        full_batch_output[idx][matches] = batch_output_

    output_locs = concatenate_none(
        old_arr=output_locs, new_arr=full_output_locs[:total_size]
    )
    full_output_locs = full_output_locs[total_size:]

    if is_last:
        output_locs = concatenate_none(old_arr=output_locs, new_arr=full_output_locs)
        for idx, batch_output_ in enumerate(batch_output):
            full_batch_output[idx] = concatenate_none(
                old_arr=full_batch_output[idx],
                new_arr=np.zeros(
                    shape=(len(full_output_locs), *batch_output_.shape[1:]),
                    dtype=np.uint8,
                ),
            )

    return full_batch_output, full_output_locs, output_locs


def merge_multitask_horizontal(
    canvas: list[None] | list[da.Array],
    count: list[None] | list[da.Array],
    output_locs_y_: np.ndarray,
    canvas_np: list[np.ndarray],
    output_locs: np.ndarray,
    change_indices: np.ndarray | list[int],
) -> tuple[list[da.Array], list[da.Array], list[np.ndarray], np.ndarray, np.ndarray]:
    """Merge horizontal patches incrementally for each row of patches."""
    start_idx = 0
    for c_idx in change_indices:
        output_locs_ = output_locs[: c_idx - start_idx]

        batch_xs = np.min(output_locs[:, 0], axis=0)
        batch_xe = np.max(output_locs[:, 2], axis=0)

        for idx, canvas_np_ in enumerate(canvas_np):
            canvas_np__ = canvas_np_[: c_idx - start_idx]
            merged_shape = (
                canvas_np__.shape[1],
                batch_xe - batch_xs,
                canvas_np__.shape[3],
            )
            canvas_merge, count_merge = merge_batch_to_canvas(
                blocks=canvas_np__,
                output_locations=output_locs_,
                merged_shape=merged_shape,
            )
            canvas_merge = da.from_array(canvas_merge, chunks=canvas_merge.shape)
            count_merge = da.from_array(count_merge, chunks=count_merge.shape)
            canvas[idx] = concatenate_none(old_arr=canvas[idx], new_arr=canvas_merge)
            count[idx] = concatenate_none(old_arr=count[idx], new_arr=count_merge)
            canvas_np[idx] = canvas_np[idx][c_idx - start_idx :]

        output_locs_y_ = concatenate_none(
            old_arr=output_locs_y_, new_arr=output_locs[:, (1, 3)]
        )

        output_locs = output_locs[c_idx - start_idx :]
        start_idx = c_idx

    return canvas, count, canvas_np, output_locs, output_locs_y_


def save_multitask_to_cache(
    canvas: list[da.Array],
    count: list[da.Array],
    canvas_zarr: list[zarr.Array | None],
    count_zarr: list[zarr.Array | None],
    save_path: str | Path = "temp.zarr",
) -> tuple[list[zarr.Array], list[zarr.Array]]:
    """Save computed canvas and count list of arrays to Zarr cache."""
    zarr_group = None
    for idx, canvas_ in enumerate(canvas):
        computed_values = compute(*[canvas_, count[idx]])
        canvas_computed, count_computed = computed_values

        chunk_shape = tuple(chunk[0] for chunk in canvas_.chunks)
        if canvas_zarr[idx] is None:
            # Only open zarr for first canvas.
            zarr_group = zarr.open(str(save_path), mode="w") if idx == 0 else zarr_group

            canvas_zarr[idx] = zarr_group.create_dataset(
                name=f"canvas/{idx}",
                shape=(0, *canvas_computed.shape[1:]),
                chunks=(chunk_shape[0], *canvas_computed.shape[1:]),
                dtype=canvas_computed.dtype,
                overwrite=True,
            )

            count_zarr[idx] = zarr_group.create_dataset(
                name=f"count/{idx}",
                shape=(0, *count_computed.shape[1:]),
                dtype=count_computed.dtype,
                chunks=(chunk_shape[0], *count_computed.shape[1:]),
                overwrite=True,
            )

        canvas_zarr[idx].resize(
            (
                canvas_zarr[idx].shape[0] + canvas_computed.shape[0],
                *canvas_zarr[idx].shape[1:],
            )
        )
        canvas_zarr[idx][-canvas_computed.shape[0] :] = canvas_computed

        count_zarr[idx].resize(
            (
                count_zarr[idx].shape[0] + count_computed.shape[0],
                *count_zarr[idx].shape[1:],
            )
        )
        count_zarr[idx][-count_computed.shape[0] :] = count_computed

    return canvas_zarr, count_zarr


def merge_multitask_vertical_chunkwise(
    canvas: list[da.Array],
    count: list[da.Array],
    output_locs_y_: np.ndarray,
    zarr_group: zarr.Group,
    save_path: Path,
    memory_threshold: int = 80,
) -> list[da.Array]:
    """Merge vertically chunked arrays into a single probability map."""
    y0s, y1s = np.unique(output_locs_y_[:, 0]), np.unique(output_locs_y_[:, 1])
    overlaps = np.append(y1s[:-1] - y0s[1:], 0)

    probabilities_zarr = [None for _ in range(len(canvas))]
    probabilities_da = [None for _ in range(len(canvas))]

    for idx, canvas_ in enumerate(canvas):
        num_chunks = canvas_.numblocks[0]
        chunk_shape = tuple(chunk[0] for chunk in canvas_.chunks)

        tqdm_ = get_tqdm()
        tqdm_loop = tqdm_(overlaps, leave=False, desc="Merging rows")

        curr_chunk = canvas_.blocks[0, 0].compute()
        curr_count = count[idx].blocks[0, 0].compute()
        next_chunk = canvas_.blocks[1, 0].compute() if num_chunks > 1 else None
        next_count = count[idx].blocks[1, 0].compute() if num_chunks > 1 else None

        for i, overlap in enumerate(tqdm_loop):
            if next_chunk is not None and overlap > 0:
                curr_chunk[-overlap:] += next_chunk[:overlap]
                curr_count[-overlap:] += next_count[:overlap]

            # Normalize
            curr_count = np.where(curr_count == 0, 1, curr_count)
            probabilities = curr_chunk / curr_count.astype(np.float32)

            probabilities_zarr[idx], probabilities_da[idx] = store_probabilities(
                probabilities=probabilities,
                chunk_shape=chunk_shape,
                probabilities_zarr=probabilities_zarr[idx],
                probabilities_da=probabilities_da[idx],
                zarr_group=zarr_group,
                name=f"probabilities/{idx}",
            )

            probabilities_zarr, probabilities_da = _save_multitask_vertical_to_cache(
                probabilities_zarr=probabilities_zarr,
                probabilities_da=probabilities_da,
                probabilities=probabilities,
                idx=idx,
                tqdm_=tqdm_,
                save_path=save_path,
                chunk_shape=chunk_shape,
                memory_threshold=memory_threshold,
            )

            if next_chunk is not None:
                curr_chunk, curr_count = next_chunk[overlap:], next_count[overlap:]

            if i + 2 < num_chunks:
                next_chunk = canvas_.blocks[i + 2, 0].compute()
                next_count = count[idx].blocks[i + 2, 0].compute()
            else:
                next_chunk, next_count = None, None

        probabilities_da[idx] = _clear_zarr(
            probabilities_zarr=probabilities_zarr[idx],
            probabilities_da=probabilities_da[idx],
            zarr_group=zarr_group,
            idx=idx,
            chunk_shape=chunk_shape,
            probabilities_shape=curr_chunk.shape[1:],
        )

    return probabilities_da


def _save_multitask_vertical_to_cache(
    probabilities_zarr: list[zarr.Array] | list[None],
    probabilities_da: list[da.Array] | list[None],
    probabilities: np.ndarray,
    idx: int,
    tqdm_: type[tqdm_notebook | tqdm],
    save_path: Path,
    chunk_shape: tuple,
    memory_threshold: int = 80,
) -> tuple[list[zarr.Array], list[da.Array] | None]:
    """Helper function to save to zarr if vertical merge is out of memory."""
    used_percent = 0
    if probabilities_da[idx] is not None:
        vm = psutil.virtual_memory()
        # Calculate total bytes for all outputs
        total_bytes = sum(0 if arr is None else arr.nbytes for arr in probabilities_da)
        used_percent = (total_bytes / vm.free) * 100
    if probabilities_zarr[idx] is None and used_percent > memory_threshold:
        msg = (
            f"Current Memory usage: {used_percent} %  "
            f"exceeds specified threshold: {memory_threshold}. "
            f"Saving intermediate results to disk."
        )
        tqdm_.write(msg)
        zarr_group = zarr.open(str(save_path), mode="a")
        probabilities_zarr[idx] = zarr_group.create_dataset(
            name=f"probabilities/{idx}",
            shape=probabilities_da[idx].shape,
            chunks=(chunk_shape[0], *probabilities.shape[1:]),
            dtype=probabilities.dtype,
            overwrite=True,
        )
        probabilities_zarr[idx][:] = probabilities_da[idx].compute()

        probabilities_da[idx] = None

    return probabilities_zarr, probabilities_da


def _clear_zarr(
    probabilities_zarr: zarr.Array | None,
    probabilities_da: da.Array | None,
    zarr_group: zarr.Group,
    idx: int,
    chunk_shape: tuple,
    probabilities_shape: tuple,
) -> da.Array | None:
    """Helper function to clear all zarr contents and return dask array."""
    if probabilities_zarr:
        if "canvas" in zarr_group:
            del zarr_group["canvas"][idx]
        if "count" in zarr_group:
            del zarr_group["count"][idx]
        return da.from_zarr(
            probabilities_zarr, chunks=(chunk_shape[0], *probabilities_shape)
        )
    return probabilities_da


def _calculate_probabilities(
    canvas_zarr: list[zarr.Array] | list[None],
    count_zarr: list[zarr.Array] | list[None],
    canvas: list[da.Array | None],
    count: list[da.Array | None],
    output_locs_y_: np.ndarray,
    save_path: Path,
    memory_threshold: int,
) -> list[da.Array]:
    """Helper function to calculate probabilities for MultiTaskSegmentor."""
    zarr_group = None
    if canvas_zarr[0] is not None:
        canvas_zarr, count_zarr = save_multitask_to_cache(
            canvas, count, canvas_zarr, count_zarr
        )
        # Wrap zarr in dask array
        for idx, canvas_zarr_ in enumerate(canvas_zarr):
            canvas[idx] = da.from_zarr(canvas_zarr_, chunks=canvas_zarr_.chunks)
            count[idx] = da.from_zarr(count_zarr[idx], chunks=count_zarr[idx].chunks)

        zarr_group = zarr.open(canvas_zarr[0].store.path, mode="a")

    # Final vertical merge
    return merge_multitask_vertical_chunkwise(
        canvas,
        count,
        output_locs_y_,
        zarr_group,
        save_path,
        memory_threshold,
    )


def _check_and_update_for_memory_overload(
    canvas: list[da.Array | None],
    count: list[da.Array | None],
    canvas_zarr: list[zarr.Array | None],
    count_zarr: list[zarr.Array | None],
    memory_threshold: int,
    tqdm_loop: DataLoader | tqdm,
    tqdm_: type[tqdm_notebook | tqdm],
    save_path: Path,
    num_expected_output: int,
) -> tuple[
    list[da.Array | None],
    list[da.Array | None],
    list[zarr.Array | None],
    list[zarr.Array | None],
    DataLoader | tqdm,
]:
    """Helper function to check and update the memory usage for multitask segmentor."""
    vm = psutil.virtual_memory()
    used_percent = vm.percent
    total_bytes = sum(arr.nbytes for arr in canvas) if canvas else 0
    canvas_used_percent = (total_bytes / vm.free) * 100

    if not (used_percent > memory_threshold or canvas_used_percent > memory_threshold):
        return canvas, count, canvas_zarr, count_zarr, tqdm_loop

    tqdm_loop.desc = "Spill intermediate data to disk"
    used_percent = (
        canvas_used_percent
        if (canvas_used_percent > memory_threshold)
        else used_percent
    )
    msg = (
        f"Current Memory usage: {used_percent} %  "
        f"exceeds specified threshold: {memory_threshold}. "
        f"Saving intermediate results to disk."
    )
    tqdm_.write(msg)
    # Flush data in Memory and clear dask graph
    canvas_zarr, count_zarr = save_multitask_to_cache(
        canvas,
        count,
        canvas_zarr,
        count_zarr,
        save_path=save_path,
    )
    canvas = [None for _ in range(num_expected_output)]
    count = [None for _ in range(num_expected_output)]
    gc.collect()
    tqdm_loop.desc = "Inferring patches"

    return canvas, count, canvas_zarr, count_zarr, tqdm_loop


def _save_annotation_store(
    curr_image: Path | None,
    keys_to_compute: list[str],
    values: list[da.Array | list[da.Array]],
    task_name: str,
    idx: int,
    save_path: Path,
    class_dict: dict,
    scale_factor: tuple[float, float],
) -> Path:
    """Helper function to save to annotation store."""
    if isinstance(curr_image, Path):
        store_file_name = (
            f"{curr_image.stem}.db"
            if task_name is None
            else f"{curr_image.stem}_{task_name}.db"
        )
    else:
        store_file_name = f"{idx}.db" if task_name is None else f"{idx}_{task_name}.db"
    predictions_ = dict(zip(keys_to_compute, values, strict=False))
    output_path = save_path.parent / store_file_name
    # Patch mode indexes the "coordinates" while calculating "values" variable.
    origin = (
        predictions_.pop("coordinates")[0][:2]
        if len(predictions_["coordinates"].shape) > 1
        else predictions_.pop("coordinates")[:2]
    )
    origin = tuple(max(0.0, float(x)) for x in origin)
    store = SQLiteStore()
    store = dict_to_store(
        store=store,
        processed_predictions=predictions_,
        class_dict=class_dict,
        scale_factor=scale_factor,
        origin=origin,
    )

    store.commit()
    store.dump(output_path)

    return output_path


def _process_instance_predictions(
    inst_dict: dict,
    ioconfig: IOSegmentorConfig,
    tile_shape: list,
    tile_flag: list,
    tile_mode: int,
    tile_tl: tuple,
    ref_inst_dict: dict,
) -> list | tuple:
    """Function to merge new tile prediction with existing prediction.

    Args:
        inst_dict (dict): Dictionary containing instance information.
        ioconfig (:class:`IOSegmentorConfig`): Object defines information
            about input and output placement of patches.
        tile_shape (list): A list of the tile shape.
        tile_flag (list): A list of flag to indicate if instances within
            an area extended from each side (by `ioconfig.margin`) of
            the tile should be replaced by those within the same spatial
            region in the accumulated output this run. The format is
            [top, bottom, left, right], 1 indicates removal while 0 is not.
            For example, [1, 1, 0, 0] denotes replacing top and bottom instances
            within `ref_inst_dict` with new ones after this processing.
        tile_mode (int): A flag to indicate the type of this tile. There
            are 4 flags:
            - 0: A tile from tile grid without any overlapping, it is not
                an overlapping tile from tile generation. The predicted
                instances are immediately added to accumulated output.
            - 1: Vertical tile strip that stands between two normal tiles
                (flag 0). It has the same height as normal tile but
                less width (hence vertical strip).
            - 2: Horizontal tile strip that stands between two normal tiles
                (flag 0). It has the same width as normal tile but
                less height (hence horizontal strip).
            - 3: tile strip stands at the cross-section of four normal tiles
                (flag 0).
        tile_tl (tuple): Top left coordinates of the current tile.
        ref_inst_dict (dict): Dictionary contains accumulated output. The
            expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.

    Returns:
        new_inst_dict (dict): A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list): List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.

    """
    # should be rare, no nuclei detected in input images
    if len(inst_dict) == 0:
        return {}, []

    # !
    m = ioconfig.margin
    w, h = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    tile_rtree = STRtree(geometries)
    # !

    # create margin bounding box, ordering should match with
    # created tile info flag (top, bottom, left, right)
    boundary_lines = [
        shapely_box(0, 0, w, 1),  # top egde
        shapely_box(0, h - 1, w, h),  # bottom edge
        shapely_box(0, 0, 1, h),  # left
        shapely_box(w - 1, 0, w, h),  # right
    ]
    margin_boxes = [
        shapely_box(0, 0, w, m),  # top egde
        shapely_box(0, h - m, w, h),  # bottom edge
        shapely_box(0, 0, m, h),  # left
        shapely_box(w - m, 0, w, h),  # right
    ]
    # ! this is wrt to WSI coord space, not tile
    margin_lines = [
        [[m, m], [w - m, m]],  # top egde
        [[m, h - m], [w - m, h - m]],  # bottom edge
        [[m, m], [m, h - m]],  # left
        [[w - m, m], [w - m, h - m]],  # right
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    margin_lines = [shapely_box(*v.flatten().tolist()) for v in margin_lines]

    # the ids within this match with those within `inst_map`, not UUID
    sel_indices = []
    if tile_mode in [0, 3]:
        # for `full grid` tiles `cross section` tiles
        # -- extend from the boundary by the margin size, remove
        #    nuclei whose entire contours lie within the margin area
        sel_boxes = [
            box
            for idx, box in enumerate(margin_boxes)
            if tile_flag[idx] or tile_mode == 3  # noqa: PLR2004
        ]

        sel_indices = [
            geo
            for bounds in sel_boxes
            for geo in tile_rtree.query(bounds)
            if bounds.contains(geometries[geo])
        ]
    elif tile_mode in [1, 2]:
        # for `horizontal/vertical strip` tiles
        # -- extend from the marked edges (top/bot or left/right) by
        #    the margin size, remove all nuclei lie within the margin
        #    area (including on the margin line)
        # -- remove all nuclei on the boundary also

        sel_boxes = [
            margin_boxes[idx] if flag else boundary_lines[idx]
            for idx, flag in enumerate(tile_flag)
        ]

        sel_indices = [geo for bounds in sel_boxes for geo in tile_rtree.query(bounds)]
    else:
        msg = f"Unknown tile mode {tile_mode}."
        raise ValueError(msg)

    def retrieve_sel_uids(sel_indices: list, inst_dict: dict) -> list:
        """Helper to retrieved selected instance uids."""
        if len(sel_indices) > 0:
            # not sure how costly this is in large dict
            inst_uids = list(inst_dict.keys())
        return [inst_uids[idx] for idx in sel_indices]

    remove_insts_in_tile = retrieve_sel_uids(sel_indices, inst_dict)

    # external removal only for tile at cross-sections
    # this one should contain UUID with the reference database
    remove_insts_in_orig = []
    if tile_mode == 3:  # noqa: PLR2004
        inst_boxes = [v["box"] for v in ref_inst_dict.values()]
        inst_boxes = np.array(inst_boxes)

        geometries = [shapely_box(*bounds) for bounds in inst_boxes]
        ref_inst_rtree = STRtree(geometries)
        sel_indices = [
            geo for bounds in margin_lines for geo in ref_inst_rtree.query(bounds)
        ]

        remove_insts_in_orig = retrieve_sel_uids(sel_indices, ref_inst_dict)

    # move inst position from tile space back to WSI space
    # an also generate universal uid as replacement for storage
    new_inst_dict = {}
    for inst_uid, inst_info in inst_dict.items():
        if inst_uid not in remove_insts_in_tile:
            inst_info["box"] += np.concatenate([tile_tl] * 2)
            if "centroid" in inst_info:
                inst_info["centroid"] += tile_tl
            inst_info["contours"] += tile_tl
            inst_uuid = uuid.uuid4().hex
            new_inst_dict[inst_uuid] = inst_info

    return new_inst_dict, remove_insts_in_orig


def _get_inst_info_dicts(post_process_output: tuple[dict]) -> list:
    """Helper to convert post processing output to dictionary list.

    This function makes the info_dict compatible with tile based processing of
    info_dictionaries from HoVerNet.

    """
    inst_dicts = []
    for _output in post_process_output:
        keys_ = list(_output["info_dict"].keys())

        inst_dicts.extend(
            [
                {
                    i + 1: {
                        key: values[i] for key, values in _output["info_dict"].items()
                    }
                    for i in range(len(_output["info_dict"][keys_[0]]))
                }
            ]
        )

    return inst_dicts


def _create_wsi_info_dict(
    post_process_output: tuple[dict],
    wsi_info_dict: tuple[dict] | None,
    wsi_proc_shape: tuple[int, ...],
    save_path: Path,
    return_predictions: tuple[bool, ...] | None,
    memory_threshold: float = 80,
) -> tuple[dict[str, dict[Any, Any] | list[Any] | Any], ...]:
    """Create or reuse WSI info dictionaries for post-processed outputs.

    This function constructs a tuple of WSI information dictionaries, one for each
    element in `post_process_output`. If an existing `wsi_info_dict` is provided,
    it is returned unchanged. Otherwise, a new dictionary is created for each item,
    containing task metadata, an allocated prediction array (NumPy or Zarr, chosen
    based on available memory), and an empty `info_dict` for downstream metadata.

    Args:
        post_process_output (tuple[dict]):
            A tuple of dictionaries produced by the post-processing step. Each
            dictionary must contain at least:
            - "task_type": str
            - "predictions": array-like with a `.dtype` and `.shape` attribute
        wsi_info_dict (tuple[dict] | None):
            Existing WSI info dictionaries. If provided, they are returned as-is.
        wsi_proc_shape (tuple[int, ...]):
            The full shape of the WSI-level prediction array to allocate for each
            output item.
        save_path (Path):
            Filesystem path where Zarr arrays will be stored if disk-backed
            allocation is required.
        return_predictions (tuple[bool, ...]):
            Whether to return predictions for individual tasks. Default is None,
            which returns no predictions.
        memory_threshold (float, optional):
            Fraction of available RAM allowed for in-memory allocation. Must be
            between 0.0 and 100. Defaults to 80.

    Returns:
        tuple[dict[str, dict[Any, Any] | list[Any] | Any], ...]:
            A tuple of dictionaries, one per post-processing output. Each dictionary
            contains:
            - "task_type": str
            - "predictions": allocated NumPy or Zarr array.
            - "info_dict": an empty dictionary for additional metadata.

    """
    if wsi_info_dict is not None:
        return wsi_info_dict

    # Convert to tuple for each task
    if return_predictions is None:
        return_predictions = [False for _ in post_process_output]

    return tuple(
        {
            "task_type": post_process_output_["task_type"],
            "predictions": None
            if not return_predictions[idx]
            else create_smart_array(
                shape=wsi_proc_shape,
                dtype=post_process_output_["predictions"].dtype,
                memory_threshold=memory_threshold,
                zarr_path=save_path,
                chunks=post_process_output_["predictions"].shape,
            ),
            "info_dict": {},
        }
        for idx, post_process_output_ in enumerate(post_process_output)
    )


def _update_tile_based_predictions_array(
    post_process_output: tuple[dict],
    wsi_info_dict: tuple[dict],
    bounds: tuple[int, int, int, int],
) -> tuple[dict]:
    """Helper function to update tile based predictions array."""
    x_start, y_start, x_end, y_end = bounds

    for idx, post_process_output_ in enumerate(post_process_output):
        if wsi_info_dict[idx]["predictions"] is None:
            continue
        max_h, max_w = wsi_info_dict[idx]["predictions"].shape
        x_end, y_end = min(x_end, max_w), min(y_end, max_h)
        wsi_info_dict[idx]["predictions"][y_start:y_end, x_start:x_end] = (
            post_process_output_["predictions"][
                0 : y_end - y_start, 0 : x_end - x_start
            ]
        )

    return wsi_info_dict
