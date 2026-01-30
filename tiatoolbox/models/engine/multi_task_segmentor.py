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
        return_predictions (tuple(bool, ...):
            Whether to return array predictions for individual tasks.
        return_probabilities (bool):
            Whether to return per-class probabilities.
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
    """MultiTask segmentation engine to run models like hovernet and hovernetplus.

    MultiTaskSegmentor performs segmentation across multiple model heads
    (e.g., semantic, instance, edge). It abstracts model invocation,
    preprocessing, and output postprocessing for multi-head segmentation.

    Args:
        model (str | ModelABC):
            A PyTorch model instance or name of a pretrained model from TIAToolbox.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter. Default is `None`.
        batch_size (int):
            Number of image patches processed per forward pass. Default is 8.
        num_workers (int):
            Number of workers for data loading. Default is 0.
        weights (str | Path | None):
            Path to model weights. If None, default weights are used.

            >>> engine = SemanticSegmentor(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
        verbose (bool):
            Whether to enable verbose logging. Default is True.

    Attributes:
        images (list[str | Path] | np.ndarray):
            Input image patches or WSI paths.
        masks (list[str | Path] | np.ndarray):
            Optional tissue masks for WSI processing.
            These are only utilized when patch_mode is False.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (bool):
            Whether input is treated as patches (`True`) or WSIs (`False`).
        model (ModelABC):
            Loaded PyTorch model.
        ioconfig (ModelIOConfigABC):
            IO configuration for patch extraction and resolution.
        return_labels (bool):
            Whether to include labels in the output.
        input_resolutions (list[dict]):
            Resolution settings for model input. Supported
            units are `level`, `power` and `mpp`. Keys should be "units" and
            "resolution" e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
            :class:`WSIReader` for details.
        patch_input_shape (tuple[int, int]):
            Shape of input patches (height, width). Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (tuple[int, int]):
            Stride used during patch extraction. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        labels (list | None):
            Optional labels for input images.
            Only a single label per image is supported.
        drop_keys (list):
            Keys to exclude from model output.
        output_type (str):
            Format of output ("dict", "zarr", "annotationstore").
        output_locations (list | None):
            Coordinates of output patches used during WSI processing.

    Examples:
    >>> # list of 2 image patches as input
    >>> wsis = ['path/img.svs', 'path/img.svs']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed")
    >>> output = mtsegmentor.run(wsis, patch_mode=False)

    >>> # array of list of 2 image patches as input
    >>> image_patches = [np.ndarray, np.ndarray]
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed")
    >>> output = mtsegmentor.run(image_patches, patch_mode=True)

    >>> # list of 2 image patch files as input
    >>> data = ['path/img.png', 'path/img.png']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(data, patch_mode=False)

    >>> # list of 2 image tile files as input
    >>> tile_file = ['path/tile1.png', 'path/tile2.png']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(tile_file, patch_mode=False)

    >>> # list of 2 wsi files as input
    >>> wsis = ['path/wsi1.svs', 'path/wsi2.svs']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(wsis, patch_mode=False)


    """

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
        """Initialize :class:`MultiTaskSegmentor`.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or name of a pretrained model from TIAToolbox.
                If a string is provided, the corresponding pretrained weights will be
                downloaded unless overridden via `weights`.
            batch_size (int):
                Number of image patches processed per forward pass. Default is 8.
            num_workers (int):
                Number of workers for data loading. Default is 0.
            weights (str | Path | None):
                Path to model weights. If None, default weights are used.
            device (str):
                Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Whether to enable verbose logging. Default is True.

        """
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
        """Run inference on a batch of image patches using the multitask model.

        This method processes patches provided by a PyTorch ``DataLoader`` and runs
        them through the model's ``infer_batch`` method. Models with multiple heads
        (e.g., semantic, instance, edge) may return multiple outputs per patch.
        Outputs are collected as Dask arrays for efficient large-scale aggregation.

        Args:
            dataloader (DataLoader):
                A PyTorch dataloader that yields dicts containing ``"image"`` tensors
                and optionally other metadata (e.g., coordinates).
            return_coordinates (bool):
                Whether to return the spatial coordinates associated with each patch
                (when available from the dataset). Default is False.

        Returns:
            dict[str, list[da.Array]]:
                A dictionary containing the model outputs for all patches.

                Keys:
                    probabilities (list[da.Array]):
                        A list of Dask arrays containing model outputs for each head.
                        Each array has shape ``(N, C, H, W)`` depending on the model.
                    coordinates (da.Array):
                        Returned only when ``return_coordinates=True``.
                        A Dask array of shape ``(N, 2)`` or ``(N, 4)`` depending on
                        how patch coordinates are stored in the dataset.

        Notes:
            - The number of model outputs (heads) is inferred dynamically from the
              first forward pass.
            - Outputs are stacked via ``dask.array.concatenate`` for scalability.
            - This method does not perform postprocessing; raw logits/probabilities
              are returned exactly as produced by the model.

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
        """Perform model inference on a whole slide image (WSI).

        This method iterates over WSI patches produced by a DataLoader,
        runs each patch through the model's ``infer_batch`` callback, and
        incrementally assembles full-resolution model outputs for each model
        head (e.g., semantic, instance, edge). Patch-level outputs are merged
        row-by-row using horizontal stitching, optionally spilling intermediate
        results to disk when memory usage exceeds a threshold. After all rows
        are processed, vertical merging is performed to generate the final
        probability maps for each multitask head.

        Raw probabilities and patch coordinates are returned as Dask arrays.
        This method does not perform any post-processing; downstream calls to
        ``post_process_wsi`` are required to convert model logits into
        task-specific outputs (e.g., instances, contours, or label maps).

        Args:
            dataloader (DataLoader):
                A PyTorch dataloader yielding dictionaries with keys such as
                ``"image"`` and ``"output_locs"`` that correspond to extracted
                WSI patches and their placement metadata.
            save_path (Path):
                A filesystem path used to store temporary Zarr cache data when
                memory spilling is triggered. The directory is created if needed.
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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
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
                        Whether to return labels with predictions.
                    return_predictions (tuple(bool, ...):
                        Whether to return array predictions for individual tasks.
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
            dict[str, da.Array]:
                A dictionary containing the raw multitask model outputs.

                Keys:
                    probabilities (list[da.Array]):
                        One Dask array per model head, each representing the final
                        WSI-sized probability map for that task. Each array has
                        shape ``(H, W, C)`` depending on the head's channel count.
                    coordinates (da.Array):
                        A Dask array of shape ``(N, 2)`` or ``(N, 4)``, containing
                        accumulated patch coordinate metadata produced during the
                        WSI dataloader iteration.

        Notes:
            - The number of model heads is inferred from the first
              ``infer_batch`` call.
            - Patch predictions are merged horizontally when the x-coordinate
              changes row, and vertically after all rows are processed.
            - Large WSIs may trigger spilling intermediate canvas data to disk
              when memory exceeds ``memory_threshold``.
            - This function returns *raw probabilities only*. For task-specific
              segmentation or instance extraction, call ``post_process_wsi``.

        """
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
        """Post-process raw patch-level predictions for multitask segmentation.

        This method applies the model's ``postproc_func`` to per-patch probability
        maps produced by ``infer_patches``. For multitask models (multiple heads),
        it zips the per-head probability arrays across patches and invokes
        ``postproc_func`` to obtain one or more task dictionaries per patch (e.g.,
        semantic labels, instance info, edges). The per-patch outputs are then
        reorganized into a task-centric structure using
        ``build_post_process_raw_predictions`` for downstream saving.

        Args:
            raw_predictions (dict):
                Dictionary containing raw model outputs from ``infer_patches``.
                Expected keys:
                    - ``"probabilities"`` (list[da.Array]):
                      One Dask array per model head. Each array typically has shape
                      ``(N, H, W, C)`` for ``N`` patches, with head-specific channels.
                      These are *raw* logits/probabilities and are not normalized
                      beyond what the model provides.
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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
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
                        Whether to return labels with predictions.
                    return_predictions (tuple(bool, ...):
                        Whether to return array predictions for individual tasks.
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
            dict:
                A task-organized dictionary suitable for saving, where each entry
                corresponds to a task produced by ``postproc_func``. For each task
                (e.g., ``"semantic"``, ``"instance"``), keys and value types depend
                on the model's post-processing output. Typical patterns include:
                    - ``"predictions"``: list[da.Array] with per-patch outputs,
                      if the model returns patch-level prediction arrays.
                    - ``"info_dict"``: list[dict] with per-patch metadata dictionaries
                      (e.g., instance tables, properties). Lists are aligned to the
                      number of input patches.
                Any pre-existing keys in ``raw_predictions`` (e.g., ``"coordinates"``)
                are preserved as returned by ``build_post_process_raw_predictions``.

        Notes:
            - This method is *patch-level* post-processing only; it does not perform
              WSI-scale tiling or stitching. For WSI outputs, use ``post_process_wsi``.
            - Inputs are typically Dask arrays; computation remains lazy until an
              explicit save step or ``dask.compute`` is invoked downstream.
            - The exact set of task keys and payload shapes are determined by the
              model's ``postproc_func`` for each head.

        """
        probabilities = raw_predictions["probabilities"]
        post_process_predictions = [
            self.model.postproc_func(list(probs_for_idx))
            for probs_for_idx in zip(*probabilities, strict=False)
        ]

        return self.build_post_process_raw_predictions(
            post_process_predictions=post_process_predictions,
            raw_predictions=raw_predictions,
        )

    def post_process_wsi(  # skipcq: PYL-R0201
        self: MultiTaskSegmentor,
        raw_predictions: dict,
        save_path: Path,
        **kwargs: Unpack[MultiTaskSegmentorRunParams],
    ) -> dict:
        """Post-process whole slide image (WSI) predictions for multitask segmentation.

        This method converts raw WSI-scale probability maps (produced by
        ``infer_wsi``) into task-specific outputs using the model's
        ``postproc_func``. If the probability maps are fully in memory, the method
        processes the entire WSI at once. If they are Zarr-backed (spilled during
        inference) or too large, it switches to tile mode: it iterates over WSI
        tiles, applies ``postproc_func`` per tile, merges instance predictions
        across tile boundaries, and optionally writes intermediate arrays to Zarr
        under ``save_path.with_suffix(".zarr")`` for memory efficiency.

        The result is organized into a task-centric dictionary (e.g., semantic,
        instance) with arrays and/or metadata suitable for saving or further use.

        Args:
            raw_predictions (dict):
                Dictionary containing WSI-scale model outputs from ``infer_wsi``.
                Expected key:
                    - ``"probabilities"`` (tuple[da.Array]):
                      One Dask array per model head. Each array is either
                      memory-backed (Dask→NumPy) or Zarr-backed depending on
                      memory spilling during inference.
            save_path (Path):
                Base path for writing intermediate Zarr arrays in tile mode and
                for allocating per-task outputs when disk-backed arrays are needed.
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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
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
                        Whether to return labels with predictions.
                    return_predictions (tuple(bool, ...):
                        Whether to return array predictions for individual tasks.
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
            dict:
                A task-organized dictionary of WSI-scale outputs. For each task
                (e.g., ``"semantic"``, ``"instance"``), typical entries include:
                    - ``"predictions"`` (da.Array or np.ndarray, optional):
                      Full-resolution task prediction map, present only where
                      enabled by ``return_predictions``.
                    - Additional task-specific keys (e.g., ``"info_dict"``,
                      per-instance dictionaries, contours, classes, probabilities).
                The set of keys and their exact shapes/types are determined by the
                model's ``postproc_func``.

        Notes:
            - Full-WSI mode is selected when probability maps are not Zarr-backed;
              otherwise tile mode is used.
            - Tile mode uses model-specific merging of instances across tile
              boundaries and may write intermediate arrays under a ``.zarr`` group
              next to ``save_path``.
            - Probability maps themselves are not modified here; this method produces
              task-centric outputs from them. Use ``save_predictions`` to persist
              results as ``dict``, ``zarr``, or ``annotationstore``.

        """
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
        """Convert full-WSI probability maps into task-specific outputs in memory.

        This helper is used when the WSI-scale probability maps (one per model head)
        fit in memory without requiring Zarr-backed tiling. It invokes the model's
        ``postproc_func`` once on the complete list of head maps and returns a list
        of per-task dictionaries (e.g., semantic, instance). Optionally, it drops
        the ``"predictions"`` array for tasks where returning the full-resolution
        map is not requested.

        Args:
            probabilities (list[da.Array | np.ndarray]):
                Full-resolution probability maps, one per model head. Each element
                is either a Dask array or NumPy array with shape ``(H, W, C)``,
                where ``C`` is head-specific. These are the outputs of
                ``infer_wsi`` after horizontal/vertical stitching.
            return_predictions (tuple[bool, ...] | None):
                Per-task flags indicating whether to keep the task's
                full-resolution ``"predictions"`` array in the result. If
                ``None``, no task predictions are returned (all ``"predictions"``
                keys are removed). The tuple length must match the number of
                task dictionaries returned by ``postproc_func``.

        Returns:
            list[dict] | None:
                A list of task dictionaries returned by the model's
                ``postproc_func``. Each dictionary must include
                ``"task_type"`` and may include keys such as
                ``"predictions"`` (``np.ndarray`` or ``da.Array``) and/or an
                ``"info_dict"`` with task-specific metadata. If all task
                predictions are dropped and no other outputs are produced,
                this may return ``None``.

        Notes:
            - This function performs no tiling or disk spilling; it assumes the
              inputs fit in memory. For large WSIs or Zarr-backed probability
              maps, use ``_process_tile_mode`` instead.
            - The exact set of task keys and value types is model-dependent and
              determined by ``postproc_func``.
            - When ``return_predictions`` is provided, it is applied positionally
              to the sequence of task dictionaries emitted by ``postproc_func``:
              if a task's flag is ``False``, that task's ``"predictions"`` key is
              removed from the output.

        """
        post_process_predictions = self.model.postproc_func(probabilities)
        if return_predictions is None:
            return_predictions = [False for _ in post_process_predictions]
        for idx, return_predictions_ in enumerate(return_predictions):
            if not return_predictions_:
                del post_process_predictions[idx]["predictions"]

        return post_process_predictions

    def _process_tile_mode(
        self: MultiTaskSegmentor,
        probabilities: list[da.Array | np.ndarray],
        save_path: Path,
        memory_threshold: float = 80,
        *,
        return_predictions: tuple[bool, ...] | None = None,
    ) -> list[dict] | None:
        """Convert WSI probability maps into outputs using tile-mode processing.

        This helper is used when WSI-scale probability maps are Zarr-backed or too
        large to fit comfortably in memory. It iterates over WSI tiles, extracts the
        corresponding sub-arrays from each model head, applies the model's
        ``postproc_func`` per tile, and merges task outputs across tile boundaries.
        For instance-type tasks, it removes duplicated/cut instances near tile
        margins using configuration from ``IOSegmentorConfig`` (tile flags, margin)
        and consolidates detections into the slide coordinate system.

        Optionally, full-resolution per-task prediction arrays (e.g., dense label or
        probability maps) are allocated as NumPy or Zarr via ``create_smart_array``
        and incrementally filled at the appropriate tile locations. Allocation and
        spilling behavior are governed by ``memory_threshold``.

        Args:
            probabilities (list[da.Array | np.ndarray]):
                WSI-scale probability maps, one per model head, with shape
                ``(H, W, C)`` per head. These are the outputs of ``infer_wsi``
                (after horizontal/vertical stitching) and may be Zarr-backed.
            save_path (Path):
                Base path used for creating a ``.zarr`` group to store
                disk-backed arrays when memory usage exceeds the threshold and for
                per-task predictions when requested by ``return_predictions``.
            memory_threshold (float):
                Maximum allowed RAM usage (percentage) for in-memory arrays before
                switching to or continuing with Zarr-backed allocation. Defaults to 80.
            return_predictions (tuple[bool, ...] | None):
                Per-task flags indicating whether to retain a full-resolution
                ``"predictions"`` array for each task. If ``None``, no task-level
                prediction arrays are retained (i.e., they are set to ``None`` and not
                allocated). The tuple length must match the number of task dictionaries
                produced by ``postproc_func``.

        Returns:
            list[dict] | None:
                A list of task dictionaries (one per multitask head output as produced
                by ``postproc_func``) with fields such as:
                    - ``"task_type"`` (str): Name/type of the task (e.g.,
                      ``"semantic"``, ``"instance"``).
                    - ``"predictions"`` (np.ndarray or Zarr-backed array | None):
                      Full-resolution task prediction array if enabled by
                      ``return_predictions``; otherwise ``None``.
                    - ``"info_dict"`` (dict): Task-specific metadata accumulated across
                      tiles. For instance tasks, this includes merged instance tables
                      (e.g., boxes, centroids, contours) keyed by UUIDs in WSI space.

                Returns ``None`` only if ``postproc_func`` yields no outputs.

        Notes:
            - Tile layout is derived from the engine IO config; each tile's bounds
              are used to slice per-head probability maps and to place results back
              into WSI space.
            - For instance tasks, objects near tile margins are pruned/merged using
              per-tile flags and a configurable margin to avoid duplicates across
              tiles. Instance coordinates (boxes, centroids, contours) are translated
              from tile space to WSI space prior to consolidation.
            - When ``return_predictions`` requests any task array, allocation is done
              via ``create_smart_array`` to choose between NumPy and Zarr based on
              ``memory_threshold``. Arrays are filled tile-by-tile using the tile
              bounds.
            - Computation remains lazy for Dask-backed inputs until explicitly
              computed or saved downstream. Probability maps themselves are not
              modified in this method; it only derives task-centric outputs.

        """
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
        """Merge per-image, per-task outputs into a task-organized prediction structure.

        This function takes a list of outputs where each element corresponds to one
        image and contains one or more task dictionaries returned by the model's
        post-processing step (e.g., semantic, instance). Each task dictionary must
        include a ``"task_type"`` key along with any number of task-specific fields
        (for example, ``"predictions"``, ``"info_dict"``, or additional metadata).
        The function reorganizes this data into ``raw_predictions`` by grouping
        entries under their respective task types and aligning values across images.

        The merging logic is as follows:
          1) For each task (identified by ``"task_type"``), values for keys other than
             ``"task_type"`` are temporarily collected into lists, one entry per image.
          2) After all images are processed, list entries are normalized:

             - If all entries for a key are array-like (``np.ndarray`` or
               ``dask.array.Array``),
               they are stacked along a new leading dimension (image axis).
             - If all entries for a key are dictionaries, their subkeys are expanded
               into separate lists aligned across images (the original composite key
               is removed).
          3) Existing content in ``raw_predictions`` is preserved and extended as
             needed.

        Args:
            post_process_predictions (list[tuple]):
                A list where each element represents a single image. Each element is
                an iterable of task dictionaries. Every task dictionary **must**
                contain:
                    - ``"task_type"`` (str): Name/type of the task
                      (e.g., ``"semantic"``, ``"instance"``, ``"edge"``).
                and **may** contain any number of additional fields, such as:
                    - ``"predictions"``: array-like output for that task
                    - ``"info_dict"``: dictionary of task-specific metadata
                    - Any other task-dependent keys
            raw_predictions (dict):
                Dictionary that will be updated **in-place**. It may already contain
                task entries or unrelated keys (e.g., ``"probabilities"``,
                ``"coordinates"``). New tasks and fields are added as they appear.

        Returns:
            dict:
                The updated ``raw_predictions`` dictionary containing one entry per
                task type. Under each task name, keys hold per-image arrays (stacked
                as Dask/NumPy where applicable) or lists/dicts aligned across images.
                Example structure:
                    {
                      "semantic": {
                        "predictions": da.Array | np.ndarray,  # stacked over images
                        "info_dict": [dict, dict, ...]         # or expanded subkeys
                      },
                      "instance": {
                        "info_dict": [...],                    # per-image metadata
                        "contours": [...], "classes": [...],   # task-dependent keys
                      },
                      "coordinates": da.Array,                 # if previously present
                    }

        Notes:
            - Array stacking occurs only when **all** per-image entries for a key are
              array-like; mixed types remain as lists.
            - Dictionary expansion occurs only when **all** per-image entries for a key
              are dictionaries; subkeys are promoted to top-level keys under the task
              and aligned across images.
            - The set ``self.tasks`` is updated to include all encountered task types.

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

        Depending on ``output_type``, this method either:
          - returns a Python dictionary (``"dict"``),
          - writes a Zarr group to disk and returns the path (``"zarr"``), or
          - writes one or more SQLite-backed AnnotationStore ``.db`` files and
            returns the resulting path(s) (``"annotationstore"``).

        For multitask outputs, this function also:
          - Preserves task separation when saving to Zarr (one group per task).
          - Optionally saves raw probability maps if ``return_probabilities=True``
            (as Zarr only; probabilities cannot be written to AnnotationStore).
          - Merges per-task keys for saving to AnnotationStore, including optional
            coordinates to establish slide origin.

        Args:
            processed_predictions (dict):
                Task-organized dictionary produced by post-processing (e.g. from
                ``post_process_patches`` or ``post_process_wsi``). For multitask
                models this typically includes:
                    - ``"probabilities"`` (optional): list[da.Array] of WSI maps,
                      present if preserved for saving.
                    - Per-task sub-dicts (e.g., ``"semantic"``, ``"instance"``),
                      each containing task-specific arrays/metadata such as
                      ``"predictions"``, ``"info_dict"``, etc.
                    - ``"coordinates"`` (optional): Dask/NumPy array used to set
                      spatial origin when saving vector outputs.
            output_type (str):
                Desired output format. Supported values are:
                ``"dict"``, ``"zarr"``, or ``"annotationstore"`` (case-sensitive).
            save_path (Path | None):
                Base filesystem path for file outputs. Required for
                ``"zarr"`` and ``"annotationstore"``. For Zarr, a
                ``save_path.with_suffix(".zarr")`` group is used. For
                AnnotationStore, ``.db`` files are written (one per image in
                patch mode, one per WSI in WSI mode). Ignored when
                ``output_type="dict"``.
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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
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
                        Whether to return labels with predictions.
                    return_predictions (tuple(bool, ...):
                        Whether to return array predictions for individual tasks.
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
            dict | AnnotationStore | Path | list[Path]:
                - If ``output_type == "dict"``:
                    Returns the (possibly simplified) prediction dictionary.
                    For a single task, the task level is flattened.
                - If ``output_type == "zarr"``:
                    Returns the ``Path`` to the saved ``.zarr`` group.
                - If ``output_type == "annotationstore"``:
                    Returns a list of paths to saved ``.db`` files (patch mode),
                    or a single path / store handle for WSI mode. If probability
                    maps were requested for saving, the Zarr path holding those
                    maps may also be included.

        Raises:
            TypeError:
                If an unsupported ``output_type`` is provided.

        Notes:
            - For ``"dict"`` and ``"zarr"``, saving is delegated to
              ``_save_predictions_as_dict_zarr`` to keep behavior aligned across
              engines.
            - When ``output_type == "annotationstore"``, arrays are first computed
              (via a Zarr/dict pass) to obtain concrete NumPy payloads suitable
              for vector export, after which per-task stores are written using
              ``_save_predictions_as_annotationstore``.
            - If ``return_probabilities=True``, probability maps are written only
              to Zarr, never to AnnotationStore. A guidance message is logged
              describing how to visualize heatmaps (e.g., converting to OME-TIFF).

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
        """Run the `MultiTaskSegmentor` engine on input images.

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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
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
                        Whether to return labels with predictions.
                    return_predictions (tuple(bool, ...):
                        Whether to return array predictions for individual tasks.
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
            >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
            >>> output = mtsegmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = mtsegmentor.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = mtsegmentor.run(wsis, patch_mode=False)
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
    """Write polygonal multitask predictions into an SQLite-backed AnnotationStore.

    Converts a task dictionary (with per-object fields) into `Annotation` records,
    applying coordinate scaling and translation to move predictions into the slide's
    baseline coordinate space. Each geometry is created from the per-object
    `"contours"` entry, validated, and shifted by `origin`. All remaining keys in
    `processed_predictions` are attached as annotation properties; the `"type"` key
    can be mapped via `class_dict`.

    Expected `processed_predictions` structure:
        - "contours": list-like of polygon coordinates per object, where each item
          is shaped like `[[x0, y0], [x1, y1], ..., [xN, yN]]`. These are interpreted
          according to `"geom_type"` (default `"Polygon"`).
        - Optional "geom_type": str (e.g., "Polygon", "MultiPolygon").
          Defaults to "Polygon".
        - Additional per-object fields (e.g., "type", "probability", scores, attributes)
          with list-like values aligned to `contours` length.

    Args:
        store (SQLiteStore):
            Target annotation store that will receive the converted annotations.
        processed_predictions (dict):
            Dictionary containing per-object fields. Must include `"contours"`;
            may include `"geom_type"` and any number of additional fields to be
            written as properties.
        class_dict (dict | None):
            Optional mapping for the `"type"` field. When provided and when
            `"type"` is present in `processed_predictions`, each `"type"` value is
            replaced by `class_dict[type_id]` in the saved annotation properties.
        origin (tuple[float, float]):
            `(x0, y0)` offset to add to the final geometry coordinates (in pixels)
            after scaling. Typically corresponds to the tile/patch origin in WSI
            space.
        scale_factor (tuple[float, float]):
            `(sx, sy)` factors applied to coordinates before translation, used to
            convert from model space to baseline slide resolution (e.g.,
            `model_mpp / slide_mpp`).

    Returns:
        AnnotationStore:
            The input `store` after appending all converted annotations.

    Notes:
        - Geometries are constructed from `processed_predictions["contours"]` using
          `geom_type` (default `"Polygon"`), scaled by `scale_factor`, and translated
          by `origin`. Invalid geometries are auto-corrected using `make_valid_poly`.
        - Per-object properties are created by taking the i-th element from each
          remaining key in `processed_predictions`. Scalars are coerced to arrays
          first, then converted with `.tolist()` to ensure JSON-serializable values.
        - If `class_dict` is provided and a `"type"` key exists, `"type"` values are
          mapped prior to saving.
        - All annotations are appended in a single batch via `store.append_many(...)`.

    """
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
    """Align patch predictions to the global output index and pad to cover gaps.

    This helper prepares a *full-sized* set of outputs for the current batch by
    aligning patch-level predictions with the remaining global output locations.
    It uses the provided `full_output_locs` (the outstanding locations yet to be
    filled) to place each patch's predictions at the correct indices, returning
    arrays sized to the current span. If this is the final batch (`is_last=True`),
    it pads the arrays with zeros to cover any remaining, unmatched output
    locations and appends those locations to `output_locs`.

    Concretely:
      1) A lookup is built over `full_output_locs` so each row in `batch_locs`
         maps to a unique index (“match”).
      2) For each head in `batch_output`, an appropriately sized zero-initialized
         array is created and the matched batch predictions are placed at the
         computed indices.
      3) `output_locs` is extended by the portion of `full_output_locs` covered
         in this call; `full_output_locs` is advanced accordingly.
      4) If `is_last=True`, the function also appends any remaining locations to
         `output_locs` and pads the per-head arrays with zeros so their first
         dimension matches the updated number of locations.

    Args:
        batch_output (tuple[np.ndarray]):
            Tuple of per-head patch predictions for the current batch. Each
            element has shape ``(N, H, W, C)`` (head-specific), where ``N`` is
            the number of patches in the batch.
        batch_locs (np.ndarray):
            Array of output locations (e.g., patch output boxes) corresponding
            to `batch_output`. Each row must uniquely identify a location and
            match rows in `full_output_locs`.
        full_output_locs (np.ndarray):
            The remaining global output location array, carrying the canonical
            order of all locations that should be filled. This is progressively
            consumed from the front as batches are placed.
        output_locs (np.ndarray):
            Accumulated output location array across previous batches. This is
            extended in-place with the portion of `full_output_locs` filled in
            this call, and with any remaining tail (zeros padded in outputs)
            when `is_last=True`.
        is_last (bool):
            Whether this is the final batch. When True, any locations left in
            `full_output_locs` after placing matches are appended to
            `output_locs`, and the per-head output arrays are padded with zeros
            to match the total number of output locations.

    Returns:
        tuple[list[np.ndarray], np.ndarray, np.ndarray]:
            - full_batch_output (list[np.ndarray]):
              One array per head containing the aligned outputs for this call.
              Each has shape ``(M, H, W, C)``, where ``M`` is the number of
              locations consumed (and possibly padded to include the remaining
              tail when `is_last=True`).
            - full_output_locs (np.ndarray):
              Updated remaining global output locations (the unconsumed tail).
            - output_locs (np.ndarray):
              Updated accumulated output locations including those added by
              this call (and any final tail when `is_last=True`).

    Notes:
        - Ordering is defined by `full_output_locs`. The number of rows
          consumed during this call equals ``max(match_indices) + 1``.
        - Padding on the last batch is performed with zeros of the same dtype
          as each head's predictions (uint8 for the padded section in the
          implementation).
        - This function is agnostic to the semantic meaning of locations; it
          only ensures that per-head arrays and the accumulated location index
          remain consistent across batches.

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
    """Merge horizontally a run of patch outputs into per-head row blocks.

    This helper performs **row-wise stitching** of patch predictions for
    multitask heads. It consumes the leftmost segment of ``canvas_np`` (per head)
    up to each index in ``change_indices``—which mark where the dataloader
    advanced to a new row of output patches—and merges that segment into a
    horizontally concatenated row block for each head. The merged blocks and
    their per-pixel hit counts are appended to ``canvas`` and ``count`` (as
    Dask arrays with chunking equal to the merged row height), while the consumed
    portion is removed from ``canvas_np``. The function also updates and returns
    ``output_locs`` (with the consumed locations removed) and accumulates the
    vertical extents of each merged row in ``output_locs_y_``.

    For each row segment:
      1) The function determines the row's horizontal span from
         ``output_locs`` (min x0, max x1).
      2) For each head, it calls ``merge_batch_to_canvas`` to place the segment's
         patch outputs into a contiguous row block and an aligned count map.
      3) The row block and count map are wrapped as Dask arrays and appended to
         the running lists in ``canvas`` and ``count`` (one list per head).
      4) The segment is removed from ``canvas_np`` and ``output_locs``; the
         segment's vertical bounds ``(y0, y1)`` are appended to ``output_locs_y_``.

    Args:
        canvas (list[da.Array] | list[None]):
            Accumulated per-head row blocks (probability/logit sums) as Dask
            arrays. Each entry grows along the first axis with each merged row.
            Pass ``None`` for each head on the first call.
        count (list[da.Array] | list[None]):
            Accumulated per-head row count maps, aligned with ``canvas``.
            Pass ``None`` for each head on the first call.
        output_locs_y_ (np.ndarray):
            Accumulated vertical extents of already-merged rows. Each appended
            element is ``[y0, y1]`` corresponding to the merged row's span.
            Pass ``None`` on the first call; it will be initialized internally
            via concatenation.
        canvas_np (list[np.ndarray]):
            In-memory patch outputs awaiting merge, one list entry per head.
            Each head's entry is a NumPy array of stacked patch outputs for the
            **current** unmerged part of the row, with shape
            ``(N_seg, H, W, C)`` for the segment being merged.
        output_locs (np.ndarray):
            Output placement boxes for the awaiting patches in ``canvas_np``,
            shaped ``(N_pending, 4)`` as ``[x0, y0, x1, y1]``. The function
            consumes from the front up to each ``change_indices`` boundary and
            returns the remaining tail.
        change_indices (np.ndarray | list[int]):
            Sorted indices (relative to the current ``output_locs``) where a
            **row change** occurs. Each index marks the end of a contiguous row
            segment to be merged in this call.

    Returns:
        tuple[list[da.Array], list[da.Array], list[np.ndarray], np.ndarray, np.ndarray]:
            - ``canvas``:
              Updated list of per-head Dask arrays containing concatenated row
              blocks (values are sums; normalization happens later).
            - ``count``:
              Updated list of per-head Dask arrays containing concatenated row
              hit counts for normalization.
            - ``canvas_np``:
              Updated in-memory per-head arrays with consumed segment removed.
            - ``output_locs``:
              Updated placement boxes with the consumed segment removed.
            - ``output_locs_y_``:
              Updated array of accumulated vertical row extents, with the new
              row's ``[y0, y1]`` appended.

    Notes:
        - The merged row block shape per head is
          ``(row_height, row_width, C)``, where:
            * ``row_height`` is the head's patch output height,
            * ``row_width`` is ``max(x1) - min(x0)`` for the row,
            * ``C`` is the number of channels for that head.
        - ``merge_batch_to_canvas`` handles placement and accumulation of
          overlapping patch outputs and produces a matching count map.
        - Normalization (division by counts) is **not** performed here; it is
          done later during vertical merging to form the final probability maps.
        - Dask chunking is set to the full row height to facilitate subsequent
          vertical concatenation and overlap handling.

    """
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
    """Write accumulated horizontal row blocks to a Zarr cache on disk.

    This function is called when intermediate per-head accumulators
    (``canvas`` and ``count``) become large enough to risk exceeding the
    memory threshold. It computes the current Dask arrays for each head,
    writes them to Zarr datasets under ``save_path``, and updates
    ``canvas_zarr`` / ``count_zarr`` so later merges operate directly on
    Zarr-backed arrays rather than holding everything in memory.

    For each head:
      1) The corresponding ``canvas`` and ``count`` Dask arrays are fully
         computed.
      2) If this is the first time spilling for that head, new Zarr datasets
         are created using chunk shapes consistent with the canvas rows.
      3) The computed rows are appended to the Zarr datasets by resizing the
         arrays and writing the new rows at the end.
      4) The updated Zarr arrays are returned to be wrapped by Dask in later
         steps.

    Args:
        canvas (list[da.Array]):
            Accumulated per-head row blocks (probability/logit sums). Each
            head's entry has shape ``(N_rows, H, W, C)`` where ``N_rows`` grows
            as horizontal rows are merged.
        count (list[da.Array]):
            Accumulated per-head row hit counts aligned with ``canvas``,
            with matching shape and chunking.
        canvas_zarr (list[zarr.Array | None]):
            List of Zarr datasets for storing accumulated ``canvas`` values
            per head. ``None`` entries indicate that no Zarr datasets have
            been created yet for those heads.
        count_zarr (list[zarr.Array | None]):
            List of Zarr datasets mirroring ``canvas_zarr`` but storing hit
            counts instead of accumulated values.
        save_path (str | Path):
            Path to the Zarr group used for caching. A new group is created
            if needed on the first spill.

    Returns:
        tuple[list[zarr.Array], list[zarr.Array]]:
            Updated ``canvas_zarr`` and ``count_zarr`` lists, where each head
            now has a Zarr dataset containing all accumulated rows up to this
            point.

    Notes:
        - Chunking for the Zarr datasets follows the Dask chunk size along
          the row axis to allow efficient later vertical merging.
        - This function does **not** normalize probabilities; normalization
          happens in the final vertical merge via
          ``merge_multitask_vertical_chunkwise``.
        - After spilling, upstream functions will reset in-memory ``canvas``
          and ``count`` to free RAM and continue populating new entries.

    """
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
    """Merge horizontally stitched row blocks into final WSI probability maps.

    After horizontal stitching, each head has a stack of row blocks (values) and
    matching row-wise count maps. This function merges those rows **vertically**,
    resolving overlaps between adjacent rows using the provided `output_locs_y_`
    spans. For each head and row boundary, overlapping rows are summed in the
    overlap region, then normalized by the corresponding summed counts. The
    normalized row is appended to a Zarr-backed or Dask-backed accumulator to
    build the final full-height probability map.

    Concretely, for each head:
      1) Iterate across row boundaries using `output_locs_y_`, compute overlap height.
      2) If there is an overlap with the next row, add overlapping slices from
         the next row's canvas and count into the tail of the current row.
      3) Normalize the current row by its count map (with zero-division guarded).
      4) Append normalized rows to Zarr (or keep in-memory) via `store_probabilities`.
      5) Periodically spill in-memory arrays to Zarr when memory exceeds
         `memory_threshold` (via `_save_multitask_vertical_to_cache`).
      6) After processing all rows, clear temporary Zarr datasets for canvas/count
         and return a Dask view (from Zarr if spilled, otherwise from memory).

    Args:
        canvas (list[da.Array]):
            Per-head Dask arrays of horizontally merged **row blocks** (sums).
            For each head `h`, `canvas[h]` has shape
            `(N_rows, row_height, row_width, C)`, chunked along the row axis.
        count (list[da.Array]):
            Per-head Dask arrays of **row-wise hit counts** matching `canvas`.
        output_locs_y_ (np.ndarray):
            Array of shape `(N_rows, 2)` where each row is `[y0, y1]` indicating
            the vertical extent of the corresponding row block in slide
            coordinates. Overlaps are computed as `prev_y1 - next_y0`.
        zarr_group (zarr.Group):
            Zarr group used to create/append the per-head probability datasets
            (under `"probabilities/{idx}"`) and to clear temporary `"canvas"` and
            `"count"` datasets after finalization.
        save_path (Path):
            Base path of the Zarr store (used when spilling additional data and
            when returning Zarr-backed Dask arrays).
        memory_threshold (int):
            Maximum allowed RAM usage (percentage) before converting in-memory
            probability accumulators to Zarr-backed arrays. Default is 80.

    Returns:
        list[da.Array]:
            One Dask array per head, each representing the **final** WSI-sized
            probability map with shape `(H, W, C)`. If spilling occurred, these
            are backed by Zarr datasets created under `zarr_group`; otherwise
            they are in-memory Dask arrays.

    Notes:
        - Overlaps along the vertical direction are handled by **additive merge**
          of both values and counts, followed by normalization. Non-overlapping
          regions are passed through unchanged.
        - Zero counts are guarded by replacing with 1 during normalization to
          avoid division by zero; this is safe because values are zero where
          counts are zero.
        - Chunking along the first axis (row blocks) is preserved to facilitate
          incremental appends and memory spill; final arrays are exposed with
          appropriate Dask chunking for downstream use.
        - Temporary row-level `"canvas/*"` and `"count/*"` datasets are deleted
          before returning when Zarr-backed accumulators are used (see
          `_clear_zarr`).

    """
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
    tile_shape: tuple[int, int],
    tile_flag: tuple[int, int, int, int],
    tile_mode: int,
    tile_tl: tuple[int, int],
    ref_inst_dict: dict,
) -> list | tuple:
    """Function to merge new tile prediction with existing prediction.

    Args:
        inst_dict (dict):
            Dictionary containing instance information.
        ioconfig (:class:`IOSegmentorConfig`):
            Object defines information
            about input and output placement of patches.
        tile_shape (tuple(int, int)):
            A list of the tile shape.
        tile_flag (list):
            A list of flag to indicate if instances within
            an area extended from each side (by `ioconfig.margin`) of
            the tile should be replaced by those within the same spatial
            region in the accumulated output this run. The format is
            [top, bottom, left, right], 1 indicates removal while 0 is not.
            For example, [1, 1, 0, 0] denotes replacing top and bottom instances
            within `ref_inst_dict` with new ones after this processing.
        tile_mode (int):
            A flag to indicate the type of this tile. There
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
        tile_tl (tuple):
            Top left coordinates of the current tile.
        ref_inst_dict (dict):
            Dictionary contains accumulated output. The
            expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.

    Returns:
        new_inst_dict (dict):
            A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list):
            List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.

    """
    # should be rare, no nuclei detected in input images
    if len(inst_dict) == 0:
        return {}, []

    sel_indices, margin_lines = _get_sel_indices_margin_lines(
        ioconfig=ioconfig,
        tile_shape=tile_shape,
        inst_dict=inst_dict,
        tile_tl=tile_tl,
        tile_mode=tile_mode,
        tile_flag=tile_flag,
    )

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

    new_inst_dict = _move_tile_space_to_wsi_space(
        inst_dict=inst_dict,
        tile_tl=tile_tl,
        remove_insts_in_tile=remove_insts_in_tile,
    )

    return new_inst_dict, remove_insts_in_orig


def _get_sel_indices_margin_lines(
    ioconfig: IOSegmentorConfig,
    tile_shape: tuple[int, int],
    tile_flag: tuple[int, int, int, int],
    tile_mode: int,
    tile_tl: tuple[int, int],
    inst_dict: dict,
) -> tuple[list, list]:
    """Helper function to retrieve margin lines and selected indices within bounds."""
    if tile_mode not in [0, 1, 2, 3]:
        msg = f"Unknown tile mode {tile_mode}."
        raise ValueError(msg)

    margin = ioconfig.margin
    width, height = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    tile_rtree = STRtree(geometries)
    # !

    # create margin bounding box, ordering should match with
    # created tile info flag (top, bottom, left, right)
    boundary_lines = [
        shapely_box(0, 0, width, 1),  # top egde
        shapely_box(0, height - 1, width, height),  # bottom edge
        shapely_box(0, 0, 1, height),  # left
        shapely_box(width - 1, 0, width, height),  # right
    ]
    margin_boxes = [
        shapely_box(0, 0, width, margin),  # top egde
        shapely_box(0, height - margin, width, height),  # bottom edge
        shapely_box(0, 0, margin, height),  # left
        shapely_box(width - margin, 0, width, height),  # right
    ]
    margin_lines = _get_margin_lines(
        margin=margin,
        height=height,
        width=width,
        tile_tl=tile_tl,
    )

    # the ids within this match with those within `inst_map`, not UUID
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
        return sel_indices, margin_lines

    # otherwise if tile_mode in [1, 2]:
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

    return sel_indices, margin_lines


def _get_margin_lines(
    margin: int,
    height: int,
    width: int,
    tile_tl: tuple[int, int],
) -> list:
    """Helper function to get margin lines."""
    # ! this is wrt to WSI coord space, not tile
    margin_lines = [
        [[margin, margin], [width - margin, margin]],  # top egde
        [[margin, height - margin], [width - margin, height - margin]],  # bottom edge
        [[margin, margin], [margin, height - margin]],  # left
        [[width - margin, margin], [width - margin, height - margin]],  # right
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    return [shapely_box(*v.flatten().tolist()) for v in margin_lines]


def _move_tile_space_to_wsi_space(
    inst_dict: dict,
    tile_tl: tuple,
    remove_insts_in_tile: list,
) -> dict:
    """Helper function to move inst dict from tile space to wsi space."""
    # move inst position from tile space back to WSI space
    # and also generate universal uid as replacement for storage
    new_inst_dict = {}
    for inst_uid, inst_info in inst_dict.items():
        if inst_uid not in remove_insts_in_tile:
            inst_info["box"] += np.concatenate([tile_tl] * 2)
            if "centroid" in inst_info:
                inst_info["centroid"] += tile_tl
            inst_info["contours"] += tile_tl
            inst_uuid = uuid.uuid4().hex
            new_inst_dict[inst_uuid] = inst_info
    return new_inst_dict


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
                name=f"{post_process_output_['task_type']}/predictions",
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
