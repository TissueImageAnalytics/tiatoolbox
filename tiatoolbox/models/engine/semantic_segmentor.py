"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import psutil
import torch
import zarr
from dask import compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.models.dataset.dataset_abc import WSIPatchDataset
from tiatoolbox.utils.misc import (
    dict_to_store_semantic_segmentor,
    dict_to_zarr,
    get_tqdm,
)

from .patch_predictor import PatchPredictor, PredictorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader


def merge_batch_to_canvas(
    blocks: np.ndarray,
    output_locations: np.ndarray,
    merged_shape: tuple[int, int, int],
    dtype_: type,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge patch-level predictions into a single canvas.

    This function aggregates overlapping patch predictions into a unified
    output canvas and maintains a count map to normalize overlapping regions.

    Args:
        blocks (np.ndarray):
            Array of predicted blocks with shape (N, H, W, C), where N is the
            number of patches.
        output_locations (np.ndarray):
            Array of coordinates for each block in the format
            [start_x, start_y, end_x, end_y] with shape (N, 4).
        merged_shape (tuple[int, int, int]):
            Shape of the final merged canvas (H, W, C).
        dtype_ (type):
            Data type of the output canvas.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - canvas: Merged prediction map of shape (H, W, C).
            - count: Count map indicating how many times each pixel was updated,
              shape (H, W).

    """
    canvas = np.zeros(merged_shape, dtype=dtype_)
    count = np.zeros(merged_shape[:2], dtype=np.uint8)
    for i, block in enumerate(blocks):
        xs, ys, xe, ye = output_locations[i]
        # To deal with edge cases
        ye, xe = min(ye, canvas.shape[0]), min(xe, canvas.shape[1])
        canvas[ys:ye, xs:xe, :] += block[0 : ye - ys, 0 : xe - xs, :]
        count[ys:ye, xs:xe] += 1
    return canvas, count


class SemanticSegmentorRunParams(PredictorRunParams, total=False):
    """Runtime parameters for configuring the `SemanticSegmentor.run()` method.

    This class extends `PredictorRunParams` with additional parameters
    specific to semantic segmentation workflows.

    Attributes:
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        class_dict (dict):
            Optional dictionary mapping classification outputs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda").
        ioconfig (ModelIOConfigABC):
            Input/output configuration for patch extraction and resolution.
        return_labels (bool):
            Whether to return labels with predictions.
        num_loader_workers (int):
            Number of workers used in DataLoader.
        num_post_proc_workers (int):
            Number of workers used for post-processing.
        output_file (str):
            Output file name for saving results (e.g., .zarr or .db).
        patch_input_shape (tuple[int, int]):
            Shape of input patches (height, width).
        input_resolutions (list[dict]):
            Resolution used for reading the image. See `WSIReader` for details.
        return_probabilities (bool):
            Whether to return per-class probabilities.
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.
        patch_output_shape (tuple[int, int]):
            Shape of output patches (height, width).
        output_resolutions (Resolution):
            Resolution used for writing output predictions.

    """

    patch_output_shape: tuple[int, int]
    output_resolutions: Resolution

    patch_output_shape: tuple
    output_resolutions: Resolution


class SemanticSegmentor(PatchPredictor):
    r"""Semantic segmentation engine for digital histology images.

    This class extends `PatchPredictor` to support semantic segmentation tasks
    using pretrained or custom models from TIAToolbox. It supports both patch-level
    and whole slide image (WSI) processing, and provides utilities for merging,
    post-processing, and saving predictions.

    Performance:
        The TIAToolbox model `fcn_resnet50_unet-bcss` achieves the following
        results on the BCSS dataset:

        .. list-table:: Semantic segmentation performance on the BCSS dataset
           :widths: 15 15 15 15 15 15 15
           :header-rows: 1

           * -
             - Tumour
             - Stroma
             - Inflammatory
             - Necrosis
             - Other
             - All
           * - Amgad et al.
             - 0.851
             - 0.800
             - 0.712
             - 0.723
             - 0.666
             - 0.750
           * - TIAToolbox
             - 0.885
             - 0.825
             - 0.761
             - 0.765
             - 0.581
             - 0.763

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
        num_loader_workers (int):
            Number of workers for data loading. Default is 0.
        num_post_proc_workers (int):
            Number of workers for post-processing. Default is 0.
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
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(wsis, patch_mode=False)

        >>> # array of list of 2 image patches as input
        >>> image_patches = [np.ndarray, np.ndarray]
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(data, patch_mode=True)

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(data, patch_mode=False)

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(tile_file, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsis = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> segmentor = SemanticSegmentor(model="resnet18-kather100k")
        >>> output = segmentor.run(wsis, patch_mode=False)

    References:
        [1] Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. Structured crowdsourcing
        enables convolutional segmentation of histology images. Bioinformatics 2019.
        doi: 10.1093/bioinformatics/btz083

    """

    def __init__(
        self: SemanticSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`SemanticSegmentor`.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or name of a pretrained model from TIAToolbox.
                If a string is provided, the corresponding pretrained weights will be
                downloaded unless overridden via `weights`.
            batch_size (int):
                Number of image patches processed per forward pass. Default is 8.
            num_loader_workers (int):
                Number of workers for data loading. Default is 0.
            num_post_proc_workers (int):
                Number of workers for post-processing. Default is 0.
            weights (str | Path | None):
                Path to model weights. If None, default weights are used.
            device (str):
                Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Whether to enable verbose logging. Default is True.

        """
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )
        self.output_locations: list | None = None

    def get_dataloader(
        self: SemanticSegmentor,
        images: str | Path | list[str | Path] | np.ndarray,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: SemanticSegmentorRunParams | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return a DataLoader for inference.

        This method prepares the dataset and returns a PyTorch DataLoader
        for either patch-based or WSI-based semantic segmentation.

        Args:
            images (str | Path | list[str | Path] | np.ndarray):
                Input images. Can be a list of file paths or a NumPy array
                of image patches in NHWC format.
            masks (Path | None):
                Optional tissue masks for WSI processing. Only used when
                `patch_mode` is False.
            labels (list | None):
                Optional labels for input images. Only one label per image is supported.
            ioconfig (SemanticSegmentorRunParams | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).

        Returns:
            torch.utils.data.DataLoader:
                A PyTorch DataLoader configured for inference.

        """
        # Overwrite when patch_mode is False.
        if not patch_mode:
            dataset = WSIPatchDataset(
                img_path=images,
                mask_path=masks,
                patch_input_shape=ioconfig.patch_input_shape,
                patch_output_shape=ioconfig.patch_output_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )

            dataset.preproc_func = self.model.preproc_func
            self.output_locations = dataset.outputs

            # preprocessing must be defined with the dataset
            return torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_loader_workers,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )

        return super().get_dataloader(
            images=images,
            masks=masks,
            labels=labels,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
        )

    def _merge_model_output_to_dask_canvas(
        self: SemanticSegmentor,
        batch_data: dict[str, Any],
        merged_shape: zarr.Array,
        canvas_dtype: zarr.Array,
    ) -> tuple[da.Array, da.Array]:
        """Merge model outputs from a batch into a shared canvas and count map.

        This method performs inference on a batch of image patches, aligns the
        outputs based on their spatial locations, and merges them into a full-resolution
        canvas. It also maintains a count map to track the number of contributions
        per pixel.

        Args:
            batch_data (dict[str, Any]):
                Dictionary containing batch input data. Expected keys:
                - "image": Batch of input images (as a Tensor) for inference.
                - "output_locs": Tensor with bounding box coordinates for each output.
            merged_shape (tuple[int, int, int]):
                Shape of the final merged canvas (height, width, channels).
            canvas_dtype (np.dtype):
                Data type for the canvas array.

        Returns:
            tuple[dask.array.Array, dask.array.Array]:
                - canvas: Dask array containing the merged outputs.
                - count: Dask array indicating the number of times
                  each pixel was updated.

        """
        canvas = da.zeros(
            merged_shape,
            dtype=canvas_dtype,
        )
        count = da.zeros(
            merged_shape[:2],
            dtype=np.uint8,
        )
        batch_output = self.model.infer_batch(
            self.model,
            batch_data["image"],
            device=self.device,
        )

        output_locs = batch_data["output_locs"].numpy()

        batch_xs, batch_ys = np.min(output_locs[:, 0:2], axis=0)
        batch_xe, batch_ye = np.max(output_locs[:, 2:4], axis=0)

        merged_shape_batch = (
            batch_ye - batch_ys,
            batch_xe - batch_xs,
            batch_output.shape[3],
        )

        merged_output, merged_count = merge_batch_to_canvas(
            batch_output,
            output_locs - np.array([batch_xs, batch_ys, batch_xs, batch_ys]),
            merged_shape_batch,
            batch_output.dtype,
        )

        batch_ye, batch_xe = (
            min(batch_ye, canvas.shape[0]),
            min(batch_xe, canvas.shape[1]),
        )

        canvas[
            batch_ys:batch_ye,
            batch_xs:batch_xe,
            :,
        ] += merged_output

        count[
            batch_ys:batch_ye,
            batch_xs:batch_xe,
        ] += merged_count

        return canvas, count

    @staticmethod
    def _write_canvas_count_to_zarr(
        slice_to_write: slice,
        canvas: da.Array,
        count: da.Array,
        canvas_zarr: zarr.Array,
        count_zarr: zarr.Array,
    ) -> None:
        lock = SerializableLock()
        write_task = []
        task = canvas[slice_to_write, :, :].to_zarr(
            canvas_zarr,
            region=(slice_to_write, slice(None), slice(None)),
            compute=False,
            lock=lock,
        )
        write_task.append(task)

        task = count[slice_to_write, :].to_zarr(
            count_zarr,
            region=(slice_to_write, slice(None)),
            compute=False,
            lock=lock,
        )
        write_task.append(task)

        print("\nWriting done... \n")  # noqa: T201
        with ProgressBar():
            compute(*write_task)
        print("\nWriting done!!!\n")  # noqa: T201

    def _spill_to_disk(
        self: SemanticSegmentor,
        batch_info: list[dict[str, Any]],
        canvas: da.Array,
        count: da.Array,
        canvas_zarr: zarr.Array,
        count_zarr: zarr.Array,
        tqdm_loop: tqdm.tqdm,
        max_save_y: int,
        min_save_y: int,
        memory_threshold: int,
    ) -> tuple[da.Array, da.Array, int, int]:
        vm = psutil.virtual_memory()
        used_percent = vm.percent

        # If within threshold limit return
        if used_percent < memory_threshold:
            return canvas, count, min_save_y, max_save_y

        # Else Spill to disk
        y_info = batch_info.pop(0)
        min_y = y_info["min_starty"]

        tqdm_loop.desc = "Memory Overload: Spilling to Disk"

        # Try to continue if there is available memory for an entire row
        if min_y == 0:
            msg = (
                f"Memory usage is too high ({used_percent}%). "
                f"Current memory usage: {vm.total} bytes. "
                f"Increase Memory threshold, reduce batch size "
                f"or switch device to CPU."
            )
            logger.warning(msg)
            return canvas, count, min_save_y, max_save_y

        # When Spill is triggered for first time
        # Avoids zero slice
        if max_save_y == 0:
            max_save_y = min_y - 1

        # Check if all rows have been processed
        if max_save_y < min_y:
            canvas_slice = slice(min_save_y, max_save_y)

            canvas = canvas.rechunk(canvas_zarr.chunks).persist()
            count = count.rechunk(count_zarr.chunks).persist()

            # Spill to disk
            self._write_canvas_count_to_zarr(
                slice_to_write=canvas_slice,
                canvas=canvas,
                count=count,
                canvas_zarr=canvas_zarr,
                count_zarr=count_zarr,
            )

            canvas[canvas_slice, :, :] = 0
            count[canvas_slice, :] = 0

            canvas = canvas.persist()
            count = count.persist()

            # # Align chunks only if needed
            # if canvas.chunks != canvas_zarr.chunks:
            #     canvas_ = canvas.rechunk(canvas_zarr.chunks)
            # else:
            #     canvas_ = canvas
            #
            # if count.chunks != count_zarr.chunks:
            #     count_ = count.rechunk(count_zarr.chunks)
            # else:
            #     count_ = count
            #
            # canvas_ = canvas_.persist()
            # count_ = count_.persist()
            #
            # # Reinitialize with sparse arrays
            # canvas = da.zeros(
            #     shape=canvas_zarr.shape,
            #     dtype=canvas_zarr.dtype,
            #     chunks=canvas_zarr.chunks,
            # ).persist()
            # count = da.zeros(
            #     shape=count_zarr.shape,
            #     dtype=np.uint8,
            #     chunks=count_zarr.chunks,
            # ).persist()
            #
            # # Restore unsaved region
            # restore_slice = slice(max_save_y, None)
            #
            # # Restore unsaved region (explicit compute)
            # restored_canvas = canvas_[restore_slice, :, :]
            # restored_count = count_[restore_slice, :]
            # canvas[restore_slice, :, :] = restored_canvas.persist()
            # count[restore_slice, :] = restored_count.persist()

            # Cleanup
            # del canvas_, count_, restored_canvas, restored_count
            gc.collect()

            # Update boundaries
            min_save_y = max_save_y
            max_save_y = y_info["max_endy"]
        tqdm_loop.desc = "Inferring patches"

        return canvas, count, min_save_y, max_save_y

    def infer_wsi(
        self: SemanticSegmentor,
        dataloader: DataLoader,
        save_path: Path,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict[str, da.Array]:
        """Perform model inference on a whole slide image (WSI).

        This method processes a WSI using the provided DataLoader, merges
        patch-level predictions into a full-resolution canvas, and returns
        the aggregated output. It supports optional inclusion of coordinates
        and labels.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader configured for WSI processing.
            save_path (Path):
                Path to save the intermediate output. The intermediate output is saved
                in a zarr file.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters, including:
                - return_probabilities (bool): Whether to return probability maps.
                - return_labels (bool): Whether to include labels in the output.

        Returns:
            dict[str, dask.array.Array]:
                Dictionary containing merged prediction results:
                - "probabilities": Full-resolution probability map.
                - "coordinates": Patch coordinates.
                - "labels": Ground truth labels (if `return_labels` is True).

        """
        # Default Memory threshold percentage is 80.
        memory_threshold = kwargs.get("memory_threshold", 80)

        keys = ["probabilities", "coordinates"]
        if self.return_labels:
            keys.append("labels")

        coordinates, labels = [], []

        # Main output dictionary
        raw_predictions = dict(zip(keys, [[]] * len(keys)))

        # sample for calculating shape for dask arrays
        sample = self.dataloader.dataset[0]  # Use only the first image
        sample_output = self.model.infer_batch(
            self.model,
            torch.Tensor(sample["image"][np.newaxis, ...]),
            device=self.device,
        )

        # Create canvas and counts
        max_location = np.max(self.output_locations, axis=0)
        merged_shape = (
            max_location[3],
            max_location[2],
            sample_output.shape[3],
        )

        zarr_group = zarr.open(str(save_path), mode="a")

        canvas_zarr = zarr_group.create_dataset(
            name="canvas",
            shape=merged_shape,
            dtype=sample_output.dtype,
            fillvalue=0,
            overwrite=True,
        )

        count_zarr = zarr_group.create_dataset(
            name="count",
            shape=merged_shape[:2],
            dtype=np.uint8,
            fillvalue=0,
            overwrite=True,
        )

        canvas = da.zeros(
            merged_shape,
            dtype=sample_output.dtype,
            chunks=canvas_zarr.chunks,
        )
        count = da.zeros(
            merged_shape[:2],
            dtype=np.uint8,
            chunks=count_zarr.chunks,
        )

        # Inference loop
        tqdm = get_tqdm()
        tqdm_loop = (
            tqdm(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else self.dataloader
        )

        locations = self.output_locations
        batch_size = dataloader.batch_size

        num_full_batches = locations.shape[0] // batch_size
        remainder = locations.shape[0] % batch_size

        batches = (
            np.array_split(locations[: num_full_batches * batch_size], num_full_batches)
            if num_full_batches
            else [locations]
        )

        # Optionally include the remainder
        if remainder > 0 and num_full_batches > 0:
            batches.append(locations[-remainder:])

        # Compute min/max for starty (col 1) and endy (col 3)
        batch_info = [
            {"min_starty": np.min(batch[:, 1]), "max_endy": np.max(batch[:, 3])}
            for batch in batches
        ]

        min_save_y = 0
        max_save_y = 0

        for batch_data in tqdm_loop:
            canvas_batch, count_batch = self._merge_model_output_to_dask_canvas(
                batch_data=batch_data,
                merged_shape=merged_shape,
                canvas_dtype=canvas.dtype,
            )

            canvas = canvas + canvas_batch.rechunk(canvas_zarr.chunks)
            count = count + count_batch.rechunk(count_zarr.chunks)

            canvas, count, min_save_y, max_save_y = self._spill_to_disk(
                batch_info=batch_info,
                canvas=canvas,
                count=count,
                canvas_zarr=canvas_zarr,
                count_zarr=count_zarr,
                tqdm_loop=tqdm_loop,
                min_save_y=min_save_y,
                max_save_y=max_save_y,
                memory_threshold=memory_threshold,
            )

            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

            if self.return_labels:
                labels.append(da.from_array(np.array(batch_data["label"])))

        self._write_canvas_count_to_zarr(
            slice_to_write=slice(min_save_y, None),
            canvas=canvas,
            count=count,
            canvas_zarr=canvas_zarr,
            count_zarr=count_zarr,
        )

        # Free up memory
        del canvas, count
        gc.collect()

        # Reinitialize using zarr
        canvas = da.from_zarr(canvas_zarr)
        count = da.from_zarr(count_zarr)
        canvas = canvas / da.maximum(count[:, :, np.newaxis], 1)

        raw_predictions["probabilities"] = canvas.rechunk("auto")
        raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)
        if self.return_labels:
            labels = [label.reshape(-1) for label in labels]
            raw_predictions["labels"] = da.concatenate(labels, axis=0)

        return raw_predictions

    def save_predictions(
        self: SemanticSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path:
        """Save semantic segmentation predictions to disk or return them in memory.

        This method saves predictions in one of the supported formats:
        - "dict": returns predictions as a Python dictionary.
        - "zarr": saves predictions as a Zarr group and returns the path.
        - "annotationstore": converts predictions to an AnnotationStore (.db file).

        If `patch_mode` is True, predictions are saved per image. If False,
        predictions are merged and saved as a single output.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
            save_path (Path | None):
                Path to save the output file. Required for "zarr" and "annotationstore".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters including:
                - scale_factor (tuple[float, float]): For coordinate transformation.
                - class_dict (dict): Mapping of class indices to names.
                - return_probabilities (bool): Whether to save probability maps.

        Returns:
            dict | AnnotationStore | Path:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved Zarr file.
                - If output_type is "annotationstore": returns AnnotationStore
                  or path to .db file.

        """
        # Conversion to annotationstore uses a different function for SemanticSegmentor
        if output_type.lower() != "annotationstore":
            return super().save_predictions(
                processed_predictions, output_type, save_path=save_path, **kwargs
            )

        logger.info("Saving predictions as AnnotationStore.")
        processed_predictions = super().save_predictions(
            processed_predictions, output_type="dict", **kwargs
        )

        return_probabilities = kwargs.get("return_probabilities", False)

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")

        # Need to add support for zarr conversion.
        save_paths = []

        if self.patch_mode:
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                out_file = dict_to_store_semantic_segmentor(
                    patch_output={"predictions": predictions},
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
        else:
            out_file = dict_to_store_semantic_segmentor(
                patch_output=processed_predictions,
                scale_factor=scale_factor,
                class_dict=class_dict,
                save_path=save_path.with_suffix(".db"),
            )
            save_paths = out_file

        if return_probabilities:
            zarr_save_path = save_path.parent.with_suffix(".zarr")
            msg = (
                f"Probability maps cannot be saved as AnnotationStore. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {zarr_save_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)
            processed_predictions = {
                "predictions": processed_predictions.get("predictions"),
            }
            dict_to_zarr(
                raw_predictions=processed_predictions,
                save_path=zarr_save_path,
            )

        return save_paths

    def run(
        self: SemanticSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the semantic segmentation engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results.
        It supports both patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            labels (list | None):
                Optional labels for input images. Only one label per image is supported.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
                Default is True.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
                Default is "dict".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to update engine attributes.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> class SemanticSegmentor(PatchPredictor):
            >>> # Define all Abstract methods.
            >>>     ...
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"
            >>> output = segmentor.run(
            >>>     image_patches,
            >>>     patch_mode=True,
            >>>     output_type="zarr")
            >>> output
            ... "/path/to/Output.zarr"
            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'/path/to/wsi1.db'}

        """
        return super().run(
            images=images,
            masks=masks,
            labels=labels,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )
