"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import psutil
import torch
import zarr
from typing_extensions import Unpack

from tiatoolbox import DuplicateFilter, logger
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

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader


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
                # auto_get_mask=False,
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
        vm = psutil.virtual_memory()

        keys = ["probabilities", "coordinates"]
        if self.return_labels:
            keys.append("labels")

        coordinates, labels = [], []

        # Main output dictionary
        raw_predictions = dict(zip(keys, [[]] * len(keys)))

        # Inference loop
        tqdm = get_tqdm()
        tqdm_loop = (
            tqdm(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else self.dataloader
        )

        canvas_np, count_np, output_locs_y_ = None, None, None
        canvas, count, output_locs = None, None, None
        canvas_zarr, count_zarr = None, None

        full_output_locs = (
            dataloader.dataset.full_outputs
            if hasattr(dataloader.dataset, "full_outputs")
            else dataloader.dataset.outputs
        )

        duplicate_filter = DuplicateFilter()
        logger.addFilter(duplicate_filter)
        for batch_idx, batch_data in enumerate(tqdm_loop):
            batch_output = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            batch_locs = batch_data["output_locs"].numpy()

            # Use np.intersect1d once numpy version is upgraded to 2.0
            full_output_dict = {tuple(row): i for i, row in enumerate(full_output_locs)}
            matches = [full_output_dict[tuple(row)] for row in batch_locs]

            total_size = np.max(matches).astype(np.uint16) + 1
            h, w, c = batch_output.shape[1:]

            # Initialize full output array
            full_batch_output = np.zeros(
                (total_size, h, w, c), dtype=batch_output.dtype
            )
            full_batch_count = np.zeros_like(full_batch_output[:, :, :, 0:1]).astype(
                np.uint8
            )

            # Place matching outputs using matching indices
            full_batch_output[matches] = batch_output
            full_batch_count[matches] = 1

            output_locs = concatenate_none(
                old_arr=output_locs, new_arr=full_output_locs[:total_size]
            )
            full_output_locs = full_output_locs[total_size:]

            if batch_idx == len(dataloader) - 1:
                output_locs = concatenate_none(
                    old_arr=output_locs, new_arr=full_output_locs
                )
                full_batch_output = concatenate_none(
                    old_arr=full_batch_output,
                    new_arr=np.zeros(
                        shape=(len(full_output_locs), h, w, c), dtype=np.uint8
                    ),
                )
                full_batch_count = concatenate_none(
                    old_arr=full_batch_count,
                    new_arr=np.zeros(
                        shape=(len(full_output_locs), h, w, 1), dtype=np.uint8
                    ),
                )

            canvas_np = concatenate_none(old_arr=canvas_np, new_arr=full_batch_output)
            count_np = concatenate_none(old_arr=count_np, new_arr=full_batch_count)

            change_indices = np.where(np.diff(output_locs[:, 1]) != 0)[0] + 1

            if change_indices.size > 0:
                canvas, count, canvas_np, count_np, output_locs, output_locs_y_ = (
                    flush_patches(
                        canvas,
                        count,
                        output_locs_y_,
                        canvas_np,
                        count_np,
                        output_locs,
                        change_indices,
                    )
                )

                used_percent = vm.percent
                # Cache the output if Memory threshold is reached
                # Or if length of dask graph is too long.
                # 50000 is estimated based on trial and error for 64 GB RAM
                if used_percent > memory_threshold or len(canvas.dask) > 25e3:
                    tqdm_loop.desc = "Spilling to disk "
                    msg = (
                        f"Current Memory usage: {used_percent} %  "
                        f"exceeds specified threshold: {memory_threshold}. "
                        if used_percent > memory_threshold
                        else f"Canvas task graph length: {len(canvas.dask)} "
                        f"exceeds specified threshold: {25e3}. "
                    ) + "Saving intermediate results to disk."
                    logger.info(msg)
                    # Flush data in Memory and clear dask graph
                    canvas_zarr, count_zarr = save_to_cache(
                        canvas,
                        count,
                        canvas_zarr,
                        count_zarr,
                        save_path=save_path,
                    )
                    canvas, count = None, None
                    gc.collect()
                    tqdm_loop.desc = "Inferring patches"

            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

            if self.return_labels:
                labels.append(da.from_array(np.array(batch_data["label"])))

        logger.removeFilter(duplicate_filter)
        canvas, count, _, _, _, output_locs_y_ = flush_patches(
            canvas,
            count,
            output_locs_y_,
            canvas_np,
            count_np,
            output_locs,
            change_indices=[len(output_locs)],
        )

        zarr_group = None
        if canvas_zarr is not None:
            canvas_zarr, count_zarr = save_to_cache(
                canvas, count, canvas_zarr, count_zarr
            )
            canvas = da.from_zarr(canvas_zarr, chunks=canvas_zarr.chunks)
            count = da.from_zarr(count_zarr, chunks=count_zarr.chunks)
            zarr_group = zarr.open(canvas_zarr.store.path, mode="a")

        np.save("output-locs-y.npy", output_locs_y_)

        # Final vertical merge
        raw_predictions["probabilities"] = merge_vertical_chunkwise(
            canvas,
            count,
            output_locs_y_,
            zarr_group,
        ).rechunk("auto")
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


def concatenate_none(
    old_arr: np.ndarray | da.Array, new_arr: np.ndarray | da.Array
) -> np.ndarray | da.Array:
    """Helper to concatenate None arrays."""
    if isinstance(new_arr, np.ndarray):
        return (
            new_arr if old_arr is None else np.concatenate((old_arr, new_arr), axis=0)
        )

    return new_arr if old_arr is None else da.concatenate([old_arr, new_arr], axis=0)


def stack_blocks_to_dask(blocks, count=False):
    """Stack a list of NumPy blocks into a Dask array with column-wise chunking."""
    if count:
        return da.concatenate(
            [
                da.from_array(b, chunks=(blocks.shape[1], blocks.shape[2], 1))
                for b in blocks
            ],
            axis=1,
        )
    return da.concatenate(
        [da.from_array(b, chunks=blocks.shape[1:]) for b in blocks], axis=1
    )


def merge_horizontal(canvas_np_, count_np_, output_locs_):
    overlaps = np.append(output_locs_[:-1, 2] - output_locs_[1:, 0], 0)
    max_overlap = np.max(overlaps)
    merge_func = horizontal_merge_func(overlaps, max_overlap)

    dask_canvas = stack_blocks_to_dask(canvas_np_)
    dask_count = stack_blocks_to_dask(count_np_, count=True)

    canvas_merge = dask_canvas.map_overlap(
        merge_func,
        depth={0: 0, 1: max_overlap, 2: 0},
        boundary="none",
        trim=False,
        dtype=dask_canvas.dtype,
    ).rechunk(dask_canvas.shape)

    count_merge = dask_count.map_overlap(
        merge_func,
        depth={0: 0, 1: max_overlap, 2: 0},
        boundary="none",
        trim=False,
        dtype=dask_count.dtype,
    ).rechunk(dask_count.shape)

    canvas_merge = canvas_merge.rechunk(chunks=canvas_merge.shape)
    count_merge = count_merge.rechunk(chunks=count_merge.shape)

    return canvas_merge, count_merge


def flush_patches(
    canvas, count, output_locs_y_, canvas_np, count_np, output_locs, change_indices
):
    start_idx = 0
    for c_idx in change_indices:
        output_locs_ = output_locs[: c_idx - start_idx]
        canvas_np_ = canvas_np[: c_idx - start_idx]
        count_np_ = count_np[: c_idx - start_idx]

        canvas_merge, count_merge = merge_horizontal(
            canvas_np_, count_np_, output_locs_
        )

        canvas = concatenate_none(old_arr=canvas, new_arr=canvas_merge)
        count = concatenate_none(old_arr=count, new_arr=count_merge)
        output_locs_y_ = concatenate_none(
            old_arr=output_locs_y_, new_arr=output_locs[:, (1, 3)]
        )

        canvas_np = canvas_np[c_idx - start_idx :]
        count_np = count_np[c_idx - start_idx :]
        output_locs = output_locs[c_idx - start_idx :]
        start_idx = c_idx

    return canvas, count, canvas_np, count_np, output_locs, output_locs_y_


def horizontal_merge_func(overlaps, max_overlap):
    def merge_horizontal_seams_var(chunk, block_info=None):
        info = block_info[0]
        j = info["chunk-location"][1]
        nJ = info["num-chunks"][1]
        LH = max_overlap if j > 0 else 0
        RH = max_overlap if j < nJ - 1 else 0
        W = chunk.shape[1] - LH - RH

        # Fold right halo
        r = overlaps[j]
        if r > 0:
            chunk[:, LH + W - r : LH + W, :] += chunk[:, LH + W : LH + W + r, :]

        # Drop right halo
        if RH > 0:
            chunk = chunk[:, : LH + W, :]

        # Drop left halo + duplicate seam cols
        if j > 0:
            l = overlaps[j - 1]
            chunk = chunk[:, LH + l :, :]

        return chunk

    return merge_horizontal_seams_var


def save_to_cache(
    canvas, count, canvas_zarr, count_zarr, save_path: str | Path = "temp.zarr"
):
    canvas_computed = canvas.compute()
    count_computed = count.compute()
    chunk_shape = tuple(chunk[0] for chunk in canvas.chunks)
    if canvas_zarr is None:
        zarr_group = zarr.open(str(save_path), mode="w")

        canvas_zarr = zarr_group.create_dataset(
            name="canvas",
            shape=(0, *canvas_computed.shape[1:]),
            chunks=(chunk_shape[0], *canvas_computed.shape[1:]),
            dtype=canvas_computed.dtype,
            overwrite=True,
        )

        count_zarr = zarr_group.create_dataset(
            name="count",
            shape=(0, *count_computed.shape[1:]),
            dtype=count_computed.dtype,
            chunks=(chunk_shape[0], *count_computed.shape[1:]),
            overwrite=True,
        )

    canvas_zarr.resize(
        (canvas_zarr.shape[0] + canvas_computed.shape[0],) + canvas_zarr.shape[1:]
    )
    canvas_zarr[-canvas_computed.shape[0] :] = canvas_computed

    count_zarr.resize(
        (count_zarr.shape[0] + count_computed.shape[0],) + count_zarr.shape[1:]
    )
    count_zarr[-count_computed.shape[0] :] = count_computed

    return canvas_zarr, count_zarr


def merge_vertical_chunkwise(canvas, count, output_locs_y_, zarr_group):
    y0s, y1s = np.unique(output_locs_y_[:, 0]), np.unique(output_locs_y_[:, 1])
    overlaps = np.append(y1s[:-1] - y0s[1:], 0)
    max_overlap = np.max(overlaps)

    num_chunks = canvas.numblocks[0]
    probabilities_zarr, probabilities_da = None, None
    chunk_shape = tuple(chunk[0] for chunk in canvas.chunks)

    tqdm = get_tqdm()
    tqdm_loop = tqdm(range(num_chunks), leave=False, desc="Merging patches")

    prev_chunk, prev_count = None, None

    curr_chunk = canvas.blocks[0, 0].compute()
    curr_count = count.blocks[0, 0].compute()
    next_chunk = canvas.blocks[1, 0].compute() if num_chunks > 1 else None
    next_count = count.blocks[1, 0].compute() if num_chunks > 1 else None

    for i in tqdm_loop:
        top_halo = max_overlap if i > 0 else 0
        bottom_halo = max_overlap if i < num_chunks - 1 else 0

        chunk = curr_chunk
        count_chunk = curr_count

        if top_halo > 0:
            chunk = np.concatenate([prev_chunk[-top_halo:], chunk], axis=0)
            count_chunk = np.concatenate([prev_count[-top_halo:], count_chunk], axis=0)

        if bottom_halo > 0:
            chunk = np.concatenate([chunk, next_chunk[:bottom_halo]], axis=0)
            count_chunk = np.concatenate(
                [count_chunk, next_count[:bottom_halo]], axis=0
            )

        # Apply seam folding
        TH = top_halo
        BH = bottom_halo
        H = chunk.shape[0] - TH - BH

        r = overlaps[i]
        if r > 0 and chunk.shape[0] >= TH + H + r:
            chunk[TH + H - r : TH + H] += chunk[TH + H : TH + H + r]
            count_chunk[TH + H - r : TH + H] += count_chunk[TH + H : TH + H + r]

        # Drop bottom halo
        if BH > 0:
            chunk = chunk[: TH + H]
            count_chunk = count_chunk[: TH + H]

        # Drop top halo + duplicate seam
        if i > 0:
            l = overlaps[i - 1]
            chunk = chunk[TH + l :]
            count_chunk = count_chunk[TH + l :]

        # Normalize
        count_safe = np.where(count_chunk == 0, 1.0, count_chunk)
        if count_safe.ndim == 2:  # noqa: PLR2004
            count_safe = count_safe[:, :, np.newaxis]

        probabilities = chunk / count_safe.astype(np.float32)

        if zarr_group is not None:
            if probabilities_zarr is None:
                probabilities_zarr = zarr_group.create_dataset(
                    name="probabilities",
                    shape=(0, *probabilities.shape[1:]),
                    chunks=(chunk_shape[0], *probabilities.shape[1:]),
                    dtype=probabilities.dtype,
                    overwrite=True,
                )

            probabilities_zarr.resize(
                (probabilities_zarr.shape[0] + probabilities.shape[0],)
                + probabilities_zarr.shape[1:]
            )
            probabilities_zarr[-probabilities.shape[0] :] = probabilities
        else:
            probabilities_da = concatenate_none(
                old_arr=probabilities_da, new_arr=da.from_array(probabilities)
            )

        prev_chunk, prev_count = curr_chunk, curr_count
        curr_chunk, curr_count = next_chunk, next_count
        if i + 2 < num_chunks:
            next_chunk = canvas.blocks[i + 2, 0].compute()
            next_count = count.blocks[i + 2, 0].compute()
        else:
            next_chunk, next_count = None, None

    if probabilities_zarr:
        return da.from_zarr(probabilities_zarr)

    return probabilities_da
