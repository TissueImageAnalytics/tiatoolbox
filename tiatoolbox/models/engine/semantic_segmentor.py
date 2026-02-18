"""Semantic Segmentation Engine for Whole Slide Images (WSIs) using TIAToolbox.

This module defines the `SemanticSegmentor` class, which extends the `PatchPredictor`
engine to support semantic segmentation workflows on digital pathology images.
It leverages deep learning models from TIAToolbox to perform patch-level and
WSI-level inference, and includes utilities for preprocessing, postprocessing,
and saving predictions in various formats.

Key Components:
---------------
Classes:
- SemanticSegmentorRunParams:
    Configuration parameters for controlling runtime behavior during segmentation.
- SemanticSegmentor:
    Core engine for performing semantic segmentation on image patches or WSIs.

Functions:
- concatenate_none:
    Concatenate arrays while gracefully handling None values.
- merge_horizontal:
    Incrementally merge horizontal patches and update location arrays.
- save_to_cache:
    Save intermediate canvas and count arrays to Zarr cache.
- merge_vertical_chunkwise:
    Merge vertically chunked canvas and count arrays into a probability map.
- store_probabilities:
    Store computed probability data in Zarr or Dask arrays.
- prepare_full_batch:
    Align patch-level predictions with global output locations.

Example:
>>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
>>> segmentor = SemanticSegmentor(model="fcn_resnet50_unet-bcss")
>>> wsis = ["slide1.svs", "slide2.svs"]
>>> output = segmentor.run(wsis, patch_mode=False)
>>>
>>> patches = [np.ndarray, np.ndarray]
>>> segmentor = SemanticSegmentor(model="fcn_resnet50_unet-bcss")
>>> output = segmentor.run(patches, patch_mode=True, output_type="dict")

Notes:
------
- Supports both patch-based and WSI-based segmentation.
- Compatible with TIAToolbox pretrained models and custom PyTorch models.
- Outputs can be saved as dictionaries, Zarr arrays, or AnnotationStore databases.
- Includes memory-aware caching and efficient merging strategies for large-scale
  inference.

"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import psutil
import torch
import zarr
from tqdm.auto import tqdm
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.models.dataset.dataset_abc import WSIPatchDataset
from tiatoolbox.utils.misc import (
    dict_to_store_semantic_segmentor,
    update_tqdm_desc,
)
from tiatoolbox.wsicore.wsireader import WSIReader, is_zarr

from .patch_predictor import PatchPredictor, PredictorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units


class SemanticSegmentorRunParams(PredictorRunParams, total=False):
    """Runtime parameters for configuring the `SemanticSegmentor.run()` method.

    This class extends `PredictorRunParams`, which itself extends `EngineABCRunParams`,
    and adds parameters specific to semantic segmentation workflows.

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
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.

    """

    patch_output_shape: tuple[int, int]
    output_resolutions: list[dict[Units, Resolution]]


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
            Format of output ("dict", "zarr", "qupath", "annotationstore").
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
        >>> output = segmentor.run(image_patches, patch_mode=True)

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
        num_workers: int = 0,
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
            num_workers (int):
                Number of workers for data loading. Default is 0.
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
            num_workers=num_workers,
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
        ioconfig: IOSegmentorConfig | None = None,
        *,
        patch_mode: bool = True,
        auto_get_mask: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return a DataLoader for inference.

        This method prepares the dataset and returns a PyTorch DataLoader
        for either patch-based or WSI-based semantic segmentation. It overrides
        the base method to support additional WSI-specific logic, including
        patch output shape and output location tracking.

        Args:
            images (str | Path | list[str | Path] | np.ndarray):
                Input images. Can be a list of file paths or a NumPy array
                of image patches in NHWC format.
            masks (Path | None):
                Optional tissue masks for WSI processing. Only used when
                `patch_mode` is False.
            labels (list | None):
                Optional labels for input images. Only one label per image is supported.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            auto_get_mask (bool):
                Whether to automatically generate a tissue mask using
                `wsireader.tissue_mask()` when `patch_mode` is False.
                If `True`, only tissue regions are processed. If `False`,
                all patches are processed. Default is `True`.

        Returns:
            torch.utils.data.DataLoader:
                A PyTorch DataLoader configured for semantic segmentation inference.

        """
        # Overwrite when patch_mode is False.
        if not patch_mode:
            dataset = WSIPatchDataset(
                input_img=images,
                mask_path=masks,
                patch_input_shape=ioconfig.patch_input_shape,
                patch_output_shape=ioconfig.patch_output_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
                auto_get_mask=auto_get_mask,
            )

            dataset.preproc_func = self._get_model_attr("preproc_func")
            self.output_locations = dataset.outputs

            # preprocessing must be defined with the dataset
            return torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_workers,
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
        the aggregated output. It supports memory-aware caching and optional
        inclusion of coordinates and labels.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader configured for WSI processing.
            save_path (Path):
                Path to save the intermediate output. The intermediate output
                is saved in a Zarr file.
            **kwargs (SemanticSegmentorRunParams):
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
            dict[str, dask.array.Array]:
                Dictionary containing merged prediction results:
                - "probabilities": Full-resolution probability map.
                - "coordinates": Patch coordinates.
                - "labels": Ground truth labels (if `return_labels` is True).

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
        tqdm_loop = tqdm(
            dataloader,
            leave=False,
            desc="Inferring patches",
            disable=not self.verbose,
        )

        canvas_np, output_locs_y_ = None, None
        canvas, count, output_locs = None, None, None
        canvas_zarr, count_zarr = None, None

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
            full_batch_output, full_output_locs, output_locs = prepare_full_batch(
                batch_output,
                batch_locs,
                full_output_locs,
                output_locs,
                canvas_np=canvas_np,
                save_path=save_path.with_name("full_batch_tmp"),
                memory_threshold=memory_threshold,
                is_last=(batch_idx == (len(dataloader) - 1)),
            )

            canvas_np = concatenate_none(old_arr=canvas_np, new_arr=full_batch_output)

            # Determine if dataloader is moved to next row of patches
            change_indices = np.where(np.diff(output_locs[:, 1]) != 0)[0] + 1

            # If a row of patches has been processed.
            if change_indices.size > 0:
                canvas, count, canvas_np, output_locs, output_locs_y_ = (
                    merge_horizontal(
                        canvas,
                        count,
                        output_locs_y_,
                        canvas_np,
                        output_locs,
                        change_indices,
                    )
                )

                vm = psutil.virtual_memory()
                used_percent = vm.percent
                # Use currently available memory (not the initial snapshot) to
                # decide when to spill intermediate results.
                canvas_used_percent = (canvas.nbytes / max(vm.available, 1)) * 100
                if (
                    used_percent > memory_threshold
                    or canvas_used_percent > memory_threshold
                ):
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
                    update_tqdm_desc(tqdm_loop=tqdm_loop, desc=msg)
                    # Flush data in Memory and clear dask graph
                    canvas_zarr, count_zarr = save_to_cache(
                        canvas,
                        count,
                        canvas_zarr,
                        count_zarr,
                        save_path=save_path,
                        verbose=self.verbose,
                    )
                    canvas, count = None, None
                    gc.collect()
                    update_tqdm_desc(tqdm_loop=tqdm_loop, desc="Inferring patches")

            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

        canvas, count, _, _, output_locs_y_ = merge_horizontal(
            canvas,
            count,
            output_locs_y_,
            canvas_np,
            output_locs,
            change_indices=[len(output_locs)],
        )

        zarr_group = None
        if canvas_zarr is not None:
            canvas_zarr, count_zarr = save_to_cache(
                canvas,
                count,
                canvas_zarr,
                count_zarr,
                verbose=self.verbose,
            )
            # Wrap zarr in dask array
            canvas = da.from_zarr(canvas_zarr, chunks=canvas_zarr.chunks)
            count = da.from_zarr(count_zarr, chunks=count_zarr.chunks)
            zarr_group = zarr.open(canvas_zarr.store.path, mode="a")

        output_shape = get_wsi_output_shape(dataloader.dataset)

        # Final vertical merge
        raw_predictions["probabilities"] = merge_vertical_chunkwise(
            canvas,
            count,
            output_locs_y_,
            zarr_group,
            save_path,
            memory_threshold,
            output_shape=output_shape,
            verbose=self.verbose,
        )
        raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)

        if save_path.with_name("full_batch_tmp").exists():
            shutil.rmtree(save_path.with_name("full_batch_tmp"))

        return raw_predictions

    def save_predictions(
        self: SemanticSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
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
                Desired output format: "dict", "zarr", "qupath" or "annotationstore".
            save_path (Path | None):
                Path to save the output file. Required for "zarr", "qupath"
                and "annotationstore".
            **kwargs (SemanticSegmentorRunParams):
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
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved Zarr file.
                - If output_type is "qupath": returns QuPath JSON
                  or path or list of paths to .json file.
                - If output_type is "annotationstore": returns AnnotationStore
                  or path or list of paths to .db file.

        """
        # Conversion to annotationstore uses a different function for SemanticSegmentor
        if output_type.lower() not in ["qupath", "annotationstore"]:
            return super().save_predictions(
                processed_predictions, output_type, save_path=save_path, **kwargs
            )

        return_probabilities = kwargs.get("return_probabilities", False)
        output_type_ = (
            "zarr"
            if is_zarr(save_path.with_suffix(".zarr")) or return_probabilities
            else "dict"
        )

        processed_predictions = super().save_predictions(
            processed_predictions,
            output_type=output_type_,
            save_path=save_path.with_suffix(".zarr"),
            **kwargs,
        )

        if isinstance(processed_predictions, Path):
            processed_predictions = zarr.open(str(processed_predictions), mode="r")

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict", self.model.class_dict)

        # Need to add support for zarr conversion.
        save_paths = []

        suffix = ".json" if output_type.lower() == "qupath" else ".db"
        msg = f"Saving predictions as f{output_type} in {suffix} format."
        logger.info(msg)
        if self.patch_mode:
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + suffix)
                else:
                    output_path = save_path.parent / (str(i) + suffix)

                out_file = dict_to_store_semantic_segmentor(
                    patch_output={"predictions": predictions},
                    scale_factor=scale_factor,
                    output_type=output_type,
                    class_dict=class_dict,
                    save_path=output_path,
                    verbose=self.verbose,
                )

                save_paths.append(out_file)
        else:
            out_file = dict_to_store_semantic_segmentor(
                patch_output=processed_predictions,
                scale_factor=scale_factor,
                output_type=output_type,
                class_dict=class_dict,
                save_path=save_path.with_suffix(suffix),
                verbose=self.verbose,
            )
            save_paths = out_file

        if return_probabilities:
            msg = (
                f"Probability maps cannot be saved as AnnotationStore or JSON. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {save_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)
        elif save_path.with_suffix(".zarr").exists():
            shutil.rmtree(save_path.with_suffix(".zarr"))

        return save_paths

    def _update_run_params(
        self: SemanticSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: tuple[int, int] | None = None,
        save_dir: os.PathLike | Path | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> Path | None:
        """Update runtime parameters for the SemanticSegmentor engine.

        This method sets internal attributes such as caching, batch size,
        IO configuration, and output format based on user input and keyword arguments.
        It also configures whether to include probabilities in the output.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution.
            output_type (str):
                Desired output format: "dict", "zarr", "qupath",
                or "annotationstore".
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            **kwargs (SemanticSegmentorRunParams):
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
            Path | None:
                Path to the save directory if applicable, otherwise None.

        Raises:
            ValueError:
                If `labels` are requested for WSI processing.

        """
        return_labels = kwargs.get("return_labels")

        if return_labels and not patch_mode:
            msg = "`return_labels` is not supported when `patch_mode` is False."
            raise ValueError(msg)

        return super()._update_run_params(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            save_dir=save_dir,
            ioconfig=ioconfig,
            overwrite=overwrite,
            patch_mode=patch_mode,
            output_type=output_type,
            **kwargs,
        )

    def run(
        self: SemanticSegmentor,
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
        **kwargs: Unpack[SemanticSegmentorRunParams],
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
                Desired output format: "dict", "zarr", "qupath",
                or "annotationstore". Default is "dict".
            **kwargs (SemanticSegmentorRunParams):
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


def concatenate_none(
    old_arr: np.ndarray | da.Array | None,
    new_arr: np.ndarray | da.Array,
) -> np.ndarray | da.Array:
    """Concatenate arrays, handling None values gracefully.

    This utility function concatenates `new_arr` to `old_arr` along the first axis.
    If `old_arr` is None, it returns `new_arr` directly. Supports both NumPy and Dask
    arrays.

    Args:
        old_arr (np.ndarray | da.Array | None):
            Existing array to append to. Can be None.
        new_arr (np.ndarray | da.Array):
            New array to append.

    Returns:
        np.ndarray | da.Array:
            Concatenated array of the same type as `new_arr`.

    """
    if isinstance(new_arr, np.ndarray):
        return (
            new_arr if old_arr is None else np.concatenate((old_arr, new_arr), axis=0)
        )

    return new_arr if old_arr is None else da.concatenate([old_arr, new_arr], axis=0)


def merge_batch_to_canvas(
    blocks: np.ndarray,
    output_locations: np.ndarray,
    merged_shape: tuple[int, int, int],
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

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - canvas: Merged prediction map of shape (H, W, C).
            - count: Count map indicating how many times each pixel was updated,
              shape (H, W).

    """
    # Ensure we operate on NumPy to avoid Dask out-parameter issues when merging.
    if not isinstance(blocks, np.ndarray):
        blocks = np.asarray(blocks)
    if not isinstance(output_locations, np.ndarray):
        output_locations = np.asarray(output_locations)

    canvas = np.zeros(merged_shape, dtype=blocks.dtype)
    count = np.zeros((*merged_shape[:2], 1), dtype=np.uint8)
    for i, block in enumerate(blocks):
        xs, ys, xe, ye = output_locations[i]
        if not np.any(block):
            continue
        # To deal with edge cases
        canvas[0 : ye - ys, xs:xe, :] += block[0 : ye - ys, 0 : xe - xs, :]
        count[0 : ye - ys, xs:xe, 0] += 1
    return canvas, count


def merge_horizontal(
    canvas: None | da.Array,
    count: None | da.Array,
    output_locs_y: np.ndarray,
    canvas_np: np.ndarray,
    output_locs: np.ndarray,
    change_indices: np.ndarray | list[int],
) -> tuple[da.Array, da.Array, np.ndarray, np.ndarray, np.ndarray]:
    """Merge horizontal patches incrementally for each row of patches.

    This function processes segments of NumPy patch arrays (`canvas_np`, `count_np`,
    `output_locs`) based on `change_indices`, merging them horizontally and appending
    the results to Dask arrays. It also updates the vertical output locations
    (`output_locs_y_`) for downstream vertical merging.

    Args:
        canvas (None | da.Array):
            Existing Dask array for canvas data, or None if uninitialized.
        count (None | da.Array):
            Existing Dask array for count data, or None if uninitialized.
        output_locs_y (np.ndarray):
            Array tracking vertical output locations for merged patches.
        canvas_np (np.ndarray):
            NumPy array of canvas patches to be merged.
        output_locs (np.ndarray):
            Array of output locations for each patch.
        change_indices (np.ndarray | list[np.ndarray]):
            Indices indicating where to flush and merge patches.

    Returns:
        tuple:
            Updated canvas and count Dask arrays, along with remaining canvas_np,
            count_np, output_locs, and output_locs_y_ arrays after processing.

    """
    start_idx = 0
    for c_idx in change_indices:
        output_locs_ = output_locs[: c_idx - start_idx]
        canvas_np_ = canvas_np[: c_idx - start_idx]

        # Compute span only for the current row to avoid allocating a canvas
        # covering the entire slide width.
        batch_xs = np.min(output_locs_[:, 0], axis=0)
        batch_xe = np.max(output_locs_[:, 2], axis=0)

        merged_shape = (canvas_np_.shape[1], batch_xe - batch_xs, canvas_np.shape[3])

        canvas_merge, count_merge = merge_batch_to_canvas(
            blocks=canvas_np_,
            output_locations=output_locs_,
            merged_shape=merged_shape,
        )

        canvas_merge = da.from_array(canvas_merge, chunks=canvas_merge.shape)
        count_merge = da.from_array(count_merge, chunks=count_merge.shape)

        canvas = concatenate_none(old_arr=canvas, new_arr=canvas_merge)
        count = concatenate_none(old_arr=count, new_arr=count_merge)

        output_locs_y = concatenate_none(
            old_arr=output_locs_y, new_arr=output_locs_[:, (1, 3)]
        )

        canvas_np = canvas_np[c_idx - start_idx :]
        output_locs = output_locs[c_idx - start_idx :]
        start_idx = c_idx

    return canvas, count, canvas_np, output_locs, output_locs_y


def save_to_cache(
    canvas: da.Array,
    count: da.Array,
    canvas_zarr: zarr.Array,
    count_zarr: zarr.Array,
    save_path: str | Path = "temp.zarr",
    zarr_dataset_name: tuple[str, str] = ("canvas", "count"),
    *,
    verbose: bool = True,
) -> tuple[zarr.Array, zarr.Array]:
    """Incrementally save computed canvas and count arrays to Zarr cache.

    This function computes the given Dask arrays (`canvas` and `count`)
    row-chunks one at a time to avoid materializing the full dask arrays
    in memory. If the datasets do not exist, they are created using the chunk
    shapes from the first block.

    Args:
        canvas (da.Array):
            Dask array representing image or feature data.
        count (da.Array):
            Dask array representing count or normalization data.
        canvas_zarr (zarr.Array):
            Existing Zarr dataset for canvas data. If None, a new one is created.
        count_zarr (zarr.Array):
            Existing Zarr dataset for count data. If None, a new one is created.
        save_path (str | Path):
            Path to the Zarr group for saving datasets. Defaults to "temp.zarr".
        zarr_dataset_name (tuple[str, str]):
            Tuple of name for zarr dataset to save canvas and count.
            Defaults to ("canvas", "count").
        verbose (bool):
            Whether to display progress bar.

    Returns:
        tuple[zarr.Array, zarr.Array]:
            Updated Zarr datasets for canvas and count arrays.
    """
    chunk0 = canvas.chunks[0][0]

    if canvas_zarr is None:
        zarr_group = zarr.open(str(save_path), mode="a")

        # Peek first block shapes to initialise datasets without computing all rows.
        # Blocks are 3D: (row_chunk, col_chunk, channel_chunk). Grab the first.
        first_canvas_block = canvas.blocks[0, 0, 0].compute()
        first_count_block = count.blocks[0, 0, 0].compute()

        canvas_zarr = zarr_group.create_dataset(
            name=zarr_dataset_name[0],
            # Append along axis 0 (height); keep width/channels fixed.
            shape=(0, *first_canvas_block.shape[1:]),
            chunks=(chunk0, *first_canvas_block.shape[1:]),
            dtype=first_canvas_block.dtype,
            overwrite=True,
        )

        count_zarr = zarr_group.create_dataset(
            name=zarr_dataset_name[1],
            shape=(0, *first_count_block.shape[1:]),
            dtype=first_count_block.dtype,
            chunks=(chunk0, *first_count_block.shape[1:]),
            overwrite=True,
        )

        # We already computed the first block; store it and start from the next.
        canvas_zarr.resize((first_canvas_block.shape[0], *canvas_zarr.shape[1:]))
        canvas_zarr[-first_canvas_block.shape[0] :] = first_canvas_block

        count_zarr.resize((first_count_block.shape[0], *count_zarr.shape[1:]))
        count_zarr[-first_count_block.shape[0] :] = first_count_block

        start_idx = 1
    else:
        start_idx = 0

    # Append remaining blocks one-at-a-time to limit peak memory.
    num_blocks = canvas.numblocks[0]
    tqdm_loop = tqdm(
        range(start_idx, num_blocks),
        leave=False,
        desc="Memory Overload, Spilling to disk",
        disable=not verbose,
    )
    for block_idx in tqdm_loop:
        canvas_block = canvas.blocks[block_idx, 0, 0].compute()
        count_block = count.blocks[block_idx, 0, 0].compute()

        canvas_zarr.resize(
            (canvas_zarr.shape[0] + canvas_block.shape[0], *canvas_zarr.shape[1:])
        )
        canvas_zarr[-canvas_block.shape[0] :] = canvas_block

        count_zarr.resize(
            (count_zarr.shape[0] + count_block.shape[0], *count_zarr.shape[1:])
        )
        count_zarr[-count_block.shape[0] :] = count_block

    return canvas_zarr, count_zarr


def get_wsi_output_shape(dataset: object) -> tuple[int, int] | None:
    """Return WSI output shape as (height, width) for the dataset if available."""
    wsi_shape = getattr(dataset, "wsi_shape", None)
    if wsi_shape is None:
        has_meta = all(
            hasattr(dataset, attr) for attr in ("img_path", "resolution", "units")
        )
        if has_meta:
            try:
                reader = getattr(dataset, "reader", None)
                if reader is None:
                    reader = WSIReader.open(dataset.img_path)
                wsi_shape = reader.slide_dimensions(
                    resolution=dataset.resolution, units=dataset.units
                )
            except (AttributeError, OSError, TypeError, ValueError):
                msg = "WSI output shape is not recognizable. Please verify outputs."
                logger.info(msg)
                return None
        else:
            msg = "No metadata found in dataset. Please verify outputs."
            logger.warning(msg)
            return None

    return int(wsi_shape[1]), int(wsi_shape[0])


def merge_vertical_chunkwise(
    canvas: da.Array,
    count: da.Array,
    output_locs_y_: np.ndarray,
    zarr_group: zarr.Group,
    save_path: Path,
    memory_threshold: int = 80,
    output_shape: tuple[int, int] | None = None,
    *,
    verbose: bool = True,
) -> da.Array:
    """Merge vertically chunked canvas and count arrays into a single probability map.

    This function processes vertically stacked image blocks (`canvas`) and their
    associated count arrays to compute normalized probabilities. It handles overlapping
    regions between chunks by applying seam folding and trimming halos to ensure smooth
    transitions. If a Zarr group is provided, the result is stored incrementally.

    Args:
        canvas (da.Array):
            Dask array containing image data split into vertical chunks.
        count (da.Array):
            Dask array containing count data corresponding to the canvas.
        output_locs_y_ (np.ndarray):
            Array of shape (N, 2) specifying vertical output locations
            for each chunk, used to compute overlaps.
        zarr_group (zarr.Group):
            Zarr group to store the merged probability dataset.
        save_path (Path):
            Path to save the intermediate output. The intermediate output
            is saved in a Zarr file.
        memory_threshold (int):
            Memory usage threshold (in percentage) to trigger caching behavior.
        output_shape (tuple[int, int] | None):
            Optional target output shape as (height, width). If provided,
            merged probabilities are clipped to this shape before being
            accumulated or written to Zarr.
        verbose (bool):
            Whether to display progress bar.

    Returns:
        da.Array:
            A merged Dask array of normalized probabilities, either loaded from Zarr
            or constructed in memory.

    """
    y0s, y1s = np.unique(output_locs_y_[:, 0]), np.unique(output_locs_y_[:, 1])
    overlaps = np.append(y1s[:-1] - y0s[1:], 0)

    num_chunks = canvas.numblocks[0]
    probabilities_zarr, probabilities_da = None, None
    chunk_shape = tuple(chunk[0] for chunk in canvas.chunks)
    written_height = 0

    tqdm_loop = tqdm(
        overlaps,
        leave=False,
        desc="Merging rows",
        disable=not verbose,
    )

    used_percent = 0

    curr_chunk = canvas.blocks[0, 0].compute()
    curr_count = count.blocks[0, 0].compute()
    next_chunk = canvas.blocks[1, 0].compute() if num_chunks > 1 else None
    next_count = count.blocks[1, 0].compute() if num_chunks > 1 else None

    probabilities = np.empty(0)

    for i, overlap in enumerate(tqdm_loop):
        if next_chunk is not None and overlap > 0:
            curr_chunk[-overlap:] += next_chunk[:overlap]
            curr_count[-overlap:] += next_count[:overlap]

        # Normalize
        curr_count = np.where(curr_count == 0, 1, curr_count)
        probabilities = curr_chunk / curr_count.astype(np.float32)

        probabilities, written_height, should_stop = clip_probabilities_to_shape(
            probabilities=probabilities,
            output_shape=output_shape,
            written_height=written_height,
        )
        if should_stop:
            break

        probabilities_zarr, probabilities_da = store_probabilities(
            probabilities=probabilities,
            chunk_shape=chunk_shape,
            probabilities_zarr=probabilities_zarr,
            probabilities_da=probabilities_da,
            zarr_group=zarr_group,
        )

        if probabilities_da is not None:
            vm = psutil.virtual_memory()
            used_percent = (probabilities_da.nbytes / vm.free) * 100
        if probabilities_zarr is None and used_percent > memory_threshold:
            desc = tqdm_loop.desc if hasattr(tqdm_loop, "desc") else ""
            msg = (
                f"Current Memory usage: {used_percent} %  "
                f"exceeds specified threshold: {memory_threshold}. "
                f"Saving intermediate results to disk."
            )
            update_tqdm_desc(tqdm_loop=tqdm_loop, desc=msg)
            zarr_group = zarr.open(str(save_path), mode="a")
            probabilities_zarr = zarr_group.create_dataset(
                name="probabilities",
                shape=probabilities_da.shape,
                chunks=(chunk_shape[0], *probabilities.shape[1:]),
                dtype=probabilities.dtype,
                overwrite=True,
            )
            probabilities_zarr[:] = probabilities_da.compute()

            probabilities_da = None
            update_tqdm_desc(tqdm_loop=tqdm_loop, desc=desc)

        if next_chunk is not None:
            curr_chunk, curr_count = next_chunk[overlap:], next_count[overlap:]

        if i + 2 < num_chunks:
            next_chunk = canvas.blocks[i + 2, 0].compute()
            next_count = count.blocks[i + 2, 0].compute()
        else:
            next_chunk, next_count = None, None

    if probabilities_zarr:
        return _get_probabilities_da_from_zarr(
            zarr_group=zarr_group,
            probabilities_zarr=probabilities_zarr,
            chunk_shape=chunk_shape,
            probabilities=probabilities,
        )

    return probabilities_da


def clip_probabilities_to_shape(
    probabilities: np.ndarray,
    output_shape: tuple[int, int] | None,
    written_height: int,
) -> tuple[np.ndarray, int, bool]:
    """Clip probability chunk to target output shape and track written height."""
    if output_shape is None:
        return probabilities, written_height, False

    target_height, target_width = map(int, output_shape)
    remaining_height = target_height - written_height
    if remaining_height <= 0:
        return probabilities[:0], written_height, True

    clipped = probabilities[:remaining_height, :target_width, ...]
    if clipped.shape[0] == 0:
        return clipped, written_height, True

    return clipped, written_height + clipped.shape[0], False


def _get_probabilities_da_from_zarr(
    zarr_group: zarr.Group,
    probabilities_zarr: zarr.Array,
    chunk_shape: tuple,
    probabilities: zarr.Array | np.ndarray,
) -> da.Array:
    """Helper function to return dask array after probabilities have been merged."""
    if "canvas" in zarr_group:
        del zarr_group["canvas"]
    if "count" in zarr_group:
        del zarr_group["count"]
    return da.from_zarr(
        probabilities_zarr, chunks=(chunk_shape[0], *probabilities.shape[1:])
    )


def store_probabilities(
    probabilities: np.ndarray,
    chunk_shape: tuple[int, ...],
    probabilities_zarr: zarr.Array | None,
    probabilities_da: da.Array | None,
    zarr_group: zarr.Group | None,
    name: str = "probabilities",
) -> tuple[zarr.Array | None, da.Array | None]:
    """Store computed probability data into a Zarr dataset or accumulate in memory.

    If a Zarr group is provided, the function appends the given probability array
    to the 'probabilities' dataset, resizing as needed. Otherwise, it concatenates
    the array into an existing Dask array for in-memory accumulation.

    Args:
        probabilities (np.ndarray):
            Computed probability array to store.
        chunk_shape (tuple[int, ...]):
            Chunk shape used for Zarr dataset creation.
        probabilities_zarr (zarr.Array | None):
            Existing Zarr dataset, or None to initialize.
        probabilities_da (da.Array | None):
            Existing Dask array for in-memory accumulation.
        zarr_group (zarr.Group | None):
            Zarr group used to create or access the dataset.
        name (str):
            Name to create Zarr dataset.

    Returns:
        tuple[zarr.Array | None, da.Array | None]:
            Updated Zarr dataset and/or Dask array.

    """
    if zarr_group is not None:
        if probabilities_zarr is None:
            probabilities_zarr = zarr_group.create_dataset(
                name=name,
                shape=(0, *probabilities.shape[1:]),
                chunks=(chunk_shape[0], *probabilities.shape[1:]),
                dtype=probabilities.dtype,
            )

        probabilities_zarr.resize(
            (
                probabilities_zarr.shape[0] + probabilities.shape[0],
                *probabilities_zarr.shape[1:],
            )
        )
        probabilities_zarr[-probabilities.shape[0] :] = probabilities
    else:
        probabilities_da = concatenate_none(
            old_arr=probabilities_da,
            new_arr=da.from_array(
                probabilities, chunks=(chunk_shape[0], *probabilities.shape[1:])
            ),
        )

    return probabilities_zarr, probabilities_da


def prepare_full_batch(
    batch_output: np.ndarray,
    batch_locs: np.ndarray,
    full_output_locs: np.ndarray,
    output_locs: np.ndarray,
    canvas_np: np.ndarray | zarr.Array | None = None,
    save_path: Path | str = "temp_fullbatch",
    memory_threshold: int = 80,
    *,
    is_last: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        canvas_np (np.ndarray | zarr.Array | None):
            Accumulated canvas array from previous batches. Used to check
            total memory footprint when deciding numpy vs zarr.
        save_path (Path | str):
            Path to a directory; a unique temp subfolder will be created within it
            to store the temporary full-batch zarr for this batch.
        memory_threshold (int):
            Memory usage threshold (in percentage) to trigger caching behavior.
        is_last (bool):
            Flag indicating whether this is the final batch.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - full_batch_output: Full-sized output array with predictions placed.
            - full_output_locs: Updated remaining global output locations.
            - output_locs: Updated accumulated output locations.

    """
    # Map batch locations back to indices in the full output grid.
    # Use a dict to avoid allocating a huge dense array when locations are sparse.
    full_output_dict = {tuple(row): i for i, row in enumerate(full_output_locs)}
    matches = np.array([full_output_dict[tuple(row)] for row in batch_locs])

    total_size = int(np.max(matches).astype(np.uint32)) + 1
    sample_shape = batch_output.shape[1:]

    # Calculate final size including potential padding
    final_size = total_size
    if is_last and len(full_output_locs):
        final_size += len(full_output_locs)

    # Check if array will fit in available memory
    # Consider BOTH: new array size AND accumulated canvas_np size
    array_bytes = final_size * np.prod(sample_shape) * batch_output.dtype.itemsize
    canvas_bytes = canvas_np.nbytes if canvas_np is not None else 0
    total_bytes = array_bytes + canvas_bytes

    vm = psutil.virtual_memory()
    # During concatenation, we temporarily need:
    # - existing canvas_np (canvas_bytes)
    # - new full_batch_output (array_bytes)
    # - concatenated result (canvas_bytes + array_bytes)
    # Total peak = 2 * (canvas_bytes + array_bytes)
    peak_bytes = 2 * total_bytes
    memory_available = vm.available * (memory_threshold / 100)

    use_numpy = peak_bytes < memory_available

    if use_numpy:
        # Array fits safely in RAM, use numpy for better performance
        full_batch_output = np.zeros(
            shape=(final_size, *sample_shape),
            dtype=batch_output.dtype,
        )
    else:
        save_path_dir = Path(save_path)
        save_path_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(prefix="full_batch_tmp_", dir=str(save_path_dir))
        )

        store = zarr.DirectoryStore(str(temp_dir))
        full_batch_output = zarr.zeros(
            shape=(total_size, *sample_shape),
            chunks=(len(batch_output), *sample_shape),
            dtype=batch_output.dtype,
            store=store,
            overwrite=True,
        )

    # Place matching outputs using matching indices
    full_batch_output[matches] = batch_output

    output_locs = concatenate_none(
        old_arr=output_locs, new_arr=full_output_locs[:total_size]
    )
    full_output_locs = full_output_locs[total_size:]

    if is_last and len(full_output_locs):
        pad_len = len(full_output_locs)
        if not use_numpy:
            # Resize zarr array to accommodate padding
            full_batch_output.resize(total_size + pad_len, *sample_shape)
        # For numpy, array is already pre-allocated to final_size
        full_batch_output[-pad_len:] = 0

        output_locs = concatenate_none(old_arr=output_locs, new_arr=full_output_locs)
        full_output_locs = np.empty(
            (0, batch_locs.shape[1]), dtype=full_output_locs.dtype
        )

    return full_batch_output, full_output_locs, output_locs
