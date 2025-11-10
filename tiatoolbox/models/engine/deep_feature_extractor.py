"""Define DeepFeatureExtractor class."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import dask.array as da
import psutil
import zarr
from dask import compute
from typing_extensions import Unpack

from tiatoolbox.utils.misc import get_tqdm

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import numpy as np
    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.wsicore import WSIReader


def save_to_cache(
    probabilities: list[da.Array],
    coordinates: list[da.Array],
    probabilities_zarr: zarr.Array,
    coordinates_zarr: zarr.Array,
    save_path: str | Path = "temp.zarr",
) -> tuple[zarr.Array, zarr.Array]:
    """Save to cache."""
    if len(probabilities) == 0:
        return probabilities_zarr, coordinates_zarr

    coordinates = da.concatenate(coordinates, axis=0)
    probabilities = da.concatenate(probabilities, axis=0)

    computed_values = compute(*[probabilities, coordinates])
    probabilities_computed, coordinates_computed = computed_values

    chunk_shape = tuple(chunk[0] for chunk in probabilities.chunks)
    if probabilities_zarr is None:
        zarr_group = zarr.open(str(save_path), mode="w")

        probabilities_zarr = zarr_group.create_dataset(
            name="canvas",
            shape=(0, *probabilities_computed.shape[1:]),
            chunks=(chunk_shape[0], *probabilities_computed.shape[1:]),
            dtype=probabilities_computed.dtype,
            overwrite=True,
        )

        coordinates_zarr = zarr_group.create_dataset(
            name="count",
            shape=(0, *coordinates_computed.shape[1:]),
            dtype=coordinates_computed.dtype,
            chunks=(chunk_shape[0], *coordinates_computed.shape[1:]),
            overwrite=True,
        )

    probabilities_zarr.resize(
        (
            probabilities_zarr.shape[0] + probabilities_computed.shape[0],
            *probabilities_zarr.shape[1:],
        )
    )
    probabilities_zarr[-probabilities_computed.shape[0] :] = probabilities_computed

    coordinates_zarr.resize(
        (
            coordinates_zarr.shape[0] + coordinates_computed.shape[0],
            *coordinates_zarr.shape[1:],
        )
    )
    coordinates_zarr[-coordinates_computed.shape[0] :] = coordinates_computed

    return probabilities_zarr, coordinates_zarr


class DeepFeatureExtractor(SemanticSegmentor):
    r"""Generic CNN-based feature extractor for digital pathology images.

    This class extends :class:`SemanticSegmentor` to extract deep features from
    whole slide images (WSIs) or image patches using a CNN model. It is designed
    for use cases where the goal is to obtain intermediate feature representations
    (e.g., embeddings) rather than final classification or segmentation outputs.

    The extracted features are returned or saved in Zarr format for downstream
    analysis, such as clustering, visualization, or training other machine learning
    models.

    Args:
        model (str | ModelABC):
            A PyTorch model instance or the name of a pretrained model from TIAToolbox.
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

    Attributes:
        process_prediction_per_batch (bool):
            Flag to control whether predictions are processed per batch.
            Default is False.

    """

    def __init__(
        self: DeepFeatureExtractor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`DeepFeatureExtractor`.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or the name of a pretrained model from
                TIAToolbox. If a string is provided, the corresponding pretrained
                weights will be downloaded unless overridden via `weights`.
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
        self.process_prediction_per_batch = False

    def infer_wsi(
        self: SemanticSegmentor,
        dataloader: DataLoader,
        save_path: Path,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict[str, da.Array]:
        """Perform model inference on a whole slide image (WSI).

        This method processes a WSI using the provided DataLoader and extracts
        deep features from each patch using the model. The extracted features
        are returned as a Dask array along with the corresponding patch coordinates.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader configured for WSI processing.
            save_path (Path):
                Path to save the intermediate output. (Unused in this implementation.)
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters, including:
                - return_probabilities (bool): Whether to return feature maps.
                - memory_threshold (int): Memory usage threshold for caching.

        Returns:
            dict[str, dask.array.Array]:
                Dictionary containing:
                - "probabilities": Extracted feature maps from the model.
                - "coordinates": Patch coordinates corresponding to the features.

        """
        # Default Memory threshold percentage is 80.
        memory_threshold = kwargs.get("memory_threshold", 80)
        vm = psutil.virtual_memory()
        _ = save_path
        keys = ["probabilities", "coordinates"]
        probabilities, coordinates = [], []

        # Main output dictionary
        raw_predictions = dict(
            zip(keys, [da.empty(shape=(0, 0))] * len(keys), strict=False)
        )

        # Inference loop
        tqdm = get_tqdm()
        tqdm_loop = (
            tqdm(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else dataloader
        )

        probabilities_zarr, coordinates_zarr = None, None

        probabilities_used_percent = 0
        for batch_data in tqdm_loop:
            batch_output = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            probabilities.append(da.from_array(batch_output[0]))
            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

            used_percent = vm.percent
            probabilities_used_percent = (
                probabilities_used_percent + (probabilities[-1].nbytes / vm.free) * 100
            )
            if (
                used_percent > memory_threshold
                or probabilities_used_percent > memory_threshold
            ):
                tqdm_loop.desc = "Spill intermediate data to disk"
                used_percent = (
                    probabilities_used_percent
                    if (probabilities_used_percent > memory_threshold)
                    else used_percent
                )
                msg = (
                    f"Current Memory usage: {used_percent} %  "
                    f"exceeds specified threshold: {memory_threshold}. "
                    f"Saving intermediate results to disk."
                )

                tqdm.write(msg)
                # Flush data in Memory and clear dask graph
                probabilities_zarr, coordinates_zarr = save_to_cache(
                    probabilities,
                    coordinates,
                    probabilities_zarr,
                    coordinates_zarr,
                    save_path=save_path,
                )

                probabilities, coordinates = [], []
                probabilities_used_percent = 0
                gc.collect()
                tqdm_loop.desc = "Inferring patches"

        if probabilities_zarr is not None:
            probabilities_zarr, coordinates_zarr = save_to_cache(
                probabilities,
                coordinates,
                probabilities_zarr,
                coordinates_zarr,
                save_path=save_path,
            )
            # Wrap zarr in dask array
            raw_predictions["probabilities"] = da.from_zarr(
                probabilities_zarr, chunks=probabilities_zarr.chunks
            )
            raw_predictions["coordinates"] = da.from_zarr(
                coordinates_zarr, chunks=coordinates_zarr.chunks
            )
        else:
            raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)
            raw_predictions["probabilities"] = da.concatenate(probabilities, axis=0)

        return raw_predictions

    def post_process_patches(
        self: DeepFeatureExtractor,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> da.Array:
        """Post-process raw patch predictions from model inference.

        This method overrides the base implementation to return raw feature maps
        without applying any additional processing. It is intended for use cases
        where intermediate CNN features are required as output.

        Args:
            raw_predictions (dask.array.Array):
                Raw model predictions as a Dask array.
            prediction_shape (tuple[int, ...]):
                Expected shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters.

        Returns:
            dask.array.Array:
                Unmodified raw predictions.

        """
        _ = kwargs.get("return_probabilities")
        _ = prediction_shape
        _ = prediction_dtype

        return raw_predictions

    def save_predictions(
        self: SemanticSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | Path:
        """Save patch-level feature predictions to disk or return them in memory.

        This method saves the extracted deep features in the specified output format.
        Only the "zarr" format is supported for this engine. The method disables
        saving the "predictions" key, as it is not relevant for feature extraction.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model outputs.
            output_type (str):
                Desired output format. Must be "zarr".
            save_path (Path | None):
                Path to save the output file. Required for "zarr" format.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters, including:
                - output_file (str): Name of the output file.
                - scale_factor (tuple[float, float]): For coordinate transformation.
                - class_dict (dict): Optional class index-to-name mapping.

        Returns:
            dict | Path:
                - If `output_type` is "zarr": returns the path to the saved Zarr file.
                - If `output_type` is "dict": returns predictions as a dictionary.

        Raises:
            ValueError:
                If an unsupported output format is provided.

        """
        # no need to compute predictions
        self.drop_keys.append("predictions")
        return super().save_predictions(
            processed_predictions, output_type, save_path=save_path, **kwargs
        )

    def _update_run_params(
        self: SemanticSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        labels: list | None = None,
        save_dir: os.PathLike | Path | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> Path | None:
        """Update runtime parameters for the DeepFeatureExtractor engine.

        This method sets internal attributes such as caching, batch size,
        IO configuration, and output format based on user input and keyword arguments.
        It also validates that the output format is supported.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing.
            labels (list | None):
                Optional labels for input images.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            output_type (str):
                Desired output format. Must be "zarr".
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters.

        Returns:
            Path | None:
                Path to the save directory if applicable, otherwise None.

        Raises:
            ValueError:
                If `output_type` is not "zarr", which is the only supported format.

        """
        if output_type not in ["zarr", "dict"]:
            msg = (
                f"output_type: `{output_type}` is not supported for "
                f"`DeepFeatureExtractor` engine."
            )
            raise ValueError(msg)

        return super()._update_run_params(
            images=images,
            masks=masks,
            labels=labels,
            save_dir=save_dir,
            ioconfig=ioconfig,
            overwrite=overwrite,
            patch_mode=patch_mode,
            output_type=output_type,
            **kwargs,
        )

    def run(
        self: DeepFeatureExtractor,
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
        """Run the DeepFeatureExtractor engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, and saving of extracted deep features. It supports both
        patch-level and whole slide image (WSI) modes. The output is returned or saved
        in Zarr format.

        Note:
            The `return_probabilities` flag is always set to True for this engine,
            as it is designed to extract intermediate feature maps.

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
                Desired output format. Must be "zarr" or "dict".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to update engine attributes.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Raises:
            ValueError:
                If `output_type` is not "zarr".
        """
        # return_probabilities is always True for FeatureExtractor.
        kwargs["return_probabilities"] = True

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
