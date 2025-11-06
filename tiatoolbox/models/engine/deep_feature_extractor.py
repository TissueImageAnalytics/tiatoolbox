"""Define DeepFeatureExtractor class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
from typing_extensions import Unpack

from tiatoolbox.models.dataset.dataset_abc import WSIStreamDataset
from tiatoolbox.utils.misc import get_tqdm

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.wsicore import WSIReader


class DeepFeatureExtractor(SemanticSegmentor):
    """Generic CNN Feature Extractor."""

    def __init__(
        self: DeepFeatureExtractor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        dataset_class: Callable = WSIStreamDataset,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`DeepFeatureExtractor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )
        self.process_prediction_per_batch = False
        self.dataset_class = dataset_class

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
                Additional runtime parameters, including:
                - return_probabilities (bool): Whether to return probability maps.
                - return_labels (bool): Whether to include labels in the output.
                - memory_threshold (int): Memory usage threshold to trigger disk
                  caching.

        Returns:
            dict[str, dask.array.Array]:
                Dictionary containing merged prediction results:
                - "probabilities": Full-resolution probability map.
                - "coordinates": Patch coordinates.
                - "labels": Ground truth labels (if `return_labels` is True).

        """
        _ = kwargs.get("patch_mode", False)
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

        raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)
        raw_predictions["probabilities"] = da.concatenate(probabilities, axis=0)

        return raw_predictions

    def post_process_patches(
        self: DeepFeatureExtractor,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[DeepFeatureExtractor],
    ) -> da.Array:
        """Post-process raw patch predictions from model inference.

        This method applies the model's post-processing function to the raw predictions
        obtained from `infer_patches()`. The output is wrapped in a Dask array for
        efficient computation and memory handling.

        Args:
            raw_predictions (da.Array | np.ndarray):
                Raw model predictions.
            prediction_shape (tuple[int, ...]):
                Expected shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (PredictorRunParams):
                Additional runtime parameters, including `return_probabilities`.

        Returns:
            dask.array.Array: Post-processed predictions as a Dask array.

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
        """Save patch predictions to disk."""
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
        """Update runtime parameters for the PatchPredictor engine.

        This method sets internal attributes such as caching, batch size,
        IO configuration, and output format based on user input and keyword arguments.
        It also configures whether to include probabilities in the output.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing.
            labels (list | None):
                Optional labels for input images.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
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
                If `labels` are requested for WSI processing.

        """
        if output_type != "zarr":
            msg = "Only zarr output is supported for `DeepFeatureExtractor`."
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
        """Run the DeepFeatureExtractor engine on input images."""
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
