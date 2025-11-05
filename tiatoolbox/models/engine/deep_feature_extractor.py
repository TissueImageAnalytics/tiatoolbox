"""Define DeepFeatureExtractor class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Unpack

from tiatoolbox.models.dataset.dataset_abc import WSIStreamDataset

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from collections.abc import Callable
    from pathlib import Path

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

    def _process_predictions(
        self: DeepFeatureExtractor,
        cum_batch_predictions: list,
        wsi_reader: WSIReader,  # skipcq: PYL-W0613  # noqa: ARG002
        ioconfig: IOSegmentorConfig,
        save_path: str,
        cache_dir: str,  # skipcq: PYL-W0613  # noqa: ARG002
    ) -> None:
        """Define how the aggregated predictions are processed.

        This includes merging the prediction if necessary and also
        saving afterward.

        Args:
            cum_batch_predictions (list):
                List of batch predictions. Each item within the list
                should be of (location, patch_predictions).
            wsi_reader (:class:`WSIReader`):
                A reader for the image where the predictions come from.
                Not used here. Added for consistency with the API.
            ioconfig (:class:`IOSegmentorConfig`):
                A configuration object contains input and output
                information.
            save_path (str):
                Root path to save current WSI predictions.
            cache_dir (str):
                Root path to cache current WSI data.
                Not used here. Added for consistency with the API.

        """
        # assume prediction_list is N, each item has L output elements
        location_list, prediction_list = list(zip(*cum_batch_predictions, strict=False))
        # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of output
        # patch, this can exceed the image bound at the requested resolution
        # remove singleton due to split.
        location_list = np.array([v[0] for v in location_list])
        np.save(f"{save_path}.position.npy", location_list)
        for idx, _ in enumerate(ioconfig.output_resolutions):
            # assume resolution idx to be in the same order as L
            # 0 idx is to remove singleton without removing other axes singleton
            prediction_list = [v[idx][0] for v in prediction_list]
            prediction_list = np.array(prediction_list)
            np.save(f"{save_path}.features.{idx}.npy", prediction_list)

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
