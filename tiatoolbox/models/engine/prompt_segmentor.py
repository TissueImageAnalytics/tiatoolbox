"""This module enables interactive segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.models_abc import model_to
from tiatoolbox.utils.misc import dict_to_store_semantic_segmentor

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntBounds, IntPair


class PromptSegmentor:
    """Engine for prompt-based segmentation of WSIs.

    This class is designed to work with the SAM model architecture.
    It allows for interactive segmentation by providing point and bounding box
    coordinates as prompts. The model can be used in both tile and WSI modes,
    where tile mode processes individual image patches and WSI mode processes
    whole-slide images. The class also supports multi-prompt segmentation,
    where multiple point and bounding box coordinates can be provided for
    segmentation.

    Args:
        model (SAM):
            Model architecture to use. If None, defaults to SAM.

    """

    def __init__(
        self,
        model: torch.nn.Module = None,
    ) -> None:
        """Initializes the PromptSegmentor."""
        if model is None:
            model = SAM()
        self.model = model

    def predict(  # skipcq: PYL-W0221
        self,
        imgs: list,
        point_coords: list[list[IntPair]] | None = None,
        box_coords: list[list[IntBounds]] | None = None,
        save_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> list[tuple[Path, Path]]:
        # use external for testing
        self._device = device
        self._model = model_to(model=self.model, device=device)
        sample_outputs = self.model.infer_batch(
            self.model,
            torch.tensor(imgs[0]).unsqueeze(0),
            point_coords=point_coords,
            box_coords=box_coords,
            device=self._device,
        )
        save_path = save_dir / f"{0}"
        mask = np.any(sample_outputs[0][0][0], axis=0, keepdims=False)
        dict_to_store_semantic_segmentor(
            patch_output={"predictions": mask[0]},
            scale_factor=self.scale,
            offset=self.offset,
            save_path=Path(f"{save_path}.{0}.db"),
        )
        return Path(f"{save_path}.{0}.db")

    def calc_mpp(
        self, area_dims: IntPair, base_mpp: float, fixed_size: int = 1500
    ) -> float:
        """Calculates the microns per pixel for a fixed area of an image.

        Args:
            area_dims (tuple):
                Dimensions of the area to be scaled.
            base_mpp (float):
                Microns per pixel of the base image.
            fixed_size (int):
                Fixed size of the area.

        Returns:
            float:
                Microns per pixel required to scale the area to a fixed size.
        """
        scale = max(area_dims) / fixed_size if max(area_dims) > fixed_size else 1.0
        self.scale = scale
        return base_mpp * scale, scale
