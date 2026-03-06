"""This module enables interactive segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.utils.misc import dict_to_store_semantic_segmentor

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from tiatoolbox.type_hints import IntPair


class PromptSegmentor:
    """Engine for prompt-based segmentation of WSIs.

    This class is designed to work with the SAM model architecture.
    It allows for interactive segmentation by providing point and bounding box
    coordinates as prompts. The model is intended to be used with image tiles
    selected interactively in some way and provided as np.arrays. At least
    one of either point_coords or box_coords must be provided to guide
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
        model = SAM() if model is None else model
        self.model = model
        self.scale = 1.0
        self.offset = np.array([0, 0])

    def run(  # skipcq: PYL-W0221
        self,
        images: list,
        point_coords: np.ndarray | None = None,
        box_coords: np.ndarray | None = None,
        save_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> list[Path]:
        """Run inference on image patches with prompts.

        Args:
            images (list):
                List of image patch arrays to run inference on.
            point_coords (np.ndarray):
                N_im x N_points x 2 array of point coordinates for each image patch.
            box_coords (np.ndarray):
                N_im x N_boxes x 4 array of bounding box coordinates for each
                image patch.
            save_dir (str or Path):
                Directory to save the output databases.
            device (str):
                Device to run inference on.

        Returns:
            list[Path]:
                Paths to the saved output databases.

        """
        paths = []
        masks, _ = self.model.infer_batch(
            self.model,
            images,
            point_coords=point_coords,
            box_coords=box_coords,
            device=device,
        )
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, _mask in enumerate(masks):
            mask = np.any(_mask[0], axis=0, keepdims=False)
            dict_to_store_semantic_segmentor(
                patch_output={"predictions": mask[0]},
                scale_factor=(self.scale, self.scale),
                offset=self.offset,
                save_path=Path(f"{save_dir}/{i}.db"),
                output_type="annotationstore",
                ignore_index=0,
            )
            paths.append(Path(f"{save_dir}/{i}.db"))
        return paths

    def calc_mpp(
        self, area_dims: IntPair, base_mpp: float, fixed_size: int = 1500
    ) -> tuple[float, float]:
        """Calculates the microns per pixel for a fixed area of an image.

        Args:
            area_dims (tuple):
                Dimensions of the area to be scaled.
            base_mpp (float):
                Microns per pixel of the base image.
            fixed_size (int):
                Fixed size of the area.

        Returns:
            tuple[float, float]:
                Tuple of the scaled mpp and the scale factor.
        """
        scale = max(area_dims) / fixed_size if max(area_dims) > fixed_size else 1.0
        self.scale = scale
        return base_mpp * scale, scale
