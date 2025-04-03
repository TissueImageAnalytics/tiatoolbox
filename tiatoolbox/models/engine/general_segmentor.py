"""Model designed for general segmentation of WSIs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from shapely import Polygon

from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.architecture.sam import SAM, SAMPrompts
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import Callable, IntBounds, IntPair


# TODO: Remove
def _prepare_save_output(
    save_path: str | Path,
    mask_shape: tuple[int, ...],
    scores_shape: tuple[int, ...],
) -> tuple:
    """Prepares for saving the cached output."""
    if save_path is not None:
        save_path = Path(save_path)
        mask_memmap = np.lib.format.open_memmap(
            save_path / "0.npy",
            mode="w+",
            shape=mask_shape,
            dtype=np.uint8,
        )
        score_memmap = np.lib.format.open_memmap(
            save_path / "1.npy",
            mode="w+",
            shape=scores_shape,
            dtype=np.float32,
        )

    return mask_memmap, score_memmap


class GeneralSegmentor(SemanticSegmentor):
    """Model designed for general segmentation of WSIs.

    Uses the SAM2 model architecture.
    """

    def __init__(
        self,
        model: SAM = None,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        dataset_class: Callable = WSIStreamDataset,
    ) -> None:
        """Initializes the GeneralSegmentor."""
        super().__init__(
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            model=model,
            dataset_class=dataset_class,
        )

    @staticmethod
    def calc_mpp(area_dims: IntPair, base_mpp: float, fixed_size: int = 1500) -> float:
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
        return base_mpp * scale

    def _load_wsi(
        self,
        file_name: str | Path,
        bounds: IntBounds | None = None,
        resolution: float = 1.0,
        units: str = "mpp",
    ) -> np.ndarray:
        """Load WSI and return image data."""
        self.reader = WSIReader.open(file_name)
        self.slide_dims = self.reader.slide_dimensions(1.0, "baseline")
        if bounds is not None:
            self.img = self.reader.read_bounds(bounds, resolution, units)
            if units == "mpp":
                base_mpp = self.reader.info["mpp"]
                self.scale_factor = base_mpp / resolution
        else:
            self.img = self.reader.slide_thumbnail(resolution, units)
        return self.img

    def _bound_prompts(self, prompts: SAMPrompts, bounds: IntBounds) -> SAMPrompts:
        """Bound the prompts to the region of interest."""
        if prompts is not None and bounds is not None:
            if prompts.point_coords is not None:
                prompts.point_coords = (
                    np.array(prompts.point_coords) - np.array(bounds[:2])
                ) * np.array(self.scale_factor)

            if prompts.box_coords is not None:
                new_box_coords = []
                for left, top, right, bottom in prompts.box_coords:
                    new_left, new_top = (
                        np.array([left, top]) - np.array(bounds[:2])
                    ) * np.array(self.scale_factor)
                    new_right, new_bottom = (
                        np.array([right, bottom]) - np.array(bounds[:2])
                    ) * np.array(self.scale_factor)

                    new_box_coords.append([new_left, new_top, new_right, new_bottom])
                prompts.box_coords = np.array(new_box_coords)
        return prompts

    def _unbound_masks(self, masks: list[np.ndarray], bounds: IntBounds) -> np.ndarray:
        """Unbound the masks to the original image size."""
        new_masks = []

        for mask in masks:
            new_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])

            # Resizes the mask into the box at base resolution
            resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

            new_mask = np.zeros(np.array(self.slide_dims)[::-1], dtype=np.uint8)

            # Stores mask into base resolution whole image
            new_mask[bounds[1] : bounds[3], bounds[0] : bounds[2]] = resized_mask
            new_masks.append(new_mask)
        return np.array(new_masks, dtype=np.uint8)

    def predict(
        self,
        file_name: str | Path,
        prompts: SAMPrompts | None = None,
        device: str = "cpu",
        save_path: str | Path | None = None,
        bounds: IntBounds | None = None,
        resolution: float = 1.0,
        units: str = "baseline",
    ) -> list[tuple[Path, Path, Path]]:
        """Predict on a WSI using prompts.

        Args:
            file_name (str):
                Path to WSI file.
            prompts (SAMPrompts):
                Prompts for SAM model.
            device (str):
                Device to run inference on.
            save_path (str):
                Location to save output prediction.
            bounds (tuple):
                Bounds for the region of interest.
            resolution (float):
                Desired resolution of the image.
            units (str):
                Units of the resolution.

        Returns:
            list:
                List of paths to the saved output prediction.
        """
        file_name = Path(file_name)
        save_dir = self._prepare_save_dir(save_dir=save_path)

        batch_data = self._load_wsi(
            file_name, resolution=resolution, units=units, bounds=bounds
        )
        prompts = self._bound_prompts(prompts, bounds=bounds)

        masks, scores = self.model.infer_batch(
            model=self.model, batch_data=batch_data, prompts=prompts, device=device
        )

        if bounds is not None:
            masks = self._unbound_masks(masks, bounds)

        mask_memmap, score_memmap = _prepare_save_output(
            save_path=save_dir, mask_shape=masks.shape, scores_shape=scores.shape
        )
        np.copyto(mask_memmap, masks)
        np.copyto(score_memmap, scores)

        self._outputs = [
            [str(file_name), str(save_dir / "0.npy"), str(save_dir / "1.npy")]
        ]

        # ? will this corrupt old version if control + c midway?
        map_file_path = save_dir / "file_map.dat"
        # backup old version first
        if Path.exists(map_file_path):
            old_map_file_path = save_dir / "file_map_old.dat"
            shutil.copy(map_file_path, old_map_file_path)
        joblib.dump(self._outputs, map_file_path)

        return self._outputs

    @staticmethod
    def to_annotation(
        mask_path: str | Path,
        score_path: str | Path,
        save_filename: str | Path | None = None,
    ) -> Path:
        """Converts the prediction output to annotation format."""
        masks = np.load(mask_path)
        scores = np.load(score_path)

        # Define annotation store path
        store_path = save_filename.with_suffix(".db")
        store = SQLiteStore()

        def mask_to_polygons(mask):
            """Extract polygons from a binary mask using OpenCV."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Avoid single-point contours
            polygons = [Polygon(c.squeeze()) for c in contours if len(c) > 2]
            return polygons

        for i, mask in enumerate(masks):
            polygons = mask_to_polygons(mask)
            # Add extracted polygons to the annotation store
            props = {"score": f"{scores[i]}", "type": f"Mask {i + 1}"}
            for poly in polygons:
                annotation = Annotation(geometry=poly, properties=props)
                store.append(annotation)

        store.create_index("id", '"id"')

        store.commit()
        store.dump(store_path)
        store.close()
        return store_path

    @staticmethod
    def create_prompts(
        point_coords: list[IntPair] | None = None,
        point_labels: list[int] | None = None,
        box_coords: list[IntBounds] | None = None,
    ) -> SAMPrompts:
        """Create prompts for the SAM model."""
        return SAMPrompts(point_coords, point_labels, box_coords)
