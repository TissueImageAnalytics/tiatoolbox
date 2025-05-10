"""Define SAM architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntBounds, IntPair


class SAM(ModelABC):
    """Segment Anything Model (SAM) Architecture.

    Meta AI's zero-shot segmentation model.
    SAM is used for interactive general-purpose segmentation.

    Currently supports SAM, which requires a checkpoint and model type.

    SAM accepts an RGB image patch along with a list of point and bounding
    box coordinates as prompts.

    Args:
        model_type (str):
            Model type.
            Currently supported: vit_b, vit_l, vit_h.
        checkpoint_path (str):
            Path to the model checkpoint.
        device (str):
            Device to run inference on.

    Examples:
        >>> # instantiate SAM with checkpoint path and model type
        >>> sam = SAM(
        ...     model_type="vit_b",
        ...     checkpoint_path="path/to/sam_checkpoint.pth"
        ... )
    """

    def __init__(
        self: SAM,
        model_path: str = "facebook/sam-vit-huge",
        *,
        device: str = "cpu",
    ) -> None:
        """Initialize :class:`SAM`."""
        super().__init__()
        self.net_name = "SAM"
        self.device = device

        self.model = SamModel.from_pretrained(model_path).to(device)
        self.processor = SamProcessor.from_pretrained(model_path)

    def forward(  # skipcq: PYL-W0221
        self: SAM,
        imgs: list,
        point_coords: list | None = None,
        box_coords: list | None = None,
    ) -> np.ndarray:
        """Torch method. Defines forward pass on each image in the batch.

        Note: This architecture only uses a single layer, so only one forward pass
        is needed.

        Args:
            imgs (list):
                List of images to process, of the shape NHWC.
            point_coords (list):
                List of point coordinates for each image.
            box_coords (list):
                Bounding box coordinates for each image.

        Returns:
            list:
                List of masks and scores for each image.

        """
        masks, scores = [], []
        for i, img in enumerate(imgs):
            image = [Image.fromarray(img)]
            image_embeddings = self._encode_image(image)
            point_labels = None
            points = None
            boxes = None

            # Processor expects coordinates to be lists
            def format_coords(coords: np.ndarray | list) -> list:
                """Helper function that converts coordinates to list format."""
                if isinstance(coords, np.ndarray):
                    return coords.tolist()
                if isinstance(coords[0], np.ndarray):
                    return [
                        item.tolist() if isinstance(item, np.ndarray) else item
                        for item in coords
                    ]
                return coords

            if point_coords is not None:
                points = point_coords[i]
                # Convert point coordinates to list
                if points is not None:
                    point_labels = [[[1] * len(points)]]
                    points = [format_coords(points)]

            if box_coords is not None:
                boxes = box_coords[i]
                # Convert box coordinates to list
                if boxes is not None:
                    boxes = [format_coords(boxes)]

            inputs = self.processor(
                image,
                input_points=points,
                input_labels=point_labels,
                input_boxes=boxes,
                return_tensors="pt",
            ).to(self.device)

            # Replaces pixel_values with image embeddings
            inputs.pop("pixel_values", None)
            inputs.update({"image_embeddings": image_embeddings})

            with torch.inference_mode():
                # Forward pass through the model
                outputs = self.model(**inputs, multimask_output=False)
                image_masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                image_scores = outputs.iou_scores.cpu()
            masks.append(image_masks)
            scores.append(image_scores)
            torch.cuda.empty_cache()

        return np.array(masks), np.array(scores)

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: list,
        point_coords: list[list[IntPair]] | None = None,
        box_coords: list[IntBounds] | None = None,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference on an input batch.

        Contains logic for forward operation as well as I/O aggregation.
        SAM accepts a list of points and a single bounding box per image.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (list):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            point_coords (list):
                Point coordinates for each image in the batch.
            box_coords (list):
                Bounding box coordinates for each image in the batch.
            device (str):
                Device to run inference on.

        Returns:
            pred_info (list):
                Tuple of masks and scores for each image in the batch.

        """
        model.eval().to(device)

        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cpu().numpy()

        with torch.inference_mode():
            masks, scores = model(batch_data, point_coords, box_coords)

        return masks, scores

    def _encode_image(self: SAM, image: np.ndarray) -> np.ndarray:
        """Encodes the image for feature extraction."""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        return self.model.get_image_embeddings(inputs["pixel_values"])

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Pre-processes an image - Converts it into a format accepted by SAM (HWC)."""
        # Move the tensor to the CPU if it's a PyTorch tensor
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        return image[..., :3]  # Remove alpha channel if present

    def to(
        self: ModelABC,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
    ) -> ModelABC | torch.nn.DataParallel[ModelABC]:
        """Moves the model to the specified device."""
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        self.device = device
        self.model.to(device)
        return self
