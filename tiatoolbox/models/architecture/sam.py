"""Define SAM architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import IntBounds, IntPair


class SAM(ModelABC):
    """Segment Anything Model (SAM) Architecture.

    Meta AI's zero-shot segmentation model.
    SAM is used for interactive general-purpose segmentation.

    Currently supports both SAM and SAM2, each of which require
    different model checkpoints and configuration files.

    SAM accepts an RGB image patch along with a list of point and bounding
    box coordinates as prompts.

    Args:
        model_type (str):
            Model type. Currently supported: vit_b, vit_l, vit_h.
            Required for SAM.
        checkpoint_path (str):
            Path to the model checkpoint.
            Required for both SAM and SAM2.
        model_cfg_path (str):
            Path to the model configuration file.
            Required for SAM2.
        model_hf_path (str):
            Huggingface path for the pretrained SAM2 model.
            If provided, it will override the checkpoint_path and model_cfg_path.
            Default is "facebook/sam2-hiera-tiny".
        device (str):
            Device to run inference on.
        use_sam2 (bool):
            Whether to use SAM2 or not. Default is True.

    Examples:
        >>> # instantiate SAM with checkpoint path and model type
        >>> sam = SAM(
        ...     model_type="vit_b",
        ...     checkpoint_path="path/to/sam_checkpoint.pth"
        ...     use_sam2=False
        ... )
        >>> # instantiate SAM2 with checkpoint and config path
        >>> sam2 = SAM(
        ...     checkpoint_path="path/to/sam2_checkpoint.pth",
        ...     model_cfg_path="path/to/sam2_config.yaml"
        ... )
        >>> # instantiate SAM2 with Huggingface path
        >>> sam2 = SAM(
        ...     model_hf_path="facebook/sam2-hiera-tiny"
        ... )
    """

    def __init__(
        self: SAM,
        model_type: str | None = None,
        checkpoint_path: str | None = None,
        model_cfg_path: str | None = None,
        model_hf_path: str = "facebook/sam2-hiera-tiny",
        *,
        device: str = "cpu",
        use_sam2: bool = True,
    ) -> None:
        """Initialize :class:`SAM`."""
        super().__init__()
        self.use_sam2 = use_sam2
        self.net_name = "SAM"

        if self.use_sam2:
            # Load SAM2
            if checkpoint_path is None or model_cfg_path is None:
                self.model = build_sam2_hf(model_hf_path, device=device)
            else:
                self.model = build_sam2(model_cfg_path, checkpoint_path)
            self.predictor = SAM2ImagePredictor(self.model)
            self.generator = SAM2AutomaticMaskGenerator(self.model)
        else:
            # Load original SAM
            if checkpoint_path is None:
                msg = "You must provide a checkpoint path for SAM."
                raise ValueError(msg)
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(
                device
            )
            self.predictor = SamPredictor(self.model)
            self.generator = SamAutomaticMaskGenerator(self.model)

    def forward(
        self: SAM,
        imgs: list,
        point_coords: list[list[IntPair]] | None = None,
        box_coords: list[list[IntBounds]] | None = None,
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
                List of bounding box coordinates for each image.

        Returns:
            list:
                List of masks and scores for each image.

        """
        batch_masks, batch_scores = [], []

        for i, image in enumerate(imgs):
            self._encode_image(image)

            # assume that prompts will be provided for all images in a batch
            points = point_coords[i] if point_coords is not None else None
            boxes = box_coords[i] if box_coords is not None else None

            if points is not None or boxes is not None:
                point_labels = [1] * len(point_coords)
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box_coords=boxes,
                    multimask_output=False,
                )
            else:
                # Use SAM's automatic mask generator
                masks = self.generator.generate(image)
                scores = np.array([mask["predicted_iou"] for mask in masks])
                masks = np.array(
                    [mask["segmentation"] for mask in masks], dtype=np.uint8
                )

            sorted_ind = np.argsort(scores)[::-1]
            sorted_masks = np.array(masks[sorted_ind], dtype=np.uint8)
            sorted_scores = np.around(scores[sorted_ind], 2)
            batch_masks.append(sorted_masks)
            batch_scores.append(sorted_scores)

        return batch_masks, batch_scores

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: list,
        point_coords: list[list[IntPair]] | None = None,
        box_coords: list[list[IntBounds]] | None = None,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference on an input batch.

        Contains logic for forward operation as well as I/O aggregation.

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

        """
        model.eval()
        model = model.to(device)

        with torch.inference_mode():
            batch_data = model.preproc(batch_data)
            masks, scores = model(batch_data, point_coords, box_coords)
            masks = model.postproc(masks)
        return masks, scores

    def _encode_image(self: SAM, image: np.ndarray) -> np.ndarray:
        """Encodes the image for feature extraction."""
        self.predictor.set_image(image)

    def load_weights(self: SAM, checkpoint_path: str) -> None:
        """Loads model weights from specified checkpoint."""
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    @staticmethod
    def preproc(images: np.ndarray) -> np.ndarray:
        """Pre-processes images - Converts them into a format accepted by SAM (HWC)."""
        # Move the tensor to the CPU if it's a PyTorch tensor
        if isinstance(images, torch.Tensor):
            return images.permute(0, 2, 3, 1).cpu().numpy()

        return images[:, :, :, :3]  # Remove alpha channel if present

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Post-processes images."""
        return image
