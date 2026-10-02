"""SegFormer semantic segmentation via Hugging Face Transformers (Apache-2.0).

This module wraps ``transformers.SegformerForSemanticSegmentation`` and keeps the
TIAToolbox ``ModelABC`` inference API. SMP-format checkpoints are auto-remapped on
``load_state_dict``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import dask.array as da
import numpy as np
import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from tiatoolbox.models.architecture.segformer_hf import (
    MIT_ENCODER_CONFIGS,
    is_smp_segformer_state_dict,
    remap_smp_segformer_state_dict,
)
from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from torch.nn.modules.module import _IncompatibleKeys


class Segformer(ModelABC):
    """SegFormer semantic segmentation model (Hugging Face Transformers).

    Args:
        encoder_name:
            Mix Transformer backbone name: ``mit_b0`` ... ``mit_b5``.
        decoder_segmentation_channels:
            Channel width of the all-MLP decoder (HF ``decoder_hidden_size``).
        in_channels:
            Number of input image channels (default RGB = 3).
        classes:
            Number of output segmentation classes (``num_labels``).
        activation:
            Optional activation applied after the classifier / upsample.
        upsampling:
            Spatial upsample factor applied to HF logits (HF outputs at 1/4
            resolution; default ``4`` restores input size).

    .. _SegFormer:
        https://arxiv.org/abs/2105.15203

    """

    def __init__(
        self,
        encoder_name: str = "mit_b5",
        decoder_segmentation_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: nn.Module | None = None,
        upsampling: int = 4,
    ) -> None:
        """Initializes the SegFormer model."""
        super().__init__()
        self.requires_divisible_input_shape = True

        if encoder_name not in MIT_ENCODER_CONFIGS:
            supported = ", ".join(sorted(MIT_ENCODER_CONFIGS))
            msg = f"Unknown encoder_name={encoder_name!r}. Supported: {supported}."
            raise ValueError(msg)

        config = SegformerConfig(
            num_labels=classes,
            num_channels=in_channels,
            decoder_hidden_size=decoder_segmentation_channels,
            reshape_last_stage=True,
            **MIT_ENCODER_CONFIGS[encoder_name],
        )
        self.model = SegformerForSemanticSegmentation(config)
        self.upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        self.activation = activation if activation is not None else nn.Identity()
        self.name = f"segformer-{encoder_name}"

    def forward(
        self: Segformer,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run encoder-decoder and upsample logits to the input resolution."""
        logits = self.model(pixel_values=x).logits
        return self.activation(self.upsampling(logits))

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,  # noqa: FBT001, FBT002
        assign: bool = False,  # noqa: FBT001, FBT002
    ) -> _IncompatibleKeys:
        """Load HF or SMP SegFormer weights.

        SMP checkpoints (keys like ``encoder.patch_embed1``) are remapped to Hugging
        Face names. Classifier tensors keep their shapes, so checkpoints that only
        differ by output class count load when ``classes`` matches.
        """
        mapped: dict[str, torch.Tensor] = dict(state_dict)
        if is_smp_segformer_state_dict(mapped):
            mapped = remap_smp_segformer_state_dict(mapped)

        # Remapped / native HF keys are ``segformer.*`` / ``decode_head.*``.
        # This wrapper stores the HF module under ``self.model``.
        needs_model_prefix = (
            bool(mapped)
            and not any(key.startswith("model.") for key in mapped)
            and any(key.startswith(("segformer.", "decode_head.")) for key in mapped)
        )
        if needs_model_prefix:
            mapped = {f"model.{key}": value for key, value in mapped.items()}

        return super().load_state_dict(mapped, strict=strict, assign=assign)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Preprocess input image for inference.

        Applies ImageNet normalization to the input image.

        Args:
            image (np.ndarray):
                Input image as a NumPy array of shape (H, W, C) in uint8 format.

        Returns:
            np.ndarray:
                Preprocessed image normalized to ImageNet statistics.

        Example:
            >>> img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
            >>> processed = Segformer.preproc(img)
            >>> processed.shape
            (448, 448, 3)

        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image / 255.0 - mean) / std

    def postproc(  # skipcq: PYL-W0221
        self: Segformer, image: np.ndarray
    ) -> np.ndarray:
        """Postprocess model output to generate a class mask.

        Applies argmax and morphological operations to classify pixels.

        Args:
            image (np.ndarray):
                Input probability map as a NumPy array of shape (H, W, C).

        Returns:
            np.ndarray:
                Tissue mask

        Example:
            >>> model = Segformer(classes=2)
            >>> mask = model.postproc(probs)
            >>> mask.shape
            (448, 448)

        """
        if isinstance(image, da.Array):
            image = image.compute()
        prediction_mask = np.argmax(image, axis=-1).astype(np.uint8)

        kernel_diameter = 3
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
        )
        prediction_mask = cv2.morphologyEx(prediction_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(prediction_mask, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def infer_batch(
        model: Segformer,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on a batch of images.

        Transfers the model and input batch to the specified device, performs
        forward pass, and returns softmax probabilities.

        Args:
            model (Segformer):
                Segformer model instance.
            batch_data (torch.Tensor):
                Batch of input images in NHWC format.
            device (str):
                Device for inference (e.g., "cpu" or "cuda").

        Returns:
            np.ndarray:
                Inference results as a NumPy array of shape (N, H, W, C).

        Example:
            >>> batch = torch.randn(4, 448, 448, 3)
            >>> probs = Segformer.infer_batch(
            ...     model, batch, device="cpu"
            ... )
            >>> probs.shape
            (4, 448, 448, 1)

        """
        model = model.to(device)
        model.eval()

        imgs = batch_data
        imgs = imgs.to(device=device, dtype=torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()
