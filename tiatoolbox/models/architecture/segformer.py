"""SegFormer architecture components used for semantic segmentation."""

from __future__ import annotations

from typing import Any

import cv2
import dask.array as da
import numpy as np
import torch
from torch import nn

from tiatoolbox.models.architecture.mix_transformer import (
    MixVisionTransformerEncoder,
    mix_transformer_encoders,
)
from tiatoolbox.models.architecture.utils import Conv2dReLU, SegmentationHead
from tiatoolbox.models.models_abc import ModelABC

MIN_SEGFORMER_DECODER_DEPTH = 3


class MLP(nn.Module):
    """Linear projection block used in the SegFormer decoder."""

    def __init__(self, skip_channels: int, segmentation_channels: int) -> None:
        """Initializes the MLP module for Segformer decoder."""
        super().__init__()

        self.linear = nn.Linear(skip_channels, segmentation_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder features into the decoder channel space."""
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


class SegformerDecoder(nn.Module):
    """Decoder head that fuses multi-scale transformer features."""

    def __init__(
        self,
        encoder_channels: list[int],
        encoder_depth: int = 5,
        segmentation_channels: int = 256,
    ) -> None:
        """Initializes the Segformer decoder module."""
        super().__init__()

        if encoder_depth < MIN_SEGFORMER_DECODER_DEPTH:
            msg = (
                "Encoder depth for Segformer decoder cannot be less than "
                f"{MIN_SEGFORMER_DECODER_DEPTH}, got {encoder_depth}."
            )
            raise ValueError(msg)

        if encoder_channels[1] == 0:
            encoder_channels = [
                channel for index, channel in enumerate(encoder_channels) if index != 1
            ]
        encoder_channels = encoder_channels[::-1]

        self.mlp_stage = nn.ModuleList(
            [MLP(channel, segmentation_channels) for channel in encoder_channels[:-1]]
        )

        self.fuse_stage = Conv2dReLU(
            in_channels=(len(encoder_channels) - 1) * segmentation_channels,
            out_channels=segmentation_channels,
            kernel_size=1,
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the Segformer decoder."""
        # Resize all features to the size of the largest feature
        target_size: list[int] = [dim // 4 for dim in features[0].shape[2:]]

        features = features[2:] if features[1].size(1) == 0 else features[1:]
        features = features[::-1]  # reverse channels to start from head of encoder

        resized_features_arr: list[torch.Tensor] = []
        for i, mlp_layer in enumerate(self.mlp_stage):
            feature = mlp_layer(features[i])
            resized_feature = torch.nn.functional.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_features_arr.append(resized_feature)

        resized_features = torch.cat(resized_features_arr, dim=1)
        return self.fuse_stage(resized_features)


class Segformer(ModelABC):
    """Segformer is a simple and efficient transformer for semantic segmentation.

    Args:
        encoder_name: Name of the classification model used as an encoder
            (a.k.a. backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
            Each stage generates features two times smaller in spatial
            dimensions than the previous one. For depth 0 we will have
            features with shapes [(N, C, H, W),], for depth 1
            [(N, C, H, W), (N, C, H // 2, W // 2)], and so on.
            Default is 5
        decoder_segmentation_channels: A number of convolution filters in
            segmentation blocks, default is 256
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask, or the number of
            output mask channels
        activation: An activation function to apply after the final
            convolution layer.
        upsampling: A number to upsample the output of the model,
            default is 4 (same size as input)

    Returns:
        ``torch.nn.Module``: **Segformer**

    .. _Segformer:
        https://arxiv.org/abs/2105.15203

    """

    def __init__(
        self,
        encoder_name: str = "mit_b5",
        encoder_depth: int = 5,
        decoder_segmentation_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: nn.Module | None = None,
        upsampling: int = 4,
    ) -> None:
        """Initializes the Segformer model."""
        super().__init__()
        self.requires_divisible_input_shape = True

        self.encoder_params = mix_transformer_encoders[encoder_name]["params"]

        self.encoder = MixVisionTransformerEncoder(
            **self.encoder_params,
        )
        self.encoder.set_in_channels(in_channels=in_channels, pretrained=False)

        self.decoder = SegformerDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            segmentation_channels=decoder_segmentation_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.name = f"segformer-{encoder_name}"

    def forward(
        self: Segformer,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Sequentially pass `x` through encoder, decoder, and head."""
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(encoder_features)
        return self.segmentation_head(decoder_output)

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
        """Postprocess model output to generate tissue mask.

        Applies argmax and morphological operations to classify pixels.

        Args:
            image (np.ndarray):
                Input probability map as a NumPy array of shape (H, W, C).

        Returns:
            np.ndarray:
                Tissue mask

        Example:
            >>> model = Segformer(num_classes=1, threshold=0.95)
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
