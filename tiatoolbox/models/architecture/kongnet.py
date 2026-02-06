"""KongNet Nuclei Detection Model Architecture [1].

This module defines the KongNet model for nuclei detection and classification
in digital pathology. It implements a multi-head encoder decoder architecture
with an EfficientNetV2-L encoder. The model is designed to detect and classify
nuclei in whole slide images (WSIs).

KongNet achieved 1st on track 1 and 2nd on track 2 during the MONKEY Challenge [2].
KongNet achieved 1st place in the 2025 MIDOG Challenge [3].
KongNet ranked among the top three in the PUMA Challenge [4].
KongNet achieved SOTA detection performance on PanNuke [5] and CoNIC [6] datasets.

Please cite the paper [1], if you use this model.

Pretrained Models:
-----------------
    - KongNet_MONKEY_1:
        MONKEY Challenge model.
    - KongNet_Det_MIDOG_1:
        MIDOG Challenge lightweight detection model.
    - KongNet_PUMA_T1_3:
        PUMA Challenge model for track 1.
    - KongNet_PUMA_T2_3:
        PUMA Challenge model for track 2.
    - KongNet_CoNIC_1:
        CoNIC model.
    - KongNet_PanNuke_1:
        PanNuke model.

Key Components:
---------------
- TimmEncoderFixed: Encoder module using TIMM models with fixed drop_path_rate handling.
- SubPixelUpsample: Sub-pixel upsampling module using PixelShuffle.
- DecoderBlock: U-Net style decoder block with attention mechanisms.
- KongNetDecoder: U-Net style decoder with multiple decoder blocks.
- KongNet: Multi-head segmentation model with shared encoder and multiple decoders.

Features:
---------
- Multi-head architecture for accurate nuclei detection and classification.
- Efficient inference pipeline for batch processing.

Example:
    >>> from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
    >>> detector = NucleusDetector(model="KongNet_CoNIC_1")
    >>> results = detector.run(
    ...     ["/example_wsi.svs"],
    ...     masks=None,
    ...     auto_get_mask=False,
    ...     patch_mode=False,
    ...     save_dir=Path("/KongNet_CoNIC/"),
    ...     output_type="annotationstore",
    ... )

References:
    [1] Lv, Jiaqi et al., "KongNet: A Multi-headed Deep Learning Model for Detection
    and Classification of Nuclei in Histopathology Images.", 2025,
    arXiv preprint arXiv:2510.23559., URL: https://arxiv.org/abs/2510.23559
    [2] L. Studer, “Structured description of the monkey challenge,” Sept. 2024.
    [3] J. Ammeling, M. Aubreville, S. Banerjee, C. A. Bertram, K. Breininger,
    D. Hirling, P. Horvath, N. Stathonikos, and M. Veta, “Mitosis domain
    generalization challenge 2025,” Mar. 2025.
    [4] M. Schuiveling, H. Liu, D. Eek, G. Breimer, K. Suijkerbuijk, W. Blokx,
    and M. Veta, “A novel dataset for nuclei and tissue segmentation in
    melanoma with baseline nuclei segmentation and tissue segmentation
    benchmarks,” GigaScience, vol. 14, 01 2025.
    [5] J. Gamper, N. A. Koohbanani, K. Benes, S. Graham, M. Jahanifar,
    S. A. Khurram, A. Azam, K. Hewitt, and N. Rajpoot, “Pannuke dataset
    extension, insights and baselines,” 2020.
    [6]  S. Graham et al., “Conic challenge: Pushing the frontiers of nuclear detection,
    segmentation, classification and counting,” Medical Image Analysis,
    vol. 92, p. 103047, 2024.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from tiatoolbox.models.architecture.utils import (
    AttentionModule,
    SegmentationHead,
    nms_on_detection_maps,
    peak_detection_map_overlap,
)
from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from tiatoolbox.type_hints import IntPair


class TimmEncoderFixed(nn.Module):
    """Fixed version of TIMM encoder that handles drop_path_rate parameter properly.

    This encoder wraps TIMM models to provide consistent feature extraction interface
    for segmentation tasks. It extracts features at multiple scales from the encoder
    backbone.

    """

    def __init__(
        self,
        name: str,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        drop_rate: float = 0.5,
        drop_path_rate: float | None = 0.0,
        *,
        pretrained: bool = True,
    ) -> None:
        """Initialize TimmEncoderFixed.

        Args:
            name (str):
                Name of the TIMM model to use as backbone.
            in_channels (int):
                Number of input channels. Default is 3.
            depth (int):
                Number of encoder stages to extract features from. Default is 5.
            output_stride (int):
                Output stride of the encoder. Default is 32.
            drop_rate (float):
                Dropout rate. Default is 0.5.
            drop_path_rate (float | None):
                Drop path rate of the encoder. Default is 0.0.
            pretrained (bool):
                Whether to use pretrained weights. Default is True.

        """
        super().__init__()
        if drop_path_rate is None:
            kwargs = {
                "in_chans": in_channels,
                "features_only": True,
                "pretrained": pretrained,
                "out_indices": tuple(range(depth)),
                "drop_rate": drop_rate,
            }
        else:
            kwargs = {
                "in_chans": in_channels,
                "features_only": True,
                "pretrained": pretrained,
                "out_indices": tuple(range(depth)),
                "drop_rate": drop_rate,
                "drop_path_rate": drop_path_rate,
            }

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [in_channels, *self.model.feature_info.channels()]
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)

        Returns:
            list[torch.Tensor]:
                List of feature tensors at different scales,
                including the input as the first element

        """
        features = self.model(x)
        return [x, *features]

    @property
    def out_channels(self) -> list[int]:
        """Get output channels for each feature level.

        Returns:
            list[int]:
                Number of channels at each feature level

        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """Get the output stride of the encoder.

        Returns:
            int:
                Output stride value

        """
        return min(self._output_stride, 2**self._depth)


class SubPixelUpsample(nn.Module):
    """Sub-pixel upsampling module using PixelShuffle.

    This module performs upsampling using sub-pixel convolution (PixelShuffle)
    which is more efficient than transposed convolution and produces better results.

    Args:
        in_channels (int):
            Number of input channels
        out_channels (int):
            Number of output channels
        upscale_factor (int):
            Factor to increase spatial resolution. Default: 2

    """

    def __init__(
        self, in_channels: int, out_channels: int, upscale_factor: int = 2
    ) -> None:
        """Initialize SubPixelUpsample.

        Args:
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            upscale_factor (int):
                Factor to increase spatial resolution. Default is 2.

        """
        super().__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sub-pixel upsampling.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Upsampled tensor of shape
                (B, out_channels, H*upscale_factor, W*upscale_factor)

        """
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        return self.conv2(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and attention.

    This block performs upsampling of the input features, concatenates
    with skip connections from the encoder, applies attention mechanisms,
    and processes through convolutions.

    Args:
        in_channels (int):
            Number of input channels
        skip_channels (int):
            Number of channels from skip connection
        out_channels (int):
            Number of output channels
        attention_type (str):
            Type of attention mechanism. Default: 'scse'.

    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_type: str = "scse",
    ) -> None:
        """Initialize DecoderBlock.

        Args:
            in_channels (int):
                Number of input channels
            skip_channels (int):
                Number of channels from skip connection
            out_channels (int):
                Number of output channels
            attention_type (str):
                Type of attention mechanism. Default: 'scse'.

        """
        super().__init__()
        self.up = SubPixelUpsample(in_channels, in_channels, upscale_factor=2)
        self.conv1 = Conv2dNormActivation(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention1 = AttentionModule(
            name=attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention2 = AttentionModule(name=attention_type, in_channels=out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x (torch.Tensor):
                Input tensor to be upsampled
            skip (Optional[torch.Tensor]):
                Skip connection tensor from encoder. Default: None

        Returns:
            torch.Tensor:
                Processed output tensor

        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention2(x)


class CenterBlock(nn.Module):
    """Center block that applies attention mechanism at the bottleneck.

    This block is placed at the center of the U-Net architecture (deepest level)
    to enhance feature representation using attention mechanisms.

    Args:
        in_channels (int):
            Number of input channels

    """

    def __init__(self, in_channels: int) -> None:
        """Initialize CenterBlock with attention.

        Args:
            in_channels (int):
                Number of input channels.

        """
        super().__init__()
        self.attention = AttentionModule(name="scse", in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through center block.

        Args:
            x (torch.Tensor):
                Input tensor.

        Returns:
            torch.Tensor:
                Output tensor with attention applied.

        """
        return self.attention(x)


class KongNetDecoder(nn.Module):
    """Decoder module for KongNet architecture.

    This decoder implements a U-Net style decoder with multiple decoder blocks,
    attention mechanisms, and optional center block at the bottleneck.

    Args:
        encoder_channels (List[int]):
            Number of channels at each encoder level
        decoder_channels (Tuple[int, ...]):
            Number of channels at each decoder level
        n_blocks (int):
            Number of decoder blocks. Default: 5
        attention_type (str):
            Type of attention mechanism. Default: 'scse'
        center (bool):
            Whether to use center block at bottleneck. Default: True

    Raises:
        ValueError:
            If n_blocks doesn't match length of decoder_channels

    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: tuple[int, ...],
        n_blocks: int = 5,
        attention_type: str = "scse",
        *,
        center: bool = True,
    ) -> None:
        """Initialize KongNetDecoder.

        Args:
            encoder_channels (List[int]):
                Number of channels at each encoder level.
            decoder_channels (Tuple[int, ...]):
                Number of channels at each decoder level.
            n_blocks (int):
                Number of decoder blocks. Default is 5.
            attention_type (str):
                Type of attention mechanism to use. Default is 'scse'.
            center (bool):
                Whether to include a center block at the bottleneck.
                Default is True.

        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            msg = (
                f"The number of blocks {n_blocks} must match the"
                f" length of decoder_channels {len(decoder_channels)}."
            )
            raise ValueError(msg)

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels, *list(decoder_channels[:-1])]
        skip_channels = [*list(encoder_channels[1:]), 0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels)
        else:
            self.center = nn.Identity()

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, attention_type=attention_type)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels, strict=True
            )
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            *features:
                Feature tensors from encoder at different scales

        Returns:
            torch.Tensor:
                Decoded output tensor

        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class KongNet(ModelABC):
    """KongNet: Multi-head nuclei detection model.

    This module defines the KongNet model for nuclei detection and classification
    in digital pathology. It implements a multi-head encoder decoder architecture
    with an EfficientNetV2-L encoder. The model is designed to detect and classify
    nuclei in whole slide images (WSIs).


    Attributes:
        encoder:
            Encoder module (e.g., TimmEncoderFixed)
        decoders:
            List of decoder modules (KongNetDecoder)
        heads:
            List of segmentation head modules (SegmentationHead)
        min_distance:
            Minimum distance between peaks in post-processing
        threshold_abs:
            Absolute threshold for peak detection in post-processing
        target_channels:
            List of target channel indices for post-processing
        output_class_dict:
            Optional dictionary mapping class names to indices
        postproc_tile_shape:
            Tile shape for post-processing with dask

    Example:
        >>> from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
        >>> detector = NucleusDetector(model="KongNet_CoNIC_1")
        >>> results = detector.run(
        ...     ["/example_wsi.svs"],
        ...     masks=None,
        ...     auto_get_mask=False,
        ...     patch_mode=False,
        ...     save_dir=Path("/KongNet_CoNIC/"),
        ...     output_type="annotationstore",
        ... )

    References:
        [1] Lv, Jiaqi et al., "KongNet: A Multi-headed Deep Learning Model for Detection
        and Classification of Nuclei in Histopathology Images.", 2025,
        arXiv preprint arXiv:2510.23559.,
        URL: https://arxiv.org/abs/2510.23559

    """

    def __init__(
        self: KongNet,
        num_heads: int,
        num_channels_per_head: list[int],
        target_channels: list[int],
        min_distance: int,
        threshold_abs: float,
        postproc_tile_shape: IntPair = (2048, 2048),
        *,
        wide_decoder: bool = False,
        class_dict: dict | None = None,
    ) -> None:
        """Initialize KongNet model.

        Args:
            num_heads (int):
                Number of decoder heads.
            num_channels_per_head (list[int]):
                List specifying number of output channels for each head.
            target_channels (list[int]):
                List of target channel indices for post-processing.
            min_distance (int):
                Minimum distance between peaks in post-processing.
            threshold_abs (float):
                Absolute threshold for peak detection in post-processing.
            postproc_tile_shape (IntPair):
                Tile shape for post-processing with dask. Defaults to (2048, 2048).
            wide_decoder (bool):
                Whether to use a wider decoder architecture. Defaults to False.
            class_dict (dict | None):
                Optional dictionary mapping class names to indices. Defaults to None.

        """
        super().__init__()

        if len(num_channels_per_head) != num_heads:
            msg = (
                f"Number of decoders {len(num_channels_per_head)}"
                f" must match number of heads {num_heads}."
            )
            raise ValueError(msg)

        self.encoder = TimmEncoderFixed(
            name="tf_efficientnetv2_l.in21k_ft_in1k",
            in_channels=3,
            depth=5,
            output_stride=32,
            drop_rate=0.5,
            drop_path_rate=0.25,
            pretrained=False,
        )

        decoder_channels = (256, 128, 64, 32, 16)
        if wide_decoder:
            decoder_channels = (512, 256, 128, 64, 32)

        decoders = [
            KongNetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                center=True,
                attention_type="scse",
            )
            for _ in range(num_heads)
        ]

        heads = [
            SegmentationHead(
                in_channels=decoders[i].blocks[-1].conv2[0].out_channels,
                out_channels=num_channels_per_head[i],  # instance channels
                activation=None,
                kernel_size=1,
            )
            for i in range(num_heads)
        ]

        self.decoders = nn.ModuleList(decoders)
        self.heads = nn.ModuleList(heads)
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.target_channels = target_channels
        self.output_class_dict = class_dict
        self.postproc_tile_shape = postproc_tile_shape

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
            >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            >>> processed = KongNet.preproc(img)
            >>> processed.shape
            ... (256, 256, 3)

        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image / 255.0 - mean) / std

    def forward(  # skipcq: PYL-W0613
        self: KongNet,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)
            *args (tuple):
                Additional positional arguments (unused).
            **kwargs (dict):
                Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Concatenated output from all heads of shape
                (B, sum(num_channels_per_head), H, W)

        """
        features = self.encoder(x)
        decoder_outputs = [decoder(*features) for decoder in self.decoders]

        segmentation_head_outputs = []
        for head, decoder_output in zip(self.heads, decoder_outputs, strict=True):
            segmentation_head_outputs.append(head(decoder_output))

        return torch.cat(segmentation_head_outputs, 1)

    @staticmethod
    def infer_batch(
        model: KongNet,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on a batch of images.

        Transfers the model and input batch to the specified device, performs
        forward pass, and returns probability maps.

        Args:
            model (torch.nn.Module):
                PyTorch model instance.
            batch_data (torch.Tensor):
                Batch of input images in NHWC format.
            device (str):
                Device for inference (e.g., "cpu" or "cuda").

        Returns:
            np.ndarray:
                Inference results as a NumPy array of shape (N, H, W, C).

        Example:
            >>> batch = torch.randn(4, 256, 256, 3)
            >>> probs = KongNet.infer_batch(model, batch, device="cpu")
            >>> probs.shape
            (4, 256, 256, len(model.target_channels))

        """
        model = model.to(device)
        model.eval()

        imgs = batch_data
        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            target_logits = logits[:, model.target_channels, :, :]
            probs = torch.nn.functional.sigmoid(target_logits)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()

    #  skipcq: PYL-W0221  # noqa: ERA001
    def postproc(
        self: KongNet,
        block: np.ndarray,
        min_distance: int | None = None,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        block_info: dict | None = None,
        depth_h: int = 0,
        depth_w: int = 0,
    ) -> np.ndarray:
        """KongNet post-processing function.

        Builds a processed mask per input channel, runs peak_local_max then
        writes 1.0 at peak pixels.

        Returns same spatial shape as the input block

        Args:
            block (np.ndarray):
                shape (H, W, C).
            min_distance (int | None):
                The minimal allowed distance separating peaks.
            threshold_abs (float | None):
                Minimum intensity of peaks.
            threshold_rel (float | None):
                Minimum intensity of peaks.
            block_info (dict | None):
                Dask block info dict. Only used when called from
                dask.array.map_overlap.
            depth_h (int):
                Halo size in pixels for height (rows). Only used
                when it's called from dask.array.map_overlap.
            depth_w (int):
                Halo size in pixels for width (cols). Only used
                when it's called from dask.array.map_overlap.

        Returns:
            out:
                NumPy array (H, W, C) with 1.0 at peaks, 0 elsewhere.

        """
        min_distance_to_use = (
            self.min_distance if min_distance is None else min_distance
        )
        threshold_abs_to_use = (
            self.threshold_abs if threshold_abs is None else threshold_abs
        )
        peak_map = peak_detection_map_overlap(
            block,
            min_distance=min_distance_to_use,
            threshold_abs=threshold_abs_to_use,
            threshold_rel=threshold_rel,
            block_info=block_info,
            depth_h=depth_h,
            depth_w=depth_w,
            return_probability=True,
        )

        return nms_on_detection_maps(
            peak_map,
            min_distance=min_distance_to_use,
        )

    def load_state_dict(
        self: KongNet,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
        assign: bool = False,
    ) -> nn.Module:
        """Load state dict with support for wrapped models."""
        return super().load_state_dict(state_dict["model"], strict, assign)
