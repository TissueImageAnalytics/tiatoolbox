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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import timm
import torch
from scipy import ndimage
from skimage.segmentation import watershed
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
    from collections.abc import Mapping, Sequence

    from tiatoolbox.type_hints import IntPair


@dataclass(frozen=True)
class KongNetOutputHeadSpec:
    """Describe one logical output head inside KongNet's concatenated tensor."""

    index: int
    name: str
    display_name: str
    channel_slice: slice
    num_channels: int
    target_channels: tuple[int, ...]
    target_channel_offsets: tuple[int, ...]

    @property
    def has_target_channels(self) -> bool:
        """Return whether this head contributes inference target channels."""
        return bool(self.target_channels)


@dataclass(frozen=True)
class KongNetComponentMaps:
    """Grouped per-class KongNet supervision maps for one patch.

    Each component is stored as a ``(H, W, C)`` NumPy array where ``C`` is the
    number of KongNet heads/classes.

    """

    mask: np.ndarray
    boundary: np.ndarray
    centroid: np.ndarray

    def __post_init__(self) -> None:
        """Validate component map shapes."""
        expected_shape = self.mask.shape
        if self.mask.ndim != 3:
            msg = "KongNet component maps must be 3D arrays of shape `(H, W, C)`."
            raise ValueError(msg)
        if self.boundary.shape != expected_shape or self.centroid.shape != expected_shape:
            msg = "Mask, boundary, and centroid maps must share the same shape."
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the common ``(H, W, C)`` component shape."""
        return self.mask.shape

    @property
    def num_classes(self) -> int:
        """Return the number of KongNet heads/classes represented."""
        return int(self.mask.shape[-1])


@dataclass(frozen=True)
class KongNetInstancePostProcResult:
    """Deterministic KongNet patch post-processing output.

    Attributes:
        instance_map:
            Integer instance-id map with ``0`` reserved for background.
        class_map:
            Integer per-pixel class-id map aligned with ``instance_map``.
        peak_map:
            Sparse ``(H, W, C)`` centroid peak map after NMS, storing peak
            probabilities at retained seed locations.
        marker_map:
            Integer seed marker map used to initialize watershed.
        foreground_mask:
            Binary foreground support derived from KongNet mask channels.
        score_map:
            Dense watershed score map derived from mask and boundary channels.
        instance_classes:
            Mapping from instance id to final class id.

    """

    instance_map: np.ndarray
    class_map: np.ndarray
    peak_map: np.ndarray
    marker_map: np.ndarray
    foreground_mask: np.ndarray
    score_map: np.ndarray
    instance_classes: dict[int, int]

    @property
    def conic_map(self) -> np.ndarray:
        """Return the CoNIC-style ``(instance_id, class_id)`` stack."""
        return np.stack((self.instance_map, self.class_map), axis=-1)


def _as_numpy_array(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a Torch or NumPy array into a detached NumPy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _normalize_single_patch_output(
    output: np.ndarray | torch.Tensor,
    *,
    expected_channels: int,
) -> np.ndarray:
    """Normalize one KongNet patch output into ``(H, W, C)`` format."""
    array = _as_numpy_array(output)
    if array.ndim != 3:
        msg = (
            "KongNet post-processing expects a single patch output with shape "
            "`(C, H, W)` or `(H, W, C)`."
        )
        raise ValueError(msg)

    if array.shape[0] == expected_channels:
        return np.moveaxis(array, 0, -1)
    if array.shape[-1] == expected_channels:
        return array

    msg = (
        "KongNet patch output does not match the expected number of full-head "
        f"channels ({expected_channels})."
    )
    raise ValueError(msg)


def _resolve_kongnet_class_ids(
    *,
    num_heads: int,
    class_dict: dict | None,
    class_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve output class ids for CoNIC-style dense label maps."""
    if class_ids is not None:
        resolved = tuple(int(class_id) for class_id in class_ids)
    elif class_dict is not None:
        ordered_keys = tuple(int(key) for key in sorted(class_dict))
        resolved = (
            ordered_keys
            if len(ordered_keys) == num_heads and all(key > 0 for key in ordered_keys)
            else tuple(range(1, num_heads + 1))
        )
    else:
        resolved = tuple(range(1, num_heads + 1))

    if len(resolved) != num_heads:
        msg = "`class_ids` must provide one positive class id per KongNet head."
        raise ValueError(msg)
    if len(set(resolved)) != len(resolved) or any(class_id <= 0 for class_id in resolved):
        msg = "KongNet dense class ids must be unique positive integers."
        raise ValueError(msg)
    return resolved


def extract_kongnet_component_maps(
    model: KongNet,
    output: np.ndarray | torch.Tensor,
    *,
    from_logits: bool = True,
) -> KongNetComponentMaps:
    """Extract per-class mask/boundary/centroid maps from one full KongNet patch.

    Args:
        model:
            The KongNet model instance describing the concatenated output layout.
        output:
            A single patch output in ``(C, H, W)`` or ``(H, W, C)`` format.
            This must be the *full-head* output of ``KongNet.forward``. The
            centroid-only output emitted by ``KongNet.infer_batch`` is not
            sufficient for instance post-processing because it lacks mask and
            boundary channels.
        from_logits:
            If ``True``, apply a sigmoid activation before unpacking channels.

    Returns:
        KongNetComponentMaps:
            Grouped ``mask``, ``boundary``, and ``centroid`` maps with shape
            ``(H, W, num_heads)``.

    """
    expected_channels = int(sum(model.num_channels_per_head))
    output_hwc = _normalize_single_patch_output(
        output,
        expected_channels=expected_channels,
    )

    if any(int(num_channels) < 3 for num_channels in model.num_channels_per_head):
        msg = (
            "KongNet instance post-processing expects every selected head to expose "
            "at least three channels ordered as `(mask, boundary, centroid)`."
        )
        raise ValueError(msg)

    output_hwc = output_hwc.astype(np.float32, copy=False)
    if from_logits:
        output_hwc = 1.0 / (1.0 + np.exp(-output_hwc))

    mask_maps = []
    boundary_maps = []
    centroid_maps = []
    for head_spec in model.training_output_spec:
        head_output = output_hwc[..., head_spec.channel_slice]
        mask_maps.append(head_output[..., 0])
        boundary_maps.append(head_output[..., 1])
        centroid_maps.append(head_output[..., 2])

    return KongNetComponentMaps(
        mask=np.stack(mask_maps, axis=-1),
        boundary=np.stack(boundary_maps, axis=-1),
        centroid=np.stack(centroid_maps, axis=-1),
    )


def kongnet_instance_postproc(
    component_maps: KongNetComponentMaps,
    *,
    class_ids: Sequence[int] | None = None,
    min_distance: int,
    threshold_abs: float,
    threshold_rel: float | None = None,
    mask_threshold: float = 0.5,
    boundary_weight: float = 1.0,
    class_assignment: Literal["seed", "mean_mask"] = "mean_mask",
    min_instance_size: int = 0,
) -> KongNetInstancePostProcResult:
    """Convert KongNet component maps into CoNIC-style instance/class outputs.

    This performs deterministic seeded watershed-style post-processing using:

    - centroid peaks as instance seeds,
    - mask probabilities as the foreground prior, and
    - boundary probabilities as the separation prior.

    Args:
        component_maps:
            One patch worth of KongNet ``mask``, ``boundary``, and ``centroid`` maps.
        class_ids:
            Dense class ids associated with the class/channel dimension. Defaults to
            ``1..C`` when omitted.
        min_distance:
            Minimum allowed distance separating centroid peaks.
        threshold_abs:
            Absolute centroid peak threshold.
        threshold_rel:
            Optional relative centroid peak threshold.
        mask_threshold:
            Foreground threshold applied to the maximum mask probability.
        boundary_weight:
            Weight applied to boundary probabilities when constructing the
            watershed score map.
        class_assignment:
            ``"seed"`` preserves the seed class for each instance.
            ``"mean_mask"`` assigns the class whose mask channel has the highest
            mean support inside the final instance region.
        min_instance_size:
            Remove predicted instances smaller than this number of pixels.

    Returns:
        KongNetInstancePostProcResult:
            Instance ids, dense class ids, and intermediate maps useful for QC.

    """
    if mask_threshold <= 0 or mask_threshold > 1:
        msg = "`mask_threshold` must be in the interval `(0, 1]`."
        raise ValueError(msg)
    if min_distance < 1:
        msg = "`min_distance` must be a positive integer."
        raise ValueError(msg)
    if min_instance_size < 0:
        msg = "`min_instance_size` must be greater than or equal to zero."
        raise ValueError(msg)

    num_classes = component_maps.num_classes
    resolved_class_ids = _resolve_kongnet_class_ids(
        num_heads=num_classes,
        class_dict=None,
        class_ids=class_ids,
    )

    peak_map = peak_detection_map_overlap(
        component_maps.centroid,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        return_probability=True,
    )
    peak_map = nms_on_detection_maps(peak_map, min_distance=min_distance)

    seed_binary = np.max(peak_map, axis=-1) > 0
    marker_map, marker_count = ndimage.label(seed_binary.astype(np.uint8))
    foreground_mask = np.max(component_maps.mask, axis=-1) >= mask_threshold
    foreground_mask = ndimage.binary_fill_holes(foreground_mask)
    foreground_mask = np.asarray(foreground_mask, dtype=bool)
    foreground_mask[seed_binary] = True

    score_map = np.max(
        component_maps.mask - (boundary_weight * component_maps.boundary),
        axis=-1,
    ).astype(np.float32)

    if marker_count == 0:
        empty_map = np.zeros(component_maps.shape[:2], dtype=np.int32)
        return KongNetInstancePostProcResult(
            instance_map=empty_map,
            class_map=empty_map.copy(),
            peak_map=peak_map.astype(np.float32, copy=False),
            marker_map=empty_map.copy(),
            foreground_mask=foreground_mask,
            score_map=score_map,
            instance_classes={},
        )

    seed_class_by_marker: dict[int, int] = {}
    for marker_id in range(1, marker_count + 1):
        marker_pixels = np.argwhere(marker_map == marker_id)
        pixel_scores = peak_map[marker_pixels[:, 0], marker_pixels[:, 1], :]
        pixel_index = int(np.argmax(np.max(pixel_scores, axis=1)))
        marker_row, marker_col = marker_pixels[pixel_index]
        class_index = int(np.argmax(peak_map[marker_row, marker_col, :]))
        seed_class_by_marker[marker_id] = resolved_class_ids[class_index]

    watershed_labels = watershed(
        -score_map,
        markers=marker_map,
        mask=foreground_mask,
    ).astype(np.int32, copy=False)

    instance_map = np.zeros_like(watershed_labels, dtype=np.int32)
    class_map = np.zeros_like(watershed_labels, dtype=np.int32)
    remapped_marker_map = np.zeros_like(marker_map, dtype=np.int32)
    instance_classes: dict[int, int] = {}

    next_instance_id = 1
    for marker_id in range(1, marker_count + 1):
        instance_mask = watershed_labels == marker_id
        if not np.any(instance_mask):
            continue
        if min_instance_size and int(np.count_nonzero(instance_mask)) < min_instance_size:
            continue

        if class_assignment == "mean_mask":
            mean_mask_scores = component_maps.mask[instance_mask].mean(axis=0)
            class_id = resolved_class_ids[int(np.argmax(mean_mask_scores))]
        else:
            class_id = seed_class_by_marker[marker_id]

        instance_map[instance_mask] = next_instance_id
        class_map[instance_mask] = int(class_id)
        remapped_marker_map[marker_map == marker_id] = next_instance_id
        instance_classes[next_instance_id] = int(class_id)
        next_instance_id += 1

    return KongNetInstancePostProcResult(
        instance_map=instance_map,
        class_map=class_map,
        peak_map=peak_map.astype(np.float32, copy=False),
        marker_map=remapped_marker_map,
        foreground_mask=foreground_mask,
        score_map=score_map,
        instance_classes=instance_classes,
    )


def _empty_kongnet_info_dict() -> dict[str, np.ndarray]:
    """Return an empty MultiTaskSegmentor-compatible instance info dictionary."""
    return {
        "box": np.empty((0, 4), dtype=np.int32),
        "centroid": np.empty((0, 2), dtype=np.float32),
        "contours": np.empty(0, dtype=object),
        "prob": np.empty(0, dtype=np.float32),
        "type": np.empty(0, dtype=np.int32),
    }


def _kongnet_instance_result_to_info_dict(
    result: KongNetInstancePostProcResult,
    *,
    class_ids: Sequence[int],
    offset: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Convert KongNet instance maps into the engine instance info schema."""
    instance_ids = np.unique(result.instance_map)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) == 0:
        return _empty_kongnet_info_dict()

    offset_array = np.asarray(offset, dtype=np.int32)
    boxes: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    contours: list[np.ndarray] = []
    probabilities: list[float] = []
    types: list[int] = []

    class_index_by_id = {class_id: index for index, class_id in enumerate(class_ids)}
    for instance_id in instance_ids:
        instance_mask = result.instance_map == instance_id
        rows, cols = np.where(instance_mask)
        if len(rows) == 0:
            continue

        x_min = int(cols.min())
        y_min = int(rows.min())
        x_max = int(cols.max()) + 1
        y_max = int(rows.max()) + 1
        cropped_mask = instance_mask[y_min:y_max, x_min:x_max].astype(np.uint8)

        contour_list, _ = cv2.findContours(
            cropped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contour_list:
            continue
        contour = max(contour_list, key=cv2.contourArea)[:, 0, :].astype(np.int32)
        min_contour_points = 3
        if contour.shape[0] < min_contour_points:  # pragma: no cover
            continue

        moments = cv2.moments(cropped_mask)
        if moments["m00"] == 0:  # pragma: no cover
            continue

        class_id = int(result.instance_classes.get(int(instance_id), 0))
        class_index = class_index_by_id.get(class_id)
        if class_index is None:
            prob = 0.0
        else:
            class_peak_map = result.peak_map[..., class_index]
            prob = float(np.max(class_peak_map[instance_mask]))

        contour += np.asarray([x_min, y_min], dtype=np.int32)
        contour += offset_array[None]

        centroid = np.asarray(
            [
                moments["m10"] / moments["m00"] + x_min,
                moments["m01"] / moments["m00"] + y_min,
            ],
            dtype=np.float32,
        )
        centroid += offset_array

        boxes.append(
            np.asarray([x_min, y_min, x_max, y_max], dtype=np.int32)
            + np.concatenate([offset_array, offset_array])
        )
        centroids.append(centroid)
        contours.append(contour)
        probabilities.append(prob)
        types.append(class_id)

    if not contours:
        return _empty_kongnet_info_dict()

    contour_array = np.empty(len(contours), dtype=object)
    contour_array[:] = contours

    return {
        "box": np.stack(boxes).astype(np.int32, copy=False),
        "centroid": np.stack(centroids).astype(np.float32, copy=False),
        "contours": contour_array,
        "prob": np.asarray(probabilities, dtype=np.float32),
        "type": np.asarray(types, dtype=np.int32),
    }


def _normalize_kongnet_head_name(name: str, fallback: str) -> str:
    """Normalize a head name into a stable snake-case key."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return normalized or fallback


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
                Number of input channels.
            skip_channels (int):
                Number of channels from skip connection.
            out_channels (int):
                Number of output channels.
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
        encoder_channels (list[int]):
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
            encoder_channels (list[int]):
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
    Please cite the paper [1], if you use this model.

    .. list-table:: KongNet detection performance (FROC)
        on the MONKEY Challenge Final Leaderboard [2]
        :widths: 15 15 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Overall Inflammatory
          - Lymphocytes
          - Monocytes
        * - KongNet_MONKEY_1
          - 0.3930
          - 0.4624
          - 0.2392

    .. list-table:: KongNet detection performance (F1)
        on the MIDOG 2025 Challenge Final Leaderboard [3]
        :widths: 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Mitotic Figures
        * - KongNet_Det_MIDOG_1
          - 0.7400

    .. list-table:: KongNet detection performance (F1)
        on the PUMA Challenge Final Leaderboard Track 1 [4]
        :widths: 15 15 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Tumour Cells
          - Lymphocytes
          - Other
        * - KongNet_PUMA_T1_3
          - 0.7948
          - 0.6746
          - 0.4704

    .. list-table:: KongNet detection performance (F1)
        on the PUMA Challenge Final Leaderboard Track 2 [4]
        :widths: 15 15 15 15 15 15 15 15 15 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Tumour Cells
          - Stroma Cells
          - Apoptotic Cells
          - Epithelium Cells
          - Histiocytes
          - Lymphocytes
          - Neutrophils
          - Endothelial Cells
          - Melanophages
          - Plasma Cells
        * - KongNet_PUMA_T1_3
          - 0.7952
          - 0.2927
          - 0.1170
          - 0.0707
          - 0.2154
          - 0.6642
          - 0.0361
          - 0.2123
          - 0.1931
          - 0.0595

    .. list-table:: KongNet detection performance (F1) on the PanNuke Dataset [5]
        :widths: 15 15 15 15 15 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Overall
          - Neoplastic Cells
          - Inflammatory Cells
          - Epithelial Cells
          - Connective Cells
          - Dead Cells
        * - KongNet_CoNIC_1
          - 0.84
          - 0.71
          - 0.72
          - 0.65
          - 0.70
          - 0.59

    .. list-table:: KongNet detection performance (F1) on the CoNIC Dataset [6]
        :widths: 15 15 15 15 15 15 15
        :header-rows: 1
        :align: left

        * - Model name
          - Neutrophils
          - Epithelial Cells
          - Lymphocytes
          - Plasma Cells
          - Eosinophils
          - Connective Cells
        * - KongNet_CoNIC_1
          - 0.510
          - 0.818
          - 0.707
          - 0.596
          - 0.591
          - 0.695


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
        class_dict:
            Optional dictionary mapping class names to indices
        tile_shape:
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

        [6]  S. Graham et al.,
        “Conic challenge: Pushing the frontiers of nuclear detection,
        segmentation, classification and counting,” Medical Image Analysis,
        vol. 92, p. 103047, 2024.

    """

    @staticmethod
    def _validate_target_channels(
        num_channels_per_head: list[int],
        target_channels: list[int],
    ) -> None:
        """Validate that target channels resolve inside the concatenated output."""
        total_channels = int(sum(num_channels_per_head))
        invalid_channels = [
            int(channel)
            for channel in target_channels
            if int(channel) < 0 or int(channel) >= total_channels
        ]
        if invalid_channels:
            msg = (
                "All `target_channels` must be valid indices into the concatenated "
                f"KongNet output with {total_channels} channels."
            )
            raise ValueError(msg)

        if len(set(target_channels)) != len(target_channels):
            msg = "`target_channels` must not contain duplicate indices."
            raise ValueError(msg)

    @classmethod
    def _build_training_output_spec(
        cls,
        *,
        num_channels_per_head: list[int],
        target_channels: list[int],
        class_dict: dict | None,
    ) -> tuple[KongNetOutputHeadSpec, ...]:
        """Build per-head metadata for training and downstream tooling."""
        cls._validate_target_channels(num_channels_per_head, target_channels)

        output_head_specs: list[KongNetOutputHeadSpec] = []
        for head_index, head_num_channels in enumerate(num_channels_per_head):
            channel_start = sum(num_channels_per_head[:head_index])
            channel_end = channel_start + int(head_num_channels)
            head_target_channels = tuple(
                channel
                for channel in target_channels
                if channel_start <= int(channel) < channel_end
            )
            output_head_specs.append(
                KongNetOutputHeadSpec(
                    index=head_index,
                    name=f"head_{head_index}",
                    display_name=f"Head {head_index}",
                    channel_slice=slice(channel_start, channel_end),
                    num_channels=int(head_num_channels),
                    target_channels=head_target_channels,
                    target_channel_offsets=tuple(
                        int(channel) - channel_start for channel in head_target_channels
                    ),
                ),
            )

        if class_dict:
            ordered_class_names = [str(class_dict[index]) for index in sorted(class_dict)]
            targeted_head_indices = [
                head.index for head in output_head_specs if head.has_target_channels
            ]

            labels_by_head_index: dict[int, str] = {}
            if len(ordered_class_names) == len(output_head_specs):
                labels_by_head_index = {
                    head_index: ordered_class_names[head_index]
                    for head_index in range(len(output_head_specs))
                }
            elif len(ordered_class_names) == len(targeted_head_indices):
                labels_by_head_index = {
                    head_index: ordered_class_names[position]
                    for position, head_index in enumerate(targeted_head_indices)
                }

            used_names: set[str] = set()
            resolved_head_specs: list[KongNetOutputHeadSpec] = []
            for head in output_head_specs:
                display_name = labels_by_head_index.get(head.index, head.display_name)
                fallback_name = f"head_{head.index}"
                normalized_name = _normalize_kongnet_head_name(
                    display_name,
                    fallback_name,
                )
                if normalized_name in used_names:
                    normalized_name = f"{normalized_name}_{head.index}"
                used_names.add(normalized_name)
                resolved_head_specs.append(
                    KongNetOutputHeadSpec(
                        index=head.index,
                        name=normalized_name,
                        display_name=display_name,
                        channel_slice=head.channel_slice,
                        num_channels=head.num_channels,
                        target_channels=head.target_channels,
                        target_channel_offsets=head.target_channel_offsets,
                    ),
                )
            output_head_specs = resolved_head_specs

        return tuple(output_head_specs)

    def __init__(
        self: KongNet,
        num_heads: int,
        num_channels_per_head: list[int],
        target_channels: list[int],
        min_distance: int,
        threshold_abs: float,
        tile_shape: IntPair = (2048, 2048),
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
            tile_shape (IntPair):
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
        self._validate_target_channels(num_channels_per_head, target_channels)

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
        self.num_heads = int(num_heads)
        self.num_channels_per_head = [int(channels) for channels in num_channels_per_head]
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.target_channels = [int(channel) for channel in target_channels]
        self.class_dict = class_dict
        self.tile_shape = tile_shape
        self._training_output_spec = self._build_training_output_spec(
            num_channels_per_head=self.num_channels_per_head,
            target_channels=self.target_channels,
            class_dict=self.class_dict,
        )

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
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image.astype(np.float32, copy=False) / np.float32(255.0)
        return (image - mean) / std

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

    @property
    def training_output_spec(self) -> tuple[KongNetOutputHeadSpec, ...]:
        """Return named metadata for KongNet's concatenated output heads."""
        return self._training_output_spec

    def resolve_instance_class_ids(
        self: KongNet,
        *,
        class_ids: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        """Resolve dense positive class ids for CoNIC-style output maps."""
        return _resolve_kongnet_class_ids(
            num_heads=len(self.training_output_spec),
            class_dict=self.class_dict,
            class_ids=class_ids,
        )

    def extract_component_maps(
        self: KongNet,
        output: np.ndarray | torch.Tensor,
        *,
        from_logits: bool = True,
    ) -> KongNetComponentMaps:
        """Extract per-class mask, boundary, and centroid maps from one patch."""
        return extract_kongnet_component_maps(
            self,
            output,
            from_logits=from_logits,
        )

    def postproc_instance_class_maps(
        self: KongNet,
        output: np.ndarray | torch.Tensor,
        *,
        from_logits: bool = True,
        class_ids: Sequence[int] | None = None,
        min_distance: int | None = None,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        mask_threshold: float = 0.5,
        boundary_weight: float = 1.0,
        class_assignment: Literal["seed", "mean_mask"] = "mean_mask",
        min_instance_size: int = 0,
    ) -> KongNetInstancePostProcResult:
        """Build CoNIC-style instance/class maps from one full KongNet patch output.

        The input must be the full per-head output of ``KongNet.forward``.
        ``KongNet.infer_batch`` only emits centroid channels and therefore cannot
        drive instance segmentation on its own.

        """
        component_maps = self.extract_component_maps(output, from_logits=from_logits)
        return kongnet_instance_postproc(
            component_maps,
            class_ids=self.resolve_instance_class_ids(class_ids=class_ids),
            min_distance=self.min_distance if min_distance is None else min_distance,
            threshold_abs=self.threshold_abs if threshold_abs is None else threshold_abs,
            threshold_rel=threshold_rel,
            mask_threshold=mask_threshold,
            boundary_weight=boundary_weight,
            class_assignment=class_assignment,
            min_instance_size=min_instance_size,
        )

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
        imgs = imgs.to(device=device, dtype=torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # unwrap DataParallel/DDP if present (happens in multi-gpu settings)
        target_channels = getattr(model, "module", model).target_channels

        with torch.inference_mode():
            logits = model(imgs)
            target_logits = logits[:, target_channels, :, :]
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


class KongNetSegmentor(KongNet):
    """KongNet variant exposing a ``MultiTaskSegmentor``-compatible contract.

    ``KongNet`` keeps its historical centroid-only inference contract for
    detection engines. This subclass emits full per-head probability maps and
    converts mask/boundary/centroid outputs into instance segmentation metadata
    that can be saved as an AnnotationStore by ``MultiTaskSegmentor``.
    """

    task_name = "nuclei_segmentation"

    @staticmethod
    def infer_batch(
        model: KongNetSegmentor,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> tuple[np.ndarray, ...]:
        """Run full-head KongNet inference for segmentation engines.

        Returns one ``(N, H, W, C_head)`` probability array per KongNet head.
        """
        model = model.to(device)
        model.eval()

        imgs = batch_data.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.nn.functional.sigmoid(logits)
            probs = probs.permute(0, 2, 3, 1).contiguous()

        probs_np = probs.cpu().numpy()
        return tuple(
            probs_np[..., head_spec.channel_slice]
            for head_spec in model.training_output_spec
        )

    def postproc(
        self: KongNetSegmentor,
        raw_maps: list[np.ndarray],
        offset: tuple[int, int] = (0, 0),
        *,
        min_distance: int | None = None,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        mask_threshold: float = 0.5,
        boundary_weight: float = 1.0,
        class_assignment: Literal["seed", "mean_mask"] = "mean_mask",
        min_instance_size: int = 0,
    ) -> tuple[dict[str, Any], ...]:
        """Convert full-head raw maps into instance segmentation output.

        Args:
            raw_maps:
                One probability map per KongNet head, ordered like
                ``KongNetSegmentor.infer_batch`` output.
            offset:
                ``(x, y)`` offset applied to contours, boxes, and centroids.
            min_distance:
                Minimum seed distance for peak detection. Uses the model default
                when ``None``.
            threshold_abs:
                Absolute seed threshold. Uses the model default when ``None``.
            threshold_rel:
                Optional relative seed threshold.
            mask_threshold:
                Probability threshold for foreground masks.
            boundary_weight:
                Weight applied to boundary evidence before watershed.
            class_assignment:
                Strategy used to assign each instance to a class head.
            min_instance_size:
                Remove predicted instances smaller than this many pixels.

        Returns:
            Tuple containing a single task dictionary compatible with
            ``MultiTaskSegmentor``.

        """
        if len(raw_maps) != len(self.training_output_spec):
            msg = (
                "KongNetSegmentor post-processing expects one raw map per "
                f"KongNet head; got {len(raw_maps)} maps for "
                f"{len(self.training_output_spec)} heads."
            )
            raise ValueError(msg)

        head_maps = [
            np.asarray(head_map.compute() if hasattr(head_map, "compute") else head_map)
            for head_map in raw_maps
        ]
        output_shape = head_maps[0].shape[:2]
        if any(
            head_map.shape[0] == 0 or head_map.shape[1] == 0
            for head_map in head_maps
        ):
            return (
                {
                    "task_type": self.task_name,
                    "predictions": np.zeros(output_shape, dtype=np.int32),
                    "info_dict": _empty_kongnet_info_dict(),
                    "seg_type": "instance",
                },
            )

        full_output = np.concatenate(head_maps, axis=-1)
        result = self.postproc_instance_class_maps(
            full_output,
            from_logits=False,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            mask_threshold=mask_threshold,
            boundary_weight=boundary_weight,
            class_assignment=class_assignment,
            min_instance_size=min_instance_size,
        )
        class_ids = self.resolve_instance_class_ids()
        return (
            {
                "task_type": self.task_name,
                "predictions": result.instance_map,
                "info_dict": _kongnet_instance_result_to_info_dict(
                    result,
                    class_ids=class_ids,
                    offset=offset,
                ),
                "seg_type": "instance",
            },
        )
