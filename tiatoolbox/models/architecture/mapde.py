"""Define MapDe architecture.

Raza, Shan E Ahmed, et al. "Deconvolving convolutional neural network
for cell detection." 2019 IEEE 16th International Symposium on Biomedical
Imaging (ISBI 2019). IEEE, 2019.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from tiatoolbox.models.architecture.micronet import MicroNet
from tiatoolbox.models.architecture.utils import peak_detection_map_overlap


class MapDe(MicroNet):
    """Initialize MapDe [1].

    The following models have been included in tiatoolbox:

    1. `mapde-crchisto`:
        This model is trained on `CRCHisto dataset
        <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/>`_
    2. `mapde-conic`:
        This model is trained on `CoNIC dataset
        <https://conic-challenge.grand-challenge.org/evaluation/challenge/leaderboard//>`_
        Centroids of ground truth masks were used to train this model.
        The results are reported on the whole test data set including preliminary
        and final set.

    The tiatoolbox model should produce the following results on the following datasets
    using 8 pixels as radius for true detection:

    .. list-table:: MapDe performance
       :widths: 15 15 15 15 15
       :header-rows: 1

       * - Model name
         - Data set
         - Precision
         - Recall
         - F1Score
       * - mapde-crchisto
         - CRCHisto
         - 0.81
         - 0.82
         - 0.81
       * - mapde-conic
         - CoNIC
         - 0.85
         - 0.85
         - 0.85

    Args:
        num_input_channels (int):
            Number of channels in input. default=3.
        num_classes (int):
            Number of cell classes to identify. default=1.
        min_distance (int):
            The minimal allowed distance separating peaks.
            To find the maximum number of peaks, use `min_distance=1`, default=6.
        threshold_abs (float):
            Minimum intensity of peaks, default=0.20.

    References:
        [1] Raza, Shan E. Ahmed, et al. "Deconvolving convolutional neural network
        for cell detection." 2019 IEEE 16th International Symposium on Biomedical
        Imaging (ISBI 2019). IEEE, 2019.

    """

    def __init__(
        self: MapDe,
        num_input_channels: int = 3,
        min_distance: int = 4,
        threshold_abs: float = 250,
        num_classes: int = 1,
        tile_shape: tuple[int, int] = (2048, 2048),
        class_dict: dict[int, str] | None = None,
    ) -> None:
        """Initialize :class:`MapDe`."""
        super().__init__(
            num_output_channels=num_classes * 2,
            num_input_channels=num_input_channels,
            out_activation="relu",
        )
        self.class_dict = class_dict
        self.tile_shape = tile_shape

        dist_filter = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.1055728,
                    0.17537889,
                    0.2,
                    0.17537889,
                    0.1055728,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.1514719,
                    0.27888975,
                    0.36754447,
                    0.4,
                    0.36754447,
                    0.27888975,
                    0.1514719,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.1055728,
                    0.27888975,
                    0.43431458,
                    0.5527864,
                    0.6,
                    0.5527864,
                    0.43431458,
                    0.27888975,
                    0.1055728,
                    0.0,
                ],
                [
                    0.0,
                    0.17537889,
                    0.36754447,
                    0.5527864,
                    0.71715724,
                    0.8,
                    0.71715724,
                    0.5527864,
                    0.36754447,
                    0.17537889,
                    0.0,
                ],
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                [
                    0.0,
                    0.17537889,
                    0.36754447,
                    0.5527864,
                    0.71715724,
                    0.8,
                    0.71715724,
                    0.5527864,
                    0.36754447,
                    0.17537889,
                    0.0,
                ],
                [
                    0.0,
                    0.1055728,
                    0.27888975,
                    0.43431458,
                    0.5527864,
                    0.6,
                    0.5527864,
                    0.43431458,
                    0.27888975,
                    0.1055728,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.1514719,
                    0.27888975,
                    0.36754447,
                    0.4,
                    0.36754447,
                    0.27888975,
                    0.1514719,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.1055728,
                    0.17537889,
                    0.2,
                    0.17537889,
                    0.1055728,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # For conv2d, filter shape = (out_channels, in_channels//groups, H, W)
        dist_filter = np.expand_dims(dist_filter, axis=(0, 1))
        dist_filter = np.repeat(dist_filter, repeats=num_classes * 2, axis=1)
        # Need to repeat for out_channels
        dist_filter = np.repeat(dist_filter, repeats=num_classes, axis=0)

        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.register_buffer(
            "dist_filter",
            torch.from_numpy(dist_filter.astype(np.float32)),
        )
        self.dist_filter.requires_grad = False

    def forward(self: MapDe, input_tensor: torch.Tensor) -> torch.Tensor:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            torch.Tensor:
                Output map for cell detection. Peak detection should be applied
                to this output for cell detection.

        """
        logits, _, _, _ = super().forward(input_tensor)
        out = F.conv2d(logits, self.dist_filter, padding="same")
        return F.relu(out)

    #  skipcq: PYL-W0221  # noqa: ERA001
    def postproc(
        self: MapDe,
        block: np.ndarray,
        min_distance: int | None = None,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        block_info: dict | None = None,
        depth_h: int = 0,
        depth_w: int = 0,
    ) -> np.ndarray:
        """MapDe post-processing function.

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
            out: NumPy array (H, W, C) with 1.0 at peaks, 0 elsewhere.
        """
        min_distance_to_use = (
            self.min_distance if min_distance is None else min_distance
        )
        threshold_abs_to_use = (
            self.threshold_abs if threshold_abs is None else threshold_abs
        )
        return peak_detection_map_overlap(
            block,
            min_distance=min_distance_to_use,
            threshold_abs=threshold_abs_to_use,
            threshold_rel=threshold_rel,
            block_info=block_info,
            depth_h=depth_h,
            depth_w=depth_w,
        )

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch I/O
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`numpy.ndarray`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list(np.ndarray):
                Probability map as numpy array.

        """
        patch_imgs = batch_data

        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        with torch.inference_mode():
            pred = model(patch_imgs_gpu)

        pred = pred.permute(0, 2, 3, 1).contiguous()
        return pred.cpu().numpy()
