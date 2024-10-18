"""Define MapDe architecture.

Raza, Shan E Ahmed, et al. "Deconvolving convolutional neural network
for cell detection." 2019 IEEE 16th International Symposium on Biomedical
Imaging (ISBI 2019). IEEE, 2019.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from skimage.feature import peak_local_max

from tiatoolbox.models.architecture.micronet import MicroNet


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
    ) -> None:
        """Initialize :class:`MapDe`."""
        super().__init__(
            num_output_channels=num_classes * 2,
            num_input_channels=num_input_channels,
            out_activation="relu",
        )

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

        dist_filter = np.expand_dims(dist_filter, axis=(0, 1))  # NCHW
        dist_filter = np.repeat(dist_filter, repeats=num_classes * 2, axis=1)

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
    def postproc(self: MapDe, prediction_map: np.ndarray) -> np.ndarray:
        """Post-processing script for MicroNet.

        Performs peak detection and extracts coordinates in x, y format.

        Args:
            prediction_map (ndarray):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pixel-wise nuclear instance segmentation
                prediction.

        """
        coordinates = peak_local_max(
            np.squeeze(prediction_map[0], axis=2),
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            exclude_border=False,
        )
        return np.fliplr(coordinates)

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> list[np.ndarray]:
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
        pred = pred.cpu().numpy()

        return [
            pred,
        ]
