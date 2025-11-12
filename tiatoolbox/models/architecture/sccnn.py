"""Define SCCNN architecture.

Sirinukunwattana, Korsuk, et al.
"Locality sensitive deep learning for detection and classification
of nuclei in routine colon cancer histology images."
IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

"""

from __future__ import annotations

from collections import OrderedDict

import dask.array as da
import numpy as np
import pandas as pd
import torch
from torch import nn

from tiatoolbox import logger

import torch
from skimage.feature import peak_local_max
from torch import nn

from tiatoolbox.models.models_abc import ModelABC


class SCCNN(ModelABC):
    """Initialize SCCNN [1].

    The following models have been included in tiatoolbox:

    1. `sccnn-crchisto`:
        This model is trained on `CRCHisto dataset
        <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/>`_
    2. `sccnn-conic`:
        This model is trained on `CoNIC dataset
        <https://conic-challenge.grand-challenge.org/evaluation/challenge/leaderboard//>`_
        Centroids of ground truth masks were used to train this model.
        The results are reported on the whole test data set including preliminary
        and final set.

    The original model was implemented in Matlab. The model has been reimplemented
    in PyTorch for Python compatibility. The original model uses HRGB as input,
    where 'H' represents hematoxylin. The model has been modified to rely on RGB
    image as input.


    The tiatoolbox model should produce the following results on the following datasets
    using 8 pixels as radius for true detection:

    .. list-table:: SCCNN performance
       :widths: 15 15 15 15 15
       :header-rows: 1

       * - Model name
         - Data set
         - Precision
         - Recall
         - F1Score
       * - sccnn-crchisto
         - CRCHisto
         - 0.82
         - 0.80
         - 0.81
       * - sccnn-conic
         - CoNIC
         - 0.79
         - 0.79
         - 0.79

    Args:
        num_input_channels (int):
            Number of channels in input. default=3.
        patch_output_shape tuple(int):
            Defines output height and output width. default=(13, 13).
        radius (int):
            Radius for nucleus detection, default = 12.
        min_distance (int):
            The minimal allowed distance separating peaks.
            To find the maximum number of peaks, use `min_distance=1`, default=6.
        threshold_abs (float):
            Minimum intensity of peaks, default=0.20.

    References:
        [1] Sirinukunwattana, Korsuk, et al.
        "Locality sensitive deep learning for detection and classification
        of nuclei in routine colon cancer histology images."
        IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

    """

    def __init__(
        self: SCCNN,
        num_input_channels: int = 3,
        patch_output_shape: tuple[int, int] = (13, 13),
        radius: int = 12,
        min_distance: int = 6,
        threshold_abs: float = 0.20,
        postproc_tile_shape: tuple[int, int] = (2048, 2048),
    ) -> None:
        """Initialize :class:`SCCNN`."""
        super().__init__()
        out_height = patch_output_shape[0]
        out_width = patch_output_shape[1]
        self.in_ch = num_input_channels
        self.out_height = out_height
        self.out_width = out_width
        self.postproc_tile_shape = postproc_tile_shape

        # Create mesh grid and convert to 3D vector
        x, y = torch.meshgrid(
            torch.arange(start=0, end=out_height),
            torch.arange(start=0, end=out_width),
            indexing="ij",
        )

        self.register_buffer("xv", torch.unsqueeze(x, dim=0).type(torch.float32))
        self.register_buffer("yv", torch.unsqueeze(y, dim=0).type(torch.float32))

        self.radius = radius
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs

        def conv_act_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
        ) -> torch.nn.ModuleDict:
            """Convolution and activation branch for SCCNN.

            This module combines the convolution and activation blocks in a single
            function.

            Args:
                in_channels (int):
                    Number of channels in input.
                out_channels (int):
                    Number of required channels in output.
                kernel_size (int):
                    Kernel size of convolution filter.

            Returns:
                torch.nn.ModuleDict:
                    Module dictionary.

            """
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(),
            )

            return nn.ModuleDict(module_dict)

        def spatially_constrained_layer1(
            in_channels: int,
            out_channels: int,
        ) -> torch.nn.ModuleDict:
            """Spatially constrained layer.

            Takes fully connected layer and returns outputs for creating probability
            map for the output. The output is Tensor is 3-dimensional where it defines
            the row, height of the centre of nucleus and its confidence value.

            Args:
                in_channels (int):
                    Number of channels in input.
                out_channels (int):
                    Number of required channels in output.

            Returns:
                torch.nn.ModuleDict:
                    Module dictionary.

            """
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
            return nn.ModuleDict(module_dict)

        module_dict = OrderedDict()
        module_dict["l1"] = conv_act_block(num_input_channels, 30, 2)
        module_dict["pool1"] = nn.MaxPool2d(2, padding=0)
        module_dict["l2"] = conv_act_block(30, 60, 2)
        module_dict["pool2"] = nn.MaxPool2d(2, padding=0)
        module_dict["l3"] = conv_act_block(60, 90, 3)
        module_dict["l4"] = conv_act_block(90, 1024, 5)
        module_dict["dropout1"] = nn.Dropout2d(p=0.5)
        module_dict["l5"] = conv_act_block(1024, 512, 1)
        module_dict["dropout2"] = nn.Dropout2d(p=0.5)
        module_dict["sc"] = spatially_constrained_layer1(512, 3)

        self.layer = nn.ModuleDict(module_dict)

    def spatially_constrained_layer2(
        self: SCCNN,
        sc1_0: torch.Tensor,
        sc1_1: torch.Tensor,
        sc1_2: torch.Tensor,
    ) -> torch.Tensor:
        """Spatially constrained layer 2.

        Estimates row, column and height for sc2 layer mapping.

        Args:
            sc1_0 (torch.Tensor):
                Output of spatially_constrained_layer1 estimating
                the x position of the nucleus.
            sc1_1 (torch.Tensor):
                Output of spatially_constrained_layer1 estimating
                the y position of the nucleus.
            sc1_2 (torch.Tensor):
                Output of spatially_constrained_layer1 estimating
                the confidence in nucleus detection.

        Returns:
            :class:`torch.Tensor`:
                Probability map using the estimates from
                spatially_constrained_layer1.

        """
        x = torch.tile(self.xv, dims=[sc1_0.size(0), 1, 1, 1])  # Tile for batch size
        y = torch.tile(self.yv, dims=[sc1_0.size(0), 1, 1, 1])
        xvr = (x - sc1_0) ** 2
        yvc = (y - sc1_1) ** 2
        out_map = xvr + yvc
        out_map_threshold = torch.lt(out_map, self.radius).type(torch.float32)
        denominator = 1 + (out_map / 2)
        sc2 = sc1_2 / denominator
        return sc2 * out_map_threshold

    @staticmethod
    def preproc(image: torch.Tensor) -> torch.Tensor:
        """Transforming network input to desired format.

        This method is model and dataset specific, meaning that it can be replaced by
        user's desired transform function before training/inference.

        Args:
            image (torch.Tensor): Input images, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The transformed input.

        """
        return image / 255.0

    def forward(  # skipcq: PYL-W0221
        self: SCCNN,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
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

        def spatially_constrained_layer1(
            layer: torch.nn.Module,
            in_tensor: torch.Tensor,
            out_height: int = 13,
            out_width: int = 13,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Spatially constrained layer 1.

            Estimates row, column and height for
            `spatially_constrained_layer2` layer mapping.

            Args:
                layer (torch.nn.Module):
                    Torch layer as ModuleDict.
                in_tensor (torch.Tensor):
                    Input Tensor.
                out_height (int):
                    Output height.
                out_width (int):
                    Output Width

            Returns:
                tuple:
                    Parameters for the requested nucleus location:
                    - torch.Tensor - Row location for the centre of the nucleus.
                    - torch.Tensor - Column location for the centre of the nucleus.
                    - torch.Tensor - Peak value for the probability function indicating
                    confidence value for the estimate.

            """
            sigmoid = layer["conv1"](in_tensor)
            sigmoid0 = sigmoid[:, 0:1, :, :] * (out_height - 1)
            sigmoid1 = sigmoid[:, 1:2, :, :] * (out_width - 1)
            sigmoid2 = sigmoid[:, 2:3, :, :]
            return sigmoid0, sigmoid1, sigmoid2

        input_tensor = self.preproc(input_tensor)
        l1 = self.layer["l1"]["conv1"](input_tensor)
        p1 = self.layer["pool1"](l1)
        l2 = self.layer["l2"]["conv1"](p1)
        p2 = self.layer["pool1"](l2)
        l3 = self.layer["l3"]["conv1"](p2)
        l4 = self.layer["l4"]["conv1"](l3)
        drop1 = self.layer["dropout1"](l4)
        l5 = self.layer["l5"]["conv1"](drop1)
        drop2 = self.layer["dropout2"](l5)
        s1_sigmoid0, s1_sigmoid1, s1_sigmoid2 = spatially_constrained_layer1(
            self.layer["sc"],
            drop2,
        )
        return self.spatially_constrained_layer2(s1_sigmoid0, s1_sigmoid1, s1_sigmoid2)

    def postproc(
        self: SCCNN, prediction_map: da.Array, prediction_shape: tuple, dtype: np.dtype
    ) -> pd.DataFrame:
        """Post-processing script for SCCNN.

        Post-process predicted probability map of the input image.
        Performs peak detection, then non-maximum suppression.
        Returns a pandas DataFrame containing detected nuclei coordinates [x, y, type, prob].

        Args:
            prediction_map (da.array):
                Predicted probability map (HxWx1) of the entire input image.
            prediction_shape (tuple):
                Shape of the prediction map.
            dtype (np.dtype):
                Data type of the prediction map.

        Returns:
            detected_nuclei (pandas.DataFrame):
                Detected nuclei coordinates stored in a pandas DataFrame.

        """
        depth = {0: self.min_distance, 1: self.min_distance, 2: 0}

        # print("maxmin debug:")  # --- DEBUG ---
        # lazy_max = prediction_map.max()
        # max_value = lazy_max.compute()
        # lazy_min = prediction_map.min()
        # min_value = lazy_min.compute()
        # print(f"lazy_max: {max_value}, lazy_min: {min_value}")

        rechunked_prediction_map = prediction_map.rechunk(
            (self.postproc_tile_shape[0], self.postproc_tile_shape[1], -1)
        )
        print(f"rechunked_prediction_map.shape: {rechunked_prediction_map.shape}")
        print(f"rechunked_prediction_map.chunks: {rechunked_prediction_map.chunks}")

        scores = da.map_overlap(
            rechunked_prediction_map,
            peak_detection_mapoverlap,
            depth=depth,
            boundary=0,
            dtype=dtype,
            block_info=True,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            depth_h=self.min_distance,
            depth_w=self.min_distance,
            calculate_probabilities=False,
        )
        ddf = centroids_map_to_dask_dataframe(scores, x_offset=0, y_offset=0)
        pandas_df = ddf.compute()

        logger.info(f"Total detections before NMS: {len(pandas_df)}")
        detected_nuclei = nucleus_detection_nms(pandas_df, radius=self.min_distance)
        logger.info(f"Total detections after NMS: {len(detected_nuclei)}")

        return detected_nuclei

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str,
    ) -> np.ndarray:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch I/O
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`numpy.ndarray` or :class:`torch.Tensor`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list of :class:`numpy.ndarray`:
                Output probability map.

        """
        patch_imgs = batch_data

        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)
        # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred = model(patch_imgs_gpu)

        pred = pred.permute(0, 2, 3, 1).contiguous()
        if torch.max(pred) > 0:
            print(torch.max(pred), torch.min(pred))  # --- DEBUG ---
        return pred.cpu().numpy()
