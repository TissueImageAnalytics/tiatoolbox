"""Defines SCCNN architecture.

Sirinukunwattana, Korsuk, et al.
"Locality sensitive deep learning for detection and classification
of nuclei in routine colon cancer histology images."
IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from skimage.feature import peak_local_max

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.utils import misc


class SCCNN(ModelABC):
    """Initialize SCCNN [1].

    The following models have been included in tiatoolbox:

    1. `sccnn-crchisto`:
        This is trained on `CRCHisto dataset
        <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/>`_ The
        model is retrained in torch as the original model was trained in Matlab. This
        model also relies on RGB image as input. The original model uses HRGB as input,
        where 'H' represents hematoxylin.

    The tiatoolbox model should produce the following results on the CRC dataset using
    8 pixels as radius for true detection:

    .. list-table:: MicroNet performance
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

    Args:
        num_input_channels (int):
            Number of channels in input. default=3.
        out_height (int):
            Output height. default=13.
        out_width (int):
            Output width. default=13.
        radius (int):
            Radius for nucleus detection, default = 12.

    References:
        [1] Sirinukunwattana, Korsuk, et al.
        "Locality sensitive deep learning for detection and classification
        of nuclei in routine colon cancer histology images."
        IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

    """

    def __init__(
        self,
        num_input_channels: int = 3,
        out_height: int = 13,
        out_width: int = 13,
        radius: int = 12,
    ):
        super().__init__()
        self.in_ch = num_input_channels
        self.out_height = out_height
        self.out_width = out_width

        # Create mesh grid and convert to 3D vector
        x, y = torch.meshgrid(
            torch.arange(start=0, end=out_height),
            torch.arange(start=0, end=out_width),
            indexing="ij",
        )
        self.x = torch.unsqueeze(x, dim=0).type(torch.float32)
        self.y = torch.unsqueeze(y, dim=0).type(torch.float32)

        self.radius = radius

        def conv_act_block(
            in_channels: int, out_channels: int, kernel_size: int
        ) -> torch.nn.ModuleDict:
            """Convolution and Activation branch for SCCNN.

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
            in_channels, out_channels
        ) -> torch.nn.ModuleDict:
            """Spatially constrained layer.

            Takes fully connected layer and returns outputs for creating probability
            map for the output.

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

    def forward(self, input_tensor: torch.Tensor):  # skipcq: PYL-W0221
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

        def conv_act_branch(
            layer: torch.nn.Module, in_tensor: torch.Tensor
        ) -> torch.Tensor:
            """Applies Convolution and Activation to the input tensor.

            Args:
                layer (torch.nn.ModuleDict):
                    Torch layer as ModuleDict.
                in_tensor (torch.Tensor):
                    Input Tensor.
            Returns:
                torch.Tensor:
                    Torch Tensor after applying convolution and activation.

            """
            return layer["conv1"](in_tensor)

        def spatially_constrained_layer1(
            layer: torch.nn.Module,
            in_tensor: torch.Tensor,
            out_height: int = 13,
            out_width: int = 13,
        ) -> tuple:
            """Spatially constrained layer 1.

            Estimates row, column and height for
            spatially_constrained_layer2 layer mapping.

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
                tuple of :class:`torch.Tensor`:
                    Row, Column and height estimates used for
                    spatially_constrained_layer2 mapping.

            """
            sigmoid = layer["conv1"](in_tensor)
            sigmoid0 = sigmoid[:, 0:1, :, :] * (out_height - 1)
            sigmoid1 = sigmoid[:, 1:2, :, :] * (out_width - 1)
            sigmoid2 = sigmoid[:, 2:3, :, :]
            return sigmoid0, sigmoid1, sigmoid2

        def spatially_constrained_layer2(network, sc1_0, sc1_1, sc1_2) -> torch.Tensor:
            """Spatially constrained layer 2.

            Estimates row, column and height for sc2 layer mapping.

            Args:
                network (:class:`.SCCNN`):
                    An initiated SCCNN class.
                sc1_0 (torch.Tensor):
                    Output of spatially_constrained_layer1 estimating
                    the x position of the nucleus.
                sc1_1 (int):
                    Output of spatially_constrained_layer1 estimating
                    the y position of the nucleus.
                sc1_2 (int):
                    Output of spatially_constrained_layer1 estimating
                    the confidence in nucleus detection.

            Returns:
                :class:`torch.Tensor`:
                    Probability map using the estimates from
                    spatially_constrained_layer1.

            """
            x = torch.tile(
                network.x, dims=[sc1_0.size(0), 1, 1, 1]
            )  # Tile for batch size
            y = torch.tile(network.y, dims=[sc1_0.size(0), 1, 1, 1])
            xvr = (x - sc1_0) ** 2
            yvc = (y - sc1_1) ** 2
            out_map = xvr + yvc
            out_map_threshold = torch.lt(out_map, network.radius).type(torch.float32)
            denominator = 1 + (out_map / 2)
            sc2 = sc1_2 / denominator
            return sc2 * out_map_threshold

        l1 = conv_act_branch(self.layer["l1"], in_tensor=input_tensor)
        p1 = self.layer["pool1"](l1)
        l2 = conv_act_branch(self.layer["l2"], in_tensor=p1)
        p2 = self.layer["pool1"](l2)
        l3 = conv_act_branch(self.layer["l3"], in_tensor=p2)
        l4 = conv_act_branch(self.layer["l4"], in_tensor=l3)
        drop1 = self.layer["dropout1"](l4)
        l5 = conv_act_branch(self.layer["l5"], in_tensor=drop1)
        drop2 = self.layer["dropout2"](l5)
        s1_sigmoid0, s1_sigmoid1, s1_sigmoid2 = spatially_constrained_layer1(
            self.layer["sc"], drop2
        )
        return spatially_constrained_layer2(self, s1_sigmoid0, s1_sigmoid1, s1_sigmoid2)

    @staticmethod
    def postproc(
        prediction_map: np.ndarray, min_distance: int = 6, threshold_abs: float = 0.10
    ) -> np.ndarray:
        """Post-processing script for MicroNet.

        Performs peak detection and extracts coordinates in x, y format.

        Args:
            prediction_map (ndarray):
                Input image of type numpy array.
            min_distance (int):
                The minimal allowed distance separating peaks.
                To find the maximum number of peaks, use `min_distance=1`, default=6.
            threshold_abs (float):
                Minimum intensity of peaks, default=0.10.


        Returns:
            :class:`numpy.ndarray`:
                Pixel-wise nuclear instance segmentation
                prediction.

        """
        coordinates = peak_local_max(
            np.squeeze(prediction_map[0], axis=2),
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        return np.fliplr(coordinates)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Preprocessing function for MicroNet.

        Performs per image standardization.

        Args:
            image (:class:`numpy.ndarray`):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pre-processed numpy array.

        """
        return image / 255.0

    @staticmethod
    def infer_batch(model, batch_data, on_gpu) -> list:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch I/O
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`numpy.ndarray` or :class:`torch.Tensor`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        Returns:
            list of :class:`numpy.ndarray`:
                Output probability map.

        """
        patch_imgs = batch_data

        device = misc.select_device(on_gpu)
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)
        # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred = model(patch_imgs_gpu)

        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.cpu().numpy()

        return [
            pred,
        ]
