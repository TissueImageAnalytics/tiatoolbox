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


def sc2_mapping(out_height, out_width):
    """Spatially Constrained layer 2 mapping.

    Args:
        out_height (int):
            Output height.
        out_width (int):
            Output Width

    Returns:
        tuple of int:
            A mesh grid with matrix indexing.

    """
    x, y = torch.meshgrid(torch.range(0, out_height - 1), torch.range(0, out_width - 1))

    x = torch.unsqueeze(x, dim=0)  # Make 3D vector
    y = torch.unsqueeze(y, dim=0)

    x = x.type(torch.float32)
    y = y.type(torch.float32)

    return x, y


class SCCNN(ModelABC):
    """Initialise SCCNN [1].

    The following models have been included in tiatoolbox:
    1. `sccnn-crchisto`:
        This is trained on `CRCHisto dataset
        <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/>`_ The
        model is retrained in torch as the original model was trained in Matlab. This
        model also relies on RGB image as input. The original model uses HRGB as input,
        where 'H' represents hematoxylin.

    The tiatoolbox model should produce the following results on the CRC dataset using
    12 pixels as radius for true detection:

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
         - 0.74
         - 0.88
         - 0.80

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

    def __init__(self, num_input_channels=3, out_height=13, out_width=13, radius=12):
        super().__init__()
        self.in_ch = num_input_channels
        self.out_height = out_height
        self.out_width = out_width
        self.x, self.y = sc2_mapping(out_height=out_height, out_width=out_width)
        self.radius = radius

        def conv_act_branch(in_ch, out_ch, k_dim):
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(k_dim, k_dim),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(),
            )

            return nn.ModuleDict(module_dict)

        def sc(in_ch, out_ch):
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
            return nn.ModuleDict(module_dict)

        module_dict = OrderedDict()
        module_dict["l1"] = conv_act_branch(num_input_channels, 30, 2)
        module_dict["pool1"] = nn.MaxPool2d(2, padding=0)
        module_dict["l2"] = conv_act_branch(30, 60, 2)
        module_dict["pool2"] = nn.MaxPool2d(2, padding=0)
        module_dict["l3"] = conv_act_branch(60, 90, 3)
        module_dict["l4"] = conv_act_branch(90, 1024, 5)
        module_dict["dropout1"] = nn.Dropout2d(p=0.5)
        module_dict["l5"] = conv_act_branch(1024, 512, 1)
        module_dict["dropout2"] = nn.Dropout2d(p=0.5)
        module_dict["sc"] = sc(512, 3)

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

        def conv_act_branch(layer, in_tensor):
            return layer["conv1"](in_tensor)

        def sc1_layer(layer, in_tensor, out_height=13, out_width=13):
            """Spatially constrained layer 1.

            Estimates row, column and height for sc2 layer mapping.

            Args:
                layer (torch.nn.layer):
                    Torch Layer.
                in_tensor (torch.Tensor):
                    Input Tensor.
                out_height (int):
                    Output height.
                out_width (int):
                    Output Width

            Returns:
                tuple of tensors:
                    Row, Column and height estimates used for sc2 mapping.

            """
            sigmoid = layer["conv1"](in_tensor)
            sigmoid0 = sigmoid[:, 0:1, :, :] * (out_height - 1)
            sigmoid1 = sigmoid[:, 1:2, :, :] * (out_width - 1)
            sigmoid2 = sigmoid[:, 2:3, :, :]
            return sigmoid0, sigmoid1, sigmoid2

        def sc2layer(network, sc1_0, sc1_1, sc1_2):
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
        s1_sigmoid0, s1_sigmoid1, s1_sigmoid2 = sc1_layer(self.layer["sc"], drop2)
        return sc2layer(self, s1_sigmoid0, s1_sigmoid1, s1_sigmoid2)

    @staticmethod
    def postproc(
        prediction_map: np.ndarray, min_distance: int = 6, threshold_abs: float = 0.10
    ):
        """Post-processing script for MicroNet.

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
            np.squeeze(prediction_map, axis=2),
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        return np.fliplr(coordinates)

    @staticmethod
    def preproc(image: np.ndarray):
        """Preprocessing function for MicroNet.

        Performs per image standardization.

        Args:
            image (:class:`numpy.ndarray`):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pre-processed numpy array.

        """
        return torch.from_numpy(image / 255.0)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch I/O
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`numpy.ndarray`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        Returns:
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
