"""Define orginal NuClick architecture#

Koohbanani, N. A., Jahanifar, M., Tajadin, N. Z., & Rajpoot, N. (2020).
NuClick: a deep learning framework for interactive segmentation of microscopic images.
Medical Image Analysis, 65, 101771.
"""
import warnings
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import (
    disk,
    reconstruction,
    remove_small_holes,
    remove_small_objects,
)

from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import misc

bn_axis = 1


class ConvBnRelu(nn.Module):
    """Convolution -> Batch Normalization -> ReLu/Sigmoid
    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernel_size (int): Size of the kernel in the convolution layer.
        strds (int): Size of the stride in the convolution layer.
        use_bias (bool): Whether to use bias in the convolution layer.
        dilatation_rate (int): Dilatation rate in the convolution layer.
        activation (str): Name of the activation function to use.
        do_batchnorm (bool): Whether to do batch normalization after the
        convolution layer.
    Returns:
        model (torch.nn.Module): a pytorch model.

    """

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        kernel_size: Union[Tuple[int, int], np.ndarray] = (3, 3),
        strds: Union[Tuple[int, int], np.ndarray] = (1, 1),
        use_bias: bool = False,
        dilatation_rate: Union[Tuple[int, int], np.ndarray] = (1, 1),
        activation: str = "relu",
        do_batchnorm: bool = True,
    ):

        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strds, int):
            strds = (strds, strds)

        self.conv_bn_relu = self.get_block(
            num_input_channels,
            num_output_channels,
            kernel_size,
            strds,
            use_bias,
            dilatation_rate,
            activation,
            do_batchnorm,
        )

    def forward(self, input_tensor):
        """Logic for using layers defined in init.
        This method defines how layers are used in forward operation.
        Args:
            input_tensor (torch.Tensor): Input, the tensor is of the shape NCHW.
        Returns:
            output (torch.Tensor): The inference output.
        """
        return self.conv_bn_relu(input_tensor)

    def get_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        strds,
        use_bias,
        dilatation_rate,
        activation,
        do_batchnorm,
    ):
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strds,
            dilation=dilatation_rate,
            bias=use_bias,
            padding="same",
            padding_mode="zeros",
        )

        torch.nn.init.xavier_uniform_(conv1.weight)

        layers = [conv1]

        if do_batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels, eps=1.001e-5))

        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)


class MultiscaleConvBlock(nn.Module):
    """Multiscale convolution block
    Defines four convolution layers.
    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernel_sizes (list): Size of the kernel in each convolution layer.
        strds (int): Size of stride in the convolution layer.
        use_bias (bool): Whether to use bias in the convolution layer.
        dilatation_rates (list): Dilation rate for each convolution layer.
        activation (str): Name of the activation function to use.
    Returns:
        model (torch.nn.Module): a pytorch model.

    """

    def __init__(
        self,
        num_input_channels,
        kernel_sizes,
        dilatation_rates,
        num_output_channels=32,
        strds=(1, 1),
        activation="relu",
        use_bias=False,
    ):

        super().__init__()

        self.conv_block_1 = ConvBnRelu(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            kernel_size=kernel_sizes[0],
            strds=strds,
            activation=activation,
            use_bias=use_bias,
            dilatation_rate=(dilatation_rates[0], dilatation_rates[0]),
        )

        self.conv_block_2 = ConvBnRelu(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            kernel_size=kernel_sizes[1],
            strds=strds,
            activation=activation,
            use_bias=use_bias,
            dilatation_rate=(dilatation_rates[1], dilatation_rates[1]),
        )

        self.conv_block_3 = ConvBnRelu(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            kernel_size=kernel_sizes[2],
            strds=strds,
            activation=activation,
            use_bias=use_bias,
            dilatation_rate=(dilatation_rates[2], dilatation_rates[2]),
        )

        self.conv_block_4 = ConvBnRelu(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            kernel_size=kernel_sizes[3],
            strds=strds,
            activation=activation,
            use_bias=use_bias,
            dilatation_rate=(dilatation_rates[3], dilatation_rates[3]),
        )

    def forward(self, input_map):
        """Logic for using layers defined in init.
        This method defines how layers are used in forward operation.
        Args:
            input (torch.Tensor): Input, the tensor is of the shape NCHW.
        Returns:
            output (torch.Tensor): The inference output.
        """

        conv0 = input_map

        conv1 = self.conv_block_1(conv0)
        conv2 = self.conv_block_2(conv0)
        conv3 = self.conv_block_3(conv0)
        conv4 = self.conv_block_4(conv0)

        return torch.cat([conv1, conv2, conv3, conv4], dim=bn_axis)


class ResidualConv(nn.Module):
    """Residual Convolution block
    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernel_size (int): Size of the kernel in all convolution layers.
        strds (int): Size of the stride in all convolution layers.
        use_bias (bool): Whether to use bias in the convolution layers.
        dilatation_rate (int): Dilation rate in all convolution layers.
        activation (str): Name of the activation function to use.
    Returns:
        model (torch.nn.Module): a pytorch model.

    """

    def __init__(
        self,
        num_input_channels,
        num_output_channels=32,
        kernel_size=(3, 3),
        strds=(1, 1),
        activation="relu",
        use_bias=False,
        dilatation_rate=(1, 1),
    ):
        super().__init__()

        self.conv_block_1 = ConvBnRelu(
            num_input_channels,
            num_output_channels,
            kernel_size=kernel_size,
            strds=strds,
            activation="None",
            use_bias=use_bias,
            dilatation_rate=dilatation_rate,
            do_batchnorm=True,
        )
        self.conv_block_2 = ConvBnRelu(
            num_output_channels,
            num_output_channels,
            kernel_size=kernel_size,
            strds=strds,
            activation="None",
            use_bias=use_bias,
            dilatation_rate=dilatation_rate,
            do_batchnorm=True,
        )

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, input_tensor):
        """Logic for using layers defined in init.
        This method defines how layers are used in forward operation.
        Args:
            input_tensor (torch.Tensor): Input, the tensor is of the shape NCHW.
        Returns:
            output (torch.Tensor): The inference output.
        """

        conv1 = self.conv_block_1(input_tensor)
        conv2 = self.conv_block_2(conv1)
        out = torch.add(conv1, conv2)

        return self.activation(out)


class NuClick(ModelABC):
    """NuClick Architecture.
    NuClick is used for interactive segmentation.
    NuClick takes an RGB image patch along with an inclusion and an exclusion map.
    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
    Returns:
        model (torch.nn.Module): a pytorch model.
    Examples:
        >>> # instantiate a NuClick model for interactive nucleus segmentation.
        >>> NuClick(num_input_channels = 5, num_output_channels = 1)
    """

    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        self.net_name = "NuClick"

        self.n_channels = num_input_channels
        self.n_classes = num_output_channels

        # -------------Convolution + Batch Normalization + ReLu blocks------------
        self.conv_block_1 = nn.Sequential(
            ConvBnRelu(
                num_input_channels=self.n_channels,
                num_output_channels=64,
                kernel_size=7,
            ),
            ConvBnRelu(num_input_channels=64, num_output_channels=32, kernel_size=5),
            ConvBnRelu(num_input_channels=32, num_output_channels=32, kernel_size=3),
        )

        self.conv_block_2 = nn.Sequential(
            ConvBnRelu(num_input_channels=64, num_output_channels=64),
            ConvBnRelu(num_input_channels=64, num_output_channels=32),
            ConvBnRelu(num_input_channels=32, num_output_channels=32),
        )

        self.conv_block_3 = ConvBnRelu(
            num_input_channels=32,
            num_output_channels=self.n_classes,
            kernel_size=(1, 1),
            activation=None,
            use_bias=True,
            do_batchnorm=False,
        )

        # -------------Residual Convolution blocks------------
        self.residual_block_1 = nn.Sequential(
            ResidualConv(num_input_channels=32, num_output_channels=64),
            ResidualConv(num_input_channels=64, num_output_channels=64),
        )

        self.residual_block_2 = ResidualConv(
            num_input_channels=64, num_output_channels=128
        )

        self.residual_block_3 = ResidualConv(
            num_input_channels=128, num_output_channels=128
        )

        self.residual_block_4 = nn.Sequential(
            ResidualConv(num_input_channels=128, num_output_channels=256),
            ResidualConv(num_input_channels=256, num_output_channels=256),
            ResidualConv(num_input_channels=256, num_output_channels=256),
        )

        self.residual_block_5 = nn.Sequential(
            ResidualConv(num_input_channels=256, num_output_channels=512),
            ResidualConv(num_input_channels=512, num_output_channels=512),
            ResidualConv(num_input_channels=512, num_output_channels=512),
        )

        self.residual_block_6 = nn.Sequential(
            ResidualConv(num_input_channels=512, num_output_channels=1024),
            ResidualConv(num_input_channels=1024, num_output_channels=1024),
        )

        self.residual_block_7 = nn.Sequential(
            ResidualConv(num_input_channels=1024, num_output_channels=512),
            ResidualConv(num_input_channels=512, num_output_channels=256),
        )

        self.residual_block_8 = ResidualConv(
            num_input_channels=512, num_output_channels=256
        )

        self.residual_block_9 = ResidualConv(
            num_input_channels=256, num_output_channels=256
        )

        self.residual_block_10 = nn.Sequential(
            ResidualConv(num_input_channels=256, num_output_channels=128),
            ResidualConv(num_input_channels=128, num_output_channels=128),
        )

        self.residual_block_11 = ResidualConv(
            num_input_channels=128, num_output_channels=64
        )

        self.residual_block_12 = ResidualConv(
            num_input_channels=64, num_output_channels=64
        )

        # -------------Multi-scale Convolution blocks------------
        self.multiscale_block_1 = MultiscaleConvBlock(
            num_input_channels=128,
            num_output_channels=32,
            kernel_sizes=[3, 3, 5, 5],
            dilatation_rates=[1, 3, 3, 6],
        )

        self.multiscale_block_2 = MultiscaleConvBlock(
            num_input_channels=256,
            num_output_channels=64,
            kernel_sizes=[3, 3, 5, 5],
            dilatation_rates=[1, 3, 2, 3],
        )

        self.multiscale_block_3 = MultiscaleConvBlock(
            num_input_channels=64,
            num_output_channels=16,
            kernel_sizes=[3, 3, 5, 7],
            dilatation_rates=[1, 3, 2, 6],
        )

        # -------------Max Pooling blocks------------
        self.pool_block_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_5 = nn.MaxPool2d(kernel_size=(2, 2))

        # -------------Transposed Convolution blocks------------
        self.conv_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_5 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=(2, 2),
        )

    def forward(self, imgs: torch.Tensor):
        """Logic for using layers defined in init.
        This method defines how layers are used in forward operation.
        Args:
            imgs (torch.Tensor): Input images, the tensor is of the shape NCHW.
        Returns:
            output (torch.Tensor): The inference output.
        """
        conv1 = self.conv_block_1(imgs)
        pool1 = self.pool_block_1(conv1)

        conv2 = self.residual_block_1(pool1)
        pool2 = self.pool_block_2(conv2)

        conv3 = self.residual_block_2(pool2)
        conv3 = self.multiscale_block_1(conv3)
        conv3 = self.residual_block_3(conv3)
        pool3 = self.pool_block_3(conv3)

        conv4 = self.residual_block_4(pool3)
        pool4 = self.pool_block_4(conv4)

        conv5 = self.residual_block_5(pool4)
        pool5 = self.pool_block_5(conv5)

        conv51 = self.residual_block_6(pool5)

        up61 = torch.cat([self.conv_transpose_1(conv51), conv5], dim=1)
        conv61 = self.residual_block_7(up61)

        up6 = torch.cat([self.conv_transpose_2(conv61), conv4], dim=1)
        conv6 = self.residual_block_8(up6)
        conv6 = self.multiscale_block_2(conv6)
        conv6 = self.residual_block_9(conv6)

        up7 = torch.cat([self.conv_transpose_3(conv6), conv3], dim=1)
        conv7 = self.residual_block_10(up7)

        up8 = torch.cat([self.conv_transpose_4(conv7), conv2], dim=1)
        conv8 = self.residual_block_11(up8)
        conv8 = self.multiscale_block_3(conv8)
        conv8 = self.residual_block_12(conv8)

        up9 = torch.cat([self.conv_transpose_5(conv8), conv1], dim=1)
        conv9 = self.conv_block_2(up9)

        return self.conv_block_3(conv9)

    @staticmethod
    def postproc(
        preds,
        thresh=0.33,
        min_size=10,
        min_hole_size=30,
        do_reconstruction=False,
        nuc_points=None,
    ):
        """Post processing.
        Args:
            preds (ndarray): list of prediction output of each patch and
                assumed to be in the order of (no.patch, h, w) (match with the output
                of `infer_batch`).
            thresh (float): Threshold value. If a pixel has a predicted value larger
                than the threshold, it will be classified as nuclei.
            min_size (int): The smallest allowable object size.
            min_hole_size (int):  The maximum area, in pixels, of a contiguous hole
                that will be filled.
            do_reconstruction (bool): Whether to perform a morphological reconstruction
                of an image.
            nuc_points (ndarray): In the order of (no.patch, h, w).
                In each patch, The pixel that has been 'clicked' is set to 1 and the
                rest pixels are set to 0.
        Returns:
            masks (ndarray): pixel-wise nuclei instance segmentation
                prediction, shape:(no.patch, h, w).
        """
        masks = preds > thresh
        masks = remove_small_objects(masks, min_size=min_size)
        masks = remove_small_holes(masks, area_threshold=min_hole_size)
        if do_reconstruction:
            for i in range(len(masks)):
                this_mask = masks[i, :, :]
                this_marker = nuc_points[i, :, :] > 0

                if np.any(this_mask[this_marker > 0]):
                    this_mask = reconstruction(this_marker, this_mask, selem=disk(1))
                    masks[i] = np.array([this_mask])
                else:
                    warnings.warn(
                        f"Nuclei reconstruction was not done for nucleus #{i}"
                    )
        return masks

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.
        This contains logic for forward operation as well as batch i/o
        aggregation.
        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.
        Returns:
            Pixel-wise nuclei prediction for each patch, shape: (no.patch, h, w).
        """
        model.eval()
        device = misc.select_device(on_gpu)

        # Assume batch_data is NCHW
        batch_data = batch_data.to(device).type(torch.float32)

        with torch.inference_mode():
            output = model(batch_data)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)

        return output.cpu().numpy()