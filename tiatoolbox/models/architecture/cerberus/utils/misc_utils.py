import torch
from torch import nn


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[2] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[3] - crop_shape[1]) * 0.5)
        x = x[:, :, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`

    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y.

    y dims must be smaller than x dims!

    """
    assert y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1], (
        "Ensure that y dimensions are smaller than x dimensions!"
    )

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


class Pytorch_Base(nn.Module):
    """Base class that enables parameter freezing."""

    def __init__(self, *args):
        super().__init__()
        self.x = nn.Sequential(*args)

    def forward(self, x, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                x = self.x(x)
        else:
            x = self.x(x)

        return x
