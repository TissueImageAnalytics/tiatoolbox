"""Utility layers for the Cerberus architecture."""

import math

from torch import nn


def weights_init_cnn(module):
    """Initialize standard CNN layers."""
    classname = module.__class__.__name__
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if "linear" in classname.lower() and module.bias is not None:
        nn.init.constant_(module.bias, 0)
    if "norm" in classname.lower():
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def weights_init_dsf(module):
    """Initialize discrete steerable filter layers."""
    classname = module.__class__.__name__
    if classname == "GConv2d":
        w_shape = module.weight.size()
        q = w_shape[2]
        fan_out = w_shape[-1]
        std = math.sqrt(2 / fan_out * q)
        nn.init.normal_(module.weight, mean=0.0, std=std)

    if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
