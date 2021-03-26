from tiatoolbox.models.models_abc import Model_Base

import torch.nn as nn

class Model_Base(nn.Module):
    """Abstract base class for backbone models used in tiatoolbox."""

    def __init__(self, weight_init=False):
        """
        """
        super().__init__()
        return

    @staticmethod
    def weight_init():
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return


