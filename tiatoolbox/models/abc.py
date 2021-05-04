import torch
import torch.nn as nn


class Model_Base(nn.Module):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def infer_batch(model, img_list, info_list=None):
        """Contains logic for forward operation as well as i/o aggregation

        image_list: Torch.Tensor (N,...).
        info_list : A list of (N,...), each item is metadata corresponding to
                    image at same index in `image_list`.

        """
        raise NotImplementedError