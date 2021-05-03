import torch
import torch.nn as nn


class Model_Base(nn.Module):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        raise NotImplementedError

    @staticmethod
    def __infer_batch(model, img_list, info_list=None):
        """Contains logic for forward operation as well as i/o aggregation

        image_list: Torch.Tensor (N,...).
        info_list : A list of (N,...), each item is metadata corresponding to
                    image at same index in `image_list`.

        """
        raise NotImplementedError

    @staticmethod
    def __postprocess(image, *args, **kwargs):
        """Process the raw model output to generate the final prediction."""
        raise NotImplementedError

    def predict(self, X, *args, **kwargs):
        """Make a prediction on the input. In line with
        sklearn model.predict(X) where X is an image list (np.array).

        Return the prediction after being post processed.

        """
        raise NotImplementedError

    def predict_wsi(self, wsi_path, *args, **kwargs):
        """Dedicated functionality to run inference on an entire WSI."""
        # currently just pass as not implemented yet
        pass
