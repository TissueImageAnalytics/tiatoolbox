import tiatoolbox.models.abc as tia_model_abc


import torch
import torch.nn as nn


class Model_Base(tia_model_abc.Model_Base):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(
        self,
        batch_size,
        infer_input_shape=None,
        infer_output_shape=None,
        nr_loader_worker=0,
        nr_posproc_worker=0,
        preproc_args=None,
        postproc_args=None,
        *args,
        **kwargs
    ):
        super().__init__()
        raise NotImplementedError

    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        raise NotImplementedError

    @staticmethod
    def __infer_batch(model, batch_data):
        """Contain logic for forward operation as well as i/o aggregation.

        image_list: Torch.Tensor (N,...)
        info_list : A list of (N,...), each item is metadata correspond to 
                    image at same index in `image_list`

        """
        raise NotImplementedError

    @staticmethod
    def postprocess(image, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def preprocess(image, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X, *args, **kwargs):
        """The most basic and is in line with sklearn model.predict(X)
        where X is an image list (np.array). Internally, this will
        create an internal dataset and call predict_dataset.
        
        ! Should X alrd be in-compliance shape wrt to infer, 
        ! or we pad it on the fly? 

        Return the prediction after being post process.

        """
        raise NotImplementedError

    def predict_dataset(self, dataset, *args, **kwargs):
        """Apply the prediction on a dataset object. Dataset object is Torch compliance
        and return output should be compatible with input of __infer_batch.

        """
        raise NotImplementedError

    def predict_wsi(self, wsi_path, *args, **kwargs):
        """Contain dedicated functionality to run inference on an entire WSI."""
        raise NotImplementedError
