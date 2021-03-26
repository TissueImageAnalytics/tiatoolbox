import tiatoolbox.models.models_abc as tia_model_abc
from tiatoolbox.models.cnn.backbone import get_model_creator

import torch
import torch.nn as nn

class Model_Base(tia_model_abc.Model_Base):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self, 
            batch_size,
            infer_input_shape=None, 
            infer_output_shape=None,
            nr_loader_worker=0,
            nr_posproc_worker=0,
            preproc_args=None,
            postproc_args=None,  
            *args, **kwargs):
        super().__init__()
        """
        """
        raise NotImplementedError

    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        raise NotImplementedError

    @staticmethod
    def __infer_batch(model, img_list, info_list=None):
        """
        Contain logic for forward operation as well as i/o aggregation

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
        """
        The most basics and is in line with sklearn model.predict(X)
        where X is an image list (np.array). Internally, this will
        create an internall dataset and call predict_dataset
        
        Return the prediction after being post process
        """
        raise NotImplementedError

    def predict_dataset(self, dataset, *args, **kwargs):
        """
        Apply the prediction on a dataset object. Dataset object is Torch compliance
        and return output should be compatible with input of __infer_batch
        """
        raise NotImplementedError

    def predict_wsi(self, wsi_path, *args, **kwargs):
        """
        Contain dedicated functionality to run inference on an entire WSI
        """
        raise NotImplementedError


import torchvision.transforms as transforms
class VanillaCNN(Model_Base):
    def __init__(self, 
        backbone,
        batch_size,
        infer_input_shape=None, 
        infer_output_shape=None,
        nr_loader_worker=0,
        nr_posproc_worker=0,
        *args, **kwargs):
        """
        `backbone`: name of a model creator within tia.models.cnn.backbone
        `args` and `kwargs` will correspond to the backbone creator input 
        args and kwargs
        """
        super().__init__()
        model_creator = get_model_creator(backbone)
        self.model = model_creator(*args, **kwargs)
        self.batch_size = batch_size
        self.nr_loader_worker = nr_loader_worker
        self.infer_input_shape = infer_input_shape
        self.infer_output_shape = infer_output_shape
        self.nr_loader_worker = nr_loader_worker
        self.nr_posproc_worker = nr_posproc_worker

        # ! A hack to hijack the self.preprocess func def
        # ! Any ways to set this normally?
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])
        return

    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        saved_state_dict = torch.load(checkpoint_path) # ! assume to be saved in single GPU mode
        self.model = self.model.load_state_dict(saved_state_dict, strict=True)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to('cuda')        
        return 

    @staticmethod
    def __infer_batch(model, batch_data):
        """
        Contain logic for forward operation as well as i/o aggregation

        image_list: Torch.Tensor (N,...)
        info_list : A list of (N,...), each item is metadata correspond to 
                    image at same index in `image_list`  
        """
        ####
        img_patches = batch_data
        img_patches_gpu = img_patches.to('cuda').type(torch.float32) # to NCHW
        img_patches_gpu = img_patches_gpu.permute(0, 3, 1, 2).contiguous()

        ####
        model.eval() # infer mode
        # --------------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            output = model(img_patches_gpu) 
        # should be a single tensor or scalar
        return output.cpu().numpy()

    def predict(self, X, *args, **kwargs):
        """
        The most basics and is in line with sklearn model.predict(X)
        where X is an image list (np.array). Internally, this will
        create an internall dataset and call predict_dataset
        
        Return the prediction after being post process
        """

        raise NotImplementedError

    def predict_dataset(self, dataset, *args, **kwargs):
        """
        Apply the prediction on a dataset object. Dataset object is Torch compliance
        and return output should be compatible with input of __infer_image_list
        """
        raise NotImplementedError

    def predict_wsi(self, wsi_path, *args, **kwargs):
        """
        Contain dedicated functionality to run inference on an entire WSI
        """
        raise NotImplementedError    

if __name__ == '__main__':
    print('Local sample test')