
import math
import tqdm
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchvision.transforms as transforms

from ..backbone import get_model

####
class PatchArray_Dataset(torch_data.Dataset):
    def __init__(self, patch_list, preproc=None):
        super().__init__()
        if preproc is None:
            self.preproc = lambda x : x
        else:
            self.preproc = preproc
        self.patch_list = patch_list
        return

    def __len__(self):
        return self.patch_list.shape[0]

    def __getitem__(self, idx):
        patch = self.patch_list[idx]
        patch = self.preproc(patch)
        return patch

# attach this personally to the @staticmethod will create multiple duplication
# so do this hacky way to avoid that
__CNN_Patch_Predictor_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

class CNN_Patch_Predictor(nn.Module):
    def __init__(self,
                backbone, 
                nr_input_ch=3, 
                nr_class=1,
                model_code=None):
        super().__init__()
        self.nr_class = nr_class
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_extract = get_model(backbone)
        dummy = torch.rand((1, 3, 64, 64))
        dummy = self.feat_extract(dummy)
        #
        nr_feat = dummy.shape[1]
        self.classifer = nn.Linear(nr_feat, nr_class)
        
    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifer(gap_feat)
        prob = torch.softmax(logit, -1)
        return prob

    @staticmethod
    def infer_batch(model, batch_data):
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

    @staticmethod
    def preprocess_one_image(image):
        # using torchvision pipeline demands input is pillow format
        image = PIL.Image.fromarray(patch)
        image = __CNN_Patch_Predictor_preprocess(image)
        image = patch.permute(1, 2, 0)
        return image


class CNN_Patch_Predictor_Engine(object):
    """
    Usage:
    >>> image_list = np.random.randint(0, 255, [4, 512, 512, 3])
    >>> model = CNN_Predictor('resnet50', 8, 4, [512, 512, 3], [1])
    >>> model.predict(image_list)
    >>> 1, 2, 3, 0
    >>> model = CNN_Predictor('densenet121', 8, 4, [512, 512, 3], [1])
    >>> model.predict(image_list)
    >>> 1, 3, 3, 1
    """
    def __init__(self, 
        batch_size,
        model=None,
        backbone='resnet50',
        nr_class=2,
        nr_input_ch=None, 
        nr_loader_worker=0,
        verbose=True,
        *args, **kwargs):
        """
        `backbone`: name of a model creator within tia.models.cnn.backbone
        `args` and `kwargs` will correspond to the backbone creator input 
        args and kwargs
        """
        super().__init__()
        self.batch_size = batch_size
        self.nr_input_ch = nr_input_ch
        self.nr_loader_worker = nr_loader_worker
        self.verbose = verbose

        ###
        if model is not None:
            self.model = model
        else:
            self.model = CNN_Patch_Predictor(
                            backbone, 
                            nr_input_ch=nr_input_ch,
                            nr_class=nr_class)
        return
    
    def load_model(self, checkpoint_path, *args, **kwargs):
        """Load model checkpoint."""
        saved_state_dict = torch.load(checkpoint_path) # ! assume to be saved in single GPU mode
        self.model = self.model.load_state_dict(saved_state_dict, strict=True)
        return 

    def predict(self, X, *args, **kwargs):
        """
        The most basics and is in line with sklearn model.predict(X)
        where X is an image list (np.array). Internally, this will
        create an internall dataset and call predict_dataset
        
        Return the prediction after being post process
        """

        ds = PatchArray_Dataset(X, preproc=self.model.preprocess_one_image)
        output = self.predict_dataset(ds)
        raise output

    def predict_dataset(self, dataset, *args, **kwargs):
        """
        Apply the prediction on a dataset object. Dataset object is Torch compliance
        and return output should be compatible with input of __infer_image_list
        """
        dataloader = torch_data.DataLoader(dataset,
                            num_workers=self.nr_loader_worker,
                            batch_size=self.batch_size,
                            drop_last=False)

        pbar = tqdm.tqdm( total=int(len(dataloader)), 
                leave=True, ncols=80, ascii=True, position=0)

        # ! may need to take into account CPU/GPU mode
        model = torch.nn.DataParallel(self.model)
        model = model.to('cuda')        

        all_output = []
        for batch_idx, batch_input in enumerate(dataloader):
            # calling the static method of that specific ModelDesc 
            # on the an instance of ModelDesc, may be there is a nicer way 
            # to go about this
            batch_output = self.model.infer_batch(model, batch_input)
            all_output.extend(batch_output.tolist())
            # may be a with block + flag would be nicer
            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()
        all_output = np.array(all_output)
        return all_output

