# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module enables patch-level prediction."""

import math
import tqdm
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.data import Patch_Dataset


class CNN_Patch_Model(nn.Module):
    """Extends the backbone model so that is performs classification
    at the output of the network.

    Attributes:
        nr_class (int): number of classes output by the model.
        feat_extract (nn.Module): backbone CNN model.
        pool (nn.Module): type of pooling applied after feature extraction.
        classifier (nn.Module): linear classifier module used to map the features
                                to the output.

    """

    def __init__(self, backbone, nr_input_ch=3, nr_class=1):
        super().__init__()
        self.nr_class = nr_class

        self.feat_extract = get_model(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        """Run inference on an input batch. Contains logic for
        forward operation as well as i/o aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by torch.utils.data.DataLoader.

        """
        img_patches = batch_data
        img_patches_gpu = img_patches.to("cuda").type(torch.float32)  # to NCHW
        img_patches_gpu = img_patches_gpu.permute(0, 3, 1, 2).contiguous()

        # inference mode
        model.eval()
        # dont compute the gradient (not training)
        with torch.no_grad():
            output = model(img_patches_gpu)
        # output should be a single tensor or scalar
        return output.cpu().numpy()


class CNN_Patch_Predictor(object):
    """Patch-level predictor.

    Attributes:
        batch_size (int): number of images fed into the model each time.
        nr_input_ch (int): number of input channels of the image. If RGB, then this is 3.
        nr_loader_worker (int): number of workers used in torch.utils.data.DataLoader.
        verbose (bool): whether to output logging information.
        model (nn.Module): Defined PyTorch model.

    Usage:
        >>> image_list = np.random.randint(0, 255, [4, 512, 512, 3])
        >>> model = CNN_Patch_Predictor('resnet50', nr_class=8, batch_size=4, nr_loader_workers=4)
        >>> model.predict(image_list)

    """

    def __init__(
        self,
        batch_size,
        model=None,
        backbone="resnet50",
        nr_class=2,
        nr_input_ch=None,
        nr_loader_worker=0,
        verbose=True,
        *args,
        **kwargs,
    ):
        """Initialise the Patch Predictor. Note, if model is supplied in the arguments, the it
        will override the backbone.

        Args:
            batch_size (int): number of images fed into the model each time.
            model (nn.Module): defined PyTorch model with the define backbone as the feature extractor.
            backbone (str): name of the backbone model. This is obtained from tiatoolbox.models.backbone.
            nr_class (int): number of classes predicted by the model.
            nr_input_ch (int): number of input channels of the image. If RGB, then this is 3.
            nr_loader_worker (int): number of workers used in torch.utils.data.DataLoader.
            verbose (bool): whether to output logging information.

        """
        super().__init__()
        self.batch_size = batch_size
        self.nr_input_ch = nr_input_ch
        self.nr_loader_worker = nr_loader_worker
        self.verbose = verbose

        if model is not None:
            self.model = model
        else:
            self.model = CNN_Patch_Model(
                backbone, nr_input_ch=nr_input_ch, nr_class=nr_class
            )
        return

    def load_model(self, model_path, *args, **kwargs):
        """Load model checkpoint.

        Args:
            model_path: path to a PyTorch trained checkpoint. Supplied model
                        must match the initialised model in __init__.

        """
        # ! assume to be saved in single GPU mode
        saved_state_dict = torch.load(model_path)
        self.model = self.model.load_state_dict(saved_state_dict, strict=True)
        return

    def predict(self, X, *args, **kwargs):
        """Make a prediction on a dataset. Internally, this will create a
        dataset using tiatoolbox.models.data.classification.Patch_Dataset
        and call predict_dataset.

        Returns:
            output: predictions of the input dataset

        """
        ds = Patch_Dataset(X)
        output = self.predict_dataset(ds)
        raise output

    def predict_dataset(self, dataset, *args, **kwargs):
        """Make a prediction on a custom dataset object. Dataset object is Torch compliant."""
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.nr_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
        )

        # ! may need to take into account CPU/GPU mode
        model = torch.nn.DataParallel(self.model)
        model = model.to("cuda")

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
