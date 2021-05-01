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
import requests
import pathlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import copy

from tiatoolbox import TIATOOLBOX_HOME
from tiatoolbox.models.abc import Model_Base
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.dataset import Patch_Dataset, predefined_preproc_func
from tiatoolbox.utils.misc import download_data
from tiatoolbox.models.classification.pretrained_info import __pretrained_model


class CNN_Patch_Model(Model_Base):
    """Retrieve the model backbone and attach an extra FCN to perform classification.

    Attributes:
        nr_classes (int): number of classes output by the model.
        feat_extract (nn.Module): backbone CNN model.
        pool (nn.Module): type of pooling applied after feature extraction.
        classifier (nn.Module): linear classifier module used to map the features
                                to the output.

    """

    def __init__(self, backbone, nr_input_ch=3, nr_classes=1):
        super().__init__()
        self.nr_classes = nr_classes

        self.feat_extract = get_model(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # best way to retrieve channel dynamically is passing a small forward pass
        prev_nr_ch = self.feat_extract(torch.rand([2, 3, 4, 4])).shape[1]
        self.classifer = nn.Linear(prev_nr_ch, nr_classes)

        self.preproc_func = None

    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifer(gap_feat)
        prob = torch.softmax(logit, -1)
        return prob

    def set_preproc_func(self, func):
        """
        Set the `preproc_func` to this `func` if it is not None.
        Else the `preproc_func` is reset to return source image.

        `func` must behave in the following manner

        >>> transformed_img = func(img)
        """
        self.preproc_func = func if func is not None else lambda x : x
        return 
    
    def get_preproc_func(self):
        return self.preproc_func
    
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
        nr_loader_worker (int): number of workers used in torch.utils.data.DataLoader.
        model (nn.Module): Defined PyTorch model.
        verbose (bool): whether to output logging information.

    Usage:
        >>> dataset = Kather_Patch_Dataset()
        >>> predictor = CNN_Patch_Predictor(predefined_model="resnet18_kather", batch_size=16)
        >>> output = predictor.predict(dataset)

    """

    def __init__(
        self,
        batch_size=8,
        nr_loader_worker=0,
        model=None,
        predefined_model=None,
        pretrained_weight=None,
        verbose=True,
        *args,
        **kwargs,
    ):
        """Initialise the Patch Predictor. Note, if model is supplied in the arguments, it
        will override the backbone.

        Args:
            model (nn.Module): use externally defined PyTorch model for prediction. Default is `None`.
                            If provided, `pretrained_model` argument is ignored,

            predefined_model (str): name of the existing models support by tiatoolbox for processing the data.
                Currently support:
                - resnet18_kather : resnet18 backbone trained on Kather dataset.

                By default, the corresponding pretrained weights will also be downloaded.
                However, you can override with your own set of weights via the
                `pretrained_weight` argument. Argument is case insensitive.

            pretrained_weight (str): path to the weight of the corresponding `predefined_model`.

            batch_size (int) : number of images fed into the model each time.
            nr_loader_worker (int) : number of workers to load the data.
                                Take note that they will also perform preprocessing.
            verbose (bool): whether to output logging information.

        """
        super().__init__()

        if model is None and predefined_model is None:
            raise ValueError("Must provide either of `model` or `predefined_model`")

        if model is not None:
            self.model = model
        else:
            self.model = get_predefined_model(predefined_model, pretrained_weight)

        self.batch_size = batch_size
        self.nr_loader_worker = nr_loader_worker
        self.verbose = verbose
        return

    def predict(self, dataset, return_probs=False, on_gpu=True, *args, **kwargs):
        """Make a prediction on a dataset. Internally will make a deep copy of the provided
        dataset to ensure user provided dataset is unchanged.

        Args:
            dataset (torch.utils.data.Dataset): PyTorch dataset object created using
                tiatoolbox.models.data.classification.Patch_Dataset. 
            return_probs (bool): whether to return per-class model probabilities.
            on_gpu (bool): whether to run model on the GPU.

        Returns:
            output (ndarray): model predictions of the input dataset

        """

        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError(
                "Dataset supplied to predict() must be a PyTorch map style dataset (torch.utils.data.Dataset)."
            )

        # may be expensive
        dataset = copy.deepcopy(dataset) # make a deep copy of this
        dataset.set_preproc_func(self.model.get_preproc_func())

        # TODO preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.nr_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
        )

        # TODO: have a single protocol later for this
        pbar = tqdm.tqdm(
            total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
        )

        # ! may need to take into account CPU/GPU mode
        model = torch.nn.DataParallel(self.model)
        if on_gpu:
            model = model.to("cuda")

        all_output = {}
        preds_output = []
        probs_output = []
        labels_output = []

        for batch_idx, batch_data in enumerate(dataloader):
            # calling the static method of that specific ModelDesc
            # on the an instance of ModelDesc, maybe there is a better way
            # to go about this
            if dataset.return_label:
                batch_input, batch_label = batch_data
            else:
                batch_input = batch_data

            batch_output_probs = self.model.infer_batch(model, batch_input)
            # get the index of the class with the maximum probability
            batch_output = np.argmax(batch_output_probs, axis=-1)
            preds_output.extend(batch_output.tolist())
            if return_probs:
                # return raw output
                probs_output.extend(batch_output_probs.tolist())

            # may be a with block + flag would be nicer
            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        pred_output = np.array(preds_output)

        all_output = {"preds": preds_output}
        if return_probs:
            all_output["probs"] = probs_output
        return all_output


def get_predefined_model(predefined_model=None, pretrained_weight=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        predefined_model (str): name of the existing models support by tiatoolbox for processing the data.
            Currently support:
            - resnet18_kather: resnet18 backbone trained on Kather dataset.

            By default, the corresponding pretrained weights will also be downloaded.
            However, you can override with your own set of weights via the
            `pretrained_weight` argument. Argument is case insensitive.

        pretrained_weight (str): path to the weight of the corresponding `predefined_model`.

    """
    assert isinstance(predefined_model, str)
    # parsing protocol
    predefined_model = predefined_model.lower()

    if predefined_model not in __pretrained_model:
        raise ValueError('Predefined model `%s` does not exist.' % predefined_model )
    cfg = __pretrained_model[predefined_model]
    backbone, dataset = predefined_model.split("_")

    preproc_func = predefined_preproc_func(dataset)
    model = CNN_Patch_Model(
        backbone=backbone, nr_input_ch=cfg["nr_input_ch"], nr_classes=cfg["nr_classes"]
    )
    model.set_preproc_func(preproc_func)

    if pretrained_weight is None:
        pretrained_weight_url = cfg["pretrained"]
        pretrained_weight_url_split = pretrained_weight_url.split("/")
        pretrained_weight = os.path.join(
            TIATOOLBOX_HOME, "models/", pretrained_weight_url_split[-1]
        )
        if not os.path.exists(pretrained_weight):
            download_data(pretrained_weight_url, pretrained_weight)

    # ! assume to be saved in single GPU mode
    saved_state_dict = torch.load(pretrained_weight)
    model.load_state_dict(saved_state_dict, strict=True)
    return model
