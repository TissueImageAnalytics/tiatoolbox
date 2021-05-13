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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module enables patch-level prediction."""

from tiatoolbox.models.dataset.classification import PatchDataset, WsiPatchDataset
import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import copy

from tiatoolbox import rcParam
from tiatoolbox.models.abc import ModelBase
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.utils.misc import download_data
from tiatoolbox.models.classification.pretrained_info import _pretrained_model


class CNNPatchModel(ModelBase):
    """Retrieve the model backbone and attach an extra FCN to perform classification.

    Attributes:
        num_classes (int): Number of classes output by the model.
        feat_extract (nn.Module): Backbone CNN model.
        pool (nn.Module): Type of pooling applied after feature extraction.
        classifier (nn.Module): Linear classifier module used to map the features
                                to the output.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.feat_extract = get_model(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Best way to retrieve channel dynamically is passing a small forward pass
        prev_num_ch = self.feat_extract(torch.rand([2, 3, 96, 96])).shape[1]
        self.classifer = nn.Linear(prev_num_ch, num_classes)

        self.preproc_func = None

    def forward(self, imgs):
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor): Model input.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifer(gap_feat)
        prob = torch.softmax(logit, -1)
        return prob

    def set_preproc_func(self, func):
        """To set function for preprocessing.

        Set the `preproc_func` to this `func` if it is not None.
        Else the `preproc_func` is reset to return source image.

        `func` must behave in the following manner

        >>> transformed_img = func(img)
        """
        self.preproc_func = func if func is not None else lambda x: x

    def get_preproc_func(self):
        """Get preprocessing function."""
        return self.preproc_func

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): A batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        """
        if on_gpu:
            device = "cuda"
        else:
            device = "cpu"

        img_patches = batch_data
        img_patches_device = img_patches.to(device).type(torch.float32)  # to NCHW
        img_patches_device = img_patches_device.permute(0, 3, 1, 2).contiguous()

        # Inference mode
        model.eval()
        # Dont compute the gradient (not training)
        with torch.no_grad():
            output = model(img_patches_device)
        # Output should be a single tensor or scalar
        return output.cpu().numpy()


class CNNPatchPredictor:
    """Patch-level predictor.

    Attributes:
        batch_size (int): Number of images fed into the model each time.
        num_loader_worker (int): Number of workers used in torch.utils.data.DataLoader.
        model (nn.Module): Defined PyTorch model.
        verbose (bool): Whether to output logging information.

    Usage:
        >>> dataset = PatchDataset()
        >>> predictor = CNNPatchPredictor(predefined_model="resnet18-kather100K")
        >>> output = predictor.predict(dataset)

    """

    def __init__(
        self,
        batch_size=8,
        num_loader_worker=0,
        model=None,
        predefined_model=None,
        pretrained_weight=None,
        verbose=True,
    ):
        """Initialise the Patch Predictor.

        Note, if model is supplied in the arguments, it will override the backbone.

        Args:
            model (nn.Module): Use externally defined PyTorch model for prediction with.
                weights already loaded. Default is `None`. If provided,
                `pretrained_model` argument is ignored.

            predefined_model (str): Name of the existing models support by tiatoolbox
                for processing the data. Currently supports:

                - alexnet-kather100K: alexnet backbone trained on Kather 100K dataset.
                - resnet18-kather100K: resnet18 backbone trained on Kather 100K dataset.
                - resnet34-kather100K: resnet34 backbone trained on Kather 100K dataset.
                - resnet50-kather100K: resnet50 backbone trained on Kather 100K dataset.
                - resnet101-kather100K: resnet101 backbone trained on
                  Kather 100K dataset.
                - resnext5032x4d-kather100K: resnext50_32x4d backbone trained on Kather
                  100K dataset.
                - resnext101_32x8d-kather100K: resnext101_32x8d backbone trained on
                  Kather 100K dataset.
                - wide_resnet50_2-kather100K: wide_resnet50_2 backbone trained on
                  Kather 100K dataset.
                - wide_resnet101_2-kather100K: wide_resnet101_2 backbone trained on
                  Kather 100K dataset.
                - densenet121-kather100K: densenet121 backbone trained on
                  Kather 100K dataset.
                - densenet161-kather100K: densenet161 backbone trained on
                  Kather 100K dataset.
                - densenet169-kather100K: densenet169 backbone trained on
                  Kather 100K dataset.
                - densenet201-kather100K: densenet201 backbone trained on
                  Kather 100K dataset.
                - mobilenet_v2-kather100K: mobilenet_v2 backbone trained on
                  Kather 100K dataset.
                - mobilenet_v3_large-kather100K: mobilenet_v3_large backbone trained on
                  Kather 100K dataset.
                - mobilenet_v3_small-kather100K: mobilenet_v3_small backbone trained on
                  Kather 100K dataset.
                - googlenet-kather100K: googlenet backbone trained on
                  Kather 100K dataset.

                By default, the corresponding pretrained weights will also be
                downloaded. However, you can override with your own set of weights
                via the `pretrained_weight` argument. Argument is case insensitive.

            pretrained_weight (str): Path to the weight of the corresponding
                `predefined_model`.

            batch_size (int) : Number of images fed into the model each time.
            num_loader_worker (int) : Number of workers to load the data.
                Take note that they will also perform preprocessing.
            verbose (bool): Whether to output logging information.

        """
        super().__init__()

        if model is None and predefined_model is None:
            raise ValueError("Must provide either of `model` or `predefined_model`")

        if model is not None:
            self.model = model
        else:
            self.model, self.patch_size, self.objective_value = get_predefined_model(
                predefined_model, pretrained_weight
            )

        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.verbose = verbose

    @staticmethod
    def __postprocess(probabilities):
        """Apply post processing to output probablities. For classification, we apply
        a simple method and simply take the class with the highest probability.

        Args:
            probabilities (ndarray): Model output probabilities.

        """
        return np.argmax(probabilities, axis=-1)

    def _predict_engine(
        self,
        dataset,
        return_probabilities=False,
        return_labels=False,
        return_coordinates=False,
        on_gpu=True,
    ):
        """Make a prediction on a dataset. Internally will make a deep copy
        of the provided dataset to ensure user provided dataset is unchanged.

        Args:
            dataset (torch.utils.data.Dataset): PyTorch dataset object created using
                tiatoolbox.models.data.classification.Patch_Dataset.
            return_probabilities (bool): Whether to return per-class probabilities.
            return_labels (bool): Whether to return labels.
            return_coordinates (bool): Whether to return patch coordinates.
            on_gpu (bool): whether to run model on the GPU.

        Returns:
            output (ndarray): Model predictions of the input dataset

        """

        # may be expensive
        # dataset = copy.deepcopy(dataset)  # make a deep copy of this
        dataset.set_preproc_func(self.model.get_preproc_func())
        dataset.return_labels = return_labels  # HACK

        # preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        pbar = tqdm.tqdm(
            total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
        )

        if on_gpu:
            # DataParallel works only for cuda
            model = torch.nn.DataParallel(self.model)
            model = model.to("cuda")
        else:
            model = self.model.to("cpu")

        all_output = {}
        predictions_output = []
        probabilities_output = []
        coordinates_output = []
        labels_output = []
        for _, batch_data in enumerate(dataloader):

            if return_labels and return_coordinates:
                batch_input, batch_labels, batch_coordinates = batch_data
            elif return_labels and not return_coordinates:
                batch_input, batch_labels = batch_data
            elif not return_labels and return_coordinates:
                batch_input, batch_coordinates = batch_data
            else:
                batch_input = batch_data

            batch_output_probabilities = self.model.infer_batch(
                model, batch_input, on_gpu
            )
            # get the index of the class with the maximum probability
            batch_output = self.__postprocess(batch_output_probabilities)
            predictions_output.extend(batch_output.tolist())
            if return_probabilities:
                # return raw output
                probabilities_output.extend(batch_output_probabilities.tolist())
            if return_labels:
                # return label per patch
                labels_output.extend(batch_labels.tolist())
            if return_coordinates:
                # return coordinates of processed patches
                coordinates_output.extend(batch_coordinates)

            # may be a with block + flag would be nicer
            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        predictions_output = np.array(predictions_output)
        all_output = {"predictions": predictions_output}
        if return_probabilities:
            probabilities_output = np.array(probabilities_output)
            all_output["probabilities"] = probabilities_output
        if return_labels:
            labels_output = np.array(labels_output)
            all_output["labels"] = labels_output
        if return_coordinates:
            coordinates_output = np.array(coordinates_output)
            all_output["coordinates"] = coordinates_output

        return all_output

    def predict(
        self,
        data,
        mode,
        return_probabilities=False,
        return_labels=False,
        objective_value=None,
        patch_size=None,
        on_gpu=True,
    ):
        """Make a prediction on a dataset. Internally will make a deep copy
        of the provided dataset to ensure user provided dataset is unchanged.

        Args:
            dataset (torch.utils.data.Dataset): PyTorch dataset object created using
                tiatoolbox.models.data.classification.Patch_Dataset.
            return_probabilities (bool): Whether to return per-class probabilities.
            return_labels (bool): Whether to return labels.
            objective_value (float): Objective power used for reading patches. Note,
                this is only utilised in `tile` and `wsi` modes.
            patch_size (tuple): Size of image to read when using `tile` and `wsi`
                mode (width, height).
            on_gpu (bool): whether to run model on the GPU.

        Returns:
            output (ndarray): Model predictions of the input dataset

        """

        if mode == "patch":
            # don't return coordinates if patches are already extracted
            return_coordinates = False
            dataset = PatchDataset(data)
            output = self._predict_engine(
                dataset, return_probabilities, return_labels, return_coordinates, on_gpu
            )

        elif mode == "tile" or mode == "wsi":
            # return coordinates of patches processed within a tile / whole-slide image
            return_coordinates = True

            if objective_value is not None:
                self.objective_value = objective_value
            if patch_size is not None:
                self.patch_size = patch_size

            if not os.path.isfile(data):
                raise ValueError(
                    "A single whole-slide image should be input to predict."
                )

            dataset = WsiPatchDataset(
                data, objective_value=self.objective_value, read_size=self.patch_size
            )

        else:
            raise ValueError("%s is not a valid mode." % mode)

        output = self._predict_engine(
            dataset, return_probabilities, return_labels, return_coordinates, on_gpu
        )

        return output


def get_predefined_model(predefined_model=None, pretrained_weight=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        predefined_model (str): Name of the existing models support by tiatoolbox
            for processing the data. Currently supports:

            - alexnet-kather100K: alexnet backbone trained on Kather 100K dataset.
            - resnet18-kather100K: resnet18 backbone trained on Kather 100K dataset.
            - resnet34-kather100K: resnet34 backbone trained on Kather 100K dataset.
            - resnet50-kather100K: resnet50 backbone trained on Kather 100K dataset.
            - resnet101-kather100K: resnet101 backbone trained on Kather 100K dataset.
            - resnext5032x4d-kather100K: resnext50_32x4d backbone trained on Kather
              100K dataset.
            - resnext101_32x8d-kather100K: resnext101_32x8d backbone trained on
              Kather 100K dataset.
            - wide_resnet50_2-kather100K: wide_resnet50_2 backbone trained on
              Kather 100K dataset.
            - wide_resnet101_2-kather100K: wide_resnet101_2 backbone trained on
              Kather 100K dataset.
            - densenet121-kather100K: densenet121 backbone trained on
              Kather 100K dataset.
            - densenet161-kather100K: densenet161 backbone trained on
              Kather 100K dataset.
            - densenet169-kather100K: densenet169 backbone trained on
              Kather 100K dataset.
            - densenet201-kather100K: densenet201 backbone trained on
              Kather 100K dataset.
            - mobilenet_v2-kather100K: mobilenet_v2 backbone trained on
              Kather 100K dataset.
            - mobilenet_v3_large-kather100K: mobilenet_v3_large backbone trained on
              Kather 100K dataset.
            - mobilenet_v3_small-kather100K: mobilenet_v3_small backbone trained on
              Kather 100K dataset.
            - googlenet-kather100K: googlenet backbone trained on Kather 100K dataset.

            By default, the corresponding pretrained weights will also be downloaded.
            However, you can override with your own set of weights via the
            `pretrained_weight` argument. Argument is case insensitive.

        pretrained_weight (str): Path to the weight of the
            corresponding `predefined_model`.

    """
    if not isinstance(predefined_model, str):
        raise ValueError("predefined_model must be a string.")

    # parsing protocol
    predefined_model = predefined_model.lower()

    if predefined_model not in _pretrained_model:
        raise ValueError("Predefined model `%s` does not exist." % predefined_model)
    cfg = _pretrained_model[predefined_model]
    patch_size = cfg["patch_size"]
    objective_power = cfg["objective_power"]
    backbone, dataset = predefined_model.split("-")

    preproc_func = predefined_preproc_func(dataset)
    model = CNNPatchModel(backbone=backbone, num_classes=cfg["num_classes"])
    model.set_preproc_func(preproc_func)

    if pretrained_weight is None:
        pretrained_weight_url = cfg["pretrained"]
        pretrained_weight_url_split = pretrained_weight_url.split("/")
        pretrained_weight = os.path.join(
            rcParam["TIATOOLBOX_HOME"], "models/", pretrained_weight_url_split[-1]
        )
        if not os.path.exists(pretrained_weight):
            download_data(pretrained_weight_url, pretrained_weight)

    # ! assume to be saved in single GPU mode
    # always load to CPU
    saved_state_dict = torch.load(pretrained_weight, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    return model, patch_size, objective_power
