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

import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib
import warnings

from tiatoolbox import rcParam
from tiatoolbox.models.abc import ModelBase
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.utils.misc import download_data, save_json
from tiatoolbox.models.classification.pretrained_info import _pretrained_model
from tiatoolbox.models.dataset.classification import PatchDataset, WSIPatchDataset


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

        `func` must behave in the following manner:

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
        >>> data = [img1, img2]
        >>> predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100K")
        >>> output = predictor.predict(data, mode='patch')

        >>> data = np.array([img1, img2])
        >>> predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100K")
        >>> output = predictor.predict(data, mode='patch')

        >>> wsi_file = 'path/wsi.svs'
        >>> predictor = CNNPatchPredictor(pretraind_model="resnet18-kather100K")
        >>> output = predictor.predict(wsi_file, mode='wsi')

    """

    def __init__(
        self,
        batch_size=8,
        num_loader_worker=0,
        model=None,
        pretrained_model=None,
        pretrained_weight=None,
        verbose=True,
    ):
        """Initialise the Patch Predictor.

        Note, if model is supplied in the arguments, it will override the backbone.

        Args:
            model (nn.Module): Use externally defined PyTorch model for prediction with.
                weights already loaded. Default is `None`. If provided,
                `pretrained_model` argument is ignored.

            pretrained_model (str): Name of the existing models support by tiatoolbox
                for processing the data. Refer to
                `tiatoolbox.models.classification.get_pretrained_model` for detail.

                By default, the corresponding pretrained weights will also be
                downloaded. However, you can override with your own set of weights
                via the `pretrained_weight` argument. Argument is case insensitive.

            pretrained_weight (str): Path to the weight of the corresponding
                `pretrained_model`.

            batch_size (int) : Number of images fed into the model each time.
            num_loader_worker (int) : Number of workers to load the data.
                Take note that they will also perform preprocessing.
            verbose (bool): Whether to output logging information.

        """
        super().__init__()

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        if model is not None:
            self.model = model
        else:
            self.model, self.patch_size, self.objective_value = get_pretrained_model(
                pretrained_model, pretrained_weight
            )

        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.verbose = verbose

    @staticmethod
    def _postprocess(probabilities):
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
        """Make a prediction on a dataset. The dataset can be mutated.

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
        dataset.set_preproc_func(self.model.get_preproc_func())

        # preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        if self.verbose:
            pbar = tqdm.tqdm(
                total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
            )

        if on_gpu:
            # DataParallel works only for cuda
            model = torch.nn.DataParallel(self.model)
            model = model.to("cuda")
        else:
            model = self.model.to("cpu")

        cum_output = {
            "probabilities": [],
            "predictions": [],
            "coordinates": [],
            "labels": [],
        }
        for _, batch_data in enumerate(dataloader):

            batch_output_probabilities = self.model.infer_batch(
                model, batch_data["image"], on_gpu
            )
            # We get the index of the class with the maximum probability
            batch_output_predictions = self._postprocess(batch_output_probabilities)
            # tolist may be very expensive
            cum_output["probabilities"].extend(batch_output_probabilities.tolist())
            cum_output["predictions"].extend(batch_output_predictions.tolist())
            if return_coordinates:
                cum_output["coordinates"].extend(batch_data["coords"].tolist())
            if return_labels:  # be careful of `s`
                # We dont use tolist here because label may be of mixed types
                # and hence collated as list by torch
                cum_output["labels"].extend(list(batch_data["label"]))

            # May be a with block + flag would be nicer
            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        if not return_probabilities:
            cum_output.pop("probabilities")
        if not return_labels:
            cum_output.pop("labels")
        if not return_coordinates:
            cum_output.pop("coordinates")
        return cum_output

    def predict(
        self,
        img_list,
        mask_list=None,
        label_list=None,
        mode="patch",
        return_probabilities=False,
        return_labels=False,
        on_gpu=True,
        patch_size=(224, 224),
        stride_size=None,
        resolution=1.0,
        units="mpp",
        save_dir=None,
    ):
        """Make a prediction for a list of input data.

        Args:
            img_list (list, ndarray): List of inputs to process. When using`patch` mode,
            the input must be either a list of images, a list of image file paths
            or a numpy array of an image list. When using `tile` or `wsi` mode, the
            input must be a list of file paths.

            mask_list (list): List of masks. Patches are only processed if they are
            witin a masked area. If not provided, then the entire image is processed.

            label_list: List of labels. If using `tile` or `wsi` mode, then only a
            single label per image tile or whole-slide image is supported.
            mode (str): Type of input to process. Choose from either `patch`, `tile` or
                `wsi`.

            return_probabilities (bool): Whether to return per-class probabilities.

            return_labels (bool): Whether to return the labels with the predictions.
            on_gpu (bool): Whether to run model on the GPU.

            patch_size (tuple): Size of patches input to the model. Patches are at
                requested read resolution, not with respect to level 0.

            stride_size (tuple): Stride using during tile and WSI processing. Stride
                is at requested read resolution, not with respect to to level 0.
            resolution (float): Resolution used for reading the image.

            units (str): Units of resolution used for reading the image. Choose from
                either `level`, `power` or `mpp`.

            save_dir (str): Output directory when processing multiple tiles and
                whole-slide images.

        Returns:
            output (ndarray, pathlib.Path): Model predictions of the input dataset.
                If multiple image tiles or whole-slide images are provided as input,
                then results are saved and the resulting file paths are returned.


        """
        if mode not in ["patch", "wsi", "tile"]:
            raise ValueError(
                "%s is not a valid mode. Use either `patch`, `tile` or `wsi`" % mode
            )
        if label_list is not None:
            # if a label_list is provided, then return with the prediction
            return_labels = bool(label_list)
            if len(label_list) != len(img_list):
                raise ValueError(
                    "len(label_list) != len(img_list) : %d != %d"
                    % (len(label_list), len(img_list))
                )

        if mode == "patch":
            # don't return coordinates if patches are already extracted
            return_coordinates = False
            dataset = PatchDataset(img_list, label_list)
            output = self._predict_engine(
                dataset, return_probabilities, return_labels, return_coordinates, on_gpu
            )

        else:
            output_files = []  # generate a list of output file paths
            if len(img_list) > 1:
                warnings.warn(
                    "When providing multiple whole-slide images / tiles, "
                    "we save the outputs and return the locations "
                    "to the corresponding files."
                )
            if len(img_list) > 1 and save_dir is None:
                warnings.warn(
                    "> 1 WSIs detected but there is no save directory set."
                    "All subsequent output will be save to current runtime"
                    "location under folder 'output'. Overwriting may happen!"
                )
                save_dir = os.path.join(os.getcwd(), "output")

            if save_dir is not None:
                save_dir = pathlib.Path(save_dir)
                if not save_dir.is_dir():
                    os.makedirs(save_dir)
                else:
                    raise ValueError("`save_dir` already exists!")

            # return coordinates of patches processed within a tile / whole-slide image
            return_coordinates = True
            if not isinstance(img_list, list):
                raise ValueError(
                    "Input to `tile` and `wsi` mode must be a list of file paths."
                )

            for idx, img_path in enumerate(img_list):
                img_path = pathlib.Path(img_path)
                img_label = None if label_list is None else label_list[idx]
                img_mask = None if mask_list is None else mask_list[idx]

                dataset = WSIPatchDataset(
                    img_path,
                    mode=mode,
                    mask_path=img_mask,
                    patch_size=patch_size,
                    stride_size=stride_size,
                    resolution=resolution,
                    units=units,
                )
                output_model = self._predict_engine(
                    dataset,
                    return_labels=False,
                    return_probabilities=return_probabilities,
                    return_coordinates=return_coordinates,
                    on_gpu=on_gpu,
                )
                output_model["label"] = img_label

                if len(img_list) > 1:
                    basename = img_path.stem
                    output_file_path = os.path.join(save_dir, basename)
                    output_files.append(output_file_path)
                    save_json(output_model, output_file_path)

                    # set output to return locations of saved files
                    output = output_files
                else:
                    output = [output_model]

        return output


def get_pretrained_model(pretrained_model=None, pretrained_weight=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        pretrained_model (str): Name of the existing models support by tiatoolbox
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
            corresponding `pretrained_model`.

    """
    if not isinstance(pretrained_model, str):
        raise ValueError("pretrained_model must be a string.")

    # parsing protocol
    pretrained_model = pretrained_model.lower()

    if pretrained_model not in _pretrained_model:
        raise ValueError("Pretrained model `%s` does not exist." % pretrained_model)
    cfg = _pretrained_model[pretrained_model]
    patch_size = cfg["patch_size"]
    objective_power = cfg["objective_power"]
    backbone, dataset = pretrained_model.split("-")

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
    # always load on to the CPU
    saved_state_dict = torch.load(pretrained_weight, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    return model, patch_size, objective_power
