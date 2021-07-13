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

import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm

from tiatoolbox import rcParam
from tiatoolbox.models.abc import IOStateBase, ModelBase
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.models.dataset.classification import PatchDataset, WSIPatchDataset
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import (
    download_data,
    get_pretrained_model_info,
    save_dict_to_json,
)
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader


class _IOStatePatchPredictor(IOStateBase):
    """Define a class to hold IO information for patch predictor."""

    # We predefine to follow enforcement, actual initialization in init
    patch_size = None
    input_resolutions = None
    output_resolutions = None

    def __init__(self, patch_size, input_resolutions, output_resolutions, **kwargs):
        self.patch_size = patch_size
        self.input_resolutions = input_resolutions
        self.output_resolutions = output_resolutions
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)


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

        self.preproc_func = lambda x: x
        self.postproc_func = lambda x: np.argmax(x, axis=-1)

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

    @property
    def preproc(self):
        """Get preprocessing function."""
        return self.preproc_func

    @preproc.setter
    def preproc(self, func):
        """Setter for preprocessing function.

        Set the `preproc_func` to this `func` if it is not None.
        Else the `preproc_func` is reset to return source image.

        `func` must behave in the following manner:

        >>> transformed_img = func(img)

        """
        self.preproc_func = func if func is not None else lambda x: x

    @property
    def postproc(self):
        """Get postprocessing function."""
        return self.postproc_func

    @postproc.setter
    def postproc(self, func):
        """Setter for postprocessing function.

        Set the `postproc_func` to this `func` if it is not None.
        Else the `preproc_func` is reset to return argmax along last
        axis of the input.

        `func` must behave in the following manner:

        >>> transformed_img = func(prediction_of_an_image)

        """
        self.postproc_func = (
            func if func is not None else lambda x: np.argmax(x, axis=-1)
        )

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
        device = misc.select_device(on_gpu)

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
        >>> predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> data = np.array([img1, img2])
        >>> predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> wsi_file = 'path/wsi.svs'
        >>> predictor = CNNPatchPredictor(pretraind_model="resnet18-kather100k")
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
                `tiatoolbox.models.classification.get_pretrained_model` for details.

                By default, the corresponding pretrained weights will also be
                downloaded. However, you can override with your own set of weights
                via the `pretrained_weight` argument. Argument is case insensitive.

            pretrained_weight (str): Path to the weight of the corresponding
                `pretrained_model`.
                >>> predictor = CNNPatchPredictor(
                ...    pretrained_model="resnet18-kather100k",
                ...    pretrained_weight="resnet18_local_weight")

            batch_size (int) : Number of images fed into the model each time.
            num_loader_worker (int) : Number of workers to load the data.
                Take note that they will also perform preprocessing.
            verbose (bool): Whether to output logging information.

        """
        super().__init__()

        self.img_list = None
        self.output_list = None
        self.mode = None

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        if model is not None:
            self.model = model
            iostate = None  # retrieve iostate from provided model ?
        else:
            model, iostate = get_pretrained_model(pretrained_model, pretrained_weight)

        self._iostate = iostate  # for storing original
        self.iostate = None  # for runtime
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_worker
        self.verbose = verbose

    @staticmethod
    def merge_predictions(
        img,
        output,
        resolution=None,
        units=None,
        postproc_func: Callable = None,
    ):
        """Merge patch-level predictions to form a 2-dimensional prediction map.

        Args:
            img (:obj:`str` or :obj:`pathlib.Path` or :class:`numpy.ndarray`):
                A HWC image or a path to WSI.
            output (list): ouput generated by the model.
            resolution (float): Resolution of merged predictions.
            units (str): Units of resolution used when merging predictions. This
                must be the same `units` used when processing the data.
            postproc_func (callable): a function to post-process raw prediction
                from model. Must be in the form
                >>> value = postproc_func(predictions)

        Returns:
            prediction_map (ndarray): Merged predictions as a 2D array.
            overlay (ndarray): Overlaid output if return_overlay is set to True.

        """
        reader = get_wsireader(img)
        if isinstance(reader, VirtualWSIReader):
            warnings.warn(
                " ".join(
                    [
                        "Image is not pyramidal hence read is forced to be",
                        "at `units='baseline'` and `resolution=1.0`.",
                    ]
                )
            )
            resolution = 1.0
            units = "baseline"

        canvas_shape = reader.slide_dimensions(resolution=resolution, units=units)
        canvas_shape = canvas_shape[::-1]  # XY to YX

        # may crash here, do we need to deal with this ?
        output_shape = reader.slide_dimensions(
            resolution=output["resolution"], units=output["units"]
        )
        output_shape = output_shape[::-1]  # XY to YX
        fx = np.array(canvas_shape) / np.array(output_shape)

        if "probabilities" not in output.keys():
            coordinates = output["coordinates"]
            predictions = output["predictions"]
            denominator = None
            output = np.zeros(list(canvas_shape), dtype=np.float32)
        else:
            coordinates = output["coordinates"]
            predictions = output["probabilities"]
            num_class = np.array(predictions[0]).shape[0]
            denominator = np.zeros(canvas_shape)
            output = np.zeros(list(canvas_shape) + [num_class], dtype=np.float32)

        for idx, bound in enumerate(coordinates):
            prediction = predictions[idx]
            # assumed to be in XY
            # top-left for output placement
            tl = np.ceil(np.array(bound[:2]) * fx).astype(np.int32)
            # bot-right for output placement
            br = np.ceil(np.array(bound[2:]) * fx).astype(np.int32)
            output[tl[1] : br[1], tl[0] : br[0]] = prediction
            if denominator is not None:
                denominator[tl[1] : br[1], tl[0] : br[0]] += 1

        # deal with overlapping regions
        if denominator is not None:
            output = output / (np.expand_dims(denominator, -1) + 1.0e-8)
            # convert raw probabilities to preditions
            if postproc_func is not None:
                output = postproc_func(output)
            else:
                output = np.argmax(output, axis=-1)
            # to make sure background is 0 while class wil be 1..N
            output[denominator > 0] += 1
        return output

    def _predict_engine(
        self,
        dataset,
        return_probabilities=False,
        return_labels=False,
        return_coordinates=False,
        on_gpu=True,
    ):
        """Make a prediction on a dataset. The dataset may be mutated.

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
        dataset.preproc = self.model.preproc

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

        # use external for testing
        model = misc.model_to(on_gpu, self.model)

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
            batch_output_predictions = self.model.postproc(batch_output_probabilities)
            # tolist may be very expensive
            cum_output["probabilities"].extend(batch_output_probabilities.tolist())
            cum_output["predictions"].extend(batch_output_predictions.tolist())
            if return_coordinates:
                cum_output["coordinates"].extend(batch_data["coords"].tolist())
            if return_labels:  # be careful of `s`
                # We dont use tolist here because label may be of mixed types
                # and hence collated as list by torch
                cum_output["labels"].extend(list(batch_data["label"]))

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
        patch_size: Tuple[int, int] = None,
        stride_size: Tuple[int, int] = None,
        resolution=None,
        units=None,
        merge_predictions=False,
        save_dir=None,
    ):
        """Make a prediction for a list of input data.

        Args:
            img_list (list, ndarray): List of inputs to process. When using `patch`
            mode, the input must be either a list of images, a list of image file paths
            or a numpy array of an image list. When using `tile` or `wsi` mode, the
            input must be a list of file paths.

            mask_list (list): List of masks. Only utilised when processing image tiles
            and whole-slide images. Patches are only processed if they are witin a
            masked area. If not provided, then a tissue mask will be automatically
            generated for whole-slide images or the entire image is processed for
            image tiles.

            label_list: List of labels. If using `tile` or `wsi` mode, then only a
            single label per image tile or whole-slide image is supported.
            mode (str): Type of input to process. Choose from either `patch`, `tile` or
                `wsi`.

            return_probabilities (bool): Whether to return per-class probabilities.

            return_labels (bool): Whether to return the labels with the predictions.

            on_gpu (bool): whether to run model on the GPU.

            patch_size (tuple): Size of patches input to the model. Patches are at
                requested read resolution, not with respect to level 0, and must be
                positive.

            stride_size (tuple): Stride using during tile and WSI processing. Stride
                is at requested read resolution, not with respect to to level 0, and
                must be positive. If not provided, `stride_size=patch_size`.

            resolution (float): Resolution used for reading the image.

            units (str): Units of resolution used for reading the image. Choose from
                either `level`, `power` or `mpp`.

            merge_predictions (bool): Whether to merge the predictions to form
            a 2-dimensional map. This is only applicable for `mode='wsi'` or
            `mode='tile'`.

            save_dir (str): Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output` where the
                running script is invoked.

        Returns:
            output (ndarray, dict): Model predictions of the input dataset.
                If multiple image tiles or whole-slide images are provided as input,
                then results are saved to `save_dir` and a dictionary indicating save
                location for each input is return.

                The dict has following format:
                - img_path: path of the input image.
                    - raw: path to save location for raw prediction, saved in .json.
                    - merged: path to .npy contain merged predictions if
                    `merge_predictions` is `True`.

                For example
                >>> wsi_list = ['wsi1.svs', 'wsi2.svs']
                >>> predictor = CNNPatchPredictor(
                ...                 pretrained_model="resnet18-kather100k")
                >>> output = predictor.predict(wsi_list, mode='wsi)
                >>> output.keys()
                ['wsi1.svs', 'wsi2.svs']
                >>> output['wsi1.svs']
                {'raw': '0.raw.json', 'merged': '0.merged.npy}
                >>> output['wsi2.svs']
                {'raw': '1.raw.json', 'merged': '1.merged.npy}

        """
        if mode not in ["patch", "wsi", "tile"]:
            raise ValueError(
                f"{mode} is not a valid mode. Use either `patch`, `tile` or `wsi`"
            )
        if mode == "patch" and label_list is not None:
            # if a label_list is provided, then return with the prediction
            return_labels = bool(label_list)
            if len(label_list) != len(img_list):
                raise ValueError(
                    f"len(label_list) != len(img_list) : "
                    f"{len(label_list)} != {len(img_list)}"
                )
        if mode == "wsi" and mask_list is not None and len(mask_list) != len(img_list):
            raise ValueError(
                f"len(mask_list) != len(img_list) : "
                f"{len(mask_list)} != {len(img_list)}"
            )

        if mode == "patch":
            # don't return coordinates if patches are already extracted
            return_coordinates = False
            dataset = PatchDataset(img_list, label_list)
            output = self._predict_engine(
                dataset, return_probabilities, return_labels, return_coordinates, on_gpu
            )

        else:
            stride_size = stride_size if stride_size is not None else patch_size

            self.iostate = self._iostate
            if patch_size is not None:
                iostate = _IOStatePatchPredictor(
                    input_resolutions=[{"resolution": resolution, "units": units}],
                    output_resolutions=[{"resolution": resolution, "units": units}],
                    patch_size=patch_size,
                    stride_size=stride_size,
                )
                self.iostate = iostate

            if len(img_list) > 1:
                warnings.warn(
                    "When providing multiple whole-slide images / tiles, "
                    "we save the outputs and return the locations "
                    "to the corresponding files."
                )
                if save_dir is None:
                    warnings.warn(
                        "> 1 WSIs detected but there is no save directory set."
                        "All subsequent output will be saved to current runtime"
                        "location under folder 'output'. Overwriting may happen!"
                    )
                    save_dir = os.path.join(os.getcwd(), "output")

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

            # generate a list of output file paths if number of input images > 1
            file_dict = OrderedDict()
            for idx, img_path in enumerate(img_list):
                img_path = pathlib.Path(img_path)
                img_label = None if label_list is None else label_list[idx]
                img_mask = None if mask_list is None else mask_list[idx]

                dataset = WSIPatchDataset(
                    img_path,
                    mode=mode,
                    mask_path=img_mask,
                    patch_size=self.iostate.patch_size,
                    stride_size=self.iostate.stride_size,
                    resolution=self.iostate.input_resolutions[0]["resolution"],
                    units=self.iostate.input_resolutions[0]["units"],
                )
                output_model = self._predict_engine(
                    dataset,
                    return_labels=False,
                    return_probabilities=return_probabilities,
                    return_coordinates=return_coordinates,
                    on_gpu=on_gpu,
                )
                output_model["label"] = img_label
                # add extra information useful for downstream analysis
                output_model["pretrained_model"] = self.pretrained_model
                output_model["resolution"] = resolution
                output_model["units"] = units

                output_list = [output_model]  # assign to a list
                if merge_predictions:
                    merged_prediction = self.merge_predictions(
                        img_path, output_model, resolution=resolution, units=units
                    )
                    output_list.append(merged_prediction)

                if len(img_list) > 1:
                    img_code = "{number:0{width}d}".format(
                        width=len(str(len(img_list))), number=idx
                    )
                    save_info = {}
                    save_path = os.path.join(save_dir, img_code)
                    raw_save_path = f"{save_path}.raw.json"
                    save_info["raw"] = raw_save_path
                    save_dict_to_json(output_model, raw_save_path)
                    if merge_predictions:
                        merged_file_path = f"{save_path}.merged.npy"
                        np.save(merged_file_path, merged_prediction)
                        save_info["merged"] = merged_file_path
                    file_dict[img_path] = save_info
                else:
                    output = output_list
            output = file_dict if len(img_list) > 1 else output

        return output


def get_pretrained_model(pretrained_model=None, pretrained_weight=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        pretrained_model (str): Name of the existing models support by tiatoolbox
            for processing the data. Currently supports:

            - alexnet-kather100k: alexnet backbone trained on Kather 100k dataset.
            - resnet18-kather100k: resnet18 backbone trained on Kather 100k dataset.
            - resnet34-kather100k: resnet34 backbone trained on Kather 100k dataset.
            - resnet50-kather100k: resnet50 backbone trained on Kather 100k dataset.
            - resnet101-kather100k: resnet101 backbone trained on Kather 100k dataset.
            - resnext5032x4d-kather100k: resnext50_32x4d backbone trained on Kather
              100k dataset.
            - resnext101_32x8d-kather100k: resnext101_32x8d backbone trained on
              Kather 100k dataset.
            - wide_resnet50_2-kather100k: wide_resnet50_2 backbone trained on
              Kather 100k dataset.
            - wide_resnet101_2-kather100k: wide_resnet101_2 backbone trained on
              Kather 100k dataset.
            - densenet121-kather100k: densenet121 backbone trained on
              Kather 100k dataset.
            - densenet161-kather100k: densenet161 backbone trained on
              Kather 100k dataset.
            - densenet169-kather100k: densenet169 backbone trained on
              Kather 100k dataset.
            - densenet201-kather100k: densenet201 backbone trained on
              Kather 100k dataset.
            - mobilenet_v2-kather100k: mobilenet_v2 backbone trained on
              Kather 100k dataset.
            - mobilenet_v3_large-kather100k: mobilenet_v3_large backbone trained on
              Kather 100k dataset.
            - mobilenet_v3_small-kather100k: mobilenet_v3_small backbone trained on
              Kather 100k dataset.
            - googlenet-kather100k: googlenet backbone trained on Kather 100k dataset.

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
    backbone, dataset = pretrained_model.split("-")

    pretrained_yml = get_pretrained_model_info()

    pretrained_info = pretrained_yml[dataset]
    pretrained_models_dict = pretrained_info["models"]

    if backbone not in pretrained_models_dict.keys():
        raise ValueError(f"Pretrained model `{pretrained_model}` does not exist.")

    patch_size = pretrained_info["patch_size"]
    resolution = pretrained_info["resolution"]
    units = pretrained_info["units"]
    num_classes = pretrained_info["num_classes"]

    model = CNNPatchModel(backbone=backbone, num_classes=num_classes)
    model.preproc = predefined_preproc_func(dataset)

    if pretrained_weight is None:
        pretrained_weight_url = pretrained_models_dict[backbone]
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

    iostate = _IOStatePatchPredictor(
        patch_size=patch_size,
        input_resolutions=[{"resolution": resolution, "units": units}],
        output_resolutions=[{"resolution": resolution, "units": units}],
    )
    return model, iostate
