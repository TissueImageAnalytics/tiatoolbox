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


from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.models.architecture.hovernet import (
    HoVerNet,
    ResidualBlock,
    TFSamepaddingLayer,
    create_decoder_branch,
)
from tiatoolbox.models.architecture.utils import UpSample2x
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import get_bounding_box


class HoVerNetPlus(HoVerNet):
    """Initialise HoVer-Net+.

    HoVer-Net+ takes an RGB input image, and provides the option to simultaneously
    segment and classify the nuclei present, aswell as semantically segment different
    regions or layers in the images.

    """

    def __init__(
        self,
        num_input_channels: int = 3,
        num_types: int = None,
        num_layers: int = None,
        mode: str = "original",
    ):
        """Initialise HoVer-Net+.

        Args:
            num_input_channels (int): The number of input channels, default = 3 for RGB.
            num_types (int): The number of types of nuclei present in the images.
            num_layers (int): The number of layers/different regions types present.

        """
        super().__init__()
        self.mode = mode
        self.num_types = num_types
        self.num_layers = num_layers
        ksize = 5 if mode == "original" else 3

        if mode not in ["original", "fast"]:
            raise ValueError(
                f"Invalid mode {mode} for HoVerNet. "
                "Only support `original` or `fast`."
            )

        modules = [
            (
                "/",
                nn.Conv2d(num_input_channels, 64, 7, stride=1, padding=0, bias=False),
            ),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        # pre-pend the padding for `fast` mode
        if mode == "fast":
            modules = [("pad", TFSamepaddingLayer(ksize=7, stride=1)), *modules]

        self.conv0 = nn.Sequential(OrderedDict(modules))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        if num_types is None and num_layers is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        elif num_types is not None and num_layers is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        elif num_types is not None and num_layers is not None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("ls", create_decoder_branch(ksize=ksize, out_ch=num_layers)),
                    ]
                )
            )
        elif num_types is None and num_layers is not None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("ls", create_decoder_branch(ksize=ksize, out_ch=num_layers)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    @staticmethod
    def __proc_ls(ls_map: np.ndarray):
        """Extract Layer Segmentation map with LS Map.

        This function takes the layer segmentation map and applies a gaussian blur to
        remove spurious segmentations.

        Args:
            ls_map: The input predicted segmentation map.

        Returns:
            ls_map: The processed segmentation map.

        """
        ls_map = np.squeeze(ls_map.astype("float32"))
        ls_map = cv2.GaussianBlur(ls_map, (7, 7), 0)
        ls_map = np.around(ls_map)
        ls_map = ls_map.astype("int")

        return ls_map

    @staticmethod
    # skipcq: PYL-W0221
    def postproc(raw_maps: List[np.ndarray]):
        """Post processing script for image tiles.

        Args:
            raw_maps (list(ndarray)): list of prediction output of each head and
                assumed to be in the order of [np, hv, tp, ls] (match with the output
                of `infer_batch`).

        Returns:
            inst_map (ndarray): pixel-wise nuclear instance segmentation
                prediction.
            inst_dict (dict): a dictionary containing a mapping of each instance
                within `inst_map` instance information. It has the following form
            >>> inst_info = {
            >>>         box: number[],
            >>>         centroids: number[],
            >>>         contour: number[][],
            >>>         type: number,
            >>>         prob: number,
            >>> }
            >>> inst_dict = {[inst_uid: number] : inst_info}
                and `inst_uid` is an integer corresponds to the instance
                having the same pixel value within `inst_map`.
            layer_map (ndarray): pixel-wise layer segmentation prediction.
            layer_dict (dict): a dictionary containing a mapping of each segmented
                layer within `layer_map`. It has the following form
            >>> layer_info = {
            >>>         contour: number[][],
            >>>         type: number,
            >>> }
            >>> layer_dict = {[layer_uid: number] : layer_info}

        """
        if len(raw_maps) == 4:
            # Nuclei and layer segmentation
            np_map, hv_map, tp_map, ls_map = raw_maps
        elif len(raw_maps) == 3:
            # Nuclei segmentation (with classes) only
            np_map, hv_map, tp_map = raw_maps
            ls_map = None
        elif len(raw_maps) == 2:
            # Nuclei segmentation (no classes) only
            tp_map = None
            np_map, hv_map = raw_maps
            ls_map = None
        elif len(raw_maps) == 1:
            # Layer segmentation only
            ls_map = raw_maps[0]
            np_map, hv_map, tp_map = None, None, None

        pred_type = tp_map
        if np_map is not None:
            pred_inst = super(HoVerNetPlus, HoVerNetPlus).__proc_np_hv(np_map, hv_map)
        else:
            pred_inst = None

        if ls_map is not None:
            pred_layer = HoVerNetPlus.__proc_ls(ls_map)
        else:
            pred_layer = None

        inst_info_dict = None
        layer_dict = None

        if pred_type is not None or pred_layer is None:
            inst_id_list = np.unique(pred_inst)[1:]  # exclude background
            inst_info_dict = {}
            for inst_id in inst_id_list:
                inst_map = pred_inst == inst_id
                inst_box = get_bounding_box(inst_map)
                inst_box_tl = inst_box[:2]
                inst_map = inst_map[
                    inst_box[1] : inst_box[3], inst_box[0] : inst_box[2]
                ]
                inst_map = inst_map.astype(np.uint8)
                inst_moment = cv2.moments(inst_map)
                inst_contour = cv2.findContours(
                    inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # * opencv protocol format may break
                inst_contour = inst_contour[0][0].astype(np.int32)
                inst_contour = np.squeeze(inst_contour)

                # < 3 points dont make a contour, so skip, likely artifact too
                # as the contours obtained via approximation => too small
                if inst_contour.shape[0] < 3:  # pragma: no cover
                    continue
                # ! check for trickery shape
                if len(inst_contour.shape) != 2:  # pragma: no cover
                    continue

                inst_centroid = [
                    (inst_moment["m10"] / inst_moment["m00"]),
                    (inst_moment["m01"] / inst_moment["m00"]),
                ]
                inst_centroid = np.array(inst_centroid)
                inst_contour += inst_box_tl[None]
                inst_centroid += inst_box_tl  # X
                inst_info_dict[inst_id] = {  # inst_id should start at 1
                    "box": inst_box,
                    "centroid": inst_centroid,
                    "contour": inst_contour,
                    "prob": None,
                    "type": None,
                }

        if pred_type is not None:
            # * Get class of each instance id, stored at index id-1
            for inst_id in list(inst_info_dict.keys()):
                cmin, rmin, cmax, rmax = inst_info_dict[inst_id]["box"]
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                (type_list, type_pixels) = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)

                inst_type = type_list[0][0]

                # ! pick the 2nd most dominant if it exists
                if inst_type == 0 and len(type_list) > 1:  # pragma: no cover
                    inst_type = type_list[1][0]

                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["prob"] = float(type_prob)

        if pred_layer is not None:

            def image2contours(image, layer_info_dict, cnt, type_class):
                """Transforms image layers/regions into contours to store in dictionary.

                Args:
                    image (ndarray): Semantic segmentation map of different
                        layers/regions following processing.
                    layer_info_dict (dict): Dictionary to store layer contours in. It
                        has the following form:
                        layer_info = {
                                contour: number[][],
                                type: number,
                        }
                        layer_dict = {[layer_uid: number] : layer_info}
                    cnt (int): Counter.
                    type_class (int): The class of the layer to be processed.

                Returns:
                    layer_info_dict (dict): Updated layer dict.
                    cnt (int): Counter.

                """
                contours, _ = cv2.findContours(
                    image.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                for layer in contours:
                    coords = layer[:, 0, :]
                    layer_info_dict[str(cnt)] = {
                        "contours": coords.tolist(),
                        "type": type_class,
                    }
                    cnt += 1

                return layer_info_dict, cnt

            layer_list = np.unique(pred_layer)
            layer_list = np.delete(layer_list, np.where(layer_list == 0))
            layer_dict = {}
            count = 1

            for lyr in layer_list:
                layer = np.where(pred_layer == lyr, 1, 0).astype("uint8")
                layer_dict, count = image2contours(layer, layer_dict, count, lyr)

        return pred_inst, inst_info_dict, pred_layer, layer_dict

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        """
        patch_imgs = batch_data

        device = misc.select_device(on_gpu)
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred_dict = model(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            if "np" in pred_dict:
                pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            if "ls" in pred_dict:
                layer_map = F.softmax(pred_dict["ls"], dim=-1)
                layer_map = torch.argmax(layer_map, dim=-1, keepdim=True)
                layer_map = layer_map.type(torch.float32)
                pred_dict["ls"] = layer_map
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

        if "tp" in pred_dict and "ls" not in pred_dict:
            return pred_dict["np"], pred_dict["hv"], pred_dict["tp"]

        if "tp" in pred_dict and "ls" in pred_dict:
            return pred_dict["np"], pred_dict["hv"], pred_dict["tp"], pred_dict["ls"]

        if "tp" not in pred_dict and "ls" in pred_dict:
            return pred_dict["ls"]

        if "tp" not in pred_dict and "np" in pred_dict:
            return pred_dict["np"], pred_dict["hv"]
