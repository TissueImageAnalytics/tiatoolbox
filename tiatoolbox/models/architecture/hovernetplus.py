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
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.architecture.utils import UpSample2x
from tiatoolbox.utils import misc


class HoVerNetPlus(HoVerNet):
    """Initialise HoVer-Net+.

    HoVer-Net+ takes an RGB input image, and provides the option to simultaneously
    segment and classify the nuclei present, aswell as semantically segment different
    regions or layers in the images. Note the HoVer-Net+ architecture assumes an image
    resolution of 0.5 mpp, in contrast to HoVer-Net at 0.25 mpp.

    """

    def __init__(
        self, num_input_channels: int = 3, num_types: int = None, num_layers: int = None
    ):
        """Initialise HoVer-Net+.

        Args:
            num_input_channels (int): The number of input channels, default = 3 for RGB.
            num_types (int): The number of types of nuclei present in the images.
            num_layers (int): The number of layers/different regions types present.

        """
        super().__init__(mode="fast")
        self.num_types = num_types
        self.num_layers = num_layers
        ksize = 3

        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    (
                        "tp",
                        HoVerNet._create_decoder_branch(ksize=ksize, out_ch=num_types),
                    ),
                    (
                        "np",
                        HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2),
                    ),
                    (
                        "hv",
                        HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2),
                    ),
                    (
                        "ls",
                        HoVerNet._create_decoder_branch(ksize=ksize, out_ch=num_layers),
                    ),
                ]
            )
        )

        self.upsample2x = UpSample2x()

    @staticmethod
    def _proc_ls(ls_map: np.ndarray):
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
    def _get_layer_info(pred_layer):
        """Transforms image layers/regions into contours to store in dictionary.

        Args:
            image (ndarray): Semantic segmentation map of different
                layers/regions following processing.

        Returns:
            layer_info_dict (dict): Dictionary to store layer contours in. It
                has the following form:
                layer_info = {
                        contour: number[][],
                        type: number,
                }
                layer_dict = {[layer_uid: number] : layer_info}

        """

        layer_list = np.unique(pred_layer)
        layer_list = np.delete(layer_list, np.where(layer_list == 0))
        layer_info_dict = {}
        count = 1

        for type_class in layer_list:
            layer = np.where(pred_layer == type_class, 1, 0).astype("uint8")
            contours, _ = cv2.findContours(
                layer.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            for layer in contours:
                coords = layer[:, 0, :]
                layer_info_dict[count] = {
                    "contours": coords,
                    "type": type_class,
                }
                count += 1

        return layer_info_dict

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
                inst_info = {
                        box: number[],
                        centroids: number[],
                        contour: number[][],
                        type: number,
                        prob: number,
                }
                inst_dict = {[inst_uid: number] : inst_info}
                and `inst_uid` is an integer corresponds to the instance
                having the same pixel value within `inst_map`.
            layer_map (ndarray): pixel-wise layer segmentation prediction.
            layer_dict (dict): a dictionary containing a mapping of each segmented
                layer within `layer_map`. It has the following form
                layer_info = {
                        contour: number[][],
                        type: number,
                }
                layer_dict = {[layer_uid: number] : layer_info}

        Examples:
            >>> from tiatoolbox.models.architecture.hovernet_plus import HoVerNetPlus
            >>> import torch
            >>> import numpy as np
            >>> batch = torch.from_numpy(image_patch)[None]
            >>> # image_patch is a 256x256x3 numpy array
            >>> weights_path = "A/weights.pth"
            >>> pretrained = torch.load(weights_path)
            >>> model = HoVerNetPlus(num_types=3, num_layers=5)
            >>> model.load_state_dict(pretrained)
            >>> output = model.infer_batch(model, batch, on_gpu=False)
            >>> output = [v[0] for v in output]
            >>> output = model.postproc(output)

        """
        np_map, hv_map, tp_map, ls_map = raw_maps

        pred_inst = HoVerNet._proc_np_hv(np_map, hv_map, fx=0.5)
        # fx=0.5 as nuclear processing is at 0.5 mpp instead of 0.25 mpp

        pred_layer = HoVerNetPlus._proc_ls(ls_map)
        pred_type = tp_map

        nuc_inst_info_dict = HoVerNet._get_instance_info(pred_inst, pred_type)
        layer_info_dict = HoVerNetPlus._get_layer_info(pred_layer)

        return pred_inst, nuc_inst_info_dict, pred_layer, layer_info_dict

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
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]

            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

            layer_map = F.softmax(pred_dict["ls"], dim=-1)
            layer_map = torch.argmax(layer_map, dim=-1, keepdim=True)
            layer_map = layer_map.type(torch.float32)
            pred_dict["ls"] = layer_map

            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

        return pred_dict["np"], pred_dict["hv"], pred_dict["tp"], pred_dict["ls"]
