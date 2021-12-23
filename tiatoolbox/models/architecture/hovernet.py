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


import math
from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture.utils import (
    UpSample2x,
    centre_crop,
    centre_crop_to_shape,
)
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import get_bounding_box


class TFSamepaddingLayer(nn.Module):
    """To align with tensorflow `same` padding.

    Putting this before any conv layer that needs padding. Here,
    we assume kernel has same height and width for simplicity.

    """

    def __init__(self, ksize: int, stride: int):
        super().__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init."""
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        x = F.pad(x, padding, "constant", 0)
        return x


class DenseBlock(nn.Module):
    """Dense Convolutional Block.

    This convolutional block supports only `valid` padding.

    References:
        Huang, Gao, et al. "Densely connected convolutional networks."
        Proceedings of the IEEE conference on computer vision and
        pattern recognition. 2017.

    """

    def __init__(
        self,
        in_ch: int,
        unit_ksizes: List[int],
        unit_chs: List[int],
        unit_count: int,
        split: int = 1,
    ):
        super().__init__()
        if len(unit_ksizes) != len(unit_chs):
            raise ValueError("Unbalance Unit Info")

        self.nr_unit = unit_count
        self.in_ch = in_ch

        # weights value may not match with tensorflow version
        # due to different default intialization scheme between
        # torch and tensorflow
        def get_unit_block(unit_in_ch):
            """Helper function to make it less long."""
            layers = OrderedDict(
                [
                    ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("preact_bna/relu", nn.ReLU(inplace=True)),
                    (
                        "conv1",
                        nn.Conv2d(
                            unit_in_ch,
                            unit_chs[0],
                            unit_ksizes[0],
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("conv1/bn", nn.BatchNorm2d(unit_chs[0], eps=1e-5)),
                    ("conv1/relu", nn.ReLU(inplace=True)),
                    (
                        "conv2",
                        nn.Conv2d(
                            unit_chs[0],
                            unit_chs[1],
                            unit_ksizes[1],
                            groups=split,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                ]
            )
            return nn.Sequential(layers)

        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for _ in range(unit_count):
            self.units.append(get_unit_block(unit_in_ch))
            unit_in_ch += unit_chs[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, prev_feat: torch.Tensor):
        """Logic for using layers defined in init."""
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = centre_crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class ResidualBlock(nn.Module):
    """Residual block.

    References:
        He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and
        pattern recognition. 2016.

    """

    def __init__(
        self,
        in_ch: int,
        unit_ksizes: List[int],
        unit_chs: List[int],
        unit_count: int,
        stride: int = 1,
    ):
        super().__init__()
        if len(unit_ksizes) != len(unit_chs):
            raise ValueError("Unbalance Unit Info")

        self.nr_unit = unit_count
        self.in_ch = in_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                (
                    "conv1",
                    nn.Conv2d(
                        unit_in_ch,
                        unit_chs[0],
                        unit_ksizes[0],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv1/bn", nn.BatchNorm2d(unit_chs[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(
                        ksize=unit_ksizes[1], stride=stride if idx == 0 else 1
                    ),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_chs[0],
                        unit_chs[1],
                        unit_ksizes[1],
                        stride=stride if idx == 0 else 1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_chs[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv2d(
                        unit_chs[1],
                        unit_chs[2],
                        unit_ksizes[2],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
            ]
            # has BatchNorm-Activation layers to conclude each
            # previous block so must not put preact for the first
            # unit of this block
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_chs[-1]

        if in_ch != unit_chs[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_chs[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, prev_feat: torch.Tensor):
        """Logic for using layers defined in init."""
        if self.shortcut is None:
            shortcut = prev_feat
        else:
            shortcut = self.shortcut(prev_feat)

        for idx in range(0, len(self.units)):
            new_feat = prev_feat
            new_feat = self.units[idx](new_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        feat = self.blk_bna(prev_feat)
        return feat


class HoVerNet(ModelABC):
    """HoVer-Net Architecture.

    Args:
        num_input_channels (int): Number of channels in input.
        num_types (int): Number of nuclei types within the predictions.
          Once define, a branch dedicated for typing is created.
          By default, no typing (`num_types=None`) is used.
        mode (str): To use architecture defined in as in original paper
          (`original`) or the one used in PanNuke paper (`fast`).

    References:
        Graham, Simon, et al. "Hover-net: Simultaneous segmentation and
        classification of nuclei in multi-tissue histology images."
        Medical Image Analysis 58 (2019): 101563.

        Gamper, Jevgenij, et al. "PanNuke dataset extension, insights and baselines."
        arXiv preprint arXiv:2003.10778 (2020).

    """

    def __init__(
        self, num_input_channels: int = 3, num_types: int = None, mode: str = "original"
    ):
        super().__init__()
        self.mode = mode
        self.num_types = num_types

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

        ksize = 5 if mode == "original" else 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        (
                            "tp",
                            HoVerNet._create_decoder_branch(
                                ksize=ksize, out_ch=num_types
                            ),
                        ),
                        ("np", HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", HoVerNet._create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    # skipcq: PYL-W0221
    def forward(self, imgs: torch.Tensor):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            imgs (torch.Tensor): Input images, the tensor is in the shape of NCHW.

        Returns:
            output (dict): A dictionary containing the inference output.
                The expected format os {decoder_name: prediction}.

        """
        imgs = imgs / 255.0  # to 0-1 range to match XY

        d0 = self.conv0(imgs)
        d0 = self.d0(d0)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        if self.mode == "original":
            d[0] = centre_crop(d[0], [184, 184])
            d[1] = centre_crop(d[1], [72, 72])
        else:
            d[0] = centre_crop(d[0], [92, 92])
            d[1] = centre_crop(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict

    @staticmethod
    def _create_decoder_branch(out_ch=2, ksize=5):
        """Helper to create a decoder branch."""
        modules = [
            ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
            ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
            (
                "convf",
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ),
        ]
        u3 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
            ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
            (
                "convf",
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ),
        ]
        u2 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
            (
                "conva",
                nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),
            ),
        ]
        u1 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
            (
                "conv",
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ),
        ]
        u0 = nn.Sequential(OrderedDict(modules))

        decoder = nn.Sequential(
            OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
        )
        return decoder

    @staticmethod
    def _proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray, fx: float = 1):
        """Extract Nuclei Instance with NP and HV Map.

        Sobel will be applied on horizontal and vertical channel in
        `hv_map` to derive a energy landscape which highligh possible
        nuclei instance boundaries. Afterward, watershed with markers
        is applied on the above energy map using the `np_map` as filter
        to remove background regions.

        Args:
            np_map (np.ndarray): An image of shape (heigh, width, 1) which
              contains the probabilities of a pixel being a nuclei.
            hv_map (np.ndarray): An array of shape (heigh, width, 2) which
              contains the horizontal (channel 0) and vertical (channel 1)
              of possible instances exist withint the images.
            fx (float): The scale factor for processing nuclei. The scale
              assumes an image of resolution 0.25 microns per pixel. Default
              is therefore 1 for HoVer-Net.

        Returns:
            An np.ndarray of shape (height, width) where each non-zero values
            within the array correspond to one detected nuclei instances.

        """
        blb_raw = np_map[..., 0]
        h_dir_raw = hv_map[..., 0]
        v_dir_raw = hv_map[..., 1]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * fx) + 1)
        obj_size = math.ceil(10 * (fx ** 2))
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=obj_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    @staticmethod
    def _get_instance_info(pred_inst, pred_type=None):
        """To collect instance information and store it within a dictionary.

        Args:
            pred_inst (np.ndarray): An image of shape (heigh, width) which
                contains the probabilities of a pixel being a nuclei.
            pred_type (np.ndarray): An image of shape (heigh, width, 1) which
                contains the probabilities of a pixel being a certain type of nuclei.

        Returns:
            inst_info_dict (dict): A dictionary containing a mapping of each instance
                    within `pred_inst` instance information. It has following form

                    inst_info = {
                            box: number[],
                            centroids: number[],
                            contour: number[][],
                            type: number,
                            prob: number,
                    }
                    inst_info_dict = {[inst_uid: number] : inst_info}

                    and `inst_uid` is an integer corresponds to the instance
                    having the same pixel value within `pred_inst`.

        """
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_box = get_bounding_box(inst_map)
            inst_box_tl = inst_box[:2]
            inst_map = inst_map[inst_box[1] : inst_box[3], inst_box[0] : inst_box[2]]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = inst_contour[0][0].astype(np.int32)
            inst_contour = np.squeeze(inst_contour)

            # < 3 points does not make a contour, so skip, likely artifact too
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

        return inst_info_dict

    @staticmethod
    # skipcq: PYL-W0221
    def postproc(raw_maps: List[np.ndarray]):
        """Post processing script for image tiles.

        Args:
            raw_maps (list(ndarray)): list of prediction output of each head and
                assumed to be in the order of [np, hv, tp] (match with the output
                of `infer_batch`).

        Returns:
            inst_map (ndarray): pixel-wise nuclear instance segmentation
                prediction.
            inst_dict (dict): a dictionary containing a mapping of each instance
                within `inst_map` instance information. It has following form

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

        Examples:
            >>> from tiatoolbox.models.architecture.hovernet import HoVerNet
            >>> import torch
            >>> import numpy as np
            >>> batch = torch.from_numpy(image_patch)[None]
            >>> # image_patch is a 256x256x3 numpy array
            >>> weights_path = "A/weights.pth"
            >>> pretrained = torch.load(weights_path)
            >>> model = HoVerNet(num_types=6, mode="fast")
            >>> model.load_state_dict(pretrained)
            >>> output = model.infer_batch(model, batch, on_gpu=False)
            >>> output = [v[0] for v in output]
            >>> output = model.postproc(output)

        """
        if len(raw_maps) == 3:
            np_map, hv_map, tp_map = raw_maps
        else:
            tp_map = None
            np_map, hv_map = raw_maps

        pred_type = tp_map
        pred_inst = HoVerNet._proc_np_hv(np_map, hv_map)
        nuc_inst_info_dict = HoVerNet._get_instance_info(pred_inst, pred_type)

        return pred_inst, nuc_inst_info_dict

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

        Returns:
            List of output from each head, each head is expected to contain
            N predictions for N input patches. There are two cases, one
            with 2 heads (Nuclei Pixels `np` and Hover `hv`) or with 2 heads
            (`np`, `hv`, and Nuclei Types `tp`).

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
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

        if "tp" in pred_dict:
            return pred_dict["np"], pred_dict["hv"], pred_dict["tp"]
        return pred_dict["np"], pred_dict["hv"]
