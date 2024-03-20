"""Define HoVerNetPlus architecture."""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from skimage import morphology
from torch import nn

from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.architecture.utils import UpSample2x


class HoVerNetPlus(HoVerNet):
    """Initialise HoVerNet+ [1].

    HoVerNet+ takes an RGB input image, and provides the option to
    simultaneously segment and classify the nuclei present, as well as
    semantically segment different regions or layers in the images. Note
    the HoVerNet+ architecture assumes an image resolution of 0.5 mpp,
    in contrast to HoVerNet at 0.25 mpp.

    The tiatoolbox model should produce following results on the specified datasets
    that it was trained on.

    .. list-table:: HoVerNet+ Performance for Nuclear Instance Segmentation
       :widths: 15 15 15 15 15 15 15
       :header-rows: 1

       * - Model name
         - Data set
         - DICE
         - AJI
         - DQ
         - SQ
         - PQ
       * - hovernetplus-oed
         - OED
         - 0.84
         - 0.69
         - 0.86
         - 0.80
         - 0.69

    .. list-table:: HoVerNet+ Mean Performance for Semantic Segmentation
       :widths: 15 15 15 15 15 15
       :header-rows: 1

       * - Model name
         - Data set
         - F1
         - Precision
         - Recall
         - Accuracy
       * - hovernetplus-oed
         - OED
         - 0.82
         - 0.82
         - 0.82
         - 0.84

    Args:
        num_input_channels (int):
            The number of input channels, default = 3 for RGB.
        num_types (int):
            The number of types of nuclei present in the images.
        num_layers (int):
            The number of layers/different regions types present.

    References:
        [1] Shephard, Adam J., et al. "Simultaneous Nuclear Instance and
        Layer Segmentation in Oral Epithelial Dysplasia." Proceedings of
        the IEEE/CVF International Conference on Computer Vision. 2021.


    """

    def __init__(
        self: HoVerNetPlus,
        num_input_channels: int = 3,
        num_types: int | None = None,
        num_layers: int | None = None,
        nuc_type_dict: dict | None = None,
        layer_type_dict: dict | None = None,
    ) -> None:
        """Initialize :class:`HoVerNetPlus`."""
        super().__init__(mode="fast")
        self.num_input_channels = num_input_channels
        self.num_types = num_types
        self.num_layers = num_layers
        self.nuc_type_dict = nuc_type_dict
        self.layer_type_dict = layer_type_dict
        ksize = 3

        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    (
                        "tp",
                        self._create_decoder_branch(ksize=ksize, out_ch=num_types),
                    ),
                    (
                        "np",
                        self._create_decoder_branch(ksize=ksize, out_ch=2),
                    ),
                    (
                        "hv",
                        self._create_decoder_branch(ksize=ksize, out_ch=2),
                    ),
                    (
                        "ls",
                        self._create_decoder_branch(ksize=ksize, out_ch=num_layers),
                    ),
                ],
            ),
        )

        self.upsample2x = UpSample2x()

    @staticmethod
    def _proc_ls(ls_map: np.ndarray) -> np.ndarray:
        """Extract Layer Segmentation map with LS Map.

        This function takes the layer segmentation map and applies various morphological
        operations remove spurious segmentations. Note, this processing is specific to
        oral epithelium, where prioirty is given to certain tissue layers.

        Args:
            ls_map:
                The input predicted segmentation map.

        Returns:
            :class:`numpy.ndarray`:
                The processed segmentation map.

        """
        ls_map = np.squeeze(ls_map)
        ls_map = np.around(ls_map).astype("uint8")  # ensure all numbers are integers
        min_size = 20000
        kernel_size = 20

        epith_all = np.where(ls_map >= 2, 1, 0).astype("uint8")  # noqa: PLR2004
        mask = np.where(ls_map >= 1, 1, 0).astype("uint8")
        epith_all = epith_all > 0
        epith_mask = morphology.remove_small_objects(
            epith_all,
            min_size=min_size,
        ).astype("uint8")
        epith_edited = epith_mask * ls_map
        epith_edited = epith_edited.astype("uint8")
        epith_edited_open = np.zeros_like(epith_edited).astype("uint8")
        for i in [3, 2, 4]:
            tmp = np.where(epith_edited == i, 1, 0).astype("uint8")
            ep_open = cv2.morphologyEx(
                tmp,
                cv2.MORPH_CLOSE,
                np.ones((kernel_size, kernel_size)),
            )
            ep_open = cv2.morphologyEx(
                ep_open,
                cv2.MORPH_OPEN,
                np.ones((kernel_size, kernel_size)),
            )
            epith_edited_open[ep_open == 1] = i

        mask_open = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size)),
        )
        mask_open = cv2.morphologyEx(
            mask_open,
            cv2.MORPH_OPEN,
            np.ones((kernel_size, kernel_size)),
        ).astype("uint8")
        ls_map = mask_open.copy()
        for i in range(2, 5):
            ls_map[epith_edited_open == i] = i

        return ls_map.astype("uint8")

    @staticmethod
    def _get_layer_info(pred_layer: np.ndarray) -> dict:
        """Transforms image layers/regions into contours to store in dictionary.

        Args:
            pred_layer (:class:`numpy.ndarray`):
                Semantic segmentation map of different layers/regions
                following processing.

        Returns:
            dict:
                A dictionary of layer contours. It has the
                following form:

                .. code-block:: json

                    {
                        1: {  # Instance ID
                            "contour": [
                                [x, y],
                                ...
                            ],
                            "type": integer,
                        },
                        ...
                    }

        """
        layer_list = np.unique(pred_layer)
        layer_list = np.delete(layer_list, np.where(layer_list == 0))
        layer_info_dict = {}
        count = 1

        for type_class in layer_list:
            layer = np.where(pred_layer == type_class, 1, 0).astype("uint8")
            contours, _ = cv2.findContours(
                layer.astype("uint8"),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
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
    # skipcq: PYL-W0221  # noqa: ERA001
    def postproc(raw_maps: list[np.ndarray]) -> tuple:
        """Post-processing script for image tiles.

        Args:
            raw_maps (list(ndarray)):
                A list of prediction outputs of each head and assumed to
                be in the order of [np, hv, tp, ls] (match with the
                output of `infer_batch`).

        Returns:
            tuple:
                - inst_map (ndarray):
                    Pixel-wise nuclear instance segmentation prediction.
                - inst_dict (dict):
                    A dictionary containing a mapping of each instance
                    within `inst_map` instance information. It has the
                    following form:

                    .. code-block:: json

                        {
                            0: {  # Instance ID
                                "box": [
                                    x_min,
                                    y_min,
                                    x_max,
                                    y_max,
                                ],
                                "centroid": [x, y],
                                "contour": [
                                    [x, y],
                                    ...
                                ],
                                "type": integer,
                                "prob": float,
                            },
                            ...
                        }

                    where the instance ID is an integer corresponding to the
                    instance at the same pixel value within `inst_map`.
                - layer_map (ndarray):
                    Pixel-wise layer segmentation prediction.
                - layer_dict (dict):
                    A dictionary containing a mapping of each segmented
                    layer within `layer_map`. It has the following form

                    .. code-block:: json

                        {
                            1: {  # Instance ID
                                "contour": [
                                    [x, y],
                                    ...
                                ],
                                "type": integer,
                            },
                            ...
                        }

        Examples:
            >>> from tiatoolbox.models.architecture.hovernetplus import HoVerNetPlus
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

        pred_inst = HoVerNetPlus._proc_np_hv(np_map, hv_map, scale_factor=0.5)
        # fx=0.5 as nuclear processing is at 0.5 mpp instead of 0.25 mpp

        pred_layer = HoVerNetPlus._proc_ls(ls_map)
        pred_type = np.around(tp_map).astype("uint8")

        nuc_inst_info_dict = HoVerNet.get_instance_info(pred_inst, pred_type)
        layer_info_dict = HoVerNetPlus._get_layer_info(pred_layer)

        return pred_inst, nuc_inst_info_dict, pred_layer, layer_info_dict

    @staticmethod
    def infer_batch(model: nn.Module, batch_data: np.ndarray, *, device: str) -> tuple:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        """
        patch_imgs = batch_data

        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred_dict = model(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()],
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
