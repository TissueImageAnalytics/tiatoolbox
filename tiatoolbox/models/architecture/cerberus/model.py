"""TIAToolbox integration wrapper for the Cerberus architecture."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.models_abc import ModelABC

from .net_desc import NetDesc
from .postproc import PostProcInstErodedContourMap

SPATIAL_NDIMS = 2

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


class Cerberus(ModelABC, NetDesc):
    """Cerberus multi-task model for glands, lumen, nuclei, and patch class."""

    head_names = (
        "Nuclei-INST",
        "Nuclei-TYPE",
        "Gland-INST",
        "Gland-TYPE",
        "Lumen-INST",
        "Patch-Class",
    )

    def __init__(
        self,
        patch_output_shape: tuple[int, int] = (144, 144),
        nuclei_type_dict: dict | None = None,
        gland_type_dict: dict | None = None,
        lumen_type_dict: dict | None = None,
    ) -> None:
        """Initialize the fixed Cerberus ResNet-34 model."""
        nn.Module.__init__(self)
        self._postproc = self.postproc
        self._preproc = self.preproc
        self.class_dict = None
        NetDesc.__init__(self)
        self.patch_output_shape = tuple(patch_output_shape)
        self.tasks = ("nuclei", "gland", "lumen")
        self.class_dict = {
            "nuclei": nuclei_type_dict
            or {
                0: "Background",
                1: "Neutrophil",
                2: "Epithelial",
                3: "Lymphocyte",
                4: "Plasma",
                5: "Eosinophil",
                6: "Connective",
            },
            "gland": gland_type_dict
            or {0: "Background", 1: "Gland", 2: "Surface Epithelium"},
            "lumen": lumen_type_dict or {0: "Background", 1: "Lumen"},
        }

    def forward(
        self, imgs: torch.Tensor, train_decoder_list: list[str] | None = None
    ) -> OrderedDict:
        """Forward pass through the shared encoder and selected Cerberus decoders."""
        return NetDesc.forward(self, imgs, train_decoder_list or [])

    def load_weights_from_file(self, weights: str | Path) -> torch.nn.Module:
        """Load Cerberus weights saved as ``weights.tar`` or a plain state dict."""
        state = torch.load(weights, map_location="cpu")
        state = state["desc"] if isinstance(state, dict) and "desc" in state else state
        state = _strip_dataparallel_prefix(state)
        self.load_state_dict(state, strict=True)
        return self

    @staticmethod
    def infer_batch(
        model: nn.Module, batch_data: np.ndarray | torch.Tensor, *, device: str
    ) -> tuple[np.ndarray, ...]:
        """Run Cerberus inference and return TIAToolbox-compatible head arrays."""
        patch_imgs = batch_data
        patch_imgs = patch_imgs.to(device).type(torch.float32)
        patch_imgs = patch_imgs.permute(0, 3, 1, 2).contiguous()

        model.eval()
        with torch.inference_mode():
            pred_dict = model(patch_imgs)
            pred_dict = OrderedDict(
                (k, v.permute(0, 2, 3, 1).contiguous()) for k, v in pred_dict.items()
            )

            pred_dict["Nuclei-INST"] = functional.softmax(
                pred_dict["Nuclei-INST"], dim=-1
            )[..., 1:]
            pred_dict["Gland-INST"] = functional.softmax(
                pred_dict["Gland-INST"], dim=-1
            )[..., 1:]
            pred_dict["Lumen-INST"] = functional.softmax(
                pred_dict["Lumen-INST"], dim=-1
            )[..., 1:]

            for key in ("Nuclei-TYPE", "Gland-TYPE"):
                type_map = functional.softmax(pred_dict[key], dim=-1)
                pred_dict[key] = torch.argmax(type_map, dim=-1, keepdim=True).type(
                    torch.float32
                )

            patch_class = functional.softmax(pred_dict["Patch-Class"], dim=-1)
            patch_class = torch.argmax(patch_class, dim=-1, keepdim=True).type(
                torch.float32
            )
            model_ = getattr(model, "module", model)
            output_shape = tuple(getattr(model_, "patch_output_shape", (144, 144)))

            pred_dict["Patch-Class"] = functional.interpolate(
                patch_class.permute(0, 3, 1, 2),
                size=output_shape,
                mode="nearest",
            ).permute(0, 2, 3, 1)

            outputs = []
            for head_name in Cerberus.head_names:
                head_output = pred_dict[head_name]
                if head_output.shape[1:3] != output_shape:
                    head_output = _crop_center_tensor(head_output, output_shape)
                outputs.append(head_output.cpu().numpy())

        return tuple(outputs)

    def postproc(
        self, raw_maps: list[np.ndarray | da.Array], offset: tuple[int, int] = (0, 0)
    ) -> tuple[dict, ...]:
        """Post-process Cerberus heads into annotation-store compatible tasks."""
        is_dask = isinstance(raw_maps[0], da.Array)
        maps = [raw_map.compute() if is_dask else raw_map for raw_map in raw_maps]

        head_map = dict(zip(self.head_names, maps, strict=False))
        outputs = []
        gland_inst_map = None
        for tissue_name, task_name in (
            ("Nuclei", "nuclei"),
            ("Gland", "gland"),
            ("Lumen", "lumen"),
        ):
            raw_map, idx_dict = _build_tissue_raw_map(head_map, tissue_name)
            inst_map, type_map = PostProcInstErodedContourMap.post_process(
                raw_map=raw_map,
                idx_dict=idx_dict,
                tissue_mode=tissue_name,
                ds_factor=1.0,
            )
            if tissue_name == "Gland":
                gland_inst_map = inst_map.copy()
            if tissue_name == "Lumen" and gland_inst_map is not None:
                inst_map = inst_map * (gland_inst_map > 0)
            if type_map is not None:
                type_map = np.squeeze(type_map).astype("uint8")

            inst_map = inst_map.astype("int32")
            inst_info_dict = HoVerNet.get_instance_info(
                inst_map,
                type_map,
                offset=offset,
                verbose=False,
            )
            info_dict = _inst_dict_for_dask_processing(inst_info_dict, is_dask=is_dask)
            outputs.append(
                {
                    "task_type": task_name,
                    "predictions": da.array(inst_map) if is_dask else inst_map,
                    "info_dict": info_dict,
                    "seg_type": "instance",
                }
            )

        return tuple(outputs)


def _strip_dataparallel_prefix(state: dict) -> dict:
    if all(key.split(".")[0] == "module" for key in state):
        return {".".join(key.split(".")[1:]): value for key, value in state.items()}
    return state


def _crop_center_tensor(
    tensor: torch.Tensor,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    h, w = tensor.shape[1:3]
    out_h, out_w = output_shape
    top = max((h - out_h) // 2, 0)
    left = max((w - out_w) // 2, 0)
    return tensor[:, top : top + out_h, left : left + out_w, :]


def _build_tissue_raw_map(
    head_map: dict[str, np.ndarray], tissue_name: str
) -> tuple[np.ndarray, dict[str, list[int]]]:
    idx_dict = {}
    maps = []
    start = 0
    for suffix in ("INST", "TYPE"):
        head_name = f"{tissue_name}-{suffix}"
        if head_name not in head_map:
            continue
        tissue_map = head_map[head_name]
        if tissue_map.ndim == SPATIAL_NDIMS:
            tissue_map = tissue_map[..., None]
        maps.append(tissue_map)
        stop = start + tissue_map.shape[-1]
        idx_dict[head_name] = [start, stop]
        start = stop

    return np.concatenate(maps, axis=-1), idx_dict


def _inst_dict_for_dask_processing(inst_info_dict: dict, *, is_dask: bool) -> dict:
    if not inst_info_dict:
        output = {
            "box": np.empty((0, 4), dtype=np.int32),
            "centroid": np.empty((0, 2), dtype=np.float32),
            "contours": np.empty((0, 0, 2), dtype=np.int32),
            "prob": np.empty((0,), dtype=np.float32),
            "type": np.empty((0,), dtype=np.int32),
        }
        if is_dask:
            return {key: da.from_array(value) for key, value in output.items()}
        return output

    inst_info_df = pd.DataFrame(inst_info_dict).transpose()
    output = {}
    for key, col in inst_info_df.items():
        col_np = col.to_numpy()
        if key == "contours":
            col_np = _pad_contours(col_np)
        elif key in {"box", "type"}:
            col_np = np.asarray(col_np.tolist(), dtype=np.int32)
        elif key in {"centroid", "prob"}:
            col_np = np.asarray(col_np.tolist(), dtype=np.float32)
        chunks = (len(col), *col_np.shape[1:])
        output[key] = da.from_array(col_np, chunks=chunks) if is_dask else col_np
    return output


def _pad_contours(contours: np.ndarray) -> np.ndarray:
    """Pad variable-length contours to a rectangular integer array."""
    max_len = max(contour.shape[0] for contour in contours)
    pad_value = np.iinfo(np.int32).min
    padded = np.full((len(contours), max_len, 2), pad_value, dtype=np.int32)
    for idx, contour in enumerate(contours):
        contour_ = np.asarray(contour, dtype=np.int32)
        padded[idx, : contour_.shape[0], :] = contour_
    return padded
