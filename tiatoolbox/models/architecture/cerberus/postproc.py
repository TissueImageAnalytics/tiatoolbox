"""Post-processing for the released Cerberus ResNet-34 checkpoint."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage import morphology
from skimage.segmentation import watershed

CONTOUR_THRESHOLD = 0.5
GLAND_INNER_THRESHOLD = 0.55


def get_bounding_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return bounding box as ``rmin, rmax, cmin, cmax``."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1


class PostProcInstErodedContourMap:
    """Cerberus eroded-contour instance post-processing."""

    @staticmethod
    def _proc_gland(inst_fg: np.ndarray, ds_factor: float = 1.0) -> np.ndarray:
        """Extract labelled gland instances from inner and contour maps."""
        ksize = int((11 - 1) * ds_factor)
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]
        inst_cnt = inst_cnt_raw.copy()
        inst_cnt[inst_cnt > CONTOUR_THRESHOLD] = 1
        inst_cnt[inst_cnt <= CONTOUR_THRESHOLD] = 0

        inst_fg = np.array((inst_inner_raw - inst_cnt) > GLAND_INNER_THRESHOLD)
        inst_fg = morphology.remove_small_objects(
            inst_fg,
            max_size=int(1000 * (ds_factor**2)),
        )
        return _dilate_labelled_instances(inst_fg, k_disk)

    @staticmethod
    def _proc_lumen(inst_fg: np.ndarray, ds_factor: float = 1.0) -> np.ndarray:
        """Extract labelled lumen instances from inner and contour maps."""
        ksize = int((3 - 1) * ds_factor)
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]
        inst_cnt = inst_cnt_raw.copy()
        inst_cnt[inst_cnt > CONTOUR_THRESHOLD] = 1
        inst_cnt[inst_cnt <= CONTOUR_THRESHOLD] = 0

        inst_fg = np.array((inst_inner_raw - inst_cnt) > CONTOUR_THRESHOLD)
        inst_fg = morphology.remove_small_objects(
            inst_fg,
            max_size=int(150 * (ds_factor**2)),
        )
        return _dilate_labelled_instances(inst_fg, k_disk)

    @staticmethod
    def _proc_nuclei(inst_fg: np.ndarray, ds_factor: float = 1.0) -> np.ndarray:
        """Extract labelled nuclei instances from inner and contour maps."""
        _ = ds_factor
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]
        inst_raw = inst_inner_raw + inst_cnt_raw
        inst_msk = np.array(inst_raw > CONTOUR_THRESHOLD)

        if np.sum(inst_msk) == 0:
            return np.zeros(inst_msk.shape)

        inst_msk = cv2.erode(inst_msk.astype("uint8"), k_disk, iterations=1)
        inst_msk = label(inst_msk)[0]
        inst_msk = morphology.remove_small_objects(inst_msk, max_size=8)
        inst_msk = np.array(inst_msk > 0)

        inst_mrk = np.array(inst_inner_raw > CONTOUR_THRESHOLD)
        inst_mrk = label(inst_mrk)[0]
        inst_mrk = morphology.remove_small_objects(inst_mrk, max_size=4)

        marker = binary_fill_holes(inst_mrk.copy())
        marker = label(marker)[0]
        return watershed(-inst_inner_raw, marker, mask=inst_msk)

    @classmethod
    def post_process(
        cls,
        raw_map: np.ndarray,
        idx_dict: dict[str, list[int]],
        tissue_mode: str,
        ds_factor: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert Cerberus raw maps into instance and optional type maps."""
        func_dict = {
            "LUMEN": cls._proc_lumen,
            "GLAND": cls._proc_gland,
            "NUCLEI": cls._proc_nuclei,
        }
        tissue_key = tissue_mode.upper()
        if tissue_key not in func_dict:
            msg = f"Unsupported Cerberus tissue mode: {tissue_mode}"
            raise ValueError(msg)

        tissue_ch = f"{tissue_mode}-INST"
        if tissue_ch not in idx_dict:
            msg = f"Missing required Cerberus map: {tissue_ch}"
            raise KeyError(msg)

        inst_fg = raw_map[..., idx_dict[tissue_ch][0] : idx_dict[tissue_ch][1]]
        inst_map = func_dict[tissue_key](inst_fg, ds_factor)

        type_ch = f"{tissue_mode}-TYPE"
        if type_ch not in idx_dict:
            return inst_map, None

        type_map = raw_map[..., idx_dict[type_ch][0] : idx_dict[type_ch][1]]
        return inst_map, np.squeeze(type_map)


def _dilate_labelled_instances(inst_fg: np.ndarray, k_disk: np.ndarray) -> np.ndarray:
    """Label foreground instances, dilate each object, and fill holes."""
    inst_lab = label(inst_fg)[0]
    output_map = np.zeros(inst_lab.shape)
    for inst_id in np.unique(inst_lab).tolist()[1:]:
        inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)
        y1, y2, x1, x2 = get_bounding_box(inst_map)
        pad = k_disk.shape[0] * 2
        y1 = max(y1 - pad, 0)
        x1 = max(x1 - pad, 0)
        x2 = min(x2 + pad, inst_map.shape[1] - 1)
        y2 = min(y2 + pad, inst_map.shape[0] - 1)
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
        inst_map_crop = binary_fill_holes(inst_map_crop)
        output_region = output_map[y1:y2, x1:x2]
        output_region[inst_map_crop > 0] = inst_id
    return output_map
