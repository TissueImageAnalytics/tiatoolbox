import copy

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, measurements
from skimage import morphology
from skimage.segmentation import watershed


def get_bounding_box(img):
    """Return bounding box as rmin, rmax, cmin, cmax."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1


def get_inst_info_dict(inst_map, type_map, ds_factor=1.0):
    # get json information
    inst_info_dict = None
    inst_id_list = np.unique(inst_map)[1:]  # exclude background
    inst_info_dict = {}
    for inst_id in inst_id_list:
        single_inst_map = inst_map == inst_id
        # TODO: change format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(single_inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        single_inst_map = single_inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        single_inst_map = single_inst_map.astype(np.uint8)
        inst_moment = cv2.moments(single_inst_map)
        inst_contour = cv2.findContours(
            single_inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue  # ! check for too small a contour
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y

        # inst_id should start at 1
        inst_info_dict[inst_id] = {
            "box": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
        }

    if type_map is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["box"]).flatten()
            inst_map_crop = inst_map[rmin:rmax, cmin:cmax]
            inst_type_crop = type_map[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # resize to resolution used for processing
    if ds_factor != 1.0:
        for inst_id in list(inst_info_dict.keys()):
            inst_bbox = inst_info_dict[inst_id]["box"]
            inst_centroid = inst_info_dict[inst_id]["centroid"]
            inst_contour = inst_info_dict[inst_id]["contour"]
            if "type" in inst_info_dict[inst_id].keys():
                inst_type = inst_info_dict[inst_id]["type"]
                inst_type_prob = inst_info_dict[inst_id]["type_prob"]
            else:
                inst_type = None
                inst_type_prob = None
            inst_info_dict[inst_id] = {
                "box": np.round(inst_bbox / ds_factor).astype("int"),
                "centroid": np.round(inst_centroid / ds_factor).astype("int"),
                "contour": np.round(inst_contour / ds_factor).astype("int"),
            }
            if inst_type is not None:
                inst_info_dict[inst_id]["type"] = inst_type
                inst_info_dict[inst_id]["type_prob"] = inst_type_prob

    return inst_info_dict


class PostProcABC:
    @classmethod
    def to_save_dict(cls, pred_inst):
        inst_info_dict = None
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: change format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "box": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }


class PostProcInstErodedMap(PostProcABC):
    @staticmethod
    def __proc_gland(inst_fg, ds=1):

        ksize = 11
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_fg = np.squeeze(inst_fg)
        inst_fg = np.array(inst_fg > 0.5)
        inst_fg = morphology.remove_small_objects(inst_fg, max_size=1500)
        inst_lab = measurements.label(inst_fg)[0]

        output_map = np.zeros([inst_lab.shape[0], inst_lab.shape[1]])
        id_list = np.unique(inst_lab).tolist()[1:]
        for inst_id in id_list:
            inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst_map)
            pad = ksize * 2
            y1 = y1 - pad if y1 - pad >= 0 else y1
            x1 = x1 - pad if x1 - pad >= 0 else x1
            x2 = x2 + pad if x2 + pad <= inst_map.shape[1] - 1 else x2
            y2 = y2 + pad if y2 + pad <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]

            inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
            inst_map_crop = binary_fill_holes(inst_map_crop)

            output_region = output_map[y1:y2, x1:x2]
            output_region[inst_map_crop > 0] = inst_id

        return output_map

    @staticmethod
    def __proc_lumen(inst_fg, ds=1):

        ksize = 3
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_fg = np.squeeze(inst_fg)
        inst_fg = np.array(inst_fg > 0.5)
        inst_fg = morphology.remove_small_objects(inst_fg, max_size=150)
        inst_lab = measurements.label(inst_fg)[0]

        output_map = np.zeros([inst_lab.shape[0], inst_lab.shape[1]])
        id_list = np.unique(inst_lab).tolist()[1:]
        for inst_id in id_list:
            inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst_map)
            pad = ksize * 2
            y1 = y1 - pad if y1 - pad >= 0 else y1
            x1 = x1 - pad if x1 - pad >= 0 else x1
            x2 = x2 + pad if x2 + pad <= inst_map.shape[1] - 1 else x2
            y2 = y2 + pad if y2 + pad <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]

            inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
            inst_map_crop = binary_fill_holes(inst_map_crop)

            output_region = output_map[y1:y2, x1:x2]
            output_region[inst_map_crop > 0] = inst_id

        return output_map

    @staticmethod
    def __proc_nuclei(inst_fg, ds=1):

        ksize = 3
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_fg = np.squeeze(inst_fg)
        inst_fg = np.array(inst_fg > 0.5)
        inst_fg = morphology.remove_small_objects(inst_fg, max_size=8)
        inst_lab = measurements.label(inst_fg)[0]

        output_map = np.zeros([inst_lab.shape[0], inst_lab.shape[1]])
        id_list = np.unique(inst_lab).tolist()[1:]
        for inst_id in id_list:
            inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst_map)
            pad = ksize * 2
            y1 = y1 - pad if y1 - pad >= 0 else y1
            x1 = x1 - pad if x1 - pad >= 0 else x1
            x2 = x2 + pad if x2 + pad <= inst_map.shape[1] - 1 else x2
            y2 = y2 + pad if y2 + pad <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]

            inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
            inst_map_crop = binary_fill_holes(inst_map_crop)

            output_region = output_map[y1:y2, x1:x2]
            output_region[inst_map_crop > 0] = inst_id

        return output_map

    @classmethod
    def post_process(cls, raw_map, idx_dict, tissue_mode, scale=1.0):
        __func_dict = {
            "LUMEN": cls.__proc_lumen,
            "GLAND": cls.__proc_gland,
            "NUCLEI": cls.__proc_nuclei,
        }
        assert tissue_mode.upper() in __func_dict
        __func = __func_dict[tissue_mode.upper()]
        tissue_ch = "%s-INST" % tissue_mode
        assert tissue_ch in list(idx_dict.keys())

        inst_fg = raw_map[..., idx_dict[tissue_ch][0] : idx_dict[tissue_ch][1]]
        inst_map = __func(inst_fg)

        type_ch = tissue_mode + "-" + "TYPE"
        if type_ch in list(idx_dict.keys()):
            type_map = raw_map[..., idx_dict[type_ch][0] : idx_dict[type_ch][1]]
        else:
            type_map = None

        return inst_map, type_map


class PostProcInstErodedContourMap(PostProcABC):
    @staticmethod
    def __proc_gland(inst_fg, ds_factor=1.0):

        ksize_ = 11
        ksize = (ksize_ - 1) * ds_factor
        ksize = int(ksize)
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]

        inst_cnt = inst_cnt_raw.copy()
        inst_cnt[inst_cnt > 0.5] = 1
        inst_cnt[inst_cnt <= 0.5] = 0

        inst_fg = inst_inner_raw - inst_cnt
        inst_fg = np.array(inst_fg > 0.55)
        # inst_fg = morphology.remove_small_objects(inst_fg, max_size=1500)
        inst_fg = morphology.remove_small_objects(
            inst_fg,
            max_size=int(1000 * (ds_factor**2)),
        )
        inst_lab = measurements.label(inst_fg)[0]

        output_map = np.zeros([inst_lab.shape[0], inst_lab.shape[1]])
        id_list = np.unique(inst_lab).tolist()[1:]
        for inst_id in id_list:
            inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst_map)
            pad = ksize * 2
            y1 = y1 - pad if y1 - pad >= 0 else y1
            x1 = x1 - pad if x1 - pad >= 0 else x1
            x2 = x2 + pad if x2 + pad <= inst_map.shape[1] - 1 else x2
            y2 = y2 + pad if y2 + pad <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]

            inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
            inst_map_crop = binary_fill_holes(inst_map_crop)

            output_region = output_map[y1:y2, x1:x2]
            output_region[inst_map_crop > 0] = inst_id

        return output_map

    @staticmethod
    def __proc_lumen(inst_fg, ds_factor=1.0):

        ksize_ = 3
        ksize = (ksize_ - 1) * ds_factor
        ksize = int(ksize)
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]

        inst_cnt = inst_cnt_raw.copy()
        inst_cnt[inst_cnt > 0.5] = 1
        inst_cnt[inst_cnt <= 0.5] = 0

        inst_fg = inst_inner_raw - inst_cnt
        inst_fg = np.array(inst_fg > 0.5)
        inst_fg = morphology.remove_small_objects(
            inst_fg,
            max_size=int(150 * (ds_factor**2)),
        )
        inst_lab = measurements.label(inst_fg)[0]

        output_map = np.zeros([inst_lab.shape[0], inst_lab.shape[1]])
        id_list = np.unique(inst_lab).tolist()[1:]
        for inst_id in id_list:
            inst_map = np.array(inst_lab == inst_id, dtype=np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst_map)
            pad = ksize * 2
            y1 = y1 - pad if y1 - pad >= 0 else y1
            x1 = x1 - pad if x1 - pad >= 0 else x1
            x2 = x2 + pad if x2 + pad <= inst_map.shape[1] - 1 else x2
            y2 = y2 + pad if y2 + pad <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]

            inst_map_crop = cv2.dilate(inst_map_crop, k_disk, iterations=1)
            inst_map_crop = binary_fill_holes(inst_map_crop)

            output_region = output_map[y1:y2, x1:x2]
            output_region[inst_map_crop > 0] = inst_id

        return output_map

    @staticmethod
    def __proc_nuclei(inst_fg, ds_factor=1.0):

        ksize = 3
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        inst_inner_raw = inst_fg[..., 0]
        inst_cnt_raw = inst_fg[..., 1]
        inst_raw = inst_inner_raw + inst_cnt_raw

        # binarise
        inst_msk = np.array(inst_raw > 0.5)
        if np.sum(inst_msk) > 0:
            inst_msk = cv2.erode(inst_msk.astype("uint8"), k_disk, iterations=1)
            inst_msk = measurements.label(inst_msk)[0]
            inst_msk = morphology.remove_small_objects(inst_msk, max_size=8)
            inst_msk = np.array(inst_msk > 0)

            inst_mrk = inst_inner_raw
            inst_mrk = np.array(inst_mrk > 0.5)
            inst_mrk = measurements.label(inst_mrk)[0]
            inst_mrk = morphology.remove_small_objects(inst_mrk, max_size=4)

            marker = inst_mrk.copy()
            marker = binary_fill_holes(marker)
            marker = measurements.label(marker)[0]
            output_map = watershed(-inst_inner_raw, marker, mask=inst_msk)
        else:
            output_map = np.zeros([inst_msk.shape[0], inst_msk.shape[1]])
        return output_map

    @classmethod
    def post_process(cls, raw_map, idx_dict, tissue_mode, ds_factor=1.0):
        __func_dict = {
            "LUMEN": cls.__proc_lumen,
            "GLAND": cls.__proc_gland,
            "NUCLEI": cls.__proc_nuclei,
        }
        assert tissue_mode.upper() in __func_dict
        __func = __func_dict[tissue_mode.upper()]
        tissue_ch = f"{tissue_mode}-INST"

        idx_dict = copy.deepcopy(idx_dict)
        assert tissue_ch in list(idx_dict.keys())

        inst_fg = raw_map[..., idx_dict[tissue_ch][0] : idx_dict[tissue_ch][1]]
        inst_map = __func(inst_fg, ds_factor)

        type_ch = tissue_mode + "-" + "TYPE"
        if type_ch in list(idx_dict.keys()):
            type_map = raw_map[..., idx_dict[type_ch][0] : idx_dict[type_ch][1]]
            type_map = np.squeeze(type_map)
        else:
            type_map = None

        return inst_map, type_map
