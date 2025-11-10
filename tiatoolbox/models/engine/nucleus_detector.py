"""This module implements nucleus detection engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from tiatoolbox import logger
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
)
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.annotation import AnnotationStore
from tiatoolbox.utils.misc import df_to_store_nucleus_detector

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.models.models_abc import ModelABC



def probability_to_peak_map(
    img2d: np.ndarray,
    min_distance: int,
    threshold_abs: float,
    threshold_rel: float = 0.0,
) -> np.ndarray:
    """Build a boolean mask (H, W) of objects from a 2D probability map using peak_local_max.

    Args:
        img2d (np.ndarray): 2D probability map.
        min_distance (int): Minimum distance between peaks.
        threshold_abs (float): Absolute threshold for peak detection.
        threshold_rel (float, optional): Relative threshold for peak detection. Defaults to 0.0.

    Returns:
        mask (np.ndarray): Boolean mask (H, W) with True at peak locations.
    """
    H, W = img2d.shape
    mask = np.zeros((H, W), dtype=bool)
    coords = peak_local_max(
        img2d,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
    )
    if coords.size:
        r, c = coords[:, 0], coords[:, 1]
        mask[r, c] = True
    return mask


def peak_detection_mapoverlap(
    block: np.ndarray,
    block_info,
    min_distance: int,
    threshold_abs: float,
    depth_h: int,
    depth_w: int,
    calculate_probabilities: bool = False,
) -> np.ndarray:
    """Runs inside Dask.da.map_overlap on a padded NumPy block: (h_pad, w_pad, C).
    Builds a processed mask per channel, runs peak_local_max then
    label+regionprops, and writes probability (mean_intensity) at centroid pixels.
    Keeps only centroids whose (row,col) lie in the interior window:
        rows [depth_h : depth_h + core_h), cols [depth_w : depth_w + core_w)
    Returns same spatial shape as input block: (h_pad, w_pad, C), float32.

    Args:
        block: NumPy array (H, W, C) with padded block data.
        block_info: Dask block info dict.
        min_distance: Minimum distance in pixels between peaks.
        threshold_abs: Minimum absolute threshold for peak detection.
        depth_h: Halo size in pixels for height (rows).
        depth_w: Halo size in pixels for width (cols).
        calculate_probabilities: If True, write mean_intensity at centroids;
            else write 1.0 at centroids.

    Returns:
        out: NumPy array (H, W, C) with probabilities at centroids, 0 elsewhere.
    """
    H, W, C = block.shape

    # --- derive core (pre-overlap) size for THIS block safely ---
    info = block_info[0]
    locs = info["array-location"]  # [(r0,r1),(c0,c1),(ch0,ch1)]
    core_h = int(locs[0][1] - locs[0][0])  # r1 - r0
    core_w = int(locs[1][1] - locs[1][0])

    rmin, rmax = depth_h, depth_h + core_h
    cmin, cmax = depth_w, depth_w + core_w

    out = np.zeros((H, W, C), dtype=np.float32)

    for ch in range(C):
        img = np.asarray(block[..., ch])  # NumPy 2D view
        pmask = probability_to_peak_map(img, min_distance, threshold_abs)
        if not pmask.any():
            continue

        lab = label(pmask)
        props = regionprops(lab, intensity_image=img)

        for reg in props:
            r, c = reg.centroid  # floats in padded-block coords
            if (rmin <= r < rmax) and (cmin <= c < cmax):
                rr = int(round(r))
                cc = int(round(c))
                if 0 <= rr < H and 0 <= cc < W:
                    if calculate_probabilities:
                        out[rr, cc, ch] = float(reg.mean_intensity)
                    else:
                        out[rr, cc, ch] = 1.0

    return out


def detection_with_map_overlap(
    probs: da.Array, min_distance: int, threshold_abs: float, depth_pixels: int
) -> da.Array:
    """probs: Dask array (H, W, C), float.
    depth_pixels: halo in pixels for H/W (use >= min_distance and >= any morphology radius).

    Returns:
      scores: da.Array (H, W, C) with mean_intensity at centroids, 0 elsewhere.
    """
    depth = {0: depth_pixels, 1: depth_pixels, 2: 0}
    scores = da.map_overlap(
        probs,
        peak_detection_mapoverlap,
        depth=depth,
        boundary=0,
        dtype=np.float32,
        block_info=True,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        depth_h=depth_pixels,
        depth_w=depth_pixels,
    )
    return scores


def centroids_map_to_dask_dataframe(
    scores: da.Array, x_offset: int = 0, y_offset: int = 0
) -> dd.DataFrame:
    """Convert centroid map (H, W, C) into a Dask DataFrame with columns: x, y, type, prob.

    Args:
        scores: Dask array (H, W, C) with probabilities at centroids, 0 elsewhere.
        x_offset: global x offset to add to all x coordinates.
        y_offset: global y offset to add to all y coordinates.

    Returns:
        ddf: Dask DataFrame with columns: x, y, type, prob.
    """
    # 1) Build a boolean mask of detections

    mask = scores > 0
    # 2) Get coordinates and class of detections (lazy 1D Dask arrays)

    yy, xx, cc = da.nonzero(mask)
    # 3) Get probability values at those detections (lazy) — same length as yy/xx/cc

    ss = da.extract(mask, scores)
    # 4) Assemble a Dask DataFrame
    # all columns are row-wise aligned (all built from arrays of the same length).
    ddf = dd.concat(
        [
            dd.from_dask_array(xx.astype("int64"), columns="x"),
            dd.from_dask_array(yy.astype("int64"), columns="y"),
            dd.from_dask_array(cc.astype("int64"), columns="type"),
            dd.from_dask_array(ss.astype("float32"), columns="prob"),
        ],
        axis=1,
        ignore_unknown_divisions=True,
    )

    # 5) Apply global offsets (if needed)
    if x_offset != 0:
        ddf["x"] = ddf["x"] + int(x_offset)
    if y_offset != 0:
        ddf["y"] = ddf["y"] + int(y_offset)

    return ddf


def nucleus_detection_nms(
    df: pd.DataFrame, radius: int, overlap_threshold: float = 0.5
) -> pd.DataFrame:
    """Greedy NMS across ALL detections.

    Keeps the highest-prob detection, removes any other point within 'radius' pixels > overlap_threshold.
    Expects dataframe columns: ['x','y','type','prob'].

    Args:
        df: pandas DataFrame of detections.
        radius: radius in pixels for suppression.
        overlap_threshold: float in [0,1], fraction of radius for suppression.

    Returns:
        filtered DataFrame with same columns/dtypes.
    """
    if df.empty:
        return df.copy()
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if not (0.0 < overlap_threshold <= 1.0):
        raise ValueError("overlap_threshold must be in (0.0, 1.0]")

    # Sort by descending probability (highest priority first)
    sub = df.sort_values("prob", ascending=False).reset_index(drop=True)

    # Coordinates as float64 for distance math
    coords = sub[["x", "y"]].to_numpy(dtype=np.float64)
    r2 = float(radius) * float(radius)

    coords = sub[["x", "y"]].to_numpy(dtype=np.float64)
    r = float(radius)
    two_r = 2.0 * r
    two_r2 = two_r * two_r  # distance^2 cutoff for any overlap

    suppressed = np.zeros(len(sub), dtype=bool)
    keep_idx = []

    for i in range(len(sub)):
        if suppressed[i]:
            continue

        keep_idx.append(i)

        # Vectorised distances to all points
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        d2 = dx * dx + dy * dy

        # Only points with d < 2r can have nonzero overlap
        cand = d2 <= two_r2
        cand[i] = False  # don't suppress the kept point itself
        if not np.any(cand):
            continue

        d = np.sqrt(d2[cand])

        # Safe cosine argument = (distance ÷ diameter), Clamp for numerical stability
        u = np.clip(d / (2.0 * r), -1.0, 1.0)
        # Exact intersection area of two equal-radius circles.
        inter = 2.0 * (r * r) * np.arccos(u) - 0.5 * d * np.sqrt(
            np.clip(4.0 * r * r - d * d, 0.0, None)
        )

        union = 2.0 * np.pi * (r * r) - inter
        iou = inter / union

        # Suppress candidates whose IoU exceeds threshold
        idx_cand = np.where(cand)[0]
        to_suppress = idx_cand[iou >= overlap_threshold]
        suppressed[to_suppress] = True

    kept = sub.iloc[keep_idx].copy()
    return kept


class NucleusDetector(SemanticSegmentor):
    r"""Nucleus detection engine.

    The models provided by tiatoolbox should give the following results:

    .. list-table:: Nucleus detection performance on the (add models list here)
       :widths: 15 15
       :header-rows: 1

    Args:
        model (str or nn.Module):
            Defined PyTorch model or name of the existing models support by
            tiatoolbox for processing the data e.g., mapde-conic, sccnn-conic.
            For a full list of pretrained models, please refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`.
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument is case insensitive.
        batch_size (int):
            Number of images fed into the model each time.
        num_workers (int):
            Number of workers used in torch.utils.data.DataLoader.
        weights (str or pathlib.Path, optional):
            Pretrained weights file path or name of the existing weights
            supported by tiatoolbox. If ``None``, and `model` is a string,
            the default pretrained weights for the specified model will be used.
            If `model` is a nn.Module, no weights will be loaded
            unless specified here.
        device (str):
            Device to run the model on, e.g., 'cpu' or 'cuda:0'.
        verbose (bool):
            Whether to output logging information.


    Examples:
        >>> # list of 2 image patches as input
        >>> data = [img1, img2]
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.run(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.run(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.run(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.run(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.run(wsi_file, mode='wsi')

    References:
        [1] Raza, Shan E. Ahmed, et al. "Deconvolving convolutional neural network
        for cell detection." 2019 IEEE 16th International Symposium on Biomedical
        Imaging (ISBI 2019). IEEE, 2019.

        [2] Sirinukunwattana, Korsuk, et al.
        "Locality sensitive deep learning for detection and classification
        of nuclei in routine colon cancer histology images."
        IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

    """

    from tiatoolbox.wsicore.wsireader import WSIReader

    def __init__(
        self: NucleusDetector,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def post_process_patches(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> list[pd.DataFrame]:
        """Define how to post-process patch predictions.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.

        Returns:
            A list of DataFrames containing the post-processed predictions for each patch.

        """
        _ = kwargs.get("return_probabilities")
        _ = prediction_shape
        _ = prediction_dtype

        batch_predictions = []
        for i in range(raw_predictions.shape[0]):
            batch_predictions.append(self.model.postproc_func(raw_predictions[i]))
        return batch_predictions

    def post_process_wsi(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> pd.DataFrame:
        """Define how to post-process WSI predictions.

        Returns:
            A DataFrame containing the post-processed predictions for the WSI.

        """
        logger.info("Post processing WSI predictions in NucleusDetector")

        logger.info(f"Raw probabilities shape: {prediction_shape}")
        logger.info(f"Raw probabilities dtype: {prediction_dtype}")
        logger.info(f"Chunk size: {raw_predictions.chunks}")

        detection_df = self.model.postproc(
            raw_predictions, prediction_shape, prediction_dtype
        )

        return detection_df

    def save_predictions(
        self: NucleusDetector,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | list[Path]:
        """Define how to save the processed predictions.

        Returns:
            A function that saves the processed predictions.

        """
        logger.info("Saving predictions in NucleusDetector")
        if output_type != "annotationstore":
            logger.warning(
                f"NucleusDetector only supports output_type='annotationstore'. "
                f"Overriding output_type='{output_type}' to 'annotationstore'."
            )
            output_type = "annotationstore"
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        class_dict = kwargs.get("class_dict")

        if self.patch_mode:
            save_paths = []
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                out_file = df_to_store_nucleus_detector(
                    predictions,
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
            return save_paths
        return df_to_store_nucleus_detector(
            processed_predictions["predictions"],
            scale_factor=scale_factor,
            save_path=save_path,
            class_dict=class_dict,
        )
