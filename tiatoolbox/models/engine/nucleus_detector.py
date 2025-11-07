"""This module implements nucleus detection engine."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed
import dask 
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

from tiatoolbox.models.engine.engine_abc import EngineABCRunParams
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams
)
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.models.models_abc import ModelABC
from shapely.geometry import Point
from typing import TYPE_CHECKING, Unpack

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader

def dataframe_to_annotation_store(
    df: pd.DataFrame,
) -> AnnotationStore:
    """
    Convert a pandas DataFrame with columns ['x','y','type','prob']
    to an AnnotationStore and save to disk.
    """
    from tiatoolbox.annotation import SQLiteStore, Annotation

    ann_store = SQLiteStore()
    for _, row in df.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        obj_type = int(row["type"])
        prob = float(row["prob"])
        ann = Annotation(geometry=Point(x, y), properties={"type": "nuclei", "probability": prob})
        ann_store.append(ann)
    return ann_store


def processed_mask_fn(img2d:np.ndarray, min_distance: int, threshold_abs: float|int) -> np.ndarray:
    """
    Build a boolean mask (H, W) of objects from a 2D probability map.
    Here: 1-pixel objects from peak_local_max. Add morphology inside if you need blobs.
    """
    H, W = img2d.shape
    mask = np.zeros((H, W), dtype=bool)
    coords = peak_local_max(img2d, min_distance=min_distance, threshold_abs=threshold_abs)
    if coords.size:
        r, c = coords[:, 0], coords[:, 1]
        mask[r, c] = True
    return mask

def block_regionprops_mapoverlap(
    block: np.ndarray,
    block_info,
    min_distance: int,
    threshold_abs: float | int,
    depth_h: int,
    depth_w: int,
) -> np.ndarray:
    """
    Runs inside da.map_overlap on a padded NumPy block: (h_pad, w_pad, C).
    Builds a processed mask per channel, runs label+regionprops, and writes
    region score (mean_intensity) at centroid pixels. Keeps only centroids
    whose (row,col) lie in the interior window:
        rows [depth_h : depth_h + core_h), cols [depth_w : depth_w + core_w)
    Returns same spatial shape as input block: (h_pad, w_pad, C), float32.
    """
    H, W, C = block.shape

    # --- derive core (pre-overlap) size for THIS block safely ---
    info = block_info[0]
    locs = info["array-location"]           # [(r0,r1),(c0,c1),(ch0,ch1)]
    core_h = int(locs[0][1] - locs[0][0])   # r1 - r0
    core_w = int(locs[1][1] - locs[1][0])


    rmin, rmax = depth_h, depth_h + core_h
    cmin, cmax = depth_w, depth_w + core_w

    out = np.zeros((H, W, C), dtype=np.float32)

    for ch in range(C):
        img = np.asarray(block[..., ch])  # NumPy 2D view
        pmask = processed_mask_fn(img, min_distance, threshold_abs)
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
                    out[rr, cc, ch] = float(reg.mean_intensity)

    return out


def detect_with_map_overlap(probs, min_distance, threshold_abs, depth_pixels):
    """
    probs: Dask array (H, W, C), float.
    depth_pixels: halo in pixels for H/W (use >= min_distance and >= any morphology radius).
    Returns:
      scores: da.Array (H, W, C) with mean_intensity at centroids, 0 elsewhere.
    """
    depth = {0: depth_pixels, 1: depth_pixels, 2: 0}
    scores = da.map_overlap(
        probs,
        block_regionprops_mapoverlap,
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

def scores_to_ddf(scores: da.Array, x_offset: int, y_offset: int) -> dd.DataFrame:
    """
    Convert (H, W, C) scores -> Dask DataFrame with columns: x, y, type, prob.
    Uses da.extract(mask, scores) to avoid vindex on Dask indexers.
    """
    # 1) Build a boolean mask of detections
    mask = scores > 0

    # 2) Global coordinates of detections (lazy 1D Dask arrays)
    yy, xx, cc = da.nonzero(mask)

    # 3) Values at those detections (lazy) — same length as yy/xx/cc
    ss = da.extract(mask, scores)

    # 4) Assemble a Dask DataFrame
    ddf = dd.concat(
        [
            dd.from_dask_array(xx.astype("int64"),   columns="x"),
            dd.from_dask_array(yy.astype("int64"),   columns="y"),
            dd.from_dask_array(cc.astype("int64"),   columns="type"),
            dd.from_dask_array(ss.astype("float32"), columns="prob"),
        ],
        axis=1,
    )

    # 5) Apply global offsets (if your WSI/crop needs them)
    ddf["x"] = ddf["x"] + int(x_offset)
    ddf["y"] = ddf["y"] + int(y_offset)

    return ddf


def greedy_radius_nms_pandas_all(df: pd.DataFrame, radius: int) -> pd.DataFrame:
    """
    Greedy NMS across ALL detections (no per-type grouping).
    Keeps the highest-prob point, suppresses any other point within 'radius' pixels.

    Expects columns: ['x','y','type','prob'].
    Returns: filtered DataFrame with same columns/dtypes.
    """
    if df.empty:
        return df.copy()

    # Sort by descending probability (highest priority first)
    sub = df.sort_values("prob", ascending=False).reset_index(drop=True)

    # Coordinates as float64 for distance math
    coords = sub[["x", "y"]].to_numpy(dtype=np.float64)
    r2 = float(radius) * float(radius)

    suppressed = np.zeros(len(sub), dtype=bool)
    keep_idx = []

    for i in range(len(sub)):
        if suppressed[i]:
            continue
        keep_idx.append(i)

        # Suppress all remaining within radius of the kept point
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        close = (dx * dx + dy * dy) <= r2
        suppressed |= close

    kept = sub.iloc[keep_idx].copy()

    # Ensure stable dtypes
    kept["x"] = kept["x"].astype("int64")
    kept["y"] = kept["y"].astype("int64")
    kept["type"] = kept["type"].astype("int64")
    kept["prob"] = kept["prob"].astype(df["prob"].dtype)

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

    def post_process_patches(self,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],  # noqa: ARG002
        prediction_dtype: type,  # noqa: ARG002
        **kwargs: Unpack[SemanticSegmentorRunParams],  # noqa: ARG002
    ) -> da.Array:
        """Define how to post-process patch predictions.

        Returns:
            A function that process the raw model predictions on patches.

        """

        pass

    def post_process_wsi(self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> da.Array:
        """Define how to post-process WSI predictions.

        Returns:
            A function that process the raw model predictions on WSI.

        """
        print("Post processing WSI predictions in NucleusDetector")

        print("Raw probabilities shape:", raw_predictions.shape)

        print("Chunk size:", raw_predictions.chunks)

        

        scores = detect_with_map_overlap(
            probs=raw_predictions,
            min_distance=3,
            threshold_abs=205,   # set your threshold
            depth_pixels=5
        )
        print("Scores shape:", scores.shape)

        # compact table:
        ddf = scores_to_ddf(scores, x_offset=0, y_offset=0)
        pandas_df = ddf.compute()

        print("Total detections before NMS:", len(pandas_df))
        nms_df = greedy_radius_nms_pandas_all(pandas_df, radius=3)
        print("Total detections after NMS:", len(nms_df))

        save_path = "/media/u1910100/data/overlays/test/mapde_conic.db"
        ann_store = dataframe_to_annotation_store(nms_df)
        ann_store.dump(save_path)


        sys.exit()
        


    def run(
        self: NucleusDetector,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the semantic segmentation engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both
        patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            labels (list | None):
                Optional labels for input images. Only one label per image is supported.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`). Default
                is True.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore". Default
                is "dict".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to update engine attributes.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = segmentor.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... "/path/to/wsi1.db"

        """
        return super().run(
            images=images,
            masks=masks,
            labels=labels,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )

 