"""This module implements nucleus detection engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
from shapely.geometry import Point
from skimage.feature import peak_local_max

from tiatoolbox import logger
from tiatoolbox.annotation import AnnotationStore
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
)
from tiatoolbox.models.models_abc import ModelABC

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
        exclude_border=False,
    )
    if coords.size:
        r, c = coords[:, 0], coords[:, 1]
        mask[r, c] = True
    return mask


# def peak_detection_mapoverlap(
#     block: np.ndarray,
#     block_info,
#     min_distance: int,
#     threshold_abs: float,
#     depth_h: int,
#     depth_w: int,
# ) -> np.ndarray:
#     """Runs inside Dask.da.map_overlap on a padded NumPy block: (h_pad, w_pad, C).
#     Builds a processed mask per channel, runs peak_local_max then
#     label+regionprops, and writes probability (mean_intensity) at centroid pixels.
#     Keeps only centroids whose (row,col) lie in the interior window:
#         rows [depth_h : depth_h + core_h), cols [depth_w : depth_w + core_w)
#     Returns same spatial shape as input block: (h_pad, w_pad, C), float32.

#     Args:
#         block: NumPy array (H, W, C) with padded block data.
#         block_info: Dask block info dict.
#         min_distance: Minimum distance in pixels between peaks.
#         threshold_abs: Minimum absolute threshold for peak detection.
#         depth_h: Halo size in pixels for height (rows).
#         depth_w: Halo size in pixels for width (cols).
#         calculate_probabilities: If True, write mean_intensity at centroids;
#             else write 1.0 at centroids.

#     Returns:
#         out: NumPy array (H, W, C) with probabilities at centroids, 0 elsewhere.
#     """
#     H, W, C = block.shape

#     # --- derive core (pre-overlap) size for THIS block safely ---
#     info = block_info[0]
#     locs = info["array-location"]  # [(r0,r1),(c0,c1),(ch0,ch1)]
#     core_h = int(locs[0][1] - locs[0][0])  # r1 - r0
#     core_w = int(locs[1][1] - locs[1][0])

#     rmin, rmax = depth_h, depth_h + core_h
#     cmin, cmax = depth_w, depth_w + core_w

#     out = np.zeros((H, W, C), dtype=np.float32)

#     for ch in range(C):
#         img = np.asarray(block[..., ch])  # NumPy 2D view
#         pmask = probability_to_peak_map(img, min_distance, threshold_abs)
#         if not pmask.any():
#             continue

#         lab = label(pmask)
#         props = regionprops(lab, intensity_image=img)

#         for reg in props:
#             r, c = reg.centroid  # floats in padded-block coords
#             if (rmin <= r < rmax) and (cmin <= c < cmax):
#                 rr = int(round(r))
#                 cc = int(round(c))
#                 if 0 <= rr < H and 0 <= cc < W:
#                     if calculate_probabilities:
#                         out[rr, cc, ch] = float(reg.mean_intensity)
#                     else:
#                         out[rr, cc, ch] = 1.0

#     return out


def _chunk_to_df(
    block: np.ndarray, block_info: dict, x_offset: int = 0, y_offset: int = 0
) -> pd.DataFrame:
    # block: np.ndarray (h, w, C) for this chunk (no halos here; use after stitching)
    info = block_info[0] if 0 in block_info else block_info[None]
    (r0, r1), (c0, c1), _ = info[
        "array-location"
    ]  # global interior coords for this chunk

    # find nonzeros per channel
    ys, xs, cs = np.nonzero(block)
    if ys.size == 0:
        DTYPES = {
            "x": "uint32",  # or "uint32" if you really want
            "y": "uint32",
            "type": "uint32",
            "prob": "float32",
        }
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in DTYPES.items()})

    probs = block[ys, xs, cs].astype(np.float32, copy=False)
    df = pd.DataFrame(
        {
            "x": xs + c0 + int(x_offset),
            "y": ys + r0 + int(y_offset),
            "type": cs.astype(np.int64, copy=False),
            "prob": probs,
        }
    )
    return df


def centroids_map_to_ddf_chunkwise(
    scores: da.Array, x_offset: int = 0, y_offset: int = 0
) -> dd.DataFrame:
    # build one delayed pandas DF per chunk
    dfs = (
        scores.map_blocks(
            _chunk_to_df,
            dtype=object,  # ignored; returning DataFrames
            block_info=True,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        .to_delayed()
        .ravel()
    )

    meta = pd.DataFrame(
        {
            "x": pd.Series([], dtype="uint32"),
            "y": pd.Series([], dtype="uint32"),
            "type": pd.Series([], dtype="uint32"),
            "prob": pd.Series([], dtype="float32"),
        }
    )
    ddf = dd.from_delayed(dfs, meta=meta)
    return ddf


def _chunk_to_records(
    block: np.ndarray, block_info
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # block: (h, w, C) NumPy chunk (post-stitching, no halos)
    info = block_info[0] if 0 in block_info else block_info[None]
    (r0, r1), (c0, c1), _ = info["array-location"]  # global interior start/stop

    ys, xs, cs = np.nonzero(block)
    if ys.size == 0:
        # return empty, dtype-stable arrays to avoid surprises
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.float32),
        )

    x = xs.astype(np.uint32, copy=False) + int(c0)
    y = ys.astype(np.uint32, copy=False) + int(r0)
    t = cs.astype(np.uint32, copy=False)
    p = block[ys, xs, cs].astype(np.float32, copy=False)
    return (x, y, t, p)


def _write_records_to_store(
    recs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    store: SQLiteStore,
    scale_factor: tuple[float, float],
    class_dict: dict | None,
    batch_size: int = 5000,
) -> int:
    x, y, t, p = recs
    n = len(x)
    if n == 0:
        return 0  # nothing to write

    x = np.rint(x * scale_factor[0]).astype(np.uint32, copy=False)
    y = np.rint(y * scale_factor[1]).astype(np.uint32, copy=False)

    # class mapping
    if class_dict is None:
        # identity over actually-present types
        uniq = np.unique(t)
        class_dict = {int(k): int(k) for k in uniq}
    labels = np.array([class_dict.get(int(k), int(k)) for k in t], dtype=object)

    def make_points(xb, yb):
        return [Point(int(xx), int(yy)) for xx, yy in zip(xb, yb)]

    written = 0
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        pts = make_points(x[i:j], y[i:j])

        anns = [
            Annotation(geometry=pt, properties={"type": lbl, "probability": float(pp)})
            for pt, lbl, pp in zip(pts, labels[i:j], p[i:j])
        ]
        store.append_many(anns)
        written += j - i
    return written


def write_centroids_to_store(
    scores: da.Array,
    scale_factor: tuple[float, float] = (1.0, 1.0),
    class_dict: dict | None = None,
    save_path: Path | None = None,
    batch_size: int = 5000,
) -> Path | SQLiteStore:
    # one delayed record-tuple per chunk
    recs_delayed = (
        scores.map_blocks(
            _chunk_to_records,
            dtype=object,  # we return Python tuples
            block_info=True,
        )
        .to_delayed()
        .ravel()
    )

    store = SQLiteStore()

    # one delayed writer per chunk (returns number written)
    writes = [
        dask.delayed(_write_records_to_store)(
            recs, store, scale_factor, class_dict, batch_size
        )
        for recs in recs_delayed
    ]

    # IMPORTANT: SQLite is single-writer; run sequentially
    with ProgressBar():
        total = dask.compute(*writes, scheduler="single-threaded")
    logger.info(f"Total detections written to store: {sum(total)}")
    # # if a save director is provided, then dump store into a file
    if save_path:
        # ensure parent directory exists
        save_path.parent.absolute().mkdir(parents=True, exist_ok=True)
        # ensure proper db extension
        save_path = save_path.parent.absolute() / (save_path.stem + ".db")
        store.commit()
        store.dump(save_path)
        return save_path

    return store


class NucleusDetector(SemanticSegmentor):
    r"""Nucleus detection engine.

    Args:
        model (str or nn.Module):
            Defined PyTorch model or name of the existing models support by
            tiatoolbox for processing the data e.g., mapde-conic, mapde-crchisto.
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

    Supported TIAToolBox Pre-trained Models:
        - `mapde-conic`
        - `mapde-crchisto`


    Examples:
        >>> model_name = "mapde-conic"
        >>> detector = NucleusDetector(model=model_name, batch_size=16, num_workers=8)
        >>> detector.run(
        ...     images=[pathlib.Path("example_wsi.tiff")],
        ...     patch_mode=False,
        ...     device="cuda",
        ...     save_dir=pathlib.Path("output_directory/"),
        ...     overwrite=True,
        ...     output_type="annotationstore",
        ...     class_dict={0: "nucleus"},
        ...     auto_get_mask=True,
        ...     memory_threshold=80
        ... )
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
    ) -> da.Array:
        """Define how to post-process WSI predictions.
        Processes the raw prediction dask array using map_overlap
        to apply the model's post-processing function on each chunk
        with appropriate overlaps on chunk boundaries.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.

        Returns:
            Post-processed dask array of detections at the WSI level.
            The array has the same shape and dtype as the input.
            Each pixel indicates the presence of a detected nucleus as a probability score.

        """
        logger.info("Post processing WSI predictions in NucleusDetector")
        logger.info(f"Raw probabilities shape: {prediction_shape}")
        logger.info(f"Raw probabilities dtype: {prediction_dtype}")
        logger.info(f"Raw chunk size: {raw_predictions.chunks}")

        # Add halo (overlap) around each block for post-processing
        depth_h = self.model.min_distance
        depth_w = self.model.min_distance
        depth = {0: depth_h, 1: depth_w, 2: 0}

        # Re-chunk to post-processing tile shape for more efficient processing
        rechunked_prediction_map = raw_predictions.rechunk(
            (self.model.postproc_tile_shape[0], self.model.postproc_tile_shape[1], -1)
        )
        logger.info(f"Post-processing chunk size: {rechunked_prediction_map.chunks}")

        detection_map = da.map_overlap(
            rechunked_prediction_map,
            self.model.postproc,
            depth=depth,
            boundary=0,
            dtype=prediction_dtype,
            block_info=True,
            depth_h=depth_h,
            depth_w=depth_w,
        )

        return detection_map

    def save_predictions(
        self: NucleusDetector,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore:
        """Save nucleus detections to disk or return them in memory.

        This method saves predictions in one of the supported formats:
        - "annotationstore": converts predictions to an AnnotationStore (.db file).

        If `patch_mode` is True, predictions are saved per image. If False,
        predictions are merged and saved as a single output.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
            output_type (str):
                "annotationstore".
            save_path (Path | None):
                Path to save the output file.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters including:
                - scale_factor (tuple[float, float]): For coordinate transformation.
                - class_dict (dict): Mapping of class indices to names.
                - return_probabilities (bool): Whether to save probability maps.

        Returns:
            dict | AnnotationStore | Path:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved Zarr file.
                - If output_type is "annotationstore": returns AnnotationStore
                  or path to .db file.

        """
        # Conversion to annotationstore uses a different function for SemanticSegmentor
        if output_type != "annotationstore":
            logger.warning(
                f"Output type '{output_type}' is not supported by NucleusDetector. "
                "Defaulting to 'annotationstore'."
            )
            output_type = "annotationstore"

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")

        # Need to add support for zarr conversion.
        save_paths = []

        logger.info("Saving predictions as AnnotationStore.")

        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        class_dict = kwargs.get("class_dict")

        if self.patch_mode:
            save_paths = []
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                out_file = write_centroids_to_store(
                    predictions,
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
            return save_paths
        return write_centroids_to_store(
            processed_predictions["predictions"],
            scale_factor=scale_factor,
            save_path=save_path,
            class_dict=class_dict,
        )

    @staticmethod
    def nucleus_detection_nms(
        df: pd.DataFrame, radius: int, overlap_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Non-Maximum Suppression across ALL detections.

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
