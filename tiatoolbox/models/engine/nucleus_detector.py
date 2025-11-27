"""This module implements nucleus detection engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
from dask.diagnostics.progress import ProgressBar
from shapely.geometry import Point

from tiatoolbox import logger
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Unpack

    import pandas as pd

    from tiatoolbox.annotation import AnnotationStore


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

    def post_process_patches(
        self: NucleusDetector,
        raw_predictions: list[da.Array],
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> list[np.ndarray]:
        """Define how to post-process patch predictions.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters

        Returns:
            A list of DataFrames containing the post-processed
            predictions for each patch.

        """
        _ = kwargs.get("return_probabilities")
        _ = prediction_shape
        _ = prediction_dtype

        return [
            self.model.postproc_func(raw_predictions[i])
            for i in range(len(raw_predictions))
        ]

    def post_process_wsi(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[SemanticSegmentorRunParams],  # noqa: ARG002
    ) -> da.Array:
        """Define how to post-process WSI predictions.

        Processes the raw prediction dask array using map_overlap
        to apply the model's post-processing function on each chunk
        with appropriate overlaps on chunk boundaries.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters

        Returns:
            Post-processed dask array of detections at the WSI level.
            The array has the same shape and dtype as the input.
            Each pixel indicates the presence of a detected nucleus
            as a probability score.

        """
        logger.info("Post processing WSI predictions in NucleusDetector")
        logger.info("Raw probabilities shape: %s", prediction_shape)
        logger.info("Raw probabilities dtype %s", prediction_dtype)
        logger.info("Raw chunk size: %s", raw_predictions.chunks)

        # Add halo (overlap) around each block for post-processing
        depth_h = self.model.min_distance
        depth_w = self.model.min_distance
        depth = {0: depth_h, 1: depth_w, 2: 0}

        # Re-chunk to post-processing tile shape for more efficient processing
        rechunked_prediction_map = raw_predictions.rechunk(
            (self.model.postproc_tile_shape[0], self.model.postproc_tile_shape[1], -1)
        )
        logger.info("Post-processing tile size: %s", rechunked_prediction_map.chunks)
        logger.info("Post-processing tiles overlap: (h=%d, w=%d)", depth_h, depth_w)

        return da.map_overlap(
            rechunked_prediction_map,
            self.model.postproc,
            depth=depth,
            boundary=0,
            dtype=prediction_dtype,
            block_info=True,
            depth_h=depth_h,
            depth_w=depth_w,
        )

    def save_predictions(
        self: NucleusDetector,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path:
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

        Returns:
            AnnotationStore | Path:
                - returns AnnotationStore or path to .db file.

        """
        # Only "annotationstore" output type is supported for NucleusDetector
        if output_type != "annotationstore":
            logger.warning(
                "Output type %s is not supported by NucleusDetector. "
                "Defaulting to 'annotationstore'.",
                output_type,
            )
            output_type = "annotationstore"

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")
        if class_dict is None:
            class_dict = self.model.output_class_dict

        # Need to add support for zarr conversion.
        save_paths = []

        logger.info("Saving predictions as AnnotationStore.")

        if self.patch_mode:
            save_paths = []
            for i, predictions in enumerate(processed_predictions["predictions"]):
                predictions_da = da.from_array(predictions, chunks=predictions.shape)

                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                out_file = self.write_centroid_maps_to_store(
                    predictions_da,
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
            return save_paths
        return self.write_centroid_maps_to_store(
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

        Keeps the highest-prob detection, removes any other point
        within 'radius' pixels > overlap_threshold.
        Expects dataframe columns: ['x','y','type','prob'].

        Args:
            df: pandas DataFrame of detections.
            radius: radius in pixels for suppression.
            overlap_threshold: float in [0,1], fraction of radius for suppression.

        Returns:
            filtered DataFrame with same columns/dtypes.
        """
        overlap_max = 1.0
        overlap_min = 0.0
        if df.empty:
            return df.copy()
        if radius <= 0:
            msg = "radius must be > 0"
            raise ValueError(msg)
        if not overlap_min < overlap_threshold <= overlap_max:
            msg = f"overlap_threshold must be in (0.0, 1.0], got {overlap_threshold}"
            raise ValueError(msg)

        # Sort by descending probability (highest priority first)
        sub = df.sort_values("prob", ascending=False).reset_index(drop=True)

        # Coordinates as float64 for distance math

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

            # Safe cosine argument = (distance ÷ diameter)
            # Clamp for numerical stability
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

        return sub.iloc[keep_idx].copy()

    @staticmethod
    def _centroid_maps_to_detection_records(
        block: np.ndarray, block_info: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert a block of centroid maps to detection records.

        Each block is a NumPy array of shape (h, w, C) containing detection
        probabilities of each class c. This function finds non-zero detections
        and returns their global coordinates, class IDs (channel), and probabilities.

        Args:
            block: NumPy array (h, w, C) for this chunk.
            block_info: Dask block info dict.

        Returns:
            Tuple of ([x_coords], [y_coords], [class_ids], [probs])
        """
        # block: (h, w, C) NumPy chunk (post-stitching, no halos)
        if block_info is not None:
            info = block_info[0]
            (r0, _), (c0, _), _ = info["array-location"]  # global interior start/stop
        else:
            r0, c0 = 0, 0

        # find the coordinates and channel indices of nonzeros
        ys, xs, cs = np.nonzero(block)

        if ys.size == 0:
            # return empty arrays
            return (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32),
            )

        x = xs.astype(np.uint32, copy=False) + int(c0)
        y = ys.astype(np.uint32, copy=False) + int(r0)
        t = cs.astype(np.uint32, copy=False)

        # read detection probabilities
        p = block[ys, xs, cs].astype(np.float32, copy=False)
        return (x, y, t, p)

    @staticmethod
    def _write_detection_records_to_store(
        recs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        store: SQLiteStore,
        scale_factor: tuple[float, float],
        class_dict: dict[int, str | int] | None,
        batch_size: int = 5000,
    ) -> int:
        """Write detection records to AnnotationStore in batches.

        Args:
            recs: Tuple of ([x_coords], [y_coords], [class_ids], [probs])
            store: SQLiteStore to write the detections to
            scale_factor: Scaling factors for x and y coordinates
            class_dict: Mapping from original class IDs to new class names
            batch_size: Number of records to write in each batch
        Returns:
            Total number of records written
        """
        x, y, t, p = recs
        n = len(x)
        if n == 0:
            return 0  # nothing to write

        # scale coordinates
        x = np.rint(x * scale_factor[0]).astype(np.uint32, copy=False)
        y = np.rint(y * scale_factor[1]).astype(np.uint32, copy=False)

        # class mapping
        if class_dict is None:
            # identity over actually-present types
            uniq = np.unique(t)
            class_dict = {int(k): int(k) for k in uniq}
        labels = np.array([class_dict.get(int(k), int(k)) for k in t], dtype=object)

        def make_points(xb: np.ndarray, yb: np.ndarray) -> list[Point]:
            """Create Shapely Point geometries from coordinate arrays."""
            return [Point(int(xx), int(yy)) for xx, yy in zip(xb, yb, strict=True)]

        written = 0
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            pts = make_points(x[i:j], y[i:j])

            anns = [
                Annotation(
                    geometry=pt, properties={"type": lbl, "probability": float(pp)}
                )
                for pt, lbl, pp in zip(pts, labels[i:j], p[i:j], strict=True)
            ]
            store.append_many(anns)
            written += j - i
        return written

    @staticmethod
    def write_centroid_maps_to_store(
        detection_maps: da.Array,
        scale_factor: tuple[float, float] = (1.0, 1.0),
        class_dict: dict | None = None,
        save_path: Path | None = None,
        batch_size: int = 5000,
    ) -> Path | SQLiteStore:
        """Write post-processed detection maps to an AnnotationStore.

        This is done in chunks using Dask for efficiency and to handle large
        detection maps at WSI level.

        Args:
            detection_maps: Dask array (H, W, C) of detection scores.
            scale_factor: Tuple (sx, sy) to scale coordinates before saving.
            class_dict: Optional dict mapping class indices to names.
            save_path: Optional Path to save the .db file.
                If None, returns in-memory store.
            batch_size: Number of records to write per batch.

        Returns:
            Path to saved .db file if save_path is provided, else in-memory SQLiteStore.
        """
        recs_delayed = (  # Convert each block to detection records
            detection_maps.map_blocks(
                NucleusDetector._centroid_maps_to_detection_records,
                dtype=object,  # we return Python tuples
                block_info=True,
            )
            .to_delayed()
            .ravel()
        )

        # create annotation store
        store = SQLiteStore()

        # one delayed writer per chunk (returns number of detections written)
        writes = [
            dask.delayed(NucleusDetector._write_detection_records_to_store)(
                recs, store, scale_factor, class_dict, batch_size
            )
            for recs in recs_delayed
        ]

        # IMPORTANT: SQLite is single-writer; run sequentially
        with ProgressBar():
            total = dask.compute(*writes, scheduler="single-threaded")
        logger.info("Total detections written to store: %s", sum(total))

        # if a save directory is provided, then dump store into a file
        if save_path:
            save_path.parent.absolute().mkdir(parents=True, exist_ok=True)
            save_path = save_path.parent.absolute() / (save_path.stem + ".db")
            store.commit()
            store.dump(save_path)
            return save_path

        return store
