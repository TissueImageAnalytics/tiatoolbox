"""This module implements nucleus detection engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import zarr
from dask import compute
from dask.diagnostics.progress import ProgressBar
from shapely.geometry import Point

from tiatoolbox import logger
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
)
from tiatoolbox.wsicore.wsireader import is_zarr

if TYPE_CHECKING:  # pragma: no cover
    import os
    from typing import Unpack

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


def _flatten_predictions_to_dask(
    arr: da.Array | list[np.ndarray] | np.ndarray,
) -> da.Array:
    """Normalise predictions to a flat 1D Dask array."""
    # # Case 1: already a Dask array
    if isinstance(arr, da.Array):
        # If it's already a numeric Dask array, just return it
        if arr.dtype != object:
            return arr
        arr = arr.compute()

    arr_list = list(arr)
    dask_parts = [
        a if isinstance(a, da.Array) else da.from_array(a, chunks="auto")
        for a in arr_list
    ]
    return da.concatenate(dask_parts, axis=0)


class NucleusDetectorRunParams(SemanticSegmentorRunParams, total=False):
    """Runtime parameters for configuring the `NucleusDetector.run()` method.

    This class extends `SemanticSegmentorRunParams`,
    and adds parameters specific to nucleus detection workflows.

    Attributes:
        auto_get_mask (bool):
            Whether to automatically generate segmentation masks using
            `wsireader.tissue_mask()` during processing.
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        class_dict (dict):
            Optional dictionary mapping classification outputs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda").
        labels (list):
            Optional labels for input images. Only a single label per image
            is supported.
        memory_threshold (int):
            Memory usage threshold (in percentage) to trigger caching behavior.
        num_workers (int):
            Number of workers used in DataLoader.
        output_file (str):
            Output file name for saving results (e.g., .zarr or .db).
        output_resolutions (Resolution):
            Resolution used for writing output predictions.
        patch_output_shape (tuple[int, int]):
            Shape of output patches (height, width).
        min_distance (int):
            Minimum distance separating two nuclei (in pixels).
        postproc_tile_shape (tuple[int, int]):
            Tile shape (height, width) for post-processing (in pixels).
        return_labels (bool):
            Whether to return labels with predictions.
        return_probabilities (bool):
            Whether to return per-class probabilities.
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.

    """

    min_distance: int
    postproc_tile_shape: IntPair


class NucleusDetector(SemanticSegmentor):
    r"""Nucleus detection engine for digital pathology images.

    This class extends SemanticSegmentor to support nucleus detection tasks
    using pretrained or custom models from TIAToolbox. It supports both patch-level
    and whole slide image (WSI) processing, and provides utilities for merging,
    post-processing, and saving predictions.

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
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict[str, da.Array]:
        """Define how to post-process patch predictions.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters

        Returns:
            dict[str, dask.array.Array]:
                Detection arrays aggregated per patch. Each key ('x', 'y',
                'types', 'probs') maps to a 1-D object dask array where each
                element corresponds to a patch's detections.
                keys:
                - "x": dask array of x coordinates (np.uint32).
                - "y": dask array of y coordinates (np.uint32).
                - "types": dask array of detection types (np.uint32).
                - "probs": dask array of detection probabilities (np.float32).

        """
        logger.info("Post processing patch predictions in NucleusDetector")
        _ = kwargs.get("return_probabilities")
        _ = prediction_shape
        _ = prediction_dtype

        detection_arrays = []
        for i in range(raw_predictions.shape[0]):
            patch_pred = raw_predictions[i]
            postproc_map = da.from_array(
                self.model.postproc(patch_pred), chunks=patch_pred.chunks
            )
            detection_arrays.append(
                self._centroid_maps_to_detection_arrays(postproc_map)
            )

        def to_object_da(arrs: list[da.Array]) -> da.Array:
            """Wrap list of variable-length arrays into object-dtype dask array."""
            # np.array(arrs, dtype=object) can collapse to 2-D (e.g., shape (n, 0))
            # when all elements are empty, which breaks dask chunk validation.
            # Build the object array explicitly to keep it 1-D.
            obj_array = np.empty(len(arrs), dtype=object)
            for idx, arr in enumerate(arrs):
                obj_array[idx] = arr
            return da.from_array(obj_array, chunks=(len(arrs),))

        return {
            "x": to_object_da([det["x"] for det in detection_arrays]),
            "y": to_object_da([det["y"] for det in detection_arrays]),
            "types": to_object_da([det["types"] for det in detection_arrays]),
            "probs": to_object_da([det["probs"] for det in detection_arrays]),
        }

    def post_process_wsi(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> da.Array:
        """Define how to post-process WSI predictions.

        Processes the raw prediction dask array using map_overlap
        to apply the model's post-processing function on each chunk
        with appropriate overlaps on chunk boundaries.

        Args:
            raw_predictions (da.Array): The raw predictions from the model.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters

        Returns:
            dict[str, da.Array]:
                Dictionary of detection records with keys:
                - "x": dask array of x coordinates.
                - "y": dask array of y coordinates.
                - "types": dask array of detection types.
                - "probs": dask array of detection probabilities.

        """
        _ = prediction_shape

        logger.info("Post processing WSI predictions in NucleusDetector")

        min_distance = kwargs.get("min_distance", (self.model.min_distance))
        postproc_tile_shape = kwargs.get(
            "postproc_tile_shape",
            self.model.postproc_tile_shape,
        )

        # Add halo (overlap) around each block for post-processing
        depth_h = min_distance
        depth_w = min_distance
        depth = {0: depth_h, 1: depth_w, 2: 0}

        # Re-chunk to post-processing tile shape for more efficient processing
        rechunked_prediction_map = raw_predictions.rechunk(
            (postproc_tile_shape[0], postproc_tile_shape[1], -1)
        )
        logger.info("Post-processing tile size: %s", rechunked_prediction_map.chunks)
        logger.info("Post-processing tiles overlap: (h=%d, w=%d)", depth_h, depth_w)

        centroid_maps = da.map_overlap(
            rechunked_prediction_map,
            self.model.postproc,
            depth=depth,
            boundary=0,
            dtype=prediction_dtype,
            block_info=True,
            depth_h=depth_h,
            depth_w=depth_w,
        )

        return self._centroid_maps_to_detection_arrays(centroid_maps)

    def save_predictions(
        self: NucleusDetector,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Save nucleus detections to disk or return them in memory.

        This method saves predictions in one of the supported formats:
        - "annotationstore": converts predictions to an AnnotationStore (.db file).

        If `patch_mode` is True, predictions are saved per image. If False,
        predictions are merged and saved as a single output.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
                keys:
                - "predictions":
                {
                    - 'x': dask array of x coordinates (np.uint32).
                    - 'y': dask array of y coordinates (np.uint32).
                    - 'types': dask array of detection types (np.uint32).
                    - 'probs': dask array of detection probabilities (np.float32).
                }
            output_type (str):
                Desired output format.
                Supported values are "dict", "zarr", and "annotationstore".
            save_path (Path | None):
                Path to save the output file.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters including:
                - scale_factor (tuple[float, float]): For coordinate transformation.
                - class_dict (dict): Mapping of class indices to names.

        Returns:
            AnnotationStore | Path:
                - returns AnnotationStore or path to .db file.

        """
        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")
        if class_dict is None:
            class_dict = self.model.output_class_dict

        if output_type.lower() == "dict":
            return super().save_predictions(
                processed_predictions,
                output_type,
                save_path=save_path,
                **kwargs,
            )
        if output_type.lower() == "annotationstore":
            return self._save_predictions_annotation_store(
                processed_predictions,
                save_path=save_path,
                scale_factor=scale_factor,
                class_dict=class_dict,
            )
        return self._save_predictions_zarr(
            processed_predictions,
            save_path=save_path,
        )

    def _save_predictions_zarr(
        self: NucleusDetector,
        processed_predictions: dict,
        save_path: Path | None = None,
    ) -> Path | list[Path]:
        """Save predictions to a Zarr store.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
                keys:
                - "predictions":
                {
                    - 'x': dask array of x coordinates (np.uint32).
                    - 'y': dask array of y coordinates (np.uint32).
                    - 'types': dask array of detection types (np.uint32).
                    - 'probs': dask array of detection probabilities (np.float32).
                }
            save_path (Path | None):
                Path to save the output Zarr store.

        Returns:
            Path | list[Path]:
                Path to the saved Zarr store(s).

        """
        predictions = processed_predictions["predictions"]

        keys_to_compute = [k for k in predictions if k not in self.drop_keys]

        # If appending to an existing Zarr, skip keys that are already present
        if is_zarr(save_path):
            zarr_group = zarr.open(save_path, mode="r")
            keys_to_compute = [k for k in keys_to_compute if k not in zarr_group]

        write_tasks = []

        # --- NEW: compute patch_offsets from 'x' if we are in patch mode ----
        patch_offsets = None
        if self.patch_mode and "x" in predictions:
            x_arr_list = predictions["x"].compute()

            # lengths[i] = number of detections in patch i
            lengths = np.array([len(a) for a in x_arr_list], dtype=np.int64)
            patch_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
            patch_offsets[0] = 0
            np.cumsum(lengths, out=patch_offsets[1:])

            # Save patch_offsets as its own 1D dataset
            offsets_da = da.from_array(patch_offsets, chunks="auto")
            write_tasks.append(
                offsets_da.to_zarr(
                    url=save_path,
                    component="patch_offsets",
                    compute=False,
                )
            )

        # ---------------- save flattened predictions -----------------
        for key in keys_to_compute:
            raw = predictions[key]

            dask_array = _flatten_predictions_to_dask(raw)
            # Type casting for storage
            if key != "probs":
                dask_array = dask_array.astype(np.uint32)
            else:
                dask_array = dask_array.astype(np.float32)

            # Normalise ragged per-patch predictions to a flat 1D Dask array

            task = dask_array.to_zarr(
                url=save_path,
                component=key,
                compute=False,
            )
            write_tasks.append(task)

        msg = f"Saving output to {save_path}."
        logger.info(msg=msg)
        with ProgressBar():
            compute(*write_tasks)

        zarr_group = zarr.open(save_path, mode="r+")
        for key in self.drop_keys:
            if key in zarr_group:
                del zarr_group[key]

        return save_path

    def _save_predictions_annotation_store(
        self: NucleusDetector,
        processed_predictions: dict,
        save_path: Path | None = None,
        scale_factor: tuple[float, float] = (1.0, 1.0),
        class_dict: dict | None = None,
    ) -> AnnotationStore | Path | list[Path]:
        """Save predictions to an AnnotationStore.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
                keys:
                - "predictions":
                {
                    - 'x': dask array of x coordinates (np.uint32).
                    - 'y': dask array of y coordinates (np.uint32).
                    - 'types': dask array of detection types (np.uint32).
                    - 'probs': dask array of detection probabilities (np.float32).
                }
            save_path (Path | None):
                Path to save the output file.
            scale_factor (tuple[float, float]):
                Scaling factors for x and y coordinates.
            class_dict (dict | None):
                Mapping from original class IDs to new class names.

        Returns:
            AnnotationStore | Path:
                - returns AnnotationStore or path to .db file.

        """
        logger.info("Saving predictions as AnnotationStore.")
        if self.patch_mode:
            save_paths = []
            detections = processed_predictions["predictions"]

            num_patches = len(detections["x"])
            for i in range(num_patches):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                detection_arrays = {
                    "x": detections["x"][i],
                    "y": detections["y"][i],
                    "types": detections["types"][i],
                    "probs": detections["probs"][i],
                }

                out_file = self.write_detection_arrays_to_store(
                    detection_arrays=detection_arrays,
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
            return save_paths
        predictions = processed_predictions["predictions"]
        return self.write_detection_arrays_to_store(
            detection_arrays=predictions,
            scale_factor=scale_factor,
            save_path=save_path,
            class_dict=class_dict,
        )

    @staticmethod
    def _extract_detection_arrays_from_block(
        block: np.ndarray, block_info: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract detection arrays from a block of centroid maps.

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
            info = block_info[0]  # block_info[0] is input block info
            (global_y_start, _), (global_x_start, _), _ = info[
                "array-location"
            ]  # global interior start/stop
        else:
            global_y_start, global_x_start = 0, 0

        # find the coordinates and channel indices of nonzeros
        ys_local, xs_local, classes = np.nonzero(block)

        if ys_local.size == 0:
            # return empty arrays
            return (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32),
            )

        xs_global = xs_local.astype(np.uint32, copy=False) + int(global_x_start)
        ys_global = ys_local.astype(np.uint32, copy=False) + int(global_y_start)
        classes = classes.astype(np.uint32, copy=False)

        # read detection probabilities
        probs = block[ys_local, xs_local, classes].astype(np.float32, copy=False)
        return xs_global, ys_global, classes, probs

    @staticmethod
    def _centroid_maps_to_detection_arrays(
        detection_maps: da.Array,
    ) -> dict[str, da.Array]:
        """Convert centroid maps to detection records stored as dask arrays.

        Returns a dictionary with four 1-D dask arrays: x, y, types, probs.

        Args:
            detection_maps: Dask array (H, W, C) of centroid maps.

        Returns:
            dict with keys:
                - "x": dask array of x coordinates (np.uint32).
                - "y": dask array of y coordinates (np.uint32).
                - "types": dask array of detection types (np.uint32).
                - "probs": dask array of detection probabilities (np.float32).

        """
        recs_delayed = (
            detection_maps.map_blocks(
                NucleusDetector._extract_detection_arrays_from_block,
                dtype=object,
                block_info=True,
            )
            .to_delayed()
            .ravel()
        )

        def make_parts(index: int, dtype: np.dtype) -> list[da.Array]:
            """Extract one element from each delayed record tuple."""
            return [
                da.from_delayed(
                    dask.delayed(lambda rec, idx=index: rec[idx])(rec_tuple),
                    shape=(np.nan,),
                    dtype=dtype,
                )
                for rec_tuple in recs_delayed
            ]

        def concat_parts(parts: list[da.Array]) -> da.Array:
            """Concatenate parts while handling empty inputs."""
            return da.concatenate(parts)

        x_parts = make_parts(0, np.uint32)
        y_parts = make_parts(1, np.uint32)
        type_parts = make_parts(2, np.uint32)
        prob_parts = make_parts(3, np.float32)

        x_da = concat_parts(x_parts)
        y_da = concat_parts(y_parts)
        types_da = concat_parts(type_parts)
        probs_da = concat_parts(prob_parts)

        # Compute once to avoid nested delayed graphs downstream.
        with ProgressBar():
            x_np, y_np, types_np, probs_np = dask.compute(
                x_da, y_da, types_da, probs_da
            )

        def wrap(arr: np.ndarray) -> da.Array:
            """Wrap computed numpy arrays back to single-chunk dask arrays."""
            if arr.size == 0:
                return da.from_array(arr, chunks=(0,))
            return da.from_array(arr, chunks="auto")

        return {
            "x": wrap(x_np),
            "y": wrap(y_np),
            "types": wrap(types_np),
            "probs": wrap(probs_np),
        }

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
    def write_detection_arrays_to_store(
        detection_arrays: dict[str, da.Array],
        scale_factor: tuple[float, float] = (1.0, 1.0),
        class_dict: dict | None = None,
        save_path: Path | None = None,
        batch_size: int = 5000,
    ) -> Path | SQLiteStore:
        """Write detection arrays to an SQLiteStore.

        Expects detection_arrays to contain dask arrays for keys
        ``x``, ``y``, ``types``, and ``probs``.

        Args:
            detection_arrays: dict with keys:
                - "x": dask array of x coordinates (np.uint32).
                - "y": dask array of y coordinates (np.uint32).
                - "types": dask array of detection types (np.uint32).
                - "probs": dask array of detection probabilities (np.float32).
            scale_factor: Tuple (sx, sy) to scale coordinates before saving.
            class_dict: Optional dict mapping class indices to names.
            save_path: Optional Path to save the .db file.
                If None, returns in-memory store.
            batch_size: Number of records to write per batch.

        Returns:
            Path to saved .db file if save_path is provided, else in-memory SQLiteStore.

        """
        xs = detection_arrays["x"]
        ys = detection_arrays["y"]
        types = detection_arrays["types"]
        probs = detection_arrays["probs"]

        xs = np.atleast_1d(np.asarray(xs))
        ys = np.atleast_1d(np.asarray(ys))
        types = np.atleast_1d(np.asarray(types))
        probs = np.atleast_1d(np.asarray(probs))

        if not (len(xs) == len(ys) == len(types) == len(probs)):
            msg = "Detection record lengths are misaligned."
            raise ValueError(msg)

        store = SQLiteStore()
        total_written = NucleusDetector._write_detection_records_to_store(
            (xs, ys, types, probs),
            store,
            scale_factor,
            class_dict,
            batch_size,
        )
        logger.info("Total detections written to store: %s", total_written)

        if save_path:
            save_path.parent.absolute().mkdir(parents=True, exist_ok=True)
            save_path = save_path.parent.absolute() / (save_path.stem + ".db")
            store.commit()
            store.dump(save_path)
            return save_path

        return store

    def run(
        self: NucleusDetector,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        *,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the nucleus detection engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both
        patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
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
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    auto_get_mask (bool):
                        Whether to automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches to feed to the model in a forward pass.
                    class_dict (dict):
                        Optional dictionary mapping classification outputs to
                        class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (in percentage) to
                        trigger caching behavior.
                    num_workers (int):
                        Number of workers used in DataLoader.
                    output_file (str):
                        Output file name for saving results (e.g., .zarr or .db).
                    output_resolutions (Resolution):
                        Resolution used for writing output predictions.
                    patch_output_shape (tuple[int, int]):
                        Shape of output patches (height, width).
                    min_distance (int):
                        Minimum distance separating two nuclei (in pixels).
                    postproc_tile_shape (tuple[int, int]):
                        Tile shape (height, width) for post-processing (in pixels).
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities.
                    scale_factor (tuple[float, float]):
                        Scale factor for converting annotations to baseline resolution.
                        Typically model_mpp / slide_mpp.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to patch_input_shape.
                    verbose (bool):
                        Whether to output logging information.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> nuc_detector = NucleusDetector(model="sccnn-conic")
            >>> output = nuc_detector.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = nuc_detector.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = nuc_detector.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... "/path/to/wsi1.db"

        """
        return super().run(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )
