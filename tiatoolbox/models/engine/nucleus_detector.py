"""This module implements nucleus detection engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from shapely.geometry import Point

from tiatoolbox import logger
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    SemanticSegmentorRunParams,
)
from tiatoolbox.utils.misc import get_tqdm

if TYPE_CHECKING:  # pragma: no cover
    import os
    from typing import Unpack

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


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

    def post_process_patches(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[NucleusDetectorRunParams],  # noqa: ARG002
    ) -> dict:
        """Define how to post-process patch predictions.

        Args:
            raw_predictions (da.Array):
                Predicted probabilities from the model (B, H, W, C),
                B is number of patches.
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters

        Returns:
            dict[str, list[da.Array]]:
                Detection arrays aggregated per patch. Each key ('x', 'y',
                'classes', 'probs') maps to a 1-D object dask array
                corresponds to a patch's detections.
                keys:
                    - "x" (list[dask array]): x coordinates (np.uint32).
                    - "y" (list[dask array]): y coordinates (np.uint32).
                    - "classes" (list[dask array]): detection classes (np.uint32).
                    - "probs" (list[dask array]): detection probabilities (np.float32).

        """
        logger.info("Post processing patch predictions in NucleusDetector")
        _ = prediction_shape
        _ = prediction_dtype

        probabilities = raw_predictions.persist()

        # Lists to hold per-patch detection arrays
        xs = []
        ys = []
        classes = []
        probs = []

        # Process each patch's predictions
        for i in range(probabilities.shape[0]):
            probs_prediction_patch = probabilities[i].compute()
            centroids_map_patch = self.model.postproc(probs_prediction_patch)
            centroids_map_patch = da.from_array(centroids_map_patch, chunks="auto")
            xs_patch, ys_patch, classes_patch, probs_patch = (
                self._centroid_maps_to_detection_arrays(centroids_map_patch).values()
            )
            xs.append(xs_patch)
            ys.append(ys_patch)
            classes.append(classes_patch)
            probs.append(probs_patch)

        return {"x": xs, "y": ys, "classes": classes, "probs": probs}

    def post_process_wsi(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict[str, da.Array]:
        """Define how to post-process WSI predictions.

        Processes the raw prediction dask array using map_overlap
        to apply the model's post-processing function on each chunk
        with appropriate overlaps on chunk boundaries.

        Args:
            raw_predictions (da.Array):
                Predicted probabilities from the model with shape (H, W, C).
            prediction_shape (tuple[int, ...]): The shape of the predictions.
            prediction_dtype (type): The data type of the predictions.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters

        Returns:
            dict[str, da.Array]:
            Each key ('x', 'y', 'classes', 'probs') maps to a 1-D dask array
            of all detections.
            - "x" (dask array): x coordinates (np.uint32).
            - "y" (dask array): y coordinates (np.uint32).
            - "classes" (dask array): detection classes (np.uint32).
            - "probs" (dask array): detection probabilities (np.float32).

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
                WSI mode:
                    - "predictions":
                    {
                        - 'x' (da.Array):
                            x coordinates (np.uint32).
                        - 'y' (da.Array):
                            y coordinates (np.uint32).
                        - 'types' (da.Array):
                            detection types (np.uint32).
                        - 'probs' (da.Array):
                            detection probabilities (np.float32).
                    }
                Patch mode:
                    - "predictions":
                    {
                        - "x" (list[dask array]):
                            x coordinates (np.uint32).
                        - "y" (list[dask array]):
                            y coordinates (np.uint32).
                        - "classes" (list[dask array]):
                            detection classes (np.uint32).
                        - "probs" (list[dask array]):
                            detection probabilities (np.float32).
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

        if output_type.lower() != "annotationstore":
            return super().save_predictions(
                processed_predictions["predictions"],
                output_type,
                save_path=save_path,
                **kwargs,
            )
        return self._save_predictions_annotation_store(
            processed_predictions,
            save_path=save_path,
            scale_factor=scale_factor,
            class_dict=class_dict,
        )

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
                    - 'classes': dask array of detection classes (np.uint32).
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
                    "classes": detections["classes"][i],
                    "probs": detections["probs"][i],
                }

                out_file = self.save_detection_arrays_to_store(
                    detection_arrays=detection_arrays,
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
            return save_paths
        predictions = processed_predictions["predictions"]
        return self.save_detection_arrays_to_store(
            detection_arrays=predictions,
            scale_factor=scale_factor,
            save_path=save_path,
            class_dict=class_dict,
        )

    @staticmethod
    def _centroid_maps_to_detection_arrays(
        detection_maps: da.Array,
    ) -> dict[str, da.Array]:
        """Convert centroid maps to detection records stored as dask arrays.

        Returns a dictionary with four 1-D dask arrays: x, y, types, probs.

        Args:
            detection_maps (da.Array):
                Dask array (H, W, C) of centroid maps,
                with detection probabilities at nuclei centroids,
                0 elsewhere.

        Returns:
            dict with keys:
                - "x": dask array of x coordinates (np.uint32).
                - "y": dask array of y coordinates (np.uint32).
                - "classes": dask array of detection classes (np.uint32).
                - "probs": dask array of detection probabilities (np.float32).

        """
        # Lists of da.Array parts from each block
        ys, xs, classes = da.nonzero(detection_maps)
        probs = detection_maps[detection_maps > 0]

        xs = xs.compute_chunk_sizes()
        ys = ys.compute_chunk_sizes()
        classes = classes.compute_chunk_sizes()
        probs = probs.compute_chunk_sizes()

        return {"x": xs, "y": ys, "classes": classes, "probs": probs}

    @staticmethod
    def _write_detection_arrays_to_store(
        detection_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        store: SQLiteStore,
        scale_factor: tuple[float, float],
        class_dict: dict[int, str | int] | None,
        batch_size: int = 5000,
    ) -> int:
        """Write detection arrays to AnnotationStore in batches.

        Args:
            detection_arrays (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
                Tuple of ([x_coords], [y_coords], [class_ids], [probs]).
            store (SQLiteStore):
                AnnotationStore to write the detections into.
            scale_factor (tuple[float, float]):
                Scaling factors for x and y coordinates.
            class_dict (dict[int, str | int] | None):
                Mapping from original class IDs to new class names.
            batch_size (int):
                Number of records to write in each batch,
                default is 5000.

        Returns:
            int:
                Total number of records written
        """
        xs, ys, classes, probs = detection_arrays
        n = len(xs)
        if n == 0:
            return 0  # nothing to write

        # scale coordinates
        xs = np.rint(xs * scale_factor[0]).astype(np.uint32, copy=False)
        ys = np.rint(ys * scale_factor[1]).astype(np.uint32, copy=False)

        # class mapping
        if class_dict is None:
            # identity over actually-present types
            uniq = np.unique(classes)
            class_dict = {int(k): int(k) for k in uniq}
        labels = np.array(
            [class_dict.get(int(k), int(k)) for k in classes], dtype=object
        )

        def make_points(xs_batch: np.ndarray, ys_batch: np.ndarray) -> list[Point]:
            """Create Shapely Point geometries from coordinate arrays in batches."""
            return [
                Point(int(xx), int(yy))
                for xx, yy in zip(xs_batch, ys_batch, strict=True)
            ]

        tqdm = get_tqdm()
        tqdm_loop = tqdm(range(0, n, batch_size), desc="Writing detections to store")
        written = 0
        for i in tqdm_loop:
            j = min(i + batch_size, n)
            pts = make_points(xs[i:j], ys[i:j])

            anns = [
                Annotation(
                    geometry=pt, properties={"class": lbl, "probability": float(pp)}
                )
                for pt, lbl, pp in zip(pts, labels[i:j], probs[i:j], strict=True)
            ]
            store.append_many(anns)
            written += j - i
        return written

    @staticmethod
    def save_detection_arrays_to_store(
        detection_arrays: dict[str, da.Array],
        scale_factor: tuple[float, float] = (1.0, 1.0),
        class_dict: dict | None = None,
        save_path: Path | None = None,
        batch_size: int = 5000,
    ) -> Path | SQLiteStore:
        """Write detection arrays to an SQLiteStore.

        Expects detection_arrays to contain dask arrays for keys
        ``x``, ``y``, ``classes``, and ``probs``.

        Args:
            detection_arrays (dict[str, da.Array]):
                - "x": dask array of x coordinates (np.uint32).
                - "y": dask array of y coordinates (np.uint32).
                - "classes": dask array of class ids (np.uint32).
                - "probs": dask array of detection probabilities (np.float32).
            scale_factor (tuple[float, float]):
                Scale factor to scale coordinates before saving.
            class_dict (dict | None):
                Optional dict mapping class indices to names.
            save_path (Path | None):
                Optional Path to save the .db file.
                If None, returns in-memory store.
            batch_size (int):
                Number of records to write per batch, default is 5000.

        Returns:
            Path to saved .db file if save_path is provided, else in-memory SQLiteStore.

        """
        xs = detection_arrays["x"]
        ys = detection_arrays["y"]
        classes = detection_arrays["classes"]
        probs = detection_arrays["probs"]

        xs = np.atleast_1d(np.asarray(xs))
        ys = np.atleast_1d(np.asarray(ys))
        classes = np.atleast_1d(np.asarray(classes))
        probs = np.atleast_1d(np.asarray(probs))

        if not (len(xs) == len(ys) == len(classes) == len(probs)):
            msg = "Detection record lengths are misaligned."
            raise ValueError(msg)

        store = SQLiteStore()
        total_written = NucleusDetector._write_detection_arrays_to_store(
            (xs, ys, classes, probs),
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
