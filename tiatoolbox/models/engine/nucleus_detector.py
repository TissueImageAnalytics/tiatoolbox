"""Nucleus Detection Engine for Digital Pathology (WSIs and patches).

This module implements the `NucleusDetector` class which extends
`SemanticSegmentor` to perform instance-level nucleus detection on
histology images. It supports patch-mode and whole slide image (WSI)
workflows using TIAToolbox or custom PyTorch models, and provides
utilities for parallel post-processing (centroid extraction, thresholding),
merging detections across patches, and exporting results in multiple
formats (in-memory dict, Zarr, AnnotationStore).

Classes
-------
NucleusDetectorRunParams
    TypedDict specifying runtime configuration keys for detection.
NucleusDetector
    Core engine for nucleus detection on image patches or WSIs.

Examples:
--------
>>> from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
>>> detector = NucleusDetector(model="mapde-conic")
>>> # WSI workflow: save to AnnotationStore (.db)
>>> out = detector.run(
...     images=[pathlib.Path("example_wsi.tiff")],
...     patch_mode=False,
...     device="cuda",
...     save_dir=pathlib.Path("output_directory/"),
...     overwrite=True,
...     output_type="annotationstore",
...     class_dict={0: "nucleus"},
...     auto_get_mask=True,
...     memory_threshold=80,
... )
>>> # Patch workflow: return in-memory detections
>>> patches = [np.ndarray, np.ndarray]  # NHWC
>>> out = detector.run(patches, patch_mode=True, output_type="dict")

Notes:
-----
- Outputs can be returned as Python dictionaries, saved as Zarr groups,
 or converted to AnnotationStore (.db).
- Post-processing uses tile rechunking and halo padding to facilitate
  centroid extraction near chunk boundaries.

"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from dask import compute
from dask.diagnostics.progress import ProgressBar
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
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


class NucleusDetectorRunParams(SemanticSegmentorRunParams, total=False):
    """Runtime parameters for configuring the `NucleusDetector.run()` method.

    This class extends `SemanticSegmentorRunParams` (and transitively
    `PredictorRunParams` → `EngineABCRunParams`) with additional options
    specific to nucleus detection workflows.

    Attributes:
        auto_get_mask (bool):
            Whether to automatically generate segmentation masks using
            `wsireader.tissue_mask()` during WSI processing.
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        class_dict (dict):
            Optional dictionary mapping numeric class IDs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda").
        labels (list):
            Optional labels for input images. Only a single label per image
            is supported.
        memory_threshold (int):
            Memory usage threshold (in percentage) to trigger caching behavior.
        num_workers (int):
            Number of workers used in the DataLoader.
        output_file (str):
            Output file name for saving results (e.g., ".zarr" or ".db").
        output_resolutions (Resolution):
            Resolution used for writing output predictions/coordinates.
        patch_output_shape (tuple[int, int]):
            Shape of output patches (height, width).
        min_distance (int):
            Minimum separation between nuclei (in pixels) used during
            centroid extraction/post-processing.
        threshold_abs (float):
            Absolute detection threshold applied to model outputs.
        threshold_rel (float):
            Relative detection threshold (e.g., with respect to local maxima).
        postproc_tile_shape (tuple[int, int]):
            Tile shape (height, width) used during post-processing
            (in pixels) to control rechunking behavior.
        return_labels (bool):
            Whether to return labels with predictions.
        return_probabilities (bool):
            Whether to include per-class probabilities in the output.
        scale_factor (tuple[float, float]):
            Scale factor for converting coordinates to baseline resolution.
            Typically, `model_mpp / slide_mpp`.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to `patch_input_shape`.
        verbose (bool):
            Whether to enable verbose logging.

    """

    min_distance: int
    threshold_abs: float
    threshold_rel: float
    postproc_tile_shape: IntPair


class NucleusDetector(SemanticSegmentor):
    r"""Nucleus detection engine for digital histology images.

    This class extends :class:`SemanticSegmentor` to support instance-level
    nucleus detection using pretrained or custom models from TIAToolbox.
    It operates in both patch-level and whole slide image (WSI) modes and
    provides utilities for post-processing (e.g., centroid extraction,
    thresholding, tile-overlap handling), merging predictions, and saving
    results in multiple output formats. Supported TIAToolbox models include
    nucleus-detection architectures such as ``mapde-conic`` and
    ``mapde-crchisto``. For the full list of pretrained models, refer to the
    model zoo documentation:
    https://tia-toolbox.readthedocs.io/en/latest/pretrained.html

    The class integrates seamlessly with the TIAToolbox engine interface,
    inheriting the data loading, inference orchestration, memory-aware
    chunking, and output-saving conventions of :class:`SemanticSegmentor`,
    while overriding only the nucleus-specific post-processing and export
    routines.

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
            Number of image patches processed per forward pass.
            Default is ``8``.
        num_workers (int):
            Number of workers for ``torch.utils.data.DataLoader``.
            Default is ``0``.
        weights (str or pathlib.Path or None):
            Optional path to pretrained weights. If ``None`` and ``model`` is
            a string, default pretrained weights for that model will be used.
            If ``model`` is an ``nn.Module``, weights are loaded only if
            provided.
        device (str):
            Device on which the model will run (e.g., ``"cpu"``, ``"cuda"``).
            Default is ``"cpu"``.
        verbose (bool):
            Whether to output logging information. Default is ``True``.

    Attributes:
        images (list[str or Path] or np.ndarray):
            Input images supplied to the engine, either as WSI paths or
            NHWC-formatted patches.
        masks (list[str or Path] or np.ndarray):
            Optional tissue masks for WSI processing. Only used when
            ``patch_mode=False``.
        patch_mode (bool):
            Whether input is treated as image patches (``True``) or as WSIs
            (``False``).
        model (ModelABC):
            Loaded PyTorch model. Can be a pretrained TIAToolbox model or a
            custom user-provided model.
        ioconfig (ModelIOConfigABC):
            IO configuration specifying patch extraction shape, stride, and
            resolution settings for inference.
        return_labels (bool):
            Whether to include labels in the output, if provided.
        input_resolutions (list[dict]):
            Resolution settings for model input heads. Supported units are
            ``"level"``, ``"power"``, and ``"mpp"``.
        patch_input_shape (tuple[int, int]):
            Height and width of input patches read from slides, expressed in
            read resolution space.
        stride_shape (tuple[int, int]):
            Stride used during patch extraction. Defaults to
            ``patch_input_shape``.
        drop_keys (list):
            Keys to exclude from model output when saving results.
        output_type (str):
            Output format (``"dict"``, ``"zarr"``, or ``"annotationstore"``).

    Examples:
        >>> from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
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
        ...     memory_threshold=80,
        ... )

    """

    def __init__(
        self: NucleusDetector,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusDetector`.

        This constructor follows the standard TIAToolbox engine initialization
        workflow. A model may be provided either as a string referring to a
        pretrained TIAToolbox architecture or as a custom ``torch.nn.Module``.
        When ``model`` is a string, the corresponding pretrained weights are
        automatically downloaded unless explicitly overridden via ``weights``.

        Args:
            model (str or ModelABC):
                A PyTorch model instance or the name of a pretrained TIAToolbox
                model. If a string is provided, default pretrained weights are
                loaded unless ``weights`` is supplied to override them.

            batch_size (int):
                Number of image patches processed per forward pass.
                Default is ``8``.

            num_workers (int):
                Number of workers used for ``torch.utils.data.DataLoader``.
                Default is ``0``.

            weights (str or Path or None):
                Path to model weights. If ``None`` and ``model`` is a string,
                the default pretrained weights for that model will be used.
                If ``model`` is a ``nn.Module``, weights are loaded only when
                specified here.

            device (str):
                Device on which the model will run (e.g., ``"cpu"``, ``"cuda"``).
                Default is ``"cpu"``.

            verbose (bool):
                Whether to enable verbose logging during initialization and
                inference. Default is ``True``.

        """
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
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict:
        """Post-process patch-level detection outputs.

        Applies the model's post-processing function (e.g., centroid extraction and
        thresholding) to each patch's probability map, yielding per-patch detection
        arrays suitable for saving or further merging.

        Args:
            raw_predictions (da.Array):
                Patch predictions of shape ``(B, H, W, C)``, where ``B`` is the number
                of patches (probabilities/logits).
            prediction_shape (tuple[int, ...]):
                Expected prediction shape.
            prediction_dtype (type):
                Expected prediction dtype.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    min_distance (int):
                        Minimum separation between nuclei (in pixels) used during
                        centroid extraction/post-processing.
                    threshold_abs (float):
                        Absolute detection threshold applied to model outputs.
                    threshold_rel (float):
                        Relative detection threshold
                        (e.g., with respect to local maxima).


        Returns:
            dict[str, list[da.Array]]:
                A dictionary of lists (one list per patch), with keys:
                - ``"x"`` (list[dask array]):
                    1-D object dask arrays of x coordinates (``np.uint32``).
                - ``"y"`` (list[dask array]):
                    1-D object dask arrays of y coordinates (``np.uint32``).
                - ``"classes"`` (list[dask array]):
                    1-D object dask arrays of class IDs (``np.uint32``).
                - ``"probabilities"`` (list[dask array]):
                    1-D object dask arrays of detection scores (``np.float32``).

        Notes:
            - If thresholds are not provided via ``kwargs``, model defaults are used.

        """
        logger.info("Post processing patch predictions in NucleusDetector")
        _ = prediction_shape
        _ = prediction_dtype

        # If these are not provided, defaults from model will be used in postproc
        min_distance = kwargs.get("min_distance")
        threshold_abs = kwargs.get("threshold_abs")
        threshold_rel = kwargs.get("threshold_rel")

        # Lists to hold per-patch detection arrays
        xs = []
        ys = []
        classes = []
        probs = []

        # Process each patch's predictions
        for i in range(raw_predictions.shape[0]):
            probs_prediction_patch = raw_predictions[i].compute()
            centroids_map_patch = self.model.postproc(
                probs_prediction_patch,
                min_distance=min_distance,
                threshold_abs=threshold_abs,
                threshold_rel=threshold_rel,
            )
            centroids_map_patch = da.from_array(centroids_map_patch, chunks="auto")
            xs_patch, ys_patch, classes_patch, probs_patch = (
                self._centroid_maps_to_detection_arrays(centroids_map_patch).values()
            )
            xs.append(xs_patch)
            ys.append(ys_patch)
            classes.append(classes_patch)
            probs.append(probs_patch)

        return {"x": xs, "y": ys, "classes": classes, "probabilities": probs}

    def post_process_wsi(
        self: NucleusDetector,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict[str, da.Array]:
        """Post-process WSI-level nucleus detection outputs.

        Processes the full-slide prediction map using Dask's block-wise operations
        to extract nuclei centroids across the entire WSI. The prediction map is
        first re-chunked to the model's preferred post-processing tile shape, and
        `dask.map_overlap` with halo padding is used to facilitate centroid
        extraction on large prediction maps. The resulting centroid maps are
        computed and saved to Zarr storage for memory-efficient processing, then
        converted into detection arrays (x, y, classes, probabilities) through
        sequential block processing.

        Args:
            raw_predictions (da.Array):
                WSI prediction map of shape ``(H, W, C)`` containing
                per-class probabilities or logits.
            prediction_shape (tuple[int, ...]):
                Expected prediction shape.
            prediction_dtype (type):
                Expected prediction dtype.
            **kwargs (NucleusDetectorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    min_distance (int):
                        Minimum distance separating two nuclei (in pixels).
                    threshold_abs (float):
                        Absolute detection threshold applied to model outputs.
                    threshold_rel (float):
                        Relative detection threshold
                        (e.g., with respect to local maxima).
                    postproc_tile_shape (tuple[int, int]):
                        Tile shape (height, width) for post-processing rechunking.
                    cache_dir (str or os.PathLike):
                        Directory for caching intermediate centroid maps as Zarr.
                        Defaults to './tmp/'.

        Returns:
            dict[str, da.Array]:
                A dictionary mapping detection fields to 1-D Dask arrays:
                - ``"x"``: x coordinates of detected nuclei (``np.uint32``).
                - ``"y"``: y coordinates of detected nuclei (``np.uint32``).
                - ``"classes"``: class IDs (``np.uint32``).
                - ``"probabilities"``: detection scores (``np.float32``).

        Notes:
            - Halo padding ensures that nuclei crossing tile/chunk boundaries
              are not fragmented or duplicated.
            - If thresholds are not explicitly provided, model defaults are used.
            - Centroid maps are computed and saved to Zarr storage to avoid
              out-of-memory errors on large WSIs.
            - The Zarr-backed centroid maps are then processed block-by-block
              to extract detections incrementally.

        """
        _ = prediction_shape

        logger.info("Post processing WSI predictions in NucleusDetector")

        # If these are not provided, defaults from model will be used in postproc
        threshold_abs = kwargs.get("threshold_abs")
        threshold_rel = kwargs.get("threshold_rel")

        # min_distance and postproc_tile_shape cannot be None here
        min_distance = kwargs.get("min_distance")
        if min_distance is None:
            min_distance = self.model.min_distance
        postproc_tile_shape = kwargs.get("postproc_tile_shape")
        if postproc_tile_shape is None:
            postproc_tile_shape = self.model.postproc_tile_shape

        # Add halo (overlap) around each block for post-processing
        depth_h = min_distance
        depth_w = min_distance
        depth = {0: depth_h, 1: depth_w, 2: 0}

        # Re-chunk to post-processing tile shape for more efficient processing
        rechunked_prediction_map = raw_predictions.rechunk(
            (postproc_tile_shape[0], postproc_tile_shape[1], -1)
        )

        centroid_maps = da.map_overlap(
            self.model.postproc,
            rechunked_prediction_map,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            depth=depth,
            boundary=0,
            dtype=prediction_dtype,
            block_info=True,
            depth_h=depth_h,
            depth_w=depth_w,
        )

        logger.info("Computing and saving centroid maps to temporary zarr file.")
        temp_zarr_file = tempfile.TemporaryDirectory(
            prefix="tiatoolbox_nucleus_detector_", suffix=".zarr"
        )
        logger.info("Temporary zarr file created at: %s", temp_zarr_file.name)
        task = centroid_maps.to_zarr(
            url=temp_zarr_file.name, compute=False, object_codec=None
        )
        with ProgressBar():
            compute(task)

        centroid_maps = da.from_zarr(temp_zarr_file.name)

        return self._centroid_maps_to_detection_arrays(centroid_maps)

    def save_predictions(
        self: NucleusDetector,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[NucleusDetectorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Save nucleus detections to disk or return them in memory.

        Saves post-processed detection outputs in one of the supported formats.
        If ``patch_mode=True``, predictions are saved per image. If
        ``patch_mode=False``, detections are merged and saved as a single output.

        Args:
            processed_predictions (dict):
                Dictionary containing processed detection results. Expected to include
                a ``"predictions"`` key with detection arrays. The internal structure
                follows TIAToolbox conventions and may differ slightly between patch
                and WSI modes:
                - Patch mode:
                  - ``"x"`` (list[da.Array]):
                    per-patch x coordinates (np.uint32).
                  - ``"y"`` (list[da.Array]):
                    per-patch y coordinates (np.uint32).
                  - ``"classes"`` (list[da.Array]):
                    per-patch class IDs (np.uint32).
                  - ``"probabilities"`` (list[da.Array]):
                    per-patch detection scores (np.float32).
                - WSI mode:
                  - ``"x"`` (da.Array):
                    x coordinates (np.uint32).
                  - ``"y"`` (da.Array):
                    y coordinates (np.uint32).
                  - ``"classes"`` (da.Array):
                    class IDs (np.uint32).
                  - ``"probabilities"`` (da.Array):
                    detection scores (np.float32).

            output_type (str):
                Desired output format: ``"dict"``, ``"zarr"``, or ``"annotationstore"``.

            save_path (Path | None):
                Path at which to save the output file(s). Required for file outputs
                (e.g., Zarr or SQLite .db). If ``None`` and ``output_type="dict"``,
                results are returned in memory.

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
            dict | AnnotationStore | Path | list[Path]:
                - If ``output_type="dict"``:
                    returns a Python dictionary of predictions.
                - If ``output_type="zarr"``:
                    returns the path to the saved ``.zarr`` group.
                - If ``output_type="annotationstore"``:
                    returns an AnnotationStore handle or the path(s) to saved
                    ``.db`` file(s). In patch mode, a list of per-image paths
                    may be returned.

        Notes:
            - For non-AnnotationStore outputs, this method delegates to the
              base engine's saving function to preserve consistency across
              TIAToolbox engines.

        """
        if output_type.lower() != "annotationstore":
            return super().save_predictions(
                processed_predictions["predictions"],
                output_type,
                save_path=save_path,
                **kwargs,
            )

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")
        if class_dict is None:
            class_dict = self.model.output_class_dict

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
        """Save nucleus detections to an AnnotationStore (.db).

        Converts the processed detection arrays into per-instance `Annotation`
        records, applies coordinate scaling and optional class-ID remapping,
        and writes the results into an SQLite-backed AnnotationStore. In patch
        mode, detections are written to separate `.db` files per input image;
        in WSI mode, all detections are merged and written to a single store.

        Args:
            processed_predictions (dict):
                Dictionary containing the computed detection outputs. Expected to
                include a top-level key ``"predictions"`` with fields:
                - ``"x"`` (da.Array):
                    dask array of x coordinates (``np.uint32``)
                - ``"y"`` (da.Array):
                    dask array of y coordinates (``np.uint32``)
                - ``"classes"`` (da.Array):
                    dask array of class IDs (``np.uint32``)
                - ``"probabilities"`` (da.Array):
                    dask array of detection scores (``np.float32``)

            save_path (Path or None):
                Output path for saving the AnnotationStore. If ``None``, an in-memory
                store is returned. When patch mode is active, this path serves as the
                directory for producing one `.db` file per patch input.

            scale_factor (tuple[float, float], optional):
                Scaling factors applied to x and y coordinates prior to writing.
                Typically corresponds to ``model_mpp / slide_mpp``.
                Defaults to ``(1.0, 1.0)``.

            class_dict (dict or None):
                Optional mapping from original class IDs to class names or remapped IDs.
                If ``None``, an identity mapping based on present classes is used.

        Returns:
            AnnotationStore or Path or list[Path]:
                - For WSI mode: a single AnnotationStore handle or the path to the saved
                  `.db` file.
                - For patch mode: a list of paths, one per saved patch-level
                  AnnotationStore.

        Notes:
            - This method centralizes the translation of detection arrays into
              `Annotation` objects and abstracts batching logic via
              ``_write_detection_arrays_to_store``.

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
                    "probabilities": detections["probabilities"][i],
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
        """Convert centroid maps into 1-D detection arrays.

        This helper function extracts non-zero centroid predictions from a
        already computed Dask array of centroid maps and flattens them into
        coordinate, class, and probability arrays suitable for saving or
        further processing. The function processes the centroid maps block
        by block to minimize memory usage, reading each block from disk
        and extracting detections incrementally.

        Args:
            detection_maps (da.Array):
                A Dask array of shape ``(H, W, C)`` representing centroid
                probability maps, where non-zero values correspond to nucleus
                detections. Each non-zero entry encodes both the class channel
                and its associated probability. This array is expected to be
                already computed.

        Returns:
            dict[str, da.Array]:
                A dictionary containing four 1-D Dask arrays:
                - ``"x"``:
                    x coordinates of detected nuclei (``np.uint32``).
                - ``"y"``:
                    y coordinates of detected nuclei (``np.uint32``).
                - ``"classes"``:
                    class IDs for each detection (``np.uint32``).
                - ``"probabilities"``:
                    detection probabilities (``np.float32``).

        Notes:
            - The centroid maps are expected to be pre-computed.
            - Blocks are processed sequentially to avoid loading the entire
              centroid map into memory at once.
            - Global coordinates are computed by adding block offsets to local
              coordinates within each block.
            - This method is used by both patch-level and WSI-level
              post-processing routines to unify detection formatting.


        """
        logger.info("Extracting detections from centroid maps block by block...")

        # Get chunk information
        num_blocks_h = detection_maps.numblocks[0]
        num_blocks_w = detection_maps.numblocks[1]

        # Lists to collect detections from each block
        ys_list = []
        xs_list = []
        classes_list = []
        probs_list = []

        tqdm = get_tqdm()
        for i in tqdm(range(num_blocks_h), desc="Processing detection blocks"):
            for j in range(num_blocks_w):
                # Get block offsets
                y_offset = sum(detection_maps.chunks[0][:i]) if i > 0 else 0
                x_offset = sum(detection_maps.chunks[1][:j]) if j > 0 else 0

                # Read this block from Zarr (already computed, so this is just I/O)
                block = np.array(detection_maps.blocks[i, j])

                # Extract nonzero detections
                ys, xs, classes = np.nonzero(block)
                probs = block[ys, xs, classes]

                # Adjust to global coordinates
                ys = ys + y_offset
                xs = xs + x_offset

                # Append to lists if we have detections
                if len(ys) > 0:
                    ys_list.append(ys.astype(np.uint32))
                    xs_list.append(xs.astype(np.uint32))
                    classes_list.append(classes.astype(np.uint32))
                    probs_list.append(probs.astype(np.float32))

        # Concatenate all block results
        if ys_list:
            ys = np.concatenate(ys_list)
            xs = np.concatenate(xs_list)
            classes = np.concatenate(classes_list)
            probs = np.concatenate(probs_list)
        else:
            ys = np.array([], dtype=np.uint32)
            xs = np.array([], dtype=np.uint32)
            classes = np.array([], dtype=np.uint32)
            probs = np.array([], dtype=np.float32)

        return {
            "y": da.from_array(ys, chunks="auto"),
            "x": da.from_array(xs, chunks="auto"),
            "classes": da.from_array(classes, chunks="auto"),
            "probabilities": da.from_array(probs, chunks="auto"),
        }

    @staticmethod
    def _write_detection_arrays_to_store(
        detection_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        store: SQLiteStore,
        scale_factor: tuple[float, float],
        class_dict: dict[int, str | int] | None,
        batch_size: int = 5000,
    ) -> int:
        """Write detection arrays to an AnnotationStore in batches.

        Converts coordinate, class, and probability arrays into `Annotation`
        objects and appends them to an SQLite-backed store in configurable
        batch sizes. Coordinates are scaled to baseline slide resolution using
        the provided `scale_factor`, and optional class-ID remapping is applied
        via `class_dict`.

        Args:
            detection_arrays (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
                Tuple of arrays in the order:
                `(x_coords, y_coords, class_ids, probabilities)`.
                Each element must be a 1-D NumPy array of equal length.
            store (SQLiteStore):
                Target `AnnotationStore` instance to receive the detections.
            scale_factor (tuple[float, float]):
                Factors applied to `(x, y)` coordinates prior to writing,
                typically `(model_mpp / slide_mpp)`. The scaled coordinates are
                rounded to `np.uint32`.
            class_dict (dict[int, str | int] | None):
                Optional mapping from original class IDs to names or remapped IDs.
                If `None`, an identity mapping is used for the set of present classes.
            batch_size (int):
                Number of records to write per batch. Default is `5000`.

        Returns:
            int:
                Total number of detection records written to the store.

        Notes:
            - Coordinates are scaled and rounded to integers to ensure consistent
              geometry creation for `Annotation` points.
            - Class mapping is applied per-record; unmapped IDs fall back to their
              original values.
            - Writing in batches reduces memory pressure and improves throughput
              on large number of detections.

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
                    geometry=pt, properties={"type": lbl, "probability": float(pp)}
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
        """Write nucleus detection arrays to an SQLite-backed AnnotationStore.

        Converts the detection arrays into NumPy form, applies coordinate scaling
        and optional class-ID remapping, and writes the results into an in-memory
        SQLiteStore. If `save_path` is provided, the store is committed and saved
        to disk as a `.db` file. This method provides a unified interface for
        converting Dask-based detection outputs into persistent annotation storage.

        Args:
            detection_arrays (dict[str, da.Array]):
                A dictionary containing the detection fields:
                - ``"x"``: dask array of x coordinates (``np.uint32``).
                - ``"y"``: dask array of y coordinates (``np.uint32``).
                - ``"classes"``: dask array of class IDs (``np.uint32``).
                - ``"probabilities"``: dask array of detection scores (``np.float32``).

            scale_factor (tuple[float, float], optional):
                Multiplicative factors applied to the x and y coordinates before
                saving. The scaled coordinates are rounded to integer pixel
                locations. Defaults to ``(1.0, 1.0)``.

            class_dict (dict or None):
                Optional mapping of class IDs to class names or remapped IDs.
                If ``None``, an identity mapping is used based on the detected
                class IDs.

            save_path (Path or None):
                Destination path for saving the `.db` file. If ``None``, the
                resulting SQLiteStore is returned in memory. If provided, the
                parent directory is created if needed, and the final store is
                written as ``save_path.with_suffix(".db")``.

            batch_size (int):
                Number of detection records to write per batch. Defaults to ``5000``.

        Returns:
            Path or SQLiteStore:
                - If `save_path` is provided: the path to the saved `.db` file.
                - If `save_path` is ``None``: an in-memory `SQLiteStore` containing
                  all detections.

        Notes:
            - The heavy lifting is delegated to
              :meth:`NucleusDetector._write_detection_arrays_to_store`,
              which performs coordinate scaling, class mapping, and batch writing.

        """
        xs = detection_arrays["x"]
        ys = detection_arrays["y"]
        classes = detection_arrays["classes"]
        probs = detection_arrays["probabilities"]

        xs = np.atleast_1d(np.asarray(xs))
        ys = np.atleast_1d(np.asarray(ys))
        classes = np.atleast_1d(np.asarray(classes))
        probs = np.atleast_1d(np.asarray(probs))

        if not len(xs) == len(ys) == len(classes) == len(probs):
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
                    threshold_abs (float):
                        Absolute detection threshold applied to model outputs.
                    threshold_rel (float):
                        Relative detection threshold
                        (e.g., with respect to local maxima).
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
            >>> from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
            >>> detector = NucleusDetector(model="mapde-conic")
            >>> # WSI workflow: save to AnnotationStore (.db)
            >>> out = detector.run(
            ...     images=[pathlib.Path("example_wsi.tiff")],
            ...     patch_mode=False,
            ...     device="cuda",
            ...     save_dir=pathlib.Path("output_directory/"),
            ...     overwrite=True,
            ...     output_type="annotationstore",
            ...     class_dict={0: "nucleus"},
            ...     auto_get_mask=True,
            ...     memory_threshold=80,
            ... )
            >>> # Patch workflow: return in-memory detections
            >>> patches = [np.ndarray, np.ndarray]  # NHWC
            >>> out = detector.run(patches, patch_mode=True, output_type="dict")


        """
        output = super().run(
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

        if not patch_mode:
            # Clean up temporary zarr directory after WSI processing
            # It should have been already deleted, but check anyway
            temp_dir = Path(tempfile.gettempdir())
            if temp_dir.exists():
                # find file starting with 'tiatoolbox_nucleus_detector_'
                # and ending with '.zarr'
                for item in temp_dir.iterdir():
                    if item.name.startswith(
                        "tiatoolbox_nucleus_detector_"
                    ) and item.name.endswith(".zarr"):
                        shutil.rmtree(item)
                        logger.info(
                            "Temporary zarr directory %s has been removed.", item
                        )

        return output