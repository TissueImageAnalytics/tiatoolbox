"""This module enables multi-task segmentor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import torch
import zarr
from shapely.geometry import shape as feature2geometry
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.annotation.storage import Annotation
from tiatoolbox.utils.misc import get_tqdm, make_valid_poly
from tiatoolbox.wsicore.wsireader import is_zarr

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOSegmentorConfig


class MultiTaskSegmentor(SemanticSegmentor):
    """A multitask segmentation engine for models like hovernet and hovernetplus."""

    def __init__(
        self: MultiTaskSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusInstanceSegmentor`."""
        self.tasks = set()
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def infer_patches(
        self: MultiTaskSegmentor,
        dataloader: DataLoader,
        *,
        return_coordinates: bool = False,
    ) -> dict[str, list[da.Array]]:
        """Run model inference on image patches and return predictions.

        This method performs batched inference using a PyTorch DataLoader,
        and accumulates predictions in Dask arrays. It supports optional inclusion
        of coordinates and labels in the output.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader containing image patches for inference.
            return_coordinates (bool):
                Whether to include coordinates in the output. Required when
                called by `infer_wsi` and `patch_mode` is False.

        Returns:
            dict[str, dask.array.Array]:
                Dictionary containing prediction results as Dask arrays.
                Keys include:
                    - "probabilities": Model output probabilities.
                    - "coordinates": Patch coordinates (if `return_coordinates` is
                      True).

        """
        keys = ["probabilities"]
        coordinates = []

        # Expected number of outputs from the model
        batch_output = self.model.infer_batch(
            self.model,
            torch.Tensor(dataloader.dataset[0]["image"][np.newaxis, ...]),
            device=self.device,
        )

        num_expected_output = len(batch_output)
        probabilities = [[] for _ in range(num_expected_output)]

        if return_coordinates:
            keys.append("coordinates")
            coordinates = []

        # Main output dictionary
        raw_predictions = {key: [] for key in keys}
        raw_predictions["probabilities"] = [[] for _ in range(num_expected_output)]

        # Inference loop
        tqdm = get_tqdm()
        tqdm_loop = (
            tqdm(dataloader, leave=False, desc="Inferring patches")
            if self.verbose
            else self.dataloader
        )

        for batch_data in tqdm_loop:
            batch_output = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            for i in range(num_expected_output):
                probabilities[i].append(
                    da.from_array(
                        batch_output[i],  # probabilities
                    )
                )

            if return_coordinates:
                coordinates.append(
                    da.from_array(
                        self._get_coordinates(batch_data),
                    )
                )

        for i in range(num_expected_output):
            raw_predictions["probabilities"][i] = da.concatenate(
                probabilities[i], axis=0
            )

        if return_coordinates:
            raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)

        return raw_predictions

    def post_process_patches(  # skipcq: PYL-R0201
        self: MultiTaskSegmentor,
        raw_predictions: dict,
        **kwargs: Unpack[SemanticSegmentorRunParams],  # noqa: ARG002
    ) -> dict:
        """Post-process raw patch predictions from inference.

        This method applies a post-processing function (e.g., smoothing, filtering)
        to the raw model predictions. It supports delayed execution using Dask
        and returns a Dask array for efficient computation.

        Args:
            raw_predictions (dask.array.Array):
                Raw model predictions as a dask array.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters used for post-processing.

        Returns:
            dask.array.Array:
                Post-processed predictions as a Dask array.

        """
        probabilities = raw_predictions["probabilities"]
        post_process_predictions = [
            self.model.postproc_func(list(probs_for_idx))
            for probs_for_idx in zip(*probabilities, strict=False)
        ]

        raw_predictions = self.build_post_process_raw_predictions(
            post_process_predictions=post_process_predictions,
            raw_predictions=raw_predictions,
        )

        # Need to update info_dict
        _ = raw_predictions

        return raw_predictions

    def build_post_process_raw_predictions(
        self: MultiTaskSegmentor,
        post_process_predictions: list[tuple],
        raw_predictions: dict,
    ) -> dict:
        """Merge per-image outputs into a task-organized prediction structure.

        This function takes a list of outputs, where each element corresponds to one
        image and contains one or more segmentation dictionaries. Each segmentation
        dictionary must include a ``"task_type"`` key along with any number of
        additional fields (e.g., ``"predictions"``, ``"info_dict"``, or others).

        The function reorganizes these outputs into ``raw_predictions`` by grouping
        entries under their respective task types. For each task, all keys except
        ``"task_type"`` are stored in dictionaries indexed by ``img_id``. Existing
        content in ``raw_predictions`` is preserved and extended as needed.

        Args:
            post_process_predictions (list[tuple]):
                A list where each element represents one image. Each element is an
                iterable of segmentation dictionaries. Each segmentation dictionary
                must contain a ``"task_type"`` field and may contain any number of
                additional fields.
            raw_predictions (dict):
                A dictionary that will be updated in-place. It may already contain
                task entries or other unrelated keys. New tasks and new fields are
                added dynamically as they appear in ``outputs``.

        Returns:
            dict:
                The updated ``raw_predictions`` dictionary, containing all tasks and
                their associated per-image fields.

        """
        tasks = set()
        for seg_list in post_process_predictions:
            for seg in seg_list:
                task = seg["task_type"]
                tasks.add(task)

                # Initialize task entry if needed
                if task not in raw_predictions:
                    raw_predictions[task] = {}

                # For every key except task_type, store values by img_id
                for key, value in seg.items():
                    if key == "task_type":
                        continue

                    # Initialize list for this key
                    if key not in raw_predictions[task]:
                        raw_predictions[task][key] = []

                    raw_predictions[task][key].append(value)

        for task in tasks:
            task_dict = raw_predictions[task]
            for key in list(task_dict.keys()):
                values = task_dict[key]
                if all(isinstance(v, (np.ndarray, da.Array)) for v in values):
                    raw_predictions[task][key] = da.stack(values, axis=0)
                    continue

                if all(isinstance(v, dict) for v in values):
                    first = values[0]

                    # Add new keys safely
                    for subkey in first:
                        raw_predictions[task][subkey] = [d[subkey] for d in values]

                    del raw_predictions[task][key]

        self.tasks = tasks
        return raw_predictions

    def _save_predictions_as_dict_zarr(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Helper function to save predictions as dictionary or zarr."""
        if output_type.lower() == "dict":
            # If there is a single task simplify the output.
            if len(self.tasks) == 1:
                task_output = processed_predictions.pop(next(iter(self.tasks)))
                processed_predictions.update(task_output)
            return super().save_predictions(
                processed_predictions, output_type, save_path=save_path, **kwargs
            )

        # Save to zarr
        for task_name in self.tasks:
            processed_predictions_ = processed_predictions.pop(task_name)
            # If there is a single task simplify the output.
            task_name_ = None if len(self.tasks) == 1 else task_name
            keys_to_compute = [
                k for k in processed_predictions_ if k not in self.drop_keys
            ]
            _ = self.save_predictions_as_zarr(
                processed_predictions=processed_predictions_,
                save_path=save_path,
                keys_to_compute=keys_to_compute,
                task_name=task_name_,
            )
        return save_path

    def _save_predictions_as_annotationstore(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        task_name: str | None = None,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Helper function to save predictions as annotationstore."""
        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")
        # Need to add support for zarr conversion.
        save_paths = []

        logger.info("Saving predictions as AnnotationStore.")

        # Not required for annotationstore
        processed_predictions.pop("predictions")

        if self.patch_mode:
            for i, predictions in enumerate(
                zip(*processed_predictions.values(), strict=False)
            ):
                predictions_ = dict(
                    zip(processed_predictions.keys(), predictions, strict=False)
                )
                if isinstance(self.images[i], Path):
                    store_file_name = (
                        f"{self.images[i].stem}.db"
                        if task_name is None
                        else f"{self.images[i].stem}_{task_name}.db"
                    )
                    output_path = save_path.parent / store_file_name
                else:
                    store_file_name = (
                        f"{i}.db" if task_name is None else f"{i}_{task_name}.db"
                    )
                    output_path = save_path.parent / store_file_name

                origin = predictions_.pop("coordinates")[:2]
                store = SQLiteStore()
                store = dict_to_store(
                    store=store,
                    processed_predictions=predictions_,
                    class_dict=class_dict,
                    scale_factor=scale_factor,
                    origin=origin,
                )

                store.commit()
                store.dump(output_path)

                save_paths.append(output_path)
        return_probabilities = kwargs.get("return_probabilities", False)
        if return_probabilities:
            msg = (
                f"Probability maps cannot be saved as AnnotationStore. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {save_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)

        return save_paths

    def save_predictions(
        self: MultiTaskSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Save model predictions to disk or return them in memory.

        Depending on the output type, this method saves predictions as a zarr group,
        an AnnotationStore (SQLite database), or returns them as a dictionary.

        Args:
            processed_predictions (dict):
                Dictionary containing processed model predictions.
            output_type (str):
                Desired output format.
                Supported values are "dict", "zarr", and "annotationstore".
            save_path (Path | None):
                Path to save the output file.
                Required for "zarr" and "annotationstore" formats.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters to update engine attributes.
                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").
                        See :class:`torch.device` for more details.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., "zarr" or "annotationstore").
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates from non-baseline to baseline
                        resolution.
                    stride_shape (IntPair):
                        Stride used during WSI processing, at requested read resolution.
                        Must be positive. Defaults to `patch_input_shape` if not
                        provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            dict | AnnotationStore | Path | list [Path]:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved zarr file.
                - If output_type is "annotationstore": returns an AnnotationStore
                  or path to .db file.

        Raises:
            TypeError:
                If an unsupported output_type is provided.

        """
        if output_type in ["dict", "zarr"]:
            return self._save_predictions_as_dict_zarr(
                processed_predictions=processed_predictions,
                output_type=output_type,
                save_path=save_path,
                **kwargs,
            )

        # Save to AnnotationStore
        return_probabilities = kwargs.get("return_probabilities", False)
        output_type_ = (
            "zarr"
            if is_zarr(save_path.with_suffix(".zarr")) or return_probabilities
            else "dict"
        )

        # This runs dask.compute and returns numpy arrays
        # for saving annotationstore output.
        processed_predictions = self._save_predictions_as_dict_zarr(
            processed_predictions,
            output_type=output_type_,
            save_path=save_path.with_suffix(".zarr"),
            **kwargs,
        )

        if isinstance(processed_predictions, Path):
            processed_predictions = zarr.open(str(processed_predictions), mode="r")

        save_paths = []
        if self.tasks & processed_predictions.keys():
            for task_name in self.tasks:
                dict_for_store = {
                    **processed_predictions[task_name],
                    "coordinates": processed_predictions["coordinates"],
                }
                out_path = self._save_predictions_as_annotationstore(
                    processed_predictions=dict_for_store,
                    task_name=task_name,
                    save_path=save_path,
                    **kwargs,
                )
                save_paths += out_path

            return save_paths

        return self._save_predictions_as_annotationstore(
            processed_predictions=processed_predictions,
            task_name=None,
            save_path=save_path,
            **kwargs,
        )

    def run(
        self: MultiTaskSegmentor,
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
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").

                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    output_resolutions (Resolution):
                        Resolution used for writing output predictions.
                    patch_output_shape (tuple[int, int]):
                        Shape of output patches (height, width).
                    return_labels (bool):
                        Whether to return labels with predictions. Should be False.
                    return_probabilities (bool):
                        Whether to return per-class probabilities.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

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
        return_labels = kwargs.get("return_labels")

        # Passing multitask labels causes unnecessary memory overheads
        if return_labels:
            msg = "`return_labels` is not supported for MultiTaskSegmentor."
            raise ValueError(msg)

        kwargs["return_labels"] = False

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


def dict_to_store(
    store: SQLiteStore,
    processed_predictions: dict,
    class_dict: dict | None = None,
    origin: tuple[float, float] = (0, 0),
    scale_factor: tuple[float, float] = (1, 1),
) -> AnnotationStore:
    """Helper function to convert dict to store."""
    contour = processed_predictions.pop("contours")

    ann = []
    for i, contour_ in enumerate(contour):
        ann_ = Annotation(
            make_valid_poly(
                feature2geometry(
                    {
                        "type": processed_predictions.get("geom_type", "Polygon"),
                        "coordinates": scale_factor * np.array([contour_]),
                    },
                ),
                tuple(origin),
            ),
            {
                prop: (
                    class_dict[processed_predictions[prop][i]]
                    if prop == "type" and class_dict is not None
                    # Intention is convert arrays to list
                    # There might be int or float values which need to be
                    # converted to arrays first and then apply tolist().
                    else np.array(processed_predictions[prop][i]).tolist()
                )
                for prop in processed_predictions
            },
        )
        ann.append(ann_)
    logger.info("Added %d annotations.", len(ann))
    store.append_many(ann)

    return store
