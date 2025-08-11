"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import torch
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.models.dataset.dataset_abc import WSIPatchDataset
from tiatoolbox.utils.misc import (
    dict_to_store_semantic_segmentor,
    dict_to_zarr,
    get_tqdm,
)

from .patch_predictor import PatchPredictor, PredictorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader


def merge_all(
    blocks: np.ndarray,
    output_locations: np.ndarray,
    merged_shape: tuple[int, int, int],
    dtype_: type,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge patch-level predictions into a single canvas.

    This function aggregates overlapping patch predictions into a unified
    output canvas and maintains a count map to normalize overlapping regions.

    Args:
        blocks (np.ndarray):
            Array of predicted blocks with shape (N, H, W, C), where N is the
            number of patches.
        output_locations (np.ndarray):
            Array of coordinates for each block in the format
            [start_x, start_y, end_x, end_y] with shape (N, 4).
        merged_shape (tuple[int, int, int]):
            Shape of the final merged canvas (H, W, C).
        dtype_ (type):
            Data type of the output canvas.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - canvas: Merged prediction map of shape (H, W, C).
            - count: Count map indicating how many times each pixel was updated,
              shape (H, W).

    """
    canvas = np.zeros(merged_shape, dtype=dtype_)
    count = np.zeros(merged_shape[:2], dtype=np.uint8)
    for i, block in enumerate(blocks):
        xs, ys, xe, ye = output_locations[i]
        # To deal with edge cases
        ye, xe = min(ye, canvas.shape[0]), min(xe, canvas.shape[1])
        canvas[ys:ye, xs:xe, :] += block[0 : ye - ys, 0 : xe - xs, :]
        count[ys:ye, xs:xe] += 1
    return canvas, count


class SemanticSegmentorRunParams(PredictorRunParams, total=False):
    """Runtime parameters for configuring the `SemanticSegmentor.run()` method.

    This class extends `PredictorRunParams` with additional parameters
    specific to semantic segmentation workflows.

    Attributes:
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        cache_mode (bool):
            Whether to run the engine in cache mode. Recommended for large datasets.
        cache_size (int):
            Number of patches to process in a batch when cache_mode is True.
        class_dict (dict):
            Optional dictionary mapping classification outputs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda").
        ioconfig (ModelIOConfigABC):
            Input/output configuration for patch extraction and resolution.
        return_labels (bool):
            Whether to return labels with predictions.
        num_loader_workers (int):
            Number of workers used in DataLoader.
        num_post_proc_workers (int):
            Number of workers used for post-processing.
        output_file (str):
            Output file name for saving results (e.g., .zarr or .db).
        patch_input_shape (tuple[int, int]):
            Shape of input patches (height, width).
        input_resolutions (list[dict]):
            Resolution used for reading the image. See `WSIReader` for details.
        return_probabilities (bool):
            Whether to return per-class probabilities.
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.
        patch_output_shape (tuple[int, int]):
            Shape of output patches (height, width).
        output_resolutions (Resolution):
            Resolution used for writing output predictions.

    """

    patch_output_shape: tuple[int, int]
    output_resolutions: Resolution

    patch_output_shape: tuple
    output_resolutions: Resolution


class SemanticSegmentor(PatchPredictor):
    r"""Semantic segmentation engine for digital histology images.

    This class extends `PatchPredictor` to support semantic segmentation tasks
    using pretrained or custom models from TIAToolbox. It supports both patch-level
    and whole slide image (WSI) processing, and provides utilities for merging,
    post-processing, and saving predictions.

    Performance:
        The TIAToolbox model `fcn_resnet50_unet-bcss` achieves the following
        results on the BCSS dataset:

        .. list-table:: Semantic segmentation performance on the BCSS dataset
           :widths: 15 15 15 15 15 15 15
           :header-rows: 1

           * -
             - Tumour
             - Stroma
             - Inflammatory
             - Necrosis
             - Other
             - All
           * - Amgad et al.
             - 0.851
             - 0.800
             - 0.712
             - 0.723
             - 0.666
             - 0.750
           * - TIAToolbox
             - 0.885
             - 0.825
             - 0.761
             - 0.765
             - 0.581
             - 0.763

    Args:
        model (str | ModelABC):
            A PyTorch model instance or name of a pretrained model from TIAToolbox.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter. Default is `None`.
        batch_size (int):
            Number of image patches processed per forward pass. Default is 8.
        num_loader_workers (int):
            Number of workers for data loading. Default is 0.
        num_post_proc_workers (int):
            Number of workers for post-processing. Default is 0.
        weights (str | Path | None):
            Path to model weights. If None, default weights are used.

            >>> engine = SemanticSegmentor(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
        verbose (bool):
            Whether to enable verbose logging. Default is True.

    Attributes:
        images (list[str | Path] | np.ndarray):
            Input image patches or WSI paths.
        masks (list[str | Path] | np.ndarray):
            Optional tissue masks for WSI processing.
            These are only utilized when patch_mode is False.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (bool):
            Whether input is treated as patches (`True`) or WSIs (`False`).
        model (ModelABC):
            Loaded PyTorch model.
        ioconfig (ModelIOConfigABC):
            IO configuration for patch extraction and resolution.
        return_labels (bool):
            Whether to include labels in the output.
        input_resolutions (list[dict]):
            Resolution settings for model input. Supported
            units are `level`, `power` and `mpp`. Keys should be "units" and
            "resolution" e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
            :class:`WSIReader` for details.
        patch_input_shape (tuple[int, int]):
            Shape of input patches (height, width). Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (tuple[int, int]):
            Stride used during patch extraction. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        cache_mode (bool):
            Whether to use caching for large datasets.
        cache_size (int):
            Number of patches to process in a batch when caching.
        labels (list | None):
            Optional labels for input images.
            Only a single label per image is supported.
        drop_keys (list):
            Keys to exclude from model output.
        output_type (str):
            Format of output ("dict", "zarr", "annotationstore").
        output_locations (list | None):
            Coordinates of output patches used during WSI processing.

    Examples:
        >>> # list of 2 image patches as input
        >>> wsis = ['path/img.svs', 'path/img.svs']
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(wsis, patch_mode=False)

        >>> # array of list of 2 image patches as input
        >>> image_patches = [np.ndarray, np.ndarray]
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(data, patch_mode=True)

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(data, patch_mode=False)

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
        >>> output = segmentor.run(tile_file, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsis = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> segmentor = SemanticSegmentor(model="resnet18-kather100k")
        >>> output = segmentor.run(wsis, patch_mode=False)

    References:
        [1] Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. Structured crowdsourcing
        enables convolutional segmentation of histology images. Bioinformatics 2019.
        doi: 10.1093/bioinformatics/btz083

    """

    def __init__(
        self: SemanticSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`SemanticSegmentor`.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or name of a pretrained model from TIAToolbox.
                If a string is provided, the corresponding pretrained weights will be
                downloaded unless overridden via `weights`.
            batch_size (int):
                Number of image patches processed per forward pass. Default is 8.
            num_loader_workers (int):
                Number of workers for data loading. Default is 0.
            num_post_proc_workers (int):
                Number of workers for post-processing. Default is 0.
            weights (str | Path | None):
                Path to model weights. If None, default weights are used.
            device (str):
                Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Whether to enable verbose logging. Default is True.

        """
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )
        self.output_locations: list | None = None

    def get_dataloader(
        self: SemanticSegmentor,
        images: str | Path | list[str | Path] | np.ndarray,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: SemanticSegmentorRunParams | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return a DataLoader for inference.

        This method prepares the dataset and returns a PyTorch DataLoader
        for either patch-based or WSI-based semantic segmentation.

        Args:
            images (str | Path | list[str | Path] | np.ndarray):
                Input images. Can be a list of file paths or a NumPy array
                of image patches in NHWC format.
            masks (Path | None):
                Optional tissue masks for WSI processing. Only used when
                `patch_mode` is False.
            labels (list | None):
                Optional labels for input images. Only one label per image is supported.
            ioconfig (SemanticSegmentorRunParams | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).

        Returns:
            torch.utils.data.DataLoader:
                A PyTorch DataLoader configured for inference.

        """
        # Overwrite when patch_mode is False.
        if not patch_mode:
            dataset = WSIPatchDataset(
                img_path=images,
                mask_path=masks,
                patch_input_shape=ioconfig.patch_input_shape,
                patch_output_shape=ioconfig.patch_output_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )

            dataset.preproc_func = self.model.preproc_func
            self.output_locations = dataset.outputs

            # preprocessing must be defined with the dataset
            return torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_loader_workers,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )

        return super().get_dataloader(
            images=images,
            masks=masks,
            labels=labels,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
        )

    def infer_wsi(
        self: SemanticSegmentor,
        dataloader: DataLoader,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict:
        """Model inference on a WSI.

        Args:
            dataloader (DataLoader):
                A torch dataloader to process WSIs.
            save_path (Path):
                Path to save the intermediate output. The intermediate output is saved
                in a zarr file.
            **kwargs (SemanticSegmentorRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            save_path (Path):
                Path to zarr file where intermediate output is saved.

        """
        _ = kwargs.get("return_probabilities", False)

        keys = ["probabilities", "coordinates"]
        coordinates = []

        if self.return_labels:
            keys.append("labels")
            labels = []

        # Main output dictionary
        raw_predictions = dict(zip(keys, [[]] * len(keys)))

        # sample for calculating shape for dask arrays
        sample = self.dataloader.dataset[0]
        sample_output = self.model.infer_batch(
            self.model,
            torch.Tensor(sample["image"][np.newaxis, ...]),
            device=self.device,
        )

        # Create canvas and counts
        max_location = np.max(self.output_locations, axis=0)
        merged_shape = (
            max_location[3],
            max_location[2],
            sample_output.shape[3],
        )
        canvas = da.zeros(merged_shape, dtype=sample_output.dtype)
        count = da.zeros(merged_shape[:2], dtype=np.uint8)

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

            output_locs = batch_data["output_locs"].numpy()

            batch_xs, batch_ys = np.min(output_locs[:, 0:2], axis=0)
            batch_xe, batch_ye = np.max(output_locs[:, 2:4], axis=0)

            merged_shape_batch = (
                batch_ye - batch_ys,
                batch_xe - batch_xs,
                sample_output.shape[3],
            )

            merged_output, merged_count = merge_all(
                batch_output,
                output_locs - np.array([batch_xs, batch_ys, batch_xs, batch_ys]),
                merged_shape_batch,
                sample_output.dtype,
            )

            batch_ye, batch_xe = (
                min(batch_ye, canvas.shape[0]),
                min(batch_xe, canvas.shape[1]),
            )

            canvas[
                batch_ys:batch_ye,
                batch_xs:batch_xe,
                :,
            ] += merged_output

            count[
                batch_ys:batch_ye,
                batch_xs:batch_xe,
            ] += merged_count

            coordinates.append(
                da.from_array(
                    self._get_coordinates(batch_data),
                )
            )

            if self.return_labels:
                labels.append(da.from_array(np.array(batch_data["label"])))

        raw_predictions["probabilities"] = canvas / da.maximum(
            count[:, :, np.newaxis], 1
        )
        raw_predictions["coordinates"] = da.concatenate(coordinates, axis=0)
        if self.return_labels:
            labels = [label.reshape(-1) for label in labels]
            raw_predictions["labels"] = da.concatenate(labels, axis=0)

        return raw_predictions

    def save_predictions(
        self: SemanticSegmentor,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> dict | AnnotationStore | Path:
        """Save semantic segmentation predictions to disk.

        Args:
            processed_predictions (dict | Path):
                A dictionary or path to zarr with model prediction information.
            save_path (Path):
                Optional output path to directory to save the patch dataset output to a
                `.zarr` or `.db` file, provided `patch_mode` is True. If the
                `patch_mode` is False then `save_dir` is required.
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (SemanticSegmentorRunParams):
                Keyword Args required to save the output.

        Returns:
            dict or Path or :class:`AnnotationStore`:
                If the `output_type` is "AnnotationStore", the function returns
                the patch predictor output as an SQLiteStore containing Annotations
                for each or the Path to a `.db` file depending on whether a
                save_dir Path is provided. Otherwise, the function defaults to
                returning patch predictor output, either as a dict or the Path to a
                `.zarr` file depending on whether a save_dir Path is provided.

        """
        # Conversion to annotationstore uses a different function for SemanticSegmentor
        if output_type.lower() != "annotationstore":
            return super().save_predictions(
                processed_predictions, output_type, save_path=save_path, **kwargs
            )

        logger.info("Saving predictions as AnnotationStore.")
        processed_predictions = super().save_predictions(
            processed_predictions, output_type="dict", **kwargs
        )

        return_probabilities = kwargs.get("return_probabilities", False)

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")

        # Need to add support for zarr conversion.
        save_paths = []

        if self.patch_mode:
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path.parent / (self.images[i].stem + ".db")
                else:
                    output_path = save_path.parent / (str(i) + ".db")

                out_file = dict_to_store_semantic_segmentor(
                    patch_output={"predictions": predictions},
                    scale_factor=scale_factor,
                    class_dict=class_dict,
                    save_path=output_path,
                )

                save_paths.append(out_file)
        else:
            out_file = dict_to_store_semantic_segmentor(
                patch_output=processed_predictions,
                scale_factor=scale_factor,
                class_dict=class_dict,
                save_path=save_path.with_suffix(".db"),
            )
            save_paths = out_file

        if return_probabilities and self.cache_mode:
            zarr_save_path = save_path.parent.with_suffix(".zarr")
            msg = (
                f"Probability maps cannot be saved as AnnotationStore. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {zarr_save_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)
            processed_predictions = {
                "predictions": processed_predictions.get("predictions"),
            }
            dict_to_zarr(
                raw_predictions=processed_predictions,
                save_path=zarr_save_path,
            )

        return save_paths

    def run(
        self: SemanticSegmentor,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the engine on input images.

        Args:
            images (list, ndarray):
                List of inputs to process. When using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list.
            masks (list | None):
                List of masks. Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            labels (list | None):
                List of labels. Only a single label per image is supported.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.
                default = True.
            ioconfig (IOSegmentorConfig):
                IO configuration.
            save_dir (str or pathlib.Path):
                Output directory to save the results.
                If save_dir is not provided when patch_mode is False,
                then for a single image the output is created in the current directory.
                If there are multiple WSIs as input then the user must provide
                path to save directory otherwise an OSError will be raised.
            overwrite (bool):
                Whether to overwrite the results. Default = False.
            output_type (str):
                The format of the output type. "output_type" can be
                "zarr" or "AnnotationStore". Default value is "zarr".
                When saving in the zarr format the output is saved using the
                `python zarr library <https://zarr.readthedocs.io/en/stable/>`__
                as a zarr group. If the required output type is an "AnnotationStore"
                then the output will be intermediately saved as zarr but converted
                to :class:`AnnotationStore` and saved as a `.db` file
                at the end of the loop.
            **kwargs (SemanticSegmentorRunParams):
                Keyword Args to update :class:`EngineABC` attributes during runtime.

        Returns:
            (:class:`numpy.ndarray`, dict, list):
                Model predictions of the input dataset. If multiple
                whole slide images are provided as input,
                or save_output is True, then results are saved to
                `save_dir` and a dictionary indicating save location for
                each input is returned.

                The dict has the following format:

                - img_path: path of the input image.
                - raw: path to save location for raw prediction,
                  saved in .json.
                - list: List of image paths to the output files.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> class SemanticSegmentor(PatchPredictor):
            >>> # Define all Abstract methods.
            >>>     ...
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"
            >>> output = segmentor.run(
            >>>     image_patches,
            >>>     patch_mode=True,
            >>>     output_type="zarr")
            >>> output
            ... "/path/to/Output.zarr"
            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'/path/to/wsi1.db'}

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
