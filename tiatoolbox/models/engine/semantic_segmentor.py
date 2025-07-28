"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import psutil
import torch
import zarr
from dask import compute
from typing_extensions import Unpack

from tiatoolbox import logger
from tiatoolbox.models.dataset.dataset_abc import WSIPatchDataset
from tiatoolbox.utils.misc import (
    dict_to_store_semantic_segmentor,
    get_tqdm,
)

from .patch_predictor import PatchPredictor, PredictorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader


def smart_divide(
    merged_probabilities: zarr.Array,
    merged_weights: zarr.Array,
    tile_size: int = 2048,
    safety_margin: float = 0.5,
    *,
    verbose: bool = False,
) -> zarr.Array:
    """Use chunked division for Zarr if memory is low.

    Divide merged_probabilities by merged_weights using full-array or chunked strategy
    based on available system memory.

    """
    h, w, c = merged_probabilities.shape
    total_elements = h * w * c
    estimated_memory = total_elements * 4 * 2  # float32 = 4 bytes, two arrays

    available_memory = psutil.virtual_memory().available
    if estimated_memory < available_memory * safety_margin:
        # Use full-array division
        merged_weights[merged_weights == 0] = 1
        merged_probabilities[:] = merged_probabilities[:] / merged_weights[:]
    else:  # pragma: no cover
        progress_bar = None
        tqdm = get_tqdm()

        if verbose:
            progress_bar = tqdm(
                total=len(range(0, h, tile_size)),
                leave=False,
                desc="Merging Patches",
            )
        # Use chunked division
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                prob_tile = merged_probabilities[i:i_end, j:j_end, :]
                weight_tile = merged_weights[i:i_end, j:j_end, :]
                weight_tile[weight_tile == 0] = 1
                merged_probabilities[i:i_end, j:j_end, :] = prob_tile / weight_tile

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

    return merged_probabilities


class SemanticSegmentorRunParams(PredictorRunParams):
    """Class describing the input parameters for the :func:`EngineABC.run()` method.

    Attributes:
        batch_size (int):
            Number of image patches to feed to the model in a forward pass.
        cache_mode (bool):
            Whether to run the Engine in cache_mode. For large datasets,
            we recommend to set this to True to avoid out of memory errors.
            For smaller datasets, the cache_mode is set to False as
            the results can be saved in memory.
        cache_size (int):
            Specifies how many image patches to process in a batch when
            cache_mode is set to True. If cache_size is less than the batch_size
            batch_size is set to cache_size.
        class_dict (dict):
            Optional dictionary mapping classification outputs to class names.
        device (str):
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device.
        ioconfig (ModelIOConfigABC):
            Input IO configuration (:class:`ModelIOConfigABC`) to run the Engine.
        return_labels (bool):
            Whether to return the labels with the predictions.
        num_loader_workers (int):
            Number of workers used in :class:`torch.utils.data.DataLoader`.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
        output_file (str):
            Output file name to save "zarr" or "db". If None, path to output is
            returned by the engine.
        patch_input_shape (tuple):
            Shape of patches input to the model as tuple of height and width (HW).
            Patches are requested at read resolution, not with respect to level 0,
            and must be positive.
        input_resolutions (list(dict(Units, Resolution)))::
            Resolution used for reading the image. Please see
            :class:`WSIReader` for details.
        return_probabilities (bool):
                Whether to return per-class probabilities.
        scale_factor (tuple[float, float]):
            The scale factor to use when loading the
            annotations. All coordinates will be multiplied by this factor to allow
            conversion of annotations saved at non-baseline resolution to baseline.
            Should be model_mpp/slide_mpp.
        stride_shape (tuple):
            Stride used during WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        verbose (bool):
            Whether to output logging information.

    """

    patch_output_shape: tuple
    output_resolutions: Resolution


class SemanticSegmentor(PatchPredictor):
    r"""Semantic Segmentor Engine for processing digital histology images.

    The tiatoolbox model should produce the following results on the BCSS dataset
    using fcn_resnet50_unet-bcss.

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
            A PyTorch model or name of pretrained model.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter. Default is `None`.
        batch_size (int):
            Number of image patches fed into the model each time in a
            forward/backward pass. Default value is 8.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. Default value is 0.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
            Default value is 0.
        weights (str or Path):
            Path to the weight of the corresponding `model`.

            >>> engine = SemanticSegmentor(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default is "cpu".
        verbose (bool):
            Whether to output logging information. Default value is False.

    Attributes:
        images (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A list of image patches in NHWC format as a numpy array
            or a list of str/paths to WSIs.
        masks (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A list of tissue masks or binary masks corresponding to processing area of
            input images. These can be a list of numpy arrays or paths to
            the saved image masks. These are only utilized when patch_mode is False.
            Patches are only generated within a masked area.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (str):
            Whether to treat input images as a set of image patches. TIAToolbox defines
            an image as a patch if HWC of the input image matches with the HWC expected
            by the model. If HWC of the input image does not match with the HWC expected
            by the model, then the patch_mode must be set to False which will allow the
            engine to extract patches from the input image.
            In this case, when the patch_mode is False the input images are treated
            as WSIs. Default value is True.
        model (str | ModelABC):
            A PyTorch model or a name of an existing model from the TIAToolbox model zoo
            for processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument
            is case-insensitive.
        ioconfig (IOSegmentorConfig):
            Input IO configuration of type :class:`IOSegmentorConfig` to run the Engine.
        _ioconfig (IOSegmentorConfig):
            Runtime ioconfig.
        return_labels (bool):
            Whether to return the labels with the predictions.
        input_resolutions (Resolution):
            Resolution used for reading the image. Please see
            :obj:`WSIReader` for details.
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `level`, `power` or `mpp`. Please see
            :obj:`WSIReader` for details.
        patch_input_shape (tuple):
            Shape of patches input to the model as tupled of HW. Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (tuple):
            Stride used during WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        batch_size (int):
            Number of images fed into the model each time.
        cache_mode (bool):
            Whether to run the Engine in cache_mode. For large datasets,
            we recommend to set this to True to avoid out of memory errors.
            For smaller datasets, the cache_mode is set to False as
            the results can be saved in memory. cache_mode is always True when
            processing WSIs i.e., when `patch_mode` is False. Default value is False.
        cache_size (int):
            Specifies how many image patches to process in a batch when
            cache_mode is set to True. If cache_size is less than the batch_size
            batch_size is set to cache_size. Default value is 10,000.
        labels (list | None):
                List of labels. Only a single label per image is supported.
        device (str):
            :class:`torch.device` to run the model.
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default value is "cpu".
        num_loader_workers (int):
            Number of workers used in :class:`torch.utils.data.DataLoader`.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
        return_labels (bool):
            Whether to return the output labels. Default value is False.
        input_resolutions (Resolution):
            Resolution used for reading the image. Please see
            :class:`WSIReader` for details.
            When `patch_mode` is True, the input image patches are expected to be at
            the correct resolution and units. When `patch_mode` is False, the patches
            are extracted at the requested resolution and units. Default value is 1.0.
        output_locations (list | None):
            A list of coordinates in `[start_x, start_y, end_x, end_y]` format to be
            used for patch extraction.
        verbose (bool):
            Whether to output logging information. Default value is False.

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
        """Initialize :class:`SemanticSegmentor`."""
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
        """Pre-process images and masks and return dataloader for inference.

        Args:
            images (list of str or :class:`Path` or :class:`numpy.ndarray`):
                A list of image patches in NHWC format as a numpy array
                or a list of str/paths to WSIs. When `patch_mode` is False
                the function expects list of str/paths to WSIs.
            masks (list | None):
                List of masks. Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            labels (list | None):
                List of labels. Only a single label per image is supported.
            ioconfig (ModelIOConfigABC):
                A :class:`ModelIOConfigABC` object.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.

        Returns:
            torch.utils.data.DataLoader:
                :class:`torch.utils.data.DataLoader` for inference.

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

    def post_process_cache_mode(
        self: SemanticSegmentor,
        raw_predictions: Path,
        **kwargs: Unpack[PredictorRunParams],
    ) -> Path:
        """Returns an array from raw predictions.

        Merges raw predictions from individual patches into a single prediction array if
        patch_mode is False.

        """
        if self.patch_mode:
            return super().post_process_cache_mode(
                raw_predictions=raw_predictions,
                **kwargs,
            )

        progress_bar = None
        tqdm = get_tqdm()

        if self.verbose:
            progress_bar = tqdm(
                total=len(self.output_locations),
                leave=False,
                desc="Merging Patch Outputs",
            )

        return_probabilities = kwargs.get("return_probabilities", False)
        num_post_proc_workers = self.num_post_proc_workers

        if num_post_proc_workers is not None and num_post_proc_workers > 0:
            dask.config.set(scheduler="threads", num_workers=num_post_proc_workers)
        else:
            dask.config.set(scheduler="threads")

        dask_patch_probabilities = da.from_zarr(
            url=str(raw_predictions), component="probabilities"
        )

        # --- Calculate canvas parameters from Dask array and locations ---
        max_location = np.max(self.output_locations, axis=0)
        merged_shape = (
            max_location[3],
            max_location[2],
            dask_patch_probabilities.shape[3],
        )

        # creating dask arrays for faster processing
        merged_probabilities = da.zeros(
            shape=merged_shape,
            dtype=dask_patch_probabilities.dtype,
            chunks=merged_shape,
        )

        merged_weights = da.zeros(
            shape=merged_shape,
            dtype=int,
            chunks=merged_shape,
        )

        for idx, location in enumerate(self.output_locations):
            start_x, start_y, end_x, end_y = location
            patch_probs = dask_patch_probabilities[
                idx, 0 : end_y - start_y, 0 : end_x - start_x, :
            ]
            merged_probabilities[start_y:end_y, start_x:end_x, :] = (
                merged_probabilities[start_y:end_y, start_x:end_x, :] + patch_probs
            )
            merged_weights[start_y:end_y, start_x:end_x] = (
                merged_weights[start_y:end_y, start_x:end_x] + 1
            )
            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        # Normalize where weight > 1
        final_probabilities_dask = da.where(
            merged_weights > 1,
            merged_probabilities / merged_weights,
            merged_probabilities,
        )

        # Applying Post-Processing
        final_predictions_dask = self.model.postproc_func(
            final_probabilities_dask,
        )

        zarr_group = zarr.open_group(
            str(raw_predictions),
            mode="r+",
        )  # Open in read/write mode

        # save merged probabilities as single output probabilities
        logger.info("Saving raw predictions to array.")

        write_predictions = da.to_zarr(
            final_predictions_dask,
            url=zarr_group.store,  # Use the underlying store object of the Zarr group
            component="predictions",
            compute=False,
        )

        if return_probabilities:
            write_probabilities = da.to_zarr(
                final_probabilities_dask,
                url=zarr_group.store,
                component="merged_probabilities",
                compute=False,
            )
            compute(write_probabilities, write_predictions)
            del zarr_group["probabilities"]
            zarr.storage.rename(
                zarr_group.store, "merged_probabilities", "probabilities"
            )
            return raw_predictions

        write_predictions.compute()
        del zarr_group["probabilities"]
        return raw_predictions

    def save_predictions(
        self: PatchPredictor,
        processed_predictions: dict | Path,
        output_type: str,
        save_dir: Path | None = None,
        **kwargs: SemanticSegmentorRunParams,
    ) -> dict | AnnotationStore | Path | list[Path]:
        """Save semantic segmentation predictions to disk.

        Args:
            processed_predictions (dict | Path):
                A dictionary or path to zarr with model prediction information.
            save_dir (Path):
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
                processed_predictions, output_type, save_dir, **kwargs
            )

        logger.info("Saving predictions as AnnotationStore.")

        save_path = Path(kwargs.get("output_file", save_dir))
        return_probabilities = kwargs.get("return_probabilities", False)

        # scale_factor set from kwargs
        scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
        # class_dict set from kwargs
        class_dict = kwargs.get("class_dict")

        processed_predictions_path: str | Path = None

        # Need to add support for zarr conversion.
        if self.cache_mode:
            processed_predictions_path = processed_predictions
            processed_predictions = zarr.open(processed_predictions, mode="r+")

        save_paths = []

        if self.patch_mode:
            for i, predictions in enumerate(processed_predictions["predictions"]):
                if isinstance(self.images[i], Path):
                    output_path = save_path / (self.images[i].stem + ".db")
                else:
                    output_path = save_path / (str(i) + ".db")

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
                save_path=processed_predictions_path.with_suffix(".db"),
            )
            save_paths = out_file

        processed_predictions.pop("predictions")

        if return_probabilities and self.cache_mode:
            new_zarr_name = out_file.parent.with_suffix(".zarr")
            processed_predictions_path.rename(new_zarr_name)
            msg = (
                f"Probability maps cannot be saved as AnnotationStore. "
                f"To visualise heatmaps in TIAToolbox Visualization tool,"
                f"convert heatmaps in {processed_predictions_path} to ome.tiff using"
                f"tiatoolbox.utils.misc.write_probability_heatmap_as_ome_tiff."
            )
            logger.info(msg)

        if processed_predictions_path and processed_predictions_path.exists():
            shutil.rmtree(processed_predictions_path)

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
