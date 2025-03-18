"""Defines SemanticSegmentor Engine."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import zarr
from typing_extensions import Unpack

from .patch_predictor import PatchPredictor, PredictorRunParams
from ...utils.misc import dict_to_zarr, dict_to_store_semantic_segmentor

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import numpy as np

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import Resolution
    from tiatoolbox.wsicore import WSIReader


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
        input_resolutions (Resolution):
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
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `level`, `power` or `mpp`. Please see
            :class:`WSIReader` for details.
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
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `baseline`, `level`, `power` or `mpp`. Please see
            :class:`WSIReader` for details.
            When `patch_mode` is True, the input image patches are expected to be at
            the correct resolution and units. When `patch_mode` is False, the patches
            are extracted at the requested resolution and units.
            Default value is `baseline`.
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

    def save_predictions(
        self: PatchPredictor,
        processed_predictions: dict | Path,
        output_type: str,
        save_dir: Path | None = None,
        **kwargs: SemanticSegmentorRunParams,
    ) -> dict | AnnotationStore | Path:
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
        if (
            self.cache_mode or not save_dir
        ) and output_type.lower() != "annotationstore":
            return processed_predictions

        save_path = Path(kwargs.get("output_file", save_dir / "output.db"))

        if output_type.lower() == "annotationstore":
            # scale_factor set from kwargs
            scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
            # class_dict set from kwargs
            class_dict = kwargs.get("class_dict")

            processed_predictions_path: str | Path | None = None

            # Need to add support for zarr conversion.
            if self.cache_mode:
                processed_predictions_path = processed_predictions
                processed_predictions = zarr.open(processed_predictions, mode="r")

            out_file = dict_to_store_semantic_segmentor(
                processed_predictions,
                scale_factor,
                class_dict,
                save_path,
            )
            if processed_predictions_path is not None:
                shutil.rmtree(processed_predictions_path)

            return out_file

        return (
            dict_to_zarr(
                processed_predictions,
                save_path,
                **kwargs,
            )
            if isinstance(processed_predictions, dict)
            else processed_predictions
        )

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
    ) -> AnnotationStore | Path | str | dict:
        """Run the engine on input images.

        Args:
            images (list, ndarray):
                List of inputs to process. when using `patch` mode, the
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
            (:class:`numpy.ndarray`, dict):
                Model predictions of the input dataset. If multiple
                whole slide images are provided as input,
                or save_output is True, then results are saved to
                `save_dir` and a dictionary indicating save location for
                each input is returned.

                The dict has the following format:

                - img_path: path of the input image.
                - raw: path to save location for raw prediction,
                  saved in .json.

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
