"""Defines PatchClassifier Engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import zarr
from typing_extensions import Unpack

from .engine_abc import EngineABCRunParams
from .patch_predictor import PatchPredictor

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import numpy as np

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import ModelIOConfigABC
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.wsicore import WSIReader


class ClassifierRunParams(EngineABCRunParams):
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
        resolution (Resolution):
            Resolution used for reading the image. Please see
            :class:`WSIReader` for details.
        return_labels (bool):
            Whether to return the output labels.
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

    return_probabilities: bool


class PatchClassifier(PatchPredictor):
    r"""Patch level classifier for digital histology images.

    The models provided by TIAToolbox should give the following results:

    .. list-table:: PatchClassifier performance on the Kather100K dataset [1]
       :widths: 15 15
       :header-rows: 1

       * - Model name
         - F\ :sub:`1`\ score
       * - alexnet-kather100k
         - 0.965
       * - resnet18-kather100k
         - 0.990
       * - resnet34-kather100k
         - 0.991
       * - resnet50-kather100k
         - 0.989
       * - resnet101-kather100k
         - 0.989
       * - resnext50_32x4d-kather100k
         - 0.992
       * - resnext101_32x8d-kather100k
         - 0.991
       * - wide_resnet50_2-kather100k
         - 0.989
       * - wide_resnet101_2-kather100k
         - 0.990
       * - densenet121-kather100k
         - 0.993
       * - densenet161-kather100k
         - 0.992
       * - densenet169-kather100k
         - 0.992
       * - densenet201-kather100k
         - 0.991
       * - mobilenet_v2-kather100k
         - 0.990
       * - mobilenet_v3_large-kather100k
         - 0.991
       * - mobilenet_v3_small-kather100k
         - 0.992
       * - googlenet-kather100k
         - 0.992

    .. list-table:: PatchClassifier performance on the PCam dataset [2]
       :widths: 15 15
       :header-rows: 1

       * - Model name
         - F\ :sub:`1`\ score
       * - alexnet-pcam
         - 0.840
       * - resnet18-pcam
         - 0.888
       * - resnet34-pcam
         - 0.889
       * - resnet50-pcam
         - 0.892
       * - resnet101-pcam
         - 0.888
       * - resnext50_32x4d-pcam
         - 0.900
       * - resnext101_32x8d-pcam
         - 0.892
       * - wide_resnet50_2-pcam
         - 0.901
       * - wide_resnet101_2-pcam
         - 0.898
       * - densenet121-pcam
         - 0.897
       * - densenet161-pcam
         - 0.893
       * - densenet169-pcam
         - 0.895
       * - densenet201-pcam
         - 0.891
       * - mobilenet_v2-pcam
         - 0.899
       * - mobilenet_v3_large-pcam
         - 0.895
       * - mobilenet_v3_small-pcam
         - 0.890
       * - googlenet-pcam
         - 0.867

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

            >>> engine = PatchClassifier(
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
        ioconfig (ModelIOConfigABC):
            Input IO configuration of type :class:`ModelIOConfigABC` to run the Engine.
        _ioconfig (ModelIOConfigABC):
            Runtime ioconfig.
        return_labels (bool):
            Whether to return the labels with the predictions.
        resolution (Resolution):
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
        resolution (Resolution):
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
        >>> data = ['path/img.svs', 'path/img.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(wsi_file, mode='wsi')

    References:
        [1] Kather, Jakob Nikolas, et al. "Predicting survival from colorectal cancer
        histology slides using deep learning: A retrospective multicenter study."
        PLoS medicine 16.1 (2019): e1002730.

        [2] Veeling, Bastiaan S., et al. "Rotation equivariant CNNs for digital
        pathology." International Conference on Medical image computing and
        computer-assisted intervention. Springer, Cham, 2018.

    """

    def __init__(
        self: PatchClassifier,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`PatchClassifier`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def post_process_cache_mode(
        self: PatchClassifier,
        raw_predictions: Path,
        **kwargs: Unpack[ClassifierRunParams],
    ) -> Path:
        """Returns an array from raw predictions."""
        return_probabilities = kwargs.get("return_probabilities")
        zarr_group = zarr.open(str(raw_predictions), mode="r+")

        num_iter = math.ceil(len(zarr_group["probabilities"]) / self.batch_size)
        start = 0
        for _ in range(num_iter):
            # Probabilities for post-processing
            probabilities = zarr_group["probabilities"][start : start + self.batch_size]
            start = start + self.batch_size
            predictions = self.model.postproc_func(
                probabilities,
            )
            if "predictions" in zarr_group:
                zarr_group["predictions"].append(predictions)
                continue

            zarr_dataset = zarr_group.create_dataset(
                name="predictions",
                shape=predictions.shape,
                compressor=zarr_group["probabilities"].compressor,
            )
            zarr_dataset[:] = predictions

        if return_probabilities is not False:
            return raw_predictions

        del zarr_group["probabilities"]

        return raw_predictions

    def post_process_patches(
        self: PatchClassifier,
        raw_predictions: dict | Path,
        **kwargs: Unpack[ClassifierRunParams],
    ) -> dict | Path:
        """Post-process raw patch predictions from inference.

        The output of :func:`infer_patches()` with patch prediction information will be
        post-processed using this function. The processed output will be saved in the
        respective input format. If `cache_mode` is True, the function processes the
        input using zarr group with size specified by `cache_size`.

        Args:
            raw_predictions (dict | Path):
                A dictionary or path to zarr with patch prediction information.
            **kwargs (ClassifierRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`ClassifierRunParams` for accepted keyword arguments.

        Returns:
            dict or Path:
                Returns patch based output after post-processing. Returns path to
                saved zarr file if `cache_mode` is True.

        """
        return_probabilities = kwargs.get("return_probabilities")
        if self.cache_mode:
            return self.post_process_cache_mode(raw_predictions, **kwargs)

        probabilities = raw_predictions.get("probabilities")

        predictions = self.model.postproc_func(
            probabilities,
        )

        raw_predictions["predictions"] = predictions

        if return_probabilities is not False:
            return raw_predictions

        del raw_predictions["probabilities"]

        return raw_predictions

    def post_process_wsi(
        self: PatchClassifier,
        raw_predictions: dict | Path,
        **kwargs: Unpack[ClassifierRunParams],
    ) -> dict | Path:
        """Post process WSI output.

        Takes the raw output from patch predictions and post-processes it to improve the
        results e.g., using information from neighbouring patches.

        """
        return self.post_process_cache_mode(raw_predictions, **kwargs)

    def run(
        self: PatchClassifier,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[ClassifierRunParams],
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
            ioconfig (IOPatchPredictorConfig):
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
            **kwargs (ClassifierRunParams):
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
            >>> class PatchClassifier(PatchPredictor):
            >>> # Define all Abstract methods.
            >>>     ...
            >>> classifier = PatchClassifier(model="resnet18-kather100k")
            >>> output = classifier.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"
            >>> output = classifier.run(
            >>>     image_patches,
            >>>     patch_mode=True,
            >>>     output_type="zarr")
            >>> output
            ... "/path/to/Output.zarr"
            >>> output = classifier.run(wsis, patch_mode=False)
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
