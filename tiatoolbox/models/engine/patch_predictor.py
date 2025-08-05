"""Defines PatchPredictor Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
from dask import delayed
from typing_extensions import Unpack

from .engine_abc import EngineABC, EngineABCRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import numpy as np

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import ModelIOConfigABC
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.wsicore import WSIReader


class PredictorRunParams(EngineABCRunParams, total=False):
    """Parameters for configuring the `PatchPredictor.run()` method.

    This class extends `EngineABCRunParams` with additional parameters specific
    to patch-level prediction workflows.

    Optional Keys:
        return_probabilities (bool):
            Whether to return per-class probabilities
            in the output. If False, only predicted labels are returned.

    """

    return_probabilities: bool


class PatchPredictor(EngineABC):
    r"""Patch-level prediction engine for digital histology images.

    This class extends `EngineABC` to support patch-based inference using
    pretrained or custom models from TIAToolbox. It supports both patch and
    whole slide image (WSI) modes, and provides utilities for post-processing
    and saving predictions.

    Supported Models:
        .. list-table:: PatchPredictor performance on the Kather100K dataset [1].
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

        .. list-table:: PatchPredictor performance on the PCam dataset [2]
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
            A PyTorch model instance or name of a pretrained model from TIAToolbox.
            If a string is provided, pretrained weights
            will be downloaded unless overridden via `weights`.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded.
        batch_size (int):
            Number of image patches processed per forward pass.
            Default is 8.
        num_loader_workers (int):
            Number of workers for data loading. Default is 0.
        num_post_proc_workers (int):
            Number of workers for post-processing. Default is 0.
        weights (str | Path | None):
            Path to model weights. If None, default weights are used.

            >>> engine = PatchPredictor(
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

    Examples:
        >>> # list of 2 image patches as input
        >>> data = ['path/img.svs', 'path/img.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, patch_mode=False)

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, patch_mode=True)

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, patch_mode=True)

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(tile_file, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(wsi_file, patch_mode=False)

    References:
        [1] Kather, Jakob Nikolas, et al. "Predicting survival from colorectal cancer
        histology slides using deep learning: A retrospective multicenter study."
        PLoS medicine 16.1 (2019): e1002730.

        [2] Veeling, Bastiaan S., et al. "Rotation equivariant CNNs for digital
        pathology." International Conference on Medical image computing and
        computer-assisted intervention. Springer, Cham, 2018.

    """

    def __init__(
        self: PatchPredictor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize the PatchPredictor engine.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or name of a pretrained model from TIAToolbox.
                If a string is provided, the corresponding pretrained
                weights will be downloaded unless overridden via `weights`.
            batch_size (int):
                Number of image patches processed per forward pass. Default is 8.
            num_loader_workers (int):
                Number of workers for data loading. Default is 0.
            num_post_proc_workers (int):
                Number of workers for post-processing. Default is 0.
            weights (str | Path | None): Path to model weights.
                If None, default weights are used.
            device (str): D
                device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
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

    def post_process_patches(
        self: PatchPredictor,
        raw_predictions: dict | Path,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[PredictorRunParams],
    ) -> dict | Path:
        """Post-process raw patch predictions from inference.

        The output of :func:`infer_patches()` with patch prediction information will be
        post-processed using this function. The processed output will be saved in the
        respective input format. If `cache_mode` is True, the function processes the
        input using zarr group with size specified by `cache_size`.

        Args:
            raw_predictions (dict | Path):
                A dictionary or path to zarr with patch prediction information.
            prediction_shape (tuple (int, ...)):
                prediction shape.
            prediction_dtype (tuple (int, ...)):
                prediction dtype.
            **kwargs (EngineABCRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            dict or Path:
                Returns patch based output after post-processing. Returns path to
                saved zarr file if `cache_mode` is True.

        """
        _ = kwargs.get("return_probabilities")

        raw_predictions = delayed(self.model.postproc_func)(
            raw_predictions,
        )

        return da.from_delayed(
            raw_predictions,
            shape=prediction_shape,
            dtype=prediction_dtype,
        )

    def post_process_wsi(
        self: PatchPredictor,
        raw_predictions: dict | Path,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[PredictorRunParams],
    ) -> dict | Path:
        """Post process WSI output.

        Takes the raw output from patch predictions and post-processes it to improve the
        results e.g., using information from neighbouring patches.

        Args:
            raw_predictions (dict | Path):
                A dictionary or path to zarr with patch prediction information.
            prediction_shape (tuple (int, ...)):
                prediction shape.
            prediction_dtype (tuple (int, ...)):
                prediction dtype.
            **kwargs (EngineABCRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            dict or Path:
                Returns patch based output after post-processing. Returns path to
                saved zarr file if `cache_mode` is True.

        """
        return self.post_process_patches(
            raw_predictions=raw_predictions,
            prediction_shape=prediction_shape,
            prediction_dtype=prediction_dtype,
            **kwargs,
        )

    def _update_run_params(
        self: EngineABC,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        save_dir: os | Path | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[PredictorRunParams],
    ) -> Path | None:
        """Updates runtime parameters.

        Updates runtime parameters for an EngineABC for EngineABC.run().

        """
        return_probabilities = kwargs.get("return_probabilities")
        if not return_probabilities:
            self.drop_keys.append("probabilities")

        return super()._update_run_params(
            images=images,
            masks=masks,
            labels=labels,
            save_dir=save_dir,
            ioconfig=ioconfig,
            overwrite=overwrite,
            patch_mode=patch_mode,
            output_type=output_type,
            **kwargs,
        )

    def run(
        self: PatchPredictor,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[PredictorRunParams],
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
            **kwargs (PredictorRunParams):
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
            >>> class PatchPredictor(EngineABC):
            >>> # Define all Abstract methods.
            >>>     ...
            >>> predictor = PatchPredictor(model="resnet18-kather100k")
            >>> output = predictor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"
            >>> output = predictor.run(
            >>>     image_patches,
            >>>     patch_mode=True,
            >>>     output_type="zarr")
            >>> output
            ... "/path/to/Output.zarr"
            >>> output = predictor.run(wsis, patch_mode=False)
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
