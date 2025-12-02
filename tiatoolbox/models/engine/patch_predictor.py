"""Defines the PatchPredictor engine for patch-level inference in digital pathology.

This module implements the PatchPredictor class, which extends the EngineABC base
class to support patch-based and whole slide image (WSI) inference using deep learning
models from TIAToolbox. It provides utilities for model initialization, post-processing,
and output management, including support for multiple output formats.

Classes:
    - PatchPredictor:
        Engine for performing patch-level predictions.
    - PredictorRunParams:
        TypedDict for configuring runtime parameters.

Example:
    >>> images = [np.ndarray, np.ndarray]
    >>> predictor = PatchPredictor(model="resnet18-kather100k")
    >>> output = predictor.run(images, patch_mode=True)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

from tiatoolbox.utils.misc import cast_to_min_dtype

from .engine_abc import EngineABC, EngineABCRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from pathlib import Path

    import dask.array as da
    import numpy as np

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.engine.io_config import ModelIOConfigABC
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader


class PredictorRunParams(EngineABCRunParams, total=False):
    """Parameters for configuring the `PatchPredictor.run()` method.

    This class extends `EngineABCRunParams` with additional parameters specific
    to patch-level prediction workflows.

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
        return_labels (bool):
            Whether to return labels with predictions.
        return_probabilities (bool):
            Whether to return per-class probabilities in the output.
            If False, only predicted labels are returned.
        scale_factor (tuple[float, float]):
            Scale factor for converting annotations to baseline resolution.
            Typically model_mpp / slide_mpp.
        stride_shape (tuple[int, int]):
            Stride used during WSI processing. Defaults to patch_input_shape.
        verbose (bool):
            Whether to output logging information.

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
        num_workers (int):
            Number of workers for data loading. Default is 0.
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
        labels (list | None):
            Optional labels for input images.
            Only a single label per image is supported.
        drop_keys (list):
            Keys to exclude from model output.
        output_type (str):
            Format of output ("dict", "zarr", "annotationstore").

    Example:
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
        num_workers: int = 0,
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
            num_workers (int):
                Number of workers for data loading. Default is 0.
            weights (str | Path | None): Path to model weights.
                If None, default weights are used.
            device (str):
                device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Whether to enable verbose logging. Default is True.

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
        self: PatchPredictor,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[PredictorRunParams],
    ) -> da.Array:
        """Post-process raw patch predictions from model inference.

        This method applies the model's post-processing function to the raw predictions
        obtained from `infer_patches()`. The output is wrapped in a Dask array for
        efficient computation and memory handling.

        Args:
            raw_predictions (da.Array | np.ndarray):
                Raw model predictions.
            prediction_shape (tuple[int, ...]):
                Expected shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (PredictorRunParams):
                Additional runtime parameters to configure prediction.

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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities in the output.
                        If False, only predicted labels are returned.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            dask.array.Array: Post-processed predictions as a Dask array.

        """
        _ = kwargs.get("return_probabilities")
        _ = prediction_shape
        _ = prediction_dtype
        raw_predictions = self.model.postproc_func(raw_predictions)
        return cast_to_min_dtype(raw_predictions)

    def post_process_wsi(
        self: PatchPredictor,
        raw_predictions: da.Array,
        prediction_shape: tuple[int, ...],
        prediction_dtype: type,
        **kwargs: Unpack[PredictorRunParams],
    ) -> da.Array:
        """Post-process predictions from whole slide image (WSI) inference.

        This method refines the raw patch-level predictions obtained from WSI inference.
        It typically applies spatial smoothing or other contextual operations using
        neighboring patch information. Internally, it delegates to
        `post_process_patches()`.

        Args:
            raw_predictions (dask.array.Array):
                Raw model predictions.
            prediction_shape (tuple[int, ...]):
                Expected shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (PredictorRunParams):
                Additional runtime parameters to configure prediction.

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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities in the output.
                        If False, only predicted labels are returned.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            dask.array.Array: Post-processed predictions as a Dask array.

        """
        return self.post_process_patches(
            raw_predictions=raw_predictions,
            prediction_shape=prediction_shape,
            prediction_dtype=prediction_dtype,
            **kwargs,
        )

    def _update_run_params(
        self: PatchPredictor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        save_dir: os.PathLike | Path | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[PredictorRunParams],
    ) -> Path | None:
        """Update runtime parameters for the PatchPredictor engine.

        This method sets internal attributes such as caching, batch size,
        IO configuration, and output format based on user input and keyword arguments.
        It also configures whether to include probabilities in the output.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            **kwargs (PredictorRunParams):
                Additional runtime parameters to configure prediction.

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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities in the output.
                        If False, only predicted labels are returned.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            Path | None:
                Path to the save directory if applicable, otherwise None.

        """
        return_probabilities = kwargs.get("return_probabilities")
        if not return_probabilities:
            self.drop_keys.append("probabilities")
        return super()._update_run_params(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            save_dir=save_dir,
            ioconfig=ioconfig,
            overwrite=overwrite,
            patch_mode=patch_mode,
            output_type=output_type,
            **kwargs,
        )

    def run(
        self: PatchPredictor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        *,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[PredictorRunParams],
    ) -> AnnotationStore | Path | str | dict:
        """Run the PatchPredictor engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both patch
        and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. When using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing.
                Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
                Default value is "zarr".
            **kwargs (PredictorRunParams):
                Additional runtime parameters to configure prediction.

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
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities in the output.
                        If False, only predicted labels are returned.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            AnnotationStore | Path | str | dict:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI to
                  its output path.

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
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )
