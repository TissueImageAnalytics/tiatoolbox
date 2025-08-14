"""Abstract Base Class for TIAToolbox Deep Learning Engines.

This module defines the `EngineABC` class, which serves as a base for implementing
deep learning inference workflows in TIAToolbox. It supports both patch-based and
whole slide image (WSI) processing, and provides a unified interface for model
initialization, data loading, inference, post-processing, and output saving.

Classes:
    - EngineABC: Abstract base class for deep learning engines.
    - EngineABCRunParams: TypedDict for runtime configuration parameters.

Functions:
    - prepare_engines_save_dir: Utility to create or validate output directories.

Features:
    - Supports patch and WSI modes.
    - Handles caching and memory-efficient inference using Dask.
    - Integrates with TIAToolbox models and IO configurations.
    - Outputs predictions in multiple formats including dict, zarr, and AnnotationStore.

Intended Usage:
    Subclass `EngineABC` to implement specific inference logic by overriding
    abstract methods such as preprocessing, postprocessing, and model-specific behavior.

Example:
    >>> class MyEngine(EngineABC):
    >>>     def __init__(self, model, weights, verbose):
    >>>         super().__init__(model=model, weights=weights, verbose=verbose)
    >>> # Implement base class functions and then call.
    >>> engine = MyEngine(model="resnet18-kather100k")
    >>> output = engine.run(images, patch_mode=True)

"""

from __future__ import annotations

import copy
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import dask
import dask.array as da
import numpy as np
import torch
from dask import compute
from dask.diagnostics import ProgressBar
from torch import nn
from typing_extensions import Unpack

from tiatoolbox import DuplicateFilter, logger, rcParam
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model
from tiatoolbox.models.dataset.dataset_abc import PatchDataset, WSIPatchDataset
from tiatoolbox.models.models_abc import load_torch_model
from tiatoolbox.utils.misc import (
    dict_to_store_patch_predictions,
    get_tqdm,
)

from .io_config import ModelIOConfigABC

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore.wsireader import WSIReader


def prepare_engines_save_dir(
    save_dir: str | Path | None,
    *,
    patch_mode: bool,
    overwrite: bool = False,
) -> Path | None:
    """Create or validate the save directory for engine outputs.

    Args:
        save_dir (str | Path | None):
            Path to the output directory.
        patch_mode (bool):
            Whether the input is treated as patches.
        overwrite (bool):
            Whether to overwrite existing directory. Default is False.

    Returns:
        Path | None:
            Path to the output directory if created or validated, else None.

    Raises:
        OSError:
            If patch_mode is False and save_dir is not provided.

    """
    if patch_mode:
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=overwrite)
            return save_dir
        return None

    if save_dir is None:
        msg = (
            "Input WSIs detected but no save directory provided. "
            "Please provide a 'save_dir'."
        )
        raise OSError(msg)

    logger.info(
        "When providing multiple whole slide images, "
        "the outputs will be saved and the locations of outputs "
        "will be returned to the calling function when `run()` "
        "finishes successfully."
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=overwrite)

    return save_dir


class EngineABCRunParams(TypedDict, total=False):
    """Parameters for configuring the :func:`EngineABC.run()` method.

    Optional Keys:
        batch_size (int):
            Number of image patches per forward pass.
        class_dict (dict):
            Mapping of classification outputs to class names.
        device (str):
            Device to run the model on (e.g., "cpu", "cuda"). Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device.
        ioconfig (ModelIOConfigABC):
            IO configuration (:class:`ModelIOConfigABC`) for model input/output.
        return_labels (bool):
            Whether to return labels with predictions.
        num_loader_workers (int):
            Number of workers for DataLoader.
        num_post_proc_workers (int):
            Number of workers for post-processing.
        output_file (str):
            Filename for saving output (e.g., .zarr or .db).
        patch_input_shape (IntPair):
            Shape of input patches (height, width).
            Patches are requested at read resolution, not with respect to level 0,
            and must be positive.
        input_resolutions (list[dict[Units, Resolution]]):
            Resolution settings for input heads. Supported
            units are `level`, `power` and `mpp`. Keys should be "units" and
            "resolution" e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
            :class:`WSIReader` for details.
        scale_factor (tuple[float, float]):
            Scale factor for annotations (model_mpp / slide_mpp). All coordinates
            are multiplied by this factor to allow conversion of annotations
            saved at non-baseline resolution to baseline. Should be model_mpp/slide_mpp.
        stride_shape (IntPair):
            Stride used during WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        verbose (bool):
            Whether to enable verbose logging.

    """

    batch_size: int
    class_dict: dict
    device: str
    ioconfig: ModelIOConfigABC
    num_loader_workers: int
    num_post_proc_workers: int
    output_file: str
    patch_input_shape: IntPair
    input_resolutions: list[dict[Units, Resolution]]
    return_labels: bool
    scale_factor: tuple[float, float]
    stride_shape: IntPair
    verbose: bool


class EngineABC(ABC):  # noqa: B024
    """Abstract base class for TIAToolbox deep learning engines to run CNN models.

    This class provides a unified interface for running inference on image patches
    or whole slide images (WSIs), handling preprocessing, batching, postprocessing,
    and saving predictions.

    Args:
        model (str | ModelABC):
            Model name from TIAToolbox or a PyTorch model instance.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter.
        batch_size (int):
            Number of patches per forward pass. Default is 8.
        num_loader_workers (int):
            Number of workers for data loading. Default is 0.
        num_post_proc_workers (int):
            Number of workers for post-processing. Default is 0.
        weights (str | Path | None):
            Path to model weights. If None, default weights are used.

            >>> engine = EngineABC(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Device to run the model on (e.g., "cpu", "cuda"). Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default is "cpu".
        verbose (bool):
            Enable verbose logging. Default is False.

    Attributes:
        images (list[str | Path] | np.ndarray):
            Input images or patches.
            A list of image patches in NHWC format as a numpy array
            or a list of str/paths to WSIs.
        masks (list[str | Path] | np.ndarray):
            Optional masks for WSIs.
            A list of tissue masks or binary masks corresponding to processing area of
            input images. These can be a list of numpy arrays or paths to
            the saved image masks. These are only utilized when patch_mode is False.
            Patches are only generated within a masked area.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (bool):
            Whether input is treated as patches. TIAToolbox defines
            an image as a patch if HWC of the input image matches with the HWC expected
            by the model. If HWC of the input image does not match with the HWC expected
            by the model, then the patch_mode must be set to False which will allow the
            engine to extract patches from the input image.
            In this case, when the patch_mode is False the input images are treated
            as WSIs. Default value is True.
        model (ModelABC):
            Loaded PyTorch model. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument
            is case-insensitive.
        ioconfig (ModelIOConfigABC):
            IO configuration (:class:`ModelIOConfigABC`) for model input/output.
        dataloader (DataLoader):
            Torch DataLoader for inference.
        return_labels (bool):
            Whether to return labels with probabilities and predictions.
        input_resolutions (list[dict[Units, Resolution]]):
            Resolution settings for input heads. Supported
            units are `level`, `power` and `mpp`. Keys should be "units" and
            "resolution" e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
            :class:`WSIReader` for details.
        patch_input_shape (IntPair):
            Shape of input patches. Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (IntPair):
            Stride used during WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        batch_size (int):
            Number of images fed into the model each time.
        labels (list | None):
            Optional labels for input images.
            Only a single label per image is supported.
        drop_keys (list):
            Keys to exclude from model output.
        output_type (str):
            Format of output ("dict", "zarr", "AnnotationStore").
        verbose (bool):
            Whether to enable verbose logging.

    Example:
        >>> # Inherit from EngineABC
        >>> class MyEngine(EngineABC):
        >>>     def __init__(self, model, weights, verbose):
        >>>         super().__init__(model=model, weights=weights, verbose=verbose)
        >>> engine = MyEngine(model="resnet18-kather100k")
        >>> output = engine.run(images, patch_mode=True)

        >>> # Define all the abstract classes

        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = TestEngineABC(model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # array of list of 2 image patches as input
        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = TestEngineABC(model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # list of 2 image files as input
        >>> image = ['path/image1.png', 'path/image2.png']
        >>> engine = TestEngineABC(model="resnet18-kather100k")
        >>> output = engine.run(image, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> engine = TestEngineABC(model="resnet18-kather100k")
        >>> output = engine.run(wsi_file, patch_mode=False)

    """

    def __init__(
        self: EngineABC,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        """Initialize the EngineABC instance.

        Args:
            model (str | ModelABC):
                Model name from TIAToolbox or a PyTorch model instance.
            batch_size (int):
                Number of patches per forward pass. Default is 8.
            num_loader_workers (int):
                Number of workers for data loading. Default is 0.
            num_post_proc_workers (int):
                Number of workers for post-processing. Default is 0.
            weights (str | Path | None):
                Path to model weights. If None, default weights are used.
            device (str):
                Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Enable verbose logging. Default is False.

        """
        self.images = None
        self.masks = None
        self.patch_mode = None
        self.device = device

        # Initialize model with specified weights and ioconfig.
        self.model, self.ioconfig = self._initialize_model_ioconfig(
            model=model, weights=weights
        )
        self.model.to(device=self.device)
        self.model = (
            compile_model(  # for runtime, such as after wrapping with nn.DataParallel
                self.model,
                mode=rcParam["torch_compile_mode"],
            )
        )
        self._ioconfig = self.ioconfig  # runtime ioconfig
        self.batch_size = batch_size
        self.labels: list | None = None
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.patch_input_shape: IntPair | None = None
        self.input_resolutions: list[dict[Units, Resolution]] | None = None
        self.return_labels: bool = False
        self.stride_shape: IntPair | None = None
        self.verbose: bool = verbose
        self.dataloader: DataLoader | None = None
        self.drop_keys: list = []
        self.output_type = None

    @staticmethod
    def _initialize_model_ioconfig(
        model: str | ModelABC,
        weights: str | Path | None,
    ) -> tuple[nn.Module, ModelIOConfigABC | None]:
        """Helper function to initialize model and IO configuration.

        If a pretrained model from TIAToolbox is specified by name, this function
        loads the model and its associated IO configuration. If a custom model is
        provided, it loads the weights if specified and returns None for IO config.

        Args:
            model (str | ModelABC):
                A model name from TIAToolbox or a PyTorch model instance.
                The user can request pretrained models from the toolbox model zoo using
                the list of pretrained models available at this `link
                <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
                By default, the corresponding pretrained weights will also
                be downloaded. However, you can override with your own set
                of weights using the `weights` parameter.

            weights (str | Path | None):
                Path to pretrained weights. If None and a TIAToolbox model is used,
                default weights are automatically downloaded.

        Returns:
            tuple[nn.Module, ModelIOConfigABC | None]:
                A tuple containing the loaded PyTorch model and its IO configuration.
                If the model is not from TIAToolbox, IO config will be None.

        Raises:
            TypeError:
                If the model is neither a string (TIAToolbox model)
                nor a torch.nn.Module.

        """
        if not isinstance(model, (str, nn.Module)):
            msg = "Input model must be a string or 'torch.nn.Module'."
            raise TypeError(msg)

        if isinstance(model, str):
            # ioconfig is retrieved from the pretrained model in the toolbox.
            # list of pretrained models in the TIA Toolbox is available here:
            # https://tia-toolbox.readthedocs.io/en/latest/pretrained.html
            # no need to provide ioconfig in EngineABC.run() this case.
            return get_pretrained_model(model, weights)

        if weights is not None:
            model = load_torch_model(model=model, weights=weights)

        return model, None

    def get_dataloader(
        self: EngineABC,
        images: str | Path | list[str | Path] | np.ndarray,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return a DataLoader for inference.

        Args:
            images (list[str | Path] | np.ndarray):
                A list of image patches in NHWC format as a numpy array,
                or a list of file paths to WSIs. When `patch_mode` is False,
                expects file paths to WSIs.
            masks (Path | None):
                Optional list of masks used when `patch_mode` is False.
                Patches are generated only within masked areas. If not provided,
                tissue masks are automatically generated.
            labels (list | None):
                Optional list of labels. Only one label per image is supported.
            ioconfig (ModelIOConfigABC | None):
                IO configuration object specifying patch size, stride, and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).

        Returns:
            torch.utils.data.DataLoader:
                A PyTorch DataLoader configured for inference.

        """
        if labels:
            # if a labels is provided, then return with the prediction
            self.return_labels = bool(labels)

        if not patch_mode:
            dataset = WSIPatchDataset(
                img_path=images,
                mask_path=masks,
                patch_input_shape=ioconfig.patch_input_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )

            dataset.preproc_func = self.model.preproc_func

            # preprocessing must be defined with the dataset
            return torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_loader_workers,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                persistent_workers=self.num_loader_workers > 0,
            )

        dataset = PatchDataset(
            inputs=images, labels=labels, patch_input_shape=ioconfig.patch_input_shape
        )

        dataset.preproc_func = self.model.preproc_func

        # preprocessing must be defined with the dataset
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_workers,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def _get_coordinates(self: EngineABC, batch_data: dict) -> np.ndarray:
        """Extract coordinates for each image patch in a batch.

        This method returns coordinates for each patch, either based on
        the patch dimensions (if in patch mode) or from precomputed values
        (if in WSI mode).

        Args:
            batch_data (dict):
                Dictionary containing batch data, including image and
                optional coordinates.

        Returns:
            np.ndarray:
                Array of coordinates for each patch in the batch.
                Shape is (N, 4), where N is the number of patches.

        """
        if self.patch_mode:
            coordinates = [0, 0, *batch_data["image"].shape[1:3]]
            return np.tile(coordinates, reps=(batch_data["image"].shape[0], 1))
        return np.array(batch_data["coords"])

    def _inference_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        raw_predictions: dict,
    ) -> dict[str, list[da.Array]]:
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

            raw_predictions["probabilities"].append(
                da.from_array(
                    batch_output,  # probabilities
                )
            )

            if "coordinates" in raw_predictions:
                raw_predictions["coordinates"].append(
                    da.from_array(
                        self._get_coordinates(batch_data),
                    )
                )

            if "output_locs" in raw_predictions:
                raw_predictions["output_locs"].append(
                    da.from_array(batch_data["output_locs"].numpy())
                )

            if "labels" in raw_predictions:
                raw_predictions["labels"].append(
                    da.from_array(np.array(batch_data["label"]))
                )

        return raw_predictions

    def infer_patches(
        self: EngineABC,
        dataloader: DataLoader,
        *,
        return_coordinates: bool = False,
    ) -> dict[str, da.Array]:
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
            dict[str, list | dask.array.Array]:
                Dictionary containing prediction results as Dask arrays.
                Keys include:
                    - "probabilities": Model output probabilities.
                    - "labels": Ground truth labels (if `return_labels` is True).
                    - "coordinates": Patch coordinates (if `return_coordinates`
                      is True).

        """
        keys = ["probabilities"]

        if self.return_labels:
            keys.append("labels")

        if return_coordinates:
            keys.append("coordinates")

        # Main output dictionary
        raw_predictions = {key: [] for key in keys}

        raw_predictions = self._inference_loop(dataloader, raw_predictions)

        raw_predictions["probabilities"] = da.concatenate(
            raw_predictions["probabilities"], axis=0
        )

        if return_coordinates:
            raw_predictions["coordinates"] = da.concatenate(
                raw_predictions["coordinates"], axis=0
            )

        if self.return_labels:
            labels = [label.reshape(-1) for label in raw_predictions["labels"]]
            raw_predictions["labels"] = da.concatenate(labels, axis=0)

        return raw_predictions

    def post_process_patches(  # skipcq: PYL-R0201
        self: EngineABC,
        raw_predictions: dict[str, da.Array | np.ndarray],
        prediction_shape: tuple[int, ...],  # noqa: ARG002
        prediction_dtype: type,  # noqa: ARG002
        **kwargs: Unpack[EngineABCRunParams],  # noqa: ARG002
    ) -> dask.array.Array:
        """Post-process raw patch predictions from inference.

        This method applies a post-processing function (e.g., smoothing, filtering)
        to the raw model predictions. It supports delayed execution using Dask
        and returns a Dask array for efficient computation.

        Args:
            raw_predictions (dask.array.Array | np.ndarray):
                Raw model predictions as a Dask array.
            prediction_shape (tuple[int, ...]):
                Shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters used for post-processing.

        Returns:
            dask.array.Array:
                Post-processed predictions as a Dask array.

        """
        return raw_predictions

    def save_predictions(
        self: EngineABC,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> dict | AnnotationStore | Path:
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
                Additional runtime parameters including:
                    - output_file: Name of the output file.
                    - scale_factor: Scaling factor for annotations.
                    - class_dict: Mapping of class indices to names.

        Returns:
            dict | AnnotationStore | Path:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved zarr file.
                - If output_type is "annotationstore":
                  returns an AnnotationStore or path to .db file.

        Raises:
            TypeError:
                If an unsupported output_type is provided.

        """
        keys_to_compute = [k for k in processed_predictions if k not in self.drop_keys]

        if output_type.lower() == "zarr":
            write_tasks = []
            for key in keys_to_compute:
                dask_array = processed_predictions[key]
                task = dask_array.to_zarr(
                    url=save_path,
                    component=key,
                    compute=False,
                    overwrite=True,
                )
                write_tasks.append(task)
            msg = f"Saving output to {save_path}."
            logger.info(msg=msg)
            with ProgressBar():
                compute(*write_tasks)

            return save_path

        values_to_compute = [processed_predictions[k] for k in keys_to_compute]

        # Compute all at once
        computed_values = compute(*values_to_compute)

        # Assign computed values
        processed_predictions = dict(zip(keys_to_compute, computed_values))

        if output_type.lower() == "dict":
            return processed_predictions

        if output_type.lower() == "annotationstore":
            save_path = Path(kwargs.get("output_file", save_path.parent / "output.db"))

            # scale_factor set from kwargs
            scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
            # class_dict set from kwargs
            class_dict = kwargs.get("class_dict")

            return dict_to_store_patch_predictions(
                processed_predictions,
                scale_factor,
                class_dict,
                save_path,
            )

        msg = f"Unsupported output type: {output_type}"
        raise TypeError(msg)

    def infer_wsi(
        self: EngineABC,
        dataloader: DataLoader,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> dict:
        """Run model inference on a whole slide image (WSI).

        This method performs inference on a WSI using the provided DataLoader,
        and accumulates predictions in Dask arrays. Optionally includes
        coordinates and labels in the output.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader configured for WSI processing.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters used during inference.

        Returns:
            dict:
                Dictionary containing prediction results as Dask arrays.

        """
        _ = kwargs.get("patch_mode", False)
        return self.infer_patches(
            dataloader=dataloader,
            return_coordinates=True,
        )

    # This is not a static model for child classes.
    def post_process_wsi(  # skipcq: PYL-R0201
        self: EngineABC,
        raw_predictions: dask.array.Array | np.ndarray,
        prediction_shape: tuple[int, ...],  # noqa: ARG002
        prediction_dtype: type,  # noqa: ARG002
        **kwargs: Unpack[EngineABCRunParams],  # noqa: ARG002
    ) -> dask.array.Array:
        """Post-process predictions from whole slide image (WSI) inference.

        This method applies a post-processing function (e.g., smoothing, filtering)
        to the raw model predictions. It supports delayed execution using Dask
        and returns a Dask array for efficient computation.

        Args:
            raw_predictions (dask.array.Array | np.ndarray):
                Raw model predictions as a Dask array.
            prediction_shape (tuple[int, ...]):
                Shape of the prediction output.
            prediction_dtype (type):
                Data type of the prediction output.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters used for post-processing.

        Returns:
            dask.array.Array:
                Post-processed predictions as a Dask array.

        """
        return raw_predictions

    def _load_ioconfig(self: EngineABC, ioconfig: ModelIOConfigABC) -> ModelIOConfigABC:
        """Load or validate the IO configuration for the engine.

        If the model is from TIAToolbox and no IO configuration is provided,
        this method attempts to use the default configuration. Otherwise,
        it validates and sets the provided configuration.

        Args:
            ioconfig (ModelIOConfigABC):
                IO configuration to use for model inference.

        Returns:
            ModelIOConfigABC:
                The IO configuration to be used during inference.

        Raises:
            ValueError:
                If no IO configuration is provided and none is available from the model.

        """
        if self.ioconfig is None and ioconfig is None:
            msg = (
                "Please provide a valid ModelIOConfigABC. "
                "No default ModelIOConfigABC found."
            )
            logger.warning(msg)

        if ioconfig and isinstance(ioconfig, ModelIOConfigABC):
            self.ioconfig = ioconfig

        return self.ioconfig

    def _update_ioconfig(
        self: EngineABC,
        ioconfig: ModelIOConfigABC,
        patch_input_shape: IntPair,
        stride_shape: IntPair,
        input_resolutions: list[dict[Units, Resolution]],
    ) -> ModelIOConfigABC:
        """Update the IO configuration used for patch-based inference.

        This method updates the patch input shape, stride, and input resolutions
        in the IO configuration. If no configuration is provided, it creates a new one.

        Args:
            ioconfig (ModelIOConfigABC):
                Existing IO configuration to update. If None, a new one is created.

            patch_input_shape (IntPair):
                Size of patches input to the model (height, width). Patches are at
                requested read resolution, not with respect to level 0,
                and must be positive.

            stride_shape (IntPair):
                Stride used during patch extraction.
                If None, defaults to patch_input_shape.
                Stride is at requested read resolution, not with respect to
                level 0, and must be positive. If not provided,
                `stride_shape=patch_input_shape`.

            input_resolutions (list[dict[Units, Resolution]]):
                List of dictionaries specifying resolution and units
                for each input head. Supported units are `level`, `power` and `mpp`.
                Keys should be "units" and "resolution"
                e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
                :class:`WSIReader` for details.

        Returns:
            ModelIOConfigABC:
                Updated IO configuration for patch-based inference.

        Raises:
            ValueError:
                If neither an IO configuration nor patch/resolution parameters
                are provided.

        """
        config_flag = (
            patch_input_shape is None,
            input_resolutions is None,
        )
        if isinstance(ioconfig, ModelIOConfigABC):
            return ioconfig

        if self.ioconfig is None and any(config_flag):
            msg = (
                "Must provide either "
                "`ioconfig` or `patch_input_shape` and `input_resolutions`."
            )
            raise ValueError(
                msg,
            )

        if stride_shape is None:
            stride_shape = patch_input_shape

        if self.ioconfig:
            ioconfig = copy.deepcopy(self.ioconfig)
            # ! not sure if there is a nicer way to set this
            if patch_input_shape is not None:
                ioconfig.patch_input_shape = patch_input_shape
            if stride_shape is not None:
                ioconfig.stride_shape = stride_shape
            if input_resolutions is not None:
                ioconfig.input_resolutions = input_resolutions

            return ioconfig

        return ModelIOConfigABC(
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            stride_shape=stride_shape,
            output_resolutions=[],
        )

    @staticmethod
    def _validate_images_masks(images: list | np.ndarray) -> list | np.ndarray:
        """Validate the format and shape of input images or masks.

        Ensures that the input is either a list of file paths or a 4D NumPy array
        in NHWC format.

        Args:
            images (list | np.ndarray):
                List of image paths or a NumPy array of image patches.

        Returns:
            list | np.ndarray:
                The validated input images or masks.

        Raises:
            TypeError:
                If the input is neither a list nor a NumPy array.

            ValueError:
                If the input is a NumPy array but not 4D (NHWC).

        """
        if not isinstance(images, (list, np.ndarray)):
            msg = "Input must be a list of file paths or a numpy array."
            raise TypeError(
                msg,
            )

        if isinstance(images, np.ndarray) and images.ndim != 4:  # noqa: PLR2004
            msg = (
                "The input numpy array should be four dimensional."
                "The shape of the numpy array should be NHWC."
            )
            raise ValueError(msg)

        return images

    @staticmethod
    def _validate_input_numbers(
        images: list | np.ndarray,
        masks: list[os.PathLike] | np.ndarray | None = None,
        labels: list | None = None,
    ) -> None:
        """Validate that the number of images, masks, and labels match.

        Ensures that the lengths of masks and labels (if provided) match
        the number of input images.

        Args:
            images (list | np.ndarray):
                List of input images or a NumPy array.
            masks (list[PathLike] | np.ndarray | None):
                Optional list of masks corresponding to the input images.
            labels (list | None):
                Optional list of labels corresponding to the input images.

        Returns:
            None

        Raises:
            ValueError:
                If the number of masks or labels does not match the number of images.

        """
        if masks is None and labels is None:
            return

        len_images = len(images)

        if masks is not None and len_images != len(masks):
            msg = (
                f"len(masks) is not equal to len(images) "
                f": {len(masks)} != {len(images)}"
            )
            raise ValueError(
                msg,
            )

        if labels is not None and len_images != len(labels):
            msg = (
                f"len(labels) is not equal to len(images) "
                f": {len(labels)} != {len(images)}"
            )
            raise ValueError(
                msg,
            )
        return

    def _update_run_params(
        self: EngineABC,
        images: list[os.PathLike | WSIReader] | np.ndarray,
        masks: list[os.PathLike] | np.ndarray | None = None,
        labels: list | None = None,
        save_dir: os.PathLike | Path | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> Path | None:
        """Update runtime parameters for the engine before running inference.

        This method sets internal attributes such as caching, batch size,
        IO configuration, and output format based on user input and keyword arguments.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                List of input images or a NumPy array of patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional list of masks for WSI processing.
            labels (list | None):
                Optional list of labels for input images.
            save_dir (PathLike | Path | None):
                Directory to save output files. Required for WSI mode.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution settings.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
            **kwargs (EngineABCRunParams):
                Additional runtime parameters to update engine attributes.

        Returns:
            Path | None:
                Path to the save directory if applicable, otherwise None.

        Raises:
            TypeError:
                If an unsupported output_type is provided.
            ValueError:
                If required configuration or input parameters are missing.

        """
        for key in kwargs:
            setattr(self, key, kwargs.get(key))

        if self.num_loader_workers is not None and self.num_loader_workers > 0:
            dask.config.set(scheduler="threads", num_workers=self.num_loader_workers)
        else:
            dask.config.set(scheduler="threads")

        if not self.return_labels:
            self.drop_keys.append("label")

        self.patch_mode = patch_mode

        self._validate_input_numbers(images=images, masks=masks, labels=labels)
        if output_type.lower() not in ["dict", "zarr", "annotationstore"]:
            msg = "output_type must be 'dict' or 'zarr' or 'annotationstore'."
            raise TypeError(msg)

        self.output_type = output_type

        if save_dir is not None and output_type.lower() not in [
            "zarr",
            "annotationstore",
        ]:
            self.output_type = "zarr"
            msg = (
                f"output_type has been updated to 'zarr' "
                f"for saving the file to {save_dir}."
                f"Remove `save_dir` input to return the output as a `dict`."
            )
            logger.info(msg)

        self.images = self._validate_images_masks(images=images)

        if masks is not None:
            self.masks = self._validate_images_masks(images=masks)

        self.labels = labels

        # if necessary move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=ioconfig)
        self.model = self.model.to(device=self.device)
        self._ioconfig = self._update_ioconfig(
            ioconfig,
            self.patch_input_shape,
            self.stride_shape,
            self.input_resolutions,
        )

        return prepare_engines_save_dir(
            save_dir=save_dir,
            patch_mode=patch_mode,
            overwrite=overwrite,
        )

    def _run_patch_mode(
        self: EngineABC,
        output_type: str,
        save_dir: Path,
        **kwargs: EngineABCRunParams,
    ) -> dict | AnnotationStore | Path:
        """Run the engine in patch mode.

        This method performs inference on image patches, post-processes the predictions,
        and saves the output in the specified format.

        Args:
            output_type (str):
                Desired output format. Supported values are "dict", "zarr",
                and "annotationstore".
            save_dir (Path):
                Directory to save the output files.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters including:
                    - output_file: Name of the output file.
                    - scale_factor: Scaling factor for annotations.
                    - class_dict: Mapping of class indices to names.

        Returns:
            dict | AnnotationStore | Path:
                - If output_type is "dict": returns predictions as a dictionary.
                - If output_type is "zarr": returns path to saved zarr file.
                - If output_type is "annotationstore": returns an AnnotationStore
                  or path to .db file.

        """
        save_path = None
        if save_dir:
            output_file = Path(kwargs.get("output_file", "output.zarr"))
            save_path = save_dir / (str(output_file.stem) + ".zarr")

        duplicate_filter = DuplicateFilter()
        logger.addFilter(duplicate_filter)

        self.dataloader = self.get_dataloader(
            images=self.images,
            masks=self.masks,
            labels=self.labels,
            patch_mode=True,
            ioconfig=self._ioconfig,
        )
        raw_predictions = self.infer_patches(
            dataloader=self.dataloader,
            return_coordinates=output_type == "annotationstore",
        )

        raw_predictions["predictions"] = self.post_process_patches(
            raw_predictions=raw_predictions["probabilities"],
            prediction_shape=raw_predictions["probabilities"].shape[:-1],
            prediction_dtype=raw_predictions["probabilities"].dtype,
            **kwargs,
        )

        logger.removeFilter(duplicate_filter)

        out = self.save_predictions(
            processed_predictions=raw_predictions,
            output_type=output_type,
            save_path=save_path,
            **kwargs,
        )

        msg = f"Output file saved at {out}."
        logger.info(msg=msg)
        return out

    @staticmethod
    def _calculate_scale_factor(dataloader: DataLoader) -> float | tuple[float, float]:
        """Calculate the scale factor for final output based on dataloader resolution.

        This method compares the resolution used during reading with the slide's
        baseline resolution to compute a scale factor for coordinate transformation.

        Args:
            dataloader (DataLoader):
                PyTorch DataLoader used for WSI inference. Must contain resolution
                and unit metadata in its dataset.

        Returns:
            float | tuple[float, float]:
                Scale factor for converting coordinates to baseline resolution.
                - If units are "mpp": returns (model_mpp / slide_mpp).
                - If units are "level": returns downsample ratio.
                - If units are "power": returns objective_power / model_power.
                - If units are "baseline": returns the resolution directly.

        """
        # get units and resolution from dataloader.
        dataloader_units = dataloader.dataset.units
        dataloader_resolution = dataloader.dataset.resolution

        # if dataloader units is baseline slide resolution is 1.0.
        # in this case dataloader resolution / slide resolution will be
        # equal to dataloader resolution.

        if dataloader_units in ["mpp", "level", "power"]:
            wsimeta_dict = dataloader.dataset.reader_info.as_dict()

        if dataloader_units == "mpp":
            slide_resolution = wsimeta_dict[dataloader_units]
            scale_factor = np.divide(dataloader_resolution, slide_resolution)
            return scale_factor[0], scale_factor[1]

        if dataloader_units == "level":
            downsample_ratio = wsimeta_dict["level_downsamples"][dataloader_resolution]
            return downsample_ratio, downsample_ratio

        if dataloader_units == "power":
            slide_objective_power = wsimeta_dict["objective_power"]
            return (
                slide_objective_power / dataloader_resolution,
                slide_objective_power / dataloader_resolution,
            )

        return dataloader_resolution

    def _run_wsi_mode(
        self: EngineABC,
        output_type: str,
        save_dir: Path,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> dict | AnnotationStore | Path:
        """Run the engine in WSI mode (patch_mode = False).

        This method performs inference on each whole slide image (WSI),
        post-processes the predictions, and saves the output in the specified format.

        Args:
            output_type (str):
                Desired output format. Supported values are "dict", "zarr",
                and "annotationstore".
            save_dir (Path):
                Directory to save the output files.
            **kwargs (EngineABCRunParams):
                Additional runtime parameters including:
                    - output_file: Name of the output file.
                    - scale_factor: Scaling factor for annotations.
                    - class_dict: Mapping of class indices to names.

        Returns:
            dict | AnnotationStore | Path:
                Dictionary mapping each input WSI to its corresponding output path.
                Output may be a zarr file, SQLite database, or in-memory dictionary.

        """
        progress_bar = None
        tqdm = get_tqdm()

        if self.verbose:
            progress_bar = tqdm(
                total=len(self.images),
                desc="Processing WSIs",
            )
        suffix = ".zarr"
        if output_type == "AnnotationStore":
            suffix = ".db"

        out = {image: save_dir / (str(image.stem) + suffix) for image in self.images}

        save_path = {
            image: save_dir / (str(image.stem) + ".zarr") for image in self.images
        }

        for image_num, image in enumerate(self.images):
            duplicate_filter = DuplicateFilter()
            logger.addFilter(duplicate_filter)
            mask = self.masks[image_num] if self.masks is not None else None
            self.dataloader = self.get_dataloader(
                images=image,
                masks=mask,
                patch_mode=False,
                ioconfig=self._ioconfig,
            )

            scale_factor = self._calculate_scale_factor(dataloader=self.dataloader)

            raw_predictions = self.infer_wsi(
                dataloader=self.dataloader,
                **kwargs,
            )

            raw_predictions["predictions"] = self.post_process_wsi(
                raw_predictions=raw_predictions["probabilities"],
                prediction_shape=raw_predictions["probabilities"].shape[:-1],
                prediction_dtype=raw_predictions["probabilities"].dtype,
                **kwargs,
            )

            kwargs["output_file"] = out[image]
            kwargs["scale_factor"] = scale_factor
            out[image] = self.save_predictions(
                processed_predictions=raw_predictions,
                output_type=output_type,
                save_path=save_path[image],
                **kwargs,
            )
            logger.removeFilter(duplicate_filter)
            msg = f"Output file saved at {out[image]}."
            logger.info(msg=msg)

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return out

    def run(
        self: EngineABC,
        images: list[os.PathLike | WSIReader] | np.ndarray,
        masks: list[os.PathLike] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[EngineABCRunParams],
    ) -> AnnotationStore | Path | str | dict:
        """Run the engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both patch
        and WSI modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                List of input images or a NumPy array of patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional list of masks for WSI processing.
                Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            labels (list | None):
                Optional list of labels for input images.
            ioconfig (ModelIOConfigABC | None):
                IO configuration for patch extraction and resolution settings.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`).
                Default is True.
            save_dir (PathLike | Path | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore".
            **kwargs (EngineABCRunParams):
                Additional runtime parameters to update engine attributes.

        Returns:
            AnnotationStore | Path | str | dict:
                - If patch_mode is True: returns predictions or path to saved output.
                - If patch_mode is False: returns a dictionary mapping each WSI to
                  its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
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
        save_dir = self._update_run_params(
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

        if patch_mode:
            return self._run_patch_mode(
                output_type=self.output_type,
                save_dir=save_dir,
                **kwargs,
            )

        # All inherited classes will get scale_factors,
        # highest_input_resolution, implement dataloader,
        # pre-processing, post-processing and save_output
        # for WSIs separately.
        return self._run_wsi_mode(
            output_type=self.output_type,
            save_dir=save_dir,
            **kwargs,
        )
