"""Defines Abstract Base Class for TIAToolbox Engines."""

from __future__ import annotations

import copy
import shutil
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import tqdm
import zarr
from torch import nn
from typing_extensions import Unpack

from tiatoolbox import DuplicateFilter, logger, rcParam
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model
from tiatoolbox.models.dataset.dataset_abc import PatchDataset, WSIPatchDataset
from tiatoolbox.models.models_abc import load_torch_model
from tiatoolbox.utils.misc import (
    dict_to_store,
    dict_to_zarr,
    write_to_zarr_in_cache_mode,
)

from .io_config import ModelIOConfigABC

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.typing import IntPair, Resolution, Units
    from tiatoolbox.wsicore.wsireader import WSIReader


def prepare_engines_save_dir(
    save_dir: os | Path | None,
    *,
    patch_mode: bool,
    overwrite: bool = False,
) -> Path | None:
    """Create a save directory.

    If patch_mode is False and the save directory is not defined,
    this function will raise an error.

    If patch_mode is True and the save directory is defined it will
    create save_dir otherwise returns None.

    Args:
        save_dir (str or Path):
            Path to output directory.
        patch_mode(bool):
            Whether to treat input image as a patch or WSI.
        overwrite (bool):
            Whether to overwrite the results. Default = False.

    Returns:
        :class:`Path`:
            Path to output directory.

    Raises:
        OSError:
            If the save directory is not defined.

    """
    if patch_mode is True:
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=overwrite)
        return save_dir

    if save_dir is None:
        msg = (
            "Input WSIs detected but no save directory provided."
            "Please provide a 'save_dir'."
        )
        raise OSError(msg)

    logger.info(
        "When providing multiple whole slide images, "
        "the outputs will be saved and the locations of outputs "
        "will be returned to the calling function when `run()`"
        "finishes successfully.",
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=overwrite)

    return save_dir


class EngineABCRunParams(TypedDict, total=False):
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

    batch_size: int
    cache_mode: bool
    cache_size: int
    class_dict: dict
    device: str
    ioconfig: ModelIOConfigABC
    num_loader_workers: int
    num_post_proc_workers: int
    output_file: str
    patch_input_shape: IntPair
    resolution: Resolution
    return_labels: bool
    scale_factor: tuple[float, float]
    stride_shape: IntPair
    units: Units
    verbose: bool


class EngineABC(ABC):  # noqa: B024
    """Abstract base class for TIAToolbox deep learning engines to run CNN models.

    Args:
        model (str | ModelABC):
            A PyTorch model. Default is `None`.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter.
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

            >>> engine = EngineABC(
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
        >>> # Inherit from EngineABC
        >>> class TestEngineABC(EngineABC):
        >>>     def __init__(
        >>>        self,
        >>>        model,
        >>>        weights,
        >>>        verbose,
        >>>     ):
        >>>       super().__init__(model=model, weights=weights, verbose=verbose)
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
        """Initialize Engine."""
        self.images = None
        self.masks = None
        self.patch_mode = None
        self.device = device

        # Initialize model with specified weights and ioconfig.
        self.model, self.ioconfig = self._initialize_model_ioconfig(
            model=model,
            weights=weights,
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
        self.cache_mode: bool = False
        self.cache_size: int = self.batch_size if self.batch_size else 10000
        self.labels: list | None = None
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.patch_input_shape: IntPair | None = None
        self.resolution: Resolution | None = None
        self.return_labels: bool = False
        self.stride_shape: IntPair | None = None
        self.units: Units | None = None
        self.verbose = verbose

    @staticmethod
    def _initialize_model_ioconfig(
        model: str | ModelABC,
        weights: str | Path | None,
    ) -> tuple[nn.Module, ModelIOConfigABC | None]:
        """Helper function to initialize model and ioconfig attributes.

        If a pretrained model provided by the TIAToolbox is requested. The model
        can be specified as a string otherwise :class:`torch.nn.Module` is required.
        This function also loads the :class:`ModelIOConfigABC` using the information
        from the pretrained models in TIAToolbox. If ioconfig is not available then it
        should be provided in the :func:`run()` function.

        Args:
            model (str | ModelABC):
                A PyTorch model. Default is `None`.
                The user can request pretrained models from the toolbox model zoo using
                the list of pretrained models available at this `link
                <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
                By default, the corresponding pretrained weights will also
                be downloaded. However, you can override with your own set
                of weights using the `weights` parameter.

            weights (str | Path | None):
                Path to pretrained weights. If no pretrained weights are provided
                and the `model` is provided by TIAToolbox, then pretrained weights will
                be automatically loaded from the TIA servers.

        Returns:
            ModelABC:
                The requested PyTorch model as a :class:`ModelABC` instance.

            ModelIOConfigABC | None:
                The model io configuration for TIAToolbox pretrained models.
                If the specified model is not in TIAToolbox model zoo, then the function
                returns None.

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
        if labels:
            # if a labels is provided, then return with the prediction
            self.return_labels = bool(labels)

        if not patch_mode:
            dataset = WSIPatchDataset(
                img_path=images,
                mode="wsi",
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
            )

        dataset = PatchDataset(inputs=images, labels=labels)
        dataset.preproc_func = self.model.preproc_func

        # preprocessing must be defined with the dataset
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_workers,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

    @staticmethod
    def _update_model_output(raw_predictions: dict, raw_output: dict) -> dict:
        """Helper function to append raw output during inference."""
        for key in raw_output:
            if raw_predictions[key] is None:
                raw_predictions[key] = raw_output[key]
            else:
                raw_predictions[key] = np.append(
                    raw_predictions[key], raw_output[key], axis=0
                )

        return raw_predictions

    def _get_coordinates(self: EngineABC, batch_data: dict) -> np.ndarray:
        """Helper function to collect coordinates for AnnotationStore."""
        if self.patch_mode:
            coordinates = [0, 0, *batch_data["image"].shape[1:3]]
            return np.tile(coordinates, reps=(batch_data["image"].shape[0], 1))
        return batch_data["coords"].numpy()

    def infer_patches(
        self: EngineABC,
        dataloader: DataLoader,
        save_path: Path | None,
        *,
        return_coordinates: bool = False,
    ) -> dict | Path:
        """Runs model inference on image patches and returns output as a dictionary.

        Args:
            dataloader (DataLoader):
                An :class:`torch.utils.data.DataLoader` object to run inference.
            save_path (Path | None):
                If `cache_mode` is True then path to save zarr file must be provided.
            return_coordinates (bool):
                Whether to save coordinates in the output. This is required when
                this function is called by `infer_wsi` and `patch_mode` is False.

        Returns:
            dict or Path:
                Result of model inference as a dictionary. Returns path to
                saved zarr file if `cache_mode` is True.

        """
        progress_bar = None

        if self.verbose:
            progress_bar = tqdm.tqdm(
                total=int(len(dataloader)),
                leave=True,
                ncols=80,
                ascii=True,
                position=0,
            )

        keys = ["probabilities"]

        if self.return_labels:
            keys.append("labels")

        if return_coordinates:
            keys.append("coordinates")

        raw_predictions = {key: None for key in keys}

        zarr_group = None

        if self.cache_mode:
            zarr_group = zarr.open(save_path, mode="w")

        for _, batch_data in enumerate(dataloader):
            batch_output = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )
            if return_coordinates:
                batch_output["coordinates"] = self._get_coordinates(batch_data)

            if self.return_labels:  # be careful of `s`
                if isinstance(batch_data["label"], torch.Tensor):
                    batch_output["labels"] = batch_data["label"].numpy()
                else:
                    batch_output["labels"] = batch_data["label"]

            raw_predictions = self._update_model_output(
                raw_predictions=raw_predictions,
                raw_output=batch_output,
            )

            if self.cache_mode:
                zarr_group = write_to_zarr_in_cache_mode(
                    zarr_group=zarr_group, output_data_to_save=raw_predictions
                )
                raw_predictions = {key: None for key in keys}

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return save_path if self.cache_mode else raw_predictions

    def post_process_patches(
        self: EngineABC,
        raw_predictions: dict | Path,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> dict | Path:
        """Post-process raw patch predictions from inference.

        The output of :func:`infer_patches()` with patch prediction information will be
        post-processed using this function. The processed output will be saved in the
        respective input format. If `cache_mode` is True, the function processes the
        input using zarr group with size specified by `cache_size`.

        Args:
            raw_predictions (dict | Path):
                A dictionary or path to zarr with patch prediction information.
            **kwargs (EngineABCRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            dict or Path:
                Returns patch based output after post-processing. Returns path to
                saved zarr file if `cache_mode` is True.

        """
        _ = kwargs.get("return_labels")  # Key values required for post-processing

        if self.cache_mode:  # cache mode
            _ = zarr.open(raw_predictions, mode="w")

        return raw_predictions

    def save_predictions(
        self: EngineABC,
        processed_predictions: dict | Path,
        output_type: str,
        save_dir: Path | None = None,
        **kwargs: dict,
    ) -> dict | AnnotationStore | Path:
        """Save model predictions.

        Args:
            processed_predictions (dict | Path):
                A dictionary or path to zarr with model prediction information.
            save_dir (Path):
                Optional output path to directory to save the patch dataset output to a
                `.zarr` or `.db` file, provided `patch_mode` is True. If the
                `patch_mode` is False then `save_dir` is required.
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (EngineABCRunParams):
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

            out_file = dict_to_store(
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

    def infer_wsi(
        self: EngineABC,
        dataloader: DataLoader,
        save_path: Path,
        **kwargs: EngineABCRunParams,
    ) -> Path:
        """Model inference on a WSI.

        Args:
            dataloader (DataLoader):
                A torch dataloader to process WSIs.

            save_path (Path):
                Path to save the intermediate output. The intermediate output is saved
                in a zarr file.
            **kwargs (EngineABCRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            save_path (Path):
                Path to zarr file where intermediate output is saved.

        """
        _ = kwargs.get("patch_mode", False)
        return self.infer_patches(
            dataloader=dataloader,
            save_path=save_path,
            return_coordinates=True,
        )

    # This is not a static model for child classes.
    def post_process_wsi(  # skipcq: PYL-R0201
        self: EngineABC,
        raw_predictions: dict | Path,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> dict | Path:
        """Post process WSI output.

        Takes the raw output from patch predictions and post-processes it to improve the
        results e.g., using information from neighbouring patches.

        """
        _ = kwargs.get("return_labels")  # Key values required for post-processing
        return raw_predictions

    def _load_ioconfig(self: EngineABC, ioconfig: ModelIOConfigABC) -> ModelIOConfigABC:
        """Helper function to load ioconfig.

        If the model is provided by TIAToolbox it will load the default ioconfig.
        Otherwise, ioconfig must be specified.

        Args:
            ioconfig (ModelIOConfigABC):
                IO configuration to run the engines.

        Raises:
             ValueError:
                If no io configuration is provided or found in the pretrained TIAToolbox
                models.

        Returns:
            ModelIOConfigABC:
                The ioconfig used for the run.

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
        resolution: Resolution,
        units: Units,
    ) -> ModelIOConfigABC:
        """Update IOConfig.

        Args:
            ioconfig (:class:`ModelIOConfigABC`):
                Input ioconfig for PatchPredictor.
            patch_input_shape (tuple):
                Size of patches input to the model. Patches are at
                requested read resolution, not with respect to level 0,
                and must be positive.
            stride_shape (tuple):
                Stride using during tile and WSI processing. Stride is
                at requested read resolution, not with respect to
                level 0, and must be positive. If not provided,
                `stride_shape=patch_input_shape`.
            resolution (Resolution):
                Resolution used for reading the image. Please see
                :obj:`WSIReader` for details.
            units (Units):
                Units of resolution used for reading the image.

        Returns:
            Updated Patch Predictor IO configuration.

        """
        config_flag = (
            patch_input_shape is None,
            resolution is None,
            units is None,
        )
        if isinstance(ioconfig, ModelIOConfigABC):
            return ioconfig

        if self.ioconfig is None and any(config_flag):
            msg = (
                "Must provide either "
                "`ioconfig` or `patch_input_shape`, `resolution`, and `units`."
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
            if resolution is not None:
                ioconfig.input_resolutions[0]["resolution"] = resolution
            if units is not None:
                ioconfig.input_resolutions[0]["units"] = units

            return ioconfig

        return ModelIOConfigABC(
            input_resolutions=[{"resolution": resolution, "units": units}],
            patch_input_shape=patch_input_shape,
            stride_shape=stride_shape,
            output_resolutions=[],
        )

    @staticmethod
    def _validate_images_masks(images: list | np.ndarray) -> list | np.ndarray:
        """Validate input images for a run."""
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
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
    ) -> None:
        """Validates number of input images, masks and labels."""
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
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        save_dir: os | Path | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        output_type: str = "dict",
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: Unpack[EngineABCRunParams],
    ) -> Path | None:
        """Updates runtime parameters.

        Updates runtime parameters for an EngineABC for EngineABC.run().

        """
        for key in kwargs:
            setattr(self, key, kwargs.get(key))

        self.patch_mode = patch_mode
        if not self.patch_mode:
            self.cache_mode = True  # if input is WSI run using cache mode.

        if self.cache_mode and self.batch_size > self.cache_size:
            self.batch_size = self.cache_size

        self._validate_input_numbers(images=images, masks=masks, labels=labels)
        if output_type.lower() not in ["dict", "zarr", "annotationstore"]:
            msg = "output_type must be 'dict' or 'zarr' or 'annotationstore'."
            raise TypeError(msg)

        self.images = self._validate_images_masks(images=images)

        if masks is not None:
            self.masks = self._validate_images_masks(images=masks)

        self.labels = labels

        # if necessary move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=ioconfig)
        self.model.to(device=self.device)
        self._ioconfig = self._update_ioconfig(
            ioconfig,
            self.patch_input_shape,
            self.stride_shape,
            self.resolution,
            self.units,
        )

        return prepare_engines_save_dir(
            save_dir=save_dir,
            patch_mode=patch_mode,
            overwrite=overwrite,
        )

    def _run_patch_mode(
        self: EngineABC, output_type: str, save_dir: Path, **kwargs: EngineABCRunParams
    ) -> dict | AnnotationStore | Path:
        """Runs the Engine in the patch mode.

        Input arguments are passed from :func:`EngineABC.run()`.

        """
        save_path = None
        if self.cache_mode:
            output_file = Path(kwargs.get("output_file", "output.zarr"))
            save_path = save_dir / (str(output_file.stem) + ".zarr")

        duplicate_filter = DuplicateFilter()
        logger.addFilter(duplicate_filter)

        dataloader = self.get_dataloader(
            images=self.images,
            masks=self.masks,
            labels=self.labels,
            patch_mode=True,
        )
        raw_predictions = self.infer_patches(
            dataloader=dataloader,
            save_path=save_path,
            return_coordinates=output_type == "annotationstore",
        )
        processed_predictions = self.post_process_patches(
            raw_predictions=raw_predictions,
            **kwargs,
        )
        logger.removeFilter(duplicate_filter)

        out = self.save_predictions(
            processed_predictions=processed_predictions,
            output_type=output_type,
            save_dir=save_dir,
            **kwargs,
        )

        if save_dir:
            msg = f"Output file saved at {out}."
            logger.info(msg=msg)
            return out

        return out

    @staticmethod
    def _calculate_scale_factor(dataloader: DataLoader) -> float | tuple[float, float]:
        """Calculates scale factor for final output.

        Uses the dataloader resolution and the WSI resolution to calculate scale
        factor for final WSI output.

        Args:
            dataloader (DataLoader):
                Dataloader for the current run.

        Returns:
            scale_factor (float | tuple[float, float]):
                Scale factor for final output.

        """
        # get units and resolution from dataloader.
        dataloader_units = dataloader.dataset.units
        dataloader_resolution = dataloader.dataset.resolution

        # if dataloader units is baseline slide resolution is 1.0.
        # in this case dataloader resolution / slide resolution will be
        # equal to dataloader resolution.

        if dataloader_units in ["mpp", "level", "power"]:
            wsimeta_dict = dataloader.dataset.reader.info.as_dict()

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
        """Runs the Engine in the WSI mode (patch_mode = False).

        Input arguments are passed from :func:`EngineABC.run()`.

        """
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
            dataloader = self.get_dataloader(
                images=image,
                masks=mask,
                patch_mode=False,
                ioconfig=self._ioconfig,
            )

            scale_factor = self._calculate_scale_factor(dataloader=dataloader)

            raw_predictions = self.infer_wsi(
                dataloader=dataloader,
                save_path=save_path[image],
                **kwargs,
            )
            processed_predictions = self.post_process_wsi(
                raw_predictions=raw_predictions,
                **kwargs,
            )
            kwargs["output_file"] = out[image]
            kwargs["scale_factor"] = scale_factor
            out[image] = self.save_predictions(
                processed_predictions=processed_predictions,
                output_type=output_type,
                save_dir=save_dir,
                **kwargs,
            )
            logger.removeFilter(duplicate_filter)
            msg = f"Output file saved at {out[image]}."
            logger.info(msg=msg)

        return out

    def run(
        self: EngineABC,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[EngineABCRunParams],
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
                "dict", "zarr" or "AnnotationStore". Default value is "zarr".
                When saving in the zarr format the output is saved using the
                `python zarr library <https://zarr.readthedocs.io/en/stable/>`__
                as a zarr group. If the required output type is an "AnnotationStore"
                then the output will be intermediately saved as zarr but converted
                to :class:`AnnotationStore` and saved as a `.db` file
                at the end of the loop.
            **kwargs (EngineABCRunParams):
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
                output_type=output_type,
                save_dir=save_dir,
                **kwargs,
            )

        # All inherited classes will get scale_factors,
        # highest_input_resolution, implement dataloader,
        # pre-processing, post-processing and save_output
        # for WSIs separately.
        return self._run_wsi_mode(
            output_type=output_type,
            save_dir=save_dir,
            **kwargs,
        )
