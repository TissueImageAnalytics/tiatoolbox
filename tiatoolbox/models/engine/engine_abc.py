"""Defines Abstract Base Class for TIAToolbox Engines to run CNN models."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import tqdm
import zarr
from torch import nn

from tiatoolbox import logger
from tiatoolbox.models.architecture import get_pretrained_model
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

    If the save directory is not defined, this function will raise an error.

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
        "will be returned to the calling function.",
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=overwrite)

    return save_dir


class EngineABCRunParams(TypedDict, total=False):
    """Class describing the input parameters for the run function.

    Defines the expected keyword arguments for the EngineABC.run() function.

    """

    batch_size: int
    cache_mode: bool
    cache_size: int
    device: str
    ioconfig: ModelIOConfigABC
    merge_predictions: bool
    num_loader_workers: int
    num_post_proc_workers: int
    patch_input_shape: IntPair
    resolution: Resolution
    return_labels: bool
    stride_shape: IntPair
    units: Units
    verbose: bool


class EngineABC(ABC):
    """Abstract base class for engines to run CNN models in TIAToolbox.

    Args:
        model (str | ModelABC):
            A PyTorch model. Default is `None`.
            The user can request pretrained models from the toolbox using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the weights parameter.
        batch_size (int):
            Number of image patches fed into the model each time. Default = 8.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. Default = 0
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model. Default = 0
        weights (str or Path):
            Path to the weight of the corresponding `model`.

            >>> engine = EngineABC(
            ...    model="pretrained-model",
            ...    weights="pretrained-local-weights.pth"
            ... )

        device (str):
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default is "cpu".
        verbose (bool):
            Whether to output logging information.

    Attributes:
        images (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A NHWC image or a path to WSI.
        masks (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            List of tissue masks or binary masks corresponding to processing area of
            input images.
        patch_mode (str):
            Whether to treat input image as a patch or WSI.
            default = True.
        model (str | nn.Module):
            Defined PyTorch model.
            Name of an existing model supported by the TIAToolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument
            is case-insensitive.
        ioconfig (ModelIOConfigABC):
            Input IO configuration to run the Engine.
        _ioconfig ():
            Runtime ioconfig.
        return_labels (bool):
            Whether to return the labels with the predictions.
        merge_predictions (bool):
            Whether to merge the predictions to form a 2-dimensional
            map. This is only applicable if `patch_mode` is False in inference.
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
        labels (list | None):
                List of labels. Only a single label per image is supported.
        device (str):
            :class:`torch.device` to run the model.
        num_loader_workers (int):
            Number of workers used in :class:`torch.utils.data.DataLoader`.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
        return_labels (bool):
            Whether to return the output labels. Default = False.
        merge_predictions (bool):
            Whether to merge WSI predictions into a single file. Default = False.
        verbose (bool):
            Whether to output logging information.

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
        >>> # define all the abstract classes
        >>> import numpy as np
        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = TestEngineABC(model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # array of list of 2 image patches as input
        >>> import numpy as np
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
        >>> output = engine.run(wsi_file, patch_mode=True)

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
        super().__init__()

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
        self._ioconfig = self.ioconfig  # runtime ioconfig

        self.batch_size = batch_size
        self.cache_mode: bool = False
        self.cache_size: int = self.batch_size if self.batch_size else 10000
        self.labels: list | None = None
        self.merge_predictions: bool = False
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.patch_input_shape: IntPair | None = None
        self.resolution: Resolution = 1.0
        self.return_labels: bool = False
        self.stride_shape: IntPair | None = None
        self.units: Units = "baseline"
        self.verbose = verbose

    @staticmethod
    def _initialize_model_ioconfig(
        model: str | nn.Module,
        weights: str | Path | None,
    ) -> tuple[nn.Module, ModelIOConfigABC | None]:
        """Helper function to initialize model and ioconfig attributes.

        If a pretrained model provided by the TIAToolbox is requested. The model
        can be specified as a string otherwise torch.nn.Module is required.
        This function also loads the :class:`ModelIOConfigABC` using the information
        from the pretrained models in TIAToolbox. If ioconfig is not available then it
        should be provided in the :func:`run` function.

        Args:
            model (str | nn.Module):
                A torch model which should be run by the engine.

            weights (str | Path | None):
                Path to pretrained weights. If no pretrained weights are provided
                and the model is provided by TIAToolbox, then pretrained weights will
                be automatically loaded from the TIA servers.

        Returns:
            nn.Module:
                The requested PyTorch model.

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
        images: Path,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return dataloader for inference.

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
            ioconfig (ModelIOConfigABC):
                :class:`ModelIOConfigABC` object.
            patch_mode(bool):
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

    def infer_patches(
        self: EngineABC,
        dataloader: DataLoader,
        save_path: Path | None,
    ) -> dict:
        """Runs model inference on image patches and returns output as a dictionary.

        Args:
            dataloader (DataLoader):
                An :class:`torch.utils.data.DataLoader` object to run inference.
            save_path (Path | None):
                If cache_mode is True then path to save zarr file must be provided.

        Returns:
            dict:
                Result of model inference as a dictionary.

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

        keys = ["predictions"]

        if self.return_labels:
            keys.append("labels")

        raw_predictions = dict.fromkeys(keys, [])

        zarr_group = None

        if self.cache_mode:
            zarr_group = zarr.open(save_path, mode="w")

        for _, batch_data in enumerate(dataloader):
            batch_output_predictions = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            raw_predictions["predictions"].extend(batch_output_predictions.tolist())

            if self.return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                raw_predictions["labels"].extend(list(batch_data["label"]))

            if self.cache_mode:
                zarr_group = write_to_zarr_in_cache_mode(
                    zarr_group=zarr_group, output_data_to_save=raw_predictions
                )

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return raw_predictions

    def save_predictions(
        self: EngineABC,
        processed_predictions: dict,
        output_type: str,
        save_dir: Path | None = None,
        **kwargs: dict,
    ) -> dict | AnnotationStore | Path:
        """Save Patch predictions.

        Args:
            processed_predictions (dict):
                A dictionary of patch prediction information.
            save_dir (Path):
                Optional Output Path to directory to save the patch dataset output to a
                `.zarr` or `.db` file, provided patch_mode is True. if the patch_mode is
                  False then save_dir is required.
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (dict):
                Keyword Args to update setup_patch_dataset() method attributes.

        Returns: (dict, Path, :class:`AnnotationStore`):
            if the output_type is "AnnotationStore", the function returns the patch
            predictor output as an SQLiteStore containing Annotations for each or the
            Path to a `.db` file depending on whether a save_dir Path is provided.
            Otherwise, the function defaults to returning patch predictor output, either
            as a dict or the Path to a `.zarr` file depending on whether a save_dir Path
            is provided.

        """
        if self.cache_mode or (not save_dir and output_type != "AnnotationStore"):
            return processed_predictions

        output_file = (
            kwargs["output_file"] and kwargs.pop("output_file")
            if "output_file" in kwargs
            else "output"
        )

        save_path = save_dir / output_file

        if output_type == "AnnotationStore":
            # scale_factor set from kwargs
            scale_factor = kwargs.get("scale_factor")
            # class_dict set from kwargs
            class_dict = kwargs.get("class_dict")

            return dict_to_store(
                processed_predictions,
                scale_factor,
                class_dict,
                save_path,
            )

        return dict_to_zarr(
            processed_predictions,
            save_path,
            **kwargs,
        )

    @staticmethod
    def post_process_patches(
        raw_predictions: dict,
        # cache_mode: bool,
        # cache_size: bool,
        **kwargs: dict,
    ) -> dict:
        """Save Patch predictions.

        Args:
            raw_predictions (dict):
                A dictionary of patch prediction information.
            **kwargs (dict):
                Keyword Args to update setup_patch_dataset() method attributes.

        Returns:
            dict:
                Return patch based output after post-processing.

        """
        _ = kwargs.get("key_values")  # Key values required for post-processing
        # _ = cache_mode # if post-processing of patches is required.
        # _ = cache_size # if post-processing of patches is required in cache_mode.

        return raw_predictions

    @abstractmethod
    def infer_wsi(
        self: EngineABC,
        dataloader: torch.utils.data.DataLoader,
        img_label: str,
        highest_input_resolution: list[dict],
        save_dir: Path,
        **kwargs: dict,
    ) -> list:
        """Model inference on a WSI."""
        # return coordinates of patches processed within a tile / whole-slide image
        raise NotImplementedError

    @abstractmethod
    def save_output(
        self: EngineABC,
        raw_output: dict,
        save_dir: Path,
        output_type: str,
        **kwargs: dict,
    ) -> AnnotationStore | Path:
        """Post-process a WSI.

        Args:
            raw_output (dict):
                A dictionary of patch prediction information.
            save_dir (Path):
                Output Path to directory to save the patch dataset output to a
                `.zarr` or `.db` file
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (dict):
                Keyword Args to update setup_patch_dataset() method attributes.

        Returns: (AnnotationStore or Path):
            if the output_type is "AnnotationStore", the function returns the patch
            predictor output as an SQLiteStore containing Annotations stored in a `.db`
            file. Otherwise, the function defaults to returning patch predictor output
            stored in a `.zarr` file.

        """
        output_file = (
            kwargs["output_file"] and kwargs.pop("output_file")
            if "output_file" in kwargs
            else "output"
        )
        save_path = save_dir / output_file

        if output_type == "AnnotationStore":
            # scale_factor set from kwargs
            scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
            # class_dict set from kwargs
            class_dict = kwargs.get("class_dict")

            return dict_to_store(raw_output, scale_factor, class_dict, save_path)

        # referring to the zarr group generated during the infer_wsi step
        return save_path.parent.absolute() / (save_path.stem + ".zarr")

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
            raise ValueError(msg)

        if ioconfig is not None:
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
        if ioconfig:
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
        *,
        overwrite: bool = False,
        patch_mode: bool,
        **kwargs: EngineABCRunParams,
    ) -> Path | None:
        """Updates runtime parameters.

        Updates runtime parameters for an EngineABC for EngineABC.run().

        """
        for key in kwargs:
            setattr(self, key, kwargs.get(key))

        self.patch_mode = patch_mode
        if self.cache_mode and self.cache_size > self.batch_size:
            self.batch_size = self.cache_size

        self._validate_input_numbers(images=images, masks=masks, labels=labels)
        self.images = self._validate_images_masks(images=images)

        if masks is not None:
            self.masks = self._validate_images_masks(images=masks)

        self.labels = labels

        # if necessary move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=self.ioconfig)
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
        dataloader = self.get_dataloader(
            images=self.images,
            labels=self.labels,
            patch_mode=True,
        )
        raw_predictions = self.infer_patches(
            dataloader=dataloader, save_path=save_dir / "out.zarr"
        )
        processed_predictions = self.post_process_patches(
            raw_predictions=raw_predictions,
            **kwargs,
        )
        return self.save_predictions(
            processed_predictions=processed_predictions,
            output_type=output_type,
            save_dir=save_dir,
            **kwargs,
        )

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
        **kwargs: EngineABCRunParams,
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
                "zarr" or "AnnotationStore". Default is "zarr".
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
                - merged: path to .npy contain merged
                  predictions if `merge_predictions` is `True`.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> predictor = EngineABC(model="resnet18-kather100k")
            >>> output = predictor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'raw': '0.raw.json', 'merged': '0.merged.npy'}
            >>> output['wsi2.svs']
            ... {'raw': '1.raw.json', 'merged': '1.merged.npy'}

            >>> predictor = EngineABC(model="alexnet-kather100k")
            >>> output = predictor.run(
            >>>     images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            >>>     labels=list(range(10)),
            >>>     on_gpu=False,
            >>>     )
            >>> output
            ... {'predictions': [[0.7716791033744812, 0.0111849969252944, ...,
            ... 0.034451354295015335, 0.004817609209567308]],
            ... 'labels': [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
            ... tensor(5), tensor(6), tensor(7), tensor(8), tensor(9)]}

            >>> predictor = EngineABC(model="alexnet-kather100k")
            >>> save_dir = Path("/tmp/patch_output/")
            >>> output = eng.run(
            >>>     images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            >>>     on_gpu=False,
            >>>     verbose=False,
            >>>     save_dir=save_dir,
            >>>     overwrite=True
            >>>     )
            >>> output
            ... '/tmp/patch_output/output.zarr'
        """
        save_dir = self._update_run_params(
            images=images,
            masks=masks,
            labels=labels,
            save_dir=save_dir,
            ioconfig=ioconfig,
            overwrite=overwrite,
            patch_mode=patch_mode,
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
        raise NotImplementedError
