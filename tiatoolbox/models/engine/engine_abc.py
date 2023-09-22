"""Defines Abstract Base Class for TIAToolbox Model Engines."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

from tiatoolbox import logger
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.dataset.dataset_abc import PatchDataset
from tiatoolbox.models.models_abc import load_torch_model, model_to

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.wsicore.wsireader import WSIReader

    from .io_config import ModelIOConfigABC


def prepare_engines_save_dir(
    save_dir: os | Path | None,
    len_images: int,
    *,
    patch_mode: bool,
    overwrite: bool,
) -> Path | None:
    """Create directory if not defined and number of images is more than 1.

    Args:
        save_dir (str or Path):
            Path to output directory.
        len_images (int):
            List of inputs to process.
        patch_mode(bool):
            Whether to treat input image as a patch or WSI.
        overwrite (bool):
                Whether to overwrite the results. Default = False.

    Returns:
        :class:`Path`:
            Path to output directory.

    """
    if patch_mode is True:
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=overwrite)
        return save_dir

    if save_dir is None:
        if len_images > 1:
            msg = (
                "More than 1 WSIs detected but there is no save directory provided."
                "Please provide a 'save_dir'."
            )
            raise OSError(msg)
        return (
            Path.cwd()
        )  # save the output to current working directory and return save_dir

    if len_images > 1:
        logger.info(
            "When providing multiple whole slide images, "
            "the outputs will be saved and the locations of outputs "
            "will be returned to the calling function.",
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=overwrite)

    return save_dir


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox.

    Args:
        model (str | nn.Module):
            A PyTorch model. Default is `None`.
            The user can request pretrained models from the toolbox using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights.
        weights (str or Path):
            Path to the weight of the corresponding `model`.

            >>> engine = EngineABC(
            ...    pretrained_model="pretrained-model-name",
            ...    weights="pretrained-local-weights.pth")

        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. default = 0
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model. default = 0
        device (str):
            Select the device to run the model. Default is "cpu".
        verbose (bool):
            Whether to output logging information.

    Attributes:
        images (str or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
            A NHWC image or a path to WSI.
        patch_mode (str):
            Whether to treat input image as a patch or WSI.
            default = True.
        model (str | nn.Module):
            Defined PyTorch model.
            Name of the existing models supported by the TIAToolbox for
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
            map. This is only applicable `patch_mode` is False in inference.
        resolution (Resolution):
            Resolution used for reading the image. Please see
            :obj:`WSIReader` for details.
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `level`, `power` or `mpp`. Please see
            :obj:`WSIReader` for details.
        patch_input_shape (tuple):
            Size of patches input to the model. Patches are at
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
            Select the device to run the model. Default is "cpu".
        num_loader_workers (int):
            Number of workers used in torch.utils.data.DataLoader.
        verbose (bool):
            Whether to output logging information.

    Examples:
        >>> # array of list of 2 image patches as input
        >>> import numpy as np
        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=False)

        >>> # list of 2 image files as input
        >>> image = ['path/image1.png', 'path/image2.png']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.run(image, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.run(wsi_file, patch_mode=True)

    """

    def __init__(
        self: EngineABC,
        model: str | nn.Module,
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

        self.masks = None
        self.images = None
        self.patch_mode = None
        self.device = device

        # Initialize model with specified weights and ioconfig.
        self.model, self.ioconfig = self._initialize_model_ioconfig(
            model=model,
            weights=weights,
        )
        self.model = model_to(model=self.model, device=self.device)
        self._ioconfig = self.ioconfig  # runtime ioconfig

        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.verbose = verbose
        self.return_labels = False
        self.merge_predictions = False
        self.units = "baseline"
        self.resolution = 1.0
        self.patch_input_shape = None
        self.stride_shape = None
        self.labels = None

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
                Otherwise, None.

        """
        if not isinstance(model, (str, nn.Module)):
            msg = "Input model must be a string or 'torch.nn.Module'."
            raise TypeError(msg)

        if isinstance(model, str):
            # ioconfig is retrieved from the pretrained model in the toolbox.
            # no need to provide ioconfig in EngineABC.run() this case.
            return get_pretrained_model(model, weights)

        if weights is not None:
            model = load_torch_model(model=model, weights=weights)

        return model, None

    def pre_process_patches(
        self: EngineABC,
        images: np.ndarray | list,
        labels: list,
    ) -> torch.utils.data.DataLoader:
        """Pre-process an image patch."""
        if labels:
            # if a labels is provided, then return with the prediction
            self.return_labels = bool(labels)

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
    def _convert_output_to_requested_type(
        output: dict,
        output_type: str,
    ) -> AnnotationStore | np.ndarray | pd.DataFrame | dict | str:
        """Converts inference output to requested type."""
        # function convert output to output_type
        if output_type.lower() == "array":
            return np.array(output["predictions"])

        if output_type.lower() == "json":
            return json.dumps(output, indent=4)

        if output_type.lower() == "dataframe":
            return pd.DataFrame.from_dict(data=output)

        return output

    def infer_patches(
        self: EngineABC,
        data_loader: DataLoader,
    ) -> dict:
        """Model inference on an image patch."""
        progress_bar = None

        if self.verbose:
            progress_bar = tqdm.tqdm(
                total=int(len(data_loader)),
                leave=True,
                ncols=80,
                ascii=True,
                position=0,
            )
        raw_predictions = {
            "predictions": [],
        }

        if self.return_labels:
            raw_predictions["labels"] = []

        for _, batch_data in enumerate(data_loader):
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

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return raw_predictions

    def post_process_patches(
        self: EngineABC,
        raw_predictions: dict,
        output_type: str,
    ) -> AnnotationStore | np.ndarray | pd.DataFrame | dict | str:
        """Post-process an image patches."""
        return self._convert_output_to_requested_type(
            output=raw_predictions,
            output_type=output_type,
        )

    @abstractmethod
    def pre_process_wsi(self: EngineABC) -> NoReturn:
        """Pre-process a WSI."""
        raise NotImplementedError

    @abstractmethod
    def infer_wsi(self: EngineABC) -> NoReturn:
        """Model inference on a WSI."""
        raise NotImplementedError

    @abstractmethod
    def post_process_wsi(self: EngineABC) -> NoReturn:
        """Post-process a WSI."""
        raise NotImplementedError

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
        **kwargs: dict,
    ) -> AnnotationStore | np.ndarray | pd.DataFrame | dict | str:
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
                "zarr", "AnnotationStore". Default is "zarr".
                When saving in the zarr format the output is saved using the
                `python zarr library <https://zarr.readthedocs.io/en/stable/>`__
                as a zarr group. If the required output type is an "AnnotationStore"
                then the output will be intermediately saved as zarr but converted
                to :class:`AnnotationStore` and saved as a `.db` file
                at the end of the loop.
            **kwargs (dict):
                Keyword Args to update :class:`EngineABC` attributes.

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
            >>> predictor = EngineABC(
            ...                 pretrained_model="resnet18-kather100k")
            >>> output = predictor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'raw': '0.raw.json', 'merged': '0.merged.npy'}
            >>> output['wsi2.svs']
            ... {'raw': '1.raw.json', 'merged': '1.merged.npy'}

        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._validate_input_numbers(images=images, masks=masks, labels=labels)
        self.images = self._validate_images_masks(images=images)

        if masks is not None:
            self.masks = self._validate_images_masks(images=masks)

        self.labels = labels

        # if necessary Move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=ioconfig)
        self.model = model_to(model=self.model, device=self.device)

        save_dir = prepare_engines_save_dir(
            save_dir,
            len(self.images),
            patch_mode=patch_mode,
            overwrite=overwrite,
        )

        if patch_mode:
            data_loader = self.pre_process_patches(
                self.images,
                self.labels,
            )
            raw_predictions = self.infer_patches(
                data_loader=data_loader,
            )
            return self.post_process_patches(
                raw_predictions=raw_predictions,
                output_type=output_type,
            )

        return {"save_dir": save_dir}
