"""Defines Abstract Base Class for TIAToolbox Model Engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import torch
import tqdm
from torch import nn

from tiatoolbox import logger
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.dataset.dataset_abc import PatchDataset
from tiatoolbox.models.models_abc import load_torch_model, model_to

if TYPE_CHECKING:  # pragma: no cover
    import os

    import numpy as np
    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore

    from .io_config import ModelIOConfigABC


def prepare_engines_save_dir(
    save_dir: os | Path | None,
    len_images: int,
    *,
    patch_mode: bool,
) -> Path | None:
    """Create directory if not defined and number of images is more than 1.

    Args:
        save_dir (str or Path):
            Path to output directory.
        len_images (int):
            List of inputs to process.
        patch_mode(bool):
            Whether to treat input image as a patch or WSI.

    Returns:
        :class:`Path`:
            Path to output directory.

    """
    if patch_mode is True:
        save_dir.mkdir(parents=True, exist_ok=False)
        return save_dir

    if save_dir is None:
        if len_images > 1:
            msg = (
                "More than 1 WSIs detected but there is no save directory provided."
                "Please provide a 'save_dir'."
                "All subsequent output will be saved to current runtime"
                "location under folder 'Path.cwd() / output'. "
                "The output might be overwritten!",
            )
            raise OSError(msg)
        return (
            Path.cwd()
        )  # save the output to current working directory and return save_dir

    if len_images > 1:
        logger.info(
            "When providing multiple whole-slide images / tiles, "
            "the outputs will be saved and the locations of outputs "
            "will be returned to the calling function.",
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    return save_dir


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox.

    Args:
        model (nn.Module):
            Use externally defined PyTorch model for prediction with.
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case-insensitive.
        weights (str or Path):
            Path to the weight of the corresponding `pretrained_model`.

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
        on_gpu (bool):

        verbose (bool):
            Whether to output logging information.

    Attributes:
        images (str or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
            A HWC image or a path to WSI.
        mode (str):
            Type of input to process. Choose from either `patch`, `tile`
            or `wsi`.
        model (str | nn.Module):
            Defined PyTorch model.
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case-insensitive.
        ioconfig (ModelIOConfigABC):
            Input IO configuration to run the Engine.
        _ioconfig ():
            Runtime ioconfig.
        return_probabilities (bool):
            Whether to return per-class probabilities.
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
            Stride using during tile and WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        batch_size (int):
            Number of images fed into the model each time.
        labels:
            List of labels. If using `tile` or `wsi` mode, then only
            a single label per image tile or whole-slide image is
            supported.
        on_gpu (bool):
            Whether to run model on the GPU. Default is False.
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

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.run(tile_file, patch_mode=False)

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
        on_gpu: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize Engine."""
        super().__init__()

        self.masks = None
        self.images = None
        self.mode = None
        self.on_gpu = on_gpu

        # Initialize model with specified weights and ioconfig.
        self.model, self.ioconfig = self._initialize_model_ioconfig(
            model=model,
            weights=weights,
        )
        self.model = model_to(model=self.model, on_gpu=self.on_gpu)
        self._ioconfig = self.ioconfig  # runtime ioconfig

        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.verbose = verbose
        self.return_probabilities = False
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

    @abstractmethod
    def pre_process_patches(
        self: EngineABC,
        images: np.ndarray | list,
        labels: list,
    ) -> torch.utils.data.DataLoader:
        """Pre-process an image patch."""
        if labels and len(labels) != len(images):
            msg = f"len(labels) != len(imgs) : {len(labels)} != {len(images)}"
            raise ValueError(
                msg,
            )

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

    @abstractmethod
    def pre_process_wsi(self: EngineABC) -> NoReturn:
        """Pre-process a WSI."""
        raise NotImplementedError

    @abstractmethod
    def infer_patches(
        self: EngineABC,
        data_loader: DataLoader,
    ) -> AnnotationStore | np.ndarray | dict | str:
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
        output = {
            "predictions": [],
            "labels": [],
        }
        if self.return_probabilities:
            output["probabilities"] = []

        for _, batch_data in enumerate(data_loader):
            batch_output_probabilities = self.model.infer_batch(
                self.model,
                batch_data["image"],
                on_gpu=self.on_gpu,
            )
            # We get the index of the class with the maximum probability
            batch_output_predictions = self.model.postproc_func(
                batch_output_probabilities,
            )

            output["predictions"].extend(batch_output_predictions.tolist())

            # tolist might be very expensive
            if self.return_probabilities:
                output["probabilities"].extend(batch_output_probabilities.tolist())

            if self.return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                output["labels"].extend(list(batch_data["label"]))

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return output

    @abstractmethod
    def infer_wsi(self: EngineABC) -> NoReturn:
        """Model inference on a WSI."""
        raise NotImplementedError

    @abstractmethod
    def post_process_patch(self: EngineABC) -> NoReturn:
        """Post-process an image patch."""
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

    def run(
        self: EngineABC,
        images: list[os | Path] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        # output_type can be np.ndarray, Annotation or Json str
        # output_type: str = "Annotation",  # noqa: ERA001
        **kwargs: dict,
    ) -> AnnotationStore | np.ndarray | dict | str:
        """Run the engine on input images.

        Args:
            images (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list. When using
                `tile` or `wsi` mode, the input must be a list of file
                paths.
            masks (list | None):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for
                whole-slide images or the entire image is processed for
                image tiles.
            labels (list | None):
                List of labels. If using `tile` or `wsi` mode, then only
                a single label per image tile or whole-slide image is
                supported.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.
                default = True.
            ioconfig (IOPatchPredictorConfig):
                IO configuration.
            save_dir (str or pathlib.Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            output_type (str):
                Whether to save output for a single file. default=False
            **kwargs (dict):
                Keyword Args to update :class:`EngineABC` attributes.

        Returns:
            (:class:`numpy.ndarray`, dict):
                Model predictions of the input dataset. If multiple
                image tiles or whole-slide images are provided as input,
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

        self.images = images
        self.masks = masks
        self.labels = labels

        # if necessary Move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=ioconfig)
        self.model = model_to(model=self.model, on_gpu=self.on_gpu)

        save_dir = prepare_engines_save_dir(
            save_dir,
            len(self.images),
            patch_mode=patch_mode,
        )

        if patch_mode:
            data_loader = self.pre_process_patches(
                self.images,
                self.labels,
            )
            return self.infer_patches(
                data_loader=data_loader,
            )

        return {"save_dir": save_dir}
