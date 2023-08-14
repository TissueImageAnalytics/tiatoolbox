"""Defines Abstract Base Class for TIAToolbox Model Engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from tiatoolbox import logger
from tiatoolbox.models.architecture import get_pretrained_model

if TYPE_CHECKING:
    import os

    import numpy as np
    from torch import nn

    from tiatoolbox.annotation.storage import Annotation

    from .io_config import IOPatchPredictorConfig


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
        weights (str):
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
        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers used in torch.utils.data.DataLoader.
        verbose (bool):
            Whether to output logging information.

    Examples:
        >>> # list of 2 image patches as input
        >>> data = ["path/to/image1.svs", "path/to/image2.svs"]
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.run(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> import numpy as np
        >>> data = np.array([img1, img2])
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.run(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.run(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.run(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.run(wsi_file, mode='wsi')

    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize Engine."""
        super().__init__()

        self.images = None
        self.mode = None

        if model is not None:
            self.model = model
            ioconfig = None  # retrieve ioconfig from provided model.
        else:
            model, ioconfig = get_pretrained_model(model, weights)

        self.ioconfig = ioconfig  # for storing original
        self._ioconfig = self.ioconfig  # runtime ioconfig
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
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

    @abstractmethod
    def pre_process_patch(self):
        """Pre-process an image patch."""
        raise NotImplementedError

    @abstractmethod
    def pre_process_wsi(self):
        """Pre-process a WSI."""
        raise NotImplementedError

    @abstractmethod
    def infer_patch(self):
        """Model inference on an image patch."""
        raise NotImplementedError

    @abstractmethod
    def infer_wsi(self):
        """Model inference on a WSI."""
        raise NotImplementedError

    @abstractmethod
    def post_process_patch(self):
        """Post-process an image patch."""
        raise NotImplementedError

    @abstractmethod
    def post_process_wsi(self):
        """Post-process a WSI."""
        raise NotImplementedError

    @staticmethod
    def _prepare_save_dir(save_dir: os | Path, images: list | np.ndarray) -> Path:
        """Create directory if not defined and number of images is more than 1.

        Args:
            save_dir (str or Path):
                Path to output directory.
            images (list, ndarray):
                List of inputs to process.

        Returns:
            :class:`Path`:
                Path to output directory.

        """
        if save_dir is None and len(images) > 1:
            logger.warning(
                "More than 1 WSIs detected but there is no save directory set."
                "All subsequent output will be saved to current runtime"
                "location under folder 'output'. Overwriting may happen!",
                stacklevel=2,
            )
            save_dir = Path.cwd() / "output"
        elif save_dir is not None and len(images) > 1:
            logger.warning(
                "When providing multiple whole-slide images / tiles, "
                "the outputs will be saved and the locations of outputs"
                "will be returned"
                "to the calling function.",
                stacklevel=2,
            )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)
            return save_dir

        return Path.cwd() / "output"

    @abstractmethod
    def run(
        self,
        images: list[os | Path] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        ioconfig: IOPatchPredictorConfig | None = None,
        *,
        patch_mode: bool = False,
        on_gpu=True,
        save_dir=None,
        # None will not save output
        # save_output can be np.ndarray, Annotation or Json str
        save_output: np.ndarray | Annotation | str | None = True,
        **kwargs: dict,
    ) -> np.ndarray | dict:
        """Run the engine on input images.

        Args:
            images (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list. When using
                `tile` or `wsi` mode, the input must be a list of file
                paths.
            masks (list):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for
                whole-slide images or the entire image is processed for
                image tiles.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.
                default = False.
            on_gpu (bool):
                Whether to run model on the GPU.
            ioconfig (IOPatchPredictorConfig):
                Patch Predictor IO configuration.
            save_dir (str or pathlib.Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            save_output (bool):
                Whether to save output for a single file. default=False
            **kwargs (dict):
                Keyword Args for ...

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
            >>> output = predictor.run(wsis, mode="wsi")
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'raw': '0.raw.json', 'merged': '0.merged.npy'}
            >>> output['wsi2.svs']
            ... {'raw': '1.raw.json', 'merged': '1.merged.npy'}

        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        save_dir = self._prepare_save_dir(save_dir, images)

        return {"save_dir": save_dir}
