"""Defines Abstract Base Class for TIAToolbox Model Engines."""
from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from tiatoolbox.models.architecture import get_pretrained_model


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox.

    Args:
        model (nn.Module):
            Use externally defined PyTorch model for prediction with.
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case-insensitive.
        pretrained_weights (str):
            Path to the weight of the corresponding `pretrained_model`.

            >>> engine = EngineABC(
            ...    pretrained_model="pretrained-model-name",
            ...    pretrained_weights="pretrained-local-weights.pth")

        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. default = 0
        num_postproc_workers (int):
            Number of workers to postprocess the results of the model. default = 0
        verbose (bool):
            Whether to output logging information.

    Attributes:
        images (str or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
            A HWC image or a path to WSI.
        mode (str):
            Type of input to process. Choose from either `patch`, `tile`
            or `wsi`.
        model (nn.Module):
            Defined PyTorch model.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case insensitive.
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
        >>> output = engine.predict(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.predict(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> engine = EngineABC(pretrained_model="resnet18-kather100k")
        >>> output = engine.predict(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.predict(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> engine = EngineABC(pretraind_model="resnet18-kather100k")
        >>> output = engine.predict(wsi_file, mode='wsi')

    """

    def __init__(
        self,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_postproc_workers: int = 0,
        model: nn.Module = None,
        pretrained_model: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize Engine."""
        super().__init__()

        self.images = None
        self.mode = None

        if model is None and pretrained_model is None:
            msg = "Must provide either `model` or `pretrained_model`."
            raise ValueError(msg)

        if model is not None:
            self.model = model
            ioconfig = None  # retrieve ioconfig from provided model.
        else:
            model, ioconfig = get_pretrained_model(pretrained_model, pretrained_weights)

        self.ioconfig = ioconfig  # for storing original
        self._ioconfig = self.ioconfig  # runtime ioconfig
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.num_postproc_workers = num_postproc_workers
        self.verbose = verbose

    @abstractmethod
    def pre_process_patch(self):
        raise NotImplementedError

    @abstractmethod
    def pre_process_tile(self):
        raise NotImplementedError

    @abstractmethod
    def pre_process_wsi(self):
        raise NotImplementedError

    @abstractmethod
    def infer_patch(self):
        raise NotImplementedError

    @abstractmethod
    def infer_tile(self):
        raise NotImplementedError

    @abstractmethod
    def infer_wsi(self):
        raise NotImplementedError

    @abstractmethod
    def post_process_patch(self):
        raise NotImplementedError

    @abstractmethod
    def post_process_tile(self):
        raise NotImplementedError

    @abstractmethod
    def post_process_wsi(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
