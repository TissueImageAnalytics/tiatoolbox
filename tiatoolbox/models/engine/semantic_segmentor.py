"""This module implements semantic segmentation."""

from __future__ import annotations

import copy
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import joblib
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm

from tiatoolbox import logger, rcParam
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model
from tiatoolbox.models.models_abc import IOConfigABC, model_to
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import imread
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from multiprocessing.managers import Namespace

    from tiatoolbox.typing import IntPair, Resolution, Units


def _estimate_canvas_parameters(
    sample_prediction: np.ndarray,
    canvas_shape: np.ndarray,
) -> tuple[tuple, tuple, bool]:
    """Estimates canvas parameters.

    Args:
        sample_prediction (:class:`numpy.ndarray`):
            Patch prediction assuming to be of shape HWC.
        canvas_shape (:class:`numpy.ndarray`):
            HW of the supposed assembled image.

    Returns:
        (tuple, tuple, bool):
            Canvas Shape, Canvas Count and whether to add singleton dimension.

    """
    if len(sample_prediction.shape) == 3:  # noqa: PLR2004
        num_output_ch = sample_prediction.shape[-1]
        canvas_cum_shape_ = (*tuple(canvas_shape), num_output_ch)
        canvas_count_shape_ = (*tuple(canvas_shape), 1)
        add_singleton_dim = num_output_ch == 1
    else:
        canvas_cum_shape_ = (*tuple(canvas_shape), 1)
        canvas_count_shape_ = (*tuple(canvas_shape), 1)
        add_singleton_dim = True

    return canvas_cum_shape_, canvas_count_shape_, add_singleton_dim


def _prepare_save_output(
    save_path: str | Path,
    cache_count_path: str | Path,
    canvas_cum_shape_: tuple[int, ...],
    canvas_count_shape_: tuple[int, ...],
) -> tuple:
    """Prepares for saving the cached output."""
    if save_path is not None:
        save_path = Path(save_path)
        cache_count_path = Path(cache_count_path)
        if Path.exists(save_path) and Path.exists(cache_count_path):
            cum_canvas = np.load(str(save_path), mmap_mode="r+")
            count_canvas = np.load(str(cache_count_path), mmap_mode="r+")
            if canvas_cum_shape_ != cum_canvas.shape:
                msg = "Existing image shape in `save_path` does not match."
                raise ValueError(msg)
            if canvas_count_shape_ != count_canvas.shape:
                msg = "Existing image shape in `cache_count_path` does not match."
                raise ValueError(
                    msg,
                )
        else:
            cum_canvas = np.lib.format.open_memmap(
                save_path,
                mode="w+",
                shape=canvas_cum_shape_,
                dtype=np.float32,
            )
            # assuming no more than 255 overlapping times
            count_canvas = np.lib.format.open_memmap(
                cache_count_path,
                mode="w+",
                shape=canvas_count_shape_,
                dtype=np.uint8,
            )
            # flush fill
            count_canvas[:] = 0
        is_on_drive = True
    else:
        is_on_drive = False
        cum_canvas = np.zeros(
            shape=canvas_cum_shape_,
            dtype=np.float32,
        )
        # for pixel occurrence counting
        count_canvas = np.zeros(canvas_count_shape_, dtype=np.float32)

    return is_on_drive, count_canvas, cum_canvas


class IOSegmentorConfig(IOConfigABC):
    """Contain semantic segmentor input and output information.

    Args:
        input_resolutions (list):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        output_resolutions (list):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        patch_input_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest input in (height, width).
        patch_output_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest output in (height, width).
        save_resolution (dict):
            Resolution to save all output.

    Examples:
        >>> # Defining io for a network having 1 input and 1 output at the
        >>> # same resolution
        >>> ioconfig = IOSegmentorConfig(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=[2048, 2048],
        ...     patch_output_shape=[1024, 1024],
        ...     stride_shape=[512, 512],
        ... )

    Examples:
        >>> # Defining io for a network having 3 input and 2 output
        >>> # at the same resolution, the output is then merged at a
        >>> # different resolution.
        >>> ioconfig = IOSegmentorConfig(
        ...     input_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...         {"units": "mpp", "resolution": 0.75},
        ...     ],
        ...     output_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...     ],
        ...     patch_input_shape=[2048, 2048],
        ...     patch_output_shape=[1024, 1024],
        ...     stride_shape=[512, 512],
        ...     save_resolution={"units": "mpp", "resolution": 4.0},
        ... )

    """

    # We pre-define to follow enforcement, actual initialisation in init
    input_resolutions = None
    output_resolutions = None

    def __init__(
        self: IOSegmentorConfig,
        input_resolutions: list[dict],
        output_resolutions: list[dict],
        patch_input_shape: IntPair,
        patch_output_shape: IntPair,
        save_resolution: dict | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize :class:`IOSegmentorConfig`."""
        self._kwargs = kwargs
        self.patch_input_shape = patch_input_shape
        self.patch_output_shape = patch_output_shape
        self.stride_shape = None
        self.input_resolutions = input_resolutions
        self.output_resolutions = output_resolutions

        self.resolution_unit = input_resolutions[0]["units"]
        self.save_resolution = save_resolution

        for variable, value in kwargs.items():
            self.__setattr__(variable, value)

        self._validate()

        if self.resolution_unit == "mpp":
            self.highest_input_resolution = min(
                self.input_resolutions,
                key=lambda x: x["resolution"],
            )
        else:
            self.highest_input_resolution = max(
                self.input_resolutions,
                key=lambda x: x["resolution"],
            )

    def _validate(self: IOSegmentorConfig) -> None:
        """Validate the data format."""
        resolutions = self.input_resolutions + self.output_resolutions
        units = [v["units"] for v in resolutions]
        units = np.unique(units)
        if len(units) != 1 or units[0] not in [
            "power",
            "baseline",
            "mpp",
        ]:
            msg = f"Invalid resolution units `{units[0]}`."
            raise ValueError(msg)

    @staticmethod
    def scale_to_highest(resolutions: list[dict], units: Units) -> np.ndarray:
        """Get the scaling factor from input resolutions.

        This will convert resolutions to a scaling factor with respect to
        the highest resolution found in the input resolutions list.

        Args:
            resolutions (list):
                A list of resolutions where one is defined as
                `{'resolution': value, 'unit': value}`
            units (Units):
                Units that the resolutions are at.

        Returns:
            :class:`numpy.ndarray`:
                A 1D array of scaling factors having the same length as
                `resolutions`

        """
        old_val = [v["resolution"] for v in resolutions]
        if units not in ["baseline", "mpp", "power"]:
            msg = (
                f"Unknown units `{units}`. "
                f"Units should be one of 'baseline', 'mpp' or 'power'."
            )
            raise ValueError(
                msg,
            )
        if units == "baseline":
            return old_val
        if units == "mpp":
            return np.min(old_val) / np.array(old_val)
        return np.array(old_val) / np.max(old_val)

    def to_baseline(self: IOSegmentorConfig) -> IOSegmentorConfig:
        """Return a new config object converted to baseline form.

        This will return a new :class:`IOSegmentorConfig` where
        resolutions have been converted to baseline format with the
        highest possible resolution found in both input and output as
        reference.

        """
        resolutions = self.input_resolutions + self.output_resolutions
        if self.save_resolution is not None:
            resolutions.append(self.save_resolution)

        scale_factors = self.scale_to_highest(resolutions, self.resolution_unit)
        num_input_resolutions = len(self.input_resolutions)
        num_output_resolutions = len(self.output_resolutions)

        end_idx = num_input_resolutions
        input_resolutions = [
            {"units": "baseline", "resolution": v} for v in scale_factors[:end_idx]
        ]
        end_idx = num_input_resolutions + num_output_resolutions
        output_resolutions = [
            {"units": "baseline", "resolution": v}
            for v in scale_factors[num_input_resolutions:end_idx]
        ]

        save_resolution = None
        if self.save_resolution is not None:
            save_resolution = {"units": "baseline", "resolution": scale_factors[-1]}
        return IOSegmentorConfig(
            input_resolutions=input_resolutions,
            output_resolutions=output_resolutions,
            patch_input_shape=self.patch_input_shape,
            patch_output_shape=self.patch_output_shape,
            save_resolution=save_resolution,
            **self._kwargs,
        )


class WSIStreamDataset(torch_data.Dataset):
    """Reading a wsi in parallel mode with persistent workers.

    To speed up the inference process for multiple WSIs. The
    `torch.utils.data.Dataloader` is set to run in persistent mode.
    Normally, this will prevent workers from altering their initial
    states (such as provided input etc.). To sidestep this, we use a
    shared parallel workspace context manager to send data and signal
    from the main thread, thus allowing each worker to load a new wsi as
    well as corresponding patch information.

    Args:
        mp_shared_space (:class:`Namespace`):
            A shared multiprocessing space, must be from
            `torch.multiprocessing`.
        ioconfig (:class:`IOSegmentorConfig`):
            An object which contains I/O placement for patches.
        wsi_paths (list): List of paths pointing to a WSI or tiles.
        preproc (Callable):
            Pre-processing function to be applied to a patch.
        mode (str):
            Either `"wsi"` or `"tile"` to indicate the format of images
            in `wsi_paths`.

    Examples:
        >>> ioconfig = IOSegmentorConfig(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=[2048, 2048],
        ...     patch_output_shape=[1024, 1024],
        ...     stride_shape=[512, 512],
        ... )
        >>> mp_manager = torch_mp.Manager()
        >>> mp_shared_space = mp_manager.Namespace()
        >>> mp_shared_space.signal = 1  # adding variable to the shared space
        >>> wsi_paths = ['A.svs', 'B.svs']
        >>> wsi_dataset = WSIStreamDataset(ioconfig, wsi_paths, mp_shared_space)

    """

    def __init__(
        self: WSIStreamDataset,
        ioconfig: IOSegmentorConfig,
        wsi_paths: list[str | Path],
        mp_shared_space: Namespace,
        preproc: Callable[[np.ndarray], np.ndarray] | None = None,
        mode: str = "wsi",
    ) -> None:
        """Initialize :class:`WSIStreamDataset`."""
        super().__init__()
        self.mode = mode
        self.preproc = preproc
        self.ioconfig = copy.deepcopy(ioconfig)

        if mode == "tile":
            logger.warning(
                "WSIPatchDataset only reads image tile at "
                '`units="baseline"`. Resolutions will be converted '
                "to baseline value.",
                stacklevel=2,
            )
            self.ioconfig = self.ioconfig.to_baseline()

        self.mp_shared_space = mp_shared_space
        self.wsi_paths = wsi_paths
        self.wsi_idx = None  # to be received externally via thread communication
        self.reader = None

    def _get_reader(self: WSIStreamDataset, img_path: str | Path) -> WSIReader:
        """Get appropriate reader for input path."""
        img_path = Path(img_path)
        if self.mode == "wsi":
            return WSIReader.open(img_path)
        img = imread(img_path)
        # initialise metadata for VirtualWSIReader.
        # here, we simulate a whole-slide image, but with a single level.
        metadata = WSIMeta(
            mpp=np.array([1.0, 1.0]),
            objective_power=10,
            axes="YXS",
            slide_dimensions=np.array(img.shape[:2][::-1]),
            level_downsamples=[1.0],
            level_dimensions=[np.array(img.shape[:2][::-1])],
        )
        return VirtualWSIReader(
            img,
            info=metadata,
        )

    def __len__(self: WSIStreamDataset) -> int:
        """Return the length of the instance attributes."""
        return len(self.mp_shared_space.patch_inputs)

    @staticmethod
    def collate_fn(batch: list | np.ndarray) -> torch.Tensor:
        """Prototype to handle reading exception.

        This will exclude any sample with `None` from the batch. As
        such, wrapping `__getitem__` with try-catch and return `None`
        upon exceptions will prevent crashing the entire program. But as
        a side effect, the batch may not have the size as defined.

        """
        batch = [v for v in batch if v is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self: WSIStreamDataset, idx: int) -> tuple:
        """Get an item from the dataset."""
        # ! no need to lock as we do not modify source value in shared space
        if self.wsi_idx != self.mp_shared_space.wsi_idx:
            self.wsi_idx = int(self.mp_shared_space.wsi_idx.item())
            self.reader = self._get_reader(self.wsi_paths[self.wsi_idx])

        # this is in XY and at requested resolution (not baseline)
        bounds = self.mp_shared_space.patch_inputs[idx]
        bounds = bounds.numpy()  # expected to be a torch.Tensor

        # be the same as bounds br-tl, unless bounds are of float
        patch_data_ = []
        scale_factors = self.ioconfig.scale_to_highest(
            self.ioconfig.input_resolutions,
            self.ioconfig.resolution_unit,
        )
        for idy, resolution in enumerate(self.ioconfig.input_resolutions):
            resolution_bounds = np.round(bounds * scale_factors[idy])
            patch_data = self.reader.read_bounds(
                resolution_bounds.astype(np.int32),
                coord_space="resolution",
                pad_constant_values=0,  # expose this ?
                **resolution,
            )

            if self.preproc is not None:
                patch_data = patch_data.copy()
                patch_data = self.preproc(patch_data)
            patch_data_.append(patch_data)
        if len(patch_data_) == 1:
            patch_data_ = patch_data_[0]

        bound = self.mp_shared_space.patch_outputs[idx]
        return patch_data_, bound


class SemanticSegmentor:
    """Pixel-wise segmentation predictor.

    The tiatoolbox model should produce the following results on the BCSS dataset
    using fcn_resnet50_unet-bcss.

    .. list-table:: Semantic segmentation performance on the BCSS dataset
       :widths: 15 15 15 15 15 15 15
       :header-rows: 1

       * -
         - Tumour
         - Stroma
         - Inflammatory
         - Necrosis
         - Other
         - All
       * - Amgad et al.
         - 0.851
         - 0.800
         - 0.712
         - 0.723
         - 0.666
         - 0.750
       * - TIAToolbox
         - 0.885
         - 0.825
         - 0.761
         - 0.765
         - 0.581
         - 0.763

    Note, if `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments.

    Args:
        model (nn.Module):
            Use externally defined PyTorch model for prediction with
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_.
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case-insensitive.
        pretrained_weights (str):
            Path to the weight of the corresponding `pretrained_model`.
        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data. Take note that they will
            also perform preprocessing.
        num_postproc_workers (int):
            This value is there to maintain input compatibility with
            `tiatoolbox.models.classification` and is not used.
        verbose (bool):
            Whether to output logging information.
        dataset_class (obj):
            Dataset class to be used instead of default.
        auto_generate_mask (bool):
            To automatically generate tile/WSI tissue mask if is not
            provided.

    Attributes:
        process_prediction_per_batch (bool):
            A flag to denote whether post-processing for inference
            output is applied after each batch or after finishing an entire
            tile or WSI.

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = SemanticSegmentor(model='fcn-tissue_mask')
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0.raw') , ('B/wsi.svs', 'output/1.raw')]
        >>> # if a network have 2 output heads, each head output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.raw.0', 'output/0.raw.1'

    """

    def __init__(
        self: SemanticSegmentor,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_postproc_workers: int = 0,
        model: torch.nn.Module | None = None,
        pretrained_model: str | None = None,
        pretrained_weights: str | None = None,
        dataset_class: Callable = WSIStreamDataset,
        *,
        verbose: bool = True,
        auto_generate_mask: bool = False,
    ) -> None:
        """Initialize :class:`SemanticSegmentor`."""
        super().__init__()

        if model is None and pretrained_model is None:
            msg = "Must provide either of `model` or `pretrained_model`"
            raise ValueError(msg)

        if model is not None:
            self.model = model
            # template ioconfig, usually coming from pretrained
            self.ioconfig = None
        else:
            model, ioconfig = get_pretrained_model(pretrained_model, pretrained_weights)
            self.ioconfig = ioconfig
            self.model = model

        # local variables for flagging mode within class,
        # subclass should have overwritten to alter some specific behavior
        self.process_prediction_per_batch = True

        # for runtime, such as after wrapping with nn.DataParallel
        self._cache_dir = None
        self._loader = None
        self._model = None
        self._device = None
        self._mp_shared_space = None
        self._postproc_workers = None
        self.num_postproc_workers = num_postproc_workers
        self._futures = None
        self._outputs = []
        self.imgs = None
        self.masks = None

        self.dataset_class: WSIStreamDataset = dataset_class
        self.model = compile_model(
            model,
            mode=rcParam["torch_compile_mode"],
        )
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.num_postproc_workers = None
        self.verbose = verbose
        self.auto_generate_mask = auto_generate_mask

    @staticmethod
    def get_coordinates(
        image_shape: tuple[int, int] | np.ndarray,
        ioconfig: IOSegmentorConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate patch tiling coordinates.

        By default, internally, it will call the
        `PatchExtractor.get_coordinates`. To use your own approach,
        either subclass to overwrite or directly assign your own
        function to this name. In either cases, the function must obey
        the API defined here.

        Args:
            image_shape (tuple(int), :class:`numpy.ndarray`):
                This argument specifies the shape of mother image (the
                image we want to extract patches from) at requested
                `resolution` and `units` and it is expected to be in
                (width, height) format.
            ioconfig (:class:`IOSegmentorConfig`):
                Object that contains information about input and output
                placement of patches. Check `IOSegmentorConfig` for
                details about available attributes.

        Returns:
            tuple:
                List of patch inputs and outputs

                - :py:obj:`list` - patch_inputs:
                    A list of corrdinates in `[start_x, start_y, end_x,
                    end_y]` format indicating the read location of the
                    patch in the mother image.

                - :py:obj:`list` - patch_outputs:
                    A list of corrdinates in `[start_x, start_y, end_x,
                    end_y]` format indicating to write location of the
                    patch in the mother image.

        Examples:
            >>> # API of function expected to overwrite `get_coordinates`
            >>> def func(image_shape, ioconfig):
            ...   patch_inputs = np.array([[0, 0, 256, 256]])
            ...   patch_outputs = np.array([[0, 0, 256, 256]])
            ...   return patch_inputs, patch_outputs
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.get_coordinates = func

        """
        results = PatchExtractor.get_coordinates(
            patch_output_shape=ioconfig.patch_output_shape,
            image_shape=image_shape,
            patch_input_shape=ioconfig.patch_input_shape,
            stride_shape=ioconfig.stride_shape,
        )
        return results[0], results[1]

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Resolution | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
        """Indicates which coordinate is valid basing on the mask.

        To use your own approaches, either subclass to overwrite or
        directly assign your own function to this name. In either cases,
        the function must obey the API defined here.

        Args:
            mask_reader (:class:`.VirtualReader`):
                A virtual pyramidal reader of the mask related to the
                WSI from which we want to extract the patches.
            bounds (ndarray and np.int32):
                Coordinates to be checked via the `func`. They must be
                in the same resolution as requested `resolution` and
                `units`. The shape of `coordinates` is (N, K) where N is
                the number of coordinate sets and K is either 2 for
                centroids or 4 for bounding boxes. When using the
                default `func=None`, K should be 4, as we expect the
                `coordinates` to be bounding boxes in `[start_x,
                start_y, end_x, end_y]` format.
            resolution (Resolution):
                Resolution of the requested patch.
            units (Units):
                Units of the requested patch.

        Returns:
            :class:`numpy.ndarray`:
                List of flags to indicate which coordinate is valid.

        Examples:
            >>> # API of function expected to overwrite `filter_coordinates`
            >>> def func(reader, bounds, resolution, units):
            ...   # as example, only select first bound
            ...   return np.array([1, 0])
            >>> coords = [[0, 0, 256, 256], [128, 128, 384, 384]]
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.filter_coordinates = func

        """
        if not isinstance(mask_reader, VirtualWSIReader):
            msg = "`mask_reader` should be VirtualWSIReader."
            raise TypeError(msg)

        if not isinstance(bounds, np.ndarray) or not np.issubdtype(
            bounds.dtype,
            np.integer,
        ):
            msg = "`coordinates` should be ndarray of integer type."
            raise ValueError(msg)

        mask_real_shape = mask_reader.img.shape[:2]
        mask_resolution_shape = mask_reader.slide_dimensions(
            resolution=resolution,
            units=units,
        )[::-1]
        mask_real_shape = np.array(mask_real_shape)
        mask_resolution_shape = np.array(mask_resolution_shape)
        scale_factor = mask_real_shape / mask_resolution_shape
        scale_factor = scale_factor[0]  # what if ratio x != y

        def sel_func(coord: np.ndarray) -> bool:
            """Accept coord as long as its box contains part of mask."""
            coord_in_real_mask = np.ceil(scale_factor * coord).astype(np.int32)
            start_x, start_y, end_x, end_y = coord_in_real_mask
            roi = mask_reader.img[start_y:end_y, start_x:end_x]
            return np.sum(roi > 0) > 0

        flags = [sel_func(bound) for bound in bounds]
        return np.array(flags)

    @staticmethod
    def get_reader(
        img_path: str | Path,
        mask_path: str | Path,
        mode: str,
        *,
        auto_get_mask: bool,
    ) -> tuple[WSIReader, WSIReader]:
        """Define how to get reader for mask and source image."""
        img_path = Path(img_path)
        reader = WSIReader.open(img_path)

        mask_reader = None
        if mask_path is not None:
            mask_path = Path(mask_path)
            if not Path.is_file(mask_path):
                msg = "`mask_path` must be a valid file path."
                raise ValueError(msg)
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            mask_reader.info = reader.info
        return reader, mask_reader

    def _predict_one_wsi(
        self: SemanticSegmentor,
        wsi_idx: int,
        ioconfig: IOSegmentorConfig,
        save_path: str,
        mode: str,
    ) -> None:
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int):
                Index of the tile/wsi to be processed within `self`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object which defines I/O placement during inference and
                when assembling back to full tile/wsi.
            save_path (str):
                Location to save output prediction as well as possible
                intermediate results.
            mode (str):
                Either `"tile"` or `"wsi"` to indicate run mode.

        """
        cache_dir = self._cache_dir / str(wsi_idx)
        cache_dir.mkdir(parents=True)

        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path,
            mask_path,
            mode,
            auto_get_mask=self.auto_generate_mask,
        )

        # assume ioconfig has already been converted to `baseline` for `tile` mode
        resolution = ioconfig.highest_input_resolution
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch and tile placement
        # this is in XY
        (patch_inputs, patch_outputs) = self.get_coordinates(wsi_proc_shape, ioconfig)
        if mask_reader is not None:
            sel = self.filter_coordinates(mask_reader, patch_outputs, **resolution)
            patch_outputs = patch_outputs[sel]
            patch_inputs = patch_inputs[sel]

        # modify the shared space so that we can update worker info
        # without needing to re-create the worker. There should be no
        # race-condition because only the following enumerate loop
        # triggers the parallelism, and this portion is still in
        # sequential execution order
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(self._loader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for _, batch_data in enumerate(self._loader):
            sample_datas, sample_infos = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)
            sample_outputs = self.model.infer_batch(
                self._model,
                sample_datas,
                device=self._device,
            )
            # repackage so that it's an N list, each contains
            # L x etc. output
            sample_outputs = [np.split(v, batch_size, axis=0) for v in sample_outputs]
            sample_outputs = list(zip(*sample_outputs))

            # tensor to numpy, costly?
            sample_infos = sample_infos.numpy()
            sample_infos = np.split(sample_infos, batch_size, axis=0)

            sample_outputs = list(zip(sample_infos, sample_outputs))
            if self.process_prediction_per_batch:
                self._process_predictions(
                    sample_outputs,
                    wsi_reader,
                    ioconfig,
                    save_path,
                    cache_dir,
                )
            else:
                cum_output.extend(sample_outputs)
            pbar.update()
        pbar.close()

        self._process_predictions(
            cum_output,
            wsi_reader,
            ioconfig,
            save_path,
            cache_dir,
        )

        # clean up the cache directories
        shutil.rmtree(cache_dir)

    def _process_predictions(
        self: SemanticSegmentor,
        cum_batch_predictions: list,
        wsi_reader: WSIReader,
        ioconfig: IOSegmentorConfig,
        save_path: str,
        cache_dir: str,
    ) -> None:
        """Define how the aggregated predictions are processed.

        This includes merging the prediction if necessary and also saving afterwards.
        Note that items within `cum_batch_predictions` will be consumed during
        the operation.

        Args:
            cum_batch_predictions (list):
                List of batch predictions. Each item within the list
                should be of (location, patch_predictions).
            wsi_reader (:class:`WSIReader`):
                A reader for the image where the predictions come from.
            ioconfig (:class:`IOSegmentorConfig`):
                A configuration object contains input and output
                information.
            save_path (str):
                Root path to save current WSI predictions.
            cache_dir (str):
                Root path to cache current WSI data.

        """
        if len(cum_batch_predictions) == 0:
            return

        # assume predictions is N, each item has L output element
        locations, predictions = list(zip(*cum_batch_predictions))
        # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of
        # output patch this can exceed the image bound at the requested
        # resolution remove singleton due to split.
        locations = np.array([v[0] for v in locations])
        for index, output_resolution in enumerate(ioconfig.output_resolutions):
            # assume resolution index to be in the same order as L
            merged_resolution = ioconfig.highest_input_resolution
            merged_locations = locations
            # ! location is w.r.t the highest resolution, hence still need conversion
            if ioconfig.save_resolution is not None:
                merged_resolution = ioconfig.save_resolution
                output_shape = wsi_reader.slide_dimensions(**output_resolution)
                merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                fx = merged_shape[0] / output_shape[0]
                merged_locations = np.ceil(locations * fx).astype(np.int64)
            merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
            # 0 idx is to remove singleton without removing other axes singleton
            to_merge_predictions = [v[index][0] for v in predictions]
            sub_save_path = f"{save_path}.raw.{index}.npy"
            sub_count_path = f"{cache_dir}/count.{index}.npy"
            self.merge_prediction(
                merged_shape[::-1],  # XY to YX
                to_merge_predictions,
                merged_locations,
                save_path=sub_save_path,
                cache_count_path=sub_count_path,
            )

    @staticmethod
    def merge_prediction(
        canvas_shape: tuple[int] | list[int] | np.ndarray,
        predictions: list[np.ndarray],
        locations: list | np.ndarray,
        save_path: str | Path | None = None,
        cache_count_path: str | Path | None = None,
    ) -> np.ndarray:
        """Merge patch-level predictions to form a 2-dimensional prediction map.

        When accumulating the raw prediction onto a same canvas (via
        calling the function multiple times), `save_path` and
        `cache_count_path` must be the same. If either of these two do
        not exist, the function will create new files. However, if
        `save_path` is `None`, the function will perform the
        accumulation using CPU-RAM as storage.

        Args:
            canvas_shape (:class:`numpy.ndarray`):
                HW of the supposed assembled image.
            predictions (list):
                List of :class:`np.ndarray`, each item is a patch prediction,
                assuming to be of shape HWC.
            locations (list):
                List of :class:`np.ndarray`, each item is the location of the patch
                at the same index within `predictions`. The location is
                in the to be assembled canvas and of the form
                `(top_left_x, top_left_y, bottom_right_x,
                bottom_right_x)`.
            save_path (str):
                Location to save the assembled image.
            cache_count_path (str):
                Location to store the canvas for counting how many times
                each pixel get overlapped when assembling.

        Returns:
            :class:`numpy.ndarray`:
                An image contains merged data.

        Examples:
        >>> SemanticSegmentor.merge_prediction(
        ...     canvas_shape=[4, 4],
        ...     predictions=[
        ...         np.full((2, 2), 1),
        ...         np.full((2, 2), 2)],
        ...     locations=[
        ...         [0, 0, 2, 2],
        ...         [2, 2, 4, 4]],
        ...     save_path=None,
        ... )
        ... array([[1, 1, 0, 0],
        ...        [1, 1, 0, 0],
        ...        [0, 0, 2, 2],
        ...        [0, 0, 2, 2]])

        """
        canvas_shape = np.array(canvas_shape)

        sample_prediction = predictions[0]

        if len(sample_prediction.shape) not in (2, 3):
            msg = f"Prediction is no HW or HWC: {sample_prediction.shape}."
            raise ValueError(msg)

        (
            canvas_cum_shape_,
            canvas_count_shape_,
            add_singleton_dim,
        ) = _estimate_canvas_parameters(sample_prediction, canvas_shape)

        is_on_drive, count_canvas, cum_canvas = _prepare_save_output(
            save_path,
            cache_count_path,
            canvas_cum_shape_,
            canvas_count_shape_,
        )

        def index(arr: np.ndarray, tl: np.ndarray, br: np.ndarray) -> np.ndarray:
            """Helper to shorten indexing."""
            return arr[tl[0] : br[0], tl[1] : br[1]]

        patch_infos = list(zip(locations, predictions))
        for _, patch_info in enumerate(patch_infos):
            # position is assumed to be in XY coordinate
            (bound_in_wsi, prediction) = patch_info
            # convert to XY to YX, and in tl, br
            tl_in_wsi = np.array(bound_in_wsi[:2][::-1])
            br_in_wsi = np.array(bound_in_wsi[2:][::-1])
            old_tl_in_wsi = tl_in_wsi.copy()

            # need to do conversion
            patch_shape_in_wsi = tuple(br_in_wsi - tl_in_wsi)
            # conversion to make cv2 happy
            prediction = prediction.astype(np.float32)
            prediction = cv2.resize(prediction, patch_shape_in_wsi[::-1])
            # ! cv2 resize will remove singleton !
            if add_singleton_dim:
                prediction = prediction[..., None]

            sel = tl_in_wsi < 0
            tl_in_wsi[sel] = 0

            if np.any(tl_in_wsi >= canvas_shape):
                continue

            sel = br_in_wsi > canvas_shape
            br_in_wsi[sel] = canvas_shape[sel]

            # re-calibrate the position in case patch passing the image bound
            br_in_patch = br_in_wsi - old_tl_in_wsi
            patch_actual_shape = br_in_wsi - tl_in_wsi
            tl_in_patch = br_in_patch - patch_actual_shape

            # now cropping the prediction region
            patch_pred = prediction[
                tl_in_patch[0] : br_in_patch[0],
                tl_in_patch[1] : br_in_patch[1],
            ]

            patch_count = np.ones(patch_pred.shape[:2])[..., None]
            if not is_on_drive:
                index(cum_canvas, tl_in_wsi, br_in_wsi)[:] += patch_pred
                index(count_canvas, tl_in_wsi, br_in_wsi)[:] += patch_count
            else:
                old_avg_pred = np.array(index(cum_canvas, tl_in_wsi, br_in_wsi))
                old_count = np.array(index(count_canvas, tl_in_wsi, br_in_wsi))
                # ! there will be precision error, but we have to live with this
                new_count = old_count + patch_count
                # retrieve old raw probabilities after summation
                old_raw_pred = old_avg_pred * old_count
                new_avg_pred = (old_raw_pred + patch_pred) / new_count
                index(cum_canvas, tl_in_wsi, br_in_wsi)[:] = new_avg_pred
                index(count_canvas, tl_in_wsi, br_in_wsi)[:] = new_count
        if not is_on_drive:
            cum_canvas /= count_canvas + 1.0e-6
        return cum_canvas

    @staticmethod
    def _prepare_save_dir(save_dir: str | Path | None) -> tuple[Path, Path]:
        """Prepare save directory and cache."""
        if save_dir is None:
            logger.warning(
                "Segmentor will only output to directory. "
                "All subsequent output will be saved to current runtime "
                "location under folder 'output'. Overwriting may happen! ",
                stacklevel=2,
            )
            save_dir = Path.cwd() / "output"

        save_dir = Path(save_dir).resolve()
        if save_dir.is_dir():
            msg = f"`save_dir` already exists! {save_dir}"
            raise ValueError(msg)
        save_dir.mkdir(parents=True)
        cache_dir = Path(f"{save_dir}/cache")
        Path.mkdir(cache_dir, parents=True)

        return save_dir, cache_dir

    def _update_ioconfig(
        self: SemanticSegmentor,
        ioconfig: IOSegmentorConfig,
        mode: str,
        patch_input_shape: IntPair,
        patch_output_shape: IntPair,
        stride_shape: IntPair,
        resolution: Resolution,
        units: Units,
    ) -> IOSegmentorConfig:
        """Update ioconfig according to input parameters.

        Args:
            ioconfig (:class:`IOSegmentorConfig`):
                Object defines information about input and output
                placement of patches. When provided,
                `patch_input_shape`, `patch_output_shape`,
                `stride_shape`, `resolution`, and `units` arguments are
                ignored. Otherwise, those arguments will be internally
                converted to a :class:`IOSegmentorConfig` object.
            mode (str):
                Type of input to process. Choose from either `tile` or
                `wsi`.
            patch_input_shape (tuple):
                Size of patches input to the model. The values
                are at requested read resolution and must be positive.
            patch_output_shape (tuple):
                Size of patches output by the model. The values are at
                the requested read resolution and must be positive.
            stride_shape (tuple):
                Stride using during tile and WSI processing. The values
                are at requested read resolution and must be positive.
                If not provided, `stride_shape=patch_input_shape` is
                used.
            resolution (Resolution):
                Resolution used for reading the image.
            units (Units):
                Units of resolution used for reading the image.

        Returns:
            :class:`IOSegmentorConfig`:
                Updated ioconfig.

        """
        if patch_output_shape is None:
            patch_output_shape = patch_input_shape
        if stride_shape is None:
            stride_shape = patch_output_shape

        if ioconfig is None and patch_input_shape is None:
            if self.ioconfig is None:
                msg = (
                    "Must provide either `ioconfig` or `patch_input_shape` "
                    "and `patch_output_shape`"
                )
                raise ValueError(
                    msg,
                )
            ioconfig = copy.deepcopy(self.ioconfig)
        elif ioconfig is None:
            ioconfig = IOSegmentorConfig(
                input_resolutions=[{"resolution": resolution, "units": units}],
                output_resolutions=[{"resolution": resolution, "units": units}],
                patch_input_shape=patch_input_shape,
                patch_output_shape=patch_output_shape,
                stride_shape=stride_shape,
            )
        if mode == "tile":
            logger.warning(
                "WSIPatchDataset only reads image tile at "
                '`units="baseline"`. Resolutions will be converted '
                "to baseline value.",
                stacklevel=2,
            )
            return ioconfig.to_baseline()

        return ioconfig

    def _prepare_workers(self: SemanticSegmentor) -> None:
        """Prepare number of workers."""
        self._postproc_workers = None
        if self.num_postproc_workers is not None:
            self._postproc_workers = ProcessPoolExecutor(
                max_workers=self.num_postproc_workers,
            )

    def _memory_cleanup(self: SemanticSegmentor) -> None:
        """Memory clean up."""
        self.imgs = None
        self.masks = None
        self._cache_dir = None
        self._model = None
        self._loader = None
        self._device = None
        self._futures = None
        self._mp_shared_space = None
        if self._postproc_workers is not None:
            self._postproc_workers.shutdown()
        self._postproc_workers = None

    def _predict_wsi_handle_exception(
        self: SemanticSegmentor,
        imgs: list,
        wsi_idx: int,
        img_path: str | Path,
        mode: str,
        ioconfig: IOSegmentorConfig,
        save_dir: str | Path,
        *,
        crash_on_exception: bool,
    ) -> None:
        """Predict on multiple WSIs.

        Args:
            imgs (list, ndarray):
                List of inputs to process. When using `"patch"` mode,
                the input must be either a list of images, a list of
                image file paths or a numpy array of an image list. When
                using `"tile"` or `"wsi"` mode, the input must be a list
                of file paths.
            wsi_idx (int):
                index of current WSI being processed.
            img_path(str or Path):
                Path to current image.
            mode (str):
                Type of input to process. Choose from either `tile` or
                `wsi`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object defines information about input and output
                placement of patches. When provided,
                `patch_input_shape`, `patch_output_shape`,
                `stride_shape`, `resolution`, and `units` arguments are
                ignored. Otherwise, those arguments will be internally
                converted to a :class:`IOSegmentorConfig` object.
            save_dir (str or Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            crash_on_exception (bool):
                If `True`, the running loop will crash if there is any
                error during processing a WSI. Otherwise, the loop will
                move on to the next wsi for processing.

        Returns:
            list:
                A list of tuple(input_path, save_path) where
                `input_path` is the path of the input wsi while
                `save_path` corresponds to the output predictions.

        """
        try:
            wsi_save_path = save_dir / f"{wsi_idx}"
            self._predict_one_wsi(wsi_idx, ioconfig, str(wsi_save_path), mode)

            # Do not use dict with file name as key, because it can be
            # overwritten. It may be user intention to provide files with a
            # same name multiple times (maybe they have different root path)
            self._outputs.append([str(img_path), str(wsi_save_path)])

            # ? will this corrupt old version if control + c midway?
            map_file_path = save_dir / "file_map.dat"
            # backup old version first
            if Path.exists(map_file_path):
                old_map_file_path = save_dir / "file_map_old.dat"
                shutil.copy(map_file_path, old_map_file_path)
            joblib.dump(self._outputs, map_file_path)

            # verbose mode, error by passing ?
            logging.info("Finish: %d", wsi_idx / len(imgs))
            logging.info("--Input: %s", str(img_path))
            logging.info("--Output: %s", str(wsi_save_path))
        # prevent deep source check because this is bypass and
        # delegating error message
        except Exception as err:  # skipcq: PYL-W0703
            wsi_save_path = save_dir.joinpath(f"{wsi_idx}")
            if crash_on_exception:
                raise err  # noqa: TRY201
            logging.exception("Crashed on %s", wsi_save_path)

    def predict(  # noqa: PLR0913
        self: SemanticSegmentor,
        imgs: list,
        masks: list | None = None,
        mode: str = "tile",
        ioconfig: IOSegmentorConfig = None,
        patch_input_shape: IntPair = None,
        patch_output_shape: IntPair = None,
        stride_shape: IntPair = None,
        resolution: Resolution = 1.0,
        units: Units = "baseline",
        save_dir: str | Path | None = None,
        device: str = "cpu",
        *,
        crash_on_exception: bool = False,
    ) -> list[tuple[Path, Path]]:
        """Make a prediction for a list of input data.

        By default, if the input model at the object instantiation time
        is a pretrained model in the toolbox as well as
        `patch_input_shape`, `patch_output_shape`, `stride_shape`,
        `resolution`, `units` and `ioconfig` are `None`. The method will
        use the `ioconfig` retrieved together with the pretrained model.
        Otherwise, either `patch_input_shape`, `patch_output_shape`,
        `stride_shape`, `resolution`, `units` or `ioconfig` must be set
        else a `Value Error` will be raised.

        Args:
            imgs (list, ndarray):
                List of inputs to process. When using `"patch"` mode,
                the input must be either a list of images, a list of
                image file paths or a numpy array of an image list. When
                using `"tile"` or `"wsi"` mode, the input must be a list
                of file paths.
            masks (list):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for
                whole-slide images or the entire image is processed for
                image tiles.
            mode (str):
                Type of input to process. Choose from either `tile` or
                `wsi`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object defines information about input and output
                placement of patches. When provided,
                `patch_input_shape`, `patch_output_shape`,
                `stride_shape`, `resolution`, and `units` arguments are
                ignored. Otherwise, those arguments will be internally
                converted to a :class:`IOSegmentorConfig` object.
            device (str):
                :class:`torch.device` to run the model.
                Select the device to run the model. Please see
                https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
                for more details on input parameters for device. Default value is "cpu".
            patch_input_shape (tuple):
                Size of patches input to the model. The values
                are at requested read resolution and must be positive.
            patch_output_shape (tuple):
                Size of patches output by the model. The values are at
                the requested read resolution and must be positive.
            stride_shape (tuple):
                Stride using during tile and WSI processing. The values
                are at requested read resolution and must be positive.
                If not provided, `stride_shape=patch_input_shape` is
                used.
            resolution (float):
                Resolution used for reading the image.
            units (Units):
                Units of resolution used for reading the image. Choose
                from either `"level"`, `"power"` or `"mpp"`.
            save_dir (str or pathlib.Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            crash_on_exception (bool):
                If `True`, the running loop will crash if there is any
                error during processing a WSI. Otherwise, the loop will
                move on to the next wsi for processing.

        Returns:
            list:
                A list of tuple(input_path, save_path) where
                `input_path` is the path of the input wsi while
                `save_path` corresponds to the output predictions.

        Examples:
            >>> # Sample output of a network
            >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
            >>> predictor = SemanticSegmentor(model='fcn-tissue_mask')
            >>> output = predictor.predict(wsis, mode='wsi')
            >>> list(output.keys())
            [('A/wsi.svs', 'output/0.raw') , ('B/wsi.svs', 'output/1.raw')]
            >>> # if a network have 2 output heads, each head output of 'A/wsi.svs'
            >>> # will be respectively stored in 'output/0.raw.0', 'output/0.raw.1'

        """
        if mode not in ["wsi", "tile"]:
            msg = f"{mode} is not a valid mode. Use either `tile` or `wsi`."
            raise ValueError(msg)

        save_dir, self._cache_dir = self._prepare_save_dir(save_dir)

        ioconfig = self._update_ioconfig(
            ioconfig,
            mode,
            patch_input_shape,
            patch_output_shape,
            stride_shape,
            resolution,
            units,
        )

        # use external for testing
        self._device = device
        self._model = model_to(model=self.model, device=device)

        # workers should be > 0 else Value Error will be thrown
        self._prepare_workers()

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        ds = self.dataset_class(
            ioconfig=ioconfig,
            preproc=self.model.preproc_func,
            wsi_paths=imgs,
            mp_shared_space=mp_shared_space,
            mode=mode,
        )

        loader = torch_data.DataLoader(
            ds,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_loader_workers,
            persistent_workers=self.num_loader_workers > 0,
        )

        self._loader = loader
        self.imgs = imgs
        self.masks = masks

        # contain input / output prediction mapping
        self._outputs = []
        # ? what will happen if this crash midway?
        # => may not be able to retrieve the result dict
        for wsi_idx, img_path in enumerate(imgs):
            self._predict_wsi_handle_exception(
                imgs=imgs,
                wsi_idx=wsi_idx,
                img_path=img_path,
                mode=mode,
                ioconfig=ioconfig,
                save_dir=save_dir,
                crash_on_exception=crash_on_exception,
            )

        # clean up the cache directories
        try:
            shutil.rmtree(self._cache_dir)
        except PermissionError:  # pragma: no cover
            logger.warning("Unable to remove %s", self._cache_dir)

        self._memory_cleanup()

        return self._outputs


class DeepFeatureExtractor(SemanticSegmentor):
    """Generic CNN Feature Extractor.

    AN engine for using any CNN model as a feature extractor. Note, if
    `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments.

    Args:
        model (nn.Module):
            Use externally defined PyTorch model for prediction with
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. By default, the corresponding
            pretrained weights will also be downloaded. However, you can
            override with your own set of weights via the
            `pretrained_weights` argument. Argument is case-insensitive.
            Refer to
            :class:`tiatoolbox.models.architecture.vanilla.CNNBackbone`
            for list of supported pretrained models.
        pretrained_weights (str):
            Path to the weight of the corresponding `pretrained_model`.
        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data. Take note that they will
            also perform preprocessing.
        num_postproc_workers (int):
            This value is there to maintain input compatibility with
            `tiatoolbox.models.classification` and is not used.
        verbose (bool):
            Whether to output logging information.
        dataset_class (obj):
            Dataset class to be used instead of default.
        auto_generate_mask(bool):
            To automatically generate tile/WSI tissue mask if is not
            provided.

    Examples:
        >>> # Sample output of a network
        >>> from tiatoolbox.models.architecture.vanilla import CNNBackbone
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> # create resnet50 with pytorch pretrained weights
        >>> model = CNNBackbone('resnet50')
        >>> predictor = DeepFeatureExtractor(model=model)
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
        >>> # If a network have 2 output heads, for 'A/wsi.svs',
        >>> # there will be 3 outputs, and they are respectively stored at
        >>> # 'output/0.position.npy'   # will always be output
        >>> # 'output/0.features.0.npy' # output of head 0
        >>> # 'output/0.features.1.npy' # output of head 1
        >>> # Each file will contain a same number of items, and the item at each
        >>> # index corresponds to 1 patch. The item in `.*position.npy` will
        >>> # be the corresponding patch bounding box. The box coordinates are at
        >>> # the inference resolution defined within the provided `ioconfig`.

    """

    def __init__(
        self: DeepFeatureExtractor,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_postproc_workers: int = 0,
        model: torch.nn.Module | None = None,
        pretrained_model: str | None = None,
        pretrained_weights: str | None = None,
        dataset_class: Callable = WSIStreamDataset,
        *,
        verbose: bool = True,
        auto_generate_mask: bool = False,
    ) -> None:
        """Initialize :class:`DeepFeatureExtractor`."""
        super().__init__(
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_postproc_workers=num_postproc_workers,
            model=model,
            pretrained_model=pretrained_model,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
            auto_generate_mask=auto_generate_mask,
            dataset_class=dataset_class,
        )
        self.process_prediction_per_batch = False

    def _process_predictions(
        self: DeepFeatureExtractor,
        cum_batch_predictions: list,
        wsi_reader: WSIReader,  # skipcq: PYL-W0613  # noqa: ARG002
        ioconfig: IOSegmentorConfig,
        save_path: str,
        cache_dir: str,  # skipcq: PYL-W0613  # noqa: ARG002
    ) -> None:
        """Define how the aggregated predictions are processed.

        This includes merging the prediction if necessary and also
        saving afterward.

        Args:
            cum_batch_predictions (list):
                List of batch predictions. Each item within the list
                should be of (location, patch_predictions).
            wsi_reader (:class:`WSIReader`):
                A reader for the image where the predictions come from.
                Not used here. Added for consistency with the API.
            ioconfig (:class:`IOSegmentorConfig`):
                A configuration object contains input and output
                information.
            save_path (str):
                Root path to save current WSI predictions.
            cache_dir (str):
                Root path to cache current WSI data.
                Not used here. Added for consistency with the API.

        """
        # assume prediction_list is N, each item has L output elements
        location_list, prediction_list = list(zip(*cum_batch_predictions))
        # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of output
        # patch, this can exceed the image bound at the requested resolution
        # remove singleton due to split.
        location_list = np.array([v[0] for v in location_list])
        np.save(f"{save_path}.position.npy", location_list)
        for idx, _ in enumerate(ioconfig.output_resolutions):
            # assume resolution idx to be in the same order as L
            # 0 idx is to remove singleton without removing other axes singleton
            prediction_list = [v[idx][0] for v in prediction_list]
            prediction_list = np.array(prediction_list)
            np.save(f"{save_path}.features.{idx}.npy", prediction_list)

    def predict(  # noqa: PLR0913
        self: DeepFeatureExtractor,
        imgs: list,
        masks: list | None = None,
        mode: str = "tile",
        ioconfig: IOSegmentorConfig | None = None,
        patch_input_shape: IntPair | None = None,
        patch_output_shape: IntPair | None = None,
        stride_shape: IntPair = None,
        resolution: Resolution = 1.0,
        units: Units = "baseline",
        save_dir: str | Path | None = None,
        device: str = "cpu",
        *,
        crash_on_exception: bool = False,
    ) -> list[tuple[Path, Path]]:
        """Make a prediction for a list of input data.

        By default, if the input model at the time of object
        instantiation is a pretrained model in the toolbox as well as
        `patch_input_shape`, `patch_output_shape`, `stride_shape`,
        `resolution`, `units` and `ioconfig` are `None`. The method will
        use the `ioconfig` retrieved together with the pretrained model.
        Otherwise, either `patch_input_shape`, `patch_output_shape`,
        `stride_shape`, `resolution`, `units` or `ioconfig` must be set
        - else a `Value Error` will be raised.

        Args:
            imgs (list, ndarray):
                List of inputs to process. When using `"patch"` mode,
                the input must be either a list of images, a list of
                image file paths or a numpy array of an image list. When
                using `"tile"` or `"wsi"` mode, the input must be a list
                of file paths.
            masks (list):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for each
                whole-slide image or all image tiles in the entire image
                are processed.
            mode (str):
                Type of input to process. Choose from either `tile` or
                `wsi`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object that defines information about input and output
                placement of patches. When provided,
                `patch_input_shape`, `patch_output_shape`,
                `stride_shape`, `resolution`, and `units` arguments are
                ignored. Otherwise, those arguments will be internally
                converted to a :class:`IOSegmentorConfig` object.
            device (str):
                :class:`torch.device` to run the model.
                Select the device to run the model. Please see
                https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
                for more details on input parameters for device. Default value is "cpu".
            patch_input_shape (IntPair):
                Size of patches input to the model. The values are at
                requested read resolution and must be positive.
            patch_output_shape (tuple):
                Size of patches output by the model. The values are at
                the requested read resolution and must be positive.
            stride_shape (tuple):
                Stride using during tile and WSI processing. The values
                are at requested read resolution and must be positive.
                If not provided, `stride_shape=patch_input_shape` is
                used.
            resolution (Resolution):
                Resolution used for reading the image.
            units (Units):
                Units of resolution used for reading the image.
            save_dir (str):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            crash_on_exception (bool):
                If `True`, the running loop will crash if there is any
                error during processing a WSI. Otherwise, the loop will
                move on to the next wsi for processing.

        Returns:
            list:
                A list of tuple(input_path, save_path) where
                `input_path` is the path of the input wsi while
                `save_path` corresponds to the output predictions.

        Examples:
            >>> # Sample output of a network
            >>> from tiatoolbox.models.architecture.vanilla import CNNBackbone
            >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
            >>> # create resnet50 with pytorch pretrained weights
            >>> model = CNNBackbone('resnet50')
            >>> predictor = DeepFeatureExtractor(model=model)
            >>> output = predictor.predict(wsis, mode='wsi')
            >>> list(output.keys())
            [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
            >>> # If a network have 2 output heads, for 'A/wsi.svs',
            >>> # there will be 3 outputs, and they are respectively stored at
            >>> # 'output/0.position.npy'   # will always be output
            >>> # 'output/0.features.0.npy' # output of head 0
            >>> # 'output/0.features.1.npy' # output of head 1
            >>> # Each file will contain a same number of items, and the item at each
            >>> # index corresponds to 1 patch. The item in `.*position.npy` will
            >>> # be the corresponding patch bounding box. The box coordinates are at
            >>> # the inference resolution defined within the provided `ioconfig`.

        """
        return super().predict(
            imgs=imgs,
            masks=masks,
            mode=mode,
            device=device,
            ioconfig=ioconfig,
            patch_input_shape=patch_input_shape,
            patch_output_shape=patch_output_shape,
            stride_shape=stride_shape,
            resolution=resolution,
            units=units,
            save_dir=save_dir,
            crash_on_exception=crash_on_exception,
        )
