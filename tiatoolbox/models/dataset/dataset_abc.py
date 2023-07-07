from __future__ import annotations

import copy
import os
import pathlib
from abc import ABC, abstractmethod
from multiprocessing.managers import Namespace
from typing import TYPE_CHECKING, Callable, List, Union

import cv2
import numpy as np
import torch
from torch.utils import data as torch_data

from tiatoolbox import logger
from tiatoolbox.tools.patchextraction import PatchExtractor

if TYPE_CHECKING:
    from tiatoolbox.models import IOSegmentorConfig

from tiatoolbox.utils import imread
from tiatoolbox.wsicore import WSIMeta, WSIReader
from tiatoolbox.wsicore.wsireader import VirtualWSIReader


class PatchDatasetABC(ABC, torch.utils.data.Dataset):
    """Defines abstract base class for patch dataset."""

    def __init__(
        self,
    ):
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []

    @staticmethod
    def _check_shape_integrity(shapes):
        """Checks the integrity of input shapes.

        Args:
            shapes (list or np.ndarray):
                input shape to check.

        Raises:
            ValueError: If the shape is not valid.

        """
        if any(len(v) != 3 for v in shapes):
            raise ValueError("Each sample must be an array of the form HWC.")

        max_shape = np.max(shapes, axis=0)
        if (shapes - max_shape[None]).sum() != 0:
            raise ValueError("Images must have the same dimensions.")

    def _check_input_integrity(self, mode):
        """Check that variables received during init are valid.

        These checks include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height
              and width.

        """
        if mode == "patch":
            self.data_is_npy_alike = False
            is_all_paths = all(isinstance(v, (pathlib.Path, str)) for v in self.inputs)
            is_all_npy = all(isinstance(v, np.ndarray) for v in self.inputs)

            if not (is_all_paths or is_all_npy or isinstance(self.inputs, np.ndarray)):
                raise ValueError(
                    "Input must be either a list/array of images "
                    "or a list of valid image paths."
                )

            shapes = None
            # When a list of paths is provided
            if is_all_paths:
                if any(not os.path.exists(v) for v in self.inputs):
                    # at least one of the paths are invalid
                    raise ValueError(
                        "Input must be either a list/array of images "
                        "or a list of valid image paths."
                    )
                # Preload test for sanity check
                shapes = [self.load_img(v).shape for v in self.inputs]
                self.data_is_npy_alike = False

            if is_all_npy:
                shapes = [v.shape for v in self.inputs]
                self.data_is_npy_alike = True

            if shapes:
                self._check_shape_integrity(shapes)

            # If input is a numpy array
            if isinstance(self.inputs, np.ndarray):
                # Check that input array is numerical
                if not np.issubdtype(self.inputs.dtype, np.number):
                    # ndarray of mixed data types
                    raise ValueError("Provided input array is non-numerical.")
                # N H W C | N C H W
                if len(self.inputs.shape) != 4:
                    raise ValueError(
                        "Input must be an array of images of the form NHWC. This can "
                        "be achieved by converting a list of images to a numpy array. "
                        " eg., np.array([img1, img2])."
                    )
                self.data_is_npy_alike = True

        elif not isinstance(self.inputs, (list, np.ndarray)):
            raise ValueError("`inputs` should be a list of patch coordinates.")

    @staticmethod
    def load_img(path):
        """Load an image from a provided path.

        Args:
            path (str): Path to an image file.

        """
        path = pathlib.Path(path)

        if path.suffix not in (".npy", ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            raise ValueError(f"Cannot load image data from `{path.suffix}` files.")

        return imread(path, as_uint8=False)

    @staticmethod
    def preproc(image):
        """Define the pre-processing of this class of loader."""
        return image

    @property
    def preproc_func(self):
        """Return the current pre-processing function of this instance.

        The returned function is expected to behave as follows:
        >>> transformed_img = func(img)

        """
        return self._preproc

    @preproc_func.setter
    def preproc_func(self, func):
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`.
        Otherwise, `func` is expected to be callable and behaves as
        follows:

        >>> transformed_img = func(img)

        """
        if func is None:
            self._preproc = self.preproc
        elif callable(func):
            self._preproc = func
        else:
            raise ValueError(f"{func} is not callable!")

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, idx):
        ...  # pragma: no cover


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
        self,
        ioconfig: IOSegmentorConfig,
        wsi_paths: List[Union[str, pathlib.Path]],
        mp_shared_space: Namespace,
        preproc: Callable[[np.ndarray], np.ndarray] = None,
        mode="wsi",
    ):
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

    def _get_reader(self, img_path):
        """Get appropriate reader for input path."""
        img_path = pathlib.Path(img_path)
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

    def __len__(self):
        return len(self.mp_shared_space.patch_inputs)

    @staticmethod
    def collate_fn(batch):
        """Prototype to handle reading exception.

        This will exclude any sample with `None` from the batch. As
        such, wrapping `__getitem__` with try-catch and return `None`
        upon exceptions will prevent crashing the entire program. But as
        a side effect, the batch may not have the size as defined.

        """
        batch = [v for v in batch if v is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, idx: int):
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
            self.ioconfig.input_resolutions, self.ioconfig.resolution_unit
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


class WSIPatchDataset(PatchDatasetABC):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`):
            A WSI Reader or Virtual Reader for reading pyramidal image
            or large tile in pyramidal way.
        inputs:
            List of coordinates to read from the `reader`, each
            coordinate is of the form `[start_x, start_y, end_x,
            end_y]`.
        patch_input_shape:
            A tuple (int, int) or ndarray of shape (2,). Expected size to
            read from `reader` at requested `resolution` and `units`.
            Expected to be `(height, width)`.
        resolution:
            See (:class:`.WSIReader`) for details.
        units:
            See (:class:`.WSIReader`) for details.
        preproc_func:
            Preprocessing function used to transform the input data. It will
            be called on each patch before returning it.

    """

    def __init__(
        self,
        img_path,
        mode="wsi",
        mask_path=None,
        patch_input_shape=None,
        stride_shape=None,
        resolution=None,
        units=None,
        auto_get_mask=True,
        min_mask_ratio=0,
        preproc_func=None,
    ):
        """Create a WSI-level patch dataset.

        Args:
            mode (str):
                Can be either `wsi` or `tile` to denote the image to
                read is either a whole-slide image or a large image
                tile.
            img_path (:obj:`str` or :obj:`pathlib.Path`):
                Valid to pyramidal whole-slide image or large tile to
                read.
            mask_path (:obj:`str` or :obj:`pathlib.Path`):
                Valid mask image.
            patch_input_shape:
                A tuple (int, int) or ndarray of shape (2,). Expected
                shape to read from `reader` at requested `resolution`
                and `units`. Expected to be positive and of (height,
                width). Note, this is not at `resolution` coordinate
                space.
            stride_shape:
                A tuple (int, int) or ndarray of shape (2,). Expected
                stride shape to read at requested `resolution` and
                `units`. Expected to be positive and of (height, width).
                Note, this is not at level 0.
            resolution:
                Check (:class:`.WSIReader`) for details. When
                `mode='tile'`, value is fixed to be `resolution=1.0` and
                `units='baseline'` units: check (:class:`.WSIReader`) for
                details.
            units:
                Units in which `resolution` is defined.
            auto_get_mask:
                If `True`, then automatically get simple threshold mask using
                WSIReader.tissue_mask() function.
            min_mask_ratio:
                Only patches with positive area percentage above this value are
                included. Defaults to 0.
            preproc_func:
                Preprocessing function used to transform the input data. If
                supplied, the function will be called on each patch before
                returning it.

        Examples:
            >>> # A user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # Create a dataset to get patches from WSI with above
            >>> # preprocessing function
            >>> ds = WSIPatchDataset(
            ...     img_path='/A/B/C/wsi.svs',
            ...     mode="wsi",
            ...     patch_input_shape=[512, 512],
            ...     stride_shape=[256, 256],
            ...     auto_get_mask=False,
            ...     preproc_func=preproc_func
            ... )

        """
        super().__init__()

        # Is there a generic func for path test in toolbox?
        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")
        if mode not in ["wsi", "tile"]:
            raise ValueError(f"`{mode}` is not supported.")
        patch_input_shape = np.array(patch_input_shape)
        stride_shape = np.array(stride_shape)

        if (
            not np.issubdtype(patch_input_shape.dtype, np.integer)
            or np.size(patch_input_shape) > 2
            or np.any(patch_input_shape < 0)
        ):
            raise ValueError(f"Invalid `patch_input_shape` value {patch_input_shape}.")
        if (
            not np.issubdtype(stride_shape.dtype, np.integer)
            or np.size(stride_shape) > 2
            or np.any(stride_shape < 0)
        ):
            raise ValueError(f"Invalid `stride_shape` value {stride_shape}.")

        self.preproc_func = preproc_func
        img_path = pathlib.Path(img_path)
        if mode == "wsi":
            self.reader = WSIReader.open(img_path)
        else:
            logger.warning(
                "WSIPatchDataset only reads image tile at "
                '`units="baseline"` and `resolution=1.0`.',
                stacklevel=2,
            )
            units = "baseline"
            resolution = 1.0
            img = imread(img_path)
            axes = "YXS"[: len(img.shape)]
            # initialise metadata for VirtualWSIReader.
            # here, we simulate a whole-slide image, but with a single level.
            # ! should we expose this so that use can provide their metadata ?
            metadata = WSIMeta(
                mpp=np.array([1.0, 1.0]),
                axes=axes,
                objective_power=10,
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
            # hack value such that read if mask is provided is through
            # 'mpp' or 'power' as varying 'baseline' is locked atm
            units = "mpp"
            resolution = 1.0
            self.reader = VirtualWSIReader(
                img,
                info=metadata,
            )

        # may decouple into misc ?
        # the scaling factor will scale base level to requested read resolution/units
        wsi_shape = self.reader.slide_dimensions(resolution=resolution, units=units)

        # use all patches, as long as it overlaps source image
        self.inputs = PatchExtractor.get_coordinates(
            image_shape=wsi_shape,
            patch_input_shape=patch_input_shape[::-1],
            stride_shape=stride_shape[::-1],
            input_within_bound=False,
        )

        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = self.reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = self.reader.tissue_mask(resolution=1.25, units="power")
            # ? will this mess up  ?
            mask_reader.info = self.reader.info

        if mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                self.inputs,  # must already be at requested resolution
                wsi_shape=wsi_shape,
                min_mask_ratio=min_mask_ratio,
            )
            self.inputs = self.inputs[selected]

        if len(self.inputs) == 0:
            raise ValueError("No patch coordinates remain after filtering.")

        self.patch_input_shape = patch_input_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def __getitem__(self, idx):
        coords = self.inputs[idx]
        # Read image patch from the whole-slide image
        patch = self.reader.read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            coord_space="resolution",
        )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        return {"image": patch, "coords": np.array(coords)}


class PatchDataset(PatchDatasetABC):
    """Defines a simple patch dataset, which inherits from the
      `torch.utils.data.Dataset` class.

    Attributes:
        inputs:
            Either a list of patches, where each patch is a ndarray or a
            list of valid path with its extension be (".jpg", ".jpeg",
            ".tif", ".tiff", ".png") pointing to an image.
        labels:
            List of labels for sample at the same index in `inputs`.
            Default is `None`.
        preproc_func:
            Preprocessing function used to transform the input data.

    Examples:
        >>> # A user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     preproc_func=preproc_func
        ... )

    """

    def __init__(self, inputs, labels=None):
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self, idx):
        patch = self.inputs[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {
            "image": patch,
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
            return data

        return data
