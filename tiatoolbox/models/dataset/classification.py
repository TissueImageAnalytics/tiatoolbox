"""Define classes and methods for classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from torchvision import transforms

from tiatoolbox import logger
from tiatoolbox.models.dataset import dataset_abc
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import imread
from tiatoolbox.wsicore.wsimeta import WSIMeta
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from PIL.Image import Image

    from tiatoolbox.typing import IntPair, Resolution, Units


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list):
            List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. For more information, visit the following
            link: https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self: _TorchPreprocCaller, preprocs: list) -> None:
        self.func = transforms.Compose(preprocs)

    def __call__(self: _TorchPreprocCaller, img: np.ndarray | Image) -> torch.Tensor:
        tensor: torch.Tensor = self.func(img)
        return tensor.permute((1, 2, 0))


def predefined_preproc_func(dataset_name: str) -> _TorchPreprocCaller:
    """Get the preprocessing information used for the pretrained model.

    Args:
        dataset_name (str):
            Dataset name used to determine what preprocessing was used.

    Returns:
        _TorchPreprocCaller:
            Preprocessing function for transforming the input data.

    """
    preproc_dict = {
        "kather100k": [
            transforms.ToTensor(),
        ],
        "pcam": [
            transforms.ToTensor(),
        ],
    }

    if dataset_name not in preproc_dict:
        msg = f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        raise ValueError(
            msg,
        )

    preprocs = preproc_dict[dataset_name]
    return _TorchPreprocCaller(preprocs)


class PatchDataset(dataset_abc.PatchDatasetABC):
    """Define PatchDataset for torch inference.

    Define a simple patch dataset, which inherits from the
      `torch.utils.data.Dataset` class.

    Attributes:
        inputs (list or np.ndarray):
            Either a list of patches, where each patch is a ndarray or a
            list of valid path with its extension be (".jpg", ".jpeg",
            ".tif", ".tiff", ".png") pointing to an image.
        labels (list):
            List of labels for sample at the same index in `inputs`.
            Default is `None`.

    Examples:
        >>> # A user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     labels=["labels1", "labels2"],
        ... )

    """

    def __init__(
        self: PatchDataset,
        inputs: np.ndarray | list,
        labels: list | None = None,
    ) -> None:
        """Initialize :class:`PatchDataset`."""
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self: PatchDataset, idx: int) -> dict:
        """Get an item from the dataset."""
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


class WSIPatchDataset(dataset_abc.PatchDatasetABC):
    """Define a WSI-level patch dataset.

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

    def __init__(  # skipcq: PY-R1000  # noqa: PLR0915
        self: WSIPatchDataset,
        img_path: str | Path,
        mode: str = "wsi",
        mask_path: str | Path | None = None,
        patch_input_shape: IntPair = None,
        stride_shape: IntPair = None,
        resolution: Resolution = None,
        units: Units = None,
        min_mask_ratio: float = 0,
        preproc_func: Callable | None = None,
        *,
        auto_get_mask: bool = True,
    ) -> None:
        """Create a WSI-level patch dataset.

        Args:
            mode (str):
                Can be either `wsi` or `tile` to denote the image to
                read is either a whole-slide image or a large image
                tile.
            img_path (str or Path):
                Valid to pyramidal whole-slide image or large tile to
                read.
            mask_path (str or Path):
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
            resolution (Resolution):
                Check (:class:`.WSIReader`) for details. When
                `mode='tile'`, value is fixed to be `resolution=1.0` and
                `units='baseline'` units: check (:class:`.WSIReader`) for
                details.
            units (Units):
                Units in which `resolution` is defined.
            auto_get_mask (bool):
                If `True`, then automatically get simple threshold mask using
                WSIReader.tissue_mask() function.
            min_mask_ratio (float):
                Only patches with positive area percentage above this value are
                included. Defaults to 0.
            preproc_func (Callable):
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
        if not Path.is_file(Path(img_path)):
            msg = "`img_path` must be a valid file path."
            raise ValueError(msg)
        if mode not in ["wsi", "tile"]:
            msg = f"`{mode}` is not supported."
            raise ValueError(msg)
        patch_input_shape = np.array(patch_input_shape)
        stride_shape = np.array(stride_shape)

        if (
            not np.issubdtype(patch_input_shape.dtype, np.integer)
            or np.size(patch_input_shape) > 2  # noqa: PLR2004
            or np.any(patch_input_shape < 0)
        ):
            msg = f"Invalid `patch_input_shape` value {patch_input_shape}."
            raise ValueError(msg)
        if (
            not np.issubdtype(stride_shape.dtype, np.integer)
            or np.size(stride_shape) > 2  # noqa: PLR2004
            or np.any(stride_shape < 0)
        ):
            msg = f"Invalid `stride_shape` value {stride_shape}."
            raise ValueError(msg)

        self.preproc_func = preproc_func
        img_path = Path(img_path)
        if mode == "wsi":
            self.reader = WSIReader.open(img_path)
        else:
            logger.warning(
                "WSIPatchDataset only reads image tile at "
                '`units="baseline"` and `resolution=1.0`.',
                stacklevel=2,
            )
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
            # infer value such that read if mask provided is through
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
            mask_path = Path(mask_path)
            if not Path.is_file(mask_path):
                msg = "`mask_path` must be a valid file path."
                raise ValueError(msg)
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
            msg = "No patch coordinates remain after filtering."
            raise ValueError(msg)

        self.patch_input_shape = patch_input_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def __getitem__(self: WSIPatchDataset, idx: int) -> dict:
        """Get an item from the dataset."""
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
