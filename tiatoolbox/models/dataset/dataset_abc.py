"""Define dataset abstract classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from tiatoolbox import logger
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import imread
from tiatoolbox.utils.exceptions import DimensionMismatchError
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable
    from typing import TypeGuard

    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReaderParams

input_type = list[str | Path | np.ndarray] | np.ndarray


class PatchDatasetABC(ABC, torch.utils.data.Dataset):
    """Define abstract base class for patch dataset."""

    inputs: input_type
    labels: list[int] | np.ndarray

    def __init__(
        self: PatchDatasetABC,
    ) -> None:
        """Initialize :class:`PatchDatasetABC`."""
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []

    @staticmethod
    def _check_shape_integrity(shapes: list | np.ndarray) -> None:
        """Checks the integrity of input shapes.

        Args:
            shapes (list or np.ndarray):
                input shape to check.

        Raises:
            ValueError: If the shape is not valid.

        """
        if any(len(v) != 3 for v in shapes):  # noqa: PLR2004
            msg = "Each sample must be an array of the form HWC."
            raise ValueError(msg)

        max_shape = np.max(shapes, axis=0)
        if (shapes - max_shape[None]).sum() != 0:
            msg = "Images must have the same dimensions."
            raise ValueError(msg)

    @staticmethod
    def _are_paths(inputs: input_type) -> TypeGuard[Iterable[Path]]:
        """TypeGuard to check that input array contains only paths."""
        return all(isinstance(v, (Path, str)) for v in inputs)

    @staticmethod
    def _are_npy_like(inputs: input_type) -> TypeGuard[Iterable[np.ndarray]]:
        """TypeGuard to check that input array contains only np.ndarray."""
        return all(isinstance(v, np.ndarray) for v in inputs)

    def _check_input_integrity(self: PatchDatasetABC, mode: str) -> None:
        """Check that variables received during init are valid.

        These checks include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height
              and width.

        """
        if mode == "patch":
            self.data_is_npy_alike = False

            msg = (
                "Input must be either a list/array of images "
                "or a list of valid image paths."
            )

            # When a list of paths is provided
            if self._are_paths(self.inputs):
                if any(not Path(v).exists() for v in self.inputs):
                    # at least one of the paths are invalid
                    raise ValueError(
                        msg,
                    )
                # Preload test for sanity check
                shapes = [self.load_img(v).shape for v in self.inputs]
                self.data_is_npy_alike = False

            elif self._are_npy_like(self.inputs):
                shapes = [v.shape for v in self.inputs]
                self.data_is_npy_alike = True

            else:
                raise ValueError(msg)

            self._check_shape_integrity(shapes)

            # If input is a numpy array
            if isinstance(self.inputs, np.ndarray):
                # Check that input array is numerical
                if not np.issubdtype(self.inputs.dtype, np.number):
                    # ndarray of mixed data types
                    msg = "Provided input array is non-numerical."
                    raise ValueError(msg)
                self.data_is_npy_alike = True

        elif not isinstance(self.inputs, (list, np.ndarray)):
            msg = "`inputs` should be a list of patch coordinates."
            raise ValueError(msg)

    @staticmethod
    def load_img(path: str | Path) -> np.ndarray:
        """Load an image from a provided path.

        Args:
            path (str or Path): Path to an image file.

        Returns:
            :class:`numpy.ndarray`:
                Image as a numpy array.

        """
        path = Path(path)

        if path.suffix not in (".npy", ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            msg = f"Cannot load image data from `{path.suffix}` files."
            raise TypeError(msg)

        return imread(path, as_uint8=False)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Define the pre-processing of this class of loader."""
        return image

    @property
    def preproc_func(self: PatchDatasetABC) -> Callable:
        """Return the current pre-processing function of this instance.

        The returned function is expected to behave as follows:
        >>> transformed_img = func(img)

        """
        return self._preproc

    @preproc_func.setter
    def preproc_func(self: PatchDatasetABC, func: Callable) -> None:
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
            msg = f"{func} is not callable!"
            raise ValueError(msg)

    def __len__(self: PatchDatasetABC) -> int:
        """Return the length of the instance attributes."""
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self: PatchDatasetABC, idx: int) -> None:
        """Get an item from the dataset."""
        ...  # pragma: no cover


class WSIPatchDataset(PatchDatasetABC):
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

    def __init__(  # skipcq: PY-R1000  # noqa: PLR0913
        self: WSIPatchDataset,
        input_img: str | Path | WSIReader,
        input_mask: str | Path | None = None,
        patch_input_shape: IntPair = None,
        patch_output_shape: IntPair = None,
        stride_shape: IntPair = None,
        resolution: Resolution = None,
        units: Units = None,
        min_mask_ratio: float = 0,
        preproc_func: Callable | None = None,
        wsireader_kwargs: WSIReaderParams | None = None,
        *,
        auto_get_mask: bool = True,
    ) -> None:
        """Create a WSI-level patch dataset.

        Args:
            input_img (str or Path or WSIReader):
                Valid path to a whole-slide image class:`WSIReader`.
            input_mask (str or Path):
                Valid mask image.
            patch_input_shape:
                A tuple (int, int) or ndarray of shape (2,). Expected
                shape to read from `reader` at requested `resolution`
                and `units`. Expected to be positive and of (height,
                width). Note, this is not at `resolution` coordinate
                space.
            patch_output_shape:
                A tuple (int, int) or ndarray of shape (2,). Expected
                output shape from the model at requested `resolution`
                and `units`. Expected to be positive and of (height,
                width). Note, this is not at `resolution` coordinate
                space.
            stride_shape:
                A tuple (int, int) or ndarray of shape (2,). Expected
                stride shape to read at requested `resolution` and
                `units`. Expected to be positive and of (height, width).
                Note, this is not at level 0.
            resolution (Resolution):
                Requested resolution corresponding to units. Check
                (:class:`WSIReader`) for details.
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
            wsireader_kwargs (WSIReaderParams):
                Specify processing images with no mpp or power in the metadata.

        Examples:
            >>> # A user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # Create a dataset to get patches from WSI with above
            >>> # preprocessing function
            >>> ds = WSIPatchDataset(
            ...     input_img='/A/B/C/wsi.svs',
            ...     patch_input_shape=[512, 512],
            ...     stride_shape=[256, 256],
            ...     auto_get_mask=False,
            ...     preproc_func=preproc_func
            ... )

        """
        super().__init__()

        valid_path = bool(
            isinstance(input_img, (str, Path)) and Path(input_img).is_file()
        )
        # Is there a generic func for path test in toolbox?
        if not valid_path and not isinstance(input_img, WSIReader):
            msg = "`input_img` must be a valid file path or a `WSIReader` instance."
            raise ValueError(msg)
        patch_input_shape = np.array(patch_input_shape)
        stride_shape = np.array(stride_shape)

        _validate_patch_stride_shape(patch_input_shape, stride_shape)

        self.preproc_func = preproc_func
        img_path = (
            input_img if not isinstance(input_img, WSIReader) else input_img.input_path
        )
        self.img_path = Path(img_path)
        wsireader_kwargs = {} if wsireader_kwargs is None else wsireader_kwargs
        reader = (
            input_img
            if isinstance(input_img, WSIReader)
            else WSIReader.open(self.img_path, **wsireader_kwargs)
        )
        # To support multi-threading
        # Helps pickle using Path
        self.reader = None
        # may decouple into misc ?
        # the scaling factor will scale base level to requested read resolution/units
        wsi_shape = reader.slide_dimensions(resolution=resolution, units=units)
        self.reader_info = reader.info

        # use all patches, as long as it overlaps source image
        self.outputs = []
        if patch_output_shape is not None:
            self.inputs, self.outputs = PatchExtractor.get_coordinates(
                image_shape=wsi_shape,
                patch_input_shape=patch_input_shape[::-1],
                stride_shape=stride_shape[::-1],
                patch_output_shape=patch_output_shape,
            )
            self.full_outputs = self.outputs
        else:
            self.inputs = PatchExtractor.get_coordinates(
                image_shape=wsi_shape,
                patch_input_shape=patch_input_shape[::-1],
                stride_shape=stride_shape[::-1],
            )

        self.mask_reader = self._setup_mask_reader(
            input_mask=input_mask,
            reader=reader,
            auto_get_mask=auto_get_mask,
        )

        if self.mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                self.mask_reader,  # must be at the same resolution
                self.inputs,  # must already be at requested resolution
                wsi_shape=wsi_shape,
                min_mask_ratio=min_mask_ratio,
            )
            self.inputs = self.inputs[selected]
            if len(self.outputs) > 0:
                self.full_outputs = self.outputs  # Full list of outputs
                self.outputs = self.outputs[selected]

        self._check_inputs()

        self.patch_input_shape = patch_input_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def _setup_mask_reader(
        self: WSIPatchDataset,
        input_mask: str | Path | None,
        reader: WSIReader,
        *,
        auto_get_mask: bool,
    ) -> VirtualWSIReader | None:
        """Create a mask reader for WSIPatchDataset if requested."""
        if isinstance(input_mask, np.ndarray):
            mask_reader = VirtualWSIReader(input_mask)
            mask_reader.info = self.reader_info
            return mask_reader

        if isinstance(input_mask, (str, Path)):
            mask = Path(input_mask)
            if not Path.is_file(mask):
                msg = "`mask_path` must be a valid file path."
                raise ValueError(msg)
            mask = imread(mask)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = self.reader_info
            return mask_reader

        if auto_get_mask and input_mask is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            try:
                mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            except ValueError:
                # if power is None, try with mpp
                mask_reader = reader.tissue_mask(resolution=6.0, units="mpp")
            # ? will this mess up  ?
            mask_reader.info = self.reader_info

            return mask_reader
        return None

    def _check_inputs(self: WSIPatchDataset) -> None:
        """Check if input length is valid after filtering."""
        if len(self.inputs) == 0:
            msg = "No patch coordinates remain after filtering."
            raise ValueError(msg)

    def _get_reader(
        self: WSIPatchDataset, img_path: str | Path, wsireader_kwargs: WSIReaderParams
    ) -> WSIReader:
        """Get a reader for the image."""
        return (
            self.reader if self.reader else WSIReader.open(img_path, **wsireader_kwargs)
        )

    def __getitem__(self: WSIPatchDataset, idx: int) -> dict:
        """Get an item from the dataset."""
        coords = self.inputs[idx]
        output_locs = None
        if len(self.outputs) > 0:
            output_locs = self.outputs[idx]
        wsireader_kwargs: WSIReaderParams = {
            "mpp": self.reader_info.mpp,
            "power": self.reader_info.objective_power,
        }
        # Read image patch from the whole-slide image
        self.reader = self._get_reader(self.img_path, wsireader_kwargs)
        patch = self.reader.read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            coord_space="resolution",
        )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        if output_locs is not None:
            return {
                "image": patch,
                "coords": np.array(coords),
                "output_locs": output_locs,
            }

        return {"image": patch, "coords": np.array(coords)}


class PatchDataset(PatchDatasetABC):
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
        patch_input_shape (tuple):
            Size of patches input to the model. Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.

    Examples:
        >>> # A user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     labels=["labels1", "labels2"],
        ...     patch_input_shape=(224, 224),
        ... )

    """

    def __init__(
        self: PatchDataset,
        inputs: np.ndarray | list,
        labels: list | None = None,
        patch_input_shape: IntPair | None = None,
    ) -> None:
        """Initialize :class:`PatchDataset`."""
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels
        self.patch_input_shape = patch_input_shape

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self: PatchDataset, idx: int) -> dict:
        """Get an item from the dataset."""
        patch = self.inputs[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        if patch.shape[:-1] != tuple(self.patch_input_shape):
            msg = (
                f"Patch size is not compatible with the model. "
                f"Expected dimensions {tuple(self.patch_input_shape)}, but got "
                f"{patch.shape[:-1]}."
            )
            logger.error(msg=msg)
            raise DimensionMismatchError(
                expected_dims=tuple(self.patch_input_shape),
                actual_dims=patch.shape[:-1],
            )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {
            "image": patch,
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
            return data

        return data


def _validate_patch_stride_shape(
    patch_input_shape: np.ndarray, stride_shape: np.ndarray
) -> None:
    """Validate patch and stride shape inputs for semantic segmentation.

    Checks that both `patch_input_shape` and `stride_shape` are integer arrays of
    length ≤ 2 and contain non-negative values. Raises a ValueError if any
    condition fails.

    Parameters:
        patch_input_shape (np.ndarray):
            Shape of the input patch (e.g., height, width).
        stride_shape (np.ndarray):
            Stride dimensions used for patch extraction.

    Raises:
        ValueError:
            If either input is not a valid integer array of appropriate
            shape and values.

    """
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
