import os
import pathlib
from typing import Literal, Tuple, Union

import cv2
import numpy as np
import PIL
import torchvision.transforms as transforms

from tiatoolbox.models.dataset import dataset_abc
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import Units, VirtualWSIReader, WSIReader


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list):
            List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. For more information, visit the following
            link: https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preprocs):
        self.func = transforms.Compose(preprocs)

    def __call__(self, img):
        img = PIL.Image.fromarray(img)
        img = self.func(img)
        return img.permute(1, 2, 0)


def predefined_preproc_func(dataset_name):
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
        raise ValueError(
            f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        )

    preprocs = preproc_dict[dataset_name]
    return _TorchPreprocCaller(preprocs)


class PatchDataset(dataset_abc.PatchDatasetABC):
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


class WSIPatchDataset(dataset_abc.PatchDatasetABC):
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
            Preprocessing function used to transform the input data. If
            supplied, then torch.Compose will be used on the input
            preprocs. preprocs is a list of torchvision transforms for
            preprocessing the image. The transforms will be applied in
            the order that they are given in the list. For more
            information, visit the following link:
            https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray, WSIReader],
        mode: Literal["wsi", "tile"] = "wsi",
        mask: Union[str, pathlib.Path, np.ndarray, VirtualWSIReader] = None,
        patch_input_shape: Union[Tuple[int, int], np.ndarray] = None,
        stride_shape: Union[Tuple[int, int], np.ndarray] = None,
        resolution: float = 1,
        units: Units = "baseline",
        auto_get_mask=True,
    ):
        """Create a WSI-level patch dataset.

        Args:
            mode (str):
                Can be either `wsi` or `tile` to denote the image to
                read is either a whole-slide image or a large image
                tile.
            input_img: (:obj:`str` or
                        :obj:`pathlib.Path` or
                        :obj:`ndarray` or
                        :obj:`WSIReader`):
                Valid to pyramidal whole-slide image or large tile to
                read.
            mask (:obj:`str` or
                    :obj:`pathlib.Path` or
                    :obj:`ndarray` or
                    :obj:`VirtualWSIReader`):
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
              Check (:class:`.WSIReader`) for details.
              If reading from an image without specified metadata,
              use `resolution=1.0` and`units='baseline'` units:
              check (:class:`.WSIReader`) for details.
            units:
                Check (:class:`.WSIReader`) for details.
            preproc_func:
                Preprocessing function used to transform the input data.

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

        if mode not in ["wsi", "tile"]:
            raise ValueError(f"`{mode}` is not supported.")

        if units not in ["baseline", "power", "mpp"]:
            raise ValueError(f"`{units}` is not supported.")

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

        self.reader = WSIReader.open(input_img)

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

        self._apply_mask(
            mask=mask,
            auto_get_mask=auto_get_mask,
            resolution=resolution,
            units=units,
            mode=mode,
        )

        if len(self.inputs) == 0:
            raise ValueError("No patch coordinates remain after filtering.")

        self.patch_input_shape = patch_input_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def _apply_mask(
        self,
        mask: Union[str, pathlib.Path, np.ndarray, VirtualWSIReader],
        auto_get_mask: bool = True,
        resolution: float = 1,
        units: Units = "baseline",
        mode: Literal["wsi", "tile"] = "wsi",
    ):
        """Reads or generates a mask for the input image and
        applies it to the dataset."""
        mask_reader = None
        if mask is not None:
            if not isinstance(mask, (str, pathlib.Path, np.ndarray, VirtualWSIReader)):
                raise ValueError(
                    "`mask` must be file path, np.ndarray or VirtualWSIReader."
                )

            if isinstance(mask, VirtualWSIReader):
                if mask.mode != "bool":
                    raise ValueError(
                        "`mask` must be binary, "
                        "i.e. VirtualWSIReader's mode has to be 'bool'"
                    )

                mask_reader = mask

            elif isinstance(mask, np.ndarray):
                if mask.dtype != np.bool:
                    raise ValueError(
                        "`mask` must be binary, i.e. `ndarray.dtype` has to be bool"
                    )

                mask_reader = VirtualWSIReader(mask.astype(np.uint8))

            else:  # assume to be file path
                if not os.path.isfile(mask):
                    raise ValueError("`mask` must be a valid file path.")

                mask = imread(mask)  # assume to be gray
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = np.array(mask > 0, dtype=np.uint8)
                mask_reader = VirtualWSIReader(mask)

            mask_reader.info = self.reader.info
        elif auto_get_mask and mode == "wsi" and mask is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = self.reader.tissue_mask(resolution=1.25, units="power")
            # ? will this mess up  ?
            mask_reader.info = self.reader.info

        if mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                self.inputs,  # must already be at requested resolution
                resolution=resolution,
                units=units,
            )
            self.inputs = self.inputs[selected]

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
