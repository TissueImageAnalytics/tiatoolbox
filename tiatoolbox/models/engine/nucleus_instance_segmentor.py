"""This module enables nucleus instance segmentation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tiatoolbox import logger

from .multi_task_segmentor import MultiTaskSegmentor

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tiatoolbox.models.models_abc import ModelABC


class NucleusInstanceSegmentor(MultiTaskSegmentor):
    """NucleusInstanceSegmentor is segmentation engine to run models like hovernet.

    .. deprecated:: 2.1.0
       `NucleusInstanceSegmentor` will be removed in a future release.
       Use :class:`MultiTaskSegmentor` instead.

    NucleusInstanceSegmentor inherits MultiTaskSegmentor as it is a special case of
    MultiTaskSegmentor with a single task.

    Args:
        model (str | ModelABC):
            A PyTorch model instance or name of a pretrained model from TIAToolbox.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter. Default is `None`.
        batch_size (int):
            Number of image patches processed per forward pass. Default is 8.
        num_workers (int):
            Number of workers for data loading. Default is 0.
        weights (str | Path | None):
            Path to model weights. If None, default weights are used.

            >>> engine = NucleusInstanceSegmentor(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
        verbose (bool):
            Whether to enable verbose logging. Default is True.

    Attributes:
        images (list[str | Path] | np.ndarray):
            Input image patches or WSI paths.
        masks (list[str | Path] | np.ndarray):
            Optional tissue masks for WSI processing.
            These are only utilized when patch_mode is False.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (bool):
            Whether input is treated as patches (`True`) or WSIs (`False`).
        model (ModelABC):
            Loaded PyTorch model.
        ioconfig (ModelIOConfigABC):
            IO configuration for patch extraction and resolution.
        return_labels (bool):
            Whether to include labels in the output.
        return_predictions_dict (dict):
            This dictionary helps keep track of which tasks require predictions in
            the output.
        input_resolutions (list[dict]):
            Resolution settings for model input. Supported
            units are `level`, `power` and `mpp`. Keys should be "units" and
            "resolution" e.g., [{"units": "mpp", "resolution": 0.25}]. Please see
            :class:`WSIReader` for details.
        patch_input_shape (tuple[int, int]):
            Shape of input patches (height, width). Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (tuple[int, int]):
            Stride used during patch extraction. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        labels (list | None):
            Optional labels for input images.
            Only a single label per image is supported.
        drop_keys (list):
            Keys to exclude from model output.
        output_type (str):
            Format of output ("dict", "zarr", "annotationstore").
        output_locations (list | None):
            Coordinates of output patches used during WSI processing.

    Examples:
    >>> # list of 2 image patches as input
    >>> wsis = ['path/img.svs', 'path/img.svs']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed")
    >>> output = mtsegmentor.run(wsis, patch_mode=False)

    >>> # array of list of 2 image patches as input
    >>> image_patches = [np.ndarray, np.ndarray]
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed")
    >>> output = mtsegmentor.run(image_patches, patch_mode=True)

    >>> # list of 2 image patch files as input
    >>> data = ['path/img.png', 'path/img.png']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(data, patch_mode=False)

    >>> # list of 2 image tile files as input
    >>> tile_file = ['path/tile1.png', 'path/tile2.png']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(tile_file, patch_mode=False)

    >>> # list of 2 wsi files as input
    >>> wsis = ['path/wsi1.svs', 'path/wsi2.svs']
    >>> mtsegmentor = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    >>> output = mtsegmentor.run(wsis, patch_mode=False)


    """

    def __init__(
        self: MultiTaskSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusInstanceSegmentor`.

        Args:
            model (str | ModelABC):
                A PyTorch model instance or name of a pretrained model from TIAToolbox.
                If a string is provided, the corresponding pretrained weights will be
                downloaded unless overridden via `weights`.
            batch_size (int):
                Number of image patches processed per forward pass. Default is 8.
            num_workers (int):
                Number of workers for data loading. Default is 0.
            weights (str | Path | None):
                Path to model weights. If None, default weights are used.
            device (str):
                Device to run the model on (e.g., "cpu", "cuda"). Default is "cpu".
            verbose (bool):
                Whether to enable verbose logging. Default is True.

        """
        warnings.warn(
            "NucleusInstanceSegmentor is deprecated and will be "
            "removed in a future release. "
            "Use MultiTaskSegmentor instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        logger.warning(
            "NucleusInstanceSegmentor is deprecated and will be "
            "removed in a future release."
        )
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )
