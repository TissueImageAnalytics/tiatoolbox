"""Defines PatchPredictor Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .engine_abc import EngineABC, EngineABCRunParams

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from torch.utils.data import DataLoader

    from tiatoolbox.models.models_abc import ModelABC


class PatchPredictor(EngineABC):
    r"""Patch level predictor for digital histology images.

    The models provided by TIAToolbox should give the following results:

    .. list-table:: PatchPredictor performance on the Kather100K dataset [1]
       :widths: 15 15
       :header-rows: 1

       * - Model name
         - F\ :sub:`1`\ score
       * - alexnet-kather100k
         - 0.965
       * - resnet18-kather100k
         - 0.990
       * - resnet34-kather100k
         - 0.991
       * - resnet50-kather100k
         - 0.989
       * - resnet101-kather100k
         - 0.989
       * - resnext50_32x4d-kather100k
         - 0.992
       * - resnext101_32x8d-kather100k
         - 0.991
       * - wide_resnet50_2-kather100k
         - 0.989
       * - wide_resnet101_2-kather100k
         - 0.990
       * - densenet121-kather100k
         - 0.993
       * - densenet161-kather100k
         - 0.992
       * - densenet169-kather100k
         - 0.992
       * - densenet201-kather100k
         - 0.991
       * - mobilenet_v2-kather100k
         - 0.990
       * - mobilenet_v3_large-kather100k
         - 0.991
       * - mobilenet_v3_small-kather100k
         - 0.992
       * - googlenet-kather100k
         - 0.992

    .. list-table:: PatchPredictor performance on the PCam dataset [2]
       :widths: 15 15
       :header-rows: 1

       * - Model name
         - F\ :sub:`1`\ score
       * - alexnet-pcam
         - 0.840
       * - resnet18-pcam
         - 0.888
       * - resnet34-pcam
         - 0.889
       * - resnet50-pcam
         - 0.892
       * - resnet101-pcam
         - 0.888
       * - resnext50_32x4d-pcam
         - 0.900
       * - resnext101_32x8d-pcam
         - 0.892
       * - wide_resnet50_2-pcam
         - 0.901
       * - wide_resnet101_2-pcam
         - 0.898
       * - densenet121-pcam
         - 0.897
       * - densenet161-pcam
         - 0.893
       * - densenet169-pcam
         - 0.895
       * - densenet201-pcam
         - 0.891
       * - mobilenet_v2-pcam
         - 0.899
       * - mobilenet_v3_large-pcam
         - 0.895
       * - mobilenet_v3_small-pcam
         - 0.890
       * - googlenet-pcam
         - 0.867

    Args:
        model (str | ModelABC):
            A PyTorch model or name of pretrained model.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter. Default is `None`.
        batch_size (int):
            Number of image patches fed into the model each time in a
            forward/backward pass. Default value is 8.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. Default value is 0.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
            Default value is 0.
        weights (str or Path):
            Path to the weight of the corresponding `model`.

            >>> engine = EngineABC(
            ...    model="pretrained-model",
            ...    weights="/path/to/pretrained-local-weights.pth"
            ... )

        device (str):
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default is "cpu".
        verbose (bool):
            Whether to output logging information. Default value is False.

    Attributes:
        images (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A list of image patches in NHWC format as a numpy array
            or a list of str/paths to WSIs.
        masks (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A list of tissue masks or binary masks corresponding to processing area of
            input images. These can be a list of numpy arrays or paths to
            the saved image masks. These are only utilized when patch_mode is False.
            Patches are only generated within a masked area.
            If not provided, then a tissue mask will be automatically
            generated for whole slide images.
        patch_mode (str):
            Whether to treat input images as a set of image patches. TIAToolbox defines
            an image as a patch if HWC of the input image matches with the HWC expected
            by the model. If HWC of the input image does not match with the HWC expected
            by the model, then the patch_mode must be set to False which will allow the
            engine to extract patches from the input image.
            In this case, when the patch_mode is False the input images are treated
            as WSIs. Default value is True.
        model (str | ModelABC):
            A PyTorch model or a name of an existing model from the TIAToolbox model zoo
            for processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument
            is case-insensitive.
        ioconfig (ModelIOConfigABC):
            Input IO configuration of type :class:`ModelIOConfigABC` to run the Engine.
        _ioconfig (ModelIOConfigABC):
            Runtime ioconfig.
        return_labels (bool):
            Whether to return the labels with the predictions.
        resolution (Resolution):
            Resolution used for reading the image. Please see
            :obj:`WSIReader` for details.
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `level`, `power` or `mpp`. Please see
            :obj:`WSIReader` for details.
        patch_input_shape (tuple):
            Shape of patches input to the model as tupled of HW. Patches are at
            requested read resolution, not with respect to level 0,
            and must be positive.
        stride_shape (tuple):
            Stride used during WSI processing. Stride is
            at requested read resolution, not with respect to
            level 0, and must be positive. If not provided,
            `stride_shape=patch_input_shape`.
        batch_size (int):
            Number of images fed into the model each time.
        cache_mode (bool):
            Whether to run the Engine in cache_mode. For large datasets,
            we recommend to set this to True to avoid out of memory errors.
            For smaller datasets, the cache_mode is set to False as
            the results can be saved in memory. cache_mode is always True when
            processing WSIs i.e., when `patch_mode` is False. Default value is False.
        cache_size (int):
            Specifies how many image patches to process in a batch when
            cache_mode is set to True. If cache_size is less than the batch_size
            batch_size is set to cache_size. Default value is 10,000.
        labels (list | None):
                List of labels. Only a single label per image is supported.
        device (str):
            :class:`torch.device` to run the model.
            Select the device to run the model. Please see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            for more details on input parameters for device. Default value is "cpu".
        num_loader_workers (int):
            Number of workers used in :class:`torch.utils.data.DataLoader`.
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model.
        return_labels (bool):
            Whether to return the output labels. Default value is False.
        resolution (Resolution):
            Resolution used for reading the image. Please see
            :class:`WSIReader` for details.
            When `patch_mode` is True, the input image patches are expected to be at
            the correct resolution and units. When `patch_mode` is False, the patches
            are extracted at the requested resolution and units. Default value is 1.0.
        units (Units):
            Units of resolution used for reading the image. Choose
            from either `baseline`, `level`, `power` or `mpp`. Please see
            :class:`WSIReader` for details.
            When `patch_mode` is True, the input image patches are expected to be at
            the correct resolution and units. When `patch_mode` is False, the patches
            are extracted at the requested resolution and units.
            Default value is `baseline`.
        verbose (bool):
            Whether to output logging information. Default value is False.

    Examples:
        >>> # list of 2 image patches as input
        >>> data = ['path/img.svs', 'path/img.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> predictor = PatchPredictor(model="resnet18-kather100k")
        >>> output = predictor.run(wsi_file, mode='wsi')

    References:
        [1] Kather, Jakob Nikolas, et al. "Predicting survival from colorectal cancer
        histology slides using deep learning: A retrospective multicenter study."
        PLoS medicine 16.1 (2019): e1002730.

        [2] Veeling, Bastiaan S., et al. "Rotation equivariant CNNs for digital
        pathology." International Conference on Medical image computing and
        computer-assisted intervention. Springer, Cham, 2018.

    """

    def __init__(
        self: PatchPredictor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`PatchPredictor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def infer_wsi(
        self: EngineABC,
        dataloader: DataLoader,
        save_path: Path,
        **kwargs: EngineABCRunParams,
    ) -> Path:
        """Model inference on a WSI.

        Args:
            dataloader (DataLoader):
                A torch dataloader to process WSIs.

            save_path (Path):
                Path to save the intermediate output. The intermediate output is saved
                in a zarr file.
            **kwargs (EngineABCRunParams):
                Keyword Args to update setup_patch_dataset() method attributes. See
                :class:`EngineRunParams` for accepted keyword arguments.

        Returns:
            save_path (Path):
                Path to zarr file where intermediate output is saved.

        """
        _ = kwargs.get("patch_mode", False)
        return self.infer_patches(
            dataloader=dataloader,
            save_path=save_path,
            return_coordinates=True,
        )
