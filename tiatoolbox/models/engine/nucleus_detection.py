"""This module implements nucleus detection engine."""


from tiatoolbox.models.engine.patch_predictor import PatchPredictor


class NucleusDetector(PatchPredictor):
    r"""Nucleus detection engine.

    The models provided by tiatoolbox should give the following results:

    .. list-table:: Nucleus detection performance on the (add models list here)
       :widths: 15 15
       :header-rows: 1

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

          >>> predictor = NucleusDetector(
          ...    pretrained_model="resnet18-kather100k",
          ...    pretrained_weights="resnet18_local_weight")

        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data. Take note that they will
            also perform preprocessing.
        verbose (bool):
            Whether to output logging information.

    Attributes:
        imgs (:obj:`str` or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
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
        >>> data = [img1, img2]
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> predictor = PatchPredictor(pretraind_model="resnet18-kather100k")
        >>> output = predictor.predict(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> predictor = PatchPredictor(pretraind_model="resnet18-kather100k")
        >>> output = predictor.predict(wsi_file, mode='wsi')

    References:
        [1] Kather, Jakob Nikolas, et al. "Predicting survival from colorectal cancer
        histology slides using deep learning: A retrospective multicenter study."
        PLoS medicine 16.1 (2019): e1002730.

        [2] Veeling, Bastiaan S., et al. "Rotation equivariant CNNs for digital
        pathology." International Conference on Medical image computing and
        computer-assisted intervention. Springer, Cham, 2018.

    """  # noqa: W605

    def __init__(
        self,
        batch_size=8,
        num_loader_workers=0,
        model=None,
        pretrained_model=None,
        pretrained_weights=None,
        verbose=True,
    ):
        super().__init__(
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            model=model,
            pretrained_model=pretrained_model,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
        )
