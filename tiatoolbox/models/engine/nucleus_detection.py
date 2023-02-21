"""This module implements nucleus detection engine."""


from typing import List, Union

import numpy as np

from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)


class IONucleusDetectorConfig(IOSegmentorConfig):
    """Contains NucleusDetector input and output information.

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
        >>> ioconfig = IONucleusDetectorConfig(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=[2048, 2048],
        ...     patch_output_shape=[1024, 1024],
        ...     stride_shape=[512, 512],
        ... )

    """

    def __init__(
        self,
        input_resolutions: List[dict],
        output_resolutions: List[dict],
        patch_input_shape: Union[List[int], np.ndarray],
        patch_output_shape: Union[List[int], np.ndarray],
        save_resolution: dict = None,
        **kwargs,
    ):
        super().__init__(
            input_resolutions=input_resolutions,
            output_resolutions=output_resolutions,
            patch_input_shape=patch_input_shape,
            patch_output_shape=patch_output_shape,
            save_resolution=save_resolution,
            **kwargs,
        )


class NucleusDetector(SemanticSegmentor):
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
          ...    pretrained_model="mapde-conic",
          ...    pretrained_weights="mapde_local_weight")

        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data. Take note that they will
            also perform preprocessing.
        verbose (bool):
            Whether to output logging information. default=False.
        auto_generate_mask (bool):
            To automatically generate tile/WSI tissue mask if is not
            provided. default=False.

    Attributes:
        imgs (:obj:`str` or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
            A HWC image or a path to WSI.
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
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.predict(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.predict(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
        >>> output = nucleus_detector.predict(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> nucleus_detector = NucleusDetector(pretraind_model="mapde-conic")
        >>> output = nucleus_detector.predict(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> nucleus_detector = NucleusDetector(pretraind_model="mapde-conic")
        >>> output = nucleus_detector.predict(wsi_file, mode='wsi')

    References:
        [1] Raza, Shan E. Ahmed, et al. "Deconvolving convolutional neural network
        for cell detection." 2019 IEEE 16th International Symposium on Biomedical
        Imaging (ISBI 2019). IEEE, 2019.

        [2] Sirinukunwattana, Korsuk, et al.
        "Locality sensitive deep learning for detection and classification
        of nuclei in routine colon cancer histology images."
        IEEE transactions on medical imaging 35.5 (2016): 1196-1206.

    """  # noqa: W605

    def __init__(
        self,
        batch_size=8,
        num_loader_workers=0,
        model=None,
        pretrained_model=None,
        pretrained_weights=None,
        verbose: bool = False,
        auto_generate_mask: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            model=model,
            pretrained_model=pretrained_model,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
            auto_generate_mask=auto_generate_mask,
        )
