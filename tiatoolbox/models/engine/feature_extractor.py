"""Define Deep Feature Extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from tiatoolbox.models.dataset.dataset_abc import WSIStreamDataset

from .semantic_segmentor import SemanticSegmentor

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    import torch

    from tiatoolbox.models.engine.io_config import IOSegmentorConfig
    from tiatoolbox.type_hints import IntPair, Resolution, Units
    from tiatoolbox.wsicore.wsireader import WSIReader


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
