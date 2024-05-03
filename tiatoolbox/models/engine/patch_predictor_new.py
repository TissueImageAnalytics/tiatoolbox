"""Defines Abstract Base Class for TIAToolbox Model Engines."""

from __future__ import annotations

import copy
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tiatoolbox.models.models_abc import model_to
from tiatoolbox.utils.misc import (
    dict_to_store,
    wsi_batch_output_to_zarr_group,
)

from .engine_abc import EngineABC, prepare_engines_save_dir
from .io_config import ModelIOConfigABC

if TYPE_CHECKING:  # pragma: no cover
    import os

    import torch

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.typing import IntPair, Resolution, Units
    from tiatoolbox.wsicore.wsireader import WSIReader


class PatchPredictor(EngineABC):
    r"""Patch level predictor for digital histology images.

    The models provided by tiatoolbox should give the following results:

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
            A PyTorch model. Default is `None`.
            The user can request pretrained models from the toolbox model zoo using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights using the `weights` parameter.
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
        merge_predictions (bool):
            Whether to merge the predictions to form a 2-dimensional
            map. This is only applicable if `patch_mode` is False in inference.
            Default is False.
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
        merge_predictions (bool):
            Whether to merge WSI predictions into a single file. Default value is False.
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

    """

    def __init__(
        self: PatchPredictor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`PatchPredictor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            num_post_proc_workers=num_post_proc_workers,
            weights=weights,
            verbose=verbose,
        )

    def get_dataloader(
        self: PatchPredictor,
        images: Path,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Pre-process images and masks and return dataloader for inference.

        Args:
            images (list of str or :class:`Path` or :class:`numpy.ndarray`):
                A list of image patches in NHWC format as a numpy array
                or a list of str/paths to WSIs. When `patch_mode` is False
                the function expects list of str/paths to WSIs.
            masks (list | None):
                List of masks. Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            labels (list | None):
                List of labels. Only a single label per image is supported.
            ioconfig (ModelIOConfigABC):
                A :class:`ModelIOConfigABC` object.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.

        Returns:
            torch.utils.data.DataLoader:
                :class:`torch.utils.data.DataLoader` for inference.


        """
        return super().get_dataloader(
            images,
            masks,
            labels,
            ioconfig,
            patch_mode=patch_mode,
        )

    @abstractmethod
    def infer_wsi(
        self: EngineABC,
        dataloader: torch.utils.data.DataLoader,
        img_label: str,
        highest_input_resolution: list[dict],
        save_dir: Path,
        **kwargs: dict,
    ) -> list:
        """Model inference on a WSI."""
        # return coordinates of patches processed within a tile / whole-slide image
        return_coordinates = True

        # prepare a persistant zarr group file
        output_file = (
            kwargs["output_file"] and kwargs.pop("output_file")
            if "output_file" in kwargs
            else "output"
        )
        save_path = save_dir / output_file

        # ensure proper zarr extension
        save_path = save_path.parent.absolute() / (save_path.stem + ".zarr")

        cum_output = {}
        wsi_batch_zarr_group = None

        # get return flags from kwargs or set to False, useful in Annotation Store
        return_labels = kwargs.get("return_labels", False)

        for _, batch_data in enumerate(dataloader):
            batch_output_probabilities = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )
            # We get the index of the class with the maximum probability
            batch_output_predictions = self.model.postproc_func(
                batch_output_probabilities,
            )
            batch_output_coordinates, batch_output_label = None, None
            if return_coordinates:
                batch_output_coordinates = batch_data["coords"]
            if return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                batch_output_label = batch_data["label"]

            wsi_batch_zarr_group = wsi_batch_output_to_zarr_group(
                wsi_batch_zarr_group,
                batch_output_probabilities,
                batch_output_predictions,
                batch_output_coordinates,
                batch_output_label,
                save_path=save_path,
                **kwargs,
            )

        cum_output["probabilities"] = wsi_batch_zarr_group["probabilities"]
        cum_output["predictions"] = wsi_batch_zarr_group["predictions"]
        if return_coordinates:
            cum_output["coordinates"] = wsi_batch_zarr_group["coordinates"]
        if return_labels:
            # We do not use tolist here because label may be of mixed types
            # and hence collated as list by torch
            cum_output["labels"] = wsi_batch_zarr_group["labels"]

        cum_output["label"] = img_label
        # add extra information useful for downstream analysis
        cum_output["resolution"] = highest_input_resolution["resolution"]
        cum_output["units"] = highest_input_resolution["units"]

        return cum_output

    @abstractmethod
    def post_process_wsi(
        self: EngineABC,
        raw_output: dict,
        save_dir: Path,
        output_type: str,
        **kwargs: dict,
    ) -> dict | AnnotationStore:
        """Post-process a WSI.

        Args:
            raw_output (dict):
                A dictionary of patch prediction information.
            save_dir (Path):
                Output Path to directory to save the patch dataset output to a
                `.zarr` or `.db` file
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (dict):
                Keyword Args to update setup_patch_dataset() method attributes.

        Returns: (dict or Path):
            if the output_type is "AnnotationStore", the function returns the patch
            predictor output as an SQLiteStore containing Annotations stored to a `.db`
            file. Otherwise, the function defaults to returning patch predictor output
            stored to a `.zarr` file.

        """
        output_file = (
            kwargs["output_file"] and kwargs.pop("output_file")
            if "output_file" in kwargs
            else "output"
        )
        save_path = save_dir / output_file

        if output_type == "AnnotationStore":
            # scale_factor set from kwargs
            scale_factor = kwargs.get("scale_factor", (1.0, 1.0))
            # class_dict set from kwargs
            class_dict = kwargs.get("class_dict", None)

            return dict_to_store(raw_output, scale_factor, class_dict, save_path)

        # referring to the zarr group generated during the infer_wsi step
        return save_path.parent.absolute() / (save_path.stem + ".zarr")

    def _load_ioconfig(self: EngineABC, ioconfig: ModelIOConfigABC) -> ModelIOConfigABC:
        """Helper function to load ioconfig.

        If the model is provided by TIAToolbox it will load the default ioconfig.
        Otherwise, ioconfig must be specified.

        Args:
            ioconfig (ModelIOConfigABC):
                IO configuration to run the engines.

        Raises:
             ValueError:
                If no io configuration is provided or found in the pretrained TIAToolbox
                models.

        Returns:
            ModelIOConfigABC:
                The ioconfig used for the run.

        """
        if self.ioconfig is None and ioconfig is None:
            msg = (
                "Please provide a valid ModelIOConfigABC. "
                "No default ModelIOConfigABC found."
            )
            raise ValueError(msg)

        if ioconfig is not None:
            self.ioconfig = ioconfig

        return self.ioconfig

    def _update_ioconfig(
        self: EngineABC,
        ioconfig: ModelIOConfigABC,
        patch_input_shape: IntPair,
        stride_shape: IntPair,
        resolution: Resolution,
        units: Units,
    ) -> ModelIOConfigABC:
        """Update the ioconfig.

        Args:
            ioconfig (:class:`ModelIOConfigABC`):
                Input ioconfig for PatchPredictor.
            patch_input_shape (tuple):
                Size of patches input to the model. Patches are at
                requested read resolution, not with respect to level 0,
                and must be positive.
            stride_shape (tuple):
                Stride using during tile and WSI processing. Stride is
                at requested read resolution, not with respect to
                level 0, and must be positive. If not provided,
                `stride_shape=patch_input_shape`.
            resolution (Resolution):
                Resolution used for reading the image. Please see
                :obj:`WSIReader` for details.
            units (Units):
                Units of resolution used for reading the image.

        Returns:
            Updated Patch Predictor IO configuration.

        """
        config_flag = (
            patch_input_shape is None,
            resolution is None,
            units is None,
        )
        if ioconfig:
            return ioconfig

        if self.ioconfig is None and any(config_flag):
            msg = (
                "Must provide either "
                "`ioconfig` or `patch_input_shape`, `resolution`, and `units`."
            )
            raise ValueError(
                msg,
            )

        if stride_shape is None:
            stride_shape = patch_input_shape

        if self.ioconfig:
            ioconfig = copy.deepcopy(self.ioconfig)
            # ! not sure if there is a nicer way to set this
            if patch_input_shape is not None:
                ioconfig.patch_input_shape = patch_input_shape
            if stride_shape is not None:
                ioconfig.stride_shape = stride_shape
            if resolution is not None:
                ioconfig.input_resolutions[0]["resolution"] = resolution
            if units is not None:
                ioconfig.input_resolutions[0]["units"] = units

            return ioconfig

        return ModelIOConfigABC(
            input_resolutions=[{"resolution": resolution, "units": units}],
            patch_input_shape=patch_input_shape,
            stride_shape=stride_shape,
            output_resolutions=[],
        )

    @staticmethod
    def _validate_images_masks(images: list | np.ndarray) -> list | np.ndarray:
        """Validate input images for a run."""
        if not isinstance(images, (list, np.ndarray)):
            msg = "Input must be a list of file paths or a numpy array."
            raise TypeError(
                msg,
            )

        if isinstance(images, np.ndarray) and images.ndim != 4:  # noqa: PLR2004
            msg = (
                "The input numpy array should be four dimensional."
                "The shape of the numpy array should be NHWC."
            )
            raise ValueError(msg)

        return images

    @staticmethod
    def _validate_input_numbers(
        images: list | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
    ) -> None:
        """Validates number of input images, masks and labels."""
        if masks is None and labels is None:
            return

        len_images = len(images)

        if masks is not None and len_images != len(masks):
            msg = (
                f"len(masks) is not equal to len(images) "
                f": {len(masks)} != {len(images)}"
            )
            raise ValueError(
                msg,
            )

        if labels is not None and len_images != len(labels):
            msg = (
                f"len(labels) is not equal to len(images) "
                f": {len(labels)} != {len(images)}"
            )
            raise ValueError(
                msg,
            )
        return

    def run(
        self: EngineABC,
        images: list[os | Path | WSIReader] | np.ndarray,
        masks: list[os | Path] | np.ndarray | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: dict,
    ) -> AnnotationStore | Path | str | dict:
        """Run the engine on input images.

        Args:
            images (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list.
            masks (list | None):
                List of masks. Only utilised when patch_mode is False.
                Patches are only generated within a masked area.
                If not provided, then a tissue mask will be automatically
                generated for whole slide images.
            labels (list | None):
                List of labels. Only a single label per image is supported.
            patch_mode (bool):
                Whether to treat input image as a patch or WSI.
                default = True.
            ioconfig (IOPatchPredictorConfig):
                IO configuration.
            save_dir (str or pathlib.Path):
                Output directory to save the results.
                If save_dir is not provided when patch_mode is False,
                then for a single image the output is created in the current directory.
                If there are multiple WSIs as input then the user must provide
                path to save directory otherwise an OSError will be raised.
            overwrite (bool):
                Whether to overwrite the results. Default = False.
            output_type (str):
                The format of the output type. "output_type" can be
                "zarr", "AnnotationStore". Default is "zarr".
                When saving in the zarr format the output is saved using the
                `python zarr library <https://zarr.readthedocs.io/en/stable/>`__
                as a zarr group. If the required output type is an "AnnotationStore"
                then the output will be intermediately saved as zarr but converted
                to :class:`AnnotationStore` and saved as a `.db` file
                at the end of the loop.
            **kwargs (dict):
                Keyword Args to update :class:`EngineABC` attributes.

        Returns:
            (:class:`numpy.ndarray`, dict):
                Model predictions of the input dataset. If multiple
                whole slide images are provided as input,
                or save_output is True, then results are saved to
                `save_dir` and a dictionary indicating save location for
                each input is returned.

                The dict has the following format:

                - img_path: path of the input image.
                - raw: path to save location for raw prediction,
                  saved in .json.
                - merged: path to .npy contain merged
                  predictions if `merge_predictions` is `True`.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> predictor = EngineABC(model="resnet18-kather100k")
            >>> output = predictor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'raw': '0.raw.json', 'merged': '0.merged.npy'}
            >>> output['wsi2.svs']
            ... {'raw': '1.raw.json', 'merged': '1.merged.npy'}

            >>> predictor = EngineABC(model="alexnet-kather100k")
            >>> output = predictor.run(
            >>>     images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            >>>     labels=list(range(10)),
            >>>     on_gpu=False,
            >>>     )
            >>> output
            ... {'predictions': [[0.7716791033744812, 0.0111849969252944, ...,
            ... 0.034451354295015335, 0.004817609209567308]],
            ... 'labels': [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
            ... tensor(5), tensor(6), tensor(7), tensor(8), tensor(9)]}

            >>> predictor = EngineABC(model="alexnet-kather100k")
            >>> save_dir = Path("/tmp/patch_output/")
            >>> output = eng.run(
            >>>     images=np.zeros((10, 224, 224, 3), dtype=np.uint8),
            >>>     on_gpu=False,
            >>>     verbose=False,
            >>>     save_dir=save_dir,
            >>>     overwrite=True
            >>>     )
            >>> output
            ... '/tmp/patch_output/output.zarr'
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.patch_mode = patch_mode

        self._validate_input_numbers(images=images, masks=masks, labels=labels)
        self.images = self._validate_images_masks(images=images)

        if masks is not None:
            self.masks = self._validate_images_masks(images=masks)

        self.labels = labels

        # if necessary Move model parameters to "cpu" or "gpu" and update ioconfig
        self._ioconfig = self._load_ioconfig(ioconfig=ioconfig)
        self.model = model_to(model=self.model, device=self.device)

        save_dir = prepare_engines_save_dir(
            save_dir=save_dir,
            patch_mode=patch_mode,
            overwrite=overwrite,
        )

        if patch_mode:
            data_loader = self.get_dataloader(
                images=self.images,
                labels=self.labels,
                patch_mode=patch_mode,
            )
            raw_predictions = self.infer_patches(
                data_loader=data_loader,
            )
            return self.post_process_patches(
                raw_predictions=raw_predictions,
                output_type=output_type,
                save_dir=save_dir,
                **kwargs,
            )

        self._ioconfig = self._update_ioconfig(
            ioconfig,
            self.patch_input_shape,
            self.stride_shape,
            self.resolution,
            self.units,
        )

        fx_list = self._ioconfig.scale_to_highest(
            self._ioconfig.input_resolutions,
            self._ioconfig.input_resolutions[0]["units"],
        )
        fx_list = zip(fx_list, self._ioconfig.input_resolutions)
        fx_list = sorted(fx_list, key=lambda x: x[0])
        highest_input_resolution = fx_list[0][1]

        wsi_output_dict = OrderedDict()

        for idx, img_path in enumerate(self.images):
            img_path_ = Path(img_path)
            img_label = None if labels is None else labels[idx]
            img_mask = None if masks is None else masks[idx]

            dataloader = self.get_dataloader(
                images=img_path_,
                masks=img_mask,
                ioconfig=self._ioconfig,
                patch_mode=patch_mode,
            )

            # Only a single label per whole-slide image is supported
            kwargs["return_labels"] = False

            # custom output file name
            output_file = img_path_.stem + f"_{idx:0{len(str(len(self.images)))}d}"

            raw_output = self.infer_wsi(
                dataloader=dataloader,
                img_label=img_label,
                highest_input_resolution=highest_input_resolution,
                save_dir=save_dir,
                output_file=output_file,
                **kwargs,
            )

            # WSI output dict can have either Zarr paths or Annotation Stores
            wsi_output_dict[output_file] = self.post_process_wsi(
                raw_output,
                save_dir=save_dir,
                output_file=output_file,
                output_type=output_type,
            )

        return wsi_output_dict
