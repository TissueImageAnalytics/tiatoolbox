"""This module implements patch level prediction."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import tqdm

from tiatoolbox import logger, rcParam
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model
from tiatoolbox.models.dataset.classification import PatchDataset, WSIPatchDataset
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig
from tiatoolbox.utils import misc, save_as_json
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.typing import IntPair, Resolution, Units


class IOPatchPredictorConfig(IOSegmentorConfig):
    """Contains patch predictor input and output information."""

    def __init__(
        self: IOPatchPredictorConfig,
        patch_input_shape: IntPair = None,
        input_resolutions: Resolution = None,
        stride_shape: IntPair = None,
        **kwargs: dict,
    ) -> None:
        """Initialize :class:`IOPatchPredictorConfig`."""
        stride_shape = patch_input_shape if stride_shape is None else stride_shape
        super().__init__(
            input_resolutions=input_resolutions,
            output_resolutions=[],
            stride_shape=stride_shape,
            patch_input_shape=patch_input_shape,
            patch_output_shape=patch_input_shape,
            save_resolution=None,
            **kwargs,
        )


class PatchPredictor:
    r"""Patch level predictor.

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

          >>> predictor = PatchPredictor(
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
        img (:obj:`str` or :obj:`pathlib.Path` or :obj:`numpy.ndarray`):
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
        num_loader_worker (int):
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

    """

    def __init__(
        self: PatchPredictor,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        model: torch.nn.Module = None,
        pretrained_model: str | None = None,
        pretrained_weights: str | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`PatchPredictor`."""
        super().__init__()

        self.imgs = None
        self.mode = None

        if model is None and pretrained_model is None:
            msg = "Must provide either `model` or `pretrained_model`."
            raise ValueError(msg)

        if model is not None:
            self.model = model
            ioconfig = None  # retrieve iostate from provided model ?
        else:
            model, ioconfig = get_pretrained_model(pretrained_model, pretrained_weights)

        self.ioconfig = ioconfig  # for storing original
        self._ioconfig = None  # for storing runtime
        self.model = (
            compile_model(  # for runtime, such as after wrapping with nn.DataParallel
                model,
                mode=rcParam["torch_compile_mode"],
            )
        )
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_workers
        self.verbose = verbose

    @staticmethod
    def merge_predictions(
        img: str | Path | np.ndarray,
        output: dict,
        resolution: Resolution | None = None,
        units: Units | None = None,
        postproc_func: Callable | None = None,
        *,
        return_raw: bool = False,
    ) -> np.ndarray:
        """Merge patch level predictions to form a 2-dimensional prediction map.

        #! Improve how the below reads.
        The prediction map will contain values from 0 to N, where N is
        the number of classes. Here, 0 is the background which has not
        been processed by the model and N is the number of classes
        predicted by the model.

        Args:
            img (:obj:`str` or :obj:`pathlib.Path` or :class:`numpy.ndarray`):
              A HWC image or a path to WSI.
            output (dict):
                Output generated by the model.
            resolution (Resolution):
                Resolution of merged predictions.
            units (Units):
                Units of resolution used when merging predictions. This
                must be the same `units` used when processing the data.
            postproc_func (callable):
                A function to post-process raw prediction from model. By
                default, internal code uses the `np.argmax` function.
            return_raw (bool):
                Return raw result without applying the `postproc_func`
                on the assembled image.

        Returns:
            :class:`numpy.ndarray`:
                Merged predictions as a 2D array.

        Examples:
            >>> # pseudo output dict from model with 2 patches
            >>> output = {
            ...     'resolution': 1.0,
            ...     'units': 'baseline',
            ...     'probabilities': [[0.45, 0.55], [0.90, 0.10]],
            ...     'predictions': [1, 0],
            ...     'coordinates': [[0, 0, 2, 2], [2, 2, 4, 4]],
            ... }
            >>> merged = PatchPredictor.merge_predictions(
            ...         np.zeros([4, 4]),
            ...         output,
            ...         resolution=1.0,
            ...         units='baseline'
            ... )
            >>> merged
            ... array([[2, 2, 0, 0],
            ...    [2, 2, 0, 0],
            ...    [0, 0, 1, 1],
            ...    [0, 0, 1, 1]])

        """
        reader = WSIReader.open(img)
        if isinstance(reader, VirtualWSIReader):
            logger.warning(
                "Image is not pyramidal hence read is forced to be "
                "at `units='baseline'` and `resolution=1.0`.",
                stacklevel=2,
            )
            resolution = 1.0
            units = "baseline"

        canvas_shape = reader.slide_dimensions(resolution=resolution, units=units)
        canvas_shape = canvas_shape[::-1]  # XY to YX

        # may crash here, do we need to deal with this ?
        output_shape = reader.slide_dimensions(
            resolution=output["resolution"],
            units=output["units"],
        )
        output_shape = output_shape[::-1]  # XY to YX
        fx = np.array(canvas_shape) / np.array(output_shape)

        if "probabilities" not in output:
            coordinates = output["coordinates"]
            predictions = output["predictions"]
            denominator = None
            output = np.zeros(list(canvas_shape), dtype=np.float32)
        else:
            coordinates = output["coordinates"]
            predictions = output["probabilities"]
            num_class = np.array(predictions[0]).shape[0]
            denominator = np.zeros(canvas_shape)
            output = np.zeros([*list(canvas_shape), num_class], dtype=np.float32)

        for idx, bound in enumerate(coordinates):
            prediction = predictions[idx]
            # assumed to be in XY
            # top-left for output placement
            tl = np.ceil(np.array(bound[:2]) * fx).astype(np.int32)
            # bot-right for output placement
            br = np.ceil(np.array(bound[2:]) * fx).astype(np.int32)
            output[tl[1] : br[1], tl[0] : br[0]] += prediction
            if denominator is not None:
                denominator[tl[1] : br[1], tl[0] : br[0]] += 1

        # deal with overlapping regions
        if denominator is not None:
            output = output / (np.expand_dims(denominator, -1) + 1.0e-8)
            if not return_raw:
                # convert raw probabilities to predictions
                if postproc_func is not None:
                    output = postproc_func(output)
                else:
                    output = np.argmax(output, axis=-1)
                # to make sure background is 0 while class will be 1...N
                output[denominator > 0] += 1
        return output

    def _predict_engine(
        self: PatchPredictor,
        dataset: torch.utils.data.Dataset,
        *,
        return_probabilities: bool = False,
        return_labels: bool = False,
        return_coordinates: bool = False,
        on_gpu: bool = True,
    ) -> np.ndarray:
        """Make a prediction on a dataset. The dataset may be mutated.

        Args:
            dataset (torch.utils.data.Dataset):
                PyTorch dataset object created using
                `tiatoolbox.models.data.classification.Patch_Dataset`.
            return_probabilities (bool):
                Whether to return per-class probabilities.
            return_labels (bool):
                Whether to return labels.
            return_coordinates (bool):
                Whether to return patch coordinates.
            on_gpu (bool):
                Whether to run model on the GPU.

        Returns:
            :class:`numpy.ndarray`:
                Model predictions of the input dataset

        """
        dataset.preproc_func = self.model.preproc_func

        # preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        if self.verbose:
            pbar = tqdm.tqdm(
                total=int(len(dataloader)),
                leave=True,
                ncols=80,
                ascii=True,
                position=0,
            )

        # use external for testing
        model = misc.model_to(model=self.model, on_gpu=on_gpu)

        cum_output = {
            "probabilities": [],
            "predictions": [],
            "coordinates": [],
            "labels": [],
        }
        for _, batch_data in enumerate(dataloader):
            batch_output_probabilities = self.model.infer_batch(
                model,
                batch_data["image"],
                on_gpu=on_gpu,
            )
            # We get the index of the class with the maximum probability
            batch_output_predictions = self.model.postproc_func(
                batch_output_probabilities,
            )

            # tolist might be very expensive
            cum_output["probabilities"].extend(batch_output_probabilities.tolist())
            cum_output["predictions"].extend(batch_output_predictions.tolist())
            if return_coordinates:
                cum_output["coordinates"].extend(batch_data["coords"].tolist())
            if return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                cum_output["labels"].extend(list(batch_data["label"]))

            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        if not return_probabilities:
            cum_output.pop("probabilities")
        if not return_labels:
            cum_output.pop("labels")
        if not return_coordinates:
            cum_output.pop("coordinates")

        return cum_output

    def _update_ioconfig(
        self: PatchPredictor,
        ioconfig: IOPatchPredictorConfig,
        patch_input_shape: IntPair,
        stride_shape: IntPair,
        resolution: Resolution,
        units: Units,
    ) -> IOPatchPredictorConfig:
        """Updates the ioconfig.

        Args:
            ioconfig (IOPatchPredictorConfig):
                Input ioconfig for PatchPredictor.
            patch_input_shape (IntPair):
                Size of patches input to the model. Patches are at
                requested read resolution, not with respect to level 0,
                and must be positive.
            stride_shape (IntPair):
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
            IOPatchPredictorConfig:
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

        return IOPatchPredictorConfig(
            input_resolutions=[{"resolution": resolution, "units": units}],
            patch_input_shape=patch_input_shape,
            stride_shape=stride_shape,
        )

    @staticmethod
    def _prepare_save_dir(save_dir: str | Path, imgs: list | np.ndarray) -> Path:
        """Create directory if not defined and number of images is more than 1.

        Args:
            save_dir (str or Path):
                Path to output directory.
            imgs (list, ndarray):
                List of inputs to process.

        Returns:
            :class:`Path`:
                Path to output directory.

        """
        if save_dir is None and len(imgs) > 1:
            logger.warning(
                "More than 1 WSIs detected but there is no save directory set."
                "All subsequent output will be saved to current runtime"
                "location under folder 'output'. Overwriting may happen!",
                stacklevel=2,
            )
            save_dir = Path.cwd() / "output"
        elif save_dir is not None and len(imgs) > 1:
            logger.warning(
                "When providing multiple whole-slide images / tiles, "
                "we save the outputs and return the locations "
                "to the corresponding files.",
                stacklevel=2,
            )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)

        return save_dir

    def _predict_patch(
        self: PatchPredictor,
        imgs: list | np.ndarray,
        labels: list,
        *,
        return_probabilities: bool,
        return_labels: bool,
        on_gpu: bool,
    ) -> np.ndarray:
        """Process patch mode.

        Args:
            imgs (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list. When using
                `tile` or `wsi` mode, the input must be a list of file
                paths.
            labels (list):
                List of labels. If using `tile` or `wsi` mode, then only
                a single label per image tile or whole-slide image is
                supported.
            return_probabilities (bool):
                Whether to return per-class probabilities.
            return_labels (bool):
                Whether to return the labels with the predictions.
            on_gpu (bool):
                Whether to run model on the GPU.

        Returns:
            :class:`numpy.ndarray`:
                Model predictions of the input dataset

        """
        if labels:
            # if a labels is provided, then return with the prediction
            return_labels = bool(labels)

        if labels and len(labels) != len(imgs):
            msg = f"len(labels) != len(imgs) : {len(labels)} != {len(imgs)}"
            raise ValueError(
                msg,
            )

        # don't return coordinates if patches are already extracted
        return_coordinates = False
        dataset = PatchDataset(imgs, labels)
        return self._predict_engine(
            dataset,
            return_probabilities=return_probabilities,
            return_labels=return_labels,
            return_coordinates=return_coordinates,
            on_gpu=on_gpu,
        )

    def _predict_tile_wsi(  # noqa: PLR0913
        self: PatchPredictor,
        imgs: list,
        masks: list | None,
        labels: list,
        mode: str,
        ioconfig: IOPatchPredictorConfig,
        save_dir: str | Path,
        highest_input_resolution: list[dict],
        *,
        save_output: bool,
        return_probabilities: bool,
        merge_predictions: bool,
        on_gpu: bool,
    ) -> list | dict:
        """Predict on Tile and WSIs.

        Args:
            imgs (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list. When using
                `tile` or `wsi` mode, the input must be a list of file
                paths.
            masks (list):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for
                whole-slide images or the entire image is processed for
                image tiles.
            labels (list):
                List of labels. If using `tile` or `wsi` mode, then only
                a single label per image tile or whole-slide image is
                supported.
            mode (str):
                Type of input to process. Choose from either `patch`,
                `tile` or `wsi`.
            return_probabilities (bool):
                Whether to return per-class probabilities.
            on_gpu (bool):
                Whether to run model on the GPU.
            ioconfig (IOPatchPredictorConfig):
                Patch Predictor IO configuration..
            merge_predictions (bool):
                Whether to merge the predictions to form a 2-dimensional
                map. This is only applicable for `mode='wsi'` or
                `mode='tile'`.
            save_dir (str or pathlib.Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            save_output (bool):
                Whether to save output for a single file. default=False
            highest_input_resolution (list(dict)):
                Highest available input resolution.


        Returns:
            dict:
                Results are saved to `save_dir` and a dictionary indicating save
                location for each input is returned. The dict is in the following
                format:
                    - img_path: path of the input image.
                        - raw: path to save location for raw prediction,
                          saved in .json.
                        - merged: path to .npy contain merged
                          predictions if
                        `merge_predictions` is `True`.

        """
        # return coordinates of patches processed within a tile / whole-slide image
        return_coordinates = True

        input_is_path_like = isinstance(imgs[0], (str, Path))
        default_save_dir = (
            imgs[0].parent / "output" if input_is_path_like else Path.cwd()
        )
        save_dir = default_save_dir if save_dir is None else Path(save_dir)

        # None if no output
        outputs = None

        self._ioconfig = ioconfig
        # generate a list of output file paths if number of input images > 1
        file_dict = OrderedDict()

        if len(imgs) > 1:
            save_output = True

        for idx, img_path in enumerate(imgs):
            img_path_ = Path(img_path)
            img_label = None if labels is None else labels[idx]
            img_mask = None if masks is None else masks[idx]

            dataset = WSIPatchDataset(
                img_path_,
                mode=mode,
                mask_path=img_mask,
                patch_input_shape=ioconfig.patch_input_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )
            output_model = self._predict_engine(
                dataset,
                return_labels=False,
                return_probabilities=return_probabilities,
                return_coordinates=return_coordinates,
                on_gpu=on_gpu,
            )
            output_model["label"] = img_label
            # add extra information useful for downstream analysis
            output_model["pretrained_model"] = self.pretrained_model
            output_model["resolution"] = highest_input_resolution["resolution"]
            output_model["units"] = highest_input_resolution["units"]

            outputs = [output_model]  # assign to a list
            merged_prediction = None
            if merge_predictions:
                merged_prediction = self.merge_predictions(
                    img_path_,
                    output_model,
                    resolution=output_model["resolution"],
                    units=output_model["units"],
                    postproc_func=self.model.postproc,
                )
                outputs.append(merged_prediction)

            if save_output:
                # dynamic 0 padding
                img_code = f"{idx:0{len(str(len(imgs)))}d}"

                save_info = {}
                save_path = save_dir / img_code
                raw_save_path = f"{save_path}.raw.json"
                save_info["raw"] = raw_save_path
                save_as_json(output_model, raw_save_path)
                if merge_predictions:
                    merged_file_path = f"{save_path}.merged.npy"
                    np.save(merged_file_path, merged_prediction)
                    save_info["merged"] = merged_file_path
                file_dict[str(img_path_)] = save_info

        return file_dict if save_output else outputs

    def predict(  # noqa: PLR0913
        self: PatchPredictor,
        imgs: list,
        masks: list | None = None,
        labels: list | None = None,
        mode: str = "patch",
        ioconfig: IOPatchPredictorConfig | None = None,
        patch_input_shape: tuple[int, int] | None = None,
        stride_shape: tuple[int, int] | None = None,
        resolution: Resolution | None = None,
        units: Units = None,
        *,
        return_probabilities: bool = False,
        return_labels: bool = False,
        on_gpu: bool = True,
        merge_predictions: bool = False,
        save_dir: str | Path | None = None,
        save_output: bool = False,
    ) -> np.ndarray | list | dict:
        """Make a prediction for a list of input data.

        Args:
            imgs (list, ndarray):
                List of inputs to process. when using `patch` mode, the
                input must be either a list of images, a list of image
                file paths or a numpy array of an image list. When using
                `tile` or `wsi` mode, the input must be a list of file
                paths.
            masks (list):
                List of masks. Only utilised when processing image tiles
                and whole-slide images. Patches are only processed if
                they are within a masked area. If not provided, then a
                tissue mask will be automatically generated for
                whole-slide images or the entire image is processed for
                image tiles.
            labels:
                List of labels. If using `tile` or `wsi` mode, then only
                a single label per image tile or whole-slide image is
                supported.
            mode (str):
                Type of input to process. Choose from either `patch`,
                `tile` or `wsi`.
            return_probabilities (bool):
                Whether to return per-class probabilities.
            return_labels (bool):
                Whether to return the labels with the predictions.
            on_gpu (bool):
                Whether to run model on the GPU.
            ioconfig (IOPatchPredictorConfig):
                Patch Predictor IO configuration.
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
                Units of resolution used for reading the image. Choose
                from either `level`, `power` or `mpp`. Please see
                :obj:`WSIReader` for details.
            merge_predictions (bool):
                Whether to merge the predictions to form a 2-dimensional
                map. This is only applicable for `mode='wsi'` or
                `mode='tile'`.
            save_dir (str or pathlib.Path):
                Output directory when processing multiple tiles and
                whole-slide images. By default, it is folder `output`
                where the running script is invoked.
            save_output (bool):
                Whether to save output for a single file. default=False

        Returns:
            (:class:`numpy.ndarray` or list or dict):
                Model predictions of the input dataset. If multiple
                image tiles or whole-slide images are provided as input,
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
            >>> predictor = PatchPredictor(
            ...                 pretrained_model="resnet18-kather100k")
            >>> output = predictor.predict(wsis, mode="wsi")
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... {'raw': '0.raw.json', 'merged': '0.merged.npy'}
            >>> output['wsi2.svs']
            ... {'raw': '1.raw.json', 'merged': '1.merged.npy'}

        """
        if mode not in ["patch", "wsi", "tile"]:
            msg = f"{mode} is not a valid mode. Use either `patch`, `tile` or `wsi`"
            raise ValueError(
                msg,
            )
        if mode == "patch":
            return self._predict_patch(
                imgs,
                labels,
                return_probabilities=return_probabilities,
                return_labels=return_labels,
                on_gpu=on_gpu,
            )

        if not isinstance(imgs, list):
            msg = "Input to `tile` and `wsi` mode must be a list of file paths."
            raise TypeError(
                msg,
            )

        if mode == "wsi" and masks is not None and len(masks) != len(imgs):
            msg = f"len(masks) != len(imgs) : {len(masks)} != {len(imgs)}"
            raise ValueError(
                msg,
            )

        ioconfig = self._update_ioconfig(
            ioconfig,
            patch_input_shape,
            stride_shape,
            resolution,
            units,
        )
        if mode == "tile":
            logger.warning(
                "WSIPatchDataset only reads image tile at "
                '`units="baseline"`. Resolutions will be converted '
                "to baseline value.",
                stacklevel=2,
            )
            ioconfig = ioconfig.to_baseline()

        fx_list = ioconfig.scale_to_highest(
            ioconfig.input_resolutions,
            ioconfig.input_resolutions[0]["units"],
        )
        fx_list = zip(fx_list, ioconfig.input_resolutions)
        fx_list = sorted(fx_list, key=lambda x: x[0])
        highest_input_resolution = fx_list[0][1]

        save_dir = self._prepare_save_dir(save_dir, imgs)

        return self._predict_tile_wsi(
            imgs=imgs,
            masks=masks,
            labels=labels,
            mode=mode,
            return_probabilities=return_probabilities,
            on_gpu=on_gpu,
            ioconfig=ioconfig,
            merge_predictions=merge_predictions,
            save_dir=save_dir,
            save_output=save_output,
            highest_input_resolution=highest_input_resolution,
        )
