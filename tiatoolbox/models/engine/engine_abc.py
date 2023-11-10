"""Defines Abstract Base Class for TIAToolbox Model Engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import torch
import tqdm
from torch import nn

from tiatoolbox import logger
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.dataset.dataset_abc import PatchDataset, WSIPatchDataset
from tiatoolbox.models.models_abc import load_torch_model, model_to
from tiatoolbox.utils.misc import dict_to_store, dict_to_zarr

if TYPE_CHECKING:  # pragma: no cover
    import os

    from torch.utils.data import DataLoader

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.wsicore.wsireader import WSIReader
    from tiatoolbox.typing import IntPair, Resolution, Units

    from .io_config import ModelIOConfigABC


def prepare_engines_save_dir(
    save_dir: os | Path | None,
    len_images: int,
    *,
    patch_mode: bool,
    overwrite: bool,
) -> Path | None:
    """Create directory if not defined and number of images is more than 1.

    Args:
        save_dir (str or Path):
            Path to output directory.
        len_images (int):
            List of inputs to process.
        patch_mode(bool):
            Whether to treat input image as a patch or WSI.
        overwrite (bool):
            Whether to overwrite the results. Default = False.

    Returns:
        :class:`Path`:
            Path to output directory.

    """
    if patch_mode is True:
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=overwrite)
        return save_dir

    if save_dir is None:
        if len_images > 1:
            msg = (
                "More than 1 WSIs detected but there is no save directory provided."
                "Please provide a 'save_dir'."
            )
            raise OSError(msg)
        return (
            Path.cwd()
        )  # save the output to current working directory and return save_dir

    if len_images > 1:
        logger.info(
            "When providing multiple whole slide images, "
            "the outputs will be saved and the locations of outputs "
            "will be returned to the calling function.",
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=overwrite)

    return save_dir


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox.

    Args:
        model (str | nn.Module):
            A PyTorch model. Default is `None`.
            The user can request pretrained models from the toolbox using
            the list of pretrained models available at this `link
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights.
        weights (str or Path):
            Path to the weight of the corresponding `model`.

            >>> engine = EngineABC(
            ...    model="pretrained-model-name",
            ...    weights="pretrained-local-weights.pth")

        batch_size (int):
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data using :class:`torch.utils.data.Dataset`.
            Please note that they will also perform preprocessing. default = 0
        num_post_proc_workers (int):
            Number of workers to postprocess the results of the model. default = 0
        device (str):
            Select the device to run the model. Default is "cpu".
        verbose (bool):
            Whether to output logging information.

    Attributes:
        images (list of str or list of :obj:`Path` or NHWC :obj:`numpy.ndarray`):
            A NHWC image or a path to WSI.
        patch_mode (str):
            Whether to treat input image as a patch or WSI.
            default = True.
        model (str | nn.Module):
            Defined PyTorch model.
            Name of an existing model supported by the TIAToolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `weights` argument. Argument
            is case-insensitive.
        ioconfig (ModelIOConfigABC):
            Input IO configuration to run the Engine.
        _ioconfig ():
            Runtime ioconfig.
        return_labels (bool):
            Whether to return the labels with the predictions.
        merge_predictions (bool):
            Whether to merge the predictions to form a 2-dimensional
            map. This is only applicable if `patch_mode` is False in inference.
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
        labels (list | None):
                List of labels. Only a single label per image is supported.
        device (str):
            Select the device to run the model. Default is "cpu".
        num_loader_workers (int):
            Number of workers used in torch.utils.data.DataLoader.
        verbose (bool):
            Whether to output logging information.

    Examples:
        >>> # array of list of 2 image patches as input
        >>> import numpy as np
        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = EngineABC(model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # array of list of 2 image patches as input
        >>> import numpy as np
        >>> data = np.array([np.ndarray, np.ndarray])
        >>> engine = EngineABC(model="resnet18-kather100k")
        >>> output = engine.run(data, patch_mode=True)

        >>> # list of 2 image files as input
        >>> image = ['path/image1.png', 'path/image2.png']
        >>> engine = EngineABC(model="resnet18-kather100k")
        >>> output = engine.run(image, patch_mode=False)

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> engine = EngineABC(model="resnet18-kather100k")
        >>> output = engine.run(wsi_file, patch_mode=True)

    """

    def __init__(
        self: EngineABC,
        model: str | nn.Module,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_post_proc_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        """Initialize Engine."""
        super().__init__()

        self.masks = None
        self.images = None
        self.patch_mode = None
        self.device = device

        # Initialize model with specified weights and ioconfig.
        self.model, self.ioconfig = self._initialize_model_ioconfig(
            model=model,
            weights=weights,
        )
        self.model = model_to(model=self.model, device=self.device)
        self._ioconfig = self.ioconfig  # runtime ioconfig

        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.num_post_proc_workers = num_post_proc_workers
        self.verbose = verbose
        self.return_labels = False
        self.merge_predictions = False
        self.units = "baseline"
        self.resolution = 1.0
        self.patch_input_shape = None
        self.stride_shape = None
        self.labels = None

    @staticmethod
    def _initialize_model_ioconfig(
        model: str | nn.Module,
        weights: str | Path | None,
    ) -> tuple[nn.Module, ModelIOConfigABC | None]:
        """Helper function to initialize model and ioconfig attributes.

        If a pretrained model provided by the TIAToolbox is requested. The model
        can be specified as a string otherwise torch.nn.Module is required.
        This function also loads the :class:`ModelIOConfigABC` using the information
        from the pretrained models in TIAToolbox. If ioconfig is not available then it
        should be provided in the :func:`run` function.

        Args:
            model (str | nn.Module):
                A torch model which should be run by the engine.

            weights (str | Path | None):
                Path to pretrained weights. If no pretrained weights are provided
                and the model is provided by TIAToolbox, then pretrained weights will
                be automatically loaded from the TIA servers.

        Returns:
            nn.Module:
                The requested PyTorch model.

            ModelIOConfigABC | None:
                The model io configuration for TIAToolbox pretrained models.
                Otherwise, None.

        """
        if not isinstance(model, (str, nn.Module)):
            msg = "Input model must be a string or 'torch.nn.Module'."
            raise TypeError(msg)

        if isinstance(model, str):
            # ioconfig is retrieved from the pretrained model in the toolbox.
            # list of pretrained models in the TIA Toolbox is available here:
            # https://tia-toolbox.readthedocs.io/en/add-bokeh-app/pretrained.html
            # no need to provide ioconfig in EngineABC.run() this case.
            return get_pretrained_model(model, weights)

        if weights is not None:
            model = load_torch_model(model=model, weights=weights)
        
        return model, None
    
    """ 
    different approach now 
    def setup_dataloader(
        self: EngineABC,
        images: np.ndarray | list,
        labels: list,
        masks: list | None = None,
        ioconfig: ModelIOConfigABC| None = None,    
        *,
        patch_mode: bool=True,
    ) -> torch.utils.data.DataLoader:

        # if a labels is provided, then return with the prediction
        self.return_labels = bool(labels)

        #ensure images is list of paths if patch mode is wsi
        input_images_is_path_like = isinstance(images[0], (str, Path))
        if patch_mode is None and not input_images_is_path_like:
            msg = "Input to `wsi` mode must be a list of file paths."
            raise ValueError(msg)

        if patch_mode:
            dataset = PatchDataset(inputs=images, labels=labels)
        
        else:
            dataset = WSIPatchDataset(
                images,
                mode='wsi',
                mask_path=masks,
                patch_input_shape=ioconfig.patch_input_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )

        dataset.preproc_func = self.model.preproc_func
        
        # preprocessing must be defined with the dataset
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_workers,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )"""
    
    def pre_process_patches(
        self: EngineABC,
        images: np.ndarray | list,
        labels: list,
    ) -> torch.utils.data.DataLoader:
        """Pre-process an image patch."""
        if labels:
            # if a labels is provided, then return with the prediction
            self.return_labels = bool(labels)

        dataset = PatchDataset(inputs=images, labels=labels)
        dataset.preproc_func = self.model.preproc_func

        # preprocessing must be defined with the dataset
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_workers,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def infer_patches(
        self: EngineABC,
        data_loader: DataLoader,
    ) -> dict:
        """Model inference on an image patch."""
        progress_bar = None

        if self.verbose:
            progress_bar = tqdm.tqdm(
                total=int(len(data_loader)),
                leave=True,
                ncols=80,
                ascii=True,
                position=0,
            )
        raw_predictions = {
            "predictions": [],
        }

        if self.return_labels:
            raw_predictions["labels"] = []

        for _, batch_data in enumerate(data_loader):
            batch_output_predictions = self.model.infer_batch(
                self.model,
                batch_data["image"],
                device=self.device,
            )

            raw_predictions["predictions"].extend(batch_output_predictions.tolist())

            if self.return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                raw_predictions["labels"].extend(list(batch_data["label"]))

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        return raw_predictions

    def post_process_patches(
        self: EngineABC,
        raw_predictions: dict,
        output_type: str,
        save_dir: Path | None = None,
        **kwargs: dict,
    ) -> Path | AnnotationStore:
        """Post-process image patches.

        Args:
            raw_predictions (dict):
                A dictionary of patch prediction information.
            save_dir (Path):
                Optional Output Path to directory to save the patch dataset output to a
                `.zarr` or `.db` file, provided patch_mode is True. if the patch_mode is
                  False then save_dir is required.
            output_type (str):
                The desired output type for resulting patch dataset.
            **kwargs (dict):
                Keyword Args to update setup_patch_dataset() method attributes.

        Returns: (dict, Path, :class:`SQLiteStore`):
            if the output_type is "AnnotationStore", the function returns the patch
            predictor output as an SQLiteStore containing Annotations for each or the
            Path to a `.db` file depending on whether a save_dir Path is provided.
            Otherwise, the function defaults to returning patch predictor output, either
            as a dict or the Path to a `.zarr` file depending on whether a save_dir Path
            is provided.

        """
        if not save_dir and output_type != "AnnotationStore":
            return raw_predictions

        output_file = (
            kwargs["output_file"] and kwargs.pop("output_file")
            if "output_file" in kwargs
            else "output"
        )

        save_path = save_dir / output_file

        if output_type == "AnnotationStore":
            # scale_factor set from kwargs
            scale_factor = kwargs["scale_factor"] if "scale_factor" in kwargs else None
            # class_dict set from kwargs
            class_dict = kwargs["class_dict"] if "class_dict" in kwargs else None

            return dict_to_store(raw_predictions, scale_factor, class_dict, save_path)

        return dict_to_zarr(
            raw_predictions,
            save_path,
            **kwargs,
        )


    @staticmethod
    def merge_predictions(
        img: str | Path | np.ndarray,
        output: dict,
        resolution: Resolution | None = None,
        units: Units | None = None,
        post_proc_func: Callable | None = None,
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
            post_proc_func (callable):
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
                if post_proc_func is not None:
                    output = post_proc_func(output)
                else:
                    output = np.argmax(output, axis=-1)
                # to make sure background is 0 while class will be 1...N
                output[denominator > 0] += 1
        
        return output


    @abstractmethod
    def pre_process_wsi(self: EngineABC,
        images: list,
        labels: list,
        masks: list | None = None,
        ioconfig: ModelIOConfigABC| None = None,    
        *,
        patch_mode: bool=True,
    ) -> list[torch.utils.data.DataLoader]:
        """Pre-process a WSI."""

        dataloaders = []
        
        for idx, img_path in enumerate(images):
            img_path_ = Path(img_path)
            img_label = None if labels is None else labels[idx]
            img_mask = None if masks is None else masks[idx]

            dataset = WSIPatchDataset(
                img_path_,
                mode='wsi',
                mask_path=img_mask,
                patch_input_shape=ioconfig.patch_input_shape,
                stride_shape=ioconfig.stride_shape,
                resolution=ioconfig.input_resolutions[0]["resolution"],
                units=ioconfig.input_resolutions[0]["units"],
            )

            dataset.preproc_func = self.model.preproc_func

            # preprocessing must be defined with the dataset
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_loader_workers,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )

            #list of dataloaders per image
            dataloaders.append(dataloader)
        
        return dataloaders


    @abstractmethod
    def infer_wsi(
        self: EngineABC,
        model: torch.nn.Module,
        dataloaders: list,
        #should be moved to wsi_post_processing
        images: list,
        labels: list,
        highest_input_resolution: list[dict],
        merge_predictions: bool = False,
        ) -> list:
        """Model inference on a WSI."""

        # return coordinates of patches processed within a tile / whole-slide image
        return_coordinates = True
        raw_predictions_per_wsi = []
        for idx, dataloader in enumerate(dataloaders):
            
            # will be moved to post processing
            img_path_ = images[idx]
            img_label = None if labels is None else labels[idx]

            pbar = None
            if self.verbose:
                pbar = tqdm.tqdm(
                    total=int(len(dataloader)),
                    leave=True,
                    ncols=80,
                    ascii=True,
                    position=0,
                )

            output_model = {
                "probabilities": [],
                "predictions": [],
                "coordinates": [],
                "labels": [],
            }

            for _, batch_data in enumerate(dataloader):
                batch_output_probabilities = self.model.infer_batch(
                    model,
                    batch_data["image"],
                    device=self.device,
                )
                # We get the index of the class with the maximum probability
                batch_output_predictions = self.model.postproc_func(
                    batch_output_probabilities,
                )

                # tolist might be very expensive
                output_model["probabilities"].extend(
                    batch_output_probabilities.tolist())
                output_model["predictions"].extend(
                    batch_output_predictions.tolist())

                if self.verbose: pbar.update()

            if self.verbose: pbar.close()

            output_model["label"] = img_label
            # add extra information useful for downstream analysis
            output_model["pretrained_model"] = self.model
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
                    post_proc_func=self.model.postproc,
                )
                outputs.append(merged_prediction)

            # should we just add to a list of raw predictions and 
            # deal with in post processing?
            raw_predictions_per_wsi.append(outputs)
            
        return raw_predictions_per_wsi


    @abstractmethod
    def post_process_wsi(self: EngineABC) -> NoReturn:

        """Post-process a WSI."""

        ## to bo implemented

        #will be moved/implemented in Post processing wsi
        # if return_coordinates:
        #     cum_output["coordinates"].extend(batch_data["coords"].tolist())
        # if return_labels:  # be careful of `s`
        #     # We do not use tolist here because label may be of mixed types
        #     # and hence collated as list by torch
        #     cum_output["labels"].extend(list(batch_data["label"]))

        #will be moved/implemented in Post processing wsi
            # if not return_probabilities:
            #     cum_output.pop("probabilities")
            # if not return_labels:
            #     cum_output.pop("labels")
            # if not return_coordinates:
            #     cum_output.pop("coordinates")

        raise NotImplementedError


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
    
    #should we revert to ModelIOConfigABC instead of IOPatchPredictorConfig?
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
        patch_input_shape: tuple[int, int] | None = None,
        stride_shape: tuple[int, int] | None = None,
        resolution: Resolution | None = None,
        units: Units = None,
        *,
        patch_mode: bool = True,
        save_dir: os | Path | None = None,  # None will not save output
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: dict,
    ) -> AnnotationStore | Path | str:
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
            save_dir,
            len(self.images),
            patch_mode=patch_mode,
            overwrite=overwrite,
        )

        if patch_mode:
            data_loader = self.pre_process_patches(
                self.images,
                self.labels,
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
        
        ioconfig = self._update_ioconfig(
            ioconfig,
            patch_input_shape,
            stride_shape,
            resolution,
            units,
        )

        #since we're not expecting mode == "tile" should the 
        #Resolutions will be converted to baseline value.
        #ioconfig = ioconfig.to_baseline()

        fx_list = ioconfig.scale_to_highest(
            ioconfig.input_resolutions,
            ioconfig.input_resolutions[0]["units"],
        )
        fx_list = zip(fx_list, ioconfig.input_resolutions)
        fx_list = sorted(fx_list, key=lambda x: x[0])
        highest_input_resolution = fx_list[0][1]

        return {"save_dir": save_dir}
