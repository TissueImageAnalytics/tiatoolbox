"""This module enables interactive segmentation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm

from tiatoolbox import logger
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.models.models_abc import model_to
from tiatoolbox.tools.patchextraction import PointsPatchExtractor
from tiatoolbox.utils.misc import dict_to_store_semantic_segmentor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import Callable, IntBounds, IntPair, Resolution, Units


class PromptSegmentor(SemanticSegmentor):
    """Engine for prompt-based segmentation of WSIs.

    This class is designed to work with the SAM model architecture.
    It allows for interactive segmentation by providing point and bounding box
    coordinates as prompts. The model can be used in both tile and WSI modes,
    where tile mode processes individual image patches and WSI mode processes
    whole-slide images. The class also supports multi-prompt segmentation,
    where multiple point and bounding box coordinates can be provided for
    segmentation.

    Args:
        model (SAM):
            Model architecture to use.
        batch_size (int):
            Batch size for processing.
        num_loader_workers (int):
            Number of workers for data loading.
        dataset_class (Callable):
            Dataset class to use.

    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        batch_size: int = 4,
        num_loader_workers: int = 0,
        dataset_class: Callable = WSIStreamDataset,
    ) -> None:
        """Initializes the PromptSegmentor."""
        if model is None:
            model = SAM()
        super().__init__(
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
            model=model,
            dataset_class=dataset_class,
        )
        self.multi_prompt = True

    def predict(  # skipcq: PYL-W0221
        self,
        imgs: list,
        masks: list | None = None,
        mode: str = "tile",
        ioconfig: IOSegmentorConfig = None,
        point_coords: list[list[IntPair]] | None = None,
        box_coords: list[list[IntBounds]] | None = None,
        save_dir: str | Path | None = None,
        device: str = "cpu",
        *,
        multi_prompt: bool = True,
        crash_on_exception: bool = False,
        **ioconfig_kwargs: dict,
    ) -> list[tuple[Path, Path]]:
        """Predict on a list of WSIs using prompts.

        Args:
            imgs (list, ndarray):
                A list of paths to the input WSIs.
            masks (list):
                A list of masks corresponding to the input WSIs.
                Used to filter the coordinates of patches for inference.
            mode (str):
                The mode of prediction. Can be either `tile` or `wsi`.
                Affects how the input images are processed and saved.
                Use 'tile' for saving as raw numpy files, or 'wsi' for
                saving as annotations.
            ioconfig (:class:`IOSegmentorConfig`):
                Configuration for input/output processing.
            point_coords (list):
                Point coordinates for each image as `[x, y]` pairs.
                Stored as a list of lists of coordinates.
            box_coords (list):
                Bounding box coordinates for each image as `[x1, y1, x2, y2]` pairs.
                Stored as a list of lists of coordinates.
            save_dir (str, Path):
                Directory to save the output predictions.
            device (str):
                Device to run inference on.
            crash_on_exception (bool):
                Whether to crash on exceptions during prediction.
            multi_prompt (bool):
                Whether to use multiple prompts simulataneously for segmentation.
                If false, the image will be processed for each prompt separately.
            **ioconfig_kwargs (dict):
                Additional keyword arguments for the IOSegmentorConfig.

        Returns:
            output_paths(list[tuple[Path, Path]]):
                A list of tuples containing the input image path and the corresponding
                output path for the predictions.
                Each tuple is of the form (input_path, save_path).

        Examples:
            >>> segmentor = PromptSegmentor(model=model)
            >>> imgs = ["path/to/image1", "path/to/image2"]
            >>> masks = ["path/to/mask1", "path/to/mask2"]
            >>> point_coords = [[[100, 200]], [[150, 250]]]
            >>> box_coords = [[[50, 50, 150, 150]], [[100, 100, 200, 200]]]
            >>> output_paths = segmentor.predict(
            ...     imgs,
            ...     masks=masks,
            ...     mode="tile",
            ...     point_coords=point_coords,
            ...     box_coords=box_coords,
            ...     save_dir="output_dir",)

        """
        if mode not in ["wsi", "tile"]:
            msg = f"{mode} is not a valid mode. Use either `tile` or `wsi`."
            raise ValueError(msg)

        save_dir, self._cache_dir = self._prepare_save_dir(save_dir=save_dir)
        ioconfig_kwargs = self.pad_ioconfig(**ioconfig_kwargs)
        ioconfig = self._update_ioconfig(ioconfig, mode, **ioconfig_kwargs)

        # use external for testing
        self._device = device
        self._model = model_to(model=self.model, device=device)

        # workers should be > 0 else Value Error will be thrown
        self._prepare_workers()

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        ds = self.dataset_class(
            ioconfig=ioconfig,
            preproc=self.model.preproc_func,
            wsi_paths=imgs,
            mp_shared_space=mp_shared_space,
            mode=mode,
        )

        loader = torch_data.DataLoader(
            ds,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_loader_workers,
            persistent_workers=self.num_loader_workers > 0,
        )

        self._loader = loader
        self.imgs = imgs
        self.masks = masks
        self.multi_prompt = multi_prompt

        self._outputs = []

        for wsi_idx, image_path in enumerate(imgs):
            self._predict_wsi_handle_exception(
                imgs,
                wsi_idx,
                image_path,
                mode,
                ioconfig,
                point_coords[wsi_idx] if point_coords is not None else None,
                box_coords[wsi_idx] if box_coords is not None else None,
                save_dir,
                crash_on_exception=crash_on_exception,
            )

        # clean up the cache directories
        try:
            shutil.rmtree(self._cache_dir)
        except PermissionError:  # pragma: no cover
            logger.warning("Unable to remove %s", self._cache_dir)

        self._memory_cleanup()

        return self._outputs

    def _predict_one_wsi(  # skipcq: PYL-W0221
        self,
        wsi_idx: int,
        ioconfig: IOSegmentorConfig,
        point_coords: np.ndarray | None = None,
        box_coords: np.ndarray | None = None,
        save_path: str | Path | None = None,
        mode: str = "tile",
    ) -> tuple[Path, Path, Path]:
        """Predict on a single WSI.

        Args:
            wsi_idx (int):
                Index of the WSI to process.
            ioconfig (:class:`IOSegmentorConfig`):
                Configuration for input/output processing.
            point_coords (list):
                Point coordinates for the current image as [x, y] pairs.
            box_coords (list):
                Bounding box coordinates for the current image as
                [x1, y1, x2, y2] pairs.
            save_path (str, Path):
                Directory to save the output predictions.
            mode (str):
                The mode of prediction. Can be either "tile" or "wsi".

        Returns:
            tuple[Path, Path, Path]:
                A tuple containing the input image path and the corresponding
                output paths for the predictions.
                Each tuple is of the form (input_path, mask_path, score_path).
        """
        cache_dir = self._cache_dir / str(wsi_idx)
        cache_dir.mkdir(parents=True)

        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path,
            mask_path,
            mode,
            auto_get_mask=self.auto_generate_mask,
        )

        resolution = ioconfig.to_baseline().highest_input_resolution

        if mask_reader is not None:
            # Filters the point coordinates to only include those within the
            # mask. filter_coordinates only accepts bounding-box style coordinates
            point_coords = (
                point_coords[
                    PromptSegmentor.filter_coordinates(
                        mask_reader,
                        point_coords,
                        **resolution,
                    )
                ]
                if point_coords is not None
                else None
            )
            if np.array(point_coords).size == 0:
                point_coords = None

            box_coords = (
                PromptSegmentor.clip_coordinates(mask_reader, box_coords, **resolution)
                if box_coords is not None
                else None
            )
            if np.array(box_coords).size == 0:
                box_coords = None

        patch_inputs, point_coords, box_coords = self.get_coordinates(
            wsi_reader=wsi_reader,
            ioconfig=ioconfig,
            mode=mode,
            point_coords=point_coords,
            box_coords=box_coords,
            multi_prompt=self.multi_prompt,
        )

        resolution = ioconfig.highest_input_resolution

        patch_inputs = (
            np.array(self.clip_coordinates(mask_reader, patch_inputs, **resolution))
            if mask_reader is not None
            else patch_inputs
        )

        patch_outputs = patch_inputs.copy()

        # modify the shared space so that we can update worker info
        # without needing to re-create the worker. There should be no
        # race-condition because only the following enumerate loop
        # triggers the parallelism, and this portion is still in
        # sequential execution order
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=len(self._loader),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for i, batch_data in enumerate(self._loader):
            sample_datas, sample_infos = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            prompt_slice = slice(i * batch_size, (i + 1) * batch_size)

            points = point_coords[prompt_slice] if point_coords is not None else None
            boxes = box_coords[prompt_slice] if box_coords is not None else None

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)

            sample_outputs = self.model.infer_batch(
                self.model,
                sample_datas,
                point_coords=points,
                box_coords=boxes,
                device=self._device,
            )

            # repackage so that it's an N list, each contains
            # L x etc. output
            sample_outputs = [
                np.split(np.array(v), batch_size, axis=0) for v in sample_outputs
            ]
            sample_outputs = list(zip(*sample_outputs))

            # tensor to numpy, costly?
            sample_infos = sample_infos.numpy()
            sample_infos = np.split(sample_infos, batch_size, axis=0)

            sample_outputs = list(zip(sample_infos, sample_outputs))
            cum_output.extend(sample_outputs)
            pbar.update()

        pbar.close()

        self._process_predictions(
            cum_output,
            wsi_reader,
            ioconfig,
            save_path,
            cache_dir,
            mode,
        )

        # clean up the cache directories
        shutil.rmtree(cache_dir)

    @staticmethod
    def pad_ioconfig(
        **kw_ioconfig: dict,
    ) -> dict:
        """Assign None to missing keyword ioconfig info."""
        # Define the expected keys
        required_keys = [
            "patch_input_shape",
            "patch_output_shape",
            "stride_shape",
            "resolution",
            "units",
        ]

        # Fill in any missing keys with None
        for key in required_keys:
            kw_ioconfig.setdefault(key, None)
        return kw_ioconfig

    @staticmethod
    def _adjust_prompt_resolution(
        wsi_reader: WSIReader,
        coords: np.ndarray | None,
        resolution: Resolution,
        units: Units,
    ) -> np.ndarray | None:
        """Adjust the resolution of the prompt coordinates.

        This function scales the provided coordinates to the specified
        resolution and units. It is used to ensure that the coordinates
        are in the correct format for processing.

        Args:
            wsi_reader (WSIReader):
                A reader for the image where the predictions come from.
            resolution (Resolution):
                The resolution of the image.
            units (Units):
                The units of the image.
            coords (np.ndarray):
                Coordinates to adjust.
        """
        if coords is not None:
            coords = coords * (
                wsi_reader.slide_dimensions(resolution, units)[0]
                / wsi_reader.slide_dimensions(1.0, "baseline")[0]
            )
        return coords

    @staticmethod
    def get_coordinates(  # skipcq: PYL-W0221
        wsi_reader: WSIReader,
        ioconfig: IOSegmentorConfig,
        mode: str,
        point_coords: np.ndarray | None = None,
        box_coords: np.ndarray | None = None,
        *,
        multi_prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate patch tiling coordinates.

        ! Update this docstring to reflect the new API.

        By default, internally, it will call the
        `PatchExtractor.get_coordinates`. To use your own approach,
        either subclass to overwrite or directly assign your own
        function to this name. In either cases, the function must obey
        the API defined here.

        Args:
            wsi_reader (WSIReader):
                A reader for the image where the predictions come from.
            ioconfig (:class:`IOSegmentorConfig`):
                Configuration for input/output processing.
            mode (str):
                The mode of prediction. Can be either `tile` or `wsi`.
            point_coords (np.ndarray):
                Point coordinates for the current image as [x, y] pairs.
            box_coords (np.ndarray):
                Bounding box coordinates for the current image as
                [x1, y1, x2, y2] pairs.
            multi_prompt (bool):
                Whether to use multiple prompts simultaneously for segmentation.
                If false, the image will be processed for each prompt separately.

        Returns:
            tuple:
                List of patch inputs and outputs

                - :py:obj:`list` - patch_inputs:
                    A list of corrdinates in `[start_x, start_y, end_x,
                    end_y]` format indicating the read location of the
                    patch in the mother image.

                - point_coords:
                    A list of point coordinates for the current image
                    as `[x, y]` pairs.
                - box_coords:
                    A list of bounding box coordinates for the current
                    image as `[x1, y1, x2, y2]` pairs.

        Examples:
            >>> # API of function expected to overwrite `get_coordinates`
            >>> def func(image_shape, ioconfig):
            ...   patch_inputs = np.array([[0, 0, 256, 256]])
            ...   patch_outputs = np.array([[0, 0, 256, 256]])
            ...   return patch_inputs, patch_outputs
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.get_coordinates = func

        """
        resolution = ioconfig.highest_input_resolution
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)
        image_patch = np.array([0, 0, wsi_proc_shape[0], wsi_proc_shape[1]])

        point_coords = PromptSegmentor._adjust_prompt_resolution(
            wsi_reader, point_coords, **resolution
        )
        box_coords = PromptSegmentor._adjust_prompt_resolution(
            wsi_reader, box_coords, **resolution
        )

        if multi_prompt:
            patch_inputs = np.array([np.copy(image_patch)])
            point_coords = (
                np.array([point_coords]) if point_coords is not None else None
            )
            # Will only use the first box passed in
            box_coords = np.array([[box_coords[0]]]) if box_coords is not None else None
        else:
            num_points = len(point_coords) if point_coords is not None else 0
            num_boxes = len(box_coords) if box_coords is not None else 0

            patch_inputs = PromptSegmentor._extract_patches(
                wsi_reader=wsi_reader,
                ioconfig=ioconfig,
                point_coords=point_coords,
                box_coords=box_coords,
                mode=mode,
            )

            # Format coordinates by adding padding
            # Required for slicing when iterating over DataLoader
            point_coords = (
                ([[x] for x in point_coords] + [None] * num_boxes)
                if point_coords is not None
                else None
            )
            box_coords = (
                [None] * num_points + [[y] for y in box_coords]
                if box_coords is not None
                else None
            )

        return patch_inputs, point_coords, box_coords

    @staticmethod
    def _extract_patches(
        wsi_reader: WSIReader,
        ioconfig: IOSegmentorConfig,
        point_coords: np.ndarray | None = None,
        box_coords: np.ndarray | None = None,
        mode: str = "tile",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract patches from the WSI, given that it is WSI mode.

        Args:
            wsi_reader (WSIReader):
                A reader for the image where the predictions come from.
            ioconfig (:class:`IOSegmentorConfig`):
                Configuration for input/output processing.
            mode (str):
                The mode of prediction. Can be either `tile` or `wsi`.
            point_coords (np.ndarray):
                Point coordinates for the current image as [x, y] pairs.
            box_coords (np.ndarray):
                Bounding box coordinates for the current image as
                [x1, y1, x2, y2] pairs.

        Returns:
            tuple:
                List of patch inputs and outputs

                - :py:obj:`list` - patch_inputs:
                    A list of corrdinates in `[start_x, start_y, end_x,
                    end_y]` format indicating the read location of the
                    patch in the mother image.

        """
        resolution = ioconfig.highest_input_resolution
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)
        image_patch = np.array([0, 0, wsi_proc_shape[0], wsi_proc_shape[1]])
        num_points = len(point_coords) if point_coords is not None else 0
        num_boxes = len(box_coords) if box_coords is not None else 0
        num_patches = num_points + num_boxes

        if mode == "tile":
            patch_inputs = np.array([np.copy(image_patch) for _ in range(num_patches)])
        else:
            patch_extractor = PointsPatchExtractor(
                wsi_reader, point_coords, ioconfig.patch_input_shape, **resolution
            )
            patch_inputs = patch_extractor.get_coordinates(
                image_shape=wsi_proc_shape,
                patch_input_shape=ioconfig.patch_input_shape,
                stride_shape=ioconfig.stride_shape,
            )
        return patch_inputs

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Resolution | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
        """Indicates which coordinate is valid basing on the mask.

        To use your own approaches, either subclass to overwrite or
        directly assign your own function to this name. In either cases,
        the function must obey the API defined here.

        Args:
            mask_reader (:class:`.VirtualReader`):
                A virtual pyramidal reader of the mask related to the
                WSI from which we want to extract the patches.
            bounds (ndarray and np.int32):
                Coordinates to be checked via the `func`. They must be
                in the same resolution as requested `resolution` and
                `units`. The shape of `coordinates` is (N, K) where N is
                the number of coordinate sets and K is either 2 for
                centroids or 4 for bounding boxes. When using the
                default `func=None`, K should be 4, as we expect the
                `coordinates` to be bounding boxes in `[start_x,
                start_y, end_x, end_y]` format.
            resolution (Resolution):
                Resolution of the requested patch.
            units (Units):
                Units of the requested patch.

        Returns:
            :class:`numpy.ndarray`:
                List of flags to indicate which coordinate is valid.

        Examples:
            >>> # API of function expected to overwrite `filter_coordinates`
            >>> def func(reader, bounds, resolution, units):
            ...   # as example, only select first bound
            ...   return np.array([1, 0])
            >>> coords = [[0, 0, 256, 256], [128, 128, 384, 384]]
            >>> segmentor = SemanticSegmentor(model='unet')
            >>> segmentor.filter_coordinates = func

        """
        if not isinstance(mask_reader, VirtualWSIReader):
            msg = "`mask_reader` should be VirtualWSIReader."
            raise TypeError(msg)

        if not isinstance(bounds, np.ndarray) or not np.issubdtype(
            bounds.dtype,
            np.integer,
        ):
            msg = "`coordinates` should be ndarray of integer type."
            raise ValueError(msg)

        mask_real_shape = mask_reader.img.shape[:2]
        mask_resolution_shape = mask_reader.slide_dimensions(
            resolution=resolution,
            units=units,
        )[::-1]
        mask_real_shape = np.array(mask_real_shape)
        mask_resolution_shape = np.array(mask_resolution_shape)
        scale_factor = mask_real_shape / mask_resolution_shape
        scale_factor = scale_factor[0]  # what if ratio x != y

        # Get mask bounding box
        mask_bbox = PromptSegmentor.get_mask_bounds(mask_reader)
        scaled_bbox = np.ceil(mask_bbox / scale_factor).astype(np.int32)

        def sel_func(coord: np.ndarray) -> bool:
            """Accept coord if it is part of mask."""
            x, y = coord
            return (scaled_bbox[0] <= x <= scaled_bbox[2]) and (
                scaled_bbox[1] <= y <= scaled_bbox[3]
            )

        flags = [sel_func(bound) for bound in bounds]
        return np.array(flags)

    def _process_predictions(
        self,
        cum_batch_predictions: list,
        wsi_reader: WSIReader,
        ioconfig: IOSegmentorConfig,
        save_path: str,
        cache_dir: str,
        mode: str = "tile",
    ) -> None:
        """Define how the aggregated predictions are processed.

        This includes merging the prediction if necessary and also saving afterwards.
        Note that items within `cum_batch_predictions` will be consumed during
        the operation.

        Args:
            cum_batch_predictions (list):
                List of batch predictions. Each item within the list
                should be of (location, patch_predictions).
            wsi_reader (:class:`WSIReader`):
                A reader for the image where the predictions come from.
            ioconfig (:class: IOSegmentorConfig):
                Configuration for input/output processing.
            save_path (str):
                Root path to save current WSI predictions.
            cache_dir (str):
                Root path to cache current WSI data.
            mode (str):
                Type of input to process. Can either be `tile` or
                `wsi`.

        """
        wsi_shape = wsi_reader.slide_dimensions(1.0, "baseline")[::-1]

        if mode == "tile":
            self._prepare_save_dir(save_path)
            for i, (_, patch_prediction) in enumerate(cum_batch_predictions):
                mask_memmap, score_memmap = self._prepare_save_output(
                    Path(save_path) / f"{i}.raw.0.npy",
                    Path(save_path) / f"{i}.raw.1.npy",
                    tuple(wsi_shape),
                    (len(cum_batch_predictions),),
                )
                mask = patch_prediction[0]
                score = patch_prediction[1]

                # store the predictions
                mask_memmap[:, :] = mask[0]
                score_memmap[i] = score[0][0][0]

                mask_memmap.flush()
                score_memmap.flush()

        else:
            locations, predictions = list(zip(*cum_batch_predictions))
            # Nx4 (N x [tl_x, tl_y, br_x, br_y), denotes the location of
            # output patch this can exceed the image bound at the requested
            # resolution remove singleton due to split.
            locations = np.array([v[0] for v in locations])
            for index, output_resolution in enumerate(ioconfig.output_resolutions):
                # assume resolution index to be in the same order as L
                merged_resolution = ioconfig.highest_input_resolution
                merged_locations = locations
                if ioconfig.save_resolution is not None:
                    merged_resolution = ioconfig.save_resolution
                    output_shape = wsi_reader.slide_dimensions(**output_resolution)
                    merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                    fx = merged_shape[0] / output_shape[0]
                    merged_locations = np.ceil(merged_locations * fx).astype(np.int64)
                merged_shape = wsi_reader.slide_dimensions(**merged_resolution)
                # ! Need to find better way to extract prediction
                to_merge_predictions = predictions[0][0][0][0][0]
                sub_save_path = f"{save_path}.raw.{index}.npy"
                sub_count_path = f"{cache_dir}/count.{index}.npy"
                merged_output = {
                    "predictions": self.merge_prediction(
                        merged_shape[::-1],  # XY to YX
                        to_merge_predictions,
                        merged_locations,
                        save_path=sub_save_path,
                        cache_count_path=sub_count_path,
                    )
                }
                # Scale the merged output to the original WSI shape
                scale_factor = np.array(wsi_shape) / np.array(merged_shape[::-1])
                # Generate annotations
                dict_to_store_semantic_segmentor(
                    patch_output=merged_output,
                    scale_factor=scale_factor,
                    save_path=Path(f"{save_path}.{index}.db"),
                )

    @staticmethod
    def get_mask_bounds(
        mask_reader: VirtualWSIReader,
    ) -> np.ndarray:
        """Generate a bounding box for the mask."""
        if not isinstance(mask_reader, VirtualWSIReader):
            msg = "`mask_reader` should be VirtualWSIReader."
            raise TypeError(msg)

        ys, xs = np.where(mask_reader.img > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return np.array([x_min, y_min, x_max, y_max])

    @staticmethod
    def clip_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Resolution | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
        """Clip coordinates to the mask bounding box.

        This function scales the provided coordinates to the mask
        resolution and clips them to the mask bounding box.
        Only non-empty boxes are kept.

        Unlike the `filter_coordinates` function in the base class, this
        function clips patches to within the mask bounding box, and discards
        patches that are completely outside. Therefore, masks should
        be overestimates of the area of interest.

        Args:
            mask_reader (VirtualWSIReader):
                A reader for the image where the predictions come from.
            bounds (np.ndarray):
                The coordinates to filter.
            resolution (Resolution):
                The resolution of the image.
            units (Units):
                The units of the image.

        Returns:
            np.ndarray:
                The filtered coordinates.
        """
        if not isinstance(mask_reader, VirtualWSIReader):
            msg = "`mask_reader` should be VirtualWSIReader."
            raise TypeError(msg)

        if not isinstance(bounds, np.ndarray) or not np.issubdtype(
            bounds.dtype,
            np.integer,
        ):
            msg = "`coordinates` should be ndarray of integer type."
            raise ValueError(msg)

        mask_real_shape = mask_reader.img.shape[:2]
        mask_resolution_shape = mask_reader.slide_dimensions(
            resolution=resolution,
            units=units,
        )[::-1]

        mask_real_shape = np.array(mask_real_shape)
        mask_resolution_shape = np.array(mask_resolution_shape)
        scale_factor = mask_real_shape / mask_resolution_shape
        scale_factor = scale_factor[0]  # what if ratio x != y

        # Get mask bounding box
        mask_bbox = PromptSegmentor.get_mask_bounds(mask_reader)
        scaled_bbox = np.ceil(mask_bbox / scale_factor).astype(np.int32)

        # Clip to mask bounding box
        new_bounds = []
        for box in bounds:
            x1, y1, x2, y2 = box
            x1_new = max(x1, scaled_bbox[0])
            y1_new = max(y1, scaled_bbox[1])
            x2_new = min(x2, scaled_bbox[2])
            y2_new = min(y2, scaled_bbox[3])

            # Only keep if box is non-empty
            if x1_new < x2_new and y1_new < y2_new:
                new_bounds.append([x1_new, y1_new, x2_new, y2_new])

        return np.array(new_bounds, dtype=np.int32)

    def _predict_wsi_handle_exception(  # skipcq: PYL-W0221
        self: PromptSegmentor,
        imgs: list,
        wsi_idx: int,
        img_path: str | Path,
        mode: str,
        ioconfig: IOSegmentorConfig,
        point_coords: list[IntPair],
        box_coords: IntBounds,
        save_dir: str | Path,
        *,
        crash_on_exception: bool,
    ) -> None:
        """Predict on multiple WSIs.

        Args:
            imgs (list, ndarray):
                List of image file paths to process.
            wsi_idx (int):
                index of current WSI being processed.
            img_path(str or Path):
                Path to current image.
            mode (str):
                Type of input to process. Can either be `tile` or
                `wsi`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object defines information about input and output
                placement of patches.
            point_coords (list):
                List of point coordinates.
            box_coords (IntBounds):
                Bounding box coordinates in [x1, y1, x2, y2] form.
            save_dir (str, Path):
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

        """
        try:
            wsi_save_path = save_dir / f"{wsi_idx}"
            self._predict_one_wsi(
                wsi_idx, ioconfig, point_coords, box_coords, str(wsi_save_path), mode
            )

            # Do not use dict with file name as key, because it can be
            # overwritten. It may be user intention to provide files with a
            # same name multiple times (maybe they have different root path)
            self._outputs.append([str(img_path), str(wsi_save_path)])

            # ? will this corrupt old version if control + c midway?
            map_file_path = save_dir / "file_map.dat"
            # backup old version first
            if Path.exists(map_file_path):
                old_map_file_path = save_dir / "file_map_old.dat"
                shutil.copy(map_file_path, old_map_file_path)
            joblib.dump(self._outputs, map_file_path)

            # verbose mode, error by passing ?
            logging.info("Finish: %d", wsi_idx / len(imgs))
            logging.info("--Input: %s", str(img_path))
            logging.info("--Output: %s", str(wsi_save_path))
        # prevent deep source check because this is bypass and
        # delegating error message
        except Exception as err:  # skipcq: PYL-W0703
            wsi_save_path = save_dir.joinpath(f"{wsi_idx}")
            if crash_on_exception:
                raise err  # noqa: TRY201
            logging.exception("Crashed on %s", wsi_save_path)

    @staticmethod
    def _prepare_save_output(
        mask_path: str | Path,
        score_path: str | Path,
        mask_shape: tuple[int, ...],
        scores_shape: tuple[int, ...],
    ) -> tuple:
        """Prepares for saving the cached output."""
        # Check if save path exists

        if mask_path is not None and score_path is not None:
            mask_path = Path(mask_path)
            score_path = Path(score_path)
            mask_memmap = np.lib.format.open_memmap(
                mask_path,
                mode="w+",
                shape=mask_shape,
                dtype=np.uint8,
            )
            score_memmap = np.lib.format.open_memmap(
                score_path,
                mode="w+",
                shape=scores_shape,
                dtype=np.float32,
            )
        return mask_memmap, score_memmap

    @staticmethod
    def calc_mpp(area_dims: IntPair, base_mpp: float, fixed_size: int = 1500) -> float:
        """Calculates the microns per pixel for a fixed area of an image.

        Args:
            area_dims (tuple):
                Dimensions of the area to be scaled.
            base_mpp (float):
                Microns per pixel of the base image.
            fixed_size (int):
                Fixed size of the area.

        Returns:
            float:
                Microns per pixel required to scale the area to a fixed size.
        """
        scale = max(area_dims) / fixed_size if max(area_dims) > fixed_size else 1.0
        return base_mpp * scale
