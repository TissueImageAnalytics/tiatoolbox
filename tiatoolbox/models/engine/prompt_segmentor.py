"""Model designed for general segmentation of WSIs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
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
        model: SAM = None,
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

    def _bound_prompts(
        self,
        bounds: IntBounds | None,
        point_coords: list[IntPair] | None = None,
        box_coords: list[IntBounds] | None = None,
    ) -> tuple[list[IntPair], list[IntBounds]]:
        """Bound the prompts to the region of interest."""
        if bounds is not None:
            if point_coords is not None:
                point_coords = (
                    np.array(point_coords) - np.array(bounds[:2])
                ) * np.array(self.scale_factor)

            if box_coords is not None:
                new_box_coords = []
                for left, top, right, bottom in box_coords:
                    new_left, new_top = (
                        np.array([left, top]) - np.array(bounds[:2])
                    ) * np.array(self.scale_factor)
                    new_right, new_bottom = (
                        np.array([right, bottom]) - np.array(bounds[:2])
                    ) * np.array(self.scale_factor)

                    new_box_coords.append([new_left, new_top, new_right, new_bottom])
                box_coords = np.array(new_box_coords)
        return point_coords, box_coords

    def _unbound_masks(self, masks: list[np.ndarray], bounds: IntBounds) -> np.ndarray:
        """Unbound the masks to the original image size."""
        new_masks = []

        for mask in masks:
            new_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])

            # Resizes the mask into the box at base resolution
            resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

            new_mask = np.zeros(np.array(self.slide_dims)[::-1], dtype=np.uint8)

            # Stores mask into base resolution whole image
            new_mask[bounds[1] : bounds[3], bounds[0] : bounds[2]] = resized_mask
            new_masks.append(new_mask)
        return np.array(new_masks, dtype=np.uint8)

    def predict(
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
    ) -> list[tuple[Path, Path, Path]]:
        """Predict on a list of WSIs using prompts.

        Args:
            imgs (list, ndarray):
                A list of paths to the input WSIs.
            masks (list):
                A list of masks corresponding to the input WSIs.
                Used to filter the coordinates of patches for inference.
            mode (str):
                The mode of prediction. Can be either `tile` or `wsi`.
                Tile mode processes individual image patches at baseline resolution,
                while WSI mode processes whole-slide images.
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

        Returns:
            output_paths(list[tuple[Path, Path, Path]]):
                A list of tuples containing the input image path and the corresponding
                output paths for the predictions.
                Each tuple is of the form (input_path, mask_path, score_path).

        Examples:
            >>> segmentor = PromptSegmentor(model=model)
            >>> imgs = ["path/to/image1", "path/to/image2"]
            >>> masks = ["path/to/mask1", "path/to/mask2"]
            >>> point_coords = [[[100, 200]], [[150, 250]]]
            >>> box_coords = [[[50, 50, 150, 150]], [[100, 100, 200, 200]]]
            >>> output_paths = segmentor.predict(
                    imgs,
                    masks=masks,
                    mode="tile",
                    point_coords=point_coords,
                    box_coords=box_coords,
                    save_dir="output_dir",
                )

        """
        save_dir, self._cache_dir = self._prepare_save_dir(save_dir=save_dir)

        ioconfig = self._update_ioconfig(ioconfig, mode)

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

    def _predict_one_wsi(
        self,
        wsi_idx: int,
        ioconfig: IOSegmentorConfig,
        point_coords: list[IntPair] | None = None,
        box_coords: list[IntBounds] | None = None,
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

        if mode == "tile":
            resolution = ioconfig.to_baseline().highest_input_resolution
            wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)
            image_patch = np.array([0, 0, wsi_proc_shape[0], wsi_proc_shape[1]])
            num_prompts = len(point_coords) + len(box_coords)

            if self.multi_prompt:
                patch_inputs = np.array([np.copy(image_patch)])
            else:
                patch_inputs = np.array(
                    [np.copy(image_patch) for _ in range(num_prompts)]
                )

        elif mode == "wsi":
            # Takes the second resolution from ioconfig
            resolution = ioconfig.input_resolutions[1]
            wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)
            if self.multi_prompt:
                patch_inputs = np.array([[0, 0, wsi_proc_shape[0], wsi_proc_shape[1]]])
            else:
                patch_extractor = PointsPatchExtractor(
                    wsi_reader, point_coords, ioconfig.patch_input_shape, **resolution
                )
                patch_inputs, patch_outputs = patch_extractor.get_coordinates(
                    patch_output_shape=ioconfig.patch_output_shape,
                    image_shape=wsi_proc_shape,
                    patch_input_shape=ioconfig.patch_input_shape,
                    stride_shape=ioconfig.stride_shape,
                )

        if mask_reader is not None:
            patch_inputs = self.filter_coordinates(
                mask_reader, patch_inputs, **resolution
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
                self._model,
                sample_datas,
                point_coords=points,
                box_coords=boxes,
                device=self._device,
            )
            # repackage so that it's an N list, each contains
            # L x etc. output
            sample_outputs = [np.split(v, batch_size, axis=0) for v in sample_outputs]
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
        )

        # clean up the cache directories
        shutil.rmtree(cache_dir)

    def _process_predictions(
        self,
        cum_batch_predictions: list,
        wsi_reader: WSIReader,
        ioconfig: IOSegmentorConfig,
        save_path: str,
        cache_dir: str,
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
            ioconfig (:class:`IOSegmentorConfig`):
                A configuration object contains input and output
                information.
            save_path (str):
                Root path to save current WSI predictions.
            cache_dir (str):
                Root path to cache current WSI data.

        """
        mask_memmap, score_memmap = self._prepare_save_output(
            save_path,
            wsi_reader.slide_dimensions(1.0, "baseline"),
            cum_batch_predictions.shape[0],
        )

        if self.multi_prompt:
            # merge the predictions
            for _, (location, mask, score) in enumerate(cum_batch_predictions):
                x1, y1, x2, y2 = location
                mask_memmap[y1:y2, x1:x2] = mask[0]
                score_memmap[y1:y2, x1:x2] = score[0]

            # save the predictions
            mask_memmap.flush()
            score_memmap.flush()

            # save the cache
            np.save(cache_dir / "0.npy", mask_memmap)
            np.save(cache_dir / "1.npy", score_memmap)
            # save the mask and score
            mask_path = Path(save_path) / "0.npy"
            score_path = Path(save_path) / "1.npy"
            np.save(mask_path, mask_memmap)
            np.save(score_path, score_memmap)

        else:
            # merge the predictions
            for _, (location, mask, score) in enumerate(cum_batch_predictions):
                x1, y1, x2, y2 = location
                mask_memmap[y1:y2, x1:x2] = mask[0]
                score_memmap[y1:y2, x1:x2] = score[0]

            if ioconfig.mode == "tile":  # incomplete
                # save the predictions
                mask_memmap.flush()
                score_memmap.flush()

                # save the cache
                np.save(cache_dir / "0.npy", mask_memmap)
                np.save(cache_dir / "1.npy", score_memmap)

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Resolution | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
        """Filter coordinates to the mask bounding box.

        This function filters the coordinates to the bounding box of the mask
        reader. It scales the coordinates to the mask resolution and clips
        them to the mask bounding box. Only non-empty boxes are kept.

        Unlike the `filter_coordinates` function in the base class, this
        function does not discard patches that are outside the mask
        bounding box. Instead, it clips them to within the mask bounding box,
        unless the patch is completely outside.

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
        ys, xs = np.where(mask_reader.img > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        mask_bbox = np.array([x_min, y_min, x_max, y_max])

        # Scale bounds to mask resolution
        scaled_bounds = np.ceil(bounds * scale_factor).astype(np.int32)

        # Clip to mask bounding box
        new_bounds = []
        for box in scaled_bounds:
            x1, y1, x2, y2 = box
            x1_new = max(x1, mask_bbox[0])
            y1_new = max(y1, mask_bbox[1])
            x2_new = min(x2, mask_bbox[2])
            y2_new = min(y2, mask_bbox[3])

            # Only keep if box is non-empty
            if x1_new < x2_new and y1_new < y2_new:
                new_bounds.append([x1_new, y1_new, x2_new, y2_new])

        return np.array(new_bounds, dtype=np.int32)

    def _predict_wsi_handle_exception(
        self: PromptSegmentor,
        imgs: list,
        wsi_idx: int,
        img_path: str | Path,
        mode: str,
        ioconfig: IOSegmentorConfig,
        point_coords: list[IntPair],
        box_coords: list[IntBounds],
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
                List of point coordinates for each image.
            box_coords (list):
                List of bounding box coordinates for each image.
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

    def _prepare_save_output(
        self,
        save_path: str | Path,
        mask_shape: tuple[int, ...],
        scores_shape: tuple[int, ...],
    ) -> tuple:
        """Prepares for saving the cached output."""
        if save_path is not None:
            save_path = Path(save_path)
            mask_memmap = np.lib.format.open_memmap(
                save_path / "0.npy",
                mode="w+",
                shape=mask_shape,
                dtype=np.uint8,
            )
            score_memmap = np.lib.format.open_memmap(
                save_path / "1.npy",
                mode="w+",
                shape=scores_shape,
                dtype=np.float32,
            )
        return mask_memmap, score_memmap
