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
from shapely import Polygon

from tiatoolbox import logger
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
    _prepare_save_output,
)
from tiatoolbox.models.models_abc import model_to
from tiatoolbox.tools.patchextraction import PointsPatchExtractor
from tiatoolbox.type_hints import Resolution, Units
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import Callable, IntBounds, IntPair


class PromptSegmentor(SemanticSegmentor):
    """Model designed for general segmentation of WSIs.

    Uses the SAM2 model architecture.

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
        batch_size: int = 8,
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
        self.bounds = None
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
        multi_prompt: bool = True,
        save_dir: str | Path | None = None,
        device: str = "cpu",
        *,
        crash_on_exception: bool = False,
    ) -> list[tuple[Path, Path, Path]]:
        """Predict on a WSI using prompts.

        Args:
            # TODO: Improve docstring
            file_name (str):
                Path to WSI file.

            device (str):
                Device to run inference on.
            save_path (str):
                Location to save output prediction.
            bounds (tuple):
                Bounds for the region of interest.
            multi_prompt (bool):
                Whether to perform inference on individual
                prompts or multiple prompts.

        Returns:
            list:
                List of paths to the saved output prediction.
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
        self, cum_batch_predictions, wsi_reader, ioconfig, save_path, cache_dir
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
        mask_memmap, score_memmap = _prepare_save_output(
            save_path, cache_dir, cum_batch_predictions.shape
        )

        # TODO: Add processing and saving output

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Resolution | None = None,
        units: Units | None = None,
    ) -> np.ndarray:
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
                List of inputs to process. When using `"patch"` mode,
                the input must be either a list of images, a list of
                image file paths or a numpy array of an image list. When
                using `"tile"` or `"wsi"` mode, the input must be a list
                of file paths.
            wsi_idx (int):
                index of current WSI being processed.
            img_path(str or Path):
                Path to current image.
            mode (str):
                Type of input to process. Choose from either `tile` or
                `wsi`.
            ioconfig (:class:`IOSegmentorConfig`):
                Object defines information about input and output
                placement of patches. When provided,
                `patch_input_shape`, `patch_output_shape`,
                `stride_shape`, `resolution`, and `units` arguments are
                ignored. Otherwise, those arguments will be internally
                converted to a :class:`IOSegmentorConfig` object.
            save_dir (str or Path):
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

    # TODO: Move to utils
    @staticmethod
    def to_annotation(
        mask_path: str | Path,
        score_path: str | Path,
        save_filename: str | Path | None = None,
    ) -> Path:
        """Converts the prediction output to annotation format."""
        masks = np.load(mask_path)
        scores = np.load(score_path)

        # Define annotation store path
        store_path = save_filename.with_suffix(".db")
        store = SQLiteStore()

        def mask_to_polygons(mask):
            """Extract polygons from a binary mask using OpenCV."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Avoid single-point contours
            polygons = [Polygon(c.squeeze()) for c in contours if len(c) > 2]
            return polygons

        for i, mask in enumerate(masks):
            polygons = mask_to_polygons(mask)
            # Add extracted polygons to the annotation store
            props = {"score": f"{scores[i]}", "type": f"Mask {i + 1}"}
            for poly in polygons:
                annotation = Annotation(geometry=poly, properties=props)
                store.append(annotation)

        store.create_index("id", '"id"')

        store.commit()
        store.dump(store_path)
        store.close()
        return store_path
