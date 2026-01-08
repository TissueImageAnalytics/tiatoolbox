"""This module enables multi-task segmentor."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Unpack

import joblib
import numpy as np
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

from tiatoolbox.models.engine.nucleus_instance_segmentor import (
    _process_instance_predictions,
)

from .semantic_segmentor import SemanticSegmentor, SemanticSegmentorRunParams

if TYPE_CHECKING:  # pragma: no cover
    import os
    from collections.abc import Callable
    from pathlib import Path

    from tiatoolbox.annotation import AnnotationStore
    from tiatoolbox.models.models_abc import ModelABC
    from tiatoolbox.type_hints import IntBounds, IntPair, Resolution, Units
    from tiatoolbox.wsicore import WSIReader

    from .io_config import IOInstanceSegmentorConfig, IOSegmentorConfig


# Python is yet to be able to natively pickle Object method/static method.
# Only top-level function is passable to multi-processing as caller.
# May need 3rd party libraries to use method/static method otherwise.
def _process_tile_predictions(  # skipcq: PY-R1000
    ioconfig: IOSegmentorConfig,
    tile_bounds: IntBounds,
    tile_flag: list,
    tile_mode: int,
    tile_output: list,
    # this would be replaced by annotation store
    # in the future
    ref_inst_dict: dict,
    postproc: Callable,
    merge_predictions: Callable,
    model_name: str,
) -> tuple:
    """Process Tile Predictions.

    Function to merge new tile prediction with existing prediction,
    using the output from each task.

    Args:
        ioconfig (:class:`IOSegmentorConfig`): Object defines information
            about input and output placement of patches.
        tile_bounds (:class:`numpy.array`): Boundary of the current tile, defined as
            (top_left_x, top_left_y, bottom_x, bottom_y).
        tile_flag (list): A list of flag to indicate if instances within
            an area extended from each side (by `ioconfig.margin`) of
            the tile should be replaced by those within the same spatial
            region in the accumulated output this run. The format is
            [top, bottom, left, right], 1 indicates removal while 0 is not.
            For example, [1, 1, 0, 0] denotes replacing top and bottom instances
            within `ref_inst_dict` with new ones after this processing.
        tile_mode (int): A flag to indicate the type of this tile. There
            are 4 flags:
            - 0: A tile from tile grid without any overlapping, it is not
                an overlapping tile from tile generation. The predicted
                instances are immediately added to accumulated output.
            - 1: Vertical tile strip that stands between two normal tiles
                (flag 0). It has the the same height as normal tile but
                less width (hence vertical strip).
            - 2: Horizontal tile strip that stands between two normal tiles
                (flag 0). It has the the same width as normal tile but
                less height (hence horizontal strip).
            - 3: tile strip stands at the cross section of four normal tiles
                (flag 0).
        tile_output (list): A list of patch predictions, that lie within this
            tile, to be merged and processed.
        ref_inst_dict (dict): Dictionary contains accumulated output. The
            expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        postproc (callable): Function to post-process the raw assembled tile.
        merge_predictions (callable): Function to merge the `tile_output` into
            raw tile prediction.
        model_name (string): Name of the existing models support by tiatoolbox
          for processing the data. Refer to [URL] for details.

    Returns:
        new_inst_dict (dict): A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list): List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.
        sem_maps (list): List of semantic segmentation maps.
        tile_bounds (:class:`numpy.array`): Boundary of the current tile, defined as
            (top_left_x, top_left_y, bottom_x, bottom_y).

    """
    locations, predictions = list(zip(*tile_output, strict=False))

    # convert from WSI space to tile space
    tile_tl = tile_bounds[:2]
    tile_br = tile_bounds[2:]
    locations = [np.reshape(loc, (2, -1)) for loc in locations]
    locations_in_tile = [loc - tile_tl[None] for loc in locations]
    locations_in_tile = [loc.flatten() for loc in locations_in_tile]
    locations_in_tile = np.array(locations_in_tile)

    tile_shape = tile_br - tile_tl  # in width height

    # as the placement output is calculated wrt highest possible resolution
    # within input, the output will need to re-calibrate if it is at different
    # resolution than the input
    ioconfig = ioconfig.to_baseline()
    fx_list = [v["resolution"] for v in ioconfig.output_resolutions]

    head_raws = []
    for idx, fx in enumerate(fx_list):
        head_tile_shape = np.ceil(tile_shape * fx).astype(np.int32)
        head_locations = np.ceil(locations_in_tile * fx).astype(np.int32)
        head_predictions = [v[idx][0] for v in predictions]
        head_raw = merge_predictions(
            head_tile_shape[::-1],
            head_predictions,
            head_locations,
        )
        head_raws.append(head_raw)

    if "hovernetplus" in model_name:
        _, inst_dict, layer_map, _ = postproc(head_raws)
        out_dicts = [inst_dict, layer_map]
    elif "hovernet" in model_name:
        _, inst_dict = postproc(head_raws)
        out_dicts = [inst_dict]
    else:
        out_dicts = postproc(head_raws)

    inst_dicts = [out for out in out_dicts if isinstance(out, dict)]
    sem_maps = [out for out in out_dicts if isinstance(out, np.ndarray)]
    # Some output maps may not be aggregated into a single map - combine these
    sem_maps = [
        np.argmax(s, axis=-1) if s.ndim == 3 else s  # noqa: PLR2004
        for s in sem_maps
    ]

    new_inst_dicts, remove_insts_in_origs = [], []
    for inst_id, inst_dict in enumerate(inst_dicts):
        new_inst_dict, remove_insts_in_orig = _process_instance_predictions(
            inst_dict,
            ioconfig,
            tile_shape,
            tile_flag,
            tile_mode,
            tile_tl,
            ref_inst_dict[inst_id],
        )
        new_inst_dicts.append(new_inst_dict)
        remove_insts_in_origs.append(remove_insts_in_orig)

    return new_inst_dicts, remove_insts_in_origs, sem_maps, tile_bounds


class MultiTaskSegmentor(SemanticSegmentor):
    """An engine specifically designed to handle tiles or WSIs inference.

    Note, if `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments. Each WSI's instance
    predictions (e.g. nuclear instances) will be store under a `.dat` file and
    the semantic segmentation predictions will be stored in a `.npy` file. The
    `.dat` files contains a dictionary of form:

    .. code-block:: yaml

        inst_uid:
            # top left and bottom right of bounding box
            box: (start_x, start_y, end_x, end_y)
            # centroid coordinates
            centroid: (x, y)
            # array/list of points
            contour: [(x1, y1), (x2, y2), ...]
            # the type of nuclei
            type: int
            # the probabilities of being this nuclei type
            prob: float

    Args:
        model (nn.Module): Use externally defined PyTorch model for prediction with.
          weights already loaded. Default is `None`. If provided,
          `pretrained_model` argument is ignored.
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. Refer to [URL] for details.
          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights
          via the `pretrained_weights` argument. Argument is case insensitive.
        pretrained_weights (str): Path to the weight of the corresponding
          `pretrained_model`.
        batch_size (int) : Number of images fed into the model each time.
        num_loader_workers (int) : Number of workers to load the data.
          Take note that they will also perform preprocessing.
        num_postproc_workers (int) : Number of workers to post-process
          predictions.
        verbose (bool): Whether to output logging information.
        dataset_class (obj): Dataset class to be used instead of default.
        auto_generate_mask (bool): To automatically generate tile/WSI tissue mask
          if is not provided.
        output_types (list): Ordered list describing what sort of segmentation the
            output from the model postproc gives for a two-task model this may be:
            ['instance', 'semantic']

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = MultiTaskSegmentor(
        ...     model='hovernetplus-oed',
        ...     output_type=['instance', 'semantic'],
        ... )
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
        >>> # Each output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.0.dat', 'output/0.1.npy'
        >>> # Here, the second integer represents the task number
        >>> # e.g. between 0 or 1 for a two task model

    """

    def __init__(
        self: MultiTaskSegmentor,
        model: str | ModelABC,
        batch_size: int = 8,
        num_workers: int = 0,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize :class:`NucleusInstanceSegmentor`."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            weights=weights,
            device=device,
            verbose=verbose,
        )

    def _predict_one_wsi(
        self: MultiTaskSegmentor,
        wsi_idx: int,
        ioconfig: IOInstanceSegmentorConfig,
        save_path: str,
        mode: str,
    ) -> None:
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int):
                Index of the tile/wsi to be processed within `self`.
            ioconfig (IOInstanceSegmentorConfig):
                Object which defines I/O placement
                during inference and when assembling back to full tile/wsi.
            save_path (str):
                Location to save output prediction as well as possible
                intermediate results.
            mode (str):
                `tile` or `wsi` to indicate run mode.

        """
        cache_dir = f"{self._cache_dir}/"
        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path,
            mask_path,
            mode,
            auto_get_mask=self.auto_generate_mask,
        )

        # assume ioconfig has already been converted to `baseline` for `tile` mode
        resolution = ioconfig.highest_input_resolution
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch placement
        # this is in XY
        (patch_inputs, patch_outputs) = self.get_coordinates(wsi_proc_shape, ioconfig)
        if mask_reader is not None:
            sel = self.filter_coordinates(mask_reader, patch_outputs, **resolution)
            patch_outputs = patch_outputs[sel]
            patch_inputs = patch_inputs[sel]

        # assume to be in [top_left_x, top_left_y, bot_right_x, bot_right_y]
        geometries = [shapely_box(*bounds) for bounds in patch_outputs]
        spatial_indexer = STRtree(geometries)

        # * retrieve tile placement and tile info flag
        # tile shape will always be corrected to be multiple of output
        tile_info_sets = self._get_tile_info(wsi_proc_shape, ioconfig)

        # ! running order of each set matters !
        self._futures = []

        indices_sem = [i for i, x in enumerate(self.output_types) if x == "semantic"]

        for s_id in range(len(indices_sem)):
            shape = tuple(map(int, np.fliplr([wsi_proc_shape])[0]))
            self.wsi_layers.append(
                np.lib.format.open_memmap(
                    f"{cache_dir}/{s_id}.npy",
                    mode="w+",
                    shape=shape,
                    dtype=np.uint8,
                ),
            )
            self.wsi_layers[s_id][:] = 0

        indices_inst = [i for i, x in enumerate(self.output_types) if x == "instance"]

        if not self._wsi_inst_info:  # pragma: no cover
            self._wsi_inst_info = []
        self._wsi_inst_info.extend({} for _ in indices_inst)

        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            for tile_idx, tile_bounds in enumerate(set_bounds):
                tile_flag = set_flags[tile_idx]

                # select any patches that have their output
                # within the current tile
                sel_box = shapely_box(*tile_bounds)
                sel_indices = list(spatial_indexer.query(sel_box))

                tile_patch_inputs = patch_inputs[sel_indices]
                tile_patch_outputs = patch_outputs[sel_indices]
                self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)

                tile_infer_output = self._infer_once()

                self._process_tile_predictions(
                    ioconfig,
                    tile_bounds,
                    tile_flag,
                    set_idx,
                    tile_infer_output,
                )
            self._merge_post_process_results()

        # Maybe change to store semantic annotations as contours in .dat file...
        for i_id, inst_idx in enumerate(indices_inst):
            joblib.dump(self._wsi_inst_info[i_id], f"{save_path}.{inst_idx}.dat")
        self._wsi_inst_info = []  # clean up

        for s_id, sem_idx in enumerate(indices_sem):
            shutil.copyfile(f"{cache_dir}/{s_id}.npy", f"{save_path}.{sem_idx}.npy")
            # may need to chain it with parents

    def _process_tile_predictions(
        self: MultiTaskSegmentor,
        ioconfig: IOSegmentorConfig,
        tile_bounds: IntBounds,
        tile_flag: list,
        tile_mode: int,
        tile_output: list,
    ) -> None:
        """Function to dispatch parallel post processing."""
        args = [
            ioconfig,
            tile_bounds,
            tile_flag,
            tile_mode,
            tile_output,
            self._wsi_inst_info,
            self.model.postproc_func,
            self.merge_prediction,
            self.pretrained_model,
        ]
        if self._postproc_workers is not None:
            future = self._postproc_workers.submit(_process_tile_predictions, *args)
        else:
            future = _process_tile_predictions(*args)
        self._futures.append(future)

    def _merge_post_process_results(self: MultiTaskSegmentor) -> None:
        """Helper to aggregate results from parallel workers."""

        def callback(
            new_inst_dicts: dict,
            remove_uuid_lists: list,
            tiles: dict,
            bounds: IntBounds,
        ) -> None:
            """Helper to aggregate worker's results."""
            # ! DEPRECATION:
            # !     will be deprecated upon finalization of SQL annotation store
            for inst_id, new_inst_dict in enumerate(new_inst_dicts):
                self._wsi_inst_info[inst_id].update(new_inst_dict)
                for inst_uuid in remove_uuid_lists[inst_id]:
                    self._wsi_inst_info[inst_id].pop(inst_uuid, None)

            x_start, y_start, x_end, y_end = bounds
            for sem_id, tile in enumerate(tiles):
                max_h, max_w = self.wsi_layers[sem_id].shape
                x_end, y_end = min(x_end, max_w), min(y_end, max_h)
                tile_ = tile[0 : y_end - y_start, 0 : x_end - x_start]
                self.wsi_layers[sem_id][y_start:y_end, x_start:x_end] = tile_
            # !

        for future in self._futures:
            #  not actually future but the results
            if self._postproc_workers is None:
                callback(*future)
                continue
            # some errors happen, log it and propagate exception
            # ! this will lead to discard a whole bunch of
            # ! inferred tiles within this current WSI
            if future.exception() is not None:
                raise future.exception()

            # aggregate the result via callback
            # manually call the callback rather than
            # attaching it when receiving/creating the future
            callback(*future.result())

    def run(
        self: MultiTaskSegmentor,
        images: list[os.PathLike | Path | WSIReader] | np.ndarray,
        *,
        masks: list[os.PathLike | Path] | np.ndarray | None = None,
        input_resolutions: list[dict[Units, Resolution]] | None = None,
        patch_input_shape: IntPair | None = None,
        ioconfig: IOSegmentorConfig | None = None,
        patch_mode: bool = True,
        save_dir: os.PathLike | Path | None = None,
        overwrite: bool = False,
        output_type: str = "dict",
        **kwargs: Unpack[SemanticSegmentorRunParams],
    ) -> AnnotationStore | Path | str | dict | list[Path]:
        """Run the semantic segmentation engine on input images.

        This method orchestrates the full inference pipeline, including preprocessing,
        model inference, post-processing, and saving results. It supports both
        patch-level and whole slide image (WSI) modes.

        Args:
            images (list[PathLike | WSIReader] | np.ndarray):
                Input images or patches. Can be a list of file paths, WSIReader objects,
                or a NumPy array of image patches.
            masks (list[PathLike] | np.ndarray | None):
                Optional masks for WSI processing. Only used when `patch_mode` is False.
            input_resolutions (list[dict[Units, Resolution]] | None):
                Resolution settings for input heads. Supported units are `level`,
                `power`, and `mpp`. Keys should be "units" and "resolution", e.g.,
                [{"units": "mpp", "resolution": 0.25}]. See :class:`WSIReader` for
                details.
            patch_input_shape (IntPair | None):
                Shape of input patches (height, width), requested at read
                resolution. Must be positive.
            ioconfig (IOSegmentorConfig | None):
                IO configuration for patch extraction and resolution.
            patch_mode (bool):
                Whether to treat input as patches (`True`) or WSIs (`False`). Default
                is True.
            save_dir (PathLike | None):
                Directory to save output files. Required for WSI mode.
            overwrite (bool):
                Whether to overwrite existing output files. Default is False.
            output_type (str):
                Desired output format: "dict", "zarr", or "annotationstore". Default
                is "dict".
            **kwargs (SemanticSegmentorRunParams):
                Additional runtime parameters to configure segmentation.

                Optional Keys:
                    auto_get_mask (bool):
                        Automatically generate segmentation masks using
                        `wsireader.tissue_mask()` during processing.
                    batch_size (int):
                        Number of image patches per forward pass.
                    class_dict (dict):
                        Mapping of classification outputs to class names.
                    device (str):
                        Device to run the model on (e.g., "cpu", "cuda").
                    labels (list):
                        Optional labels for input images. Only a single label per image
                        is supported.
                    memory_threshold (int):
                        Memory usage threshold (percentage) to trigger caching behavior.
                    num_workers (int):
                        Number of workers for DataLoader and post-processing.
                    output_file (str):
                        Filename for saving output (e.g., ".zarr" or ".db").
                    output_resolutions (Resolution):
                        Resolution used for writing output predictions.
                    patch_output_shape (tuple[int, int]):
                        Shape of output patches (height, width).
                    return_labels (bool):
                        Whether to return labels with predictions.
                    return_probabilities (bool):
                        Whether to return per-class probabilities.
                    scale_factor (tuple[float, float]):
                        Scale factor for annotations (model_mpp / slide_mpp).
                        Used to convert coordinates to baseline resolution.
                    stride_shape (tuple[int, int]):
                        Stride used during WSI processing.
                        Defaults to `patch_input_shape` if not provided.
                    verbose (bool):
                        Whether to enable verbose logging.

        Returns:
            AnnotationStore | Path | str | dict | list[Path]:
                - If `patch_mode` is True: returns predictions or path to saved output.
                - If `patch_mode` is False: returns a dictionary mapping each WSI
                  to its output path.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> image_patches = [np.ndarray, np.ndarray]
            >>> segmentor = SemanticSegmentor(model="fcn-tissue_mask")
            >>> output = segmentor.run(image_patches, patch_mode=True)
            >>> output
            ... "/path/to/Output.db"

            >>> output = segmentor.run(
            ...     image_patches,
            ...     patch_mode=True,
            ...     output_type="zarr"
            ... )
            >>> output
            ... "/path/to/Output.zarr"

            >>> output = segmentor.run(wsis, patch_mode=False)
            >>> output.keys()
            ... ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            ... "/path/to/wsi1.db"

        """
        return super().run(
            images=images,
            masks=masks,
            input_resolutions=input_resolutions,
            patch_input_shape=patch_input_shape,
            ioconfig=ioconfig,
            patch_mode=patch_mode,
            save_dir=save_dir,
            overwrite=overwrite,
            output_type=output_type,
            **kwargs,
        )
