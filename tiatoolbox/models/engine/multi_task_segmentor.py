# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module enables multi-task segmentors."""

import shutil
import uuid
from typing import Callable, List

# replace with the sql database once the PR in place
import joblib
import numpy as np
import torch
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    WSIStreamDataset,
)


def _process_instance_predictions(
    inst_dict,
    ioconfig,
    tile_shape,
    tile_flag,
    tile_mode,
    tile_tl,
    ref_inst_dict,
):
    """Function to merge new tile prediction with existing prediction.

    Args:
        isnt_dict (dict): Dictionary containing instance information.
        ioconfig (:class:`IOSegmentorConfig`): Object defines information
            about input and output placement of patches.
        tile_shape (list): A list of the tile shape.
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
        tile_tl (tuple): Top left coordinates of the current tile.
        ref_inst_dict (dict): Dictionary contains accumulated output. The
            expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.

    Returns:
        new_inst_dict (dict): A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list): List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.

    """
    # should be rare, no nuclei detected in input images
    if len(inst_dict) == 0:
        return {}, []

    # !
    m = ioconfig.margin
    w, h = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    # An auxiliary dictionary to actually query the index within the source list
    index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
    tile_rtree = STRtree(geometries)
    # !

    # create margin bounding box, ordering should match with
    # created tile info flag (top, bottom, left, right)
    boundary_lines = [
        shapely_box(0, 0, w, 1),  # noqa top egde
        shapely_box(0, h - 1, w, h),  # noqa bottom edge
        shapely_box(0, 0, 1, h),  # noqa left
        shapely_box(w - 1, 0, w, h),  # noqa right
    ]
    margin_boxes = [
        shapely_box(0, 0, w, m),  # noqa top egde
        shapely_box(0, h - m, w, h),  # noqa bottom edge
        shapely_box(0, 0, m, h),  # noqa left
        shapely_box(w - m, 0, w, h),  # noqa right
    ]
    # ! this is wrt to WSI coord space, not tile
    margin_lines = [
        [[m, m], [w - m, m]],  # noqa top egde
        [[m, h - m], [w - m, h - m]],  # noqa bottom edge
        [[m, m], [m, h - m]],  # noqa left
        [[w - m, m], [w - m, h - m]],  # noqa right
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    margin_lines = [shapely_box(*v.flatten().tolist()) for v in margin_lines]

    # the ids within this match with those within `inst_map`, not UUID
    sel_indices = []
    if tile_mode in [0, 3]:
        # for `full grid` tiles `cross section` tiles
        # -- extend from the boundary by the margin size, remove
        #    nuclei whose entire contours lie within the margin area
        sel_boxes = [
            box
            for idx, box in enumerate(margin_boxes)
            if tile_flag[idx] or tile_mode == 3
        ]

        sel_indices = [
            index_by_id[id(geo)]
            for bounds in sel_boxes
            for geo in tile_rtree.query(bounds)
            if bounds.contains(geo)
        ]
    elif tile_mode in [1, 2]:
        # for `horizontal/vertical strip` tiles
        # -- extend from the marked edges (top/bot or left/right) by
        #    the margin size, remove all nuclei lie within the margin
        #    area (including on the margin line)
        # -- remove all nuclei on the boundary also

        sel_boxes = [
            margin_boxes[idx] if flag else boundary_lines[idx]
            for idx, flag in enumerate(tile_flag)
        ]

        sel_indices = [
            index_by_id[id(geo)]
            for bounds in sel_boxes
            for geo in tile_rtree.query(bounds)
        ]
    else:
        raise ValueError(f"Unknown tile mode {tile_mode}.")

    def retrieve_sel_uids(sel_indices, inst_dict):
        """Helper to retrieved selected instance uids."""
        if len(sel_indices) > 0:
            # not sure how costly this is in large dict
            inst_uids = list(inst_dict.keys())
        return [inst_uids[idx] for idx in sel_indices]

    remove_insts_in_tile = retrieve_sel_uids(sel_indices, inst_dict)

    # external removal only for tile at cross sections
    # this one should contain UUID with the reference database
    remove_insts_in_orig = []
    if tile_mode == 3:
        inst_boxes = [v["box"] for v in ref_inst_dict.values()]
        inst_boxes = np.array(inst_boxes)

        geometries = [shapely_box(*bounds) for bounds in inst_boxes]
        # An auxiliary dictionary to actually query the index within the source list
        index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
        ref_inst_rtree = STRtree(geometries)
        sel_indices = [
            index_by_id[id(geo)]
            for bounds in margin_lines
            for geo in ref_inst_rtree.query(bounds)
        ]

        remove_insts_in_orig = retrieve_sel_uids(sel_indices, ref_inst_dict)

    # move inst position from tile space back to WSI space
    # an also generate universal uid as replacement for storage
    new_inst_dict = {}
    for inst_uid, inst_info in inst_dict.items():
        if inst_uid not in remove_insts_in_tile:
            inst_info["box"] += np.concatenate([tile_tl] * 2)
            inst_info["centroid"] += tile_tl
            inst_info["contour"] += tile_tl
            inst_uuid = uuid.uuid4().hex
            new_inst_dict[inst_uuid] = inst_info
    return new_inst_dict, remove_insts_in_orig


# Python is yet to be able to natively pickle Object method/static method.
# Only top-level function is passable to multi-processing as caller.
# May need 3rd party libraries to use method/static method otherwise.
def _process_tile_predictions(
    ioconfig,
    tile_bounds,
    tile_flag,
    tile_mode,
    tile_output,
    # this would be replaced by annotation store
    # in the future
    ref_inst_dict,
    postproc,
    merge_predictions,
    model_name,
):
    """Function to merge new tile prediction with existing prediction,
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

    Returns:
        new_inst_dict (dict): A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list): List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.

    """
    locations, predictions = list(zip(*tile_output))
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

    out_dicts = postproc(head_raws)

    # debate as to whether to check size of objects
    if "hovernetplus" in model_name:
        _, inst_dict, layer_map, _ = postproc(head_raws)
        out_dicts = [inst_dict, layer_map]
    elif "hovernet" in model_name:
        _, inst_dict = postproc(head_raws)
        out_dicts = [inst_dict]
    else:
        out_dicts = postproc(head_raws)

    inst_dicts = [out for out in out_dicts if type(out) is dict]
    sem_maps = [out for out in out_dicts if type(out) is np.ndarray]

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


class MultiTaskSegmentor(NucleusInstanceSegmentor):
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
            ['instance', semantic']

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = MultiTaskSegmentor(model='hovernetplus-oed',
            output_type=['instance', 'semantic'])
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
        >>> # Each output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.0.dat', 'output/0.1.npy'
        >>> # Here, the second integer represents the task number
        >>> # e.g. between 0 or 1 for a two task model

    """

    def __init__(
        self,
        batch_size: int = 8,
        num_loader_workers: int = 0,
        num_postproc_workers: int = 0,
        model: torch.nn.Module = None,
        pretrained_model: str = None,
        pretrained_weights: str = None,
        verbose: bool = True,
        auto_generate_mask: bool = False,
        dataset_class: Callable = WSIStreamDataset,
        output_types: List = None,
    ):
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
        # default is None in base class and is un-settable
        # hence we redefine the namespace here
        self.num_postproc_workers = (
            num_postproc_workers if num_postproc_workers > 0 else None
        )

        # adding more runtime placeholder
        self._wsi_inst_info = []
        self.wsi_layers = []
        self.output_types = output_types

        if pretrained_model is not None:
            if "hovernetplus" in pretrained_model:
                self.output_types = ["instance", "semantic"]
            elif "hovernet" in pretrained_model:
                self.output_types = ["instance"]

    def _predict_one_wsi(
        self,
        wsi_idx: int,
        ioconfig: IOSegmentorConfig,
        save_path: str,
        mode: str,
    ):
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int): Index of the tile/wsi to be processed within `self`.
            ioconfig (IOSegmentorConfig): Object which defines I/O placement during
                inference and when assembling back to full tile/wsi.
            loader (torch.Dataloader): The loader object which return batch of data
                to be input to model.
            save_path (str): Location to save output prediction as well as possible
                intermediate results.
            mode (str): `tile` or `wsi` to indicate run mode.

        """
        cache_dir = f"{self._cache_dir}/"
        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path, mask_path, mode, self.auto_generate_mask
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
        # An auxiliary dictionary to actually query the index within the source list
        index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
        spatial_indexer = STRtree(geometries)

        # * retrieve tile placement and tile info flag
        # tile shape will always be corrected to be multiple of output
        tile_info_sets = self._get_tile_info(wsi_proc_shape, ioconfig)

        # ! running order of each set matters !
        self._futures = []

        # ! DEPRECATION:
        # !     will be deprecated upon finalization of SQL annotation store
        # !
        indices_sem = [i for i, x in enumerate(self.output_types) if x == "semantic"]
        indices_inst = [i for i, x in enumerate(self.output_types) if x == "instance"]

        for _ in indices_inst:
            self._wsi_inst_info.append({})

        for s_id in range(len(indices_sem)):
            self.wsi_layers.append(
                np.lib.format.open_memmap(
                    f"{cache_dir}/{s_id}.npy",
                    mode="w+",
                    shape=tuple(np.fliplr([wsi_proc_shape])[0]),
                    dtype=np.uint8,
                )
            )
            self.wsi_layers[s_id][:] = 0

        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            for tile_idx, tile_bounds in enumerate(set_bounds):
                tile_flag = set_flags[tile_idx]

                # select any patches that have their output
                # within the current tile
                sel_box = shapely_box(*tile_bounds)
                sel_indices = [
                    index_by_id[id(geo)] for geo in spatial_indexer.query(sel_box)
                ]

                # Empty tile
                if len(sel_indices) == 0:
                    continue

                tile_patch_inputs = patch_inputs[sel_indices]
                tile_patch_outputs = patch_outputs[sel_indices]
                self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)

                tile_infer_output = self._infer_once()

                self._process_tile_predictions(
                    ioconfig, tile_bounds, tile_flag, set_idx, tile_infer_output
                )
            self._merge_post_process_results()

        # Maybe store semantic annotations as contours in .dat file...
        for i_id, inst_idx in enumerate(indices_inst):
            joblib.dump(self._wsi_inst_info[i_id], f"{save_path}.{inst_idx}.dat")

        for s_id, sem_idx in enumerate(indices_sem):
            shutil.copyfile(f"{cache_dir}/{s_id}.npy", f"{save_path}.{sem_idx}.npy")
            # may need to chain it with parents

        self._wsi_inst_info = []  # clean up

    def _process_tile_predictions(
        self, ioconfig, tile_bounds, tile_flag, tile_mode, tile_output
    ):
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

    def _merge_post_process_results(self):
        """Helper to aggregate results from parallel workers."""

        def callback(new_inst_dicts, remove_uuid_lists, tiles, bounds):
            """Helper to aggregate worker's results."""
            # ! DEPRECATION:
            # !     will be deprecated upon finalization of SQL annotation store
            for inst_id, new_inst_dict in enumerate(new_inst_dicts):
                self._wsi_inst_info[inst_id].update(new_inst_dict)
                remove_uuid_list = remove_uuid_lists[inst_id]
                for inst_uuid in remove_uuid_list:
                    self._wsi_inst_info[inst_id].pop(inst_uuid, None)

            x_start, y_start, x_end, y_end = bounds
            for sem_id, tile in enumerate(tiles):
                max_h, max_w = self.wsi_layers[sem_id].shape
                x_end = min(x_end, max_w)
                y_end = min(y_end, max_h)
                w = x_end - x_start
                h = y_end - y_start
                self.wsi_layers[sem_id][y_start:y_end, x_start:x_end] = tile[0:h, 0:w]

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
            result = future.result()
            # manually call the callback rather than
            # attaching it when receiving/creating the future
            callback(*result)
