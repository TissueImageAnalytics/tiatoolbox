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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module enables nucleus instance segmentation."""

import uuid
from typing import List, Union

import numpy as np

# replace with the sql database once the PR in place
import joblib
import pygeos

import torch
import tqdm

from tiatoolbox.models.segmentation import IOConfigSegmentor, SemanticSegmentor
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader


# till the day Python can natively pickle Object method
# the only way to passing function is top-level or may be
# static instance method
def _process_tile_predictions(
    ioconfig,
    tile_bound,
    tile_flag,
    tile_mode,
    tile_output,
    # this would be replaced by annotation store
    # in the future
    ref_inst_dict,
    postproc,
    merge_predictions,
):
    locations, predictions = list(zip(*tile_output))

    # convert from WSI space to Tile space
    tile_tl = tile_bound[:2]
    tile_br = tile_bound[2:]
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
            free_prediction=True,
        )
        head_raws.append(head_raw)
    inst_img, inst_dict = postproc(head_raws)

    # should be extremely rare
    if len(inst_dict) == 0:
        return {}, []

    # ! DEPRECATION:
    # !     will be deprecated upon finalization of SQL annotation store
    m = ioconfig.margin
    w, h = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    inst_boxes = np.array(inst_boxes)
    tile_rtree = pygeos.STRtree(
        pygeos.box(
            inst_boxes[:, 0], inst_boxes[:, 1], inst_boxes[:, 2], inst_boxes[:, 3]
        )
    )
    # !

    # create margin bounding box, ordering should match with
    # created tile info flag (top, bottom, left, right)
    boundary_lines = [
        pygeos.box(0, 0, w, 1),  # noqa top egde
        pygeos.box(0, h - 1, w, h),  # noqa bottom edge
        pygeos.box(0, 0, 1, h),  # noqa left
        pygeos.box(w - 1, 0, w, h),  # noqa right
    ]
    margin_boxes = [
        pygeos.box(0, 0, w, m),  # noqa top egde
        pygeos.box(0, h - m, w, h),  # noqa bottom edge
        pygeos.box(0, 0, m, h),  # noqa left
        pygeos.box(w - m, 0, w, h),  # noqa right
    ]
    # ! this is wrt to WSI coord space, not Tile
    margin_lines = [
        [[m, m], [w - m, m]],  # noqa top egde
        [[m, h - m], [w - m, h - m]],  # noqa bottom edge
        [[m, m], [m, h - m]],  # noqa left
        [[w - m, m], [w - m, h - m]],  # noqa right
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    margin_lines = [pygeos.box(*v.flatten().tolist()) for v in margin_lines]

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
            tile_rtree.query(bound, predicate="contains") for bound in sel_boxes
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
            tile_rtree.query(bound, predicate="intersects") for bound in sel_boxes
        ]

    def retrieve_sel_uids(sel_indices, inst_dict):
        sel_uids = []
        if len(sel_indices) > 0:
            # not sure how costly this is in large dict
            inst_uids = list(inst_dict.keys())
            sel_indices = [idx for sub_sel in sel_indices for idx in sub_sel]
            sel_uids = [inst_uids[idx] for idx in sel_indices]
        return sel_uids

    remove_insts_in_tile = retrieve_sel_uids(sel_indices, inst_dict)

    # external removal only for tile at cross sections
    # this one should contain UUID with the reference database
    remove_insts_in_orig = []
    if tile_mode == 3:
        # ! DEPRECATION:
        # !     will be deprecated upon finalization of SQL annotation store
        inst_boxes = [v["box"] for v in ref_inst_dict.values()]
        inst_boxes = np.array(inst_boxes)
        ref_inst_rtree = pygeos.STRtree(
            pygeos.box(
                inst_boxes[:, 0],
                inst_boxes[:, 1],
                inst_boxes[:, 2],
                inst_boxes[:, 3],
            )
        )
        # !

        # remove existings insts in old prediction that intersect
        # with the margin lines
        sel_indices = [
            ref_inst_rtree.query(bound, predicate="intersects")
            for bound in margin_lines
        ]
        remove_insts_in_orig = retrieve_sel_uids(sel_indices, ref_inst_dict)

    # move inst position from Tile space back to WSI space
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


class NucleusInstanceSegmentor(SemanticSegmentor):
    """Nucleus Instance Segmentor."""

    def _get_tile_info(
        self,
        image_shape: Union[List[int], np.ndarray],
        ioconfig: IOConfigSegmentor,
    ):
        """Generating tile information.

        To avoid out of memory problem when processing WSI-scale in general,
        the predictor will perform the inference and assemble on a large
        image tiles (each may have size of 4000x4000 compared to patch
        output of 256x256) first before sticthing every tiles at the end
        to complete the WSI output. For nuclei instance segmentation,
        the stiching process will require removal of predictions within
        some bounding areas. This function generates both the tile placement
        as well as the flag to indicate how the removal should be done to
        achieve the goal above.

        Args:
            image_shape (:class:`numpy.ndarray`, list(int)): the shape of WSI
                to extract the tile from, assumed to be in [width, height].
            ioconfig (:obj:IOConfigSegmentor): the input and output
                configuration objects.
        Returns:
            grid_tiles, removal_flags: ndarray
            vertical_strip_tiles, removal_flags: ndarray
            horizontal_strip_tiles, removal_flags: ndarray
            cross_section_tiles, removal_flags: ndarray

        """
        margin = np.array(ioconfig.margin)
        tile_shape = np.array(ioconfig.tile_shape)
        tile_shape = (
            np.floor(tile_shape / ioconfig.patch_output_shape)
            * ioconfig.patch_output_shape
        ).astype(np.int32)
        image_shape = np.array(image_shape)
        (tile_inputs, tile_outputs) = PatchExtractor.get_coordinates(
            image_shape=image_shape,
            patch_input_shape=tile_shape,
            patch_output_shape=tile_shape,
            stride_shape=tile_shape,
        )

        # * === Now generating the flags to indicate which side should
        # * === be removed in postproc callback
        boxes = tile_outputs

        # * remove all sides for boxes
        # unset for those lie within the selection
        def unset_removal_flag(boxes, removal_flag):
            """Unset removal flags for tiles intersecting image boundaries."""
            # ! DEPRECATION:
            # !     will be deprecated upon finalization of SQL annotation store
            sel_boxes = [
                pygeos.box(0, 0, w, 0),  # top egde
                pygeos.box(0, h, w, h),  # bottom edge
                pygeos.box(0, 0, 0, h),  # left
                pygeos.box(w, 0, w, h),  # right
            ]
            spatial_indexer = pygeos.STRtree(
                pygeos.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            )
            # !
            for idx, sel_box in enumerate(sel_boxes):
                sel_indices = spatial_indexer.query(sel_box)
                removal_flag[sel_indices, idx] = 0
            return removal_flag

        h, w = image_shape
        boxes = tile_outputs
        #  expand to full four corners
        # boxes_tl = boxes[:, :2]
        boxes_br = boxes[:, 2:]
        boxes_tr = np.dstack([boxes[:, 2], boxes[:, 1]])[0]
        boxes_bl = np.dstack([boxes[:, 0], boxes[:, 3]])[0]

        info = []
        # * remove edges on all sides, excluding edges at on WSI boundary
        flag = np.ones([boxes.shape[0], 4], dtype=np.int32)
        flag = unset_removal_flag(boxes, flag)
        info.append([boxes, flag])

        # * create vertical boxes at tile boundary and
        # * flag top and bottom removal, excluding those
        # * on the WSI boundary
        # -------------------
        # |    =|=   =|=    |
        # |    =|=   =|=    |
        # |   >=|=  >=|=    |
        # -------------------
        # |   >=|=  >=|=    |
        # |    =|=   =|=    |
        # |   >=|=  >=|=    |
        # -------------------
        # |   >=|=  >=|=    |
        # |    =|=   =|=    |
        # |    =|=   =|=    |
        # -------------------
        # only select boxes having right edges removed
        sel_indices = np.nonzero(flag[..., 3])
        _boxes = np.concatenate(
            [
                boxes_tr[sel_indices] - np.array([margin, 0])[None],
                boxes_br[sel_indices] + np.array([margin, 0])[None],
            ],
            axis=-1,
        )
        _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
        _flag[:, [0, 1]] = 1
        _flag = unset_removal_flag(_boxes, _flag)
        info.append([_boxes, _flag])

        # * create horizontal boxes at tile boundary and
        # * flag left and right removal, excluding those
        # * on the WSI boundary
        # -------------
        # |   |   |   |
        # |  v|v v|v  |
        # |===|===|===|
        # -------------
        # |===|===|===|
        # |   |   |   |
        # |   |   |   |
        # -------------
        # only select boxes having bottom edges removed
        sel_indices = np.nonzero(flag[..., 1])
        # top bottom left right
        _boxes = np.concatenate(
            [
                boxes_bl[sel_indices] - np.array([0, margin])[None],
                boxes_br[sel_indices] + np.array([0, margin])[None],
            ],
            axis=-1,
        )
        _flag = np.full([_boxes.shape[0], 4], 0, dtype=np.int32)
        _flag[:, [2, 3]] = 1
        _flag = unset_removal_flag(_boxes, _flag)
        info.append([_boxes, _flag])

        # * create boxes at tile cross-section and all sides
        # ------------------------
        # |     |     |     |    |
        # |    v|     |     |    |
        # |  > =|=   =|=   =|=   |
        # -----=-=---=-=---=-=----
        # |    =|=   =|=   =|=   |
        # |     |     |     |    |
        # |    =|=   =|=   =|=   |
        # -----=-=---=-=---=-=----
        # |    =|=   =|=   =|=   |
        # |     |     |     |    |
        # |     |     |     |    |
        # ------------------------

        # only select boxes having both right and bottom edges removed
        sel_indices = np.nonzero(np.prod(flag[:, [1, 3]], axis=-1))
        _boxes = np.concatenate(
            [
                boxes_br[sel_indices] - np.array([2 * margin, 2 * margin])[None],
                boxes_br[sel_indices] + np.array([2 * margin, 2 * margin])[None],
            ],
            axis=-1,
        )
        flag = np.full([_boxes.shape[0], 4], 1, dtype=np.int32)
        info.append([_boxes, flag])

        return info

    def _to_shared_space(self, wsi_idx, patch_inputs, patch_outputs):
        """Helper functions to transfer variable to shared space.

        We modify the shared space so that we can update worker info without
        needing to re-create the worker. There should be no race-condition
        because only by looping `self._loader` in main thread will trigger querying
        new data from each worker, and this portion should still be in sequential
        execution order in the main thread.

        """
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

    def _infer_once(self):
        num_steps = len(self._loader)

        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(num_steps),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for _, batch_data in enumerate(self._loader):
            sample_datas, sample_infos = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)
            sample_outputs = self.model.infer_batch(
                self._model,
                sample_datas,
                self._on_gpu,
            )
            # repackage so that its a N list, each contains
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
        return cum_output

    def _predict_one_wsi(
        self,
        wsi_idx: int,
        ioconfig: IOConfigSegmentor,
        save_path: str,
        mode: str,
    ):
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int): index of the tile/wsi to be processed within `self`.
            ioconfig (IOConfigSegmentor): object which defines I/O placement during
                inference and when assembling back to full tile/wsi.
            loader (torch.Dataloader): loader object which return batch of data
                to be input to model.
            save_path (str): location to save output prediction as well as possible
                intermediat results.
            mode (str): `tile` or `wsi` to indicate run mode.
        """
        wsi_path = self.imgs[wsi_idx]
        mask_path = None if self.masks is None else self.masks[wsi_idx]
        wsi_reader, mask_reader = self.get_reader(
            wsi_path, mask_path, mode, self.auto_generate_mask
        )

        resolution = ioconfig.highest_input_resolution
        if (
            isinstance(wsi_reader, VirtualWSIReader)
            and resolution["units"] != "baseline"
        ):
            raise ValueError("Inference on `tile` only use `units='baseline'` !")
        wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

        # * retrieve patch placement
        # this is in XY
        (patch_inputs, patch_outputs) = self.get_coordinates(wsi_proc_shape, ioconfig)
        if mask_reader is not None:
            sel = self.filter_coordinates(mask_reader, patch_outputs, **resolution)
            patch_outputs = patch_outputs[sel]
            patch_inputs = patch_inputs[sel]

        # assume to be in [topleft_x, topleft_y, botright_x, botright_y]
        # ! DEPRECATION:
        # !     will be deprecated upon finalization of SQL annotation store
        spatial_indexer = pygeos.STRtree(
            pygeos.box(
                patch_outputs[:, 0],
                patch_outputs[:, 1],
                patch_outputs[:, 2],
                patch_outputs[:, 3],
            )
        )
        # !

        # * retrieve tile placement and tile info flag
        # tile shape will always be corrected to be multiple of output
        tile_info_sets = self._get_tile_info(wsi_proc_shape, ioconfig)

        # ! running order of each set matters !
        self._futures = []

        # ! DEPRECATION:
        # !     will be deprecated upon finalization of SQL annotation store
        self._wsi_inst_info = {}
        # !

        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            for tile_idx, tile_bound in enumerate(set_bounds):
                tile_flag = set_flags[tile_idx]

                # select any patches that have their output
                # within the current tile
                sel_indices = spatial_indexer.query(pygeos.box(*tile_bound))
                tile_patch_inputs = patch_inputs[sel_indices]
                tile_patch_outputs = patch_outputs[sel_indices]
                self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)

                tile_infer_output = self._infer_once()

                # dump_path = (
                #     f"local/test_output/tile_set={set_idx}_tile={tile_idx}.dat"
                # )
                # # joblib.dump(tile_infer_output, dump_path)
                # tile_infer_output = joblib.load(dump_path)

                self._process_tile_predictions(
                    ioconfig, tile_bound, tile_flag, set_idx, tile_infer_output
                )

            self._merge_post_process_results()
        joblib.dump(self._wsi_inst_info, f"{save_path}.dat")
        # may need to chain it with parents
        self._wsi_inst_info = None  # clean up

    def _process_tile_predictions(
        self, ioconfig, tile_bound, tile_flag, tile_mode, tile_output
    ):
        """Function to dispatch parallel post processing."""
        args = [
            ioconfig,
            tile_bound,
            tile_flag,
            tile_mode,
            tile_output,
            self._wsi_inst_info,
            self.model.postproc_func,
            self.merge_prediction,
        ]
        if self._postproc_workers is not None:
            future = self._postproc_workers.submit(_process_tile_predictions, *args)
        else:
            future = _process_tile_predictions(*args)
        self._futures.append(future)
        return

    def _merge_post_process_results(self):
        """"Helper to aggregate results from parallel workers."""

        def callback(new_inst_dict, remove_uuid_list):
            """Helper to aggregate worker's results."""
            # ! DEPRECATION:
            # !     will be deprecated upon finalization of SQL annotation store
            self._wsi_inst_info.update(new_inst_dict)
            for inst_uuid in remove_uuid_list:
                self._wsi_inst_info.pop(inst_uuid, None)
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

        return
