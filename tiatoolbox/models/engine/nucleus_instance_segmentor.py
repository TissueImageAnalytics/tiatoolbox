"""This module enables nucleus instance segmentation."""

import uuid
from collections import deque
from typing import Callable, List, Union

# replace with the sql database once the PR in place
import joblib
import numpy as np
import torch
import tqdm
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.tools.patchextraction import PatchExtractor


# Python is yet to be able to natively pickle Object method/static
# method. Only top-level function is passable to multiprocessing as
# caller. May need 3rd party libraries to use method/static method
# otherwise.
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
):
    """Function to merge new tile prediction with existing prediction.

    Args:
        ioconfig (:class:`IOSegmentorConfig`):
            Object defines information about input and output placement
            of patches.
        tile_bounds (:class:`numpy.array`):
            Boundary of the current tile, defined as `(top_left_x,
            top_left_y, bottom_x, bottom_y)`.
        tile_flag (list):
            A list of flag to indicate if instances within an area
            extended from each side (by `ioconfig.margin`) of the tile
            should be replaced by those within the same spatial region
            in the accumulated output this run. The format is `[top,
            bottom, left, right]`, 1 indicates removal while 0 is not.
            For example, `[1, 1, 0, 0]` denotes replacing top and bottom
            instances within `ref_inst_dict` with new ones after this
            processing.
        tile_mode (int):
            A flag to indicate the type of this tile. There are 4 flags:
            - 0: A tile from tile grid without any overlapping, it is
                 not an overlapping tile from tile generation. The
                 predicted instances are immediately added to
                 accumulated output.
            - 1: Vertical tile strip that stands between two normal
                 tiles (flag 0). It has the same height as normal tile
                 but less width (hence vertical strip).
            - 2: Horizontal tile strip that stands between two normal
                 tiles (flag 0). It has the same width as normal tile
                 but less height (hence horizontal strip).
            - 3: Tile strip stands at the cross-section of four normal
                 tiles (flag 0).
        tile_output (list):
            A list of patch predictions, that lie within this tile, to
            be merged and processed.
        ref_inst_dict (dict):
            Dictionary contains accumulated output. The expected format
            is `{instance_id: {type: int, contour: List[List[int]],
            centroid:List[float], box:List[int]}`.
        postproc (callable):
            Function to post-process the raw assembled tile.
        merge_predictions (callable):
            Function to merge the `tile_output` into raw tile
            prediction.

    Returns:
        tuple:
            - :py:obj:`dict` - New instances dictionary:
                A dictionary contain new instances to be accumulated.
                The expected format is `{instance_id: {type: int,
                contour: List[List[int]], centroid:List[float],
                box:List[int]}`.
            - :py:obj:`list` - Instances IDs to remove:
                List of instance IDs within `ref_inst_dict` to be
                removed to prevent overlapping predictions. These
                instances are those get cut off at the boundary due to
                the tiling process.

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

    # As the placement output is calculated wrt the highest possible
    # resolution within input, the output will need to re-calibrate if
    # it is at different resolution than the input.
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
    _, inst_dict = postproc(head_raws)

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

    # create margin bounding box, ordering should match with created
    # tile info flag (top, bottom, left, right)
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
        # for `full grid` tiles `cross-section` tiles
        # -- extend from the boundary by the margin size, remove nuclei
        #    whose entire contours lie within the margin area
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
        # -- extend from the marked edges (top/bot or left/right) by the
        #    margin size, remove all nuclei lie within the margin area
        #    (including on the margin line)
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
        return []

    remove_insts_in_tile = retrieve_sel_uids(sel_indices, inst_dict)

    # external removal only for tile at cross-sections this one should
    # contain UUID with the reference database
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

    # move inst position from tile space back to WSI space and also
    # generate universal uid as replacement for storage
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
    """An engine specifically designed to handle tiles or WSIs inference.

    Note, if `model` is supplied in the arguments, it will ignore the
    `pretrained_model` and `pretrained_weights` arguments. Additionally,
    unlike `SemanticSegmentor`, this engine assumes each input model
    will ultimately predict one single target: the nucleus instance
    within the tiles/WSIs. Each WSI prediction will be store under a
    `.dat` file which contains a dictionary of form:

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
        model (nn.Module):
            Use externally defined PyTorch model for prediction with.
            weights already loaded. Default is `None`. If provided,
            `pretrained_model` argument is ignored.
        pretrained_model (str):
            Name of the existing models support by tiatoolbox for
            processing the data. For a full list of pretrained models,
            refer to the `docs
            <https://tia-toolbox.readthedocs.io/en/latest/pretrained.html>`_.
            By default, the corresponding pretrained weights will also
            be downloaded. However, you can override with your own set
            of weights via the `pretrained_weights` argument. Argument
            is case insensitive.
        pretrained_weights (str):
            Path to the weight of the corresponding `pretrained_model`.
        batch_size (int) :
            Number of images fed into the model each time.
        num_loader_workers (int):
            Number of workers to load the data.
          Take note that they will also perform preprocessing.
        num_postproc_workers (int):
            Number of workers to post-process predictions.
        verbose (bool):
            Whether to output logging information.
        dataset_class (obj):
            Dataset class to be used instead of default.
        auto_generate_mask (bool):
            To automatically generate tile/WSI tissue mask
          if is not provided.

    Examples:
        >>> # Sample output of a network
        >>> wsis = ['A/wsi.svs', 'B/wsi.svs']
        >>> predictor = SemanticSegmentor(model='hovernet_fast-pannuke')
        >>> output = predictor.predict(wsis, mode='wsi')
        >>> list(output.keys())
        [('A/wsi.svs', 'output/0') , ('B/wsi.svs', 'output/1')]
        >>> # Each output of 'A/wsi.svs'
        >>> # will be respectively stored in 'output/0.dat', 'output/0.dat'

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
        self._wsi_inst_info = None
        self._futures = []

    @staticmethod
    def _get_tile_info(
        image_shape: Union[List[int], np.ndarray],
        ioconfig: IOSegmentorConfig,
    ):
        """Generating tile information.

        To avoid out of memory problem when processing WSI-scale in
        general, the predictor will perform the inference and assemble
        on a large image tiles (each may have size of 4000x4000 compared
        to patch output of 256x256) first before stitching every tiles
        by the end to complete the WSI output. For nuclei instance
        segmentation, the stitching process will require removal of
        predictions within some bounding areas. This function generates
        both the tile placement and the flag to indicate how the removal
        should be done to achieve the above goal.

        Args:
            image_shape (:class:`numpy.ndarray`, list(int)):
                The shape of WSI to extract the tile from, assumed to be
                in `[width, height]`.
            ioconfig (:obj:IOSegmentorConfig):
                The input and output configuration objects.

        Returns:
            list:
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Grid tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Vertical strip tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Horizontal strip tiles
                    - :class:`numpy.ndarray` - Removal flags
                - :py:obj:`list` - Tiles and flags
                    - :class:`numpy.ndarray` - Cross section tiles
                    - :class:`numpy.ndarray` - Removal flags

        """
        margin = np.array(ioconfig.margin)
        tile_shape = np.array(ioconfig.tile_shape)
        tile_shape = (
            np.floor(tile_shape / ioconfig.patch_output_shape)
            * ioconfig.patch_output_shape
        ).astype(np.int32)
        image_shape = np.array(image_shape)
        (_, tile_outputs) = PatchExtractor.get_coordinates(
            image_shape=image_shape,
            patch_input_shape=tile_shape,
            patch_output_shape=tile_shape,
            stride_shape=tile_shape,
        )

        # * === Now generating the flags to indicate which side should
        # * === be removed in postproc callback
        boxes = tile_outputs

        # This saves computation time if the image is smaller than the expected tile
        if np.all(image_shape <= tile_shape):
            flag = np.zeros([boxes.shape[0], 4], dtype=np.int32)
            return [[boxes, flag]]

        # * remove all sides for boxes
        # unset for those lie within the selection
        def unset_removal_flag(boxes, removal_flag):
            """Unset removal flags for tiles intersecting image boundaries."""
            sel_boxes = [
                shapely_box(0, 0, w, 0),  # top edge
                shapely_box(0, h, w, h),  # bottom edge
                shapely_box(0, 0, 0, h),  # left
                shapely_box(w, 0, w, h),  # right
            ]
            geometries = [shapely_box(*bounds) for bounds in boxes]
            # An auxiliary dictionary to actually query the index within the source list
            index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
            spatial_indexer = STRtree(geometries)

            for idx, sel_box in enumerate(sel_boxes):
                sel_indices = [
                    index_by_id[id(geo)] for geo in spatial_indexer.query(sel_box)
                ]
                removal_flag[sel_indices, idx] = 0
            return removal_flag

        w, h = image_shape
        boxes = tile_outputs
        #  expand to full four corners
        boxes_br = boxes[:, 2:]
        boxes_tr = np.dstack([boxes[:, 2], boxes[:, 1]])[0]
        boxes_bl = np.dstack([boxes[:, 0], boxes[:, 3]])[0]

        # * remove edges on all sides, excluding edges at on WSI boundary
        flag = np.ones([boxes.shape[0], 4], dtype=np.int32)
        flag = unset_removal_flag(boxes, flag)
        info = deque([[boxes, flag]])

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

        We modify the shared space so that we can update worker info
        without needing to re-create the worker. There should be no
        race-condition because only by looping `self._loader` in main
        thread will trigger querying new data from each worker, and this
        portion should still be in sequential execution order in the
        main thread.

        Args:
            wsi_idx (int):
                The index of the WSI to be processed. This is used to
                retrieve the file path.
            patch_inputs (list):
                A list of coordinates in `[start_x, start_y, end_x,
                end_y]` format indicating the read location of the patch
                in the WSI image. The coordinates are in the highest
                resolution defined in `self.ioconfig`.
            patch_outputs (list):
                A list of coordinates in `[start_x, start_y, end_x,
                end_y]` format indicating the write location of the
                patch in the WSI image. The coordinates are in the
                highest resolution defined in `self.ioconfig`.

        """
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

    def _infer_once(self):
        """Running the inference only once for the currently active dataloader."""
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
            # repackage so that it's a N list, each contains
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
        ioconfig: IOSegmentorConfig,
        save_path: str,
        mode: str,
    ):
        """Make a prediction on tile/wsi.

        Args:
            wsi_idx (int):
                Index of the tile/wsi to be processed within `self`.
            ioconfig (IOSegmentorConfig):
                Object which defines I/O placement during inference and
                when assembling back to full tile/wsi.
            save_path (str):
                Location to save output prediction as well as possible
                intermediate results.
            mode (str):
                `tile` or `wsi` to indicate run mode.

        """
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
        self._wsi_inst_info = {}
        # !

        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            for tile_idx, tile_bounds in enumerate(set_bounds):
                tile_flag = set_flags[tile_idx]

                # select any patches that have their output
                # within the current tile
                sel_box = shapely_box(*tile_bounds)
                sel_indices = [
                    index_by_id[id(geo)] for geo in spatial_indexer.query(sel_box)
                ]

                # there is nothing in the tile
                # Ignore coverage as the condition is difficult
                # to reproduce on travis.
                if len(sel_indices) == 0:  # pragma: no cover
                    continue

                tile_patch_inputs = patch_inputs[sel_indices]
                tile_patch_outputs = patch_outputs[sel_indices]
                self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)

                tile_infer_output = self._infer_once()

                self._process_tile_predictions(
                    ioconfig, tile_bounds, tile_flag, set_idx, tile_infer_output
                )

            self._merge_post_process_results()
        joblib.dump(self._wsi_inst_info, f"{save_path}.dat")
        # may need to chain it with parents
        self._wsi_inst_info = None  # clean up

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
        ]
        if self._postproc_workers is not None:
            future = self._postproc_workers.submit(_process_tile_predictions, *args)
        else:
            future = _process_tile_predictions(*args)
        self._futures.append(future)

    def _merge_post_process_results(self):
        """Helper to aggregate results from parallel workers."""

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
            # ! this will lead to discard a bunch of
            # ! inferred tiles within this current WSI
            if future.exception() is not None:
                raise future.exception()

            # aggregate the result via callback
            result = future.result()
            # manually call the callback rather than
            # attaching it when receiving/creating the future
            callback(*result)
