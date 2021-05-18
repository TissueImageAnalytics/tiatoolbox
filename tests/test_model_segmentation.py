import os
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pytest
import torch


# sys.path.append('.')
sys.path.append('..')

# need to persist through 3 level
# level of task Thread A
# list of wsi_reader Thread A->B
# ----chunk within wsi_reader Thread A->B
# ----patch within chuck Thread A->B->C

import multiprocessing as mp
from concurrent.futures import (FIRST_EXCEPTION, ProcessPoolExecutor,
                                as_completed, wait)
mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data

from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import get_wsireader

from tiatoolbox.models.segmentation.predictor import Predictor, colorize, visualize_instances_dict
from tiatoolbox.models.segmentation.hovernet import HoVerNet

import time

# class SerializeReader(torch_data.Dataset):
#     """
#     """
#     def __init__(self, lock, wsi_path_list, mp_shared_space):
#         super().__init__()
#         self.mp_shared_space = mp_shared_space
#         self.lock = lock
#         self.wsi_path_list = wsi_path_list
#         self.wsi_idx = None # to be received externally via thread communication
#         return

#     def __len__(self):
#         return len(self.mp_shared_space.tile_info_idx)

#     def __getitem__(self, idx):
#         # ! no need to lock as we dont modify source value in shared space
#         if self.wsi_idx != self.mp_shared_space.wsi_idx:
#             self.wsi_idx = self.mp_shared_space.wsi_idx
#             holder = self.mp_shared_space.tile_info_idx.clone() # ?
#             self.tile_info_idx = holder
#             self.reader = get_wsireader(self.wsi_path_list[self.wsi_idx])

#         worker_info = torch.utils.data.get_worker_info()
#         time.sleep(worker_info.id + 1)
#         print(self.reader.info.file_path, self.tile_info_idx[idx])
#         return self.tile_info_idx[idx], np.array(worker_info.id + 1 )


# def sub_func(mp_forward_output_queue):
#     wsi_path_list = [
#         '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_001.svs',
#         '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_002.svs',
#         '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_001.svs',
#         '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_002.svs'
#     ]

#     mp_manager = torch_mp.Manager()
#     mp_shared_space = mp_manager.Namespace()
#     mp_mutext = mp_manager.Lock()

#     # ! by default, torch will split idxing across worker
#     # ! hence, for each batch, they coming entirely from different worker id
#     ds = SerializeReader(mp_mutext, wsi_path_list, mp_shared_space)
#     loader = torch_data.DataLoader(ds,
#                             batch_size=6,
#                             num_workers=2,
#                             drop_last=False,
#                             persistent_workers=True,
#                         )
#     for wsi_idx in range(len(wsi_path_list)):
#         dummy = np.array([1,2,3,4,5,6,7,8])
#         mp_shared_space.wsi_idx = torch.from_numpy(np.array(wsi_idx)).share_memory_()
#         mp_shared_space.tile_info_idx = torch.from_numpy(dummy*(wsi_idx+1)).share_memory_()
#         for _, data in enumerate(loader):
#             print(data)
#     return

# if __name__ == '__main__':
#     mp_manager_A = torch_mp.Manager()
#     # contain at most 5 tile ouput before polling forward func
#     mp_forward_output_queue = mp_manager_A.Queue(maxsize=16)

#     forward_process = mp.Process(target=sub_func, 
#                             args=(mp_forward_output_queue,))
#     forward_process.start()

#     while True:
#         if forward_process.exitcode is not None \
#             and mp_forward_output_queue.empty():
#             break

#     if forward_process.exitcode > 0:
#         raise ValueError(f'Forward process exited with code {forward_process.exitcode}')
#     forward_process.join()

from termcolor import colored
def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

def align_shape(arr_list, cval=0):
    """
    Return a new `a` and `b` such that
    they will have the shape=maximum(a.shape, b.shape, axis=-1)
    """
    shape_list = [v.shape[:2] for v in arr_list]
    max_h, max_w = np.max(shape_list, axis=0)

    new_arr_list = []
    for v in arr_list:
        if len(v.shape) == 3:    
            new_v = np.full((max_h, max_w, v.shape[-1]), cval, dtype=v.dtype)
        else:
            new_v = np.full((max_h, max_w), cval, dtype=v.dtype)
        new_v[:v.shape[0],:v.shape[1]] = v
        new_arr_list.append(new_v)
    return new_arr_list
###
if __name__ == '__main__':
    wsi_path_list = [
        '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_002.svs',
        '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_001.svs',
        '/home/tialab-dang/local/project/tiatoolbox/tests/data/TCGA-HE-7130-01Z-00-DX1.png',
        '/home/tialab-dang/local/project/tiatoolbox/tests/data/PCA-mini.svs',
    ]

    mask_path_list = [
        '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_002-mask.png',
        None,
        None,
        None
    ]

    # pretrained = '/home/tialab-dang/local/project/tiatoolbox/tests/pretrained/pecan-hover-net-pytorch.tar'
    # hovernet = HoVerNet(mode='fast', num_types=6)
    # pretrained = torch.load(pretrained)['desc']
    # hovernet.load_state_dict(pretrained)
    # predictor = Predictor(model=hovernet, num_loader_worker=4, num_postproc_worker=0)

    # pretrained = '/home/tialab-dang/local/project/tiatoolbox/tests/pretrained/pannuke.pth'
    # predictor = Predictor(
    #                 pretrained_model='hovernet-pannuke',
    #                 pretrained_weight=pretrained,
    #                 num_loader_worker=4,
    #                 num_postproc_worker=0)
    # predictor.predict(wsi_path_list, mask_path_list, mode='wsi', resolution=0.25, units='mpp')

    # idx = -1
    # output_dict = predictor.predict(wsi_path_list[idx:], mask_path_list[idx:],
    #                 mode='tile', resolution=2.0, units='baseline', on_gpu=False)

    # reader = get_wsireader(wsi_path_list[-1:][0])
    # thumb = reader.slide_thumbnail(resolution=2.0, units='baseline')
    # overlay = visualize_instances_dict(thumb, output_dict, line_thickness=1)
    # imwrite('dump.png', overlay)

    idx = -1
    from tiatoolbox.models.segmentation.semantic import Predictor as SemanticSegmentor
    from tiatoolbox.models.segmentation.generic import FCN_Model
    pretrained = '/home/tialab-dang/local/project/tiatoolbox/tests/pretrained/fcn-tissue_mask.tar'
    model = FCN_Model(nr_output_ch=3)
    pretrained = torch.load(pretrained)#['desc']
    # pretrained = convert_pytorch_checkpoint(pretrained)
    # torch.save(pretrained, '/home/tialab-dang/local/project/tiatoolbox/tests/pretrained/fcn-tissue_mask.tar')
    model.load_state_dict(pretrained)
    predictor = SemanticSegmentor(
                    model=model, 
                    num_loader_worker=4, 
                    num_postproc_worker=0,)
    output = predictor.predict([wsi_path_list[idx]], [mask_path_list[idx]],
                     mode='wsi', resolution=1.0, units='mpp')

    reader = get_wsireader(wsi_path_list[idx])
    thumb = reader.slide_thumbnail(resolution=1.0, units='mpp')
    output, thumb = align_shape([output, thumb])
    import matplotlib.pyplot as plt
    sel = output > 0
    alpha = 0.25
    colorize_output = (output * 255)[...,None]
    overlay = thumb.copy()
    overlay[sel] = thumb[sel] * alpha + (1-alpha) * colorize_output[sel]
    imwrite('dump.png', overlay)
    # exit()

# TODO: cpu mode
# TODO: pretrained model
# TODO: Tumor Seg
# reader = get_wsireader('/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini_002.svs')
# roi = reader.read_bounds((-150, -150, 150, 151), pad_mode='reflect')
# imwrite('dump.png', roi)
