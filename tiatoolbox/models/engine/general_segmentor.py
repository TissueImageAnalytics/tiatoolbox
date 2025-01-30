from pathlib import Path
import shutil
import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from PIL import Image
from skimage.io import imread as skimread
from torchvision import transforms
from torchvision.models import inception_v3

from tiatoolbox import logger
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
)

from tiatoolbox.models.architecture.sam import SAM, SAMPrompts
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils.misc import download_data, imread, select_device
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, WSIReader
from tiatoolbox.models.engine.semantic_segmentor import _prepare_save_output

def _prepare_save_output(
    save_path: str | Path,
    img_shape: tuple[int, ...],
) -> tuple:
    """Prepares for saving the cached output."""
    if save_path is not None:
        save_path = Path(save_path)
        #if Path.exists(save_path):
            # Return error
        #else:
        memmap = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            shape=img_shape,
            dtype=np.float32,
        )
    #else:
        # Return error
        
    return memmap

def _prepare_save_dir( save_dir: str | Path | None) -> tuple[Path, Path]:
    """Prepare save directory and cache."""
    if save_dir is None:
        logger.warning(
            "Segmentor will only output to directory. "
            "All subsequent output will be saved to current runtime "
            "location under folder 'output'. Overwriting may happen! ",
            stacklevel=2,
        )
        save_dir = Path.cwd() / "output"

    save_dir = Path(save_dir).resolve()
    # if save_dir.is_dir():
    #     msg = f"`save_dir` already exists! {save_dir}"
    #     raise ValueError(msg)
    save_dir.mkdir(parents=True)
    return save_dir

class GeneralSegmentor:

    """ Model designed for general segmentation of WSIs. 
        Uses the SAM2 model architecture. """

    def __init__(self, 
                 model: SAM = None):
        
        self.model = model if model is not None else SAM() 
 
        # Defining ioconfig
        self.ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {"units": "mpp", "resolution": 1.0},
            ],
            output_resolutions=[
                {"units": "mpp", "resolution": 1.0},
            ],
            patch_input_shape=[512, 512],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            save_resolution={"units": "mpp", "resolution": 1.0},
        )

    def load_wsi(self, file_name):
        reader = WSIReader.open(file_name)
        self.img = reader.slide_thumbnail(
            resolution=self.ioconfig.save_resolution["resolution"],
            units=self.ioconfig.save_resolution["units"],
        )
        return self.img


    def predict(self, file_name, prompts: SAMPrompts = None, device = "cpu", save_path = None):
        """Predict on a WSI using prompts.
        Args:
            file_name (str): 
                Path to WSI file.
            prompts (SAMPrompts): 
                Prompts for SAM model.
            device (str): 
                Device to run inference on.
            save_path (str): 
                Location to save output prediction.
        """

        save_dir = _prepare_save_dir(save_dir=save_path)
        wsi_save_dir = save_dir / "0.npy"

        batch_data = self.load_wsi(file_name)
        self.prediction = self.model.infer_batch(model=self.model, batch_data=batch_data, prompts=prompts, device=device)

        save_memmap = _prepare_save_output(save_path=wsi_save_dir, img_shape=self.prediction.shape)
        np.copyto(save_memmap, self.prediction)

        self._outputs = [str(file_name), str(wsi_save_dir)]

        # ? will this corrupt old version if control + c midway?
        map_file_path = save_dir / "file_map.dat"
        # backup old version first
        if Path.exists(map_file_path):
            old_map_file_path = save_dir / "file_map_old.dat"
            shutil.copy(map_file_path, old_map_file_path)
        joblib.dump(self._outputs, map_file_path)

        return self.prediction
    
    def predict_wsi(self, file_name, device="cpu", save_path=None):
        return self.predict(file_name, device=device, save_path=save_path)
    
    def display_prediction(self, prediction=None):
        if prediction is None:
            prediction = self.prediction
        plt.figure(figsize=(20, 20))
        plt.imshow(self.img)
        self.show_anns(prediction)
        plt.axis('off')
        plt.show()

    # Imported from SAM2 example Jupyter notebook
    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
        ax.imshow(img)
        ax.axis('off')

    def show_mask(self, mask, ax, random_color=False, borders = True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=20):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.05)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=1))    

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(2, 2))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            plt.axis('off')
            plt.show()