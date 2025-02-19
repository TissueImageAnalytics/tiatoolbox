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

from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from shapely.geometry import Polygon

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

    def load_wsi(self, file_name): # Should include MPP as parameter
        reader = WSIReader.open(file_name,(1,1), 1.0)
        self.img = reader.slide_thumbnail(
            resolution=self.ioconfig.save_resolution["resolution"],
            units=self.ioconfig.save_resolution["units"],
        )
        return self.img


    def predict(self, file_name, prompts: SAMPrompts = None, device = "cpu", save_path: Path = None):
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
        file_name = Path(file_name)
        save_dir = _prepare_save_dir(save_dir=save_path)
        wsi_save_dir = save_dir / "0.npy"

        batch_data = self.load_wsi(file_name)
        self.prediction = self.model.infer_batch(model=self.model, batch_data=batch_data, prompts=prompts, device=device)

        save_memmap = _prepare_save_output(save_path=wsi_save_dir, img_shape=self.prediction.shape)
        np.copyto(save_memmap, self.prediction)

        self._outputs = [[str(file_name), str(wsi_save_dir)]]

        # ? will this corrupt old version if control + c midway?
        map_file_path = save_dir / "file_map.dat"
        # backup old version first
        if Path.exists(map_file_path):
            old_map_file_path = save_dir / "file_map_old.dat"
            shutil.copy(map_file_path, old_map_file_path)
        joblib.dump(self._outputs, map_file_path)

        print(f"Prediction stored at {wsi_save_dir}")

        return self._outputs
    
    def predict_wsi(self, file_name, device="cpu", save_path=None):
        return self.predict(file_name, device=device, save_path=save_path)
    
    def to_annotation(self, mask_path, save_filename: Path | str = None):
        """Converts the prediction output to annotation format."""
        
        masks = np.load(mask_path)

        def mask_to_polygons(mask):
            """Extract polygons from a binary mask using OpenCV."""
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [Polygon(c.squeeze()) for c in contours if len(c) > 2]  # Avoid single-point contours
            return polygons
        
        for mask in masks:
            polygons = mask_to_polygons(mask)

        # Define annotation store path
        store_path = save_filename.with_suffix(".db")
        store = SQLiteStore()

        # Add extracted polygons to the annotation store
        props = {"score": 1, "type": "Mask"}
        for poly in polygons:
            annotation = Annotation(geometry=poly, properties=props)
            store.append(annotation)

        store.create_index("area", '"area"')
        
        store.commit()
        store.dump(store_path)
        store.close()
        print(f"Annotations stored at {store_path}")
        return store_path
    
    def create_prompts(self, point_coords = None, point_labels = None, box_coords = None):
        return SAMPrompts(point_coords, point_labels, box_coords)

        
        
    
