from pathlib import Path
import shutil
import cv2
import joblib
import numpy as np 

from tiatoolbox import logger

from tiatoolbox.models.architecture.sam import SAM, SAMPrompts
from tiatoolbox.typing import IntPair, IntBounds
from tiatoolbox.wsicore.wsireader import WSIReader, WSIMeta

from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from shapely.geometry import Polygon

def _prepare_save_output(
    save_path: str | Path,
    mask_shape: tuple[int, ...],
    scores_shape: tuple[int, ...],
) -> tuple:
    """Prepares for saving the cached output."""
    if save_path is not None:
        save_path = Path(save_path)
        #if Path.exists(save_path):
            # Return error
        #else:
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
    #else:
        # Return error
        
    return mask_memmap, score_memmap

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
    if save_dir.is_dir():
        # msg = f"`save_dir` already exists! {save_dir}"
        # raise ValueError(msg)
        save_dir.rmdir()
    save_dir.mkdir(parents=True)
    return save_dir

class GeneralSegmentor:

    """ Model designed for general segmentation of WSIs. 
        Uses the SAM2 model architecture. """

    def __init__(self, 
                 model: SAM = None):
        self.model = SAM() if model is None else model
        self.scale_factor = 1.0

    def load_wsi(
            self, 
            file_name, 
            bounds: IntBounds = None,
            resolution: float = 1.0, 
            units: str = "mpp",
            ):
        self.reader = WSIReader.open(file_name)
        self.slide_dims = self.reader.slide_dimensions(1.0, "baseline")
        print(self.slide_dims)
        print(self.reader._info().slide_dimensions)
        if bounds is not None:
            self.img = self.reader.read_bounds(bounds, resolution, units)
            if units == "mpp":
                base_mpp = self.reader._info().mpp
                self.scale_factor = base_mpp / resolution
        else:
            self.img = self.reader.slide_thumbnail(1.0, "baseline")
        return self.img
    
    def bound_prompts(self, prompts, bounds):
        if prompts is not None and bounds is not None:
            if prompts.point_coords is not None:
                prompts.point_coords = (np.array(prompts.point_coords) - np.array(bounds[:2])) * np.array(self.scale_factor)

            if prompts.box_coords is not None:
                prompts.box_coords = (np.array(prompts.box_coords) - np.array(bounds[:2])) * np.array(self.scale_factor)

        return prompts
    
    def unbound_masks(self, masks, bounds):
        new_masks = []

        for mask in masks:
            new_size = (bounds[2]-bounds[0],bounds[3]-bounds[1])
            print(new_size)

            resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST) # Resizes the mask into the box at base resolution
        
            new_mask = np.zeros(np.array(self.slide_dims)[::-1], dtype=np.uint8)

            new_mask[bounds[1]:bounds[3],bounds[0]:bounds[2]] = resized_mask # Stores mask into base resolution whole image
            new_masks.append(new_mask)
        return np.array(new_masks, dtype=np.uint8)
                                                                             
    def predict(
            self, 
            file_name, 
            prompts: SAMPrompts = None, 
            device = "cpu", 
            save_path: Path = None, 
            bounds: IntBounds = None, 
            resolution = 1.0,
            units = "mpp"
    ) -> list[tuple[Path, Path, Path]]:
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
            bounds (tuple):
                Bounds for the region of interest.
        Returns:
            list: 
                List of paths to the saved output prediction.
        """
        file_name = Path(file_name)
        save_dir = _prepare_save_dir(save_dir=save_path)

        batch_data = self.load_wsi(file_name, resolution=resolution, units=units, bounds=bounds)
        prompts = self.bound_prompts(prompts, bounds=bounds)

        masks, scores = self.model.infer_batch(model=self.model, batch_data=batch_data, prompts=prompts, device=device)
        
        print(masks)

        if bounds is not None:
            masks = self.unbound_masks(masks, bounds)

        print(masks)
        
        mask_memmap, score_memmap = _prepare_save_output(save_path=save_dir, mask_shape=masks.shape, scores_shape=scores.shape)
        np.copyto(mask_memmap, masks)
        np.copyto(score_memmap, scores)

        self._outputs = [[str(file_name), str(save_dir / "0.npy"), str(save_dir / "1.npy")]]

        # ? will this corrupt old version if control + c midway?
        map_file_path = save_dir / "file_map.dat"
        # backup old version first
        if Path.exists(map_file_path):
            old_map_file_path = save_dir / "file_map_old.dat"
            shutil.copy(map_file_path, old_map_file_path)
        joblib.dump(self._outputs, map_file_path)

        print(f"Prediction stored at {save_dir}")

        return self._outputs
    
    def predict_wsi(self, file_name, device="cpu", save_path=None):
        return self.predict(file_name, device=device, save_path=save_path)
    
    def to_annotation(self, mask_path, score_path, save_filename: Path | str = None):
        """Converts the prediction output to annotation format."""
        
        masks = np.load(mask_path)
        scores = np.load(score_path)

        # Define annotation store path
        store_path = save_filename.with_suffix(".db")
        store = SQLiteStore()

        def mask_to_polygons(mask):
            """Extract polygons from a binary mask using OpenCV."""
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [Polygon(c.squeeze()) for c in contours if len(c) > 2]  # Avoid single-point contours
            return polygons
        
        for i in range(len(masks)):
            polygons = mask_to_polygons(masks[i])
            # Add extracted polygons to the annotation store
            props = {"score": f"{scores[i]}", "type": f"Mask {i+1}"}
            for poly in polygons:
                annotation = Annotation(geometry=poly, properties=props)
                store.append(annotation)

        store.create_index("id", '"id"')
        
        store.commit()
        store.dump(store_path)
        store.close()
        print(f"Annotations stored at {store_path}")
        return store_path
    
    def create_prompts(self, point_coords = None, point_labels = None, box_coords = None):
        return SAMPrompts(point_coords, point_labels, box_coords)

        
        
    
