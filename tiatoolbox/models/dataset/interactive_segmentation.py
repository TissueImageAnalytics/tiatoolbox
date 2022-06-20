import numpy as np
import os

from tiatoolbox.models.dataset import abc
from tiatoolbox.utils.misc import read_locations
from tiatoolbox.tools.patchextraction import get_patch_extractor


class InteractiveSegmentorDataset(abc.PatchDatasetABC):

    def __init__(self, img_path, points, resolution = 0, units = "level", patch_size = (128,128), label=None):
        """Creates an interactive segmentation dataset, which inherits from the
            torch.utils.data.Dataset class.
            This dataset extract a small patch around each point from the input image.

        Args:
            img_path (:obj:`str` or :obj:`pathlib.Path`): Path to a standard image,
                a whole-slide image or a large tile to read.
            points (ndarray, pd.DataFrame, str, pathlib.Path): Points ('clicks') for the image. 
            label: Label of the image, optional. Default is `None`.
            mode (str): Type of the image to process. Choose from either `patch`, `tile`
                or `wsi`.
            resolution (int or float or tuple of float): resolution at
                which to read the image, default = 0. Either a single
                number or a sequence of two numbers for x and y are
                valid. This value is in terms of the corresponding
                units. For example: resolution=0.5 and units="mpp" will
                read the slide at 0.5 microns per-pixel, and
                resolution=3, units="level" will read at level at
                pyramid level / resolution layer 3.
            units (str): the units of resolution, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                Only pyramid / resolution levels (level) embedded in
                the whole slide image are supported.
            patch_size: Size of the patch to extract (width, height), default = (128, 128)

        Examples:
            >>> # create a dataset to extract small patches around each point on a patch image
            >>> ds = InteractiveSegmentorDataset(
            ...     img_path = 'example_image.png',
            ...     points = 'example_points.csv',
            ...     mode = 'patch',         
            ... )

        """
        super().__init__()

        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")

        self.img_path = img_path
        self.label = label
        self.patch_size = patch_size    
        self.resolution = resolution    
        self.units = units             

        # Read the points('clicks') into a panda df 
        self.locations = read_locations(points)

        self.patch_extractor = get_patch_extractor("point",  
            input_img = self.img_path, locations_list = points, patch_size=self.patch_size,
            resolution = self.resolution, units = self.units)


    def __getitem__(self, idx):
        patch = self.patch_extractor.__getitem__(idx)

        bounding_box = self.get_bounding_box(idx)

        # we know the click is at the centre of the patch:
        nuc_point = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=np.uint8)
        nuc_point[0,int((self.patch_size[1]-1)/2),int((self.patch_size[0]-1)/2)] = 1

        exclusion_map = self.get_exclusion_map(idx, bounding_box)

        patch = np.moveaxis(patch, 2, 0)
        patch = patch / 255

        input = np.concatenate((patch, nuc_point, exclusion_map), axis=0, dtype=np.float32)   # shape=(c=5,h,w)
 
        data = {
            "input": input,
            "bounding_box": bounding_box,
            "click": (self.locations["x"][idx], self.locations["y"][idx])
        }
        if self.label is not None:
            data["label"] = self.label
        
        return data

    def get_bounding_box(self, idx):
        """This function returns a bounding box of size patch_size that has the click as its centre.
            The bounding box is the same box that is used in patch extraction.

        Args:
            idx (int): The index of the point ("Click") to get a bounding box for.
        Returns:
            bounds: a list of coordinates in `[start_x, start_y, end_x, end_y]`.
        """

        #Coordinates of the top left corner of each patch:
        location = (self.patch_extractor.locations_df["x"][idx], self.patch_extractor.locations_df["y"][idx])

        tl = np.array(location)
        br = location + np.array(self.patch_size)
        bounds = np.concatenate([tl, br])

        return  bounds.astype(int)

    def get_exclusion_map(self, idx, bounding_box):
        """This function returns an exclusion map for click at the given index.

        Args:
            idx (int): The index of the point ("Click") to get an exclusionMap for.
            boundingBox: a list of coordinates in `[start_x, start_y, end_x, end_y]`
                This is the bounding box for the click at the given index.
        Returns:
            exclusionMap (ndarray)
        """
        other_points = self.locations.drop(idx, axis = 0)
        x_locations = other_points["x"].to_numpy().astype(int)
        y_locations = other_points["y"].to_numpy().astype(int)

        xy_locations = np.stack((x_locations, y_locations), axis=1)
        sel = xy_locations[np.all((xy_locations>bounding_box[:2]) & (xy_locations<bounding_box[2:]), axis=1)]

        exclusion_map = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=np.uint8)
        exclusion_map[0, sel[:, 1]-bounding_box[1], sel[:, 0]-bounding_box[0]] = 1

        return exclusion_map


    def __len__(self):
        return self.locations.shape[0]       