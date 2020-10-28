from abc import ABC
from tiatoolbox.utils.exceptions import MethodNotSupported


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        img_patch_h: input image patch height
        img_patch_w: input image patch width
    """
    def __init__(self, img_patch_h, img_patch_w):
        raise NotImplementedError

    def get_last_steps(self, image_dim, label_patch_dim, stride):
        """
        Get the last location for patch extraction in a specific
        direction (horizontal or vertical).

        Args:
            image_dim: 1D size of image
            label_patch_dim: 1D size of patches
            stride: 1D size of stride for patch extraction

        Returns:
            last_step: the final location for patch extraction
        """
        nr_step = math.ceil((image_dim - label_patch_dim) / stride)
        last_step = (nr_step + 1) * stride
        return int(last_step)

    def extract_patches(self,
                        input_img_value,
                        labels=None,
                        save_output=False,
                        save_path=None,
                        save_name=None,
                        tile_objective_value=0):
        """
        Extract patches from an image

        Args:
            input_img_value (str, ndarray): input image
            labels (str, ndarray):
            save_output: whether to save extracted patches
            save_path: path where saved patches will be saved (only if save_output = True)
            save_name: filename for saving patches (only if save_output = True)
            tile_objective_value: level of WSI pyramid for patch extraction

        Returns:
            img_patches: extracted image patches
        """

        raise NotImplementedError

    def merge_patches(self, patches):
        """
        Merge the patch-level results to get the overall image-level prediction

        Args:
            patches: patch-level predictions

        Returns:
            image: merged prediction
        """

        raise NotImplementedError


class PointsPatchExtractor(PatchExtractor):
    """
    Class for extracting patches in standard and whole-slide images with specified point
    as a centre.

    Args:
        img_patch_h: input image patch height
        img_patch_w: input image patch width
    """

    def __init__(self, img_patch_h, img_patch_w):
        raise NotImplementedError

    def merge_patches(self, patches=None):
        raise MethodNotSupported(message="Merge patches not supported for "
                                         "PointsPatchExtractor")


def get_patch_extractor(method):
    """Return a patch extractor object as requested.
    Args:
        method (str): name of patch extraction method, must be one of
                            "window", "point".
    Return:
        PatchExtractor : an object with base 'PatchExtractor' as base class.
    Examples:
        >>> from tiatoolbox.tools.patchextraction import get_patch_extractor
        >>> patch_extractor = get_patch_extractor('window')

    """

    raise NotImplementedError
