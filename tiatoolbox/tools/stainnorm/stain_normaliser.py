import numpy as np

from tiatoolbox.utils.transforms import convert_OD2RGB, convert_RGB2OD
from tiatoolbox.tools.stainnorm.stain_extraction.ruifrok_stain_extractor import (
    RuifrokStainExtractor,
)


class StainNormaliser(object):
    """Stain normalisation class

    Attributes:
        method (string): stain normalisation method to use

    Examples:
        >>> from tiatoolbox.tools.stainnorm.stain_normaliser import StainNormaliser
        >>> norm = StainNormaliser('ruifrok')
        >>> norm.fit(target_img)
        >>> transformed = norm.transform(source_img)

    """

    def __init__(self, method):
        if method.lower() == "ruifrok":
            self.extractor = RuifrokStainExtractor
        else:
            raise Exception("Method not recognized.")

    @staticmethod
    def get_concentrations(I, stain_matrix):
        """Estimate concentration matrix given an image and stain matrix.

        Args:
            I (ndarray): input image
            stain_matrix (ndarray): 2x3 (hxw) stain matrix for haematoxylin and eosin stains
        
        Returns:
            ndarray: stain concentrations of image I

        """
        OD = convert_RGB2OD(I).reshape((-1, 3))
        x, residuals, rank, s = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=-1)
        return x.T

    def fit(self, target):
        """Fit to a target image.

        Args:
            target (ndarray uint8): reference image that each source image shall be mapped to

        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target, self.stain_matrix_target
        )
        self.maxC_target = np.percentile(
            self.target_concentrations, 99, axis=0
        ).reshape((1, 2))
        self.stain_matrix_target_RGB = convert_OD2RGB(
            self.stain_matrix_target
        )  # useful to visualize.

    def transform(self, I):
        """Transform an image.

        Args:
            I (ndarray uint8): RGB input source image
        
        Returns:
            ndarray: RGB stain normalised image 

        """
        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = self.get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / maxC_source
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)
