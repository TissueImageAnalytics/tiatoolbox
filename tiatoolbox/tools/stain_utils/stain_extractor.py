import numpy as np

from tiatoolbox.tools.stain_utils.abc_stain_extractor import (
    ABCStainExtractor,
)


class RuifrokStainExtractor(ABCStainExtractor):
    """Normalize a patch colour to the target image using the method of:

    A.C. Ruifrok & D.A. Johnston 'Quantification of histochemical staining
    by color deconvolution'. Analytical and quantitative cytology and histology
    / the International Academy of Cytology and American Society of Cytology

    """

    @staticmethod
    def get_stain_matrix(_):
        """Uses pre-defined stain matrix. Top row corresponds
        to haematoxylin and bottom row corresponds to eosin

        """
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
