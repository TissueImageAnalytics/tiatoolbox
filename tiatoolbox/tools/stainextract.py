"""Stain matrix extraction for stain normalization."""
import numpy as np
from sklearn.decomposition import DictionaryLearning

from tiatoolbox.utils.misc import get_luminosity_tissue_mask
from tiatoolbox.utils.transforms import rgb2od


def vectors_in_correct_direction(e_vectors):
    """Points the eigen vectors in the right direction.

    Args:
        e_vectors (:class:`numpy.ndarray`):
            Eigen vectors.

    Returns:
        :class:`numpy.ndarray`:
            Pointing in the correct direction.

    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1, v2):
    """Rearrange input vectors for H&E in correct order with H as first output.

    Args:
        v1 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`):
            Input vector for stain extraction.

    Returns:
        :class:`numpy.ndarray`:
            Input vectors in the correct order.

    """
    if v1[0] > v2[0]:
        return np.array([v1, v2])

    return np.array([v2, v1])


def dl_output_for_h_and_e(dictionary):
    """Return correct value for H and E from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        :class:`numpy.ndarray`:
            With correct values for H and E.

    """
    if dictionary[0, 0] < dictionary[1, 0]:
        return dictionary[[1, 0], :]

    return dictionary


class CustomExtractor:
    """Get the user-defined stain matrix.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import CustomExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = CustomExtractor(stain_matrix)
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, stain_matrix):
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError("Stain matrix must have shape (2, 3) or (3, 3).")

    def get_stain_matrix(self, _):
        """Get the user defined stain matrix.

        Returns:
            :class:`numpy.ndarray`:
                User defined stain matrix.

        """
        return self.stain_matrix


class RuifrokExtractor:
    """Reuifrok stain extractor.

    Get the stain matrix as defined in:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import RuifrokExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = RuifrokExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self):
        self.__stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

    def get_stain_matrix(self, _):
        """Get the pre-defined stain matrix.

        Returns:
            :class:`numpy.ndarray`:
                Pre-defined  stain matrix.

        """
        return self.__stain_matrix.copy()


class MacenkoExtractor:
    """Macenko stain extractor.

    Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection
        angular_percentile (int):
            Percentile of angular coordinates to be selected
            with respect to the principle, orthogonal eigenvectors.

    Examples:
        >>> from tiatoolbox.tools.stainextract import MacenkoExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = MacenkoExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, luminosity_threshold=0.8, angular_percentile=99):
        self.__luminosity_threshold = luminosity_threshold
        self.__angular_percentile = angular_percentile

    def get_stain_matrix(self, img):
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation.

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        angular_percentile = self.__angular_percentile

        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # eigenvectors of covariance in OD space (orthogonal as covariance symmetric)
        _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))

        # the two principle eigenvectors
        eigen_vectors = eigen_vectors[:, [2, 1]]

        # make sure vectors are pointing the right way
        eigen_vectors = vectors_in_correct_direction(e_vectors=eigen_vectors)

        # project on this basis.
        proj = np.dot(img_od, eigen_vectors)

        # angular coordinates with respect to the principle, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # min and max angles
        min_phi = np.percentile(phi, 100 - angular_percentile)
        max_phi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # order of H&E - H first row
        he = h_and_e_in_right_order(v1, v2)

        return he / np.linalg.norm(he, axis=1)[:, None]


class VahadaneExtractor:
    """Vahadane stain extractor.

    Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection.
        regularizer (float):
            Regularizer used in dictionary learning.

    Examples:
        >>> from tiatoolbox.tools.stainextract import VahadaneExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = VahadaneExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, luminosity_threshold=0.8, regularizer=0.1):
        self.__luminosity_threshold = luminosity_threshold
        self.__regularizer = regularizer

    def get_stain_matrix(self, img):
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        regularizer = self.__regularizer
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=2,
            alpha=regularizer,
            transform_alpha=regularizer,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=3,
            transform_max_iter=1000,
        )
        dictionary = dl.fit_transform(X=img_od.T).T

        # order H and E.
        # H on first row.
        dictionary = dl_output_for_h_and_e(dictionary)

        return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]
