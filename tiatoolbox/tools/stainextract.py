# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# This file contains code inspired by StainTools
# [https://github.com/Peter554/StainTools] written by Peter Byfield.
#
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Stain matrix extraction for stain normalisation."""
import numpy as np
from sklearn.decomposition import DictionaryLearning

from tiatoolbox.utils.transforms import convert_RGB2OD
from tiatoolbox.utils.misc import get_luminosity_tissue_mask


def vectors_in_correct_direction(e_vectors):
    """Points the eigen vectors in the right direction.

    Args:
        e_vectors (:class:`numpy.ndarray`): eigen vectors

    Returns:
        ndarray pointing in the correct direction

    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1, v2):
    """Rearranges input vectors for H&E in correct order with H as first output.

    Args:
        v1 (:class:`numpy.ndarray`): Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`): Input vector for stain extraction.

    Returns:
        input vectors in the correct order.

    """
    if v1[0] > v2[0]:
        he = np.array([v1, v2])
    else:
        he = np.array([v2, v1])

    return he


def dl_output_for_h_and_e(dictionary):
    """Return correct value for H and E from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
         :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        ndarray with correct values for H and E

    """
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]

    return dictionary


class CustomExtractor:
    """Get the user-defined stain matrix.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import CustomExtractor()
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = CustomExtractor(stain_matrix)
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, stain_matrix):
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError("Stain matrix must be either (2,3) or (3,3)")

    def get_stain_matrix(self, _):
        """Get the user defined stain matrix.

        Returns:
            :class:`numpy.ndarray`: user defined stain matrix.

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
        >>> from tiatoolbox.tools.stainextract import RuifrokExtractor()
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = RuifrokExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(_):
        """Get the pre-defined stain matrix.

        Returns:
            ndarray: pre-defined  stain matrix.

        """
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])


class MacenkoExtractor:
    """Macenko stain extractor.

    Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import MacenkoExtractor()
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = MacenkoExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(img, luminosity_threshold=0.8, angular_percentile=99):
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`): input image used for stain matrix estimation
            luminosity_threshold (float): threshold used for tissue area selection
            angular_percentile (int):

        Returns:
            :class:`numpy.ndarray`: estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = convert_RGB2OD(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, e_vects = np.linalg.eigh(np.cov(img_od, rowvar=False))

        # the two principle eigenvectors
        e_vects = e_vects[:, [2, 1]]

        # make sure vectors are pointing the right way
        e_vects = vectors_in_correct_direction(e_vectors=e_vects)

        # project on this basis.
        proj = np.dot(img_od, e_vects)

        # angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # min and max angles
        min_phi = np.percentile(phi, 100 - angular_percentile)
        max_phi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(e_vects, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(e_vects, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # order of H&E - H first row
        he = h_and_e_in_right_order(v1, v2)

        normalised_rows = he / np.linalg.norm(he, axis=1)[:, None]
        return normalised_rows


class VahadaneExtractor:
    """Vahadane stain extractor.

    Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import VahadaneExtractor()
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = VahadaneExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(img, luminosity_threshold=0.8, regulariser=0.1):
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`): input image used for stain matrix estimation
            luminosity_threshold (float): threshold used for tissue area selection
            regulariser (float): regulariser used in dictionary learning

        Returns:
            :class:`numpy.ndarray`: estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = convert_RGB2OD(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=2,
            alpha=regulariser,
            transform_alpha=regulariser,
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

        normalised_rows = dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

        return normalised_rows
