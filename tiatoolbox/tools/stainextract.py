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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Stain normalisation utilities used in tiatoolbox"""
import numpy as np

from tiatoolbox.utils.transforms import convert_RGB2OD


class CustomExtractor:
    """Get the user-defined stain matrix

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    Examples:
        >>> from tiatoolbox.tools.staiextract import CustomExtractor()
        >>> extractor = CustomExtractor(stain_matrix)
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, stain_matrix):
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape != (2, 3) and self.stain_matrix.shape != (3, 3):
            raise Exception("Stain matrix must be either (2,3) or (3,3)")

    def get_stain_matrix(self, _):
        """Get the user defined stain matrix

        Returns:
            ndarray: user defined stain matrix

        """
        return self.stain_matrix


class RuifrokExtractor:
    """Get the stain matrix as defined in:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    Examples:
        >>> from tiatoolbox.tools.staiextract import RuifrokExtractor()
        >>> extractor = RuifrokExtractor()
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(_):
        """Get the pre-defined stain matrix

        Returns:
            ndarray: pre-defined  stain matrix

        """
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])


class MacenkoExtractor:
    """Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    Examples:
        >>> from tiatoolbox.tools.staiextract import MacenkoExtractor()
        >>> extractor = MacenkoExtractor()
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(img, luminosity_threshold=0.8, angular_percentile=99):
        """Stain matrix estimation

        Args:
            img (ndarray): input image used for stain matrix estimation
            luminosity_threshold (float): threshold used for tissue area selection
            angular_percentile (int):

        Returns:
            ndarray: estimated stain matrix

        """
        img = img.astype("uint8")  # ensure input image is uint8
        # Convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
            img, luminosity_threshold=luminosity_threshold
        ).reshape((-1,))
        OD = convert_RGB2OD(img).reshape((-1, 3))
        OD = OD[tissue_mask]

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1

        # Project on this basis.
        That = np.dot(OD, V)

        # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])

        # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        return normalize_matrix_rows(HE)


class VahadaneExtractor:
    """Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    Examples:
        >>> from tiatoolbox.tools.staiextract import VahadaneExtractor()
        >>> extractor = VahadaneExtractor()
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    @staticmethod
    def get_stain_matrix(img, luminosity_threshold=0.8, regulariser=0.1):
        """Stain matrix estimation

        Args:
            img (ndarray): input image used for stain matrix estimation
            luminosity_threshold (float): threshold used for tissue area selection
            regulariser (float): regulariser used in dictionary learning

        Returns:
            ndarray: estimated stain matrix

        """
        img = img.astype("uint8")  # ensure input image is uint8
        # convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
            img, luminosity_threshold=luminosity_threshold
        ).reshape((-1,))
        OD = convert_RGB2OD(img).reshape((-1, 3))
        OD = OD[tissue_mask]

        # do the dictionary learning
        dictionary = spams.trainDL(
            X=OD.T,
            K=2,
            lambda1=regulariser,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False,
        ).T

        # order H and E.
        # H on first row.
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return normalize_matrix_rows(dictionary)
