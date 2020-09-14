import numpy as np
import cv2


class ReinhardColourNormaliser(object):
    """ Normalize a patch color to the target image using the method of:
    A.C. Ruifrok & D.A. Johnston 
    'Quantification of histochemical staining by color deconvolution'.
    Analytical and quantitative cytology and histology / the International
    Academy of Cytology and American Society of Cytology, vol. 23, no. 4

    Examples:
        >>> from tiatoolbox.tools.stainnorm.reinhard_colour_normaliser import ReinhardColourNormaliser
        >>> norm = ReinhardColourNormaliser()
        >>> norm.fit(target_img)
        >>> trans = norm.transform(src_img)

    """
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        """Fit to a target image

        Args:
            target (RGB uint8): target image

        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        """Transform an image.

        Args:
            I (RGB uint8): Input image

        Returns:
            RGB float: colour normalised RGB image

        """
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(I):
        """Convert from RGB uint8 to LAB and split into channels.

        Args:
            I (RGB uint8): Input image

        Returns:
            I1 (float): L
            I2 (float): A
            I3 (float): B

        """
        I = I.astype('uint8') # ensure input image is uint8
        I = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        I_float = I.astype(np.float32)
        I1, I2, I3 = cv2.split(I_float)
        I1 /= 2.55  # should now be in range [0,100]
        I2 -= 128.0  # should now be in range [-127,127]
        I3 -= 128.0  # should now be in range [-127,127]
        return I1, I2, I3

    @staticmethod
    def merge_back(I1, I2, I3):
        """Take seperate LAB channels and merge back to give RGB uint8.

        Args:
            I1 (float): L channel
            I2 (float): A channel
            I3 (float): B channel
        Returns:
            RGB uint8: merged image

        """
        I1 *= 2.55  # should now be in range [0,255]
        I2 += 128.0  # should now be in range [0,255]
        I3 += 128.0  # should now be in range [0,255]
        I = np.clip(cv2.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(I, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, I):
        """Get mean and standard deviation of each channel.

        Args:
            I (RGB uint8): Input image

        Returns:
            means (float): mean values for each RGB channel
            stds (float): standard deviation for each RGB channel

        """
        I = I.astype('uint8') # ensure input image is uint8
        I1, I2, I3 = self.lab_split(I)
        m1, sd1 = cv2.meanStdDev(I1)
        m2, sd2 = cv2.meanStdDev(I2)
        m3, sd3 = cv2.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds