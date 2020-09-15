import numpy as np
import cv2


class ReinhardColourNormaliser(object):
    """ Normalize a patch color to the target image using the method of:
    A.C. Ruifrok & D.A. Johnston 'Quantification of histochemical staining 
    by color deconvolution'. Analytical and quantitative cytology and histology 
    / the International Academy of Cytology and American Society of Cytology, 
    vol. 23, no. 4

    Examples:
        >>> from tiatoolbox.tools.stainnorm.stain_extraction.ruifrok_stain_extractor import (
                RuifrokStainExtractor,
            )
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

    def transform(self, img):
        """Transform an image.

        Args:
            img (RGB uint8): Input image

        Returns:
            RGB float: colour normalised RGB image

        """
        chan1, chan2, chan3 = self.lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = (
            (chan1 - means[0]) * (self.target_stds[0] / stds[0])
        ) + self.target_means[0]
        norm2 = (
            (chan2 - means[1]) * (self.target_stds[1] / stds[1])
        ) + self.target_means[1]
        norm3 = (
            (chan3 - means[2]) * (self.target_stds[2] / stds[2])
        ) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(img):
        """Convert from RGB uint8 to LAB and split into channels.

        Args:
            img (RGB uint8): Input image

        Returns:
            chan1 (float): L
            chan2 (float): A
            chan3 (float): B

        """
        img = img.astype("uint8")  # ensure input image is uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= 2.55  # should now be in range [0,100]
        chan2 -= 128.0  # should now be in range [-127,127]
        chan3 -= 128.0  # should now be in range [-127,127]
        return chan1, chan2, chan3

    @staticmethod
    def merge_back(chan1, chan2, chan3):
        """Take seperate LAB channels and merge back to give RGB uint8.

        Args:
            chan1 (float): L channel
            chan2 (float): A channel
            chan3 (float): B channel
        Returns:
            RGB uint8: merged image

        """
        chan1 *= 2.55  # should now be in range [0,255]
        chan2 += 128.0  # should now be in range [0,255]
        chan3 += 128.0  # should now be in range [0,255]
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, img):
        """Get mean and standard deviation of each channel.

        Args:
            img (RGB uint8): Input image

        Returns:
            means (float): mean values for each RGB channel
            stds (float): standard deviation for each RGB channel

        """
        img = img.astype("uint8")  # ensure input image is uint8
        chan1, chan2, chan3 = self.lab_split(img)
        m1, sd1 = cv2.meanStdDev(chan1)
        m2, sd2 = cv2.meanStdDev(chan2)
        m3, sd3 = cv2.meanStdDev(chan3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds
