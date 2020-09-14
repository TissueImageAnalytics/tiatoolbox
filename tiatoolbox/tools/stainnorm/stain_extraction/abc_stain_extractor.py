from abc import ABC, abstractmethod


class ABCStainExtractor(ABC):
    """Abstract base class for stain extraction"""
    @staticmethod
    @abstractmethod
    def get_stain_matrix(I):
        """Estimate the stain matrix given an image.
        
        Args:
            I (ndarray): input image

        """
        pass