"""WSIMeta to save metadata information for WSIs"""


class WSIMeta:
    """Whole slide image meta data class"""
    def __init__(self,
                 input_dir,
                 file_name,
                 objective_power=None,
                 slide_dimension=None,
                 level_count=None,
                 level_dimensions=None,
                 level_downsamples=None,
                 vendor=None,
                 mpp_x=None,
                 mpp_y=None,
                 ):
        self.input_dir = input_dir
        self.file_name = file_name
        self.objective_power = objective_power
        self.slide_dimension = slide_dimension
        self.level_count = level_count
        self.level_dimensions = level_dimensions
        self.level_downsamples = level_downsamples
        self.vendor = vendor
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y

    def as_dict(self):
        """
        Converts WSIMeta to dictionary to assist print and save in various formats

        Args:
            self (WSIMeta):

        Returns:
            dict: whole slide image meta data as dictionary

        """
        param = {
            "input_dir": self.input_dir,
            "objective_power": self.objective_power,
            "slide_dimension": self.slide_dimension,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "file_name": self.file_name,
        }
        return param
