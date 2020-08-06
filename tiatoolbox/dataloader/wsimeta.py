"""WSIMeta to save metadata information for WSIs"""


class WSIMeta:
    """Whole slide image meta data class"""
    def __init__(self,
                 input_dir,
                 file_name,
                 objective_power=None,
                 slide_dimension=None,
                 rescale=None,
                 tile_objective_value=None,
                 tile_read_size=None,
                 level_count=None,
                 level_dimensions=None,
                 level_downsamples=None,
                 vendor=None,
                 mpp_x=None,
                 mpp_y=None,
                 magnification_levels=None,
                 ):
        self.input_dir = input_dir
        self.file_name = file_name
        self.objective_power = objective_power
        self.slide_dimension = slide_dimension
        self.rescale = rescale
        self.tile_objective_value = tile_objective_value
        self.tile_read_size = tile_read_size
        self.level_count = level_count
        self.level_dimensions = level_dimensions
        self.level_downsamples = level_downsamples
        self.vendor = vendor
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y
        self.magnification_levels = magnification_levels

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
            "rescale": self.rescale,
            "tile_objective_value": self.tile_objective_value,
            "tile_read_size": self.tile_read_size,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "file_name": self.file_name,
            "magnification_levels": self.magnification_levels,
        }
        return param
