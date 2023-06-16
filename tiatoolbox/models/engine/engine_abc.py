from abc import ABC, abstractmethod


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_patch(self):
        raise NotImplementedError

    # how to deal with patches, list of patches/numpy arrays, WSIs
    # how to communicate with sub-processes.
    # define how to deal with patches as numpy/zarr arrays.
    # convert list of patches/numpy arrays to zarr and then pass to each sub-processes.
    # define how to read WSIs, read the image and convert to zarr array.
