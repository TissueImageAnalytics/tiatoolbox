"""Multiprocessing decorators required by the tiatoolbox."""

import multiprocessing
from functools import partial

from pathos.multiprocessing import ProcessingPool as Pool


class TIAMultiProcess:
    """Multiprocessing class decorator for the toolbox, requires a list `iter_on`
    as input on which multiprocessing will run

    Attributes:
        iter_on (str): Variable on which iterations will be performed.
        workers (int): num of cpu cores to use for multiprocessing.

    Examples:
        >>> from tiatoolbox.decorators.multiproc import TIAMultiProcess
        >>> import cv2
        >>> @TIAMultiProcess(iter_on="input_path")
        ... def read_images(input_path, output_dir=None):
        ...    img = cv2.imread(input_path)
        ...    return img
        >>> imgs = read_images(input_path)

    """

    def __init__(self, iter_on):
        """
        Args:
            iter_on: Variable on which iterations will be performed.
        """
        self.iter_on = iter_on
        self.workers = multiprocessing.cpu_count()

    def __call__(self, func):
        """
        Args:
            func: function to be run with multiprocessing

        Returns:

        """

        def func_wrap(*args, **kwargs):
            """Wrapping function for decorator call
            Args:
                *args: args inputs
                **kwargs: kwargs inputs

            Returns:

            """
            if "workers" in kwargs:
                self.workers = kwargs.pop("workers")
            try:
                iter_value = kwargs.pop(self.iter_on)
            except ValueError:
                raise ValueError("Please specify iter_on in multiprocessing decorator")

            with Pool(self.workers) as p:
                results = p.map(partial(func, **kwargs), iter_value,)
                p.clear()

            return results

        func_wrap.__doc__ = func.__doc__

        return func_wrap
