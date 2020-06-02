"""
This file defines multiprocessing decorators required by the tiatoolbox.
"""

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


class TIAMultiProcess:
    """
    This class defines the multiprocessing decorator for the toolbox, requires a list iter_on as input on which
    multiprocessing will run
    """

    def __init__(self, iter_on):
        """
        __init__ function for TIAMultiProcess decorator
        Args:
            iter_on: Variable on which iterations will be performed.
        """
        self.iter_on = iter_on
        self.workers = multiprocessing.cpu_count()

    def __call__(self, func):
        """
        This is the function which will be called on a function on which decorator is applied
        Args:
            func: function to be run with multiprocessing

        Returns:

        """
        def func_wrap(*args, **kwargs):
            """
            Wrapping function for decorator call
            Args:
                *args: args inputs
                **kwargs: kwargs inputs

            Returns:

            """
            iter_value = None
            if "workers" in kwargs:
                self.workers = kwargs.pop("workers")
            try:
                iter_value = kwargs.pop(self.iter_on)
            except ValueError:
                print("Please specify iter_on in function decorator")

            with Pool(self.workers) as p:
                results = p.map(partial(func, **kwargs), iter_value,)
                p.close()

            return results

        return func_wrap
