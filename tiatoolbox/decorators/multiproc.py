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
# The Original Code is Copyright (C) 2006, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

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
