import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


class TIAMultiProcess:
    def __init__(self, iter_on):
        self.iter_on = iter_on
        self.workers = multiprocessing.cpu_count()

    def __call__(self, func):
        def func_wrap(*args, **kwargs):
            iter_value = None
            if 'workers' in kwargs:
                self.workers = kwargs.pop('workers')
            try:
                iter_value = kwargs.pop(self.iter_on)
            except ValueError:
                print("Please specify iter_on in function decorator")

            with Pool(self.workers) as p:
                results = p.map(
                    partial(
                        func,
                        **kwargs
                    ),
                    iter_value,
                )

            return results
        return func_wrap
