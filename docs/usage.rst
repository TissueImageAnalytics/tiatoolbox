=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox


----------
Dataloader
----------
.. autoclass:: tiatoolbox.dataloader.wsireader.WSIReader
    :members: __init__, slide_info

.. automodule:: tiatoolbox.dataloader.slide_info
    :members: slide_info

----------
Decorators
----------
.. automodule:: tiatoolbox.decorators.multiproc
    :members: TIAMultiProcess

------
Utils
------
.. automodule:: tiatoolbox.utils
.. automodule:: tiatoolbox.utils.misc
    :members: save_yaml, split_path_name_ext, grab_files_from_dir
