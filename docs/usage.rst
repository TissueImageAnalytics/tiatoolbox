=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox


----------
Dataloader
----------
.. automodule:: tiatoolbox.dataloader

^^^^^^^^^^^^^^^^^^^^
dataloader.wsireader
^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.dataloader.wsireader
    :members: WSIReader
    :special-members: __init__

^^^^^^^^^^^^^^^^^^^^^
dataloader.slide_info
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.dataloader.slide_info
    :members: slide_info

^^^^^^^^^^^^^^^^^^^^^
dataloader.save_tiles
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.dataloader.save_tiles
    :members: save_tiles

----------
Decorators
----------
.. automodule:: tiatoolbox.decorators

^^^^^^^^^^^^^^^^^^^^
decorators.multiproc
^^^^^^^^^^^^^^^^^^^^
.. automodule:: tiatoolbox.decorators.multiproc
    :members: TIAMultiProcess
    :special-members: __init__, __call__

------
Utils
------
.. automodule:: tiatoolbox.utils

^^^^^^^^^^
utils.misc
^^^^^^^^^^
.. automodule:: tiatoolbox.utils.misc
    :members: save_yaml, split_path_name_ext, grab_files_from_dir, imwrite, imresize

^^^^^^^^^^
utils.transforms
^^^^^^^^^^
.. automodule:: tiatoolbox.utils.transforms
    :members: background_composite
