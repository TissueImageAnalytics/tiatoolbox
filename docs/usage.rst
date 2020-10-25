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

.. currentmodule:: tiatoolbox.dataloader.wsireader
.. autoclass:: WSIReader
    :members: read_region, slide_thumbnail, save_tiles

.. autoclass:: OpenSlideWSIReader
    :members: slide_info, read_region, slide_thumbnail, save_tiles
    :show-inheritance:

.. autoclass:: OmnyxJP2WSIReader
    :members: slide_info, read_region, slide_thumbnail, save_tiles
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
dataloader.wsimeta
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.dataloader.wsimeta
    :members: WSIMeta

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

-----
Tools
-----
.. automodule:: tiatoolbox.tools

^^^^^^^^^^^^^^^
tools.stainnorm
^^^^^^^^^^^^^^^
.. automodule:: tiatoolbox.tools.stainnorm
    :members: StainNormaliser, CustomNormaliser, RuifrokNormaliser, MacenkoNormaliser, VahadaneNormaliser, ReinhardNormaliser, get_stain_normaliser
    :special-members: __init__
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
tools.stainextract
^^^^^^^^^^^^^^^^^^
.. automodule:: tiatoolbox.tools.stainextract
    :members: CustomExtractor, RuifrokExtractor, MacenkoExtractor, VahadaneExtractor

------
Utils
------
.. automodule:: tiatoolbox.utils

^^^^^^^^^^
utils.misc
^^^^^^^^^^
.. automodule:: tiatoolbox.utils.misc
    :members: save_yaml, split_path_name_ext, grab_files_from_dir, imwrite, load_stain_matrix, imread, get_luminosity_tissue_mask

^^^^^^^^^^^^^^^^
utils.transforms
^^^^^^^^^^^^^^^^
.. automodule:: tiatoolbox.utils.transforms
    :members: background_composite, imresize

^^^^^^^^^^^^^^^^
utils.exceptions
^^^^^^^^^^^^^^^^
.. automodule:: tiatoolbox.utils.exceptions
    :members: FileNotSupported, MethodNotSupported
