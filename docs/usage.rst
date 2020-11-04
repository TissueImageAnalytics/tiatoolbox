=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox

^^^^^^^^^^^^^^^^^^^^^^
Reading WSI Image Data
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tiatoolbox.dataloader.wsireader
.. autoclass:: WSIReader
    :members: read_rect, read_bounds, read_region, slide_thumbnail, save_tiles

.. autoclass:: OpenSlideWSIReader
    :members: slide_info
    :show-inheritance:

.. autoclass:: OmnyxJP2WSIReader
    :members: slide_info
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
Accessing Metadata
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.dataloader.wsimeta
    :members: WSIMeta

^^^^^^^^^^
Functional
^^^^^^^^^^

The dataloader module also includes some functional syntax for quickly
obtaining information about a slide or generating tiles.

.. automodule:: tiatoolbox.dataloader.slide_info
    :members: slide_info

.. automodule:: tiatoolbox.dataloader.save_tiles
    :members: save_tiles


^^^^^^^^^^^^^^^^^^^
Stain Normalisation
^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.stainnorm
    :members: StainNormaliser, CustomNormaliser, RuifrokNormaliser, MacenkoNormaliser, VahadaneNormaliser, ReinhardNormaliser, get_stain_normaliser
    :special-members: __init__
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
Stain Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.stainextract
    :members: CustomExtractor, RuifrokExtractor, MacenkoExtractor, VahadaneExtractor

^^^^^^^^^
Utilities
^^^^^^^^^

.. automodule:: tiatoolbox.utils

-------------
Miscellaneous
-------------

.. automodule:: tiatoolbox.utils.misc
    :members: save_yaml, split_path_name_ext, grab_files_from_dir, imwrite, load_stain_matrix, imread, get_luminosity_tissue_mask

----------
Transforms
----------

.. automodule:: tiatoolbox.utils.transforms
    :members: background_composite, imresize

----------
Exceptions
----------

.. automodule:: tiatoolbox.utils.exceptions
    :members: FileNotSupported, MethodNotSupported
