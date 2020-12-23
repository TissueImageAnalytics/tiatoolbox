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
    :show-inheritance:

.. autoclass:: OmnyxJP2WSIReader
    :show-inheritance:

.. autoclass:: VirtualWSIReader
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

-----
Image
-----

.. automodule:: tiatoolbox.utils.image
    :members:

-------------
Miscellaneous
-------------

.. automodule:: tiatoolbox.utils.misc
    :members:

    .. autofunction:: mpp2objective_power(mpp)
    .. autofunction:: objective_power2mpp(objective_power)
    .. autofunction:: mpp2common_objective_power(mpp, common_powers)

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
