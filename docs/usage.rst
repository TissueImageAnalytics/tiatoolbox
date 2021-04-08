=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox

^^^^^^^^^^^^^^^^^^^^^^
Reading WSI Image Data
^^^^^^^^^^^^^^^^^^^^^^

<<<<<<< HEAD
.. currentmodule:: tiatoolbox.wsi.wsireader
.. autoclass:: WSIReader
    :members: read_rect, read_bounds, read_region, slide_thumbnail, save_tiles
=======
.. automodule:: tiatoolbox.dataloader.wsireader
    :members: WSIReader, get_wsireader
    :private-members:
>>>>>>> d307a1328e49af96321e7139f59aa24ef17f2956

.. autoclass:: OpenSlideWSIReader
    :show-inheritance:

.. autoclass:: OmnyxJP2WSIReader
    :show-inheritance:

.. autoclass:: VirtualWSIReader
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
Accessing Metadata
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.wsi.wsimeta
    :members: WSIMeta

^^^^^^^^^^
Functional
^^^^^^^^^^

The wsi module also includes some functional syntax for quickly
obtaining information about a slide or generating tiles.

.. automodule:: tiatoolbox.wsi.slide_info
    :members: slide_info

.. automodule:: tiatoolbox.wsi.save_tiles
    :members: save_tiles


^^^^^^^^^^^^^^^^^^^
Stain Normalisation
^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.stainnorm
    :members: StainNormaliser, get_normaliser

.. autoclass:: CustomNormaliser
    :show-inheritance:

.. autoclass:: RuifrokNormaliser
    :show-inheritance:

.. autoclass:: MacenkoNormaliser
    :show-inheritance:

.. autoclass:: VahadaneNormaliser
    :show-inheritance:

.. autoclass:: ReinhardNormaliser
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
Stain Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.stainextract
    :members: CustomExtractor, RuifrokExtractor, MacenkoExtractor, VahadaneExtractor

^^^^^^^^^^^^^^^^^^
Patch Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.patchextraction
    :members: PatchExtractor, get_patch_extractor, convert_input_image_for_patch_extraction

.. autoclass:: PointsPatchExtractor
    :show-inheritance:

.. autoclass:: FixedWindowPatchExtractor
    :show-inheritance:

.. autoclass:: VariableWindowPatchExtractor
    :show-inheritance:

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
    .. autofunction:: conv_out_size(in_size, kernel_size=1, padding=0, stride=1)

----------
Transforms
----------

.. automodule:: tiatoolbox.utils.transforms
    :members:

    .. autofunction:: background_composite
    .. autofunction:: imresize
    .. autofunction:: convert_RGB2OD
    .. autofunction:: convert_OD2RGB
    .. autofunction:: bounds2locsize
    .. autofunction:: locsize2bounds

----------
Exceptions
----------

.. automodule:: tiatoolbox.utils.exceptions
    :members: FileNotSupported, MethodNotSupported
