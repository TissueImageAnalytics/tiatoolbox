=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox

^^^^^^^^^^^^^^^^^^^^^^
Reading WSI Image Data
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.wsicore.wsireader
    :members: WSIReader, get_wsireader
    :private-members:

.. autoclass:: OpenSlideWSIReader
    :show-inheritance:

.. autoclass:: OmnyxJP2WSIReader
    :show-inheritance:

.. autoclass:: VirtualWSIReader
    :show-inheritance:

^^^^^^^^^^^^^^^^^^
Accessing Metadata
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.wsicore.wsimeta
    :members: WSIMeta

^^^^^^^^^^
Functional
^^^^^^^^^^

The wsicore module also includes some functional syntax for quickly
obtaining information about a slide or generating tiles.

.. automodule:: tiatoolbox.wsicore.slide_info
    :members: slide_info

.. automodule:: tiatoolbox.wsicore.save_tiles
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


^^^^^^^^^^^^^^^
Tissue Masking
^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.tissuemask
    :members:


^^^^^^^^^^^^^^^^^^
Stain Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.stainextract
    :members:

^^^^^^^^^^^^^^^^^^
Patch Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.patchextraction
    :members: PatchExtractor, get_patch_extractor, convert_input_image_for_patch_extraction

.. autoclass:: PointsPatchExtractor
    :show-inheritance:

.. autoclass:: SlidingWindowPatchExtractor
    :show-inheritance:

^^^^^^^^^^^^^^^^^^^^
Deep Learning Models
^^^^^^^^^^^^^^^^^^^^

.. automodule::tiatoolbox.models
    :members:

.. automodule:: tiatoolbox.models.abc
    :members:

--------
Backbone
--------

.. automodule:: tiatoolbox.models.backbone
    :members:

--------------
Classification
--------------

.. automodule:: tiatoolbox.models.classification
    :members:

.. automodule:: tiatoolbox.models.classification.abc
    :members:

.. automodule:: tiatoolbox.models.classification.patch_predictor
    :members:

.. automodule:: tiatoolbox.models.classification.pretrained_info
    :members:

-------
DataSet
-------

.. automodule:: tiatoolbox.models.dataset
    :members:

.. automodule:: tiatoolbox.models.dataset.classification
    :members:

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
