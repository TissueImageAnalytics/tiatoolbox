=====
Usage
=====

To use TIA Toolbox in a project::

    import tiatoolbox

^^^^^^^^^^^^^^^^^^^^^^
Reading WSI Image Data
^^^^^^^^^^^^^^^^^^^^^^

- :obj:`wsireader <tiatoolbox.wsicore.wsireader>`
- :obj:`OpenSlideWSIReader <tiatoolbox.wsicore.wsireader.OpenSlideWSIReader>`
- :obj:`OmnyxJP2WSIReader <tiatoolbox.wsicore.wsireader.OmnyxJP2WSIReader>`
- :obj:`VirtualWSIReader <tiatoolbox.wsicore.wsireader.VirtualWSIReader>`
- :obj:`TIFFWSIReader <tiatoolbox.wsicore.wsireader.TIFFWSIReader>`

^^^^^^^^^^^^^^^^^^
Accessing Metadata
^^^^^^^^^^^^^^^^^^

- :obj:`WSIMeta <tiatoolbox.wsicore.wsimeta.WSIMeta>`

^^^^^^^^^^
Functional
^^^^^^^^^^

The wsicore module also includes some functional syntax for quickly
obtaining information about a slide or generating tiles.

- :obj:`slide_info <tiatoolbox.wsicore.slide_info>`
- :obj:`save_tiles <tiatoolbox.wsicore.save_tiles>`

^^^^^^^^^^^^^^^^^^
Stain Extraction
^^^^^^^^^^^^^^^^^^

- :obj:`Stain Extraction <tiatoolbox.tools.stainextract>`

^^^^^^^^^^^^^^^^^^^
Stain Normalisation
^^^^^^^^^^^^^^^^^^^

- :obj:`get_normaliser <tiatoolbox.tools.stainnorm.get_normaliser>`
- :obj:`StainNormaliser <tiatoolbox.tools.stainnorm.StainNormaliser>`
- :obj:`CustomNormaliser <tiatoolbox.tools.stainnorm.CustomNormaliser>`
- :obj:`RuifrokNormaliser <tiatoolbox.tools.stainnorm.RuifrokNormaliser>`
- :obj:`MacenkoNormaliser <tiatoolbox.tools.stainnorm.MacenkoNormaliser>`
- :obj:`VahadaneNormaliser <tiatoolbox.tools.stainnorm.VahadaneNormaliser>`
- :obj:`ReinhardNormaliser <tiatoolbox.tools.stainnorm.ReinhardNormaliser>`

^^^^^^^^^^^^^^^
Tissue Masking
^^^^^^^^^^^^^^^

- :obj:`Tissue Mask <tiatoolbox.tools.tissuemask>`

^^^^^^^^^^^^^^^^^^
Stain Augmentation
^^^^^^^^^^^^^^^^^^

- :obj:`Augmentation <tiatoolbox.tools.stainaugment>`


^^^^^^^^^^^^^^^^^^
Patch Extraction
^^^^^^^^^^^^^^^^^^

- :obj:`get_patch_extractor <tiatoolbox.tools.patchextraction.get_patch_extractor>`
- :obj:`PointsPatchExtractor <tiatoolbox.tools.patchextraction.PointsPatchExtractor>`
- :obj:`SlidingWindowPatchExtractor <tiatoolbox.tools.patchextraction.SlidingWindowPatchExtractor>`

^^^^^^^^^
Utilities
^^^^^^^^^

- :obj:`Image <tiatoolbox.utils.image>`
- :obj:`Transforms <tiatoolbox.utils.transforms>`
- :obj:`Miscellaneous <tiatoolbox.utils.misc>`
- :obj:`Exceptions <tiatoolbox.utils.exceptions>`


^^^^^^^^^^^^^^^^^^
Graph Construction
^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.graph
    :members:

^^^^^^^^^^^^^^^^^^^^^^^
Tile Pyramid Generation
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tiatoolbox.tools.pyramid
    :members:

.. autoclass:: TilePyramidGenerator
    :show-inheritance:

.. autoclass:: DeepZoomGenerator
    :show-inheritance:

.. autoclass:: ZoomifyGenerator
    :show-inheritance:

^^^^^^^^^^^^^^^^^^^^
Dataset
^^^^^^^^^^^^^^^^^^^^

- :obj:`Kather Dataset <tiatoolbox.models.dataset.info.KatherPatchDataset>`

^^^^^^^^^^^^^^^^^^^^
Deep Learning Models
^^^^^^^^^^^^^^^^^^^^

------------
Data Loader
------------

.. automodule:: tiatoolbox.models.architecture
    :members:

--------------
Engine
--------------

.. automodule:: tiatoolbox.models.engine
    :members:

.. automodule:: tiatoolbox.models.engine.patch_predictor
    :members:

.. automodule:: tiatoolbox.models.engine.semantic_segmentor
    :members:

.. automodule:: tiatoolbox.models.engine.nucleus_instance_segmentor
    :members:

-------
Dataset
-------

.. automodule:: tiatoolbox.models.dataset
    :members:

.. automodule:: tiatoolbox.models.dataset.classification
    :members:

----------------------------
Neural Network Architectures
----------------------------

-----
Visualization
-----

.. automodule:: tiatoolbox.utils.visualization
    :members:

-------------
Miscellaneous
-------------


- :obj:`Torch Vision CNNs <tiatoolbox.models.architecture.vanilla>`
- :obj:`Simplified U-Nets <tiatoolbox.models.architecture.unet>`

--------------
Engine
--------------

- :obj:`Patch Predictor <tiatoolbox.models.engine.patch_predictor>`
- :obj:`Semantic Segmentor <tiatoolbox.models.engine.semantic_segmentor>`
