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
Stain Normalization
^^^^^^^^^^^^^^^^^^^

- :obj:`get_normalizer <tiatoolbox.tools.stainnorm.get_normalizer>`
- :obj:`StainNormalizer <tiatoolbox.tools.stainnorm.StainNormalizer>`
- :obj:`CustomNormalizer <tiatoolbox.tools.stainnorm.CustomNormalizer>`
- :obj:`RuifrokNormalizer <tiatoolbox.tools.stainnorm.RuifrokNormalizer>`
- :obj:`MacenkoNormalizer <tiatoolbox.tools.stainnorm.MacenkoNormalizer>`
- :obj:`VahadaneNormalizer <tiatoolbox.tools.stainnorm.VahadaneNormalizer>`
- :obj:`ReinhardNormalizer <tiatoolbox.tools.stainnorm.ReinhardNormalizer>`

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

^^^^^^^^^^^^^^^^^^
Graph Construction
^^^^^^^^^^^^^^^^^^

- :obj:`Slide Graph Constructor <tiatoolbox.tools.graph.SlideGraphConstructor>`

^^^^^^^^^^^^^^^^^^^^^^^
Tile Pyramid Generation
^^^^^^^^^^^^^^^^^^^^^^^

- :obj:`Tile Pyramid Generator <tiatoolbox.tools.pyramid.TilePyramidGenerator>`
- :obj:`Zoomify <tiatoolbox.tools.pyramid.ZoomifyGenerator>`

^^^^^^^^^^^^^^^^^^^^
Dataset
^^^^^^^^^^^^^^^^^^^^

- :obj:`Kather Dataset <tiatoolbox.models.dataset.info.KatherPatchDataset>`

^^^^^^^^^^^^^^^^^^^^
Deep Learning Models
^^^^^^^^^^^^^^^^^^^^

--------------
Engine
--------------

- :obj:`Patch Prediction <tiatoolbox.models.engine.patch_predictor.PatchPredictor>`
- :obj:`Semantic Segmentation <tiatoolbox.models.engine.semantic_segmentor.SemanticSegmentor>`
- :obj:`Feature Extraction <tiatoolbox.models.engine.semantic_segmentor.DeepFeatureExtractor>`
- :obj:`Nucleus Instance Segmnetation <tiatoolbox.models.engine.nucleus_instance_segmentor.NucleusInstanceSegmentor>`

----------------------------
Neural Network Architectures
----------------------------

- :obj:`Torch Vision CNNs <tiatoolbox.models.architecture.vanilla>`
- :obj:`Simplified U-Nets <tiatoolbox.models.architecture.unet>`
- :obj:`HoVerNet <tiatoolbox.models.architecture.hovernet.HoVerNet>`
- :obj:`HoVerNet+ <tiatoolbox.models.architecture.hovernetplus.HoVerNetPlus>`

Pipelines:
    - :obj:`IDARS <tiatoolbox.models.architecture.idars>`

^^^^^^^^^
Utilities
^^^^^^^^^

- :obj:`Image <tiatoolbox.utils.image>`
- :obj:`Transforms <tiatoolbox.utils.transforms>`
- :obj:`Miscellaneous <tiatoolbox.utils.misc>`
- :obj:`Exceptions <tiatoolbox.utils.exceptions>`
- :obj:`Visualization <tiatoolbox.utils.visualization>`
