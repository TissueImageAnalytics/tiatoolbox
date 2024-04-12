.. _pretrained-info-page:

Pretrained Neural Network Models
================================

Despite the source code of TIAToolbox being held under a permissive license, the licenses of model weights are dependent on the datasets that they are trained on. We provide the licenses associated with the utilised datasets, but recommend that users also do their own due diligence for confirmation.

Patch Classification
^^^^^^^^^^^^^^^^^^^^

Kather Patch Dataset
--------------------

The following models are trained using :obj:`Kather Dataset <tiatoolbox.models.dataset.info.KatherPatchDataset>`.
Model weights obtained from training on the Kather100K dataset are held under the `Creative Commons Attribution 4.0 International License <https://creativecommons.org/licenses/by/4.0/legalcode>`_.
They share the same input output configuration defined below:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(224, 224),
            stride_shape=(224, 224),
            input_resolutions=[{"resolution": 0.5, "units": "mpp"}]
        )


.. collapse:: Model names

    - alexnet-kather100k
    - resnet18-kather100k
    - resnet34-kather100k
    - resnet50-kather100k
    - resnet101-kather100k
    - resnext50_32x4d-kather100k
    - resnext101_32x8d-kather100k
    - wide_resnet50_2-kather100k
    - wide_resnet101_2-kather100k
    - densenet121-kather100k
    - densenet161-kather100k
    - densenet169-kather100k
    - densenet201-kather100k
    - mobilenet_v2-kather100k
    - mobilenet_v3_large-kather100k
    - mobilenet_v3_small-kather100k
    - googlenet-kather100k

Patch Camelyon (PCam) Dataset
-----------------------------

The following models are trained using the `PCam dataset <https://github.com/basveeling/pcam/>`_.
The model weights obtained from training on the PCam dataset are held under the `CC0 License <https://choosealicense.com/licenses/cc0-1.0/>`_.
They share the same input output configuration defined below:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(96, 96),
            stride_shape=(96, 96),
            input_resolutions=[{"resolution": 1.0, "units": "mpp"}]
        )


.. collapse:: Model names

    - alexnet-pcam
    - resnet18-pcam
    - resnet34-pcam
    - resnet50-pcam
    - resnet101-pcam
    - resnext50-pcam
    - resnext101-pcam
    - wide_resnet50_2-pcam
    - wide_resnet101_2-pcam
    - densenet121-pcam
    - densenet161-pcam
    - densenet169-pcam
    - densenet201-pcam
    - mobilenet_v2-pcam
    - mobilenet_v3_large-pcam
    - mobilenet_v3_small-pcam
    - googlenet-pcam

Semantic Segmentation
^^^^^^^^^^^^^^^^^^^^^

Tissue Masking
--------------

The following models are trained using internal data of TIA Centre and are held under
the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
They share the same input output configuration defined below:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 2.0}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 2.0}
            ],
            patch_input_shape=(1024, 1024),
            patch_output_shape=(512, 512),
            stride_shape=(256, 256),
            save_resolution={'units': 'mpp', 'resolution': 8.0}
        )


.. collapse:: Model names

    - fcn-tissue_mask

Breast Cancer
-------------

The following models are trained using the `BCSS dataset <https://bcsegmentation.grand-challenge.org/>`_.
The model weights obtained from training on the BCSS dataset are held under the `CC0 License <https://choosealicense.com/licenses/cc0-1.0/>`_.
They share the same input output configuration defined below:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            patch_input_shape=(1024, 1024),
            patch_output_shape=(512, 512),
            stride_shape=(256, 256),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )


.. collapse:: Model names

    - fcn_resnet50_unet-bcss

Nucleus Instance Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PanNuke Dataset
---------------

We provide the following models trained using the `PanNuke dataset <https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke>`_.
All model weights trained on PanNuke are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25}
            ],
            margin=128
            tile_shape=[1024, 1024]
            patch_input_shape=(256, 256),
            patch_output_shape=(164, 164),
            stride_shape=(164, 164),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )

.. collapse:: Model names

    - hovernet_fast-pannuke

.. collapse:: Output Nuclear Classes

    - 0: Background
    - 1: Neoplastic
    - 2: Inflammatory
    - 3: Connective
    - 4: Dead
    - 5: Non-Neoplastic Epithelial

MoNuSAC Dataset
---------------

We provide the following models trained using the `MoNuSAC dataset <https://monusac.grand-challenge.org/>`_.
All model weights trained on MoNuSAC are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25}
            ],
            margin=128
            tile_shape=[1024, 1024]
            patch_input_shape=(256, 256),
            patch_output_shape=(164, 164),
            stride_shape=(164, 164),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )

.. collapse:: Model names

    - hovernet_fast-monusac

.. collapse:: Output Nuclear Classes

    - 0: Background
    - 1: Epithelial
    - 2: Lymphocyte
    - 3: Macrophage
    - 4: Neutrophil

CoNSeP Dataset
--------------

We provide the following models trained using the `CoNSeP dataset <https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/>`_.
The model weights obtained from training on the CoNSeP dataset are held under the `Apache 2.0 License <https://www.apache.org/licenses/LICENSE-2.0>`_.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25}
            ],
            margin=128
            tile_shape=[1024, 1024]
            patch_input_shape=(270, 270),
            patch_output_shape=(80, 80),
            stride_shape=(80, 80),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )

.. collapse:: Model names

    - hovernet_original-consep

.. collapse:: Output Nuclear Classes

    - 0: Background
    - 1: Epithelial
    - 2: Inflammatory
    - 3: Spindle-Shaped
    - 4: Miscellaneous


.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            tile_shape=[2048, 2048]
            patch_input_shape=(252, 252),
            patch_output_shape=(252, 252),
            stride_shape=(150, 150),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )

.. collapse:: Model names

    - micronet_hovernet-consep


Kumar Dataset
-------------

We provide the following models trained using the `Kumar dataset <https://monuseg.grand-challenge.org/>`_.
All model weights trained on Kumar are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The Kumar dataset does not contain nuclear class information, and so TIAToolbox pretrained models based on Kumar for nuclear segmentation, will only perform segmentation and not classification.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.25}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25},
                {'units': 'mpp', 'resolution': 0.25}
            ],
            margin=128
            tile_shape=[1024, 1024]
            patch_input_shape=(270, 270),
            patch_output_shape=(80, 80),
            stride_shape=(80, 80),
            save_resolution={'units': 'mpp', 'resolution': 0.25}
        )

.. collapse:: Model names

    - hovernet_original_kumar

Nucleus Detection
^^^^^^^^^^^^^^^^^

CRCHisto Dataset
--------------

We provide the following models trained using the `CRCHisto dataset <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe//>`_.
All model weights trained on CRCHisto are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(31, 31),
            stride_shape=(8, 8),
            input_resolutions=[{"resolution": 0.25, "units": "mpp"}]
        )


.. collapse:: Model names

    - sccnn-crchisto

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(252, 252),
            stride_shape=(150, 150),
            input_resolutions=[{"resolution": 0.25, "units": "mpp"}]
        )


.. collapse:: Model names

    - mapde-crchisto


CoNIC Dataset
--------------

We provide the following models trained using the `CoNIC dataset <https://conic-challenge.grand-challenge.org/>`_.
All model weights trained on CoNIC are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The input output configuration is as follows:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(31, 31),
            stride_shape=(8, 8),
            input_resolutions=[{"resolution": 0.25, "units": "mpp"}]
        )


.. collapse:: Model names

    - sccnn-conic

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOPatchPredictorConfig
        ioconfig = IOPatchPredictorConfig(
            patch_input_shape=(252, 252),
            stride_shape=(150, 150),
            input_resolutions=[{"resolution": 0.25, "units": "mpp"}]
        )


.. collapse:: Model names

    - mapde-conic


Multi-Task Segmentation
^^^^^^^^^^^^^^^^^^^^^^^

Oral Epithelial Dysplasia (OED) Dataset
---------------------------------------

We provide the following model trained using a private OED dataset. The model outputs nuclear instance segmentation
and classification results, as well as semantic segmentation of epithelial layers.
All model weights trained on the private OED dataset are held under the `Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
The model uses the following input output configuration:

.. collapse:: Input Output Configuration Details

   .. code-block:: python

        from tiatoolbox.models import IOSegmentorConfig
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 0.5}
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 0.5},
                {'units': 'mpp', 'resolution': 0.5},
                {'units': 'mpp', 'resolution': 0.5},
                {'units': 'mpp', 'resolution': 0.5}
            ],
            margin=128
            tile_shape=[1024, 1024]
            patch_input_shape=(256, 256),
            patch_output_shape=(164, 164),
            stride_shape=(164, 164),
            save_resolution={'units': 'mpp', 'resolution': 0.5}
        )

.. collapse:: Model names

    - hovernetplus-oed

.. collapse:: Output Nuclear Classes

    - 0: Background
    - 1: Other
    - 2: Epithelial

.. collapse:: Output Region Classes

    - 0: Background
    - 1: Other Tissue
    - 2: Basal Epithelium
    - 3: (Core) Epithelium
    - 4: Keratin
