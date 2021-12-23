.. _pretrained-info-page:

=================================
Pre-trained Neural Network Models
=================================

^^^^^^^^^^^^^^^^^^^^^^
Patch Classification
^^^^^^^^^^^^^^^^^^^^^^

--------------------
Kather Patch Dataset
--------------------

The following models are trained using :obj:`Kather Dataset <tiatoolbox.models.dataset.info.KatherPatchDataset>`.
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

-----------------------------
Patch Camelyon (PCam) Dataset
-----------------------------

The following models are trained using the PCam dataset.
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
    - resnext50_pcam
    - resnext101_pcam
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


^^^^^^^^^^^^^^^^^^^^^^
Semantic Segmentation
^^^^^^^^^^^^^^^^^^^^^^

--------------------
Tissue Masking
--------------------

The following models are trained using internal data of TIACentre.
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


--------------------
Breast Cancer
--------------------

The following models are trained using `BCSS dataset <https://bcsegmentation.grand-challenge.org/>`_.
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

