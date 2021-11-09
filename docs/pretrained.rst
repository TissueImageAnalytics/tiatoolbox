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

    - googlenet-kather100k
    - mobilenet_v3_small-kather100k


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

