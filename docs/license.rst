.. _license-page:

License Information
===================

Source Code
^^^^^^^^^^^

All model code is held under the `BSD-3 Clause License <https://github.com/TissueImageAnalytics/tiatoolbox/blob/develop/LICENSE>`_.

Model Weights
^^^^^^^^^^^^^

Despite the code being held under a permissive license, the licenses of model weights are dependent on the datasets that they are trained on.
We provide the licenses associated with the utilised datasets, but recommend that users also do their own due diligence for confirmation.

Patch Classification
--------------------

The model weights obtained from training on the `PCam dataset <https://github.com/basveeling/pcam>`_ are held under the `CC0 License <https://choosealicense.com/licenses/cc0-1.0/>`_.
Model weights obtained from training on the `Kather100K dataset <https://zenodo.org/record/1214456#.ZDgceOzMKqB>`_ are held under the
`Creative Commons Attribution 4.0 International License <https://creativecommons.org/licenses/by/4.0/legalcode>`_.

.. collapse:: PCam Dataset Weights (`License <https://choosealicense.com/licenses/cc0-1.0/>`_)

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

.. collapse:: Kather100K Dataset Weights (`License <https://creativecommons.org/licenses/by/4.0/legalcode>`_)

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


Semantic Segmentation
---------------------

Models weights obtained using internal data associated with the TIA Centre is held under a
`Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
Model weights optimized using the `BCSS dataset <https://bcsegmentation.grand-challenge.org/>`_ are held under a `CC0 License <https://choosealicense.com/licenses/cc0-1.0/>`_.

.. collapse:: Internal Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - fcn-tissue_mask

.. collapse:: BCSS Dataset Weights (`License <https://choosealicense.com/licenses/cc0-1.0/>`_)

    - fcn_resnet50_unet-bcss

Nucleus Instance Segmentation
-----------------------------

Models weights trained on the `PanNuke <https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke>`_, `MoNuSAC <https://monusac-2020.grand-challenge.org/>`_ and `Kumar <https://monuseg.grand-challenge.org/>`_ datasets are held under a
`Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.
Models trained on the CoNSeP dataset are held under an `Apache 2.0 License <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. collapse:: PanNuke Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - hovernet_fast-pannuke

.. collapse:: MoNuSAC Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0//>`_)

    - hovernet_fast-monusac

.. collapse:: CoNSeP Dataset Weights (`License <https://www.apache.org/licenses/LICENSE-2.0>`_)

    - hovernet_original-consep
    - micronet_hovernet-consep

.. collapse:: Kumar Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - hovernet_original_kumar


Nucleus Detection
-----------------

We provide the following models trained using the `CRCHisto <https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe//>`_
and CoNIC datasets. All model weights obtained from training on these datasets are held under a
`Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

.. collapse:: CRCHisto Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - sccnn-crchisto
    - mapde-crchisto

.. collapse:: CoNIC Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - sccnn-conic
    - mapde-conic


Multi-Task Segmentation
-----------------------

We provide the following model trained using a private OED dataset. The weights are held under a
`Creative Commons Attribution-NonCommercial-ShareAlike Version 4 (CC BY-NC-SA 4.0) License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.


.. collapse:: Private OED Dataset Weights (`License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_)

    - hovernetplus-oed
