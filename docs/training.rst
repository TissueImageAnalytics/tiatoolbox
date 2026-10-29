Training
========

``tiatoolbox.models.training`` trains PyTorch models on pathology data and
hands them to TIAToolbox's inference engines. You supply the model, an
optimizer and PyTorch ``DataLoader`` objects. The package provides:

- **Datasets** for patch folders, image/mask pairs, annotation stores and
  whole-slide images, with target builders that turn annotations into masks,
  heatmaps or labels.
- **Tasks** that define the loss and metrics: ``ClassificationTask``,
  ``SegmentationTask``, and ``StructuredDenseTask`` for models with several
  output heads.
- A single-device **trainer** with validation, mixed precision, gradient
  accumulation, learning-rate scheduling, early stopping and resumable
  checkpoints.
- **Export**: ``ModelExport`` writes a ``model_config.yaml`` next to the
  weights, and ``load_trained_model`` rebuilds the model and its IO
  configuration for an inference engine.

Three example notebooks walk through complete runs:

- :doc:`_notebooks/jnb/13-training-patch-folder-classification`: a patch
  classifier trained from per-class folders, with checkpointing, resuming and
  whole-slide inference.
- :doc:`_notebooks/jnb/14-training-slide-annotation-segmentation`: tissue
  segmentation trained directly from whole-slide images and annotation stores.
- :doc:`_notebooks/jnb/15-training-kongnet-conic`: a multi-head KongNet nucleus
  model trained on CoNIC patches with ``StructuredDenseTask``.

Quick start
-----------

This example trains a classifier on a folder ``patches`` that holds one
subfolder of images per class.

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, Subset
    from tiatoolbox.models.architecture.vanilla import CNNModel
    from tiatoolbox.models.dataset import predefined_preproc_func
    from tiatoolbox.models.training import (
        ClassificationTask, PatchFolderClassificationDataset, Trainer,
        TrainerConfig, get_classification_augmentation, set_seed, split_indices,
    )

    set_seed(42)  # Before building the model, so its initial weights are seeded.
    dataset = PatchFolderClassificationDataset("patches")
    model = CNNModel("resnet18", num_classes=len(dataset.classes))
    model.preproc_func = predefined_preproc_func("kather100k")  # Scales to [0, 1].
    dataset = dataset.with_transforms(preproc=model.preproc_func)
    train_idx, val_idx = split_indices(
        len(dataset), stratify=dataset.targets, seed=42,
    )
    augmented = dataset.with_transforms(
        pair_transform=get_classification_augmentation("light"),
    )
    train_loader = DataLoader(
        Subset(augmented, train_idx), batch_size=16, shuffle=True,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(max_epochs=5, seed=42, output_dir="run"),
    )
    history = trainer.fit()

Three things in this example apply to every workflow:

- The dataset applies ``model.preproc_func``, the preprocessing that
  TIAToolbox's inference engines apply, so the model sees the same inputs in
  training and inference.
- ``split_indices`` assigns samples to splits, and ``with_transforms`` gives
  the training split its augmentation. Both subsets share one sample index, so
  validation stays unaugmented without scanning the files twice.
- ``set_seed`` runs before the model is built, so it also seeds the initial
  weights. ``TrainerConfig(seed=...)`` seeds the data order and augmentation.

``stratify`` takes one class label per sample. To keep each patient or slide
in a single split, pass ``groups`` instead:

.. code-block:: python

    train_idx, val_idx = split_indices(
        len(patient_ids), groups=patient_ids, val_fraction=0.2, seed=42,
    )

``val_fraction`` then applies to groups, so the validation share of patches can
differ from it. ``split_indices`` does not combine grouping with
stratification; for that, or for cross-validation, use a scikit-learn splitter
and pass its indices to ``Subset``.

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Dataset
     - Reads
   * - ``PatchFolderClassificationDataset``
     - One subfolder of images per class; integer labels.
   * - ``PatchMaskPairDataset``
     - Images and integer label masks: two folders matched by file stem, lists
       of files or arrays, or ``.npy`` stacks.
   * - ``PatchAnnotationDataset``
     - Patch images with annotation stores; a target builder turns the
       annotations into targets. ``from_dirs`` pairs two folders by file stem.
   * - ``SlideAnnotationPatchDataset``
     - A grid of patches read from whole-slide images at a chosen resolution,
       with targets built from the slides' annotation stores.

Each item is a dictionary with ``image`` and ``target`` entries;
``SlideAnnotationPatchDataset`` adds ``slide_index`` and ``bounds``. A custom
dataset can return the same dictionary, with ``label`` in place of ``target``
if preferred, or an ``(image, target)`` pair. Targets may be nested
dictionaries, lists or tuples of tensors.

Sample pipeline
~~~~~~~~~~~~~~~

Every dataset accepts four optional stages, applied in this order:

1. ``pair_transform(image, target)``: shared geometry and augmentation. The
   augmentation presets go here.
2. ``image_transform(image)``: appearance changes.
3. ``target_transform(target)``: label conversion, such as mapping
   colour-coded masks to class IDs.
4. ``preproc(image)``: the model's inference preprocessing. Pass
   ``preproc=model.preproc_func``.

Nothing is rescaled implicitly: without ``preproc``, a uint8 image reaches the
model as values from 0 to 255, as it would in an inference engine for a model
without preprocessing. ``preproc`` receives the HWC image (uint8 for image
files and slide reads) and returns an HWC array or tensor. The trainer warns
when a dataset's ``preproc`` differs from a TIAToolbox model's
``preproc_func``.

``dataset.with_transforms(...)`` returns a copy that shares the sample index,
with the stages you name replaced; passing ``None`` removes a stage.

Images and targets
~~~~~~~~~~~~~~~~~~

- Image files decode to RGB, including grayscale and RGBA files, as slide
  reads do. Arrays, in memory or in ``.npy`` files, keep their channels, so
  single-channel and multiplexed images also work.
- Masks keep their integer values, including palette indices and 16-bit
  labels. ``PatchMaskPairDataset`` is for class masks; for regression maps or
  several targets per patch, use annotations or a custom dataset.
- Three-dimensional NumPy targets are read as HWC and become CHW tensors;
  target tensors keep their layout.
- ``PatchMaskPairDataset`` memory-maps ``.npy`` stacks in each process, so
  data-loader workers do not receive copies; in-memory arrays are copied to
  spawned workers.

Whole-slide datasets
~~~~~~~~~~~~~~~~~~~~

``SlideAnnotationPatchDataset`` reads patches on a grid, optionally limited to
tissue with ``input_masks`` and ``min_mask_ratio``.
``generate_slide_patch_coordinates`` returns the same grid on its own.

- ``jitter=N`` shifts each patch by up to ``N`` pixels in training passes, with
  a new seeded offset every epoch. Validation passes and direct indexing use
  the grid.
- With a preset's ``scale_jitter``, a zoomed patch is read from a larger or
  smaller window centred on it and resized to the patch size; its ``bounds``
  give the window read.
- ``compute_targets(builder)`` builds every patch's target from the
  annotations alone, without reading pixels. With ``PresenceTargetBuilder``, it
  labels patches for a stratified split or a ``ClassBalancedIndexSampler``.
- Each data-loader worker caches up to ``cache_size`` slide readers and stores.
  With more slides than that,
  ``GroupedBlockSampler(dataset.sample_slide_indices, block_size)`` visits
  shuffled blocks of patches from one slide at a time, so the caches stay warm
  while every patch is still seen once per epoch.

Resources and workers
~~~~~~~~~~~~~~~~~~~~~

Datasets close only the readers and stores they open themselves; objects you
pass in remain yours to close. Call ``dataset.close()`` or use the dataset as a
context manager.

Commit file-backed SQLite stores before data-loader workers start. In-memory
stores are copied to workers as snapshots, so workers do not see later
changes. On Windows and macOS, workers are spawned processes, so transforms and
other callables must be importable: avoid lambdas and functions defined inside
other functions.

Augmentation
------------

``get_classification_augmentation``, ``get_segmentation_augmentation`` and
``get_annotation_augmentation`` return presets that you pass as a dataset's
``pair_transform``. Each takes a level:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Level
     - Operations
   * - ``light``
     - Flips and quarter turns; brightness, contrast or gamma changes.
   * - ``medium`` (default)
     - Adds noise, blur or JPEG compression, and stain augmentation.
   * - ``heavy``
     - The ``medium`` operations with larger ranges and higher probabilities.
   * - ``strong``
     - Flips and quarter turns, with ``StrongAugment`` in place of the other
       appearance changes: each patch gets zero, two, three or four random
       colour, intensity, filter, noise or JPEG operations, many of them far
       from realistic staining.

Stronger colour and stain augmentation helps a model cope with slides stained
or scanned differently from its training data. ``strong`` pushes this
furthest, so that a model learns morphology rather than stain appearance. It
uses :class:`~tiatoolbox.tools.strongaugment.StrongAugment`, TIAToolbox's port
of `StrongAugment <https://github.com/jopo666/StrongAugment>`_ (Pohjonen et
al., `arXiv:2206.15274 <https://arxiv.org/abs/2206.15274>`_), which can also be
used on its own.

Presets change geometry first, then appearance. At every level, they flip
patches and turn them by multiples of 90 degrees, which moves pixels exactly.
Other geometry is opt-in, through arguments of all three functions:

- ``scale_jitter=0.1`` zooms half the patches by a factor between 0.9 and 1.1.
  The range is centred on one, so the scale seen in training stays unbiased;
  a zoom that only magnifies trains at a finer scale than inference sees and
  did worse than no zoom at all. Zooming out pads the patch.
  ``crop_size=224`` trains on random crops of that size, which also give the
  zoom room: a crop of 224 in a 256-pixel patch zooms by 10% without padding.
  ``SlideAnnotationPatchDataset`` reads a larger or smaller window around each
  patch instead, so it never pads.
- ``rotation=20`` rotates half the patches by up to 20 degrees. It resamples
  the targets, which can break one-pixel boundaries, and pads the corners.

Padding is white in images, ``target_fill`` in masks and zero in heatmaps. To
keep padded mask pixels out of the loss, pass ``target_fill=task.ignore_index``.
The value must fit the mask's dtype, so uint8 masks need an ignore value such
as 255, with the task's ``ignore_index`` set to match.

``PatchAnnotationDataset`` and ``SlideAnnotationPatchDataset`` build their
targets for the cropped or zoomed window, so masks, one-pixel boundaries and
centroids are drawn at the training resolution rather than resampled. Other
spatial targets follow the image, using the interpolation their target builder
declares: nearest-neighbour for ``"mask"`` targets, bilinear for ``"image"``
targets such as heatmaps. Non-spatial labels are unchanged. Pass
``target_builder=`` to ``get_annotation_augmentation`` so that it knows which
targets are which.

Stain augmentation applies to RGB images only: it estimates the stains of each
patch separately and skips patches with almost no tissue. Preset randomness is
seeded from the run seed, the epoch and the sample index, so it is repeatable,
including with persistent workers, and leaves the model's random state alone.

Tasks, losses and metrics
-------------------------

A task turns model outputs and targets into a loss and metrics:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Task
     - Use it for
   * - ``ClassificationTask``
     - One label per patch: single-label, binary or multi-label.
   * - ``SegmentationTask``
     - One dense output of per-pixel classes, scored with Dice and IoU.
   * - ``StructuredDenseTask``
     - Several dense heads, such as masks, boundaries, heatmaps or regression
       maps, each with its own loss, weight and metrics (``DenseHeadSpec``).

Models that describe their concatenated output channels as named groups, such
as ``KongNet.training_output_spec``, can build one head per group with
``StructuredDenseTask.from_output_spec``.

Outputs and targets
~~~~~~~~~~~~~~~~~~~

Built-in tasks expect raw logits; regression heads take predictions directly.
TIAToolbox's ``CNNModel`` and ``TimmModel`` return logits from ``forward``.
Their ``infer_batch``, which engines call, turns logits into probabilities
according to the model's ``output_mode``: softmax for ``single_label`` heads,
whose ``postproc_func`` then takes the argmax, and sigmoid for ``binary`` and
``multi_label`` heads, whose ``postproc_func`` labels each output above 0.5.
``output_mode="auto"`` (the default) is ``binary`` for one output and
``single_label`` otherwise, so a multi-label head must be built with
``output_mode="multi_label"``. The trainer raises when a classifier's mode
disagrees with its ``ClassificationTask``, and ``ModelExport`` checks that
``model_kwargs`` rebuilds the same head. Apply the activation yourself when
calling ``forward`` directly.

Output and target sizes must match: tasks do not resize or crop. When a model
returns several outputs, a task selects one with ``output_key`` for a
dictionary or ``output_index`` for a sequence, and otherwise uses the first
tensor. Dense heads can also select a nested key and a channel range
(``channel_slice``).

.. list-table:: Batched shapes
   :header-rows: 1
   :widths: 33 27 40

   * - Task / target mode
     - Prediction
     - Target
   * - Classification ``single_label``
     - ``(N, C)``, ``C >= 2``
     - ``(N,)`` integer class IDs (also accepts ``(N, 1)``), or ``(N, C)``
       floating class probabilities (soft labels, as from mixup).
   * - Classification ``binary``
     - ``(N,)`` or ``(N, 1)``
     - Matching binary labels.
   * - Classification ``multi_label``
     - ``(N, C)``
     - ``(N, C)`` independent labels in [0, 1].
   * - Dense ``multiclass``
     - ``(N, C, H, W)``, ``C >= 2``
     - ``(N, H, W)`` integer IDs (also accepts ``(N, 1, H, W)``).
   * - Dense ``binary`` or ``regression``
     - ``(N, C, H, W)``
     - Matching tensor (also accepts ``(N, H, W)`` when ``C == 1``).

Losses
~~~~~~

- ``ClassificationTask`` uses cross-entropy, for single-label targets, by
  default; binary and multi-label targets need ``loss="bce_with_logits"``.
- Dense heads with ``loss="auto"`` choose by target mode: BCE for ``binary``,
  MSE for ``regression`` and cross-entropy for ``multiclass``. When the target
  mode is also automatic, one output channel means BCE and several mean
  cross-entropy. ``l1`` and ``smooth_l1`` are also available.
- A named loss fixes the target mode, and an incompatible combination, such as
  cross-entropy with binary targets, raises when the task is built.

Positions whose target equals the task's ``ignore_index`` (default -100) are
left out of losses and metrics. Each loss is a mean over supervised positions:
samples for single-label classification, pixels for dense multiclass, and
elements for binary, multi-label and regression targets. Dense heads are
averaged separately, multiplied by their ``loss_weight`` and summed; a head
with nothing to supervise contributes zero.

Metrics
~~~~~~~

Metrics are computed from counts accumulated over the whole epoch, not
averaged over batches.

- **Classification**: accuracy and macro-F1. Macro-F1 averages over every
  class, scoring a class missing from both predictions and labels as zero.
  Multi-label accuracy counts a sample only when all its labels match, and
  soft labels are scored by their argmax.
- **Dense classes**: Dice and IoU. Multiclass scores skip classes missing from
  both prediction and target; binary scores count an empty but supervised
  channel as a perfect match. ``metric_classes=(1, 2)`` averages only the
  listed classes and also reports each one, such as ``val_dice_1``, which keeps
  a dominant background class out of the score used to pick checkpoints.
- **Regression**: MAE and MSE.

History keys have ``train_`` and ``val_`` prefixes. ``StructuredDenseTask``
also prefixes each head's name, as in ``val_mask_dice``, and reports each
head's weighted loss as ``<head>_loss``.

Custom losses
~~~~~~~~~~~~~

Pass a callable as ``loss`` and set ``target_mode`` explicitly. The task still
selects outputs, checks shapes, masks ignored positions and computes metrics:

.. code-block:: python

    from torch import nn
    from tiatoolbox.models.training import (
        ClassificationTask, DenseHeadSpec, StructuredDenseTask,
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.5, 2.0]), label_smoothing=0.1, reduction="none",
    )
    task = ClassificationTask(loss=criterion, target_mode="single_label")

    # Model outputs and targets are dictionaries with these two keys.
    task = StructuredDenseTask([
        DenseHeadSpec(
            "mask", loss=nn.BCEWithLogitsLoss(reduction="none"),
            target_mode="binary",
        ),
        DenseHeadSpec(
            "distance", loss=nn.SmoothL1Loss(reduction="none"),
            target_mode="regression", loss_weight=0.2,
        ),
    ])

The callable receives the selected predictions and the targets, with ignored
positions set to zero, and returns one unreduced value per supervised target:
per element, or per sample for soft labels. The task masks and averages these
values, so it rejects scalar results. Elementwise losses, such as focal or
combined losses, fit this contract; batch-level objectives such as Dice or
contrastive losses need a custom task. Use stateless criteria, as their state
is not checkpointed.

Class weights and BCE ``pos_weight`` scale individual losses but not the
denominator, so weighted cross-entropy here is a plain mean of weighted values.
PyTorch's ``CrossEntropyLoss(reduction="mean")`` divides by the sum of the
weights instead. The plain mean keeps batches of different sizes, and the
epoch's reported loss, consistent.

Custom tasks
~~~~~~~~~~~~

Subclass ``TrainingTaskABC`` for another objective or metric:

- ``loss_components(output, targets)`` returns a dictionary of weighted mean
  losses whose sum is the objective, such as
  ``{"loss": criterion(output, targets)}``. With several components, the
  history also reports each one as ``<name>_loss``.
- ``loss_normalizers(targets, batch_size)`` returns each component's
  denominator, computed from the targets before the forward pass, so that
  accumulated batches are weighted correctly. The default treats a single
  ``"loss"`` as a mean over samples.
- Metrics are optional: ``reset_epoch_state``, ``update_epoch_state`` and
  ``compute_epoch_metrics`` accumulate detached statistics of bounded size over
  an epoch.

``compute_loss`` and ``compute_metrics`` apply these methods to a single batch
for interactive checks; the trainer does not use them.

Target builders
~~~~~~~~~~~~~~~

A target builder turns the annotations that fall in one patch into its target.
The supplied builders make masks (``MaskTargetBuilder``), boundaries, binary
disks and Gaussian heatmaps at annotation centroids, presence labels, the
class with the largest coverage, and multi-label coverage vectors. Two builders
combine others:

- ``CompositeTargetBuilder`` returns a dictionary of named targets.
- ``StackedTargetBuilder`` stacks channels of the same size into one array.

Query options (``where``, ``geometry_predicate``, ``min_area`` and
``distance``) set on either combiner run one store query for all its children,
which can narrow the result with their own ``where``.
``MaskTargetBuilder(instance_ids=True)`` paints one ID per annotation, largest
first so nested annotations stay visible; a ``target_transform`` can derive
distance or horizontal/vertical maps from it for instance segmentation.

To write a builder, subclass ``TargetBuilderABC``, implement ``build_target``
and, for spatial targets, declare ``spatial_target_spec``. Accept the shared
query options as ``**query: Unpack[AnnotationQuery]`` and pass them to
``super().__init__``.

Training configuration
----------------------

``TrainerConfig`` holds the settings for a run; all have defaults.

Hardware and precision
~~~~~~~~~~~~~~~~~~~~~~

- ``device="auto"`` uses CUDA when available, otherwise the CPU.
- ``amp="auto"`` turns on mixed precision on CUDA. With the default
  ``amp_dtype="float16"``, training passes run under float16 autocast with
  loss scaling, while validation and ``evaluate`` run in float32, as inference
  engines do: in eval mode, BatchNorm's running statistics can push
  activations past float16's range. ``amp_dtype="bfloat16"`` uses autocast for
  every pass, without loss scaling, on CUDA GPUs that support it or on the CPU
  with ``amp=True``.
- ``channels_last=True`` uses the channels-last memory format, which speeds up
  convolutions on recent GPUs. ``cudnn_benchmark=True`` lets cuDNN pick the
  fastest kernels for fixed input shapes.
- Batches move to the device with non-blocking copies, which overlap with
  computation when the loader uses ``pin_memory=True``. A single image tensor
  is converted to float32; a dictionary, list or tuple of tensors reaches a
  multi-input model unchanged.

Optimization
~~~~~~~~~~~~

- ``grad_accum_steps=N`` updates the weights once every ``N`` batches. Losses
  are weighted by how many positions each batch supervises, so the update
  matches one large batch, except that BatchNorm and dropout still see the
  smaller batches.
- ``grad_clip_norm`` clips the gradient norm before each update.
- A scheduler passed as ``Trainer(scheduler=...)`` steps once per epoch, and
  ``ReduceLROnPlateau`` receives the monitored value. For schedules counted in
  updates, such as ``OneCycleLR``, set ``scheduler_interval="step"`` and count
  optimizer updates rather than batches. Updates that mixed precision skips do
  not advance the scheduler.

Validation and early stopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Validation runs every ``val_interval`` epochs, by default every epoch.
- ``monitor`` names the history value that picks the best weights and drives
  early stopping; it defaults to ``"val_loss"``. Any history key works, such
  as a head's loss (``val_mask_loss``) or a per-class score (``val_dice_1``).
  Set ``monitor_mode="max"`` for scores such as Dice, and monitor a training
  value when there is no validation loader. A missing key raises an error,
  except in epochs without validation.
- ``early_stopping_patience=N`` stops after ``N`` consecutive monitored epochs
  without improvement; with ``N=1`` the first epoch that does not improve is
  the last. Epochs without validation do not count.

Results and logging
~~~~~~~~~~~~~~~~~~~

``fit`` returns the history: one dictionary per epoch, with ``epoch``, the
learning rate at the start of the epoch (``lr``), the ``train_`` and ``val_``
metrics, and ``epoch_time`` in seconds. Each epoch's monitored value is
logged. ``log_every_n_steps`` also logs the loss every so many batches (``0``
turns this off), and ``progress_bar=True`` shows a bar for each pass.

``Trainer(..., callbacks=[fn])`` calls ``fn(trainer, epoch_metrics)`` once each
epoch has been recorded and checkpointed. ``trainer.evaluate(loader)`` runs a
validation pass over any loader, by default the validation loader, and returns
metrics without prefixes.

Checkpoints and resuming
------------------------

The trainer writes its files to ``output_dir`` (default ``training_output``):

- ``last.ckpt``: the full training state, written after every epoch. With
  ``CheckpointConfig(save_last_every=N)`` it is written every ``N`` epochs and
  after the last one, which saves time for large models.
- ``best_model_weights.pth``: the weights from the best monitored epoch.
- ``model_config.yaml``: written when the trainer has a ``ModelExport``; see
  below.

``CheckpointConfig`` also renames these files or turns either checkpoint off.
Files are written atomically, so an interrupted write never leaves a corrupt
checkpoint.

``trainer.fit(resume_from="run/last.ckpt")`` continues from the next epoch.
Build the model, task, optimizer, scheduler and loaders as for the original
run; ``max_epochs`` is the total number of epochs, not the number to add. The
checkpoint restores the model, optimizer, scheduler and loss-scaler states, the
history, the best score, the early-stopping patience and the random state of
standard loaders and samplers; custom task or criterion state is not restored.
Resuming into a new output directory copies the previous best weights there.

A resumed run continues exactly only when the data, transforms, batching,
software and hardware are unchanged and operations are deterministic.
``deterministic=True`` selects deterministic cuDNN algorithms and turns off
cuDNN benchmarking, but does not make every PyTorch operation deterministic.
Your own transforms and samplers need their own seeding.

From training to inference
--------------------------

Give the trainer a ``ModelExport`` to write ``model_config.yaml`` next to the
weights, in the format of TIAToolbox's pretrained-model registry: the
architecture class and its keyword arguments, the engine IO configuration, the
class names and the preprocessing. ``load_trained_model`` reads it back and
returns the model with its weights and the IO configuration, as
``get_pretrained_model`` does for pretrained models:

.. code-block:: python

    from tiatoolbox.models.engine.io_config import IOPatchPredictorConfig
    from tiatoolbox.models.engine.patch_predictor import PatchPredictor
    from tiatoolbox.models.training import ModelExport, load_trained_model

    ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=(224, 224),
        stride_shape=(224, 224),
    )
    trainer = Trainer(
        ...,
        export=ModelExport(
            model_kwargs={"backbone": "resnet18", "num_classes": 2},
            ioconfig=ioconfig,
            class_dict={0: "normal", 1: "tumour"},
        ),
    )
    trainer.fit()

    model, ioconfig = load_trained_model("run")
    PatchPredictor(model=model).run(images=..., ioconfig=ioconfig, ...)

- ``model_kwargs`` must rebuild the trained head: for ``CNNModel`` and
  ``TimmModel``, ``num_classes`` and ``output_mode`` are checked against the
  model when the trainer starts and when the configuration is written, so a
  multi-label head needs ``"output_mode": "multi_label"`` in ``model_kwargs``.
- Only classes from ``tiatoolbox.models.architecture`` are rebuilt from the
  file, so loading never imports other code. For any other model, pass
  ``model=`` an instance to load the weights into, or a class to build from the
  recorded arguments.
- A class can also select an inference variant of the trained architecture:
  ``model=KongNetSegmentor`` loads a trained ``KongNet`` for
  ``MultiTaskSegmentor``. ``weights="last.ckpt"`` loads another weights file.
- Only constructor arguments in ``model_kwargs`` are recorded. Post-processing
  settings travel with the model when they are constructor arguments, as
  KongNet's thresholds are. Keep training-only arguments out of
  ``model_kwargs``: ``KongNet(**model_kwargs, pretrained_encoder=True)`` starts
  training from ImageNet weights without making ``load_trained_model``
  download them again.
- Preprocessing from ``predefined_preproc_func`` is recorded by name. The
  trainer warns when a model's ``preproc_func`` is a custom function, which you
  then set again after loading.

``load_model_state_dict`` in ``tiatoolbox.models.models_abc`` loads either bare
state dictionaries or trainer checkpoints, removing ``DataParallel`` and
``torch.compile`` wrappers.

API reference
-------------

The package root exports the workflow components. Type aliases, checkpoint
file utilities and device resolution stay in their own modules.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   tiatoolbox.models.training.Trainer
   tiatoolbox.models.training.TrainerConfig
   tiatoolbox.models.training.CheckpointConfig
   tiatoolbox.models.training.ClassificationTask
   tiatoolbox.models.training.SegmentationTask
   tiatoolbox.models.training.StructuredDenseTask
   tiatoolbox.models.training.DenseHeadSpec
   tiatoolbox.models.training.TrainingTaskABC
   tiatoolbox.models.training.PatchFolderClassificationDataset
   tiatoolbox.models.training.PatchMaskPairDataset
   tiatoolbox.models.training.PatchAnnotationDataset
   tiatoolbox.models.training.SlideAnnotationPatchDataset
   tiatoolbox.models.training.TrainingAugmentationPreset
   tiatoolbox.models.training.ModelExport
   tiatoolbox.models.training.TargetBuilderABC
   tiatoolbox.models.training.AnnotationQuery
   tiatoolbox.models.training.MaskTargetBuilder
   tiatoolbox.models.training.BoundaryTargetBuilder
   tiatoolbox.models.training.BinaryDiskTargetBuilder
   tiatoolbox.models.training.GaussianHeatmapTargetBuilder
   tiatoolbox.models.training.PresenceTargetBuilder
   tiatoolbox.models.training.CoverageClassTargetBuilder
   tiatoolbox.models.training.MultiLabelTargetBuilder
   tiatoolbox.models.training.CompositeTargetBuilder
   tiatoolbox.models.training.StackedTargetBuilder
   tiatoolbox.models.training.StackedSpatialTargetSpec
   tiatoolbox.models.training.ClassBalancedIndexSampler
   tiatoolbox.models.training.GroupedBlockSampler

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst

   tiatoolbox.models.training.split_indices
   tiatoolbox.models.training.generate_slide_patch_coordinates
   tiatoolbox.models.training.set_seed
   tiatoolbox.models.training.get_classification_augmentation
   tiatoolbox.models.training.get_segmentation_augmentation
   tiatoolbox.models.training.get_annotation_augmentation
   tiatoolbox.models.training.load_trained_model
