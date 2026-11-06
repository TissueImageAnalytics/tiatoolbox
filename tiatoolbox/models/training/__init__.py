"""Single-device training with annotation-aware datasets and explicit tasks.

Start with ``Trainer``, ``TrainerConfig`` and a classification or segmentation
task. Use ``StructuredDenseTask`` for multiple heads. Standard PyTorch optimizers,
DataLoaders and Subsets remain the data/optimization interface; ``split_indices``
keeps split assignment independent of transforms, and ``ModelExport`` with
``load_trained_model`` hands trained models to inference engines. See the
Training guide for input/loss contracts, extension points and resume.
"""

from tiatoolbox.models.training.augmentations import (
    TrainingAugmentationPreset,
    get_annotation_augmentation,
    get_classification_augmentation,
    get_segmentation_augmentation,
)
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    TrainerConfig,
)
from tiatoolbox.models.training.datasets import (
    PatchAnnotationDataset,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    SlideAnnotationPatchDataset,
)
from tiatoolbox.models.training.export import ModelExport, load_trained_model
from tiatoolbox.models.training.samplers import (
    ClassBalancedIndexSampler,
    GroupedBlockSampler,
    generate_slide_patch_coordinates,
)
from tiatoolbox.models.training.targets import (
    AnnotationQuery,
    BinaryDiskTargetBuilder,
    BoundaryTargetBuilder,
    CompositeTargetBuilder,
    CoverageClassTargetBuilder,
    GaussianHeatmapTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PresenceTargetBuilder,
    StackedSpatialTargetSpec,
    StackedTargetBuilder,
    TargetBuilderABC,
)
from tiatoolbox.models.training.tasks import (
    ClassificationTask,
    DenseHeadSpec,
    SegmentationTask,
    StructuredDenseTask,
    TrainingTaskABC,
)
from tiatoolbox.models.training.trainer import (
    Trainer,
    set_seed,
)
from tiatoolbox.models.training.utils import split_indices

__all__ = [
    "AnnotationQuery",
    "BinaryDiskTargetBuilder",
    "BoundaryTargetBuilder",
    "CheckpointConfig",
    "ClassBalancedIndexSampler",
    "ClassificationTask",
    "CompositeTargetBuilder",
    "CoverageClassTargetBuilder",
    "DenseHeadSpec",
    "GaussianHeatmapTargetBuilder",
    "GroupedBlockSampler",
    "MaskTargetBuilder",
    "ModelExport",
    "MultiLabelTargetBuilder",
    "PatchAnnotationDataset",
    "PatchFolderClassificationDataset",
    "PatchMaskPairDataset",
    "PresenceTargetBuilder",
    "SegmentationTask",
    "SlideAnnotationPatchDataset",
    "StackedSpatialTargetSpec",
    "StackedTargetBuilder",
    "StructuredDenseTask",
    "TargetBuilderABC",
    "Trainer",
    "TrainerConfig",
    "TrainingAugmentationPreset",
    "TrainingTaskABC",
    "generate_slide_patch_coordinates",
    "get_annotation_augmentation",
    "get_classification_augmentation",
    "get_segmentation_augmentation",
    "load_trained_model",
    "set_seed",
    "split_indices",
]
