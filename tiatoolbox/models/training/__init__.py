"""Training utilities for TIAToolbox models."""

from tiatoolbox.models.training.augmentations import (
    TrainingAugmentationPreset,
    get_annotation_augmentation,
    get_classification_augmentation,
    get_segmentation_augmentation,
)
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    load_model_state_dict,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    TrainerConfig,
)
from tiatoolbox.models.training.datasets import (
    PatchFolderClassificationDataset,
    PatchAnnotationDataset,
    PatchMaskPairDataset,
    SlideAnnotationPatchDataset,
)
from tiatoolbox.models.training.kongnet import (
    build_kongnet_dense_heads,
    build_kongnet_training_task,
)
from tiatoolbox.models.training.samplers import (
    ClassBalancedIndexSampler,
    generate_slide_patch_coordinates,
)
from tiatoolbox.models.training.targets import (
    BinaryDiskTargetBuilder,
    BoundaryTargetBuilder,
    CompositeTargetBuilder,
    CoverageClassTargetBuilder,
    GaussianHeatmapTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PresenceTargetBuilder,
    SpatialTargetKind,
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
from tiatoolbox.models.training.utils import (
    stratified_split_indices,
)

__all__ = [
    "TrainingAugmentationPreset",
    "BinaryDiskTargetBuilder",
    "BoundaryTargetBuilder",
    "CheckpointConfig",
    "ClassificationTask",
    "CompositeTargetBuilder",
    "CoverageClassTargetBuilder",
    "DenseHeadSpec",
    "GaussianHeatmapTargetBuilder",
    "MaskTargetBuilder",
    "MultiLabelTargetBuilder",
    "PatchAnnotationDataset",
    "PatchFolderClassificationDataset",
    "PatchMaskPairDataset",
    "SlideAnnotationPatchDataset",
    "ClassBalancedIndexSampler",
    "PresenceTargetBuilder",
    "SpatialTargetKind",
    "StackedSpatialTargetSpec",
    "StackedTargetBuilder",
    "SegmentationTask",
    "StructuredDenseTask",
    "TargetBuilderABC",
    "Trainer",
    "TrainerConfig",
    "TrainingTaskABC",
    "build_kongnet_dense_heads",
    "build_kongnet_training_task",
    "extract_model_state_dict",
    "load_checkpoint",
    "load_model_state_dict",
    "save_checkpoint",
    "save_model_weights",
    "set_seed",
    "generate_slide_patch_coordinates",
    "get_annotation_augmentation",
    "get_classification_augmentation",
    "get_segmentation_augmentation",
    "stratified_split_indices",
]
