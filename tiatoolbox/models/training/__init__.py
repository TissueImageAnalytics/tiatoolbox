"""Training utilities for TIAToolbox models."""

from tiatoolbox.models.training.artifact import (
    EngineConfigSpec,
    TrainingArtifactManifest,
    ioconfig_from_dict,
    ioconfig_to_dict,
    load_training_artifact,
)
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
    PatchAnnotationDataset,
    PatchFolderClassificationDataset,
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
    SpatialTargetSpec,
    StackedSpatialTargetSpec,
    StackedTargetBuilder,
    TargetBuilderABC,
    TargetType,
)
from tiatoolbox.models.training.tasks import (
    ClassificationTargetMode,
    ClassificationTask,
    DenseHeadSpec,
    DenseLossMode,
    DenseMetricName,
    DenseTargetMode,
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
    "BinaryDiskTargetBuilder",
    "BoundaryTargetBuilder",
    "CheckpointConfig",
    "ClassBalancedIndexSampler",
    "ClassificationTargetMode",
    "ClassificationTask",
    "CompositeTargetBuilder",
    "CoverageClassTargetBuilder",
    "DenseHeadSpec",
    "DenseLossMode",
    "DenseMetricName",
    "DenseTargetMode",
    "EngineConfigSpec",
    "GaussianHeatmapTargetBuilder",
    "MaskTargetBuilder",
    "MultiLabelTargetBuilder",
    "PatchAnnotationDataset",
    "PatchFolderClassificationDataset",
    "PatchMaskPairDataset",
    "PresenceTargetBuilder",
    "SegmentationTask",
    "SlideAnnotationPatchDataset",
    "SpatialTargetKind",
    "SpatialTargetSpec",
    "StackedSpatialTargetSpec",
    "StackedTargetBuilder",
    "StructuredDenseTask",
    "TargetBuilderABC",
    "TargetType",
    "Trainer",
    "TrainerConfig",
    "TrainingArtifactManifest",
    "TrainingAugmentationPreset",
    "TrainingTaskABC",
    "build_kongnet_dense_heads",
    "build_kongnet_training_task",
    "extract_model_state_dict",
    "generate_slide_patch_coordinates",
    "get_annotation_augmentation",
    "get_classification_augmentation",
    "get_segmentation_augmentation",
    "ioconfig_from_dict",
    "ioconfig_to_dict",
    "load_checkpoint",
    "load_model_state_dict",
    "load_training_artifact",
    "save_checkpoint",
    "save_model_weights",
    "set_seed",
    "stratified_split_indices",
]
