"""Training utilities for TIAToolbox models."""

from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    DataLoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainerConfig,
)
from tiatoolbox.models.training.datasets import (
    PatchFolderClassificationDataset,
    PatchAnnotationDataset,
    PatchMaskPairDataset,
    SlideAnnotationPatchDataset,
    create_dataset,
)
from tiatoolbox.models.training.samplers import (
    ClassBalancedIndexSampler,
    generate_slide_patch_coordinates,
)
from tiatoolbox.models.training.targets import (
    CoverageClassTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PresenceTargetBuilder,
    TargetBuilderABC,
)
from tiatoolbox.models.training.tasks import (
    ClassificationTask,
    SegmentationTask,
    TrainingTaskABC,
)
from tiatoolbox.models.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    create_task,
    set_seed,
)

__all__ = [
    "CheckpointConfig",
    "ClassificationTask",
    "DataLoaderConfig",
    "CoverageClassTargetBuilder",
    "MaskTargetBuilder",
    "MultiLabelTargetBuilder",
    "OptimizerConfig",
    "PatchAnnotationDataset",
    "PatchFolderClassificationDataset",
    "PatchMaskPairDataset",
    "SlideAnnotationPatchDataset",
    "ClassBalancedIndexSampler",
    "PresenceTargetBuilder",
    "SchedulerConfig",
    "SegmentationTask",
    "TaskConfig",
    "TargetBuilderABC",
    "Trainer",
    "TrainerConfig",
    "TrainingTaskABC",
    "create_dataset",
    "create_optimizer",
    "create_scheduler",
    "create_task",
    "extract_model_state_dict",
    "load_checkpoint",
    "save_checkpoint",
    "save_model_weights",
    "set_seed",
    "generate_slide_patch_coordinates",
]
