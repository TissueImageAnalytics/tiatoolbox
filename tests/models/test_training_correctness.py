"""Regression tests for training reductions, resource ownership, and continuation."""

# Pickle inputs below are created within the same test from local dataset objects.
# ruff: noqa: S301

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from typing import BinaryIO

import copy
import itertools
import pickle
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image
from shapely.geometry import LineString, MultiPolygon, Point, box
from torch import nn
from torch.nn import functional
from torch.overrides import TorchFunctionMode
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

from tiatoolbox.annotation import Annotation, DictionaryStore, SQLiteStore
from tiatoolbox.models.models_abc import (
    ModelABC,
    load_model_state_dict,
    unwrap_model,
)
from tiatoolbox.models.training import (
    BoundaryTargetBuilder,
    CheckpointConfig,
    ClassificationTask,
    CoverageClassTargetBuilder,
    DenseHeadSpec,
    GroupedBlockSampler,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PatchAnnotationDataset,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    PresenceTargetBuilder,
    SegmentationTask,
    SlideAnnotationPatchDataset,
    StructuredDenseTask,
    Trainer,
    TrainerConfig,
    get_classification_augmentation,
    get_segmentation_augmentation,
)
from tiatoolbox.models.training.augmentations import (
    _RandomQuarterTurn,
    _StainAugmentation,
    _TorchvisionImageTransform,
    _TorchvisionSpatialPairTransform,
    _TrainingAffine,
)
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    save_checkpoint,
)
from tiatoolbox.models.training.config import resolve_trainer_amp
from tiatoolbox.models.training.datasets import (
    _ensure_tensor_image,
    _ensure_tensor_mask,
    _load_image,
    _load_mask,
)
from tiatoolbox.models.training.tasks import TrainingTaskABC
from tiatoolbox.wsicore.wsireader import TIFFWSIReader, VirtualWSIReader


class MeanSquaredTask(TrainingTaskABC):
    """Exercise the mean-over-samples contract for custom tasks."""

    def loss_components(
        self, output: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return mean squared error over samples."""
        return {"loss": (output - targets).square().mean()}


def make_trainer(
    tmp_path: Path,
    model: nn.Module,
    task: TrainingTaskABC,
    loader: DataLoader,
    **kwargs: object,
) -> Trainer:
    """Build a CPU trainer with a fixed, inspectable optimizer."""
    return Trainer(
        model,
        task,
        torch.optim.SGD(model.parameters(), lr=0.1),
        loader,
        config=TrainerConfig(
            device="cpu",
            amp=False,
            monitor="train_loss",
            output_dir=tmp_path,
            log_every_n_steps=0,
            **kwargs,
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )


@pytest.mark.parametrize(("batch_size", "accum"), [(1, 4), (2, 2), (2, 4)])
@pytest.mark.parametrize("targets", [[1.0, 1.0, 1.0], [0.0, 0.0, 3.0]])
def test_accumulation_matches_full_batch(
    tmp_path: Path, batch_size: int, accum: int, targets: list[float]
) -> None:
    """Incomplete windows and smaller final batches use the actual sample count."""
    data = TensorDataset(torch.ones(3, 1), torch.tensor(targets).reshape(3, 1))
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    expected = copy.deepcopy(model)
    full = make_trainer(
        tmp_path, expected, MeanSquaredTask(), DataLoader(data, batch_size=3)
    )
    micro = make_trainer(
        tmp_path,
        model,
        MeanSquaredTask(),
        DataLoader(data, batch_size=batch_size),
        grad_accum_steps=accum,
    )
    assert micro._run_epoch(micro.train_loader, training=True) == pytest.approx(
        full._run_epoch(full.train_loader, training=True)
    )
    torch.testing.assert_close(model.weight, expected.weight)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_masked_multihead_accumulation(tmp_path: Path, batch_size: int) -> None:
    """Each head uses its own valid-element denominator for loss and gradients."""
    targets = torch.tensor(
        [
            [[1.0, -100], [-100, -100]],
            [[0.0, 1], [-100, -100]],
            [[-100.0, -100], [3, 2]],
        ]
    ).reshape(3, 2, 1, 2)
    data = [
        {"image": torch.ones(1, 1, 2), "target": {"a": target[:1], "b": target[1:]}}
        for target in targets
    ]
    heads = [
        DenseHeadSpec("a", loss="bce_with_logits", channel_slice=0, loss_weight=2),
        DenseHeadSpec("b", loss="mse", channel_slice=1),
    ]
    model = nn.Conv2d(1, 2, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    expected = copy.deepcopy(model)
    micro = make_trainer(
        tmp_path,
        model,
        StructuredDenseTask(heads),
        DataLoader(data, batch_size=batch_size),
        grad_accum_steps=4,
    )
    full = make_trainer(
        tmp_path, expected, StructuredDenseTask(heads), DataLoader(data, batch_size=3)
    )
    assert micro._run_epoch(micro.train_loader, training=True) == pytest.approx(
        full._run_epoch(full.train_loader, training=True)
    )
    for value, reference in zip(
        model.parameters(), expected.parameters(), strict=False
    ):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize(
    "kind", ["classification", "multilabel", "segmentation", "binary", "regression"]
)
def test_epoch_metrics_do_not_depend_on_batching(tmp_path: Path, kind: str) -> None:
    """Counts and error sums agree with evaluating the whole dataset at once."""
    logits = torch.tensor([[8.0, -8], [-8.0, 8], [8.0, -8]])
    targets = torch.tensor([0, 1, -100])
    if kind == "classification":
        task = ClassificationTask()
    elif kind == "multilabel":
        task = ClassificationTask(loss="bce_with_logits")
        targets = torch.tensor([[1, -100], [0, 1], [-100, -100]])
    else:
        logits = logits.reshape(3, 2, 1, 1)
        if kind == "segmentation":
            task = SegmentationTask()
            targets = targets.reshape(3, 1, 1)
        else:
            task = StructuredDenseTask(
                [
                    DenseHeadSpec(
                        "head",
                        loss="mse" if kind == "regression" else "bce_with_logits",
                    )
                ]
            )
            targets = torch.tensor([[1, -100], [0, 1], [-100, -100]]).reshape(
                3, 2, 1, 1
            )

    class Lookup(nn.Module):
        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return logits[indices[:, 0].long()]

    model = Lookup()
    model.register_parameter("unused", nn.Parameter(torch.zeros(())))
    data = TensorDataset(torch.arange(3).reshape(3, 1), targets)
    trainer = make_trainer(tmp_path, model, task, DataLoader(data, batch_size=1))
    metrics = trainer._run_epoch(trainer.train_loader, training=False)
    expected = {
        **task.compute_metrics(logits, targets),
        "loss": float(task.compute_loss(logits, targets)),
    }
    assert metrics == pytest.approx(expected)
    if kind == "classification":
        assert metrics["f1"] == 1.0


@pytest.mark.parametrize("dense", [False, True])
def test_all_ignored_targets_do_not_update_parameters(
    tmp_path: Path, *, dense: bool
) -> None:
    """Ignored supervision produces finite zero loss and no weight-decay update."""
    model = nn.Conv2d(1, 2, 1) if dense else nn.Linear(1, 2)
    images = torch.ones(2, 1, 2, 2) if dense else torch.ones(2, 1)
    targets = torch.full((2, 2, 2) if dense else (2,), -100)
    task = SegmentationTask() if dense else ClassificationTask()
    output = model(images)
    loss = task.compute_loss(output, targets)
    assert float(loss) == 0
    loss.backward()
    assert all(
        torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters()
    )
    trainer = make_trainer(
        tmp_path, model, task, DataLoader(TensorDataset(images, targets))
    )
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    before = copy.deepcopy(model.state_dict())
    assert trainer._run_epoch(trainer.train_loader, training=True)["loss"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert not trainer.optimizer.state


def test_half_precision_loss_reduction_is_finite() -> None:
    """Large FP16 maps reduce their loss in float32 without overflowing."""
    task = StructuredDenseTask([DenseHeadSpec("mask", loss="cross_entropy")])
    logits = torch.zeros(2, 2, 256, 256, dtype=torch.float16)
    targets = torch.zeros(2, 256, 256, dtype=torch.long)
    loss = task.loss_components(logits, targets)["mask"]
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(np.log(2), rel=1e-3)


def test_trainer_does_not_recompute_dense_losses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Head-loss reporting reuses the loss from the forward pass."""
    model = nn.Conv2d(1, 2, 1)
    task = StructuredDenseTask([DenseHeadSpec("mask", loss="cross_entropy")])
    original = task._compute_head_loss
    calls = []

    def compute(*args: object) -> tuple[torch.Tensor, str]:
        calls.append(None)
        return original(*args)

    monkeypatch.setattr(task, "_compute_head_loss", compute)
    loader = DataLoader(
        TensorDataset(torch.ones(3, 1, 2, 2), torch.zeros(3, 2, 2, dtype=torch.long))
    )
    trainer = make_trainer(tmp_path, model, task, loader)
    trainer._run_epoch(loader, training=True)
    assert len(calls) == len(loader)


@pytest.mark.parametrize("method", ["exact", "rasterized"])
def test_coverage_requires_positive_overlap(method: str) -> None:
    """Zero threshold does not turn missing classes into positive labels."""
    bounds, shape = (0, 0, 32, 32), (32, 32)
    annotations = {"a": Annotation(box(1, 1, 10, 10), {"class": "a"})}
    assert (
        PresenceTargetBuilder(overlap_method=method).build_target({}, bounds, shape)
        == 0
    )
    builder = MultiLabelTargetBuilder(
        class_mapping={"a": 0, "b": 1}, overlap_method=method
    )
    np.testing.assert_array_equal(
        builder.build_target(annotations, bounds, shape), [1, 0]
    )
    annotations["b"] = Annotation(box(1, 1, 10, 10), {"class": "b"})
    np.testing.assert_array_equal(
        builder.build_target(annotations, bounds, shape), [1, 1]
    )
    outside = {"a": Annotation(box(40, 40, 50, 50), {"class": "a"})}
    assert (
        CoverageClassTargetBuilder(
            class_mapping={"a": 1}, default_label=-1, overlap_method=method
        ).build_target(outside, bounds, shape)
        == -1
    )


def test_holes_preserve_other_annotations_and_crop_edges_are_not_boundaries() -> None:
    """Polygon holes and cropped polygon contours retain their geometric meaning."""
    annotations = {
        "inside": Annotation(box(12, 12, 20, 20), {"class": "a"}),
        "donut": Annotation(
            box(0, 0, 32, 32).difference(box(8, 8, 24, 24)), {"class": "b"}
        ),
    }
    mask = MaskTargetBuilder(class_mapping={"a": 1, "b": 2}).build_target(
        annotations, (0, 0, 32, 32), (32, 32)
    )
    assert mask[16, 16] == 1
    enclosing = {"large": Annotation(box(-10, -10, 42, 42))}
    boundary = BoundaryTargetBuilder().build_target(enclosing, (0, 0, 32, 32), (32, 32))
    assert not boundary.any()


@pytest.mark.parametrize(
    "bounds",
    [
        (16, 4, 20, 12),
        (4, 16, 12, 20),
        (-4, 4, 0, 12),
        (4, -4, 12, 0),
        (16, 16, 20, 20),
    ],
)
def test_polygon_touching_patch_does_not_create_foreground(
    bounds: tuple[int, int, int, int],
) -> None:
    """A zero-area polygon intersection must not become a line/point target."""
    store = SQLiteStore()
    try:
        store.append(Annotation(box(*bounds)))
        for builder in (
            MaskTargetBuilder(),
            PresenceTargetBuilder(overlap_method="rasterized"),
        ):
            target = builder.create_target(
                store=store, patch_bounds=(0, 0, 16, 16), output_shape=(16, 16)
            )
            assert not np.any(target)
    finally:
        store.close()


def test_clipping_preserves_polygon_area_and_explicit_lines_and_points() -> None:
    """Filter collapsed polygon parts without dropping legitimate target types."""
    inside = box(2, 2, 4, 4)
    touching = box(16, 8, 20, 12)
    builder = MaskTargetBuilder()
    expected = builder.build_target(
        {"inside": Annotation(inside)}, (0, 0, 16, 16), (16, 16)
    )
    actual = builder.build_target(
        {"mixed": Annotation(MultiPolygon([inside, touching]))},
        (0, 0, 16, 16),
        (16, 16),
    )
    np.testing.assert_array_equal(actual, expected)
    for geometry in (LineString([(3, 8), (12, 8)]), Point(8, 8)):
        target = builder.build_target(
            {"explicit": Annotation(geometry)}, (0, 0, 16, 16), (16, 16)
        )
        assert target[8, 8] == 1


@pytest.mark.parametrize("channel", [-1, -3, 0, 2])
def test_dense_integer_channel_selection_preserves_gradients(channel: int) -> None:
    """Integer selection follows tensor indexing while retaining a channel axis."""
    output = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    output.requires_grad_()
    task = StructuredDenseTask(
        [DenseHeadSpec("selected", channel_slice=channel, loss="mse")]
    )
    target = torch.zeros(1, 1, 2, 2)
    loss = task.compute_loss(output, target)
    torch.testing.assert_close(loss, output[:, channel].square().mean())
    loss.backward()
    expected_grad = torch.zeros_like(output)
    expected_grad[:, channel] = 2 * output.detach()[:, channel] / target.numel()
    torch.testing.assert_close(output.grad, expected_grad)


def test_disabled_dense_metrics_skip_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling a head's metrics avoids reductions but retains its objective."""
    task = StructuredDenseTask(
        [
            DenseHeadSpec("disabled", loss="mse", metrics=()),
            DenseHeadSpec("enabled", loss="mse", metrics=("mae",)),
        ],
    )
    original = task._head_metric_state

    def checked(head: DenseHeadSpec, *args: object) -> object:
        assert head.name == "enabled"
        return original(head, *args)

    monkeypatch.setattr(task, "_head_metric_state", checked)
    output = {name: torch.ones(1, 1, 2, 2) for name in ("enabled", "disabled")}
    targets = {name: torch.zeros_like(value) for name, value in output.items()}
    assert task.compute_loss(output, targets).item() == 2
    assert task.compute_metrics(output, targets) == {"enabled_mae": 1.0}
    task.reset_epoch_state(training=True)
    task.update_epoch_state(output, targets)
    assert task.compute_epoch_metrics() == {"enabled_mae": 1.0}


@pytest.mark.parametrize("seed", [None, 7])
def test_deterministic_backend_configuration_does_not_require_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int | None
) -> None:
    """The deterministic option applies independently of RNG initialization."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    make_trainer(
        tmp_path,
        nn.Linear(2, 2),
        ClassificationTask(),
        DataLoader(TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))),
        seed=seed,
        deterministic=True,
    )
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_continuous_targets_use_zero_affine_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targets share geometry but do not inherit white RGB padding."""
    op = _TrainingAffine(degrees=0)
    monkeypatch.setattr(op, "get_params", lambda *_args: (0, (3, 0), 1, (0, 0)))
    transform = _TorchvisionSpatialPairTransform(
        [op], {"heat": "image", "mask": "mask"}
    )
    image, targets = transform(
        np.zeros((16, 16, 3), np.uint8),
        {"heat": np.ones((16, 16), np.float32), "mask": np.ones((16, 16), np.uint8)},
    )
    assert np.all(image[:, :3] == 255)
    assert np.all(targets["heat"][:, :3] == 0)
    assert np.all(targets["heat"][:, 3:] == 1)
    assert np.all(targets["mask"][:, :3] == 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_image_conversion_keeps_values_and_channels(
    dtype: type[np.generic],
) -> None:
    """Conversion never rescales values or adds channels, for any layout."""
    image = np.full((7, 9, 1), 255, dtype=dtype)
    array_result = _ensure_tensor_image(image)
    chw_tensor = torch.from_numpy(image).permute(2, 0, 1)
    torch.testing.assert_close(array_result, _ensure_tensor_image(chw_tensor))
    hwc_tensor = torch.from_numpy(image)
    torch.testing.assert_close(
        array_result, _ensure_tensor_image(hwc_tensor, channels_last=True)
    )
    assert array_result.shape == (1, 7, 9)
    assert array_result.dtype == torch.float32
    assert float(array_result[0, 0, 0]) == 255


def test_mask_loading_preserves_labels_and_rejects_colour_codes(tmp_path: Path) -> None:
    """High label values and palette indices survive image decoding."""
    path = tmp_path / "labels.png"
    Image.fromarray(np.full((8, 8), 300, np.uint16)).save(path)
    assert np.all(_load_mask(path) == 300)
    assert np.all(_load_image(path) == 300)
    palette = Image.fromarray(np.full((8, 8), 3, np.uint8)).convert("P")
    palette.save(path)
    assert np.all(_load_mask(path) == 3)
    Image.fromarray(np.full((8, 8, 3), [255, 0, 0], np.uint8)).save(path)
    with pytest.raises(ValueError, match="colour-coded"):
        _ensure_tensor_mask(_load_mask(path))
    with pytest.raises(ValueError, match="colour-coded"):
        _ensure_tensor_mask(_load_mask(path)[:1])
    dataset = PatchMaskPairDataset(
        images=[path],
        masks=[path],
        target_transform=lambda mask: (mask[..., 0] == 255).astype(np.int64),
    )
    assert torch.all(dataset[0]["target"] == 1)


def test_dataset_preserves_borrowed_memory_store_and_reader() -> None:
    """In-memory data and reader metadata survive direct use and serialization."""
    store = SQLiteStore()
    try:
        store.append(Annotation(box(0, 0, 16, 16)))
        image = np.zeros((32, 32, 3), np.uint8)
        reader = VirtualWSIReader(image, mpp=0.5, power=20)
        with SlideAnnotationPatchDataset(
            [reader], store, MaskTargetBuilder(), 16, resolution=0.5, units="mpp"
        ) as dataset:
            assert dataset._get_reader(0) is reader
            assert dataset._get_store(0) is store
            expected = dataset[0]
            with pickle.loads(pickle.dumps(dataset)) as restored:
                torch.testing.assert_close(restored[0]["target"], expected["target"])
        assert len(store) == 1
        assert reader.info.mpp[0] == 0.5
        with PatchAnnotationDataset([image], store, PresenceTargetBuilder()) as patches:
            assert patches[0]["target"] == 1
    finally:
        store.close()


@pytest.mark.parametrize("mask_kind", ["otsu", "str", "path", "store"])
def test_annotation_dataset_accepts_mask_specifications(
    tmp_path: Path, mask_kind: str
) -> None:
    """Mask method names and annotation databases retain their interpretation."""
    path = tmp_path / "mask.db"
    store = SQLiteStore(path)
    try:
        store.append(Annotation(box(0, 0, 32, 32)))
        mask = {"otsu": "otsu", "str": str(path), "path": path, "store": store}[
            mask_kind
        ]
        reader = VirtualWSIReader(
            np.full((32, 32, 3), 100, np.uint8), mpp=0.5, power=20
        )
        with SlideAnnotationPatchDataset(
            [reader], store, MaskTargetBuilder(), 16, input_masks=mask
        ) as dataset:
            assert len(dataset) > 0
            assert dataset[0]["image"].shape == (3, 16, 16)
    finally:
        store.close()


def test_dataset_cache_is_bounded_and_closes_owned_stores(tmp_path: Path) -> None:
    """Visiting many per-patch databases does not keep every database open."""
    paths = []
    for index in range(20):
        path = tmp_path / f"{index}.db"
        store = SQLiteStore(path)
        store.close()
        paths.append(path)
    with PatchAnnotationDataset(
        [np.zeros((4, 4, 3), np.uint8)] * 20, paths, PresenceTargetBuilder()
    ) as dataset:
        for index in range(20):
            dataset[index]
        assert len(dataset._store_cache) <= 16
        owned = list(dataset._store_cache.values())
    assert all(not store.cons for store in owned)
    for path in paths:
        path.unlink()


def test_configured_tiff_reader_survives_worker_serialization(tmp_path: Path) -> None:
    """A worker retains the selected TIFF series, metadata, and postprocessing."""
    path = tmp_path / "slide.tiff"
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        writer.write(np.zeros((32, 32, 3), np.uint8), tile=(16, 16), photometric="rgb")
        writer.write(
            np.full((32, 32, 3), 100, np.uint8), tile=(16, 16), photometric="rgb"
        )
    reader = TIFFWSIReader(path, series=1, mpp=(0.5, 0.6), power=20, post_proc=None)
    store = SQLiteStore()
    try:
        with (
            SlideAnnotationPatchDataset(
                [reader], store, MaskTargetBuilder(), 16
            ) as dataset,
            pickle.loads(pickle.dumps(dataset)) as restored,
        ):
            worker_reader = restored._get_reader(0)
            assert worker_reader.series_n == 1
            np.testing.assert_array_equal(worker_reader.info.mpp, [0.5, 0.6])
            torch.testing.assert_close(restored[0]["image"], dataset[0]["image"])
            assert float(restored[0]["image"].mean()) == pytest.approx(100)
        assert not reader.tiff.filehandle.closed
    finally:
        store.close()
        reader.close()
    path.unlink()


def test_memory_store_snapshot_preserves_shared_references() -> None:
    """Repeated store inputs produce a single in-memory snapshot when pickled."""
    store = SQLiteStore()
    try:
        with (
            PatchAnnotationDataset(
                [np.zeros((4, 4, 3), np.uint8)] * 20, store, PresenceTargetBuilder()
            ) as dataset,
            pickle.loads(pickle.dumps(dataset)) as restored,
        ):
            assert all(
                spec is restored.annotation_store_specs[0]
                for spec in restored.annotation_store_specs
            )
    finally:
        store.close()


def test_sqlite_close_does_not_reopen_a_closed_database(tmp_path: Path) -> None:
    """Repeated close, including destruction, must not recreate a removed file."""
    path = tmp_path / "closed.db"
    store = SQLiteStore(path)
    store.append(Annotation(box(0, 0, 2, 2)))
    store.close()
    reopened = SQLiteStore(path)
    try:
        assert len(reopened) == 1
    finally:
        reopened.close()
    path.unlink()
    store.close()
    assert not path.exists()


def test_stain_augmentation_uses_the_pipeline_seed() -> None:
    """The adapter preserves seeded augmentation with the standalone augmentor."""
    transform = _StainAugmentation(sigma1=0.4, sigma2=0.2, p=1)
    pipeline = _TorchvisionImageTransform(v2.Compose([transform]))
    image = np.random.default_rng(8).integers(40, 190, (16, 16, 3), dtype=np.uint8)
    pipeline.set_random_seed(123)
    expected = pipeline(image)
    pipeline.set_random_seed(123)
    np.testing.assert_array_equal(pipeline(image), expected)
    pipeline.set_random_seed(456)
    assert not np.array_equal(pipeline(image), expected)
    transform.p = 0
    np.testing.assert_array_equal(pipeline(image), image)


@pytest.mark.parametrize(("workers", "persistent"), [(0, False), (2, False), (2, True)])
def test_resume_matches_uninterrupted_training(
    tmp_path: Path, workers: int, *, persistent: bool
) -> None:
    """Shuffle, dropout, presets, and persistent workers continue reproducibly."""
    root = tmp_path / "images" / "class_a"
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(8):
        np.save(root / f"{index}.npy", rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))

    def build(output_dir: Path, epochs: int) -> Trainer:
        torch.manual_seed(9)
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(192, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2)
        )
        preset = get_classification_augmentation("heavy")
        dataset = PatchFolderClassificationDataset(
            root.parent, image_transform=preset.image_transform
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=workers,
            persistent_workers=persistent,
        )
        trainer = make_trainer(
            output_dir, model, ClassificationTask(), loader, seed=13, max_epochs=epochs
        )
        trainer.checkpoint_config = CheckpointConfig()
        return trainer

    full = build(tmp_path / "full", 2)
    first = build(tmp_path / "split", 1)
    resumed = build(tmp_path / "resumed", 2)
    try:
        # Trainer construction seeds globals; reseed just before each run too.
        torch.manual_seed(13)
        full.fit()
        torch.manual_seed(13)
        first.fit()
        resumed.fit(first.output_dir / "last.ckpt")
        # Wall-clock timing is the only row entry expected to differ.
        assert [
            {k: v for k, v in row.items() if k != "epoch_time"}
            for row in resumed.history
        ] == [
            {k: v for k, v in row.items() if k != "epoch_time"} for row in full.history
        ]
        for value, expected in zip(
            resumed.model.parameters(), full.model.parameters(), strict=False
        ):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert (resumed.output_dir / "best_model_weights.pth").is_file()
    finally:
        for trainer in (full, first, resumed):
            iterator = getattr(trainer.train_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()


def test_resume_retains_patience_and_previous_best_weights(tmp_path: Path) -> None:
    """A new output directory retains the best weights even without improvement.

    With a zero learning rate only the first epoch improves, so patience two
    stops after the third epoch, including when that epoch follows a resume.
    """
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    )

    def build(directory: Path, epochs: int) -> Trainer:
        model = nn.Linear(1, 2)
        trainer = make_trainer(
            directory,
            model,
            ClassificationTask(),
            loader,
            max_epochs=epochs,
            early_stopping_patience=2,
        )
        trainer.optimizer.param_groups[0]["lr"] = 0
        trainer.checkpoint_config = CheckpointConfig()
        return trainer

    first = build(tmp_path / "first", 2)
    first.fit()
    resumed = build(tmp_path / "second", 5)
    resumed.fit(first.output_dir / "last.ckpt")
    assert len(resumed.history) == 3
    assert resumed.best_epoch == 1
    best = load_checkpoint(resumed.output_dir / "best_model_weights.pth")
    first_best = load_checkpoint(first.output_dir / "best_model_weights.pth")
    torch.testing.assert_close(best, first_best)
    stopped = build(tmp_path / "third", 6)
    assert len(stopped.fit(resumed.output_dir / "last.ckpt")) == 3


def test_scheduler_steps_on_optimizer_updates(tmp_path: Path) -> None:
    """OneCycleLR advances per accumulation window, including a short tail."""
    model = nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(5, 1), torch.zeros(5, dtype=torch.long))
    )
    trainer = make_trainer(
        tmp_path,
        model,
        ClassificationTask(),
        loader,
        max_epochs=2,
        grad_accum_steps=2,
        scheduler_interval="step",
    )
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer, max_lr=0.1, total_steps=6
    )
    trainer.fit()
    assert trainer.scheduler.last_epoch == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_amp_overflow_skips_step_scheduler(tmp_path: Path) -> None:
    """A skipped AMP optimizer update must not advance a per-step scheduler."""
    model = nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)), batch_size=2
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, total_steps=2
    )
    trainer = Trainer(
        model,
        ClassificationTask(),
        optimizer,
        loader,
        scheduler=scheduler,
        config=TrainerConfig(
            device="cuda",
            amp=True,
            monitor="train_loss",
            output_dir=tmp_path,
            scheduler_interval="step",
        ),
    )
    before = copy.deepcopy(model.state_dict())
    handle = model.weight.register_hook(lambda gradient: gradient * float("inf"))
    try:
        metrics = trainer._run_epoch(loader, training=True)
    finally:
        handle.remove()
    assert np.isfinite(metrics["loss"])
    assert scheduler.last_epoch == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_resume_rejects_changed_monitor(tmp_path: Path) -> None:
    """Best-score and patience state cannot carry across different objectives."""
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    )
    trainer = make_trainer(tmp_path, nn.Linear(1, 2), ClassificationTask(), loader)
    save_checkpoint(trainer._build_checkpoint_state(1), tmp_path / "last.ckpt")
    trainer.config.monitor = "train_accuracy"
    with pytest.raises(ValueError, match="different `monitor`"):
        trainer.fit(tmp_path / "last.ckpt")


def test_monitor_typo_raises_when_validation_has_run(tmp_path: Path) -> None:
    """Missing validation metrics are distinct from a skipped validation epoch."""
    model = nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    )
    trainer = make_trainer(tmp_path, model, ClassificationTask(), loader)
    trainer.val_loader = loader
    trainer.config.monitor = "val_typo"
    with pytest.raises(KeyError, match="val_typo"):
        trainer.fit()
    with pytest.raises(ValueError, match="monitor_mode"):
        TrainerConfig(monitor_mode="typo")


def test_failed_checkpoint_write_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted save leaves a loadable previous checkpoint and no temp file."""
    path = tmp_path / "last.ckpt"
    save_checkpoint({"epoch": 1}, path)

    def fail(state: dict, stream: BinaryIO) -> None:
        del state
        stream.write(b"incomplete")
        msg = "simulated write failure"
        raise OSError(msg)

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint({"epoch": 2}, path)
    assert torch.load(path)["epoch"] == 1
    assert list(tmp_path.iterdir()) == [path]


class _ForbidDeviceSync(TorchFunctionMode):
    """Fail on tensor operations whose result or shape depends on device values."""

    _reads = frozenset(
        {
            "__bool__",
            "__int__",
            "__float__",
            "item",
            "tolist",
            "numpy",
            "bincount",
            "nonzero",
            "masked_select",
            "unique",
        }
    )

    def __torch_function__(
        self,
        func: Callable,
        types: Iterable[type],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> object:
        """Dispatch the call unless it would read device-side values."""
        del types
        name = getattr(func, "__name__", "")
        if name in self._reads:
            pytest.fail(f"`{name}` synchronises with the device")
        if name in {"__getitem__", "__setitem__"}:
            index = args[1]
            for item in index if isinstance(index, tuple) else (index,):
                if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
                    pytest.fail("Boolean mask indexing synchronises with the device")
        return func(*args, **(kwargs or {}))


_IGNORED_KINDS = [
    "classification",
    "binary",
    "multilabel",
    "segmentation",
    "dense_binary",
    "regression",
]


def _random_supervised_batch(
    kind: str,
) -> tuple[TrainingTaskABC, torch.Tensor, torch.Tensor]:
    """Build a task with random logits and targets, about a third of them ignored."""
    generator = torch.Generator().manual_seed(11)
    shapes = {
        "classification": ((12, 4), (12,), 4),
        "binary": ((12,), (12,), 2),
        "multilabel": ((12, 3), (12, 3), 2),
        "segmentation": ((2, 3, 5, 6), (2, 5, 6), 3),
        "dense_binary": ((2, 2, 5, 6), (2, 2, 5, 6), 2),
        "regression": ((2, 1, 5, 6), (2, 1, 5, 6), None),
    }
    logit_shape, target_shape, classes = shapes[kind]
    logits = torch.randn(logit_shape, generator=generator)
    if classes is None:
        targets = torch.randn(target_shape, generator=generator)
    else:
        targets = torch.randint(classes, target_shape, generator=generator)
    ignored = torch.rand(target_shape, generator=generator) < 0.3
    targets = torch.where(ignored, torch.full_like(targets, -100), targets)
    builders = {
        "classification": ClassificationTask,
        "binary": partial(
            ClassificationTask, loss="bce_with_logits", target_mode="binary"
        ),
        "multilabel": partial(ClassificationTask, loss="bce_with_logits"),
        "segmentation": SegmentationTask,
        "dense_binary": partial(SegmentationTask, loss="bce_with_logits"),
        "regression": partial(
            StructuredDenseTask, [DenseHeadSpec("head", loss="smooth_l1")]
        ),
    }
    return builders[kind](), logits, targets


def _reference_loss(
    kind: str, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Average supervised losses with boolean indexing, as the tasks once did."""
    valid = targets != -100
    if kind == "classification":
        return functional.cross_entropy(logits[valid], targets[valid])
    if kind == "segmentation":
        return functional.cross_entropy(logits, targets, ignore_index=-100)
    if kind == "regression":
        return functional.smooth_l1_loss(logits[valid], targets[valid])
    return functional.binary_cross_entropy_with_logits(
        logits[valid], targets[valid].float()
    )


def _reference_statistics(
    kind: str, logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Accumulate metric statistics with bincount and boolean indexing."""
    valid = targets != -100
    if kind in {"classification", "segmentation", "binary"}:
        if kind == "binary":
            predictions, num_classes = (logits > 0).long(), 2
        else:
            predictions, num_classes = logits.argmax(1), logits.shape[1]
        predictions, labels = predictions[valid], targets[valid]
        tp = torch.bincount(labels[predictions == labels], minlength=num_classes)
        pred = torch.bincount(predictions, minlength=num_classes)
        truth = torch.bincount(labels, minlength=num_classes)
        count = torch.full_like(tp, labels.numel())
        return torch.stack((tp, pred, truth, count)), None
    if kind == "regression":
        errors = (logits - targets)[valid].double()
        return torch.stack(
            (errors.abs().sum(), errors.square().sum(), errors.new_tensor(len(errors)))
        ), None
    predictions, truth = logits > 0, targets > 0.5
    columns = []
    for channel in range(logits.shape[1]):
        chosen = valid[:, channel]
        p, t = predictions[:, channel][chosen], truth[:, channel][chosen]
        columns.append(torch.stack(((p & t).sum(), p.sum(), t.sum(), chosen.sum())))
    subset = None
    if kind == "multilabel":
        rows = [index for index in range(len(valid)) if valid[index].any()]
        matched = [
            bool(((predictions[index] == truth[index]) | ~valid[index]).all())
            for index in rows
        ]
        subset = torch.tensor([sum(matched), len(rows)])
    return torch.stack(columns, dim=1), subset


@pytest.mark.parametrize("kind", _IGNORED_KINDS)
def test_sync_free_reductions_match_indexed_reference(kind: str) -> None:
    """Weighted sums and scatter counts equal boolean indexing and bincount."""
    task, logits, targets = _random_supervised_batch(kind)
    logits.requires_grad_()
    torch.testing.assert_close(
        task.compute_loss(logits, targets), _reference_loss(kind, logits, targets)
    )
    if isinstance(task, ClassificationTask):
        state = task._metric_state(logits, targets)
    else:
        head = task.heads[0]
        state = task._head_metric_state(
            head, logits, targets, task._resolve_head_loss(head, logits)
        )
    totals, subset = _reference_statistics(kind, logits.detach(), targets)
    torch.testing.assert_close(state.totals, totals)
    if subset is not None:
        torch.testing.assert_close(state.subset, subset)

    ignored = torch.full_like(targets, -100)
    loss = task.compute_loss(logits, ignored)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert set(task.compute_metrics(logits.detach(), ignored).values()) == {0.0}


@pytest.mark.parametrize("kind", _IGNORED_KINDS)
def test_batch_loss_and_metric_paths_do_not_read_device_values(kind: str) -> None:
    """After the first forward, losses and epoch statistics never synchronise."""
    task, logits, targets = _random_supervised_batch(kind)
    logits.requires_grad_()
    task.loss_normalizers(targets, logits.shape[0])  # CPU targets, pre-transfer.
    task.compute_loss(logits, targets)  # The first forward may check class ranges.
    task.reset_epoch_state(training=True)
    with _ForbidDeviceSync():
        for batch_targets in (targets, torch.full_like(targets, -100)):
            sum(task.loss_components(logits, batch_targets).values()).backward()
            task.update_epoch_state(logits.detach(), batch_targets)
    assert task.compute_epoch_metrics()


def test_class_index_validation_runs_on_cpu_targets(tmp_path: Path) -> None:
    """Out-of-range labels are reported by ``loss_normalizers`` before transfer."""
    task = ClassificationTask()
    with pytest.raises(ValueError, match="negative class indices"):
        task.loss_normalizers(torch.tensor([0, -1]), 2)
    task.compute_loss(torch.zeros(2, 3), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match=r"outside the valid range \[0, 2\]"):
        task.loss_normalizers(torch.tensor([0, 3]), 2)
    dense = SegmentationTask()
    assert dense.loss_normalizers(torch.tensor([[[0, 7]]]), 1) == {"segmentation": 2}
    dense.compute_loss(torch.zeros(1, 2, 1, 2), torch.tensor([[[0, 1]]]))
    with pytest.raises(ValueError, match=r"outside the valid range \[0, 1\]"):
        dense.loss_normalizers(torch.tensor([[[0, 7]]]), 1)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 2), torch.tensor([0, 5], dtype=torch.long))
    )
    trainer = make_trainer(tmp_path, nn.Linear(2, 2), ClassificationTask(), loader)
    with pytest.raises(ValueError, match="outside the valid range"):
        trainer.fit()


def test_epoch_metrics_are_empty_before_any_epoch() -> None:
    """Tasks start with empty epoch state, before ``reset_epoch_state`` runs."""
    tasks = (
        ClassificationTask(),
        SegmentationTask(),
        StructuredDenseTask([DenseHeadSpec("a"), DenseHeadSpec("b", metrics=())]),
    )
    assert all(task.compute_epoch_metrics() == {} for task in tasks)


def test_early_stopping_counts_consecutive_non_improving_epochs(
    tmp_path: Path,
) -> None:
    """Patience ``N`` stops training once ``N`` monitored epochs fail to improve."""
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    )
    trainer = make_trainer(
        tmp_path,
        nn.Linear(1, 2),
        ClassificationTask(),
        loader,
        max_epochs=6,
        early_stopping_patience=2,
    )
    losses = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def fake_run_epoch(loader: DataLoader, *, training: bool) -> dict[str, float]:
        del loader, training
        return {"loss": next(losses)}

    trainer._run_epoch = fake_run_epoch  # type: ignore[method-assign]
    history = trainer.fit()
    assert [row["train_loss"] for row in history] == [1.0, 2.0, 3.0]
    assert trainer.best_epoch == 1
    assert trainer.patience_counter == 2
    with pytest.raises(ValueError, match="positive integer"):
        TrainerConfig(early_stopping_patience=0)


def test_numpy_metric_values_keep_checkpoints_resumable(tmp_path: Path) -> None:
    """Metrics reported as NumPy scalars are stored as floats in resumable state."""

    class NumpyMetricTask(ClassificationTask):
        """Report epoch metrics as NumPy scalars, as sklearn-based tasks do."""

        def compute_epoch_metrics(self) -> dict[str, float]:
            """Wrap the built-in metrics in NumPy scalar types."""
            metrics = super().compute_epoch_metrics()
            return {key: np.float64(value) for key, value in metrics.items()}

    loader = DataLoader(
        TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2
    )

    def run(max_epochs: int, resume_from: Path | None = None) -> list[dict]:
        model = nn.Linear(2, 2)
        trainer = Trainer(
            model,
            NumpyMetricTask(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            loader,
            config=TrainerConfig(
                max_epochs=max_epochs,
                device="cpu",
                amp=False,
                monitor="train_accuracy",
                monitor_mode="max",
                output_dir=tmp_path,
                log_every_n_steps=0,
            ),
        )
        return trainer.fit(resume_from=resume_from)

    history = run(1)
    assert all(type(value) is float for value in history[0].values())
    assert len(run(2, resume_from=tmp_path / "last.ckpt")) == 2


def test_compiled_model_checkpoints_use_plain_model_keys(tmp_path: Path) -> None:
    """``torch.compile`` and parallel wrappers do not leak into saved weights."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 2))
    compiled = torch.compile(model, backend="eager")
    assert unwrap_model(nn.DataParallel(compiled)) is model
    assert extract_model_state_dict(compiled).keys() == model.state_dict().keys()

    loader = DataLoader(
        TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    Trainer(
        compiled,
        ClassificationTask(),
        torch.optim.SGD(compiled.parameters(), lr=0.1),
        loader,
        config=TrainerConfig(
            device="cpu",
            amp=False,
            monitor="train_loss",
            output_dir=tmp_path,
            log_every_n_steps=0,
        ),
    ).fit()
    plain = nn.Sequential(nn.Flatten(), nn.Linear(2, 2))
    load_model_state_dict(plain, load_checkpoint(tmp_path / "best_model_weights.pth"))
    assert torch.equal(plain[1].weight, model[1].weight)
    recompiled = torch.compile(nn.Sequential(nn.Flatten(), nn.Linear(2, 2)))
    load_model_state_dict(recompiled, load_checkpoint(tmp_path / "last.ckpt"))


@pytest.mark.parametrize("preset", ["classification", "segmentation"])
def test_rectangular_patches_keep_their_shape(preset: str) -> None:
    """Quarter turns never swap the height and width of rectangular patches."""
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    mask = np.zeros((16, 24), dtype=np.int64)
    for seed in range(24):
        if preset == "classification":
            augmentation = get_classification_augmentation("heavy")
            augmentation.image_transform.set_random_seed(seed)
            assert augmentation.image_transform(image).shape == image.shape
        else:
            augmentation = get_segmentation_augmentation("heavy")
            augmentation.pair_transform.set_random_seed(seed)
            out_image, out_mask = augmentation.pair_transform(image, mask)
            assert (out_image.shape, out_mask.shape) == (image.shape, mask.shape)

    square = torch.arange(9).reshape(1, 3, 3)
    outcomes = set()
    for seed in range(32):
        torch.manual_seed(seed)
        outcomes.add(tuple(_RandomQuarterTurn()(square).flatten().tolist()))
    # Square patches still use every multiple of 90 degrees.
    assert len(outcomes) == 4


def test_supplied_class_mapping_allows_absent_classes(tmp_path: Path) -> None:
    """A shared mapping may name classes missing from one split's folder."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / name / "x.png")
    dataset = PatchFolderClassificationDataset(
        tmp_path, class_to_idx={"a": 0, "b": 1, "c": 2}
    )
    assert sorted(dataset.targets) == [0, 1]
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="empty"):
        PatchFolderClassificationDataset(tmp_path)


def test_mask_overlaps_do_not_depend_on_store_or_insertion_order(
    tmp_path: Path,
) -> None:
    """Overlapping annotations paint largest first, whatever the query order."""
    rng = np.random.default_rng(0)
    annotations = []
    for index in range(60):
        x, y = rng.uniform(0, 48, size=2)
        width, height = rng.uniform(4, 20, size=2)
        annotations.append(
            Annotation(box(x, y, x + width, y + height), {"class": index % 3 + 1})
        )
    keys = [f"{index:03d}" for index in range(len(annotations))]
    sqlite = SQLiteStore(tmp_path / "overlaps.db")
    sqlite.append_many(annotations, keys)
    sqlite.commit()
    reversed_store = DictionaryStore()
    reversed_store.append_many(annotations[::-1], keys[::-1])
    builder = MaskTargetBuilder(class_mapping={1: 1, 2: 2, 3: 3})
    for bounds in [(0, 0, 64, 64), (10, 5, 50, 45)]:
        expected = builder.create_target(
            store=sqlite, patch_bounds=bounds, output_shape=(32, 32)
        )
        actual = builder.create_target(
            store=reversed_store, patch_bounds=bounds, output_shape=(32, 32)
        )
        np.testing.assert_array_equal(actual, expected)
    sqlite.close()

    nested = [
        Annotation(box(0, 0, 8, 8), {"class": 1}),
        Annotation(box(2, 2, 4, 4), {"class": 2}),
    ]
    for order in (nested, nested[::-1]):
        store = DictionaryStore()
        store.append_many(order)
        mask = builder.create_target(
            store=store, patch_bounds=(0, 0, 8, 8), output_shape=(8, 8)
        )
        assert mask[3, 3] == 2  # The nested annotation wins.
        assert mask[6, 6] == 1


def test_history_reports_weighted_loss_components(tmp_path: Path) -> None:
    """Several loss components are reported as ``<name>_loss``, summing to loss."""
    samples = [
        {
            "image": torch.ones(1, 2, 2),
            "target": {"a": torch.zeros(1, 2, 2), "b": torch.ones(1, 2, 2)},
        }
        for _ in range(4)
    ]
    task = StructuredDenseTask(
        [
            DenseHeadSpec("a", loss="mse", channel_slice=0),
            DenseHeadSpec("b", loss="mse", channel_slice=1, loss_weight=0.5),
        ]
    )
    model = nn.Conv2d(1, 2, 1)
    loader = DataLoader(samples, batch_size=2)
    metrics = make_trainer(tmp_path, model, task, loader).evaluate(loader)
    with torch.no_grad():
        output = model(torch.ones(1, 1, 2, 2))
    assert metrics["a_loss"] == pytest.approx(float(output[:, 0].square().mean()))
    expected_b = 0.5 * float((output[:, 1] - 1).square().mean())
    assert metrics["b_loss"] == pytest.approx(expected_b)
    assert metrics["loss"] == pytest.approx(metrics["a_loss"] + metrics["b_loss"])

    single = make_trainer(tmp_path, model, SegmentationTask(loss="mse"), loader)
    segmentation_loader = DataLoader(
        [
            {"image": sample["image"], "target": torch.zeros(2, 2, 2)}
            for sample in samples
        ]
    )
    assert not any(
        key.endswith("_loss") for key in single.evaluate(segmentation_loader)
    )


def test_single_batch_metrics_leave_epoch_state_untouched() -> None:
    """``compute_metrics`` evaluates one batch without resetting the epoch."""
    task = ClassificationTask()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    task.reset_epoch_state(training=True)
    task.update_epoch_state(logits, torch.tensor([0, 1]))
    assert task.compute_metrics(logits, torch.tensor([1, 0]))["accuracy"] == 0.0
    assert task.compute_epoch_metrics()["accuracy"] == 1.0


def test_bfloat16_autocast_runs_on_cpu_without_loss_scaling(tmp_path: Path) -> None:
    """bfloat16 mixed precision needs no GradScaler and also works on the CPU."""
    assert resolve_trainer_amp(amp=True, device="cpu", dtype="bfloat16")
    assert not resolve_trainer_amp(amp=True, device="cpu", dtype="float16")
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    model = nn.Linear(3, 2)
    trainer = make_trainer(tmp_path, model, ClassificationTask(), loader)
    trainer.config = TrainerConfig(
        device="cpu", amp=True, amp_dtype="bfloat16", output_dir=tmp_path
    )
    trainer._configure_amp()
    assert trainer.use_amp
    assert not trainer.grad_scaler.is_enabled()
    metrics = trainer._run_epoch(loader, training=True)
    assert np.isfinite(metrics["loss"])
    with pytest.raises(ValueError, match="amp_dtype"):
        TrainerConfig(amp_dtype="float8")


class _EvalScaledLinear(nn.Module):
    """Scale features in eval mode only, as drifting BatchNorm statistics can."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed the layer values beyond float16's range in eval mode."""
        scale = 1.0 if self.training else 1e5
        return self.linear(x * scale) / scale


def test_float16_autocast_covers_training_passes_only(tmp_path: Path) -> None:
    """Validation runs in float32, as eval-mode activations can exceed float16."""
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    trainer = make_trainer(tmp_path, _EvalScaledLinear(), ClassificationTask(), loader)
    trainer.use_amp = True
    trainer._amp_dtype = torch.float16
    images, targets = next(iter(loader))

    def forward_dtype(*, training: bool) -> torch.dtype:
        """Return the model output dtype of one trainer forward pass."""
        trainer.model.train(mode=training)
        output, _, _ = trainer._forward(images, targets, None, training=training)
        return output.dtype

    assert forward_dtype(training=True) == torch.float16
    assert forward_dtype(training=False) == torch.float32
    assert np.isfinite(trainer._run_epoch(loader, training=False)["loss"])
    # bfloat16 has float32's range, so every pass keeps its autocast.
    trainer._amp_dtype = torch.bfloat16
    assert forward_dtype(training=False) == torch.bfloat16


def test_non_finite_epoch_loss_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An epoch whose loss is not finite says so instead of passing silently."""
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    trainer = make_trainer(tmp_path, model, ClassificationTask(), loader)
    assert not np.isfinite(trainer._run_epoch(loader, training=False)["loss"])
    assert "val loss is not finite" in caplog.text


def test_channels_last_and_structured_inputs(tmp_path: Path) -> None:
    """Channels-last training works, and multi-input models receive dicts."""

    class TwoInputs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 1)

        def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            features = self.conv(inputs["rgb"]).mean((2, 3))
            return features + inputs["scale"].float()[:, None]

    samples = [
        {
            "image": {"rgb": torch.rand(3, 4, 4), "scale": torch.tensor(index)},
            "target": torch.tensor(index % 2),
        }
        for index in range(4)
    ]
    loader = DataLoader(samples, batch_size=2)
    model = TwoInputs()
    trainer = make_trainer(tmp_path, model, ClassificationTask(), loader)
    trainer.config = TrainerConfig(
        device="cpu", amp=False, channels_last=True, output_dir=tmp_path
    )
    metrics = trainer._run_epoch(loader, training=True)
    assert np.isfinite(metrics["loss"])

    class ConvModel(ModelABC):
        """TIAToolbox model, whose ``to`` takes no ``memory_format``."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 3)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.conv(images).mean((2, 3))

        @staticmethod
        def infer_batch(
            model: nn.Module, batch_data: torch.Tensor, *, device: str
        ) -> None:
            del model, batch_data, device

    for conv_model in (
        nn.Sequential(nn.Conv2d(3, 2, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()),
        ConvModel(),
    ):
        conv_trainer = make_trainer(
            tmp_path,
            conv_model,
            ClassificationTask(),
            DataLoader(
                TensorDataset(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])),
                batch_size=2,
            ),
            channels_last=True,
        )
        weight = next(conv_trainer.model.parameters())
        assert weight.is_contiguous(memory_format=torch.channels_last)
        conv_trainer.fit()


def test_last_checkpoint_frequency_still_saves_the_final_epoch(
    tmp_path: Path,
) -> None:
    """``save_last_every`` thins checkpoints but never skips the final epoch."""
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    )
    saved_epochs = []

    def record(trainer: Trainer, _metrics: dict[str, float]) -> None:
        path = trainer.output_dir / "last.ckpt"
        saved_epochs.append(load_checkpoint(path)["epoch"] if path.exists() else None)

    model = nn.Linear(1, 2)
    trainer = Trainer(
        model,
        ClassificationTask(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        loader,
        config=TrainerConfig(
            max_epochs=3,
            device="cpu",
            monitor="train_loss",
            output_dir=tmp_path,
            log_every_n_steps=0,
            progress_bar=True,
        ),
        checkpoint_config=CheckpointConfig(save_last_every=2),
        callbacks=[record],
    )
    trainer.fit()
    assert saved_epochs == [None, 2, 3]
    with pytest.raises(ValueError, match="save_last_every"):
        CheckpointConfig(save_last_every=0)


def _slide_dataset(**kwargs: object) -> SlideAnnotationPatchDataset:
    """Build a two-slide in-memory dataset with one annotated corner."""
    store = DictionaryStore()
    store.append(Annotation(box(0, 0, 16, 16), {"class": "tumour"}))
    image = np.arange(48 * 48 * 3, dtype=np.uint32).reshape(48, 48, 3) % 251
    readers = [
        VirtualWSIReader(image.astype(np.uint8), mpp=0.5, power=20) for _ in range(2)
    ]
    return SlideAnnotationPatchDataset(
        readers,
        store,
        MaskTargetBuilder(),
        patch_size=16,
        within_bound=True,
        **kwargs,
    )


def test_slide_jitter_shifts_training_patches_reproducibly() -> None:
    """Training passes jitter patches inside the slide; others use the grid."""
    dataset = _slide_dataset(jitter=6)
    grid = [tuple(dataset[index]["bounds"].tolist()) for index in range(len(dataset))]

    def bounds(epoch: int, *, training: bool) -> list[tuple[int, ...]]:
        dataset.set_epoch(epoch, 5, training=training)
        return [tuple(dataset[i]["bounds"].tolist()) for i in range(len(dataset))]

    first = bounds(0, training=True)
    assert first != grid
    assert first == bounds(0, training=True)
    assert first != bounds(2, training=True)
    assert bounds(1, training=False) == grid
    for x_min, y_min, x_max, y_max in first:
        assert 0 <= x_min < x_max <= 48
        assert 0 <= y_min < y_max <= 48
        assert (x_max - x_min, y_max - y_min) == (16, 16)
    with pytest.raises(ValueError, match="jitter"):
        _slide_dataset(jitter=-1)


def test_slide_targets_can_be_computed_without_reading_pixels() -> None:
    """``compute_targets`` matches the targets built while reading patches."""
    dataset = _slide_dataset()
    presence = dataset.compute_targets(PresenceTargetBuilder())
    assert presence == [int(dataset[i]["target"].any()) for i in range(len(dataset))]
    assert presence.count(1) == 2  # The corner patch of each slide.
    masks = dataset.compute_targets()
    np.testing.assert_array_equal(masks[0], dataset[0]["target"].numpy())


def test_reader_cache_size_is_configurable() -> None:
    """Datasets keep at most ``cache_size`` owned readers per process."""
    image = np.zeros((16, 16, 3), np.uint8)
    paths = []
    with TemporaryDirectory() as directory:
        for index in range(3):
            path = Path(directory) / f"{index}.png"
            Image.fromarray(image).save(path)
            paths.append(path)
        store = DictionaryStore()
        with SlideAnnotationPatchDataset(
            paths, store, MaskTargetBuilder(), patch_size=16, cache_size=1
        ) as dataset:
            for index in range(len(dataset)):
                dataset[index]
                assert len(dataset._reader_cache) <= 1


def test_grouped_block_sampler_keeps_groups_together() -> None:
    """Every index appears once, in blocks drawn from a single group."""
    groups = np.repeat([0, 1, 2], [7, 5, 6])
    generator = torch.Generator().manual_seed(3)
    sampler = GroupedBlockSampler(groups, block_size=4, generator=generator)
    order = list(sampler)
    assert sorted(order) == list(range(len(groups)))
    runs = [len(list(run)) for _, run in itertools.groupby(groups[order])]
    assert all(run >= 1 for run in runs)
    # Blocks of four can only split a group at block boundaries.
    assert len(runs) <= sum(-(-count // 4) for count in (7, 5, 6))
    again = GroupedBlockSampler(groups, 4, generator=torch.Generator().manual_seed(3))
    assert list(again) == order
    with pytest.raises(ValueError, match="block_size"):
        GroupedBlockSampler(groups, 0)
