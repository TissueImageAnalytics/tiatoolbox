"""Training datasets with a common pair/image/target transform pipeline.

``pair_transform(image, target)`` runs first, followed by ``image_transform`` and
``target_transform``, then ``preproc``. ``preproc`` should be the model's
``preproc_func``: like TIAToolbox inference engines, it receives the HWC image
and returns an HWC array or tensor. Values are never rescaled implicitly, so
without ``preproc`` a uint8 image reaches the model as 0-255 floats, exactly as
in inference. Returned samples contain float32 CHW ``image`` tensors and a
``target``; annotation targets can be nested structures. See the Training guide
for label formats and worker ownership.
"""

from __future__ import annotations

import copy
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.models.training._resources import (
    DEFAULT_CACHE_SIZE,
    _cache_resource,
    _ReaderSpec,
    _worker_spec,
)
from tiatoolbox.models.training.augmentations import _window_layout, _zoom_image
from tiatoolbox.models.training.samplers import generate_slide_patch_coordinates
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from types import EllipsisType

    from tiatoolbox.models.training.augmentations import _ScaleJitter
    from tiatoolbox.models.training.targets import TargetBuilderABC, TargetType
    from tiatoolbox.type_hints import Resolution, Units
    from tiatoolbox.wsicore.wsireader import VirtualWSIReader

_SPATIAL_DIMS = 2
_IMAGE_DIMS = 3
_BATCH_IMAGE_DIMS = 4
_RGB_CHANNELS = 3

DEFAULT_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
}

DEFAULT_ANNOTATION_STORE_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
}


def _load_image(path: str | Path) -> np.ndarray:
    """Decode an image file as RGB, like slide reads, or load ``.npy`` unchanged.

    Grayscale and RGBA files become RGB. ``.npy`` arrays keep their channels,
    which allows training models on single- or multi-channel inputs.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"Unable to read image: {path}"
        raise ValueError(msg)
    if image.ndim == _SPATIAL_DIMS:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB if image.shape[-1] == _RGB_CHANNELS else cv2.COLOR_BGRA2RGB,
    )


def _load_mask(path: Path) -> np.ndarray:
    """Load label values, including palette indices and 16-bit integers."""
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        with Image.open(path) as image:
            mask = np.array(image)
    if mask.ndim == _IMAGE_DIMS:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        elif np.all(mask == mask[..., :1]):
            mask = mask[..., 0]
        # Preserve genuine colour codes for the caller's target_transform.
    return mask


def _ensure_tensor_image(
    image: np.ndarray | torch.Tensor, *, channels_last: bool = False
) -> torch.Tensor:
    """Convert an image to a float32 CHW tensor without rescaling its values.

    Arrays are HWC (or HW). Tensors are CHW (or HW) unless ``channels_last``,
    which marks the HWC output of an inference-style ``preproc`` function.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(np.ascontiguousarray(image))
        channels_last = True
    if image.ndim == _SPATIAL_DIMS:
        image = image.unsqueeze(0)
    elif channels_last and image.ndim == _IMAGE_DIMS:
        image = image.permute(2, 0, 1)
    if image.ndim != _IMAGE_DIMS:
        msg = "Images must have shape HxW, HxWxC (arrays) or CxHxW (tensors)."
        raise ValueError(msg)
    return image.float()


def _ensure_tensor_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert mask data into an integer tensor of shape HxW."""
    if isinstance(mask, torch.Tensor):
        tensor = mask
        if tensor.ndim == _IMAGE_DIMS and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(mask))

    if tensor.ndim == _IMAGE_DIMS and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != _SPATIAL_DIMS:
        msg = (
            "Mask arrays must have shape HxW. For colour-coded masks, use "
            "`target_transform` to map colours to class indices."
        )
        raise ValueError(msg)

    return tensor.long()


# Keep each supported container/scalar conversion explicit.
def _ensure_tensor_target(  # noqa: PLR0911
    target: object,
) -> object:
    """Convert a generic target output to tensors recursively."""
    if isinstance(target, torch.Tensor):
        return target
    if isinstance(target, dict):
        return {key: _ensure_tensor_target(value) for key, value in target.items()}
    if isinstance(target, list):
        return [_ensure_tensor_target(value) for value in target]
    if isinstance(target, tuple):
        return tuple(_ensure_tensor_target(value) for value in target)
    if isinstance(target, np.ndarray):
        if target.ndim == 0:
            return torch.tensor(target.item())
        if target.ndim == _IMAGE_DIMS:
            target = np.transpose(target, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(target))
    return torch.tensor(target)


def _window_bounds(
    bounds: tuple[float, float, float, float],
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a pixel window of a ``shape`` patch with ``bounds`` to the same space."""
    x_min, y_min, x_max, y_max = bounds
    height, width = shape
    scale_x, scale_y = (x_max - x_min) / width, (y_max - y_min) / height
    left, top, right, bottom = window
    return (
        x_min + left * scale_x,
        y_min + top * scale_y,
        x_min + right * scale_x,
        y_min + bottom * scale_y,
    )


def _discover_files(
    root: Path,
    suffixes: set[str],
) -> list[Path]:
    """Discover valid image files under a directory."""
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ]
    )


def _normalize_repeated_specs(
    specs: object,
    count: int,
    *,
    argument_name: str,
    reference_name: str,
) -> list[object]:
    """Normalize a shared-or-per-sample runtime spec into a list."""
    if isinstance(specs, list):
        if len(specs) != count:
            msg = (
                f"When `{argument_name}` is a list it must have the same "
                f"length as `{reference_name}`."
            )
            raise ValueError(msg)
        values = specs
    else:
        values = [specs for _ in range(count)]

    return list(values)


def _format_key_preview(keys: set[str], *, limit: int = 5) -> str:
    """Format a short preview of relative keys for error messages."""
    preview = sorted(keys)[:limit]
    suffix = "" if len(keys) <= limit else ", ..."
    return ", ".join(preview) + suffix


def _build_relative_stem_map(
    paths: list[Path],
    root: Path,
    *,
    kind: str,
) -> dict[str, Path]:
    """Build a relative-stem lookup and reject duplicate keys."""
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        relative_key = str(path.relative_to(root).with_suffix(""))
        grouped_paths[relative_key].append(path)

    duplicate_keys = {key for key, grouped in grouped_paths.items() if len(grouped) > 1}
    if duplicate_keys:
        msg = (
            f"Duplicate {kind} files were found for relative keys: "
            f"{_format_key_preview(duplicate_keys)}."
        )
        raise ValueError(msg)

    return {key: grouped[0] for key, grouped in grouped_paths.items()}


def _collect_strict_relative_stem_pairs(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    left_suffixes: set[str],
    right_suffixes: set[str],
    left_argument_name: str,
    right_argument_name: str,
    left_kind: str,
    right_kind: str,
) -> list[tuple[Path, Path]]:
    """Collect two file sets by strict nested relative-stem matching."""
    left_root = Path(left_dir)
    right_root = Path(right_dir)
    if not left_root.exists() or not left_root.is_dir():
        msg = f"`{left_argument_name}` must be an existing directory."
        raise ValueError(msg)
    if not right_root.exists() or not right_root.is_dir():
        msg = f"`{right_argument_name}` must be an existing directory."
        raise ValueError(msg)

    left_files = _discover_files(
        left_root,
        {suffix.lower() for suffix in left_suffixes},
    )
    right_files = _discover_files(
        right_root,
        {suffix.lower() for suffix in right_suffixes},
    )
    left_map = _build_relative_stem_map(left_files, left_root, kind=left_kind)
    right_map = _build_relative_stem_map(right_files, right_root, kind=right_kind)

    matched_keys = sorted(set(left_map).intersection(right_map))
    unmatched_left_keys = set(left_map).difference(right_map)
    unmatched_right_keys = set(right_map).difference(left_map)
    pair_label = f"{left_kind}/{right_kind}"
    if not matched_keys:
        msg = f"No {pair_label} pairs were found using file stem matching."
        raise ValueError(msg)

    if unmatched_left_keys or unmatched_right_keys:
        details = []
        if unmatched_left_keys:
            details.append(
                f"missing {right_kind}s for {_format_key_preview(unmatched_left_keys)}"
            )
        if unmatched_right_keys:
            details.append(
                f"missing {left_kind}s for {_format_key_preview(unmatched_right_keys)}"
            )
        msg = f"Unmatched {pair_label} files were found: {'; '.join(details)}."
        raise ValueError(msg)

    return [(left_map[key], right_map[key]) for key in matched_keys]


class _TrainingDataset:
    """Manage transforms, augmentation epochs and owned dataset resources."""

    def _set_transforms(
        self,
        pair_transform: Callable | None,
        image_transform: Callable | None,
        target_transform: Callable | None,
        preproc: Callable | None,
    ) -> None:
        """Store the sample pipeline, applied in argument order."""
        self.pair_transform = pair_transform
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.preproc = preproc

    def _scale_jitter(self) -> _ScaleJitter | None:
        """Return the preset zoom that annotation datasets apply while reading."""
        return getattr(self.pair_transform, "scale_jitter", None)

    def _transform_sample(
        self, image: object, target: object, *, zoomed: bool = False
    ) -> tuple[torch.Tensor, object]:
        """Apply the transforms and ``preproc``; return a CHW image tensor.

        ``zoomed`` marks a sample to which the dataset applied the preset's
        scale jitter itself.
        """
        if zoomed:
            image, target = self.pair_transform(image, target, apply_scale_jitter=False)
        elif self.pair_transform is not None:
            image, target = self.pair_transform(image, target)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.preproc is None:
            return _ensure_tensor_image(image), target
        # Inference-style preprocessing returns HWC arrays or tensors.
        return _ensure_tensor_image(self.preproc(image), channels_last=True), target

    def with_transforms(
        self,
        *,
        pair_transform: Callable | EllipsisType | None = ...,
        image_transform: Callable | EllipsisType | None = ...,
        target_transform: Callable | EllipsisType | None = ...,
        preproc: Callable | EllipsisType | None = ...,
    ) -> Self:
        """Return a copy with some transforms replaced; ``...`` keeps one.

        The copy shares the sample index (files, coordinates, class mapping)
        but has its own resource caches and epoch state. Use it to apply
        training augmentation and deterministic validation preprocessing to
        the same split indices without scanning files or slides again.
        """
        dataset = copy.copy(self)
        dataset._init_training_state()  # noqa: SLF001 - same class
        for name in ("_store_cache", "_reader_cache"):
            if hasattr(self, name):
                setattr(dataset, name, OrderedDict())
        for name, value in (
            ("pair_transform", pair_transform),
            ("image_transform", image_transform),
            ("target_transform", target_transform),
            ("preproc", preproc),
        ):
            if value is not ...:
                setattr(dataset, name, value)
        return dataset

    def _init_training_state(self) -> None:
        """Initialize process ownership and shared augmentation epoch state."""
        self._resource_pid = os.getpid()
        self._augmentation_epoch = torch.tensor(
            [-1, 0, 1], dtype=torch.int64
        ).share_memory_()

    def set_epoch(self, epoch: int, seed: int, *, training: bool = True) -> None:
        """Set repeatable sample seeds, including for persistent workers."""
        self._augmentation_epoch[:] = torch.tensor([seed, epoch, int(training)])

    def _sample_seed(self, index: int, stream: int) -> int | None:
        """Return a sample's seed for one random stream, once the trainer set one."""
        seed, epoch, _ = self._augmentation_epoch.tolist()
        if seed < 0:
            return None
        sequence = np.random.SeedSequence([seed, epoch, index, stream])
        return int(sequence.generate_state(1)[0])

    def _prepare_sample(self, index: int) -> None:
        """Reopen worker resources and seed supported transforms for a sample."""
        if self._resource_pid != os.getpid():
            self.close()
            memo = {}
            for name in ("annotation_store_specs", "slide_inputs"):
                if hasattr(self, name):
                    setattr(
                        self,
                        name,
                        [_worker_spec(spec, memo) for spec in getattr(self, name)],
                    )
            self._resource_pid = os.getpid()
        for stream, name in enumerate(
            ("pair_transform", "image_transform", "target_transform")
        ):
            transform = getattr(self, name, None)
            if hasattr(transform, "set_random_seed"):
                sample_seed = self._sample_seed(index, stream)
                if sample_seed is None:
                    return
                transform.set_random_seed(sample_seed)

    def close(self) -> None:
        """Release owned cached handles; user-supplied resources remain open."""
        for name in ("_store_cache", "_reader_cache"):
            cache = getattr(self, name, {})
            while cache:
                _, resource = cache.popitem()
                resource.close()

    def __enter__(self) -> Self:
        """Return a dataset whose owned resources close on context exit."""
        return self

    def __exit__(self, *args: object) -> None:
        """Release owned resources after dataset use."""
        self.close()

    def __del__(self) -> None:
        """Release owned resources when a dataset is collected."""
        self.close()

    def __getstate__(self) -> dict:
        """Serialize specifications, with no open cached handles."""
        state = self.__dict__.copy()
        memo = {}
        for name in ("_store_cache", "_reader_cache"):
            if name in state:
                state[name] = OrderedDict()
        for name in ("annotation_store_specs", "slide_inputs"):
            if name in state:
                state[name] = [_worker_spec(spec, memo) for spec in state[name]]
        # Input masks are used only while building the index in the parent.
        if "input_mask_specs" in state:
            state["input_mask_specs"] = [None] * len(state["input_mask_specs"])
        return state


def _open_cached_store(
    store_spec: AnnotationStore | str | Path,
    cache: OrderedDict[str | int, AnnotationStore],
    capacity: int = DEFAULT_CACHE_SIZE,
) -> AnnotationStore:
    """Resolve and cache an annotation store."""
    if isinstance(store_spec, AnnotationStore):
        return store_spec

    store_path = Path(store_spec)
    if not store_path.exists():
        msg = f"Annotation store path does not exist: `{store_path}`."
        raise ValueError(msg)

    cache_key = str(store_path.resolve())
    return _cache_resource(cache, cache_key, lambda: SQLiteStore(store_path), capacity)


def _open_cached_reader(
    slide_spec: WSIReader | str | Path | _ReaderSpec,
    cache: OrderedDict[str | int, WSIReader],
    capacity: int = DEFAULT_CACHE_SIZE,
) -> WSIReader:
    """Resolve and cache a slide reader."""
    if isinstance(slide_spec, WSIReader):
        return slide_spec

    if isinstance(slide_spec, _ReaderSpec):
        return _cache_resource(cache, id(slide_spec), slide_spec.open, capacity)

    slide_path = Path(slide_spec)
    cache_key = str(slide_path.resolve())
    return _cache_resource(
        cache, cache_key, lambda: WSIReader.open(slide_path), capacity
    )


class PatchFolderClassificationDataset(_TrainingDataset, DatasetFolder):
    """Classification dataset where each class is represented by one folder.

    Targets are the folder's class index as an int64 tensor. A
    ``target_transform`` may return anything :class:`ClassificationTask`
    accepts, such as a float vector of class probabilities or a multi-hot
    label vector; its dtype is kept.

    Example directory structure:

    .. code-block:: text

        root/
          class_a/
            patch_1.png
            patch_2.png
          class_b/
            patch_3.png

    """

    def __init__(
        self: PatchFolderClassificationDataset,
        root_dir: str | Path,
        *,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preproc: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
        file_extensions: set[str] | None = None,
    ) -> None:
        """Initialize :class:`PatchFolderClassificationDataset`."""
        self._init_training_state()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            msg = "`root_dir` must be an existing directory."
            raise ValueError(msg)

        self._provided_class_to_idx = class_to_idx
        self.file_extensions = tuple(sorted(file_extensions or DEFAULT_IMAGE_SUFFIXES))

        # A supplied mapping may name classes absent from this folder, such as a
        # rare class missing from a validation split. Discovered classes must
        # each contain at least one image.
        super().__init__(
            root=self.root_dir,
            loader=_load_image,
            extensions=self.file_extensions,
            allow_empty=class_to_idx is not None,
        )
        # VisionDataset resets target_transform, so set the pipeline afterwards.
        self._set_transforms(pair_transform, image_transform, target_transform, preproc)

    def find_classes(
        self: PatchFolderClassificationDataset,
        directory: str | Path,
    ) -> tuple[list[str], dict[str, int]]:
        """Return sorted class folders, optionally filtered by a provided mapping."""
        class_dirs = sorted(path for path in Path(directory).iterdir() if path.is_dir())
        if not class_dirs:
            msg = "`root_dir` does not contain any class directories."
            raise ValueError(msg)

        if self._provided_class_to_idx is None:
            classes = [path.name for path in class_dirs]
            return classes, {
                class_name: index for index, class_name in enumerate(classes)
            }

        classes = [
            path.name for path in class_dirs if path.name in self._provided_class_to_idx
        ]
        return classes, self._provided_class_to_idx

    def __getitem__(self: PatchFolderClassificationDataset, index: int) -> dict:
        """Get one sample from the dataset."""
        self._prepare_sample(index)
        path, target = self.samples[index]
        image, target = self._transform_sample(self.loader(path), target)
        # Folder labels become int64 tensors; a ``target_transform`` keeps the
        # dtype it returns, such as float soft labels or multi-hot vectors.
        return {"image": image, "target": _ensure_tensor_target(target)}


class _NpyArray:
    """A ``.npy`` file memory-mapped lazily in each process, never pickled whole."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._array: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        """Return the read-only memory map, opening it on first use."""
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __len__(self) -> int:
        """Return the number of samples on the first axis."""
        return len(self.array)

    def __getitem__(self, index: int) -> np.ndarray:
        """Read one sample into memory."""
        return np.array(self.array[index])

    def __getstate__(self) -> dict:
        """Send only the path to data-loader workers."""
        return {"path": self.path, "_array": None}


def _sample_source(
    source: str | Path | np.ndarray | list[str | Path | np.ndarray],
) -> _NpyArray | np.ndarray | list[Path | np.ndarray]:
    """Index images or masks given as files, arrays, or one ``.npy`` stack."""
    if isinstance(source, (str, Path)):
        if Path(source).suffix.lower() != ".npy":
            msg = "A single image or mask source must be a `.npy` file of samples."
            raise ValueError(msg)
        return _NpyArray(source)
    if isinstance(source, np.ndarray):
        return source
    return [Path(item) if isinstance(item, (str, Path)) else item for item in source]


class PatchMaskPairDataset(_TrainingDataset, Dataset):
    """Segmentation dataset of image/mask pairs.

    Pairs come from ``image_dir``/``mask_dir`` (matched by relative file stem)
    or from ``images``/``masks``: equal-length lists of file paths or arrays, an
    ``(N, H, W[, C])`` array, or the path of a ``.npy`` file holding such a stack.
    ``.npy`` stacks are memory-mapped in each process, so data-loader workers
    never receive copies; in-memory arrays are copied to spawned workers. Use
    ``target_transform`` to select one channel of multi-channel label arrays.
    """

    def __init__(
        self: PatchMaskPairDataset,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        *,
        images: str | Path | np.ndarray | list[str | Path | np.ndarray] | None = None,
        masks: str | Path | np.ndarray | list[str | Path | np.ndarray] | None = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preproc: Callable | None = None,
        image_extensions: set[str] | None = None,
        mask_extensions: set[str] | None = None,
    ) -> None:
        """Initialize :class:`PatchMaskPairDataset`."""
        self._init_training_state()
        super().__init__()
        self._set_transforms(pair_transform, image_transform, target_transform, preproc)
        if images is not None or masks is not None:
            if images is None or masks is None:
                msg = "Both `images` and `masks` must be provided together."
                raise ValueError(msg)
            self.images, self.masks = _sample_source(images), _sample_source(masks)
            if len(self.images) != len(self.masks):
                msg = "`images` and `masks` must have the same length."
                raise ValueError(msg)
            return
        if image_dir is None or mask_dir is None:
            msg = "Provide either (`image_dir`, `mask_dir`) or (`images`, `masks`)."
            raise ValueError(msg)
        pairs = _collect_strict_relative_stem_pairs(
            image_dir,
            mask_dir,
            left_suffixes=image_extensions or DEFAULT_IMAGE_SUFFIXES,
            right_suffixes=mask_extensions or DEFAULT_IMAGE_SUFFIXES,
            left_argument_name="image_dir",
            right_argument_name="mask_dir",
            left_kind="image",
            right_kind="mask",
        )
        self.images = [image for image, _ in pairs]
        self.masks = [mask for _, mask in pairs]

    def __len__(self: PatchMaskPairDataset) -> int:
        """Return number of available pairs."""
        return len(self.images)

    def __getitem__(self: PatchMaskPairDataset, index: int) -> dict:
        """Get one image/mask sample pair from the dataset."""
        self._prepare_sample(index)
        image, mask = self.images[index], self.masks[index]
        image = _load_image(image) if isinstance(image, Path) else np.asarray(image)
        mask = _load_mask(mask) if isinstance(mask, Path) else np.asarray(mask)
        image, mask = self._transform_sample(image, mask)
        return {"image": image, "target": _ensure_tensor_mask(mask)}


class PatchAnnotationDataset(_TrainingDataset, Dataset):
    """Dataset for patch images paired with TIAToolbox annotation stores.

    This dataset supports training targets built dynamically from annotations
    stored in :class:`tiatoolbox.annotation.SQLiteStore`. Annotation geometries
    and optional ``patch_bounds`` are interpreted in the same slide/baseline
    coordinate space. The generated target arrays are in patch pixel
    coordinates matching the loaded image shape.

    With an augmentation preset as ``pair_transform``, the dataset applies the
    preset's crops and zooms itself: it resizes a window of the patch and
    builds the target for that window, so thin targets stay exact. Where the
    window extends past the patch, the image is padded white and the target
    with the preset's ``target_fill``.

    Use the dataset as a context manager or call ``close()`` to release cached
    files promptly. Supplied stores are borrowed; an in-memory SQLite store is
    snapshotted when workers start, and file-backed stores must first be committed.
    """

    @classmethod
    def from_dirs(
        cls: type[PatchAnnotationDataset],
        patch_dir: str | Path,
        annotation_store_dir: str | Path,
        target_builder: TargetBuilderABC,
        *,
        patch_bounds: (
            list[tuple[float, float, float, float]] | np.ndarray | None
        ) = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preproc: Callable | None = None,
        patch_extensions: set[str] | None = None,
        annotation_store_extensions: set[str] | None = None,
    ) -> PatchAnnotationDataset:
        """Create a dataset from patch and annotation-store directories.

        Patch files and store files are paired by strict relative stem. For
        example, ``patch_dir/case_1/tile_0.png`` matches
        ``annotation_store_dir/case_1/tile_0.db``. Extra files, missing files,
        duplicate relative stems, and empty matches are rejected to avoid silent
        training target misalignment.
        """
        pairs = _collect_strict_relative_stem_pairs(
            patch_dir,
            annotation_store_dir,
            left_suffixes=patch_extensions or DEFAULT_IMAGE_SUFFIXES,
            right_suffixes=(
                annotation_store_extensions or DEFAULT_ANNOTATION_STORE_SUFFIXES
            ),
            left_argument_name="patch_dir",
            right_argument_name="annotation_store_dir",
            left_kind="patch image",
            right_kind="annotation store",
        )

        return cls(
            patch_inputs=[patch_path for patch_path, _ in pairs],
            annotation_stores=[store_path for _, store_path in pairs],
            target_builder=target_builder,
            patch_bounds=patch_bounds,
            pair_transform=pair_transform,
            image_transform=image_transform,
            target_transform=target_transform,
            preproc=preproc,
        )

    def __init__(
        self: PatchAnnotationDataset,
        patch_inputs: list[str | Path | np.ndarray] | np.ndarray,
        annotation_stores: (
            AnnotationStore | str | Path | list[AnnotationStore | str | Path]
        ),
        target_builder: TargetBuilderABC,
        patch_bounds: (
            list[tuple[float, float, float, float]] | np.ndarray | None
        ) = None,
        *,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preproc: Callable | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Initialize :class:`PatchAnnotationDataset`."""
        self._init_training_state()
        super().__init__()

        self.patch_inputs = self._normalize_patch_inputs(patch_inputs)
        if not self.patch_inputs:
            msg = "`patch_inputs` must contain at least one sample."
            raise ValueError(msg)

        self.annotation_store_specs = _normalize_repeated_specs(
            annotation_stores,
            len(self.patch_inputs),
            argument_name="annotation_stores",
            reference_name="patch_inputs",
        )
        self.target_builder = target_builder

        self.patch_bounds = self._normalize_patch_bounds(patch_bounds)
        if self.patch_bounds is not None and len(self.patch_bounds) != len(
            self.patch_inputs
        ):
            msg = "`patch_bounds` must have the same length as `patch_inputs`."
            raise ValueError(msg)

        self._set_transforms(pair_transform, image_transform, target_transform, preproc)
        self.cache_size = int(cache_size)
        self._store_cache: OrderedDict[str | int, AnnotationStore] = OrderedDict()

    @staticmethod
    def _normalize_patch_inputs(
        patch_inputs: list[str | Path | np.ndarray] | np.ndarray,
    ) -> list[str | Path | np.ndarray]:
        """Normalize supported patch input variants to a list."""
        if isinstance(patch_inputs, np.ndarray):
            if patch_inputs.ndim == _BATCH_IMAGE_DIMS:
                return list(patch_inputs)
            if patch_inputs.dtype == object:
                return list(patch_inputs)
            msg = (
                "`patch_inputs` as ndarray must be either object dtype "
                "or a 4D tensor-like array (N, H, W, C)."
            )
            raise ValueError(msg)

        if isinstance(patch_inputs, list):
            return patch_inputs

        msg = "`patch_inputs` must be a list or a numpy.ndarray."
        raise ValueError(msg)

    @staticmethod
    def _normalize_patch_bounds(
        patch_bounds: (list[tuple[float, float, float, float]] | np.ndarray | None),
    ) -> list[tuple[float, float, float, float]] | None:
        """Normalize optional patch bounds argument."""
        if patch_bounds is None:
            return None

        if isinstance(patch_bounds, np.ndarray):
            patch_bounds = patch_bounds.tolist()

        normalized = []
        for bounds in patch_bounds:
            if len(bounds) != 4:  # noqa: PLR2004
                msg = "Each `patch_bounds` item must have length 4."
                raise ValueError(msg)
            x_min, y_min, x_max, y_max = [float(value) for value in bounds]
            if x_max <= x_min or y_max <= y_min:
                msg = (
                    "Each patch bounds item must satisfy "
                    "x_max > x_min and y_max > y_min."
                )
                raise ValueError(msg)
            normalized.append((x_min, y_min, x_max, y_max))
        return normalized

    def _get_store(self: PatchAnnotationDataset, index: int) -> AnnotationStore:
        """Get an AnnotationStore for one sample index."""
        return _open_cached_store(
            self.annotation_store_specs[index], self._store_cache, self.cache_size
        )

    def __len__(self: PatchAnnotationDataset) -> int:
        """Return number of samples."""
        return len(self.patch_inputs)

    def __getitem__(self: PatchAnnotationDataset, index: int) -> dict:
        """Get a patch image and its target built from paired annotations."""
        self._prepare_sample(index)
        image_item = self.patch_inputs[index]
        image = (
            _load_image(Path(image_item))
            if isinstance(image_item, (str, Path))
            else image_item
        )
        height, width = image.shape[:2]

        bounds = (
            self.patch_bounds[index]
            if self.patch_bounds is not None
            else (0.0, 0.0, float(width), float(height))
        )
        # Crop or zoom a window of the patch and build the target for that
        # window, so thin targets are drawn at the training resolution. Where
        # the window extends past the patch, the image is padded white and the
        # target is built for the part inside, then padded to match.
        output_shape = target_shape = (height, width)
        box = None
        scale_jitter = self._scale_jitter()
        if scale_jitter is not None:
            window, output_shape = scale_jitter.sample_window((height, width))
            layout = _window_layout(window, (height, width), output_shape)
            if layout is None:
                output_shape = (height, width)
            elif (window, output_shape) != ((0, 0, width, height), (height, width)):
                inner, box = layout
                image = _zoom_image(image, window, output_shape)
                bounds = _window_bounds(bounds, inner, (height, width))
                target_shape = (box[3] - box[1], box[2] - box[0])
        store = self._get_store(index)
        target = self.target_builder.create_target(
            store=store,
            patch_bounds=bounds,
            output_shape=target_shape,
        )
        if box is not None and box != (0, 0, output_shape[1], output_shape[0]):
            target = scale_jitter.pad_target(target, box, output_shape)

        image, target = self._transform_sample(
            image, target, zoomed=scale_jitter is not None
        )
        return {"image": image, "target": _ensure_tensor_target(target)}


class SlideAnnotationPatchDataset(_TrainingDataset, Dataset):
    """Dataset reading patches on-the-fly from slides with annotation stores.

    Patch images may be read at any supported resolution, but annotation queries
    are issued in baseline slide coordinates. Target builders then rescale
    annotation-derived dense outputs into patch pixel coordinates matching the
    returned image.

    Owned readers and stores are cached per process, up to ``cache_size`` of
    each; with more slides than that, a :class:`GroupedBlockSampler` keeps
    consecutive samples on one slide. Use ``close()`` or a context manager to
    release them promptly. Supplied runtime objects remain open in the caller;
    worker copies retain reader metadata and the selected TIFF series.

    ``jitter`` shifts each patch by up to that many pixels (at the read
    resolution) in x and y during training passes, with a fresh seeded offset
    per sample and epoch, so the grid does not repeat exactly. Validation
    passes and direct indexing use the grid. ``within_bound`` keeps jittered
    patches inside the slide.

    With an augmentation preset as ``pair_transform``, the dataset applies the
    preset's crops and zooms itself: it reads a smaller or larger window
    centred on the patch, resizes it to the training size and builds the target
    for that window, so zooming out needs no padding and thin targets stay
    exact. ``bounds`` then gives the window read.
    """

    def __init__(  # noqa: PLR0913
        self: SlideAnnotationPatchDataset,
        slide_inputs: list[str | Path | WSIReader],
        annotation_stores: (
            AnnotationStore | str | Path | list[AnnotationStore | str | Path]
        ),
        target_builder: TargetBuilderABC,
        patch_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        resolution: Resolution = 0,
        units: Units = "level",
        within_bound: bool = False,
        input_masks: (
            str
            | Path
            | np.ndarray
            | VirtualWSIReader
            | AnnotationStore
            | list[str | Path | np.ndarray | VirtualWSIReader | AnnotationStore | None]
            | None
        ) = None,
        min_mask_ratio: float = 0.0,
        store_filter: str | None = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
        preproc: Callable | None = None,
        jitter: int = 0,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Initialize :class:`SlideAnnotationPatchDataset`."""
        self._init_training_state()
        super().__init__()
        if not isinstance(slide_inputs, list) or not slide_inputs:
            msg = "`slide_inputs` must be a non-empty list."
            raise ValueError(msg)
        self.slide_inputs = list(slide_inputs)
        self.annotation_store_specs = _normalize_repeated_specs(
            annotation_stores,
            len(self.slide_inputs),
            argument_name="annotation_stores",
            reference_name="slide_inputs",
        )
        self.input_mask_specs = _normalize_repeated_specs(
            input_masks,
            len(self.slide_inputs),
            argument_name="input_masks",
            reference_name="slide_inputs",
        )

        self.target_builder = target_builder
        self.patch_size = patch_size
        self.stride = stride
        self.resolution = resolution
        self.units = units
        self.within_bound = within_bound
        self.min_mask_ratio = float(min_mask_ratio)
        self.store_filter = store_filter
        self._set_transforms(pair_transform, image_transform, target_transform, preproc)
        if jitter < 0:
            msg = "`jitter` must be a non-negative number of pixels."
            raise ValueError(msg)
        self.jitter = int(jitter)
        self.cache_size = int(cache_size)

        self._reader_cache: OrderedDict[str | int, WSIReader] = OrderedDict()
        self._store_cache: OrderedDict[str | int, AnnotationStore] = OrderedDict()
        try:
            self.sample_slide_indices, self.sample_bounds = self._build_patch_index()
        finally:
            self.close()

    def _get_reader(self: SlideAnnotationPatchDataset, index: int) -> WSIReader:
        """Get a cached WSI reader for one slide index."""
        return _open_cached_reader(
            self.slide_inputs[index], self._reader_cache, self.cache_size
        )

    def _get_store(self: SlideAnnotationPatchDataset, index: int) -> AnnotationStore:
        """Get a cached annotation store for one slide index."""
        return _open_cached_store(
            self.annotation_store_specs[index], self._store_cache, self.cache_size
        )

    def _jittered_bounds(
        self: SlideAnnotationPatchDataset,
        index: int,
        bounds: tuple[int, int, int, int],
        reader: WSIReader,
    ) -> tuple[int, int, int, int]:
        """Shift a training patch by its seeded offset; keep it in the slide."""
        seed = self._sample_seed(index, stream=3)
        if not self.jitter or seed is None or not self._augmentation_epoch[2]:
            return bounds
        offset_x, offset_y = np.random.default_rng(seed).integers(
            -self.jitter, self.jitter + 1, size=2
        )
        x_min, y_min, x_max, y_max = bounds
        if self.within_bound:
            width, height = reader.slide_dimensions(self.resolution, self.units)
            offset_x = np.clip(offset_x, -x_min, width - x_max)
            offset_y = np.clip(offset_y, -y_min, height - y_max)
        offset_x, offset_y = int(offset_x), int(offset_y)
        return (x_min + offset_x, y_min + offset_y, x_max + offset_x, y_max + offset_y)

    def _zoomed_bounds(
        self: SlideAnnotationPatchDataset,
        scale_jitter: _ScaleJitter,
        bounds: tuple[int, int, int, int],
        reader: WSIReader,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        """Return the window to read around a patch and the shape to resize it to."""
        x_min, y_min, x_max, y_max = bounds
        (left, top, right, bottom), output_shape = scale_jitter.sample_window(
            (y_max - y_min, x_max - x_min), centred=True
        )
        x_start, y_start = x_min + left, y_min + top
        if self.within_bound:
            width, height = reader.slide_dimensions(self.resolution, self.units)
            x_start = int(np.clip(x_start, 0, max(width - (right - left), 0)))
            y_start = int(np.clip(y_start, 0, max(height - (bottom - top), 0)))
        window = (x_start, y_start, x_start + right - left, y_start + bottom - top)
        return window, output_shape

    def compute_targets(
        self: SlideAnnotationPatchDataset,
        target_builder: TargetBuilderABC | None = None,
    ) -> list[TargetType]:
        """Build every sample's target from annotations alone, without pixels.

        Uses the dataset's own builder by default. With a scalar builder such
        as :class:`PresenceTargetBuilder`, the result labels patches for
        stratified splits or a :class:`ClassBalancedIndexSampler`. Targets use
        the patch grid, without ``jitter``.
        """
        builder = self.target_builder if target_builder is None else target_builder
        targets = []
        try:
            for slide_index, bounds in zip(
                self.sample_slide_indices.tolist(),
                self.sample_bounds.tolist(),
                strict=True,
            ):
                baseline = self._get_reader(
                    slide_index
                ).bounds_at_resolution_to_baseline(
                    tuple(bounds), self.resolution, self.units
                )
                targets.append(
                    builder.create_target(
                        store=self._get_store(slide_index),
                        patch_bounds=tuple(float(value) for value in baseline),
                        output_shape=(bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    )
                )
        finally:
            self.close()
        return targets

    def _build_patch_index(
        self: SlideAnnotationPatchDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build flattened slide-index and patch-bounds arrays."""
        per_slide_indices: list[np.ndarray] = []
        per_slide_bounds: list[np.ndarray] = []

        for slide_index in range(len(self.slide_inputs)):
            reader = self._get_reader(slide_index)
            bounds = generate_slide_patch_coordinates(
                input_img=reader,
                patch_size=self.patch_size,
                stride=self.stride,
                resolution=self.resolution,
                units=self.units,
                within_bound=self.within_bound,
                input_mask=self.input_mask_specs[slide_index],
                min_mask_ratio=self.min_mask_ratio,
                store_filter=self.store_filter,
            )
            if bounds.size == 0:
                continue

            per_slide_indices.append(
                np.full((bounds.shape[0],), fill_value=slide_index, dtype=np.int64),
            )
            per_slide_bounds.append(bounds.astype(np.int64, copy=False))

        if not per_slide_bounds:
            msg = "No patch coordinates were generated for the provided slides."
            raise ValueError(msg)

        return (
            np.concatenate(per_slide_indices, axis=0),
            np.concatenate(per_slide_bounds, axis=0),
        )

    def __len__(self: SlideAnnotationPatchDataset) -> int:
        """Return total number of generated patch coordinates."""
        return int(self.sample_slide_indices.shape[0])

    def __getitem__(self: SlideAnnotationPatchDataset, index: int) -> dict:
        """Get one on-the-fly slide patch and annotation-derived target."""
        self._prepare_sample(index)
        slide_index = int(self.sample_slide_indices[index])
        reader = self._get_reader(slide_index)
        bounds_at_resolution = self._jittered_bounds(
            index,
            tuple(int(value) for value in self.sample_bounds[index].tolist()),
            reader,
        )
        # Zooming reads a larger or smaller window around the patch and resizes
        # it to the training shape; targets are built for that window below.
        scale_jitter = self._scale_jitter()
        if scale_jitter is not None:
            bounds_at_resolution, output_shape = self._zoomed_bounds(
                scale_jitter, bounds_at_resolution, reader
            )
        image = reader.read_bounds(
            bounds_at_resolution,
            resolution=self.resolution,
            units=self.units,
            coord_space="resolution",
        )
        if scale_jitter is not None and image.shape[:2] != output_shape:
            image = _zoom_image(
                image, (0, 0, image.shape[1], image.shape[0]), output_shape
            )
        height, width = image.shape[:2]

        # Annotation stores use baseline slide coordinates. Convert the patch
        # bounds from the image-read resolution before querying target builders;
        # target arrays are still emitted at the patch image pixel shape above.
        bounds_at_baseline = tuple(
            float(value)
            for value in reader.bounds_at_resolution_to_baseline(
                bounds_at_resolution,
                self.resolution,
                self.units,
            )
        )
        store = self._get_store(slide_index)
        target = self.target_builder.create_target(
            store=store,
            patch_bounds=bounds_at_baseline,
            output_shape=(height, width),
        )

        image, target = self._transform_sample(
            image, target, zoomed=scale_jitter is not None
        )
        bounds_tensor = torch.as_tensor(bounds_at_resolution, dtype=torch.long)
        return {
            "image": image,
            "target": _ensure_tensor_target(target),
            "slide_index": torch.tensor(slide_index, dtype=torch.long),
            "bounds": bounds_tensor,
        }
