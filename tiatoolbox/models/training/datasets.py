"""Datasets used for model training workflows."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.models.training.samplers import generate_slide_patch_coordinates
from tiatoolbox.models.training.targets import TargetBuilderABC
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from tiatoolbox.type_hints import Resolution, Units
    from tiatoolbox.wsicore.wsireader import VirtualWSIReader

DEFAULT_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
}


def _load_image(path: str | Path) -> np.ndarray:
    """Load an image-like array from disk."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return imread(path, as_uint8=False)


def _load_mask(path: Path) -> np.ndarray:
    """Load a segmentation mask array from disk."""
    mask = _load_image(path)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = mask[..., 0]
    return mask


def _ensure_tensor_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert image data into a channel-first float tensor."""
    if isinstance(image, torch.Tensor):
        return image.float()

    if image.ndim == 2:
        image = np.repeat(image[..., None], repeats=3, axis=-1)

    if image.ndim != 3:
        msg = "Image arrays must have shape HxW or HxWxC."
        raise ValueError(msg)

    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)

    tensor = torch.from_numpy(np.ascontiguousarray(image))
    tensor = tensor.permute(2, 0, 1).float()

    if np.issubdtype(image.dtype, np.integer):
        tensor = tensor / 255.0

    return tensor


def _ensure_tensor_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert mask data into an integer tensor of shape HxW."""
    if isinstance(mask, torch.Tensor):
        tensor = mask
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(mask))

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2:
        msg = "Mask arrays must have shape HxW."
        raise ValueError(msg)

    return tensor.long()


def _ensure_tensor_target(
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
        return torch.from_numpy(np.ascontiguousarray(target))
    return torch.tensor(target)


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


def _normalize_runtime_spec_path(spec: object) -> Path | None:
    """Extract a path from a path-backed runtime object when possible."""
    if isinstance(spec, (str, Path)):
        return Path(spec)

    for attribute_name in ("input_path", "path", "connection"):
        attribute_value = getattr(spec, attribute_name, None)
        if isinstance(attribute_value, (str, Path)):
            return Path(attribute_value)

    return None


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

    return [_normalize_runtime_spec_path(spec) or spec for spec in values]


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

    duplicate_keys = {
        key for key, grouped in grouped_paths.items() if len(grouped) > 1
    }
    if duplicate_keys:
        msg = (
            f"Duplicate {kind} files were found for relative keys: "
            f"{_format_key_preview(duplicate_keys)}."
        )
        raise ValueError(msg)

    return {key: grouped[0] for key, grouped in grouped_paths.items()}


def _open_cached_store(
    store_spec: AnnotationStore | str | Path,
    cache: dict[str, AnnotationStore],
) -> AnnotationStore:
    """Resolve and cache an annotation store."""
    if isinstance(store_spec, AnnotationStore):
        return store_spec

    store_path = Path(store_spec)
    if not store_path.exists():
        msg = f"Annotation store path does not exist: `{store_path}`."
        raise ValueError(msg)

    cache_key = str(store_path.resolve())
    if cache_key not in cache:
        cache[cache_key] = SQLiteStore(store_path)
    return cache[cache_key]


def _open_cached_reader(
    slide_spec: WSIReader | str | Path,
    cache: dict[str, WSIReader],
) -> WSIReader:
    """Resolve and cache a slide reader."""
    if isinstance(slide_spec, WSIReader):
        return slide_spec

    slide_path = Path(slide_spec)
    cache_key = str(slide_path.resolve())
    if cache_key not in cache:
        cache[cache_key] = WSIReader.open(slide_path)
    return cache[cache_key]


class PatchFolderClassificationDataset(DatasetFolder):
    """Classification dataset where each class is represented by one folder.

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
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
        file_extensions: set[str] | None = None,
    ) -> None:
        """Initialize :class:`PatchFolderClassificationDataset`."""
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            msg = "`root_dir` must be an existing directory."
            raise ValueError(msg)

        self._provided_class_to_idx = class_to_idx
        self.file_extensions = tuple(sorted(file_extensions or DEFAULT_IMAGE_SUFFIXES))

        super().__init__(
            root=self.root_dir,
            loader=_load_image,
            extensions=self.file_extensions,
            transform=transform,
            target_transform=target_transform,
        )

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
            return classes, {class_name: index for index, class_name in enumerate(classes)}

        classes = [
            path.name for path in class_dirs if path.name in self._provided_class_to_idx
        ]
        return classes, self._provided_class_to_idx

    def __len__(self: PatchFolderClassificationDataset) -> int:
        """Return number of discovered samples."""
        return len(self.samples)

    def __getitem__(self: PatchFolderClassificationDataset, index: int) -> dict:
        """Get one sample from the dataset."""
        image, target = super().__getitem__(index)
        image_tensor = _ensure_tensor_image(image)

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)

        return {"image": image_tensor, "target": target}


class PatchMaskPairDataset(Dataset):
    """Segmentation dataset based on image/mask pairs on disk."""

    def __init__(
        self: PatchMaskPairDataset,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        image_paths: list[str | Path] | None = None,
        mask_paths: list[str | Path] | None = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        mask_transform: Callable | None = None,
        image_extensions: set[str] | None = None,
        mask_extensions: set[str] | None = None,
    ) -> None:
        """Initialize :class:`PatchMaskPairDataset`."""
        super().__init__()
        self.pair_transform = pair_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_extensions = image_extensions or DEFAULT_IMAGE_SUFFIXES
        self.mask_extensions = mask_extensions or DEFAULT_IMAGE_SUFFIXES

        self.samples = self._collect_pairs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_paths=image_paths,
            mask_paths=mask_paths,
        )

    def _collect_pairs(
        self: PatchMaskPairDataset,
        image_dir: str | Path | None,
        mask_dir: str | Path | None,
        image_paths: list[str | Path] | None,
        mask_paths: list[str | Path] | None,
    ) -> list[tuple[Path, Path]]:
        """Collect image/mask pairs from directories or explicit lists."""
        if image_paths is not None or mask_paths is not None:
            if image_paths is None or mask_paths is None:
                msg = "Both `image_paths` and `mask_paths` must be provided together."
                raise ValueError(msg)
            if len(image_paths) != len(mask_paths):
                msg = "`image_paths` and `mask_paths` must have the same length."
                raise ValueError(msg)

            pairs = [
                (Path(image_path), Path(mask_path))
                for image_path, mask_path in zip(image_paths, mask_paths)
            ]
            return pairs

        if image_dir is None or mask_dir is None:
            msg = (
                "Provide either (`image_dir`, `mask_dir`) "
                "or (`image_paths`, `mask_paths`)."
            )
            raise ValueError(msg)

        image_root = Path(image_dir)
        mask_root = Path(mask_dir)
        if not image_root.exists() or not image_root.is_dir():
            msg = "`image_dir` must be an existing directory."
            raise ValueError(msg)
        if not mask_root.exists() or not mask_root.is_dir():
            msg = "`mask_dir` must be an existing directory."
            raise ValueError(msg)

        image_files = _discover_files(image_root, self.image_extensions)
        mask_files = _discover_files(mask_root, self.mask_extensions)
        image_map = _build_relative_stem_map(
            image_files,
            image_root,
            kind="image",
        )
        mask_map = _build_relative_stem_map(
            mask_files,
            mask_root,
            kind="mask",
        )

        matched_keys = sorted(set(image_map).intersection(mask_map))
        unmatched_image_keys = set(image_map).difference(mask_map)
        unmatched_mask_keys = set(mask_map).difference(image_map)
        if not matched_keys:
            msg = "No image/mask pairs were found using file stem matching."
            raise ValueError(msg)

        if unmatched_image_keys or unmatched_mask_keys:
            details = []
            if unmatched_image_keys:
                details.append(
                    "missing masks for "
                    f"{_format_key_preview(unmatched_image_keys)}"
                )
            if unmatched_mask_keys:
                details.append(
                    "missing images for "
                    f"{_format_key_preview(unmatched_mask_keys)}"
                )
            msg = f"Unmatched image/mask files were found: {'; '.join(details)}."
            raise ValueError(msg)

        return [(image_map[key], mask_map[key]) for key in matched_keys]

    def __len__(self: PatchMaskPairDataset) -> int:
        """Return number of available pairs."""
        return len(self.samples)

    def __getitem__(self: PatchMaskPairDataset, index: int) -> dict:
        """Get one image/mask sample pair from the dataset."""
        image_path, mask_path = self.samples[index]
        image = _load_image(image_path)
        mask = _load_mask(mask_path)

        if self.pair_transform is not None:
            image, mask = self.pair_transform(image, mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        image_tensor = _ensure_tensor_image(image)
        mask_tensor = _ensure_tensor_mask(mask)

        return {"image": image_tensor, "target": mask_tensor}


class PatchAnnotationDataset(Dataset):
    """Dataset for patch images paired with TIAToolbox annotation stores.

    This dataset supports training targets built dynamically from annotations
    stored in :class:`tiatoolbox.annotation.SQLiteStore`.
    """

    def __init__(
        self: PatchAnnotationDataset,
        patch_inputs: list[str | Path | np.ndarray] | np.ndarray,
        annotation_stores: (
            AnnotationStore
            | str
            | Path
            | list[AnnotationStore | str | Path]
        ),
        target_builder: TargetBuilderABC,
        patch_bounds: (
            list[tuple[float, float, float, float]]
            | np.ndarray
            | None
        ) = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize :class:`PatchAnnotationDataset`."""
        super().__init__()

        self.patch_inputs = self._normalize_patch_inputs(patch_inputs)
        if not self.patch_inputs:
            msg = "`patch_inputs` must contain at least one sample."
            raise ValueError(msg)

        self.annotation_store_specs = self._normalize_annotation_stores(
            annotation_stores,
            len(self.patch_inputs),
        )
        self.target_builder = target_builder

        self.patch_bounds = self._normalize_patch_bounds(patch_bounds)
        if (
            self.patch_bounds is not None
            and len(self.patch_bounds) != len(self.patch_inputs)
        ):
            msg = "`patch_bounds` must have the same length as `patch_inputs`."
            raise ValueError(msg)

        self.pair_transform = pair_transform
        self.image_transform = image_transform
        self.target_transform = target_transform
        self._store_cache: dict[str, AnnotationStore] = {}

    @staticmethod
    def _normalize_patch_inputs(
        patch_inputs: list[str | Path | np.ndarray] | np.ndarray,
    ) -> list[str | Path | np.ndarray]:
        """Normalize supported patch input variants to a list."""
        if isinstance(patch_inputs, np.ndarray):
            if patch_inputs.ndim == 4:
                return [patch for patch in patch_inputs]
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
    def _normalize_annotation_stores(
        annotation_stores: (
            AnnotationStore
            | str
            | Path
            | list[AnnotationStore | str | Path]
        ),
        num_patches: int,
    ) -> list[AnnotationStore | str | Path]:
        """Normalize annotation store input to a per-patch list."""
        return _normalize_repeated_specs(
            annotation_stores,
            num_patches,
            argument_name="annotation_stores",
            reference_name="patch_inputs",
        )

    def __getstate__(self: PatchAnnotationDataset) -> dict:
        """Drop cached runtime handles before pickling dataset state."""
        state = self.__dict__.copy()
        state["_store_cache"] = {}
        return state

    @staticmethod
    def _normalize_patch_bounds(
        patch_bounds: (
            list[tuple[float, float, float, float]]
            | np.ndarray
            | None
        ),
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
        return _open_cached_store(self.annotation_store_specs[index], self._store_cache)

    def __len__(self: PatchAnnotationDataset) -> int:
        """Return number of samples."""
        return len(self.patch_inputs)

    def __getitem__(self: PatchAnnotationDataset, index: int) -> dict:
        """Get a patch image and its target built from paired annotations."""
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
        store = self._get_store(index)
        target = self.target_builder.create_target(
            store=store,
            patch_bounds=bounds,
            output_shape=(height, width),
        )

        if self.pair_transform is not None:
            image, target = self.pair_transform(image, target)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        image_tensor = _ensure_tensor_image(image)
        target_tensor = _ensure_tensor_target(target)

        return {"image": image_tensor, "target": target_tensor}


class SlideAnnotationPatchDataset(Dataset):
    """Dataset reading patches on-the-fly from slides with paired annotation stores."""

    def __init__(  # noqa: PLR0913
        self: SlideAnnotationPatchDataset,
        slide_inputs: list[str | Path | WSIReader],
        annotation_stores: (
            AnnotationStore
            | str
            | Path
            | list[AnnotationStore | str | Path]
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
            | list[
                str
                | Path
                | np.ndarray
                | VirtualWSIReader
                | AnnotationStore
                | None
            ]
            | None
        ) = None,
        min_mask_ratio: float = 0.0,
        store_filter: str | None = None,
        pair_transform: Callable | None = None,
        image_transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize :class:`SlideAnnotationPatchDataset`."""
        super().__init__()
        self.slide_inputs = self._normalize_slide_inputs(slide_inputs)
        self.annotation_store_specs = self._normalize_annotation_stores(
            annotation_stores,
            num_slides=len(self.slide_inputs),
        )
        self.input_mask_specs = self._normalize_input_masks(
            input_masks,
            num_slides=len(self.slide_inputs),
        )

        self.target_builder = target_builder
        self.patch_size = patch_size
        self.stride = stride
        self.resolution = resolution
        self.units = units
        self.within_bound = within_bound
        self.min_mask_ratio = float(min_mask_ratio)
        self.store_filter = store_filter
        self.pair_transform = pair_transform
        self.image_transform = image_transform
        self.target_transform = target_transform

        self._reader_cache: dict[str, WSIReader] = {}
        self._store_cache: dict[str, AnnotationStore] = {}
        (
            self.sample_slide_indices,
            self.sample_bounds,
        ) = self._build_patch_index()
        self._reader_cache = {}
        self._store_cache = {}

    @staticmethod
    def _normalize_slide_inputs(
        slide_inputs: list[str | Path | WSIReader],
    ) -> list[str | Path | WSIReader]:
        """Validate and normalize slide input list."""
        if not isinstance(slide_inputs, list):
            msg = "`slide_inputs` must be a list."
            raise ValueError(msg)
        if not slide_inputs:
            msg = "`slide_inputs` must contain at least one slide."
            raise ValueError(msg)
        return [
            _normalize_runtime_spec_path(slide_spec) or slide_spec
            for slide_spec in slide_inputs
        ]

    @staticmethod
    def _normalize_annotation_stores(
        annotation_stores: (
            AnnotationStore
            | str
            | Path
            | list[AnnotationStore | str | Path]
        ),
        num_slides: int,
    ) -> list[AnnotationStore | str | Path]:
        """Normalize annotation store specs to per-slide entries."""
        return _normalize_repeated_specs(
            annotation_stores,
            num_slides,
            argument_name="annotation_stores",
            reference_name="slide_inputs",
        )

    @staticmethod
    def _normalize_input_masks(
        input_masks: (
            str
            | Path
            | np.ndarray
            | VirtualWSIReader
            | AnnotationStore
            | list[
                str
                | Path
                | np.ndarray
                | VirtualWSIReader
                | AnnotationStore
                | None
            ]
            | None
        ),
        num_slides: int,
    ) -> list[
        str
        | Path
        | np.ndarray
        | VirtualWSIReader
        | AnnotationStore
        | None
    ]:
        """Normalize optional input-mask specs to per-slide entries."""
        return _normalize_repeated_specs(
            input_masks,
            num_slides,
            argument_name="input_masks",
            reference_name="slide_inputs",
        )

    def __getstate__(self: SlideAnnotationPatchDataset) -> dict:
        """Drop cached runtime handles before pickling dataset state."""
        state = self.__dict__.copy()
        state["_reader_cache"] = {}
        state["_store_cache"] = {}
        return state

    def _get_reader(self: SlideAnnotationPatchDataset, index: int) -> WSIReader:
        """Get a cached WSI reader for one slide index."""
        return _open_cached_reader(self.slide_inputs[index], self._reader_cache)

    def _get_store(self: SlideAnnotationPatchDataset, index: int) -> AnnotationStore:
        """Get a cached annotation store for one slide index."""
        return _open_cached_store(self.annotation_store_specs[index], self._store_cache)

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
        slide_index = int(self.sample_slide_indices[index])
        bounds_at_resolution = tuple(
            int(value) for value in self.sample_bounds[index].tolist()
        )

        reader = self._get_reader(slide_index)
        image = reader.read_bounds(
            bounds_at_resolution,
            resolution=self.resolution,
            units=self.units,
            coord_space="resolution",
        )
        height, width = image.shape[:2]

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

        if self.pair_transform is not None:
            image, target = self.pair_transform(image, target)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        image_tensor = _ensure_tensor_image(image)
        target_tensor = _ensure_tensor_target(target)
        bounds_tensor = torch.as_tensor(bounds_at_resolution, dtype=torch.long)
        return {
            "image": image_tensor,
            "target": target_tensor,
            "slide_index": torch.tensor(slide_index, dtype=torch.long),
            "bounds": bounds_tensor,
        }
