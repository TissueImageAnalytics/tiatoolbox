"""Datasets used for model training workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from tiatoolbox.utils.misc import imread

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

DEFAULT_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
}


def _load_image(path: Path) -> np.ndarray:
    """Load an image-like array from disk."""
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


class PatchFolderClassificationDataset(Dataset):
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
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            msg = "`root_dir` must be an existing directory."
            raise ValueError(msg)

        self.transform = transform
        self.target_transform = target_transform
        self.file_extensions = file_extensions or DEFAULT_IMAGE_SUFFIXES

        class_dirs = sorted([path for path in self.root_dir.iterdir() if path.is_dir()])
        if not class_dirs:
            msg = "`root_dir` does not contain any class directories."
            raise ValueError(msg)

        if class_to_idx is None:
            class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}

        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            class_index = self.class_to_idx[class_name]
            for image_path in _discover_files(class_dir, self.file_extensions):
                self.samples.append((image_path, class_index))

        if not self.samples:
            msg = "No training samples were found under `root_dir`."
            raise ValueError(msg)

    def __len__(self: PatchFolderClassificationDataset) -> int:
        """Return number of discovered samples."""
        return len(self.samples)

    def __getitem__(self: PatchFolderClassificationDataset, index: int) -> dict:
        """Get one sample from the dataset."""
        image_path, target = self.samples[index]
        image = _load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        image_tensor = _ensure_tensor_image(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

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
            for image_path, mask_path in pairs:
                if not image_path.exists() or not mask_path.exists():
                    msg = "All image and mask paths must exist."
                    raise ValueError(msg)
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
        image_map = {image_path.stem: image_path for image_path in image_files}
        mask_map = {mask_path.stem: mask_path for mask_path in mask_files}

        matched_keys = sorted(set(image_map).intersection(mask_map))
        pairs = [(image_map[key], mask_map[key]) for key in matched_keys]
        if not pairs:
            msg = "No image/mask pairs were found using file stem matching."
            raise ValueError(msg)
        return pairs

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


def create_dataset(
    dataset_type: Literal[
        "patch_folder_classification",
        "patch_mask_pair",
    ],
    **kwargs: dict,
) -> Dataset:
    """Create a training dataset instance from a dataset type key."""
    if dataset_type == "patch_folder_classification":
        return PatchFolderClassificationDataset(**kwargs)
    if dataset_type == "patch_mask_pair":
        return PatchMaskPairDataset(**kwargs)

    msg = (
        "Unsupported `dataset_type`. "
        "Supported values are: `patch_folder_classification`, `patch_mask_pair`."
    )
    raise ValueError(msg)
