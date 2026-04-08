"""Convert CoNIC `.npy` arrays into PNG patches and annotation stores.

This script expects the standard CoNIC training arrays:

- `images.npy` with shape `(N, H, W, 3)`
- `labels.npy` with shape `(N, H, W, 2)`

The labels array is assumed to use:

- `labels[..., 0]`: instance ids
- `labels[..., 1]`: class ids

For each sample, the script writes:

- one PNG image patch
- one SQLite annotation store (`.db`) containing one annotation per nucleus

Example:

```bash
python examples/prepare_conic_training_data.py \
  --input-dir /media/mark-eastwood/Work/Data/Conic
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.utils.misc import process_contours

CONIC_CLASS_MAP = {
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert CoNIC images.npy and labels.npy into PNG patches and "
            "SQLite annotation stores."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing `images.npy` and `labels.npy`.",
    )
    parser.add_argument(
        "--images-out",
        type=Path,
        default=None,
        help="Output directory for PNG patches. Defaults to <input-dir>/images_png.",
    )
    parser.add_argument(
        "--stores-out",
        type=Path,
        default=None,
        help=(
            "Output directory for annotation stores. Defaults to "
            "<input-dir>/annotation_stores."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to convert from the start of the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing PNGs or `.db` stores.",
    )
    return parser.parse_args()


def _validate_arrays(images: np.ndarray, labels: np.ndarray) -> None:
    """Validate the expected CoNIC array layout."""
    if images.ndim != 4 or images.shape[-1] != 3:
        msg = "`images.npy` must have shape `(N, H, W, 3)`."
        raise ValueError(msg)
    if labels.ndim != 4 or labels.shape[-1] != 2:
        msg = "`labels.npy` must have shape `(N, H, W, 2)`."
        raise ValueError(msg)
    if images.shape[:3] != labels.shape[:3]:
        msg = "`images.npy` and `labels.npy` must agree on sample and spatial shape."
        raise ValueError(msg)


def _save_png(image: np.ndarray, output_path: Path) -> None:
    """Save one RGB patch as a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_path),
        cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR),
    )
    if not success:
        msg = f"Failed to write PNG image to `{output_path}`."
        raise RuntimeError(msg)


def _instance_annotations(
    instance_map: np.ndarray,
    class_map: np.ndarray,
) -> tuple[list, list[str]]:
    """Convert one CoNIC instance/class map pair into annotations and keys."""
    annotations = []
    annotation_keys: list[str] = []

    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue

        instance_mask = (instance_map == instance_id).astype(np.uint8)
        class_ids = np.unique(class_map[instance_mask.astype(bool)])
        class_ids = class_ids[class_ids != 0]
        if len(class_ids) != 1:
            msg = (
                f"Instance `{int(instance_id)}` does not map to exactly one non-zero "
                "CoNIC class id."
            )
            raise ValueError(msg)

        class_id = int(class_ids[0])
        class_name = CONIC_CLASS_MAP.get(class_id, f"class_{class_id}")
        contours, hierarchy = cv2.findContours(
            instance_mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE,
        )
        if hierarchy is None:
            continue

        instance_annotations = process_contours(
            contours=contours,
            hierarchy=hierarchy,
            properties={
                "type": class_name,
                "class": class_name,
                "class_id": class_id,
                "instance_id": int(instance_id),
                "dataset": "CoNIC",
            },
        )
        for annotation_index, annotation in enumerate(instance_annotations):
            annotations.append(annotation)
            annotation_keys.append(f"{int(instance_id)}_{annotation_index}")

    return annotations, annotation_keys


def convert_conic_dataset(
    input_dir: Path,
    *,
    images_out: Path | None = None,
    stores_out: Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    """Convert CoNIC arrays into a PNG folder and a store-per-image folder."""
    images_path = input_dir / "images.npy"
    labels_path = input_dir / "labels.npy"
    if not images_path.exists() or not labels_path.exists():
        msg = f"`{input_dir}` must contain both `images.npy` and `labels.npy`."
        raise ValueError(msg)

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    _validate_arrays(images, labels)

    images_out = input_dir / "images_png" if images_out is None else images_out
    stores_out = (
        input_dir / "annotation_stores" if stores_out is None else stores_out
    )
    images_out.mkdir(parents=True, exist_ok=True)
    stores_out.mkdir(parents=True, exist_ok=True)

    total_samples = int(images.shape[0])
    if limit is not None:
        if limit <= 0:
            msg = "`limit` must be a positive integer when provided."
            raise ValueError(msg)
        total_samples = min(total_samples, int(limit))

    name_width = max(5, len(str(total_samples - 1)))
    for sample_index in tqdm(range(total_samples), desc="Converting CoNIC"):
        image_output_path = images_out / f"{sample_index:0{name_width}d}.png"
        store_output_path = stores_out / f"{sample_index:0{name_width}d}.db"

        if (
            not overwrite
            and image_output_path.exists()
            and store_output_path.exists()
        ):
            continue

        image = np.asarray(images[sample_index])
        sample_labels = np.asarray(labels[sample_index])
        instance_map = sample_labels[..., 0]
        class_map = sample_labels[..., 1]

        _save_png(image, image_output_path)
        annotations, annotation_keys = _instance_annotations(instance_map, class_map)

        if store_output_path.exists():
            store_output_path.unlink()
        store = SQLiteStore(store_output_path, auto_commit=False)
        if annotations:
            store.append_many(annotations, annotation_keys)
        store.commit()
        store.close()


def main() -> None:
    """Run the CLI entrypoint."""
    args = _parse_args()
    convert_conic_dataset(
        args.input_dir,
        images_out=args.images_out,
        stores_out=args.stores_out,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
