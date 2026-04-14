"""Quality-check KongNet CoNIC instance post-processing on a few patch examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture.kongnet import (
    KongNetComponentMaps,
    KongNet,
    kongnet_instance_postproc,
)
from tiatoolbox.models.models_abc import load_torch_model
from tiatoolbox.models.training import (
    BoundaryTargetBuilder,
    CompositeTargetBuilder,
    GaussianHeatmapTargetBuilder,
    MaskTargetBuilder,
    StackedTargetBuilder,
)

CONIC_CLASS_DICT = {
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}

CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([228, 26, 28], dtype=np.uint8),
    2: np.array([55, 126, 184], dtype=np.uint8),
    3: np.array([77, 175, 74], dtype=np.uint8),
    4: np.array([152, 78, 163], dtype=np.uint8),
    5: np.array([255, 127, 0], dtype=np.uint8),
    6: np.array([166, 86, 40], dtype=np.uint8),
}

ORDERED_HEAD_KEYS = tuple(label.lower() for label in CONIC_CLASS_DICT.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run KongNet full-head patch inference, apply deterministic instance "
            "post-processing, and save CoNIC QC figures."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/media/mark-eastwood/Work/Data/Conic"),
        help="CoNIC dataset root containing `images_png/` and `annotation_stores/`.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("examples/tmp/15-kongnet-conic-smoke/best_model_weights.pth"),
        help="Path to a local KongNet CoNIC checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/tmp/kongnet-conic-instance-postproc-qc"),
        help="Directory in which QC figures and arrays will be written.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Numeric patch ids to quality-check.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use for inference.",
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        default="predicted",
        choices=["predicted", "target"],
        help="Use model predictions or exact KongNet-style target maps as input.",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Override the centroid peak minimum-distance parameter.",
    )
    parser.add_argument(
        "--centroid-threshold",
        type=float,
        default=None,
        help="Override the absolute centroid peak threshold.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Foreground threshold applied to the max KongNet mask probability.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="Boundary penalty weight applied inside the watershed score map.",
    )
    parser.add_argument(
        "--min-instance-size",
        type=int,
        default=8,
        help="Drop predicted instances smaller than this many pixels.",
    )
    parser.add_argument(
        "--class-assignment",
        type=str,
        default="mean_mask",
        choices=["mean_mask", "seed"],
        help="How final instance classes are assigned after watershed.",
    )
    return parser.parse_args()


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _normalize_ring(ring: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    coords = np.round(np.asarray(ring), decimals=0).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, shape[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, shape[0] - 1)
    return coords.reshape((-1, 1, 2))


def _annotation_store_to_maps(
    store_path: Path,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    instance_map = np.zeros(shape, dtype=np.int32)
    class_map = np.zeros(shape, dtype=np.int32)
    store = SQLiteStore.open(store_path)
    try:
        for _, annotation in store.items():
            instance_id = int(annotation.properties.get("instance_id", 0))
            class_id = int(annotation.properties.get("class_id", 0))
            coords = annotation.coords
            if not isinstance(coords, list) or not coords:
                continue

            outer = _normalize_ring(coords[0], shape)
            cv2.fillPoly(instance_map, [outer], instance_id)
            cv2.fillPoly(class_map, [outer], class_id)
            for hole in coords[1:]:
                cv2.fillPoly(instance_map, [_normalize_ring(hole, shape)], 0)
                cv2.fillPoly(class_map, [_normalize_ring(hole, shape)], 0)
    finally:
        store.close()
    return instance_map, class_map


def _overlay_instance_map(
    image: np.ndarray,
    instance_map: np.ndarray,
    class_map: np.ndarray,
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        mask = instance_map == instance_id
        class_id = int(class_map[mask][0]) if np.any(mask) else 0
        color = CLASS_COLORS.get(class_id, np.array([255, 255, 255], dtype=np.uint8))
        overlay[mask] = (0.65 * overlay[mask]) + (0.35 * color)

        contour_mask = ndimage.binary_dilation(mask) ^ mask
        overlay[contour_mask] = color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _seed_points_from_peak_map(peak_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_y, peak_x, peak_class = np.where(peak_map > 0)
    return peak_y, peak_x, peak_class


def _build_model(weights: Path, device: str) -> KongNet:
    model = KongNet(
        num_heads=6,
        num_channels_per_head=[3, 3, 3, 3, 3, 3],
        target_channels=[2, 5, 8, 11, 14, 17],
        min_distance=5,
        threshold_abs=0.35,
        class_dict=CONIC_CLASS_DICT,
    )
    model = load_torch_model(model, weights)
    model = model.to(device)
    model.eval()
    return model


def _build_target_builder() -> CompositeTargetBuilder:
    return CompositeTargetBuilder(
        {
            label.lower(): StackedTargetBuilder(
                {
                    "mask": MaskTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        class_mapping=None,
                        default_label=0,
                    ),
                    "boundary": BoundaryTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        line_width=1,
                    ),
                    "centroid": GaussianHeatmapTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        sigma=2.0,
                    ),
                }
            )
            for label in CONIC_CLASS_DICT.values()
        }
    )


def _component_maps_from_target_builder(
    store_path: Path,
    shape: tuple[int, int],
    target_builder: CompositeTargetBuilder,
) -> KongNetComponentMaps:
    store = SQLiteStore.open(store_path)
    try:
        targets = target_builder.create_target(
            store=store,
            patch_bounds=(0, 0, shape[1], shape[0]),
            output_shape=shape,
        )
    finally:
        store.close()

    return KongNetComponentMaps(
        mask=np.stack([targets[key][..., 0] for key in ORDERED_HEAD_KEYS], axis=-1),
        boundary=np.stack([targets[key][..., 1] for key in ORDERED_HEAD_KEYS], axis=-1),
        centroid=np.stack([targets[key][..., 2] for key in ORDERED_HEAD_KEYS], axis=-1),
    )


def _infer_patch(model: KongNet, image: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(KongNet.preproc(image).astype(np.float32)[None, ...])
    tensor = tensor.permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        return model(tensor)[0].detach().cpu()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.weights.exists():
        msg = f"Checkpoint not found: {args.weights}"
        raise FileNotFoundError(msg)

    model = _build_model(args.weights, args.device)
    target_builder = _build_target_builder() if args.source_mode == "target" else None
    summary: list[dict[str, int | float | str]] = []
    min_distance = model.min_distance if args.min_distance is None else args.min_distance
    centroid_threshold = (
        model.threshold_abs if args.centroid_threshold is None else args.centroid_threshold
    )

    for index in args.indices:
        patch_name = f"{index:05d}"
        image_path = args.dataset_root / "images_png" / f"{patch_name}.png"
        store_path = args.dataset_root / "annotation_stores" / f"{patch_name}.db"
        image = _load_image(image_path)
        gt_instance_map, gt_class_map = _annotation_store_to_maps(store_path, image.shape[:2])
        if args.source_mode == "predicted":
            logits = _infer_patch(model, image, args.device)
            component_maps = model.extract_component_maps(logits, from_logits=True)
        else:
            component_maps = _component_maps_from_target_builder(
                store_path,
                image.shape[:2],
                target_builder=target_builder,
            )
        result = kongnet_instance_postproc(
            component_maps,
            class_ids=model.resolve_instance_class_ids(),
            min_distance=min_distance,
            threshold_abs=centroid_threshold,
            threshold_rel=None,
            mask_threshold=args.mask_threshold,
            boundary_weight=args.boundary_weight,
            class_assignment=args.class_assignment,
            min_instance_size=args.min_instance_size,
        )

        pred_instances = int(len(result.instance_classes))
        gt_instances = int(len(np.unique(gt_instance_map)) - 1)
        summary.append(
            {
                "patch": patch_name,
                "source_mode": args.source_mode,
                "pred_instances": pred_instances,
                "gt_instances": gt_instances,
                "peak_count": int(np.count_nonzero(np.max(result.peak_map, axis=-1))),
                "max_mask_mean": float(np.max(component_maps.mask, axis=-1).mean()),
                "max_boundary_mean": float(np.max(component_maps.boundary, axis=-1).mean()),
                "max_centroid_mean": float(np.max(component_maps.centroid, axis=-1).mean()),
            }
        )

        pred_overlay = _overlay_instance_map(image, result.instance_map, result.class_map)
        gt_overlay = _overlay_instance_map(image, gt_instance_map, gt_class_map)
        peak_y, peak_x, peak_class = _seed_points_from_peak_map(result.peak_map)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        axes[0].imshow(image)
        axes[0].set_title(f"Patch {patch_name} ({args.source_mode})")
        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"GT instances ({gt_instances})")
        axes[2].imshow(pred_overlay)
        axes[2].set_title(f"Pred instances ({pred_instances})")
        axes[3].imshow(np.max(component_maps.mask, axis=-1), cmap="gray", vmin=0.0, vmax=1.0)
        axes[3].set_title("Max mask prob")
        axes[4].imshow(np.max(component_maps.boundary, axis=-1), cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].set_title("Max boundary prob")
        axes[5].imshow(np.max(component_maps.centroid, axis=-1), cmap="magma", vmin=0.0, vmax=1.0)
        if len(peak_x) > 0:
            peak_colors = [CLASS_COLORS.get(int(cls + 1), CLASS_COLORS[0]) / 255.0 for cls in peak_class]
            axes[5].scatter(peak_x, peak_y, c=peak_colors, s=16)
        axes[5].set_title("Max centroid prob + seeds")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{patch_name}_qc.png", bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            args.output_dir / f"{patch_name}_maps.npz",
            pred_instance_map=result.instance_map,
            pred_class_map=result.class_map,
            gt_instance_map=gt_instance_map,
            gt_class_map=gt_class_map,
            max_mask=np.max(component_maps.mask, axis=-1),
            max_boundary=np.max(component_maps.boundary, axis=-1),
            max_centroid=np.max(component_maps.centroid, axis=-1),
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
