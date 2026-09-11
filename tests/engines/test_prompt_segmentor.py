"""Test PromptSegmentor."""

from pathlib import Path

import numpy as np
import pytest

from tiatoolbox.models.engine.prompt_segmentor import PromptSegmentor


def test_prompt_segmentor_calc_mpp_no_scaling() -> None:
    """Test calc_mpp when no scaling is required."""
    segmentor = PromptSegmentor(model=object())

    mpp, scale = segmentor.calc_mpp(
        area_dims=(500, 1000),
        base_mpp=0.5,
        fixed_size=1500,
    )

    assert mpp == 0.5
    assert scale == 1.0
    assert segmentor.scale == 1.0


def test_prompt_segmentor_calc_mpp_with_scaling() -> None:
    """Test calc_mpp when scaling is required."""
    segmentor = PromptSegmentor(model=object())

    mpp, scale = segmentor.calc_mpp(
        area_dims=(3000, 2000),
        base_mpp=0.5,
        fixed_size=1500,
    )

    assert scale == 2.0
    assert mpp == 1.0
    assert segmentor.scale == 2.0


def test_prompt_segmentor_run(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test prompt segmentor inference workflow."""
    called: dict[str, object] = {}

    class FakeSAM:
        """Minimal stand-in for SAM."""

        @staticmethod
        def infer_batch(
            model: object,
            images: list,
            point_coords: np.ndarray | None = None,
            box_coords: np.ndarray | None = None,
            *,
            device: str = "cpu",
        ) -> tuple[np.ndarray, np.ndarray]:
            _ = (
                model,
                images,
                point_coords,
                box_coords,
                device,
            )

            masks = np.array(
                [
                    [
                        [
                            [True, False],
                            [False, True],
                        ],
                    ],
                ],
                dtype=bool,
            )

            scores = np.array([1.0])

            return masks, scores

    def _fake_dict_to_store_semantic_segmentor(
        patch_output: dict,
        scale_factor: tuple[float, float],
        offset: np.ndarray,
        save_path: Path,
        output_type: str,
        ignore_index: int,
    ) -> None:
        """Capture annotation-store calls."""
        called["patch_output"] = patch_output
        called["scale_factor"] = scale_factor
        called["offset"] = offset
        called["save_path"] = save_path
        called["output_type"] = output_type
        called["ignore_index"] = ignore_index

    monkeypatch.setattr(
        "tiatoolbox.models.engine.prompt_segmentor.dict_to_store_semantic_segmentor",
        _fake_dict_to_store_semantic_segmentor,
    )

    segmentor = PromptSegmentor(model=FakeSAM())

    result = segmentor.run(
        images=[np.zeros((10, 10, 3), dtype=np.uint8)],
        point_coords=np.array([[[1, 1]]]),
        save_dir=track_tmp_path,
    )

    expected_path = track_tmp_path / "0.db"

    assert result == [expected_path]

    assert called["save_path"] == expected_path
    assert called["output_type"] == "annotationstore"
    assert called["ignore_index"] == 0


def test_prompt_segmentor_run_multiple_masks(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test processing multiple returned masks."""
    save_paths: list[Path] = []

    class FakeSAM:
        """Minimal stand-in for SAM."""

        @staticmethod
        def infer_batch(
            model: object,
            images: list,
            point_coords: np.ndarray | None = None,
            box_coords: np.ndarray | None = None,
            *,
            device: str = "cpu",
        ) -> tuple[np.ndarray, np.ndarray]:
            _ = (
                model,
                images,
                point_coords,
                box_coords,
                device,
            )

            masks = np.zeros(
                (2, 1, 1, 4, 4),
                dtype=bool,
            )

            scores = np.zeros((2,))

            return masks, scores

    def _fake_dict_to_store_semantic_segmentor(
        patch_output: dict,
        scale_factor: tuple[float, float],
        offset: np.ndarray,
        save_path: Path,
        output_type: str,
        ignore_index: int,
    ) -> None:
        _ = (
            patch_output,
            scale_factor,
            offset,
            output_type,
            ignore_index,
        )

        save_paths.append(save_path)

    monkeypatch.setattr(
        "tiatoolbox.models.engine.prompt_segmentor.dict_to_store_semantic_segmentor",
        _fake_dict_to_store_semantic_segmentor,
    )

    segmentor = PromptSegmentor(model=FakeSAM())

    result = segmentor.run(
        images=[np.zeros((10, 10, 3), dtype=np.uint8)],
        point_coords=np.array([[[1, 1]]]),
        save_dir=track_tmp_path,
    )

    assert result == [
        track_tmp_path / "0.db",
        track_tmp_path / "1.db",
    ]

    assert len(save_paths) == 2
