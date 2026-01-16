"""Unit test package for prompt segmentor."""

from pathlib import Path

import cv2
import numpy as np

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.prompt_segmentor import PromptSegmentor


def test_prompt_segmentor(track_tmp_path: Path) -> None:
    """Test for Prompt Segmentor."""
    # create dummy image patch 256x256x3 with small circle in middle
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, (255, 255, 255), -1)

    expected_area = np.pi * (50**2)

    # prompt with pt in center of circle
    points = np.array([[[128, 128]]], np.uint32)
    boxes = None

    # instantiate prompt segmentor with SAM model
    sam_model = SAM()
    prompt_segmentor = PromptSegmentor(model=sam_model)

    # run prediction
    output_paths = prompt_segmentor.run(
        images=[img],
        point_coords=points,
        box_coords=boxes,
        save_dir=track_tmp_path / "sam_test_output",
        device="cpu",
    )

    assert isinstance(output_paths, list), "Output should be a list of paths"
    assert len(output_paths) == 1, "Output list should contain one path"

    # load the saved annotation db
    store = SQLiteStore(output_paths[0])
    ann = next(iter(store.values()))
    # area should be close to expected area
    assert abs(ann.geometry.area - expected_area) < expected_area * 0.1, (
        "should segment circle area correctly"
    )

    # check with model=None
    prompt_segmentor_default = PromptSegmentor(model=None)
    assert isinstance(prompt_segmentor_default.model, SAM), (
        "Default model should be SAM"
    )
