"""Unit test package for SAM."""

from pathlib import Path
from typing import Callable

from tiatoolbox.models import SAM
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread

ON_GPU = toolbox_env.has_gpu()

# Test pretrained Model =============================


def test_functional_sam(
    remote_sample: Callable,
) -> None:
    """Test for SAM."""
    # convert to pathlib Path to prevent wsireader complaint
    tile_path = Path(remote_sample("patch-extraction-vf"))
    img = imread(tile_path)

    # test creation
    _ = SAM()

    # test inference
    # create prompts

    points1 = [[[64, 64]]]
    points2 = [[[64, 64], [128, 128]]]
    boxes1 = [[[64, 64, 128, 128]]]

    model = SAM()

    _ = model.infer_batch(model, img, on_gpu=ON_GPU)  # no prompts
    _ = model.infer_batch(model, img, points1, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, points2, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, box_coords=boxes1, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, points2, boxes1, on_gpu=ON_GPU)
