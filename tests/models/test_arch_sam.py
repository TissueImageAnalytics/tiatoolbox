"""Unit test package for SAM."""

from pathlib import Path
from typing import Callable

from tiatoolbox.models import SAM
from tiatoolbox.models.architecture.sam import SAMPrompts
from tiatoolbox.utils import imread

ON_GPU = False

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

    prompts1 = SAMPrompts(point_coords=[[64, 64]])
    prompts2 = SAMPrompts(point_coords=[[64, 64]], point_labels=[1])
    prompts3 = SAMPrompts(box_coords=[[64, 64, 128, 128]])
    prompts4 = SAMPrompts(
        point_coords=[[64, 64]], point_labels=[1], box_coords=[[64, 64, 128, 128]]
    )

    model = SAM()

    _ = model.infer_batch(model, img, on_gpu=ON_GPU)  # no prompts
    _ = model.infer_batch(model, img, prompts=prompts1, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, prompts=prompts2, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, prompts=prompts3, on_gpu=ON_GPU)
    _ = model.infer_batch(model, img, prompts=prompts4, on_gpu=ON_GPU)
