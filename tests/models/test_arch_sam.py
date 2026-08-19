"""Unit test package for SAM."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread
from tiatoolbox.utils.misc import select_device

ON_GPU = toolbox_env.has_gpu()
_RUNNING_ON_CI = toolbox_env.running_on_ci()


def test_sam_preproc_torch_tensor() -> None:
    """Test SAM pre-processing for PyTorch tensor input."""
    image = torch.arange(
        4 * 2 * 3,
        dtype=torch.float32,
    ).reshape(4, 2, 3)

    result = SAM.preproc(image)

    assert isinstance(result, np.ndarray)

    # CHW -> HWC and alpha channel removed.
    assert result.shape == (2, 3, 3)

    expected = image.permute(1, 2, 0).cpu().numpy()[..., :3]

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_sam_preproc_numpy_array_with_alpha() -> None:
    """Test SAM pre-processing for NumPy input."""
    image = np.zeros(
        (8, 8, 4),
        dtype=np.uint8,
    )

    result = SAM.preproc(image)

    assert result.shape == (8, 8, 3)


def test_sam_infer_batch_requires_prompts() -> None:
    """Test infer_batch raises when no prompts are provided."""
    model = torch.nn.Identity()

    batch_data = [
        np.zeros(
            (8, 8, 3),
            dtype=np.uint8,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="At least one of point_coords or box_coords must be provided",
    ):
        SAM.infer_batch(
            model=model,
            batch_data=batch_data,
            point_coords=None,
            box_coords=None,
            device="cpu",
        )


def test_sam_forward_point_prompts_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test SAM forward pass using point prompts only."""
    sam = SAM.__new__(SAM)

    captured: dict[str, object] = {}

    def _fake_encode_image(
        image: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fake embeddings and size metadata."""
        _ = image

        return (
            torch.zeros((1, 1)),
            torch.tensor([[8, 8]]),
            torch.tensor([[8, 8]]),
        )

    def _fake_process_prompts(
        image: list,
        embeddings: torch.Tensor,
        orig_sizes: torch.Tensor,
        reshaped_sizes: torch.Tensor,
        points: list | None,
        boxes: list | None,
        point_labels: list | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capture point prompt inputs."""
        _ = (
            image,
            embeddings,
            orig_sizes,
            reshaped_sizes,
        )

        captured["points"] = points
        captured["boxes"] = boxes
        captured["point_labels"] = point_labels

        return (
            np.ones((1, 8, 8), dtype=np.uint8),
            np.array([[0.95]], dtype=np.float32),
        )

    monkeypatch.setattr(
        sam,
        "_encode_image",
        _fake_encode_image,
    )

    monkeypatch.setattr(
        sam,
        "_process_prompts",
        _fake_process_prompts,
    )

    image = np.zeros(
        (8, 8, 3),
        dtype=np.uint8,
    )

    point_coords = [
        np.array(
            [
                [2, 3],
                [4, 5],
            ],
            dtype=np.int32,
        ),
    ]

    masks, scores = sam.forward(
        imgs=[image],
        point_coords=point_coords,
        box_coords=None,
    )

    #
    # box_coords branch skipped
    #
    assert captured["boxes"] is None

    #
    # point_coords branch executed
    #
    assert captured["points"] is not None
    assert captured["point_labels"] == [[[1], [1]]]

    assert masks.shape[2] == 1
    assert scores.shape[2] == 1


# Test pretrained Model =============================
@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
def test_functional_sam(remote_sample: Callable) -> None:
    """Test for SAM."""
    # convert to pathlib Path to prevent wsireader complaint
    tile_path = Path(remote_sample("patch-extraction-vf"))
    img = imread(tile_path)

    # test creation

    model = SAM(device=select_device(on_gpu=ON_GPU))

    # create image patch and prompts
    points = np.array([[[64, 64]]])
    boxes = np.array([[[64, 64, 128, 128]]])

    # test preproc
    tensor = torch.from_numpy(img)
    patch = np.expand_dims(model.preproc(tensor), axis=0)
    patch = model.preproc(patch)

    # test inference

    mask_output, score_output = model.infer_batch(
        model, patch, points, device=select_device(on_gpu=ON_GPU)
    )

    assert mask_output is not None, "Output should not be None"
    assert len(mask_output) > 0, "Output should have at least one element"
    assert len(score_output) > 0, "Output should have at least one element"

    mask_output, score_output = model.infer_batch(
        model, patch, box_coords=boxes, device=select_device(on_gpu=ON_GPU)
    )

    assert len(mask_output) > 0, "Output should have at least one element"
    assert len(score_output) > 0, "Output should have at least one element"

    # test error when no prompts provided
    with pytest.raises(
        ValueError,
        match=r"At least one of point_coords or box_coords must be provided.",
    ):
        _ = model.infer_batch(model, patch, device=select_device(on_gpu=ON_GPU))
