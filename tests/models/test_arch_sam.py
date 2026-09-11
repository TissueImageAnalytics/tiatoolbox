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


def test_sam_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test SAM initialization."""
    fake_model = object()
    fake_processor = object()

    class FakeLoadedModel:
        """Fake SAM model."""

        def to(self, device: str) -> object:
            """Return fake model."""
            assert device == "cpu"
            return fake_model

    def _fake_model_from_pretrained(
        model_path: str,
    ) -> FakeLoadedModel:
        """Return fake SAM model."""
        assert model_path == "fake-model"
        return FakeLoadedModel()

    def _fake_processor_from_pretrained(
        model_path: str,
    ) -> object:
        """Return fake SAM processor."""
        assert model_path == "fake-model"
        return fake_processor

    monkeypatch.setattr(
        "tiatoolbox.models.architecture.sam.SamModel.from_pretrained",
        _fake_model_from_pretrained,
    )

    monkeypatch.setattr(
        "tiatoolbox.models.architecture.sam.SamProcessor.from_pretrained",
        _fake_processor_from_pretrained,
    )

    sam = SAM(
        model_path="fake-model",
        device="cpu",
    )

    assert sam.net_name == "SAM"
    assert sam.device == "cpu"
    assert sam.model is fake_model
    assert sam.processor is fake_processor


def test_sam_encode_image() -> None:
    """Test image encoding pipeline."""
    sam = SAM.__new__(SAM)

    sam.device = "cpu"

    processed = {
        "original_sizes": torch.tensor([[100, 200]]),
        "reshaped_input_sizes": torch.tensor([[64, 64]]),
        "pixel_values": torch.ones((1, 3, 64, 64)),
    }

    class FakeProcessed(dict):
        """Fake processor output."""

        def to(self, device: str) -> "FakeProcessed":
            """Return self after device transfer."""
            assert device == "cpu"
            return self

    class FakeProcessor:
        """Fake SAM processor."""

        def __call__(
            self,
            image: np.ndarray,
            return_tensors: str,
        ) -> FakeProcessed:
            """Return encoded image."""
            _ = image

            assert return_tensors == "pt"

            return FakeProcessed(processed)

    class FakeModel:
        """Fake SAM model."""

        @staticmethod
        def get_image_embeddings(
            pixel_values: torch.Tensor,
        ) -> torch.Tensor:
            """Return fake embeddings."""
            assert torch.equal(
                pixel_values,
                processed["pixel_values"],
            )

            return torch.tensor([[123]])

    sam.processor = FakeProcessor()
    sam.model = FakeModel()

    image = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    embeddings, original_sizes, reshaped_sizes = sam._encode_image(
        image,
    )

    assert torch.equal(
        embeddings,
        torch.tensor([[123]]),
    )

    assert torch.equal(
        original_sizes,
        processed["original_sizes"],
    )

    assert torch.equal(
        reshaped_sizes,
        processed["reshaped_input_sizes"],
    )


def test_sam_process_prompts() -> None:
    """Test SAM prompt processing."""
    sam = SAM.__new__(SAM)

    sam.device = "cpu"

    image_masks = np.array([[1]])
    image_scores = torch.tensor([[0.99]])

    captured: dict[str, object] = {}

    class FakeInputs(dict):
        """Fake processor outputs."""

        def to(self, device: str) -> "FakeInputs":
            """Fake tensor transfer."""
            assert device == "cpu"
            return self

    class FakeOutputs:
        """Fake SAM outputs."""

        def __init__(self) -> None:
            self.pred_masks = torch.ones((1, 1, 4, 4))
            self.iou_scores = image_scores

    class FakeImageProcessor:
        """Fake image processor."""

        @staticmethod
        def post_process_masks(
            pred_masks: torch.Tensor,
            original_sizes: torch.Tensor,
            reshaped_input_sizes: torch.Tensor,
        ) -> np.ndarray:
            """Return fake masks."""
            _ = (
                pred_masks,
                original_sizes,
                reshaped_input_sizes,
            )

            return image_masks

    class FakeProcessor:
        """Fake SAM processor."""

        image_processor = FakeImageProcessor()

        def __call__(
            self,
            image: object,
            input_points: list | None = None,
            input_labels: list | None = None,
            input_boxes: list | None = None,
            return_tensors: str = "pt",
        ) -> FakeInputs:
            """Return fake processor inputs."""
            _ = image
            captured["input_points"] = input_points
            captured["input_labels"] = input_labels
            captured["input_boxes"] = input_boxes

            assert return_tensors == "pt"

            return FakeInputs(
                {
                    "pixel_values": torch.ones((1, 3, 4, 4)),
                },
            )

    class FakeModel:
        """Fake SAM model."""

        def __call__(
            self,
            **kwargs: object,
        ) -> FakeOutputs:
            """Capture forwarded kwargs."""
            captured["forward_kwargs"] = kwargs

            assert kwargs["multimask_output"] is False
            assert "pixel_values" not in kwargs
            assert "image_embeddings" in kwargs

            return FakeOutputs()

    sam.processor = FakeProcessor()
    sam.model = FakeModel()

    embeddings = torch.tensor([[1]])
    original_sizes = torch.tensor([[100, 200]])
    reshaped_sizes = torch.tensor([[64, 64]])

    masks, scores = sam._process_prompts(
        image=[np.zeros((4, 4, 3), dtype=np.uint8)],
        embeddings=embeddings,
        orig_sizes=original_sizes,
        reshaped_sizes=reshaped_sizes,
        points=[[[1, 1]]],
        boxes=None,
        point_labels=[[[1]]],
    )

    assert np.array_equal(masks, image_masks)
    assert torch.equal(scores, image_scores)

    assert captured["input_points"] == [[[1, 1]]]
    assert captured["input_boxes"] is None


def test_sam_to_updates_device() -> None:
    """Test SAM.to updates device."""
    sam = SAM.__new__(SAM)
    torch.nn.Module.__init__(sam)

    class FakeModel:
        """Fake model."""

        def __init__(self) -> None:
            self.called_device: str | None = None

        def to(self, device: str) -> "FakeModel":
            """Record device."""
            self.called_device = device
            return self

    fake_model = FakeModel()

    sam.model = fake_model
    sam.device = "cpu"

    result = sam.to(device="cpu")

    assert result is sam
    assert sam.device == "cpu"
    assert fake_model.called_device == "cpu"


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
    assert captured["boxes"] is None
    assert captured["points"] is not None
    assert captured["point_labels"] == [[[1], [1]]]

    assert isinstance(masks, np.ndarray)
    assert isinstance(scores, np.ndarray)

    assert masks.shape == (1, 1, 8, 8)
    assert scores.shape == (1, 1, 1)


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
