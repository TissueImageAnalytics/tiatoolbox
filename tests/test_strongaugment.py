"""Tests for the StrongAugment port."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tiatoolbox.tools.strongaugment import StrongAugment, get_augment_space

ALL_OPERATIONS = {**get_augment_space(), "solarize": (0, 256), "posterize": (1, 8)}


@pytest.fixture
def image() -> np.ndarray:
    """Return a random RGB uint8 image."""
    return np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)


def test_defaults_leave_out_extreme_operations() -> None:
    """Defaults match the training preset and leave out solarize and posterize."""
    augmentor = StrongAugment()
    assert augmentor.operations == (0, 2, 3, 4)
    assert augmentor.probabilities == (0.3, 0.3, 0.3, 0.1)
    assert augmentor.augment_space == get_augment_space()
    assert not {"solarize", "posterize"} & set(augmentor.augment_space)
    assert "probabilities=(0.3, 0.3, 0.3, 0.1)" in repr(augmentor)


@pytest.mark.parametrize("name", list(ALL_OPERATIONS))
def test_every_operation_keeps_shape_dtype_and_input(
    name: str, image: np.ndarray
) -> None:
    """Each operation, including posterize under NumPy 2, returns a new image."""
    augmentor = StrongAugment(
        operations=(1,),
        probabilities=(1.0,),
        augment_space={name: ALL_OPERATIONS[name]},
    )
    original = image.copy()
    for seed in range(5):
        operations = augmentor.sample_operations(np.random.default_rng(seed))
        assert [operation for operation, _ in operations] == [name]
        result = augmentor.apply(image, operations)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("red", {"magnitude": 1.0}),
        ("green", {"magnitude": 1.0}),
        ("blue", {"magnitude": 1.0}),
        ("saturation", {"magnitude": 1.0}),
        ("brightness", {"magnitude": 1.0}),
        ("contrast", {"magnitude": 1.0}),
        ("gamma", {"magnitude": 1.0}),
        ("sharpen", {"magnitude": 0.0}),
        ("emboss", {"magnitude": 0.0}),
        ("blur", {"magnitude": 0.0}),
        ("noise", {"magnitude": 0.0, "seed": 1}),
        ("solarize", {"magnitude": 256}),
        ("posterize", {"magnitude": 8}),
        ("tone", {"magnitude_0": 1 / 3, "magnitude_1": 2 / 3}),
    ],
)
def test_neutral_magnitudes_leave_images_unchanged(
    name: str, parameters: dict[str, float], image: np.ndarray
) -> None:
    """Magnitudes at the identity point are exact no-ops, including gamma 1.0."""
    np.testing.assert_array_equal(
        StrongAugment.apply(image, [(name, parameters)]), image
    )


def test_zero_magnitudes_remove_intensity(image: np.ndarray) -> None:
    """Zero channel and brightness scales give black; zero gamma gives white."""
    for channel, name in enumerate(("red", "green", "blue")):
        result = StrongAugment.apply(image, [(name, {"magnitude": 0.0})])
        assert not result[..., channel].any()
    assert not StrongAugment.apply(image, [("brightness", {"magnitude": 0.0})]).any()
    gamma = StrongAugment.apply(image, [("gamma", {"magnitude": 0.0})])
    assert (gamma == 255).all()


def test_image_types_and_greyscale_round_trip(image: np.ndarray) -> None:
    """PIL and NumPy, RGB and greyscale inputs come back in the same form."""
    augmentor = StrongAugment(operations=(3,), probabilities=(1.0,), seed=0)
    assert augmentor(Image.fromarray(image)).mode == "RGB"
    grey = np.ascontiguousarray(image[..., 0])
    assert augmentor(grey).shape == grey.shape
    assert augmentor(Image.fromarray(grey)).mode == "L"


def test_seeds_and_generators_reproduce_results(image: np.ndarray) -> None:
    """A seed replays a sequence, and a per-call generator overrides the seed."""
    first, second = StrongAugment(seed=3), StrongAugment(seed=3)
    for _ in range(10):
        np.testing.assert_array_equal(first(image), second(image))
    np.testing.assert_array_equal(
        StrongAugment(seed=1)(image, rng=np.random.default_rng(5)),
        StrongAugment(seed=2)(image, rng=np.random.default_rng(5)),
    )
    noise = StrongAugment(
        operations=(1,), probabilities=(1.0,), augment_space={"noise": (0.1, 0.2)}
    )
    operations = noise.sample_operations()
    np.testing.assert_array_equal(
        noise.apply(image, operations), noise.apply(image, operations)
    )


def test_sampling_draws_distinct_operations_and_continuous_magnitudes() -> None:
    """Counts follow `operations`; float operations stay continuous for int bounds."""
    augmentor = StrongAugment(operations=(0, 4), probabilities=(0.5, 0.5), seed=1)
    counts = set()
    for _ in range(30):
        names = [name for name, _ in augmentor.sample_operations()]
        assert len(set(names)) == len(names)
        counts.add(len(names))
    assert counts == {0, 4}
    red = StrongAugment(
        operations=(1,), probabilities=(1.0,), augment_space={"red": (1, 2)}, seed=0
    )
    magnitudes = {red.sample_operations()[0][1]["magnitude"] for _ in range(20)}
    assert len(magnitudes) == 20
    assert all(1 <= magnitude <= 2 for magnitude in magnitudes)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"augment_space": {"posterizzze": (1, 2)}}, ValueError, "Unsupported"),
        ({"augment_space": {"red": 1.0}}, TypeError, "pair"),
        ({"augment_space": {"red": (True, False)}}, TypeError, "numbers"),
        ({"augment_space": {"jpeg": (0.5, 1.0)}}, TypeError, "integers"),
        ({"augment_space": {"equalize": (1, 2)}}, TypeError, "True, True"),
        ({"augment_space": {"posterize": (0, 2)}}, ValueError, "1 <= low"),
        ({"augment_space": {"hue": (-0.6, 0.0)}}, ValueError, "-0.5 <= low"),
        ({"augment_space": {"red": (2.0, 1.0)}}, ValueError, "low <= high"),
        ({"operations": (1, 2), "probabilities": (1.0,)}, ValueError, "same length"),
        ({"operations": (1,), "probabilities": (0.5,)}, ValueError, "sum to 1"),
        ({"operations": (30,), "probabilities": (1.0,)}, ValueError, "whole numbers"),
    ],
)
def test_invalid_settings_raise(
    kwargs: dict, error: type[Exception], match: str
) -> None:
    """Settings are checked when the augmentor is built."""
    with pytest.raises(error, match=match):
        StrongAugment(**kwargs)


def test_unsupported_images_and_operations_raise(image: np.ndarray) -> None:
    """Only RGB or greyscale uint8 images and known operations are accepted."""
    with pytest.raises(ValueError, match="uint8"):
        StrongAugment.apply(image.astype(np.float32), [])
    with pytest.raises(ValueError, match="uint8"):
        StrongAugment.apply(np.zeros((8, 8, 4), np.uint8), [])
    with pytest.raises(ValueError, match="RGBA"):
        StrongAugment.apply(Image.new("RGBA", (8, 8)), [])
    with pytest.raises(ValueError, match="Unsupported"):
        StrongAugment.apply(image, [("warp", {})])
