"""Tests model architecture init."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from torch import nn

from tiatoolbox.models.architecture import get_pretrained_model

if TYPE_CHECKING:  # pragma: no cover
    import pytest


class FakeArchModel(nn.Module):
    """Minimal fake architecture."""

    def __init__(self) -> None:
        """Initialize a fake architecture."""
        super().__init__()


class FakeIOConfig:
    """Minimal fake IO config returned by the patched pretrained model metadata."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize a fake IO config object."""
        self.kwargs = kwargs


def _fake_locate(module_path: str) -> object:
    """Return a fake module object exposing the expected architecture classes.

    Args:
        module_path (str):
            Dotted path of the module requested by ``get_pretrained_model``.

    Returns:
        object:
            A lightweight namespace containing the fake architecture and IO classes.

    """
    _ = module_path
    return SimpleNamespace(
        FakeArchModel=FakeArchModel,
        FakeIOConfig=FakeIOConfig,
    )


def _raise_if_called(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Fail the test if an unexpected helper is called.

    Args:
        *args (Any):
            Positional arguments supplied to the unexpected call.
        **kwargs (Any):
            Keyword arguments supplied to the unexpected call.

    Raises:
        AssertionError:
            Always raised to indicate the function should not have been called.

    """
    _ = args, kwargs
    msg = "fetch_pretrained_weights should not be called"
    raise AssertionError(msg)


def _fake_load_torch_model(
    model: nn.Module,
    weights: str | Path,
) -> nn.Module:
    """Return the model unchanged while validating the explicit weights argument.

    Args:
        model (nn.Module):
            The model instance passed to the loader.
        weights (str | Path):
            The explicit pretrained weights path supplied by the test.

    Returns:
        nn.Module:
            The model instance, unchanged.
    """
    assert isinstance(model, nn.Module)
    assert isinstance(weights, Path)
    return model


def test_get_pretrained_model_with_explicit_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover the explicit-weights branch of ``get_pretrained_model``.

    This test verifies that when ``pretrained_weights`` is provided, the function:
    1. skips downloading pretrained weights,
    2. loads the supplied weights path,
    3. returns the model and IO config objects created from the patched metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest fixture used to monkeypatch network-dependent helpers.
        tmp_path (Path):
            Temporary directory provided by pytest for isolated test files.

    Returns:
        None:
            The test asserts behavior and does not return a value.
    """
    weights_path = tmp_path / "manual_weights.pth"
    weights_path.write_bytes(b"fake weights")

    fake_info: dict[str, dict[str, Any]] = {
        "architecture": {
            "class": "fake_module.FakeArchModel",
            "kwargs": {},
        },
        "ioconfig": {
            "class": "fake_io.FakeIOConfig",
            "kwargs": {
                "patch_input_shape": (224, 224),
            },
        },
    }

    monkeypatch.setattr(
        "tiatoolbox.models.architecture.PRETRAINED_INFO",
        {"resnet18-kather100k": fake_info},
    )
    monkeypatch.setattr(
        "tiatoolbox.models.architecture.locate",
        _fake_locate,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.architecture.fetch_pretrained_weights",
        _raise_if_called,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.architecture.load_torch_model",
        _fake_load_torch_model,
    )

    model, ioconfig = get_pretrained_model(
        pretrained_model="resnet18-kather100k",
        pretrained_weights=weights_path,
    )

    assert isinstance(model, FakeArchModel)
    assert isinstance(ioconfig, FakeIOConfig)
    assert ioconfig.kwargs["patch_input_shape"] == (224, 224)
