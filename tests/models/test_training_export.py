"""Model export serialization and checkpoint failure handling."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from tiatoolbox.models.architecture import _build_ioconfig
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.dataset.classification import (
    _TorchPreprocCaller,
    predefined_preproc_func,
)
from tiatoolbox.models.training import ModelExport, load_trained_model
from tiatoolbox.models.training.checkpoint import save_checkpoint, save_model_weights
from tiatoolbox.models.training.export import _default_argument

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


def test_export_normalizes_numpy_and_paths(tmp_path: Path) -> None:
    """YAML exports use portable values and can load an absolute weights override."""
    model = nn.Linear(2, 3)
    weights = tmp_path / "weights.pth"
    save_model_weights(model, weights)
    config = tmp_path / "model.yaml"
    ModelExport(
        model_kwargs={
            "sizes": np.array([2, 3]),
            "source": tmp_path,
            "scale": np.float32(0.5),
        },
        class_dict={np.int64(0): "background"},
    ).save(config, model, "missing.pth")
    recorded = yaml.safe_load(config.read_text())
    assert recorded["architecture"]["kwargs"] == {
        "sizes": [2, 3],
        "source": str(tmp_path),
        "scale": 0.5,
    }
    loaded, io = load_trained_model(config, model=nn.Linear(2, 3), weights=weights)
    assert io is None
    assert loaded.class_dict == {0: "background"}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_custom_preprocessing_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Exports explain when preprocessing must be restored by the caller."""
    model = CNNModel("resnet18", num_classes=2)
    model.preproc_func = lambda image: image / 255
    config = ModelExport(
        model_kwargs={"backbone": "resnet18", "num_classes": 2}
    ).to_config(model, tmp_path / "weights.pth")
    assert "dataset" not in config
    assert "set it again after load_trained_model" in caplog.text


@pytest.mark.parametrize("name", ["in_features", "missing"])
def test_constructor_argument_without_default(name: str) -> None:
    """Required and absent constructor arguments do not invent export defaults."""
    assert _default_argument(nn.Linear, name) is None


def test_checkpoint_temporary_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure to create a temporary file preserves an existing checkpoint."""
    path = tmp_path / "checkpoint.pth"
    save_checkpoint({"epoch": 1}, path)
    original = path.read_bytes()
    monkeypatch.setattr(
        "tiatoolbox.models.training.checkpoint.NamedTemporaryFile",
        Mock(side_effect=OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint({"epoch": 2}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_predefined_preprocessing_equality_and_hashing() -> None:
    """Named preprocessors compare by dataset; unnamed transforms keep identity."""
    first = predefined_preproc_func("kather100k")
    second = predefined_preproc_func("kather100k")
    assert first == second
    assert hash(first) == hash(second)
    custom = _TorchPreprocCaller([])
    assert custom != first
    assert custom != _TorchPreprocCaller([])
    assert custom != object()
    assert len({custom, first, second}) == 2
    assert hash(custom) == id(custom)


def test_invalid_registry_ioconfig() -> None:
    """Unknown IO configuration classes fail without importing arbitrary modules."""
    with pytest.raises(ValueError, match="not a TIAToolbox engine IO config class"):
        _build_ioconfig({"class": "not_a_module.Unknown"})
