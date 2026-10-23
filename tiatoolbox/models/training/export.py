"""Save trained models in the pretrained-model registry format and load them back.

The trainer writes ``model_config.yaml`` next to the weights. It has the same
``architecture``/``ioconfig``/``dataset`` layout as the entries in
``tiatoolbox/data/pretrained_model.yaml``, so a trained model is rebuilt the same
way as a pretrained one, and can be passed straight to an inference engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from tiatoolbox import logger
from tiatoolbox.models.architecture import _build_architecture, _build_ioconfig
from tiatoolbox.models.dataset.classification import predefined_preproc_func
from tiatoolbox.models.models_abc import load_model_state_dict, unwrap_model
from tiatoolbox.models.training.checkpoint import _atomic_output, load_checkpoint

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn

    from tiatoolbox.models.engine.io_config import ModelIOConfigABC

DEFAULT_CONFIG_FILENAME = "model_config.yaml"


def _plain(value: object) -> object:
    """Convert configuration values to YAML-safe builtins."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_plain(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _class_path(cls: type, package: str) -> str:
    """Name a class relative to a TIAToolbox package, or in full outside it."""
    return f"{cls.__module__}.{cls.__qualname__}".removeprefix(f"{package}.")


@dataclass
class ModelExport:
    """Describe how to rebuild a trained model for inference.

    Give one to the :class:`Trainer` to have it write ``model_config.yaml``
    next to the trained weights; :func:`load_trained_model` reads it back. The
    architecture class is taken from the trained model, and predefined
    preprocessing (``predefined_preproc_func``) is recorded as the registry's
    ``dataset`` key. Other custom ``preproc_func`` values cannot be recorded
    and must be set again after loading.

    Args:
        model_kwargs (dict):
            Keyword arguments that construct the trained architecture.
        ioconfig (ModelIOConfigABC):
            Inference IO configuration, such as the training resolution and
            patch shape. Optional.
        class_dict (dict):
            Class names, set as the loaded model's ``class_dict`` so engines
            label their outputs. Optional.

    """

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    ioconfig: ModelIOConfigABC | None = None
    class_dict: dict | None = None

    def to_config(self, model: nn.Module, weights: str | Path) -> dict[str, Any]:
        """Return the registry-format entry for ``model`` and its weight file."""
        model = unwrap_model(model)
        config: dict[str, Any] = {
            "architecture": {
                "class": _class_path(type(model), "tiatoolbox.models.architecture"),
                "kwargs": _plain(self.model_kwargs),
            },
        }
        if self.ioconfig is not None:
            config["ioconfig"] = {
                "class": _class_path(type(self.ioconfig), "tiatoolbox.models.engine"),
                "kwargs": {
                    item.name: _plain(getattr(self.ioconfig, item.name))
                    for item in fields(self.ioconfig)
                },
            }
        preproc = getattr(model, "preproc_func", None)
        if hasattr(preproc, "dataset_name"):
            config["dataset"] = preproc.dataset_name
        elif preproc is not None and preproc != type(model).preproc:
            logger.warning(
                "%s.preproc_func is a custom function, which model_config.yaml "
                "cannot record; set it again after load_trained_model.",
                type(model).__name__,
            )
        if self.class_dict is not None:
            config["class_dict"] = _plain(self.class_dict)
        config["weights"] = str(weights)
        return config

    def save(self, path: str | Path, model: nn.Module, weights: str | Path) -> Path:
        """Write the configuration for ``model`` atomically and return its path."""
        text = yaml.safe_dump(self.to_config(model, weights), sort_keys=False)
        with _atomic_output(path) as stream:
            stream.write(text.encode("utf-8"))
        return Path(path)


def load_trained_model(
    path: str | Path,
    *,
    model: nn.Module | type[nn.Module] | None = None,
    weights: str | Path | None = None,
) -> tuple[nn.Module, ModelIOConfigABC | None]:
    """Rebuild a trained model and its IO config, like ``get_pretrained_model``.

    Args:
        path (str or Path):
            A training output directory or its ``model_config.yaml``.
        model (nn.Module or type):
            Needed when the trained architecture is not a TIAToolbox class: an
            instance to load the weights into, or a class to construct with the
            recorded keyword arguments. A class also selects a compatible
            inference variant, such as ``KongNetSegmentor`` for a trained
            ``KongNet``.
        weights (str or Path):
            Weights or full trainer checkpoint to load instead of the recorded
            file. Relative paths resolve from the configuration's directory.

    Returns:
        tuple:
            The model with weights, recorded predefined preprocessing and class
            names, and the recorded IO config (``None`` if none was recorded).

    """
    path = Path(path)
    config_path = path / DEFAULT_CONFIG_FILENAME if path.is_dir() else path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architecture = config["architecture"]
    if model is None:
        try:
            model = _build_architecture(architecture)
        except ValueError as error:
            msg = f"{error} Pass `model=` (the class or an instance) to rebuild it."
            raise ValueError(msg) from error
    elif isinstance(model, type):
        model = model(**architecture.get("kwargs", {}))

    weights_path = Path(config["weights"] if weights is None else weights)
    if not weights_path.is_absolute():
        weights_path = config_path.parent / weights_path
    load_model_state_dict(model, load_checkpoint(weights_path))
    if "dataset" in config:
        model.preproc_func = predefined_preproc_func(config["dataset"])
    if "class_dict" in config:
        model.class_dict = config["class_dict"]
    ioconfig = _build_ioconfig(config["ioconfig"]) if "ioconfig" in config else None
    return model, ioconfig
