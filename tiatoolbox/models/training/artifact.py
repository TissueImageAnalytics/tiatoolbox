"""Lightweight training artifact manifests for inference handoff."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch import nn

from tiatoolbox.models.engine.io_config import (
    IOInstanceSegmentorConfig,
    IOPatchPredictorConfig,
    IOSegmentorConfig,
    ModelIOConfigABC,
)
from tiatoolbox.models.training.checkpoint import load_checkpoint, load_model_state_dict

MANIFEST_SCHEMA_VERSION = 1

_SUPPORTED_IO_CONFIGS = {
    "IOPatchPredictorConfig": IOPatchPredictorConfig,
    "IOSegmentorConfig": IOSegmentorConfig,
    "IOInstanceSegmentorConfig": IOInstanceSegmentorConfig,
}
_SUPPORTED_ENGINE_CONFIGS = {
    "PatchPredictor": IOPatchPredictorConfig,
    "SemanticSegmentor": IOSegmentorConfig,
    "NucleusInstanceSegmentor": IOInstanceSegmentorConfig,
    "MultiTaskSegmentor": IOSegmentorConfig,
}


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy of common Python/numpy values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _restore_numeric_mapping_keys(value: Any) -> Any:
    """Restore integer-looking JSON object keys without touching labels."""
    if isinstance(value, Mapping):
        restored: dict[Any, Any] = {}
        for key, item in value.items():
            restored_key: Any = key
            if isinstance(key, str):
                try:
                    restored_key = int(key)
                except ValueError:
                    restored_key = key
            restored[restored_key] = _restore_numeric_mapping_keys(item)
        return restored
    if isinstance(value, list):
        return [_restore_numeric_mapping_keys(item) for item in value]
    return value


def _callable_identifier(value: Any) -> str | None:
    """Return a readable, non-executable identifier for a callable."""
    if value is None or not callable(value):
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None or "<lambda>" in qualname:
        return None
    return f"{module}.{qualname}"


def _normalise_relative_path(path: str | Path, *, relative_to: str | Path) -> str:
    """Store paths relative to the manifest directory when possible."""
    path = Path(path)
    relative_to = Path(relative_to)
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def ioconfig_to_dict(ioconfig: ModelIOConfigABC) -> dict[str, Any]:
    """Serialize a supported engine IO config into explicit constructor args."""
    config_type = type(ioconfig).__name__
    if config_type not in _SUPPORTED_IO_CONFIGS:
        msg = f"Unsupported IO config type `{config_type}`."
        raise ValueError(msg)

    init_args = {
        field.name: _json_safe(getattr(ioconfig, field.name))
        for field in fields(ioconfig)
    }
    return {"type": config_type, "init_args": init_args}


def ioconfig_from_dict(payload: Mapping[str, Any]) -> ModelIOConfigABC:
    """Deserialize a supported engine IO config from manifest metadata."""
    config_type = str(payload.get("type", ""))
    try:
        config_cls = _SUPPORTED_IO_CONFIGS[config_type]
    except KeyError as error:
        msg = f"Unsupported IO config type `{config_type}`."
        raise ValueError(msg) from error

    init_args = dict(payload.get("init_args", {}))
    return config_cls(**init_args)


@dataclass
class EngineConfigSpec:
    """Recommended inference-engine IO metadata for a trained model."""

    engine: str
    ioconfig: dict[str, Any]
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_ioconfig(
        cls,
        ioconfig: ModelIOConfigABC,
        *,
        engine: str | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> EngineConfigSpec:
        """Build a config spec from a concrete TIAToolbox IO config."""
        if engine is None:
            if isinstance(ioconfig, IOPatchPredictorConfig):
                engine = "PatchPredictor"
            elif isinstance(ioconfig, IOInstanceSegmentorConfig):
                engine = "NucleusInstanceSegmentor"
            else:
                engine = "SemanticSegmentor"
        return cls(
            engine=engine,
            ioconfig=ioconfig_to_dict(ioconfig),
            run_kwargs=_json_safe(dict(run_kwargs or {})),
            notes=notes,
        )

    def to_ioconfig(self) -> ModelIOConfigABC:
        """Return the concrete IO config object represented by this spec."""
        return ioconfig_from_dict(self.ioconfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spec to a JSON-compatible mapping."""
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EngineConfigSpec:
        """Deserialize a spec from a JSON-compatible mapping."""
        return cls(
            engine=str(payload["engine"]),
            ioconfig=dict(payload["ioconfig"]),
            run_kwargs=dict(payload.get("run_kwargs", {})),
            notes=payload.get("notes"),
        )


@dataclass
class EngineSetup:
    """Split inference-engine construction kwargs from run-time kwargs."""

    constructor_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the setup to a JSON-compatible mapping."""
        return _json_safe(asdict(self))


@dataclass
class TrainingArtifactManifest:
    """JSON manifest describing a trained model handoff artifact.

    The manifest intentionally stores explicit metadata only: model identity and
    constructor hints, class labels, pre/post-processing notes, IO config
    constructor args, and relative paths to weights/checkpoints. It never pickles
    model factories or imports code while loading.
    """

    task_type: Literal[
        "classification",
        "semantic_segmentation",
        "instance_segmentation",
        "multi_task_segmentation",
        "other",
    ]
    model: dict[str, Any]
    schema_version: int = MANIFEST_SCHEMA_VERSION
    class_dict: dict[Any, Any] | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)
    engine_configs: dict[str, EngineConfigSpec] = field(default_factory=dict)
    weights: dict[str, str] = field(default_factory=dict)
    checkpoints: dict[str, str] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        task_type: Literal[
            "classification",
            "semantic_segmentation",
            "instance_segmentation",
            "multi_task_segmentation",
            "other",
        ],
        model_constructor: Mapping[str, Any] | None = None,
        model_description: str | None = None,
        class_dict: Mapping[Any, Any] | None = None,
        preprocessing: Mapping[str, Any] | None = None,
        postprocessing: Mapping[str, Any] | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        engine: str | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingArtifactManifest:
        """Create a manifest with safe metadata discovered from a model instance."""
        model_payload = {
            "module": type(model).__module__,
            "class_name": type(model).__qualname__,
            "constructor": _json_safe(dict(model_constructor or {})),
        }
        if model_description is not None:
            model_payload["description"] = model_description

        resolved_class_dict = class_dict
        if resolved_class_dict is None:
            resolved_class_dict = getattr(model, "class_dict", None)

        preproc_payload = dict(preprocessing or {})
        if "callable" not in preproc_payload:
            preproc_identifier = _callable_identifier(getattr(model, "preproc_func", None))
            if preproc_identifier is not None:
                preproc_payload["callable"] = preproc_identifier

        postproc_payload = dict(postprocessing or {})
        if "callable" not in postproc_payload:
            postproc_identifier = _callable_identifier(getattr(model, "postproc_func", None))
            if postproc_identifier is not None:
                postproc_payload["callable"] = postproc_identifier

        engine_configs: dict[str, EngineConfigSpec] = {}
        if ioconfig is not None:
            spec = EngineConfigSpec.from_ioconfig(
                ioconfig,
                engine=engine,
                run_kwargs=run_kwargs,
            )
            engine_configs[spec.engine] = spec

        return cls(
            task_type=task_type,
            model=model_payload,
            class_dict=_restore_numeric_mapping_keys(_json_safe(resolved_class_dict))
            if resolved_class_dict is not None
            else None,
            preprocessing=_json_safe(preproc_payload),
            postprocessing=_json_safe(postproc_payload),
            engine_configs=engine_configs,
            metadata=_json_safe(dict(metadata or {})),
        )

    def record_weight(
        self,
        name: str,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> None:
        """Record a named weight file path in the manifest."""
        self.weights[name] = (
            _normalise_relative_path(path, relative_to=relative_to)
            if relative_to is not None
            else str(path)
        )

    def record_checkpoint(
        self,
        name: str,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> None:
        """Record a named checkpoint path in the manifest."""
        self.checkpoints[name] = (
            _normalise_relative_path(path, relative_to=relative_to)
            if relative_to is not None
            else str(path)
        )

    def record_training_state(
        self,
        *,
        best_epoch: int,
        best_monitor_value: float,
        monitor: str,
        monitor_mode: str,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        """Record lightweight training summary metadata."""
        self.training.update(
            _json_safe(
                {
                    "best_epoch": best_epoch,
                    "best_monitor_value": best_monitor_value,
                    "monitor": monitor,
                    "monitor_mode": monitor_mode,
                    "history": history or [],
                }
            )
        )

    def _manifest_base_path(
        self,
        manifest_path: str | Path | None = None,
    ) -> Path | None:
        """Return a manifest path argument or path captured by ``load``."""
        resolved_path = manifest_path or self.source_path
        return None if resolved_path is None else Path(resolved_path)

    def _resolve_recorded_path(
        self,
        path: str | Path,
        *,
        manifest_path: str | Path | None = None,
    ) -> Path:
        """Resolve a recorded path relative to a manifest path when available."""
        resolved_path = Path(path)
        base_path = self._manifest_base_path(manifest_path)
        if resolved_path.is_absolute() or base_path is None:
            return resolved_path
        base_dir = base_path if base_path.is_dir() else base_path.parent
        return base_dir / resolved_path

    def resolve_weight_path(
        self,
        name: str = "best",
        *,
        manifest_path: str | Path | None = None,
    ) -> Path:
        """Resolve a named weight path, relative to a manifest path if provided."""
        try:
            weights_path = self.weights[name]
        except KeyError as error:
            msg = f"No `{name}` weights path recorded in the training artifact."
            raise KeyError(msg) from error
        return self._resolve_recorded_path(
            weights_path,
            manifest_path=manifest_path,
        )

    def resolve_checkpoint_path(
        self,
        name: str = "last",
        *,
        manifest_path: str | Path | None = None,
    ) -> Path:
        """Resolve a named checkpoint path, relative to a manifest path if provided."""
        try:
            checkpoint_path = self.checkpoints[name]
        except KeyError as error:
            msg = f"No `{name}` checkpoint path recorded in the training artifact."
            raise KeyError(msg) from error
        return self._resolve_recorded_path(
            checkpoint_path,
            manifest_path=manifest_path,
        )

    def resolve_weights_or_checkpoint_path(
        self,
        name: str = "best",
        *,
        manifest_path: str | Path | None = None,
    ) -> Path:
        """Resolve a named weights/checkpoint artifact path for model loading."""
        if name in self.weights:
            return self.resolve_weight_path(name, manifest_path=manifest_path)
        if name in self.checkpoints:
            return self.resolve_checkpoint_path(name, manifest_path=manifest_path)
        msg = (
            f"No `{name}` weights or checkpoint path recorded in the training artifact."
        )
        raise KeyError(msg)

    def load_weights(
        self,
        model: nn.Module,
        name: str = "best",
        *,
        manifest_path: str | Path | None = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Any:
        """Load recorded weights into a user-supplied model.

        The manifest only resolves a recorded weights/checkpoint path. The caller
        remains responsible for constructing the model explicitly. Payload loading
        is delegated to the training checkpoint helpers so bare state dicts and
        full trainer checkpoints follow the same compatibility path as trainer
        resume and :meth:`ModelABC.load_weights_from_file`.
        """
        weights_path = self.resolve_weights_or_checkpoint_path(
            name,
            manifest_path=manifest_path,
        )
        payload = load_checkpoint(weights_path, map_location=map_location)
        return load_model_state_dict(model, payload, strict=strict)

    def get_engine_config(self, engine: str | None = None) -> EngineConfigSpec:
        """Return a recommended engine config spec.

        If no engine is supplied, the only stored config is returned. Supplying an
        engine name is required when the manifest contains multiple configs.
        """
        if engine is not None:
            try:
                return self.engine_configs[engine]
            except KeyError as error:
                msg = f"No engine config named `{engine}` in training artifact."
                raise KeyError(msg) from error

        if len(self.engine_configs) != 1:
            msg = "Specify an engine when the artifact has zero or multiple configs."
            raise ValueError(msg)
        return next(iter(self.engine_configs.values()))

    def to_engine_setup(
        self,
        engine: str | None = None,
        *,
        manifest_path: str | Path | None = None,
        weights_key: str | None = "best",
        include_weights: bool = True,
    ) -> EngineSetup:
        """Return split engine constructor and ``run`` kwargs from the manifest.

        Constructor kwargs contain only engine construction parameters recorded in
        the artifact, currently the selected ``weights`` path when requested.
        Runtime kwargs contain the reconstructed IO config, class dictionary, and
        stored engine run kwargs. The method does not instantiate or import a
        model class.
        """
        constructor_kwargs: dict[str, Any] = {}
        if include_weights and weights_key is not None and weights_key in self.weights:
            constructor_kwargs["weights"] = self.resolve_weight_path(
                weights_key,
                manifest_path=manifest_path,
            )

        config = self.get_engine_config(engine)
        run_kwargs = {"ioconfig": config.to_ioconfig()}
        run_kwargs.update(config.run_kwargs)
        if self.class_dict is not None:
            run_kwargs["class_dict"] = self.class_dict
        return EngineSetup(
            constructor_kwargs=constructor_kwargs,
            run_kwargs=run_kwargs,
        )

    def to_engine_kwargs(
        self,
        engine: str | None = None,
        *,
        manifest_path: str | Path | None = None,
        weights_key: str = "best",
    ) -> dict[str, Any]:
        """Return common inference kwargs from the manifest.

        This compatibility helper merges :meth:`to_engine_setup` into a single
        mapping. Prefer :meth:`to_engine_setup` in new code when separating engine
        construction from ``run`` arguments matters.
        """
        setup = self.to_engine_setup(
            engine,
            manifest_path=manifest_path,
            weights_key=weights_key,
        )
        return {**setup.constructor_kwargs, **setup.run_kwargs}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-compatible mapping."""
        payload = asdict(self)
        payload.pop("source_path", None)
        payload["engine_configs"] = {
            name: spec.to_dict() if isinstance(spec, EngineConfigSpec) else spec
            for name, spec in self.engine_configs.items()
        }
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingArtifactManifest:
        """Deserialize a manifest mapping without importing model code."""
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            msg = (
                f"Unsupported training artifact schema version `{schema_version}`; "
                f"expected `{MANIFEST_SCHEMA_VERSION}`."
            )
            raise ValueError(msg)

        return cls(
            schema_version=schema_version,
            task_type=payload["task_type"],
            model=dict(payload["model"]),
            class_dict=_restore_numeric_mapping_keys(payload.get("class_dict")),
            preprocessing=dict(payload.get("preprocessing", {})),
            postprocessing=dict(payload.get("postprocessing", {})),
            engine_configs={
                name: EngineConfigSpec.from_dict(spec)
                for name, spec in dict(payload.get("engine_configs", {})).items()
            },
            weights=dict(payload.get("weights", {})),
            checkpoints=dict(payload.get("checkpoints", {})),
            training=dict(payload.get("training", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write the manifest to disk as formatted JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrainingArtifactManifest:
        """Load a training artifact manifest from JSON."""
        path = Path(path)
        artifact = cls.from_dict(json.loads(path.read_text()))
        artifact.source_path = path
        return artifact


def load_training_artifact(path: str | Path) -> TrainingArtifactManifest:
    """Load a training artifact manifest from JSON."""
    return TrainingArtifactManifest.load(path)
