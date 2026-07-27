"""Utilities to enable Monte Carlo Dropout without touching BatchNorm."""

from __future__ import annotations

from torch import nn

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


class mc_dropout_mode:
    """Context manager: enable only Dropout layers, everything else stays in eval."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        # Save training state of ALL modules (not just Dropout) so BatchNorm
        # and the root module are restored correctly on exit, since
        # `self.model.eval()` puts the whole tree in eval mode.
        self._prev_states: dict[nn.Module, bool] = {}

    def __enter__(self) -> nn.Module:
        self._prev_states = {m: m.training for m in self.model.modules()}

        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, _DROPOUT_TYPES):
                module.train()

        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for module, was_training in self._prev_states.items():
            module.train(was_training)

        return False  # do not suppress exceptions


def has_dropout_layers(model: nn.Module) -> bool:
    """Check if the model has any dropout layers."""
    return any(isinstance(m, _DROPOUT_TYPES) for m in model.modules())


def inject_dropout_before_classifier(
    model: nn.Module,
    p: float = 0.2,
) -> nn.Module:
    """Insert a `nn.Dropout` before the model's linear classifier head.

    Pretrained TIAToolbox patch classifiers (e.g. `resnet18-kather100k`,
    `alexnet-kather100k`) ship with no Dropout layers, which would make
    Monte Carlo Dropout a no-op. This injects a single `nn.Dropout` between
    the global-average-pool features and the linear classifier, the
    canonical placement for MC Dropout on a ResNet/AlexNet classifier.

    Idempotent: if a `nn.Dropout` already immediately precedes the
    classifier it is left untouched.
    """
    head = getattr(model, "classifier", None)
    if head is None:
        msg = (
            "Expected a CNNModel-like module with a `classifier` head, "
            f"got {type(model).__name__}."
        )
        raise TypeError(msg)
    if isinstance(head, nn.Sequential) and len(head) >= 2 and isinstance(head[0], nn.Dropout):
        return model
    if not isinstance(head, nn.Linear):
        msg = (
            "Expected `model.classifier` to be an nn.Linear, "
            f"got {type(head).__name__}."
        )
        raise TypeError(msg)
    model.classifier = nn.Sequential(nn.Dropout(p=p), head)
    return model
