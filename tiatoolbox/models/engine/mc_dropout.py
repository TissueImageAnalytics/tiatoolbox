"""Utilities to enable Monte Carlo Dropout without touching BatchNorm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from types import TracebackType
    from typing import Self

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)

# Minimum number of children a ``nn.Sequential`` classifier must have before we
# consider its leading ``nn.Dropout`` as already injected.
_MIN_SEQ_LEN_FOR_DROPOUT = 2


class mc_dropout_mode:  # noqa: N801 - intentional snake_case context-manager name
    """Context manager: enable only Dropout layers, everything else stays in eval."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize :class:`mc_dropout_mode`.

        Args:
            model (nn.Module):
                The model whose Dropout layers will be put in train mode
                inside the context.

        """
        self.model = model
        # Save training state of ALL modules (not just Dropout) so BatchNorm
        # and the root module are restored correctly on exit, since
        # `self.model.eval()` puts the whole tree in eval mode.
        self._prev_states: dict[nn.Module, bool] = {}

    def __enter__(self) -> Self:
        """Enter the context, switching only Dropout layers to train mode.

        Returns:
            Self:
                The model being managed (returned so callers can use it inside
                the ``with`` block).

        """
        self._prev_states = {m: m.training for m in self.model.modules()}

        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, _DROPOUT_TYPES):
                module.train()

        return self.model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context, restoring each module's prior training state.

        Args:
            exc_type (type[BaseException] | None):
                Type of the raised exception, if any.
            exc_val (BaseException | None):
                The raised exception, if any.
            exc_tb (TracebackType | None):
                The associated traceback, if any.

        """
        for module, was_training in self._prev_states.items():
            module.train(was_training)


def has_dropout_layers(model: nn.Module) -> bool:
    """Check if the model has any dropout layers.

    Args:
        model (nn.Module):
            The model to inspect.

    Returns:
        bool:
            True if at least one submodule is an instance of a Dropout type.

    """
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
    if (
        isinstance(head, nn.Sequential)
        and len(head) >= _MIN_SEQ_LEN_FOR_DROPOUT
        and isinstance(head[0], nn.Dropout)
    ):
        return model
    if not isinstance(head, nn.Linear):
        msg = (
            "Expected `model.classifier` to be an nn.Linear, "
            f"got {type(head).__name__}."
        )
        raise TypeError(msg)
    model.classifier = nn.Sequential(nn.Dropout(p=p), head)
    return model
