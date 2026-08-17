"""Unit tests for the three MC-Dropout building blocks, no network/WSI needed.

- tiatoolbox.models.engine.mc_dropout      -> mc_dropout_mode, has_dropout_layers,
                                               inject_dropout_before_classifier
- tiatoolbox.models.engine.uncertainty     -> decompose_uncertainty
- tiatoolbox.models.architecture.bayesian_wrapper -> BayesianModelWrapper

Run all:
    pytest tests/test_mc_dropout_units.py -v -s

Run one file's tests only, e.g. just mc_dropout:
    pytest tests/test_mc_dropout_units.py -k mc_dropout -v -s
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.models.architecture.bayesian_wrapper import BayesianModelWrapper
from tiatoolbox.models.engine.mc_dropout import (
    has_dropout_layers,
    inject_dropout_before_classifier,
    mc_dropout_mode,
)
from tiatoolbox.models.engine.uncertainty import decompose_uncertainty

# ----------------------------------------------------------------------------
# Small dummy models for testing the MC Dropout utilities without needing a
# full CNN or WSI.
# ----------------------------------------------------------------------------


class DummyCNNModel(nn.Module):
    """Minimal stand-in for tiatoolbox's CNNModel: feat_extract -> classifier."""

    def __init__(self, *, with_dropout: bool = False) -> None:
        """Initialize :class:`DummyCNNModel`.

        Args:
            with_dropout (bool):
                If True, prepend a ``nn.Dropout(0.5)`` to the classifier head.

        """
        super().__init__()
        self.feat_extract = nn.Conv2d(3, 4, kernel_size=3)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        linear = nn.Linear(4, 2)
        self.classifier = (
            nn.Sequential(nn.Dropout(0.5), linear) if with_dropout else linear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            x (torch.Tensor):
                Model input.

        Returns:
            torch.Tensor:
                The output logits after passing through the model.

        """
        x = self.bn(self.feat_extract(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    @staticmethod
    def infer_batch(
        model: DummyCNNModel, batch_data: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        """Run a forward pass and return softmax probabilities.

        Mimics tiatoolbox's CNNModel.infer_batch(model, images, device) interface.
        """
        with torch.no_grad():
            logits = model(batch_data.to(device))
            return torch.softmax(logits, dim=-1)


# ----------------------------------------------------------------------------
# mc_dropout.py
# ----------------------------------------------------------------------------
def test_has_dropout_layers_true_and_false() -> None:
    """``has_dropout_layers`` returns True when Dropout is present, else False."""
    assert has_dropout_layers(DummyCNNModel(with_dropout=True)) is True
    assert has_dropout_layers(DummyCNNModel(with_dropout=False)) is False


def test_inject_dropout_before_classifier_wraps_linear() -> None:
    """Injection wraps a bare ``nn.Linear`` classifier in a Dropout -> Linear stack."""
    model = DummyCNNModel(with_dropout=False)
    model = inject_dropout_before_classifier(model, p=0.3)

    assert isinstance(model.classifier, nn.Sequential)
    assert isinstance(model.classifier[0], nn.Dropout)
    assert model.classifier[0].p == 0.3
    assert has_dropout_layers(model) is True


def test_inject_dropout_before_classifier_is_idempotent() -> None:
    """Re-injecting on a model that already has a leading Dropout is a no-op."""
    model = DummyCNNModel(with_dropout=False)
    model = inject_dropout_before_classifier(model, p=0.3)
    n_children_before = len(list(model.classifier.children()))

    model = inject_dropout_before_classifier(model, p=0.9)  # should be a no-op

    assert len(list(model.classifier.children())) == n_children_before
    assert model.classifier[0].p == 0.3  # untouched, not overwritten to 0.9


def test_mc_dropout_mode_activates_only_dropout() -> None:
    """Inside the context only Dropout is in train mode; BatchNorm stays in eval."""
    model = DummyCNNModel(with_dropout=True)
    model.eval()  # everything off to start

    with mc_dropout_mode(model) as m:
        assert m.classifier[0].training is True  # Dropout -> train mode
        assert m.bn.training is False  # BatchNorm must stay in eval mode

    # After the context exits, everything is restored to eval
    assert model.classifier[0].training is False
    assert model.bn.training is False


def test_mc_dropout_mode_restores_previous_training_state() -> None:
    """The previous module training state is restored on context exit."""
    model = DummyCNNModel(with_dropout=True)
    model.train()  # whole model was in train mode before entering

    with mc_dropout_mode(model):
        pass

    assert model.training is True  # restored, not forced to eval afterwards
    assert model.bn.training is True


def test_inject_dropout_before_classifier_raises_typeerror_no_classifier() -> None:
    """TypeError if model has no `classifier` attribute."""
    model = nn.Sequential(nn.Linear(10, 2))
    with pytest.raises(TypeError, match=r"classifier"):
        inject_dropout_before_classifier(model)


def test_inject_dropout_before_classifier_raises_typeerror_invalid_head() -> None:
    """TypeError if classifier is neither Linear nor Sequential with leading Dropout."""
    model = DummyCNNModel(with_dropout=False)
    model.classifier = nn.Conv2d(4, 2, 1)
    with pytest.raises(TypeError, match=r"nn\.Linear"):
        inject_dropout_before_classifier(model)


def test_has_dropout_layers_detects_alpha_dropout() -> None:
    """has_dropout_layers detects AlphaDropout and FeatureAlphaDropout."""
    model_alpha = nn.Sequential(nn.Linear(10, 2), nn.AlphaDropout())
    model_feat_alpha = nn.Sequential(nn.Linear(10, 2), nn.FeatureAlphaDropout())
    assert has_dropout_layers(model_alpha) is True
    assert has_dropout_layers(model_feat_alpha) is True


class _TestError(Exception):
    """Test exception for context manager testing."""


def _raise_test_error() -> None:
    """Raise a test exception for context manager testing."""
    msg = "boom"
    raise _TestError(msg)


def test_mc_dropout_mode_exit_restores_on_exception() -> None:
    """__exit__ restores training state even when an exception is raised."""
    model = DummyCNNModel(with_dropout=True)
    model.eval()

    with pytest.raises(_TestError), mc_dropout_mode(model):
        _raise_test_error()

    assert model.training is False
    assert model.bn.training is False
    assert model.classifier[0].training is False


# ----------------------------------------------------------------------------
# uncertainty.py
# ----------------------------------------------------------------------------
def test_decompose_uncertainty_shapes_and_keys() -> None:
    """``decompose_uncertainty`` returns expected keys and shapes; total = ep + al."""
    torch.manual_seed(0)
    t, batch, n_classes = 20, 5, 3
    logits = torch.randn(t, batch, n_classes)
    mc_probs = torch.softmax(logits, dim=-1)

    stats = decompose_uncertainty(mc_probs, class_dim=-1)

    assert set(stats) == {"mean_probs", "epistemic", "aleatoric", "total"}
    assert stats["mean_probs"].shape == (batch, n_classes)
    assert stats["epistemic"].shape == (batch,)
    assert stats["aleatoric"].shape == (batch,)
    assert stats["total"].shape == (batch,)
    assert torch.allclose(stats["total"], stats["epistemic"] + stats["aleatoric"])


def test_decompose_uncertainty_zero_variance_gives_zero_epistemic() -> None:
    """If every MC sample is identical, epistemic uncertainty must be exactly 0."""
    single_pass = torch.tensor([[0.7, 0.2, 0.1]])
    mc_probs = single_pass.unsqueeze(0).repeat(
        10, 1, 1
    )  # [T=10, B=1, C=3], all identical

    stats = decompose_uncertainty(mc_probs, class_dim=-1)

    assert torch.allclose(stats["epistemic"], torch.zeros_like(stats["epistemic"]))
    assert stats["aleatoric"].item() > 0  # data itself is still uncertain


# ----------------------------------------------------------------------------
# bayesian_wrapper.py
# ----------------------------------------------------------------------------
def test_bayesian_wrapper_raises_without_dropout() -> None:
    """Wrapping a Dropout-less model with ``inject_dropout=False`` raises ValueError."""
    model = DummyCNNModel(with_dropout=False)
    with pytest.raises(ValueError, match=r"Dropout"):
        BayesianModelWrapper(base_model=model, n_samples=5, inject_dropout=False)


def test_bayesian_wrapper_raises_runtime_error_if_injection_fails() -> None:
    """RuntimeError if inject_dropout_before_classifier fails to add dropout."""
    model_without_dropout = DummyCNNModel(with_dropout=False)

    with (
        patch(
            "tiatoolbox.models.architecture.bayesian_wrapper."
            "inject_dropout_before_classifier",
            return_value=model_without_dropout,
        ),
        pytest.raises(RuntimeError, match=r"failed to insert"),
    ):
        BayesianModelWrapper(model_without_dropout, inject_dropout=True)


def test_getattr_delegates_to_base_model() -> None:
    """__getattr__ delegates to base_model for attributes not on wrapper."""
    base_model = DummyCNNModel(with_dropout=True)
    base_model.preproc_func = lambda x: x

    wrapper = BayesianModelWrapper(base_model, inject_dropout=False)

    assert wrapper.preproc_func is base_model.preproc_func


def test_getattr_raises_if_attribute_missing_everywhere() -> None:
    """__getattr__ raises AttributeError if attribute missing everywhere."""
    base_model = DummyCNNModel(with_dropout=True)
    wrapper = BayesianModelWrapper(base_model, inject_dropout=False)

    with pytest.raises(AttributeError, match=r"has no attribute"):
        _ = wrapper.this_attribute_does_not_exist_anywhere


def test_bayesian_wrapper_infer_batch_populates_uncertainty_stats() -> None:
    """``infer_batch`` returns mean probs and appends a per-batch uncertainty dict."""
    model = DummyCNNModel(with_dropout=True)
    model.eval()

    wrapper = BayesianModelWrapper(base_model=model, n_samples=8, class_dim=-1)
    dummy_batch = torch.rand(4, 3, 8, 8)

    mean_probs = wrapper.infer_batch(wrapper, dummy_batch, device="cpu")

    assert isinstance(mean_probs, np.ndarray)
    assert mean_probs.shape == (4, 2)  # [B, C]
    assert len(wrapper.uncertainty_stats) == 1

    batch_stats = wrapper.uncertainty_stats[0]
    for key in ("mean_probs", "aleatoric", "epistemic", "total", "mc_probs"):
        assert key in batch_stats
    assert batch_stats["mc_probs"].shape == (8, 4, 2)  # [T, B, C]
