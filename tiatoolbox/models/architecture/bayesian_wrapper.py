"""Transparent Bayesian wrapper for TIAToolbox models compatible with infer_batch.

The engine (`EngineABC.infer_patches`) calls
`infer_batch(self.model, batch_data["image"], device=self.device)`, hence the
static-method signature `(model, images, device)` is preserved so the engine
can keep passing the wrapper as the first argument. Per-batch uncertainty
statistics are accumulated on `wrapper.uncertainty_stats` (a list, in
inference order) so that downstream code — after `predictor.run()` finishes —
can concatenate them along axis 0 to obtain one entry per patch, aligned
with the `coordinates` / `probabilities` arrays written by the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tiatoolbox.models.engine.mc_dropout import (
    has_dropout_layers,
    inject_dropout_before_classifier,
    mc_dropout_mode,
)
from tiatoolbox.models.engine.uncertainty import decompose_uncertainty

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class BayesianModelWrapper(torch.nn.Module):
    """Wrap a TIAToolbox model and add Monte Carlo Dropout at `infer_batch`.

    The wrapped model must contain at least one Dropout layer. For pretrained
    models that ship without Dropout (e.g. `resnet18-kather100k`,
    `alexnet-kather100k`), pass `inject_dropout=True` (default) and a
    `nn.Dropout` is inserted before the linear classifier automatically,
    via `inject_dropout_before_classifier`.

    After each `infer_batch` call, the per-batch uncertainty decomposition
    (mean_probs, aleatoric, epistemic, total, mc_probs) is appended to
    `self.uncertainty_stats` in inference order, so it can be aligned with
    the `coordinates` array written to the engine's output `.zarr`.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        n_samples: int = 30,
        class_dim: int = -1,
        *,
        inject_dropout: bool = True,
        dropout_p: float = 0.2,
    ) -> None:
        """Initialize :class:`BayesianModelWrapper`.

        Args:
            base_model (torch.nn.Module):
                TIAToolbox model to wrap. Must expose an ``infer_batch`` static
                method matching ``(model, images, device)``.
            n_samples (int):
                Number of Monte Carlo forward passes per batch.
            class_dim (int):
                Class axis in the per-sample probabilities (excluding T).
            inject_dropout (bool):
                If True and the model has no Dropout layers, insert one before
                the linear classifier automatically.
            dropout_p (float):
                Dropout probability used by ``inject_dropout``.

        Raises:
            ValueError:
                If the model has no Dropout layers and ``inject_dropout`` is False.
            RuntimeError:
                If dropout injection was requested but failed.

        """
        super().__init__()
        if not has_dropout_layers(base_model):
            if not inject_dropout:
                msg = (
                    "Model has no Dropout layers: MC Dropout is not applicable. "
                    "Pass inject_dropout=True or add an nn.Dropout before wrapping."
                )
                raise ValueError(msg)
            base_model = inject_dropout_before_classifier(base_model, p=dropout_p)
            if not has_dropout_layers(base_model):
                msg = (
                    "inject_dropout_before_classifier failed to insert "
                    "any Dropout layer into the provided model."
                )
                raise RuntimeError(msg)

        self.base_model = base_model
        self.n_samples = n_samples
        self.class_dim = class_dim
        # Per-batch stats populated by `infer_batch`, in inference order.
        # Concatenate along axis 0 after `run()` to align with the engine's
        # `coordinates` / `probabilities` outputs.
        self.uncertainty_stats: list[dict[str, np.ndarray]] = []

    def __getattr__(self, name: str) -> object:
        """Delegate undefined attributes to the wrapped model.

        This lets the engine find attributes such as ``preproc_func``,
        ``postproc_func``, and ``class_dict`` on the wrapped model through
        ``_get_model_attr``. ``__getattr__`` is invoked only when normal
        lookup fails, so by the time we reach here ``self.base_model`` is
        already accessible normally.

        Args:
            name (str):
                Attribute name to look up.

        Returns:
            object:
                The attribute found on the wrapped model.

        Raises:
            AttributeError:
                If the attribute is not found on the wrapper or the model.

        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        base_model = self.__dict__.get("_modules", {}).get("base_model")
        if base_model is not None:
            return getattr(base_model, name)
        msg = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(msg)

    @staticmethod
    def infer_batch(
        model: BayesianModelWrapper,
        batch_data: torch.Tensor,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run `n_samples` MC forward passes under `mc_dropout_mode`.

        Returns the MC-mean per-class probabilities of shape [B, C] (numpy),
        and appends a per-batch uncertainty dict to `model.uncertainty_stats`.
        """
        wrapper = model  # the engine passes the wrapper as the first argument
        base_infer = wrapper.base_model.infer_batch

        samples: list[torch.Tensor] = []
        with mc_dropout_mode(wrapper.base_model):
            for _ in range(wrapper.n_samples):
                sample = base_infer(wrapper.base_model, batch_data, device)
                samples.append(torch.as_tensor(sample))

        mc_probs = torch.stack(samples, dim=0)  # [T, B, C]
        stats = decompose_uncertainty(mc_probs, class_dim=wrapper.class_dim)

        wrapper.uncertainty_stats.append(
            {
                "mean_probs": stats["mean_probs"].cpu().numpy(),
                "aleatoric": stats["aleatoric"].cpu().numpy(),
                "epistemic": stats["epistemic"].cpu().numpy(),
                "total": stats["total"].cpu().numpy(),
                "mc_probs": mc_probs.cpu().numpy(),
            }
        )

        try:
            target_device = next(wrapper.base_model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

        return stats["mean_probs"].to(target_device).cpu().numpy()
