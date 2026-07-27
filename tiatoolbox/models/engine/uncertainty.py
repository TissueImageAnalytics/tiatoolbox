"""
Uncertainty Quantification with Monte Carlo (MC Dropout).
Gal & Ghahramani (2015), used by Policastro (2020) WSI analysis project.
"""

from __future__ import annotations

import torch


def decompose_uncertainty(
    mc_probs: torch.Tensor, class_dim: int = -1
) -> dict[str, torch.Tensor]:
    """Decompose MC-Dropout uncertainty into epistemic and aleatoric terms.

    Args:
        mc_probs: tensor [T, ..., C] with T Monte Carlo forward passes,
            each already softmax-normalized along class_dim.
        class_dim: class axis in mc_probs (excluding T).

    Returns:
        dict: mean_probs, epistemic, aleatoric, total uncertainty.
    """
    mean_probs = mc_probs.mean(dim=0)          # [..., C]
    var_probs = mc_probs.var(dim=0)            # [..., C]

    epistemic = var_probs.mean(dim=class_dim)  # variance across T passes
    aleatoric = (mean_probs * (1 - mean_probs)).mean(dim=class_dim)
    total = epistemic + aleatoric

    return {
        "mean_probs": mean_probs,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": total,
    }