# =====================================================================
# model/losses.py — Custom loss functions for variant classification
#
# Focal Loss focuses training on hard-to-classify examples by
# down-weighting easy negatives. This is critical for genomic variant
# classification where many variants have clear signals but a subset
# are genuinely ambiguous.
# =====================================================================

import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional label smoothing.

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    γ = 0  → equivalent to standard cross-entropy
    γ > 0  → reduces loss for well-classified examples, focusing
              training on hard misclassified samples

    Args:
        gamma: Focusing parameter (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output [B, C]
            targets: Ground truth labels [B]

        Returns:
            Scalar focal loss value
        """
        ce = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = (-ce).exp()  # p_t = exp(-CE)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal
