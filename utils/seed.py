# =====================================================================
# utils/seed.py — Reproducibility utilities
# =====================================================================

import torch
import numpy as np


def set_seed(seed: int = 42, benchmark: bool = False):
    """
    Set all random seeds for full reproducibility.

    Covers: Python stdlib, NumPy, PyTorch CPU/GPU, cuDNN.

    Args:
        seed:      Random seed
        benchmark: If True, enable cuDNN benchmark mode (faster for
                   fixed-size inputs but slightly non-deterministic).
                   Recommended True for dedicated GPU training.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        if benchmark:
            # Faster on fixed-size inputs (same seq_length every batch)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
