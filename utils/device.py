# =====================================================================
# utils/device.py — Device detection and setup
# =====================================================================

import torch


def get_device() -> torch.device:
    """
    Detect and return the best available device (GPU preferred).
    Prints GPU info if CUDA is available.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"   GPU: {torch.cuda.get_device_name(0)} "
              f"({props.total_memory / 1e9:.1f} GB)")
        print(f"   Compute capability: sm_{cap[0]}{cap[1]}")
        return device

    print("   ⚠️ No GPU detected — using CPU (training will be slow)")
    return torch.device('cpu')


def supports_amp() -> bool:
    """
    Check if the current GPU supports mixed precision (AMP).

    - bfloat16 requires sm_80+ (Ampere: A100, A10G, etc.)
    - float16 requires sm_70+ (Volta: V100, T4, etc.)
    - Pascal GPUs (P100 = sm_60) do NOT support AMP properly
      because modern PyTorch may lack compiled fp16 kernels for sm_60.

    Returns:
        True if AMP can be safely used, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    cap = torch.cuda.get_device_capability(0)
    major = cap[0]

    if major >= 7:
        # Volta (V100), Turing (T4), Ampere (A100), Hopper — all fine
        return True

    # Pascal (P100, sm_60/61) — AMP is unreliable with modern PyTorch
    print(f"   ⚠️ GPU compute capability sm_{cap[0]}{cap[1]} < sm_70 — "
          f"disabling AMP (mixed precision)")
    print(f"   Training will use float32 — slightly slower but compatible")
    return False
