# =============================================================================
# compression/scorer_ema.py
# LTH compression using ONLY EMA gradient magnitude as the importance signal.
#
# Rationale: EMA captures the time-averaged gradient activity of each weight
# across the full training trajectory. A weight that consistently receives
# large gradient signals is important regardless of its current magnitude.
#
# EMA update: s_i^(t) = β · s_i^(t-1) + (1-β) · |∂L/∂w_i|
# Bias-corrected: ŝ_i = s_i / (1 - β^t)
#
# Threshold: global k-th percentile on log-smoothed EMA ranks.
# =============================================================================

import os
import sys
import json
import time
from copy import deepcopy

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compression.shared import (
    load_dense_model, evaluate_full,
    count_sparsity, save_weights_only, save_results_json,
    print_metrics_block, print_sparsity_table,
)
from compression.lth_core import (
    EMAGradientTracker,
    log_percentile_rank,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
    run_ema_warmup,
)


def run(
    train_loader,
    val_loader,
    device: torch.device,
    dense_path: str,
    out_dir: str,
    sparsities: list,
    cfg: dict,
    baseline_metrics: dict,
) -> dict:
    """
    Run EMA-gradient-only LTH at each sparsity level.

    Phases:
      A. EMA warmup — train for `ema_warmup_epochs` epochs to accumulate
         gradient statistics. Warmup weights are discarded afterward (we rewind
         to initial weights for each sparsity level).
      B. Per-sparsity LTH — mask by EMA rank → rewind → fine-tune → save.

    Returns:
        results: {sparsity_float: metrics_dict}
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  SCORER: EMA GRADIENT — LTH Compression")
    print(f"{'═'*70}")
    print(f"  Importance signal: EMA_β(|∂L/∂w_i|), β={cfg['ema_beta']}")
    print(f"  Warmup epochs:     {cfg['ema_warmup_epochs']}")
    print(f"  Threshold method:  global k-th percentile on log-smoothed EMA")
    print(f"  Sparsity levels:   {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"{'═'*70}\n")

    # ── Load fresh dense model + save initial weights ──────────────────────
    model = load_dense_model(dense_path, device)
    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors for LTH rewind")

    # ── Baseline ───────────────────────────────────────────────────────────
    print("\n  Evaluating dense model (baseline)...")
    if not baseline_metrics:
        baseline_metrics = evaluate_full(model, val_loader, device,
                                         desc="Dense Baseline")
    print_metrics_block(baseline_metrics, label="Dense Baseline", indent="  ")

    # ── Phase A: EMA Warmup ────────────────────────────────────────────────
    # Train for ema_warmup_epochs to accumulate rich gradient statistics.
    # The model weights will drift during warmup; we RELOAD initial weights
    # before each LTH sparsity step.
    ema_tracker = EMAGradientTracker(model, beta=cfg['ema_beta'])
    run_ema_warmup(model, train_loader, val_loader, device, cfg, ema_tracker)

    # Get bias-corrected EMA scores (computed once, reused for all sparsities)
    ema_scores = ema_tracker.get_scores()
    # Convert to log-smoothed percentile ranks (once, for efficiency)
    ema_ranked = {name: log_percentile_rank(s) for name, s in ema_scores.items()}
    print(f"\n  EMA ranked scores computed for {len(ema_ranked)} parameter tensors")

    results = {}

    # ── Phase B: Per-sparsity LTH ─────────────────────────────────────────
    for sp in sparsities:
        print(f"\n{'─'*70}")
        print(f"  EMA @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        # Reload dense weights (LTH requires starting from init)
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Create masks from EMA ranks
        masks, achieved_sp = create_masks(ema_ranked, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in ema_ranked.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # LTH Rewind
        lth_rewind(model, masks, initial_weights, device)

        # Fine-tune
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, cfg,
            sparsity_label=f"ema@{sp*100:.0f}%",
        )

        # Load best, verify sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\n  ✅ EMA @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"EMA @ {sp*100:.0f}% sparsity",
            indent="  ",
        )

        # Save weights-only .pth
        pth_path = os.path.join(out_dir, f"ema_sparsity{int(sp*100):02d}.pth")
        save_weights_only(model, pth_path)

        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
        }

    # ── Save results JSON ──────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'ema_results.json')
    save_results_json(
        {
            'scorer': 'ema',
            'ema_beta': cfg['ema_beta'],
            'ema_warmup_epochs': cfg['ema_warmup_epochs'],
            'baseline': baseline_metrics,
            'sparsity_results': {
                f"{int(sp*100):02d}pct": v for sp, v in results.items()
            },
        },
        json_path,
    )

    print_sparsity_table('ema', results, baseline_metrics)

    return results
