# =============================================================================
# compression/scorer_magnitude.py
# LTH compression using ONLY weight magnitude |w_i| as the importance signal.
#
# Rationale: Magnitude pruning (Han et al. 2015) removes the smallest
# weights globally. Surviving weights have the largest absolute values
# and are least sensitive to zeroing under a linear approximation.
#
# Threshold: kth-value on |w_i| globally across all prunable layers.
# =============================================================================

import os
import sys
import json
import time
from copy import deepcopy

import torch

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compression.shared import (
    load_dense_model, evaluate_full,
    count_sparsity, save_weights_only, save_results_json,
    print_metrics_block, print_sparsity_table,
    COMPRESSION_CFG,
)
from compression.lth_core import (
    compute_magnitude_scores,
    percentile_rank,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
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
    Run magnitude-only LTH at each sparsity level.

    For each sparsity:
      1. Score: rank global |w_i|
      2. Mask: prune bottom-k%
      3. Rewind: surviving w ← w_init
      4. Fine-tune with masks frozen
      5. Evaluate + save weights-only .pth

    Args:
        train_loader, val_loader: DataLoaders (shared, built once in pipeline)
        device:          torch.device
        dense_path:      Path to ntv2_consolidated_full_final.pth
        out_dir:         Output directory for this scorer
        sparsities:      List of target sparsity fractions [0.10, 0.20, ...]
        cfg:             Compression config dict
        baseline_metrics: Dense model metrics dict (for Δ reporting)

    Returns:
        results: {sparsity_float: metrics_dict}
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  SCORER: MAGNITUDE — LTH Compression")
    print(f"{'═'*70}")
    print(f"  Importance signal: |w_i|  (no gradient info needed)")
    print(f"  Threshold method: global k-th percentile")
    print(f"  Sparsity levels:  {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"{'═'*70}\n")

    # ── Load fresh dense model + save initial weights ──────────────────────
    model = load_dense_model(dense_path, device)

    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors for LTH rewind")

    # ── Baseline evaluation ────────────────────────────────────────────────
    print("\n  Evaluating dense model (baseline)...")
    if not baseline_metrics:
        baseline_metrics = evaluate_full(model, val_loader, device,
                                         desc="Dense Baseline")
    print_metrics_block(baseline_metrics, label="Dense Baseline", indent="  ")

    results = {}

    # ── Per-sparsity loop ──────────────────────────────────────────────────
    for sp in sparsities:
        sp_label = f"magnitude_{int(sp*100):02d}pct"
        print(f"\n{'─'*70}")
        print(f"  MAGNITUDE @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        # Step 1: Reload dense weights (fresh start for each sparsity)
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Step 2: Compute magnitude scores and convert to percentile ranks
        raw_scores = compute_magnitude_scores(model)
        ranked_scores = {name: percentile_rank(s) for name, s in raw_scores.items()}

        # Step 3: Create global masks
        masks, achieved_sp = create_masks(ranked_scores, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in ranked_scores.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # Step 4: LTH Rewind — surviving weights get INITIAL values
        lth_rewind(model, masks, initial_weights, device)

        # Step 5: Fine-tune with frozen masks
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, cfg,
            sparsity_label=f"magnitude@{sp*100:.0f}%",
        )

        # Step 6: Load best and verify actual sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)  # ensure masks enforced
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\n  ✅ Magnitude @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"Magnitude @ {sp*100:.0f}% sparsity",
            indent="  ",
        )

        # Step 7: Save weights-only .pth
        pth_path = os.path.join(out_dir, f"magnitude_sparsity{int(sp*100):02d}.pth")
        save_weights_only(model, pth_path)

        # Store result
        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
        }

    # ── Save all results JSON ──────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'magnitude_results.json')
    save_results_json(
        {
            'scorer': 'magnitude',
            'baseline': baseline_metrics,
            'sparsity_results': {
                f"{int(sp*100):02d}pct": v for sp, v in results.items()
            },
        },
        json_path,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print_sparsity_table('magnitude', results, baseline_metrics)

    return results
