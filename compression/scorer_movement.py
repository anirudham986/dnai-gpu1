# =============================================================================
# compression/scorer_movement.py
# LTH compression using ONLY weight movement as the importance signal.
#
# Rationale: Weight movement |w_i^final - w_i^init| captures how much
# a weight changed during fine-tuning. Weights that moved a lot were
# actively being learned by gradient descent and are therefore important
# for the task. Weights that barely moved contributed little.
#
# This signal is complementary to magnitude: a small weight that moved
# far from its initialisation is critical; a large weight that didn't
# move may have been irrelevant to the task.
#
# Protocol:
#   1. Short warmup training (`movement_warmup_epochs`) to let weights move
#   2. Compute movement = |w_final - w_init| for all prunable layers
#   3. LTH: mask by movement rank → rewind → fine-tune → save
# =============================================================================

import os
import sys
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
    compute_movement_scores,
    percentile_rank,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
    run_movement_warmup,
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
    Run weight-movement-only LTH at each sparsity level.

    Phases:
      A. Movement warmup — train for `movement_warmup_epochs` so weights drift
         from their initial positions. Compute movement scores.
      B. Per-sparsity LTH — mask by movement rank → rewind → fine-tune → save.

    Returns:
        results: {sparsity_float: metrics_dict}
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  SCORER: WEIGHT MOVEMENT — LTH Compression")
    print(f"{'═'*70}")
    print(f"  Importance signal: |w_i^final - w_i^init|")
    print(f"  Movement warmup:   {cfg['movement_warmup_epochs']} epochs")
    print(f"  Threshold method:  global k-th percentile on movement scores")
    print(f"  Sparsity levels:   {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"{'═'*70}\n")

    # ── Load fresh dense model + save initial weights ──────────────────────
    model = load_dense_model(dense_path, device)
    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors")

    # ── Baseline ───────────────────────────────────────────────────────────
    print("\n  Evaluating dense model (baseline)...")
    if not baseline_metrics:
        baseline_metrics = evaluate_full(model, val_loader, device,
                                         desc="Dense Baseline")
    print_metrics_block(baseline_metrics, label="Dense Baseline", indent="  ")

    # ── Phase A: Movement Warmup ───────────────────────────────────────────
    # Train model briefly so weights move away from their initial values.
    # After warmup, movement = |current_w - initial_w|.
    print(f"\n  ── Phase A: Movement Warmup ({cfg['movement_warmup_epochs']} epochs) ──")
    run_movement_warmup(model, train_loader, val_loader, device, cfg)

    # Compute movement scores from the warmed-up model
    movement_scores = compute_movement_scores(model, initial_weights)
    # Convert to percentile ranks (once, reused for all sparsities)
    movement_ranked = {
        name: percentile_rank(s) for name, s in movement_scores.items()
    }
    print(f"  ✅ Movement scores computed for {len(movement_ranked)} parameter tensors")

    # Print movement score statistics
    all_moves = torch.cat([s.flatten() for s in movement_scores.values()])
    print(f"  Movement stats: "
          f"min={all_moves.min():.6f} "
          f"mean={all_moves.mean():.6f} "
          f"max={all_moves.max():.6f}")

    results = {}

    # ── Phase B: Per-sparsity LTH ─────────────────────────────────────────
    for sp in sparsities:
        print(f"\n{'─'*70}")
        print(f"  MOVEMENT @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        # Reload dense weights (LTH must start from init each time)
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Create masks from movement ranks
        masks, achieved_sp = create_masks(movement_ranked, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in movement_ranked.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # LTH Rewind
        lth_rewind(model, masks, initial_weights, device)

        # Fine-tune
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, cfg,
            sparsity_label=f"movement@{sp*100:.0f}%",
        )

        # Load best, verify sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\n  ✅ Movement @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"Movement @ {sp*100:.0f}% sparsity",
            indent="  ",
        )

        # Save weights-only .pth
        pth_path = os.path.join(out_dir, f"movement_sparsity{int(sp*100):02d}.pth")
        save_weights_only(model, pth_path)

        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
        }

    # ── Save results JSON ──────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'movement_results.json')
    save_results_json(
        {
            'scorer': 'movement',
            'movement_warmup_epochs': cfg['movement_warmup_epochs'],
            'baseline': baseline_metrics,
            'sparsity_results': {
                f"{int(sp*100):02d}pct": v for sp, v in results.items()
            },
        },
        json_path,
    )

    print_sparsity_table('movement', results, baseline_metrics)

    return results
