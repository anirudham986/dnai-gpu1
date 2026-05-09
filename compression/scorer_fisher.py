# =============================================================================
# compression/scorer_fisher.py
# LTH compression using ONLY Fisher Information as the importance signal.
#
# Rationale: The diagonal Fisher Information F_i = E[(∂L/∂w_i)²] measures
# the curvature of the loss landscape with respect to each weight. A large
# F_i means the loss is highly sensitive to perturbations of w_i — i.e.,
# the weight is critical and should NOT be pruned.
#
# This is mathematically motivated: the second-order Taylor expansion of
# the loss change when removing weight w_i is (1/2) * w_i² * F_i.
# Pruning by Fisher therefore minimises the actual loss perturbation.
#
# Estimation: Monte Carlo over `fisher_batches` training batches.
# Threshold: global k-th percentile on Fisher scores.
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
    FisherInformationEstimator,
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
    Run Fisher-information-only LTH at each sparsity level.

    Phases:
      A. Fisher estimation — accumulate (∂L/∂w)² over `fisher_batches` batches.
         Uses the dense model. Fisher scores are computed once and reused.
      B. Per-sparsity LTH — mask by Fisher rank → rewind → fine-tune → save.

    Returns:
        results: {sparsity_float: metrics_dict}
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  SCORER: FISHER INFORMATION — LTH Compression")
    print(f"{'═'*70}")
    print(f"  Importance signal: F_i = E[(∂L/∂w_i)²]  (diagonal FIM)")
    print(f"  Fisher batches:    {cfg['fisher_batches']}")
    print(f"  Threshold method:  global k-th percentile on Fisher scores")
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

    # ── Phase A: Fisher Estimation ─────────────────────────────────────────
    # Use the clean dense model — Fisher on the trained weights gives the
    # most accurate curvature estimate for the task loss.
    print(f"\n  ── Phase A: Fisher Information Estimation ──")
    print(f"  Accumulating over {cfg['fisher_batches']} training batches...")
    fisher_est = FisherInformationEstimator(model, device)
    fisher_est.accumulate(model, train_loader, n_batches=cfg['fisher_batches'])

    fisher_scores = fisher_est.get_scores()
    # Convert to percentile ranks once, reused for all sparsities
    fisher_ranked = {name: percentile_rank(s) for name, s in fisher_scores.items()}
    print(f"  ✅ Fisher ranked for {len(fisher_ranked)} parameter tensors")

    results = {}

    # ── Phase B: Per-sparsity LTH ─────────────────────────────────────────
    for sp in sparsities:
        print(f"\n{'─'*70}")
        print(f"  FISHER @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        # Reload dense weights (fresh init for each sparsity)
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Create masks
        masks, achieved_sp = create_masks(fisher_ranked, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in fisher_ranked.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # LTH Rewind
        lth_rewind(model, masks, initial_weights, device)

        # Fine-tune
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, cfg,
            sparsity_label=f"fisher@{sp*100:.0f}%",
        )

        # Load best, verify sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\n  ✅ Fisher @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"Fisher @ {sp*100:.0f}% sparsity",
            indent="  ",
        )

        # Save weights-only .pth
        pth_path = os.path.join(out_dir, f"fisher_sparsity{int(sp*100):02d}.pth")
        save_weights_only(model, pth_path)

        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
        }

    # ── Save results JSON ──────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'fisher_results.json')
    save_results_json(
        {
            'scorer': 'fisher',
            'fisher_batches': cfg['fisher_batches'],
            'baseline': baseline_metrics,
            'sparsity_results': {
                f"{int(sp*100):02d}pct": v for sp, v in results.items()
            },
        },
        json_path,
    )

    print_sparsity_table('fisher', results, baseline_metrics)

    return results
