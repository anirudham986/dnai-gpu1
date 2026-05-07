#!/usr/bin/env python3
# =====================================================================
# train.py — Main Entry Point for NTv2 Multi-Dataset Training
# =====================================================================
#
# Usage:
#   python train.py                         # Interactive dataset selection
#   python train.py --dataset clinvar       # ClinVar (direct)
#   python train.py --dataset dbsnp         # dbSNP (direct)
#   python train.py --dataset cbioportal    # cBioPortal + gnomAD
#   python train.py --dataset consolidated  # All 4 merged (RECOMMENDED)
#   python train.py --dataset consolidated --fold 2   # Specific validation fold
#   python train.py --dataset consolidated --resume /kaggle/working/ntv2_consolidated_trained/ntv2_consolidated_fold0_timed_ep5_*.pth
#
# GPU:   Designed for NVIDIA A100/T4/P100 on Kaggle
# Model: InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# =====================================================================

import subprocess
import sys

# Install dependencies (safe for Kaggle — skips if already installed)
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu124']
)
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'transformers==4.40.2', 'pyfaidx', 'scikit-learn']
)

import argparse
import os
import json
import glob
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# Local imports
from config import get_config, DATASET_CHOICES
from data import load_dataset, DualSeqDataset, run_audit, LeakageAuditError
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate, load_checkpoint, find_latest_checkpoint
from utils import load_hg38, set_seed, get_device, supports_amp


# =====================================================================
# 1. DATASET SELECTION (interactive or CLI argument)
# =====================================================================
def select_dataset() -> str:
    """Interactive dataset selection prompt."""
    print("\n" + "=" * 70)
    print("   NTv2 MULTI-DATASET TRAINING PIPELINE")
    print("=" * 70)
    print("\n   Available datasets:\n")
    print("     [1] clinvar      — ClinVar 75k (Pathogenic + Benign)")
    print("                        Clinical significance annotations")
    print()
    print("     [2] dbsnp        — dbSNP 62k (Pathogenic + Benign)")
    print("                        Common variants + ClinVar cross-ref")
    print()
    print("     [3] cbioportal   — cBioPortal 63k + gnomAD 55k (Combined)")
    print("                        Cancer somatic (P) + population (B)")
    print()
    print("     [4] consolidated — ALL 4 datasets merged & deduplicated ★ K-FOLD")
    print("                        Zero leakage • 5-fold stratified • source-diverse")
    print()
    print("     [5] consolidated_full — Full 100k train + 25k unseen holdout ★ COMPRESSION")
    print("                        Train on ALL 100k • Test on 25k never-seen variants")
    print("                        Final model for compression pipeline")
    print()

    while True:
        choice = input("   Enter your choice [1/2/3/4 or name]: ").strip().lower()
        if choice in DATASET_CHOICES:
            dataset_name = DATASET_CHOICES[choice]
            print(f"\n   ✅ Selected: {dataset_name}")
            return dataset_name
        print(f"   ❌ Invalid choice '{choice}'. "
              f"Please enter 1, 2, 3, 4, or a dataset name.")


def parse_args():
    """Parse CLI arguments for non-interactive usage."""
    parser = argparse.ArgumentParser(
        description="NTv2 Multi-Dataset Variant Effect Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                               # Interactive mode
  python train.py --dataset clinvar             # ClinVar dataset
  python train.py --dataset consolidated        # All 4 merged (recommended)
  python train.py --dataset consolidated --fold 2  # Specific validation fold
  python train.py --dataset consolidated --resume /kaggle/working/.../checkpoint.pth
  python train.py --dataset consolidated --auto_resume  # Find & resume latest checkpoint
        """
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['clinvar', 'dbsnp', 'cbioportal', 'consolidated', 'consolidated_full'],
        help='Dataset to train on (skips interactive prompt)'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Validation fold ID (0–4) for consolidated dataset'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='Override checkpoint save directory'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a checkpoint .pth file to resume training from'
    )
    parser.add_argument(
        '--auto_resume', action='store_true',
        help='Automatically find and resume from the latest checkpoint in save_dir'
    )
    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================
def main():
    args = parse_args()

    # --- Dataset selection ---
    if args.dataset:
        dataset_name = args.dataset
        print(f"\n   Dataset (CLI): {dataset_name}")
    else:
        dataset_name = select_dataset()

    # --- Configuration ---
    cfg = get_config(dataset_name, val_fold=args.fold)

    # Apply CLI overrides
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.save_dir:
        cfg['save_dir'] = args.save_dir

    # --- Resolve resume path ---
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(cfg['save_dir'])
        if resume_path:
            print(f"\n   🔍 Auto-resume: found {os.path.basename(resume_path)}")
        else:
            print(f"\n   🔍 Auto-resume: no checkpoint found — starting fresh")

    # Handle glob pattern in resume path (e.g., timed_ep5_*.pth)
    if resume_path and '*' in resume_path:
        matches = sorted(glob.glob(resume_path), key=os.path.getmtime)
        if matches:
            resume_path = matches[-1]
        else:
            raise FileNotFoundError(f"No checkpoint matched pattern: {resume_path}")

    # --- Banner ---
    print("\n" + "=" * 70)
    print("   NT v2 — VARIANT EFFECT PREDICTION (GLRB-OPTIMIZED)")
    print("=" * 70)
    print(f"   Target:    Beat GLRB benchmark (≥75% AUROC)")
    print(f"   Model:     {cfg['model_name']}")
    print(f"   Dataset:   {dataset_name}")
    if dataset_name == 'consolidated':
        print(f"   Val fold:  {cfg['val_fold']} / {cfg.get('n_folds', 5) - 1}")
    elif dataset_name == 'consolidated_full':
        print(f"   Mode:      Full 100k train + 25k unseen holdout test")
        print(f"   Purpose:   Final model for compression pipeline")
    print(f"   Approach:  Dual-sequence + Full FT + Focal loss")
    print(f"   Context:   {cfg['seq_length']}bp from hg38")
    print(f"   Epochs:    {cfg['epochs']} | Batch: {cfg['batch_size']} "
          f"(eff: {cfg['batch_size'] * cfg['grad_accum_steps']})")
    print(f"   Resume:    {resume_path or 'None (fresh start)'}")
    print("=" * 70)

    # --- Seed ---
    set_seed(cfg['seed'])

    # --- Device ---
    print("\n" + "-" * 70)
    print("1. Device Setup")
    print("-" * 70)
    device  = get_device()
    use_amp = supports_amp()

    # --- Reference Genome ---
    print("\n" + "-" * 70)
    print("2. Loading hg38 Reference Genome")
    print("-" * 70)
    genome, has_chr = load_hg38()

    # --- Tokenizer ---
    print("\n" + "-" * 70)
    print("3. Loading NT v2 Tokenizer")
    print("-" * 70)
    tokenizer  = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Model ---
    print("\n" + "-" * 70)
    print("4. Building NTv2 Dual-Sequence Classifier")
    print("-" * 70)
    model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Total: {total_params:,} | "
          f"Trainable: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    # --- Data Loading ---
    print("\n" + "-" * 70)
    print(f"5. Loading Dataset: {dataset_name}")
    print("-" * 70)

    # Auto-generate holdout CSV if using consolidated_full and it doesn't exist
    if dataset_name == 'consolidated_full':
        from data.build_holdout import build_holdout, _find_data_dir
        try:
            data_dir_check = _find_data_dir()
            holdout_path = os.path.join(data_dir_check, '06_holdout_25k_unseen.csv')
            if not os.path.exists(holdout_path):
                print("\n   ⚠️ Holdout CSV not found — generating automatically...")
                build_holdout(data_dir_check, data_dir_check)
                print()
        except Exception as e:
            print(f"   ⚠️ Auto-generation failed: {e}")
            print("   Run 'python data/build_holdout.py' manually.")
            sys.exit(1)

    train_df, val_df = load_dataset(
        dataset_name,
        max_per_class=cfg['max_per_class'],
        seed=cfg['seed'],
        val_fold=cfg.get('val_fold', 0),
    )

    # ================================================================
    # MANDATORY LEAKAGE AUDIT — Training will ABORT if this fails
    # ================================================================
    try:
        run_audit(
            train_df, val_df,
            dataset_name=dataset_name,
            val_fold=cfg.get('val_fold', -1)
        )
    except LeakageAuditError as e:
        print("\n" + "!" * 70)
        print("   🚨 LEAKAGE AUDIT FAILED — TRAINING ABORTED")
        print("!" * 70)
        print(str(e))
        print("\n   Fix the data issue above before rerunning. "
              "Results on leaky data are scientifically invalid.")
        sys.exit(1)

    # --- Build DualSeqDatasets ---
    print("\n" + "-" * 70)
    print("6. Building Dual-Sequence Datasets")
    print("-" * 70)

    print("\n   --- Train Set ---")
    train_dataset = DualSeqDataset(
        train_df, genome, tokenizer, has_chr,
        seq_len=cfg['seq_length'],
        max_tokens=max_tokens,
        seed=cfg['seed'],
    )

    print("\n   --- Validation Set ---")
    val_dataset = DualSeqDataset(
        val_df, genome, tokenizer, has_chr,
        seq_len=cfg['seq_length'],
        max_tokens=max_tokens,
        seed=cfg['seed'] + 1,
    )

    # --- DataLoaders ---
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=pin_memory,
    )

    print(f"\n   Train: {len(train_dataset):,} samples | "
          f"Val: {len(val_dataset):,} samples")
    print(f"   Batches/epoch: {len(train_loader)} | "
          f"Effective batch: {cfg['batch_size'] * cfg['grad_accum_steps']}")

    # --- Training ---
    print("\n" + "-" * 70)
    print("7. Training (Full Fine-Tuning)")
    print("-" * 70)

    criterion = FocalLoss(
        gamma=cfg['focal_gamma'],
        label_smoothing=cfg['label_smoothing'],
    )

    best_acc, best_auroc, history = train(
        model, train_loader, val_loader, device, criterion, cfg,
        use_amp=use_amp,
        resume_from=resume_path,
    )

    # --- Final Evaluation ---
    print("\n" + "-" * 70)
    print("8. Final Evaluation")
    print("-" * 70)

    final = evaluate(model, val_loader, device, full=True, use_amp=use_amp)

    print(f"\n   Accuracy:    {final['accuracy']:.2f}%")
    print(f"   AUROC:       {final['auroc']:.2f}%")
    print(f"   F1:          {final['f1']:.2f}%")
    print(f"   MCC:         {final['mcc']:.4f}")
    print(f"   Precision:   {final['precision']:.2f}%")
    print(f"   Recall:      {final['recall']:.2f}%")
    print(f"   Specificity: {final['specificity']:.2f}%")
    print(f"   TP={final['tp']} FP={final['fp']} "
          f"FN={final['fn']} TN={final['tn']}")

    glrb_threshold = 75.0
    if final['auroc'] >= glrb_threshold:
        print(f"\n   ✅ BEATS GLRB benchmark "
              f"({final['auroc']:.2f}% ≥ {glrb_threshold}%)")
    else:
        gap = glrb_threshold - final['auroc']
        print(f"\n   📊 GLRB benchmark: {glrb_threshold}% — gap: {gap:.2f}%")

    # --- Save Final Checkpoint ---
    print("\n" + "-" * 70)
    print("9. Saving Final Model")
    print("-" * 70)

    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == 'consolidated_full':
        fold_tag = ''
        model_filename = f"ntv2_{dataset_name}_final.pth"
    elif dataset_name == 'consolidated':
        fold_tag = f"fold{cfg['val_fold']}_"
        model_filename = f"ntv2_{dataset_name}_{fold_tag}final.pth"
    else:
        fold_tag = ''
        model_filename = f"ntv2_{dataset_name}_final.pth"

    model_path = os.path.join(save_dir, model_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name':             cfg['model_name'],
            'hidden_size':            model.hidden_size,
            'num_layers_to_unfreeze': cfg['num_layers_to_unfreeze'],
            'dropout':                cfg['dropout'],
            'best_accuracy':          best_acc,
            'best_auroc':             best_auroc,
            'pooling':                'mean',
            'approach':               'dual_sequence_focal',
            'seq_length':             cfg['seq_length'],
            'full_finetune':          True,
            'dataset':                dataset_name,
            'val_fold':               cfg.get('val_fold', -1),
        },
        'final_metrics':   final,
        'hyperparameters': cfg,
        'history':         history,
    }, model_path)
    print(f"   ✅ Model: {model_path}")

    # Save training history JSON
    history_path = os.path.join(save_dir, f'training_history_{fold_tag}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save dataset info
    info_path = os.path.join(save_dir, f'dataset_info_{fold_tag}.json')
    with open(info_path, 'w') as f:
        json.dump({
            'dataset':        dataset_name,
            'val_fold':       cfg.get('val_fold', -1),
            'train_samples':  len(train_dataset),
            'val_samples':    len(val_dataset),
            'approach':       'dual_seq_focal_fullft_kfold' if dataset_name == 'consolidated' else 'dual_seq_focal_fullft_holdout',
            'seq_length':     cfg['seq_length'],
            'leakage_audit':  'PASSED',
        }, f, indent=2)
    print(f"   ✅ History + dataset info saved")

    # --- Save Standalone Weights for Compression Pipeline ---
    print("\n" + "-" * 70)
    print("10. Saving Compression-Ready Weights")
    print("-" * 70)

    weights_filename = f"ntv2_{dataset_name}_{fold_tag}weights.pth"
    weights_path     = os.path.join(save_dir, weights_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name':       cfg['model_name'],
        'hidden_size':      model.hidden_size,
        'num_layers_to_unfreeze': cfg['num_layers_to_unfreeze'],
        'dropout':          cfg['dropout'],
        'seq_length':       cfg['seq_length'],
        'dataset':          dataset_name,
        'metrics': {
            'accuracy':    final['accuracy'],
            'auroc':       final['auroc'],
            'f1':          final['f1'],
            'mcc':         final['mcc'],
            'precision':   final['precision'],
            'recall':      final['recall'],
            'specificity': final['specificity'],
            'tp':          final['tp'],
            'fp':          final['fp'],
            'fn':          final['fn'],
            'tn':          final['tn'],
        },
        'best_accuracy': best_acc,
        'best_auroc':    best_auroc,
        'train_acc':     history['train_acc'][-1] if history['train_acc'] else None,
        'train_loss':    history['train_loss'][-1] if history['train_loss'] else None,
    }, weights_path)

    print(f"   ✅ Compression-ready weights: {weights_path}")
    print(f"   📦 Contains: state_dict + baseline metrics ONLY")
    print(f"   🔗 Use this single file as input to the compression pipeline")

    if dataset_name == 'consolidated_full':
        print(f"\n   🎯 This is the FINAL model for compression.")
        print(f"   📁 Load with: torch.load('{weights_path}')")

    # --- Checkpoint Verification ---
    print("\n" + "-" * 70)
    print("11. Checkpoint Verification")
    print("-" * 70)

    verify_model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    verify_model.load_state_dict(ckpt['model_state_dict'])
    v_acc, v_auroc, _ = evaluate(verify_model, val_loader, device, use_amp=use_amp)
    match = abs(v_acc - final['accuracy']) < 0.01
    print(f"   Reload: Acc={v_acc:.2f}%, AUROC={v_auroc:.2f}% — "
          f"{'✅ PASS' if match else '❌ FAIL'}")
    del verify_model

    # --- Summary ---
    print("\n" + "=" * 70)
    print("🎯 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Dataset:    {dataset_name}")
    if dataset_name == 'consolidated':
        print(f"   Val fold:   {cfg['val_fold']} / {cfg.get('n_folds', 5) - 1}")
    elif dataset_name == 'consolidated_full':
        print(f"   Mode:       Full 100k train + 25k unseen holdout")
    print(f"   Approach:   Dual-seq + Focal loss + Full fine-tuning")
    print(f"   Samples:    {len(train_dataset) + len(val_dataset):,} total")
    print(f"   Context:    {cfg['seq_length']}bp from hg38")
    print(f"   Accuracy:   {final['accuracy']:.2f}%")
    print(f"   AUROC:      {final['auroc']:.2f}%")
    print(f"   F1:         {final['f1']:.2f}%")
    print(f"   MCC:        {final['mcc']:.4f}")
    print(f"   GLRB ref:   {glrb_threshold}% AUROC")
    print(f"   Leakage:    ✅ ZERO (audited before training)")
    print(f"   Saved:      {save_dir}/")
    if dataset_name == 'consolidated_full':
        print("   ✅ READY FOR COMPRESSION (this is the final model)")
    else:
        print("   ✅ READY FOR COMPRESSION (next stage)")
    print("=" * 70)


if __name__ == '__main__':
    main()
