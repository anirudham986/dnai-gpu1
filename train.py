#!/usr/bin/env python3
# =====================================================================
# train.py — Main Entry Point for NTv2 Multi-Dataset Training
# =====================================================================
#
# Usage:
#   python train.py                    # Interactive dataset selection
#   python train.py --dataset clinvar  # Direct dataset specification
#   python train.py --dataset dbsnp
#   python train.py --dataset cbioportal
#
# GPU:  Designed for NVIDIA A100 (16-20 GB) on Kaggle/remote
# Model: InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# =====================================================================

import subprocess
import sys

# Install dependencies (safe for Kaggle — skips if already installed)
# NOTE: Kaggle's default PyTorch may not support older GPUs (P100 = sm_60).
# We install a compatible version + pin transformers for reproducibility.
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu124']
)
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'transformers==4.40.2', 'pyfaidx']
)

import argparse
import os
import json
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# Local imports
from config import get_config, DATASET_CHOICES
from data import load_dataset, DualSeqDataset
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate
from utils import load_hg38, set_seed, get_device, supports_amp


# =====================================================================
# 1. DATASET SELECTION (interactive or CLI argument)
# =====================================================================
def select_dataset() -> str:
    """
    Interactive dataset selection prompt.
    Returns the canonical dataset name.
    """
    print("\n" + "=" * 70)
    print("   NTv2 MULTI-DATASET TRAINING PIPELINE")
    print("=" * 70)
    print("\n   Available datasets:\n")
    print("     [1] clinvar     — ClinVar 75k (Pathogenic + Benign)")
    print("                       Clinical significance annotations")
    print()
    print("     [2] dbsnp       — dbSNP 62k (Pathogenic + Benign)")
    print("                       Common variants + ClinVar cross-ref")
    print()
    print("     [3] cbioportal  — cBioPortal 63k + gnomAD 55k (Combined)")
    print("                       Cancer somatic (P) + population (B)")
    print()

    while True:
        choice = input("   Enter your choice [1/2/3 or name]: ").strip().lower()
        if choice in DATASET_CHOICES:
            dataset_name = DATASET_CHOICES[choice]
            print(f"\n   ✅ Selected: {dataset_name}")
            return dataset_name
        print(f"   ❌ Invalid choice '{choice}'. Please enter 1, 2, 3, "
              f"or a dataset name (clinvar/dbsnp/cbioportal)")


def parse_args():
    """Parse CLI arguments for non-interactive usage."""
    parser = argparse.ArgumentParser(
        description="NTv2 Multi-Dataset Variant Effect Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                       # Interactive mode
  python train.py --dataset clinvar     # ClinVar dataset
  python train.py --dataset dbsnp       # dbSNP dataset
  python train.py --dataset cbioportal  # cBioPortal + gnomAD combined
        """
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['clinvar', 'dbsnp', 'cbioportal'],
        help='Dataset to train on (skips interactive prompt)'
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
    cfg = get_config(dataset_name)

    # Apply CLI overrides
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.save_dir:
        cfg['save_dir'] = args.save_dir

    # --- Banner ---
    print("\n" + "=" * 70)
    print("   NT v2 — VARIANT EFFECT PREDICTION (GLRB-OPTIMIZED)")
    print("=" * 70)
    print(f"   Target:    Beat GLRB benchmark (0.75 AUROC)")
    print(f"   Model:     {cfg['model_name']}")
    print(f"   Dataset:   {dataset_name}")
    print(f"   Approach:  Dual-sequence + Full FT + Focal loss")
    print(f"   Context:   {cfg['seq_length']}bp from hg38")
    print(f"   Epochs:    {cfg['epochs']} | Batch: {cfg['batch_size']} "
          f"(eff: {cfg['batch_size'] * cfg['grad_accum_steps']})")
    print("=" * 70)

    # --- Seed ---
    set_seed(cfg['seed'])

    # --- Device ---
    print("\n" + "-" * 70)
    print("1. Device Setup")
    print("-" * 70)
    device = get_device()
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
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model_name'], trust_remote_code=True
    )
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer: vocab={tokenizer.vocab_size}, "
          f"max_tokens={max_tokens}")

    # --- Model ---
    print("\n" + "-" * 70)
    print("4. Building NTv2 Dual-Sequence Classifier")
    print("-" * 70)
    model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"   ✅ Total: {total_params:,} | "
          f"Trainable: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"   Full fine-tuning: {cfg['num_layers_to_unfreeze']} layers")
    print(f"   LR: backbone={cfg['backbone_lr']}, head={cfg['head_lr']}")

    # --- Data Loading ---
    print("\n" + "-" * 70)
    print(f"5. Loading Dataset: {dataset_name}")
    print("-" * 70)
    train_df, test_df = load_dataset(
        dataset_name,
        max_per_class=cfg['max_per_class'],
        seed=cfg['seed'],
    )

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

    print("\n   --- Test Set ---")
    test_dataset = DualSeqDataset(
        test_df, genome, tokenizer, has_chr,
        seq_len=cfg['seq_length'],
        max_tokens=max_tokens,
        seed=cfg['seed'] + 1,  # Different seed for test
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=pin_memory,
    )

    print(f"\n   Train: {len(train_dataset):,} samples | "
          f"Test: {len(test_dataset):,} samples")
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
        model, train_loader, test_loader, device, criterion, cfg,
        use_amp=use_amp,
    )

    # --- Final Evaluation ---
    print("\n" + "-" * 70)
    print("8. Final Evaluation")
    print("-" * 70)

    final = evaluate(model, test_loader, device, full=True,
                     use_amp=use_amp)

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

    # --- Save Checkpoint ---
    print("\n" + "-" * 70)
    print("9. Saving Checkpoint")
    print("-" * 70)

    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    model_filename = f"ntv2_{dataset_name}_trained.pth"
    model_path = os.path.join(save_dir, model_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': cfg['model_name'],
            'hidden_size': model.hidden_size,
            'num_layers_to_unfreeze': cfg['num_layers_to_unfreeze'],
            'dropout': cfg['dropout'],
            'best_accuracy': best_acc,
            'best_auroc': best_auroc,
            'pooling': 'mean',
            'approach': 'dual_sequence_focal',
            'seq_length': cfg['seq_length'],
            'full_finetune': True,
            'dataset': dataset_name,
        },
        'final_metrics': final,
        'hyperparameters': cfg,
    }, model_path)
    print(f"   ✅ Model: {model_path}")

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save dataset info
    info_path = os.path.join(save_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'approach': 'dual_seq_focal_fullft',
            'seq_length': cfg['seq_length'],
        }, f, indent=2)
    print(f"   ✅ History + dataset info saved")

    # --- Verify Checkpoint ---
    print("\n" + "-" * 70)
    print("10. Checkpoint Verification")
    print("-" * 70)

    verify_model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device,
                            weights_only=False)
    verify_model.load_state_dict(checkpoint['model_state_dict'])

    v_acc, v_auroc, _ = evaluate(verify_model, test_loader, device,
                                  use_amp=use_amp)
    match = abs(v_acc - final['accuracy']) < 0.01
    print(f"   Reload: Acc={v_acc:.2f}%, AUROC={v_auroc:.2f}% — "
          f"{'✅ PASS' if match else '❌ FAIL'}")
    del verify_model

    # --- Summary ---
    print("\n" + "=" * 70)
    print("🎯 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Dataset:    {dataset_name}")
    print(f"   Approach:   Dual-seq + Focal loss + Full fine-tuning")
    print(f"   Samples:    {len(train_dataset) + len(test_dataset):,} total")
    print(f"   Context:    {cfg['seq_length']}bp from hg38")
    print(f"   Accuracy:   {final['accuracy']:.2f}%")
    print(f"   AUROC:      {final['auroc']:.2f}%")
    print(f"   F1:         {final['f1']:.2f}%")
    print(f"   MCC:        {final['mcc']:.4f}")
    print(f"   GLRB ref:   {glrb_threshold}% AUROC")
    print(f"   Saved:      {save_dir}/")
    print("   ✅ READY FOR COMPRESSION (next stage)")
    print("=" * 70)


if __name__ == '__main__':
    main()
