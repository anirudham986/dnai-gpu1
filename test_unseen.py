#!/usr/bin/env python3
# =====================================================================
# test_unseen.py — Evaluate trained NTv2 model on 3 unseen test sets
# =====================================================================
#
# Usage (Kaggle or local):
#   python test_unseen.py --weights /path/to/ntv2_weights.pth
#   python test_unseen.py --weights /path/to/ntv2_weights.pth --data_dir "crct dataset"
#
# This script evaluates the trained NTv2DualSeqClassifier on the 3 held-out
# test sets that were NEVER seen during training or validation:
#
#   07_clinvar_test_unseen.csv       — ClinVar only
#   08_dbsnp_test_unseen.csv         — dbSNP only
#   09_cbio_gnomad_test_unseen.csv   — cBioPortal + gnomAD
#
# Outputs:
#   - Per-test-set metrics (accuracy, AUROC, F1, MCC, precision, recall,
#     specificity, confusion matrix)
#   - Comparative summary table across all 3 test sets
#   - JSON results file for downstream analysis
# =====================================================================

import argparse
import json
import os
import sys
import time
import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports
from model import NTv2DualSeqClassifier
from engine import evaluate, print_metrics
from data import DualSeqDataset
from utils import load_hg38, set_seed, get_device, supports_amp

import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

TEST_FILES = {
    'clinvar': {
        'file': '07_clinvar_test_unseen.csv',
        'name': 'ClinVar (Unseen)',
        'description': 'Clinical variant significance — single-source test',
    },
    'dbsnp': {
        'file': '08_dbsnp_test_unseen.csv',
        'name': 'dbSNP (Unseen)',
        'description': 'Common/ClinVar cross-referenced variants',
    },
    'cbio_gnomad': {
        'file': '09_cbio_gnomad_test_unseen.csv',
        'name': 'cBioPortal + gnomAD (Unseen)',
        'description': 'Cancer somatic (P) + population frequency (B)',
    },
}

# Default model hyperparameters — must match the model that was trained
MODEL_DEFAULTS = {
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'num_layers_to_unfreeze': 22,
    'dropout': 0.2,
    'seq_length': 1000,
    'batch_size': 32,
    'num_workers': 4,
    'seed': 42,
}


# =====================================================================
# HELPERS
# =====================================================================

def find_data_dir(data_dir_hint: str = None) -> str:
    """Find the directory containing the test CSV files."""
    candidates = []
    if data_dir_hint:
        candidates.append(data_dir_hint)
    candidates.extend([
        os.path.join(os.path.dirname(__file__), "crct dataset"),
        "/kaggle/input",
        "/kaggle/working",
        "crct dataset",
        ".",
    ])

    test_file = TEST_FILES['clinvar']['file']  # 07_clinvar_test_unseen.csv

    for d in candidates:
        if os.path.exists(d):
            if os.path.isfile(os.path.join(d, test_file)):
                return d
            # Search subdirectories
            for dirpath, _, files in os.walk(d):
                if test_file in files:
                    return dirpath
    raise FileNotFoundError(
        f"Could not find test files (e.g., {test_file}). "
        f"Searched: {candidates}. "
        f"Use --data_dir to specify the directory."
    )


def load_test_csv(data_dir: str, file_key: str) -> pd.DataFrame:
    """Load a test CSV and standardize columns."""
    info = TEST_FILES[file_key]
    path = os.path.join(data_dir, info['file'])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)

    # Standardize label column: INT_LABEL → LABEL
    if 'LABEL' not in df.columns and 'INT_LABEL' in df.columns:
        df = df.rename(columns={'INT_LABEL': 'LABEL'})
    elif 'LABEL' not in df.columns:
        raise KeyError(f"No LABEL or INT_LABEL column in {path}")

    return df


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NTv2 model on 3 unseen holdout test sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_unseen.py --weights output/ntv2_consolidated_full_trained/ntv2_consolidated_full_final.pth
  python test_unseen.py --weights model.pth --data_dir "crct dataset" --batch_size 64
        """
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to the trained model weights (.pth file, state_dict only)'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Directory containing 07/08/09 test CSV files'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save results JSON (default: same as weights dir)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=MODEL_DEFAULTS['batch_size'],
        help=f"Batch size for evaluation (default: {MODEL_DEFAULTS['batch_size']})"
    )
    parser.add_argument(
        '--model_name', type=str, default=MODEL_DEFAULTS['model_name'],
        help='HuggingFace model name (must match training)'
    )
    parser.add_argument(
        '--num_layers_to_unfreeze', type=int,
        default=MODEL_DEFAULTS['num_layers_to_unfreeze'],
        help='Number of layers unfrozen during training (must match)'
    )
    parser.add_argument(
        '--dropout', type=float, default=MODEL_DEFAULTS['dropout'],
        help='Dropout rate (must match training)'
    )
    parser.add_argument(
        '--seq_length', type=int, default=MODEL_DEFAULTS['seq_length'],
        help='Sequence context length in bp (must match training)'
    )
    parser.add_argument(
        '--seed', type=int, default=MODEL_DEFAULTS['seed'],
        help='Random seed'
    )
    args = parser.parse_args()

    start_time = time.time()

    # --- Validate weights path ---
    if not os.path.exists(args.weights):
        print(f"\n❌ Weights file not found: {args.weights}")
        sys.exit(1)

    # --- Banner ---
    print("\n" + "=" * 70)
    print("   NTv2 — UNSEEN HOLDOUT TEST EVALUATION")
    print("=" * 70)
    print(f"   Weights:  {args.weights}")
    print(f"   Model:    {args.model_name}")
    print(f"   Context:  {args.seq_length}bp | Batch: {args.batch_size}")
    print(f"   Tests:    07 ClinVar | 08 dbSNP | 09 cBioPortal+gnomAD")
    print("=" * 70)

    # --- Seed & Device ---
    set_seed(args.seed)
    device = get_device()
    use_amp = supports_amp()

    # --- Reference Genome ---
    print("\n" + "-" * 70)
    print("1. Loading hg38 Reference Genome")
    print("-" * 70)
    genome, has_chr = load_hg38()

    # --- Tokenizer ---
    print("\n" + "-" * 70)
    print("2. Loading Tokenizer")
    print("-" * 70)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer loaded: vocab={tokenizer.vocab_size}, "
          f"max_tokens={max_tokens}")

    # --- Load Model ---
    print("\n" + "-" * 70)
    print("3. Loading Trained Model")
    print("-" * 70)

    model = NTv2DualSeqClassifier(
        model_name=args.model_name,
        num_layers_to_unfreeze=args.num_layers_to_unfreeze,
        dropout=args.dropout,
    ).to(device)

    # Load weights
    state_dict = torch.load(
        args.weights, map_location=device, weights_only=True
    )

    # Handle both raw state_dict and wrapped dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded: {total_params:,} parameters")
    weights_size_mb = os.path.getsize(args.weights) / 1e6
    print(f"   📦 Weights file: {weights_size_mb:.1f} MB")

    # --- Find test data ---
    print("\n" + "-" * 70)
    print("4. Loading Test Datasets")
    print("-" * 70)

    data_dir = find_data_dir(args.data_dir)
    print(f"   Data dir: {data_dir}")

    # --- Evaluate each test set ---
    all_results = {}
    pin_memory = (device.type == 'cuda')

    for i, (key, info) in enumerate(TEST_FILES.items(), 1):
        print(f"\n{'─' * 70}")
        print(f"   TEST {i}/3: {info['name']}")
        print(f"   {info['description']}")
        print(f"{'─' * 70}")

        # Load CSV
        test_df = load_test_csv(data_dir, key)
        n_p = int((test_df['LABEL'] == 1).sum())
        n_b = int((test_df['LABEL'] == 0).sum())
        print(f"\n   Loaded: {info['file']} — {len(test_df):,} samples")
        print(f"   Pathogenic: {n_p:,} | Benign: {n_b:,}")

        if 'SOURCE_TAG' in test_df.columns:
            sources = test_df['SOURCE_TAG'].value_counts()
            for src, cnt in sources.items():
                print(f"     {src}: {cnt:,}")

        # Build DualSeqDataset
        test_dataset = DualSeqDataset(
            test_df, genome, tokenizer, has_chr,
            seq_len=args.seq_length,
            max_tokens=max_tokens,
            seed=args.seed + 100 + i,  # Different seed than training
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=MODEL_DEFAULTS['num_workers'],
            pin_memory=pin_memory,
        )

        # Evaluate
        metrics = evaluate(
            model, test_loader, device, use_amp=use_amp,
            desc=f"Test {i} ({key})"
        )

        # Store results
        all_results[key] = {
            'name': info['name'],
            'file': info['file'],
            'description': info['description'],
            'total_csv_samples': len(test_df),
            'extracted_samples': len(test_dataset),
            'pathogenic': n_p,
            'benign': n_b,
            'metrics': metrics,
        }

        # Print results
        print(f"\n   📊 Results for {info['name']}:")
        print_metrics(metrics, prefix="   ")

    # --- Comparative Summary ---
    print("\n" + "=" * 70)
    print("   COMPARATIVE RESULTS — ALL 3 UNSEEN TEST SETS")
    print("=" * 70)

    header = (
        f"{'Test Set':<30} {'Acc%':>7} {'AUROC%':>7} {'F1%':>7} "
        f"{'MCC':>7} {'Prec%':>7} {'Rec%':>7} {'Spec%':>7} {'N':>7}"
    )
    print(f"\n   {header}")
    print(f"   {'─' * len(header)}")

    for key, result in all_results.items():
        m = result['metrics']
        row = (
            f"{result['name']:<30} "
            f"{m['accuracy']:>7.2f} {m['auroc']:>7.2f} {m['f1']:>7.2f} "
            f"{m['mcc']:>7.4f} {m['precision']:>7.2f} {m['recall']:>7.2f} "
            f"{m['specificity']:>7.2f} {m['n_samples']:>7,}"
        )
        print(f"   {row}")

    # --- Aggregate stats ---
    accs = [r['metrics']['accuracy'] for r in all_results.values()]
    aurocs = [r['metrics']['auroc'] for r in all_results.values()]
    f1s = [r['metrics']['f1'] for r in all_results.values()]
    mccs = [r['metrics']['mcc'] for r in all_results.values()]

    print(f"\n   {'─' * len(header)}")
    print(f"   {'MEAN':<30} "
          f"{sum(accs)/len(accs):>7.2f} {sum(aurocs)/len(aurocs):>7.2f} "
          f"{sum(f1s)/len(f1s):>7.2f} {sum(mccs)/len(mccs):>7.4f}")

    total_samples = sum(r['metrics']['n_samples'] for r in all_results.values())
    print(f"\n   Total unseen test samples evaluated: {total_samples:,}")

    # --- Save results JSON ---
    output_dir = args.output_dir or os.path.dirname(args.weights)
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'unseen_test_results.json')
    save_data = {
        'weights_file': os.path.abspath(args.weights),
        'model_name': args.model_name,
        'seq_length': args.seq_length,
        'timestamp': datetime.datetime.now().isoformat(),
        'test_sets': all_results,
        'aggregate': {
            'mean_accuracy': sum(accs) / len(accs),
            'mean_auroc': sum(aurocs) / len(aurocs),
            'mean_f1': sum(f1s) / len(f1s),
            'mean_mcc': sum(mccs) / len(mccs),
            'total_samples': total_samples,
        },
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n   ✅ Results saved: {results_path}")

    # --- Timing ---
    elapsed = time.time() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

    print(f"\n" + "=" * 70)
    print(f"   🎯 UNSEEN TEST EVALUATION COMPLETE")
    print(f"   Time: {elapsed_str}")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
