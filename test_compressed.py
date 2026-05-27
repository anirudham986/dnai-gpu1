#!/usr/bin/env python3
# =====================================================================
# test_compressed.py — Evaluate ALL compressed .pth files on the 3
#                       unseen holdout test sets (07/08/09)
# =====================================================================
#
# Usage (on GPU server):
#   python3 test_compressed.py
#
# This script:
#   1. Finds every .pth file in output/lth_compressed/
#   2. For each .pth: loads it into NTv2DualSeqClassifier, evaluates
#      on all 3 unseen test sets (07/08/09) — SAME data as test_unseen.py
#   3. Prints a final cross-scorer × cross-dataset comparison matrix
#   4. Saves results JSON for each scorer + a master summary JSON
#
# The .pth files are weights-only state_dicts, so this works identically
# to test_unseen.py — just automated over all 25 compressed models.
# =====================================================================

import os
import sys
import json
import time
import datetime
import argparse

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

MODEL_NAME = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
SEQ_LENGTH = 1000
BATCH_SIZE = 8       # match compression pipeline (safe for VRAM)
NUM_WORKERS = 2      # match compression pipeline
NUM_LAYERS_TO_UNFREEZE = 22
DROPOUT = 0.2
SEED = 42

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

# Scorers and sparsity levels to look for
SCORERS = ['magnitude', 'ema', 'fisher', 'movement', 'hybrid']
SPARSITIES = [10, 20, 30, 50, 60]

_HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# HELPERS
# =====================================================================

def find_data_dir(data_dir_hint=None):
    """Find the directory containing the test CSV files."""
    candidates = []
    if data_dir_hint:
        candidates.append(data_dir_hint)
    candidates.extend([
        os.path.join(_HERE, "crct dataset"),
        "crct dataset",
        ".",
    ])
    test_file = TEST_FILES['clinvar']['file']
    for d in candidates:
        if os.path.exists(d) and os.path.isfile(os.path.join(d, test_file)):
            return d
        if os.path.exists(d):
            for dirpath, _, files in os.walk(d):
                if test_file in files:
                    return dirpath
    raise FileNotFoundError(f"Could not find {test_file}. Use --data_dir.")


def find_compressed_pths(output_root):
    """Discover all compressed .pth files, return list of (scorer, sparsity_int, path)."""
    found = []
    for scorer in SCORERS:
        scorer_dir = os.path.join(output_root, scorer)
        if not os.path.isdir(scorer_dir):
            continue
        for sp in SPARSITIES:
            fn = f"{scorer}_sparsity{sp:02d}.pth"
            fp = os.path.join(scorer_dir, fn)
            if os.path.isfile(fp):
                found.append((scorer, sp, fp))
    return found


def evaluate_single_pth(pth_path, model, test_loaders, device, use_amp):
    """Load weights from pth into model, evaluate on all test sets."""
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    results = {}
    for key, loader in test_loaders.items():
        metrics = evaluate(model, loader, device, use_amp=use_amp,
                           desc=f"  {key}")
        results[key] = metrics

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all compressed LTH models on 3 unseen test sets."
    )
    parser.add_argument(
        '--output_root', type=str,
        default=os.path.join(_HERE, 'output', 'lth_compressed'),
        help='Root directory containing scorer subdirs with .pth files'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Directory containing 07/08/09 test CSV files'
    )
    parser.add_argument(
        '--dense_weights', type=str, default=None,
        help='Optional: path to dense model .pth to include as baseline'
    )
    args = parser.parse_args()

    start_time = time.time()

    # --- Banner ---
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  COMPRESSED MODEL EVALUATION — 3 UNSEEN HOLDOUT TEST SETS")
    print(f"{sep}")
    print(f"  Tests: 07 ClinVar | 08 dbSNP | 09 cBioPortal+gnomAD")
    print(f"  Output root: {args.output_root}")
    print(f"{sep}\n")

    # --- Discover .pth files ---
    pth_list = find_compressed_pths(args.output_root)
    if not pth_list:
        print("❌ No compressed .pth files found!")
        sys.exit(1)

    print(f"  Found {len(pth_list)} compressed .pth files:")
    for scorer, sp, fp in pth_list:
        sz = os.path.getsize(fp) / 1e6
        print(f"    {scorer:>12} @ {sp:>2}%  → {fp}  [{sz:.1f} MB]")

    # --- Seed & Device ---
    set_seed(SEED)
    device = get_device()
    use_amp = supports_amp()
    print(f"\n  Device: {device}")

    # --- Reference Genome ---
    print(f"\n{'─'*78}")
    print(f"  1. Loading hg38 Reference Genome")
    print(f"{'─'*78}")
    genome, has_chr = load_hg38()

    # --- Tokenizer ---
    print(f"\n{'─'*78}")
    print(f"  2. Loading Tokenizer")
    print(f"{'─'*78}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Build Model (architecture only — weights loaded per .pth) ---
    print(f"\n{'─'*78}")
    print(f"  3. Building Model Architecture")
    print(f"{'─'*78}")
    model = NTv2DualSeqClassifier(
        model_name=MODEL_NAME,
        num_layers_to_unfreeze=NUM_LAYERS_TO_UNFREEZE,
        dropout=DROPOUT,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model architecture: {total_params:,} parameters")

    # --- Load Test Data (once, shared across all .pth evaluations) ---
    print(f"\n{'─'*78}")
    print(f"  4. Loading Test Datasets")
    print(f"{'─'*78}")

    data_dir = find_data_dir(args.data_dir)
    print(f"   Data dir: {data_dir}")

    test_loaders = {}
    pin_memory = (device.type == 'cuda')

    for i, (key, info) in enumerate(TEST_FILES.items(), 1):
        path = os.path.join(data_dir, info['file'])
        df = pd.read_csv(path)
        if 'LABEL' not in df.columns and 'INT_LABEL' in df.columns:
            df = df.rename(columns={'INT_LABEL': 'LABEL'})

        n_p = int((df['LABEL'] == 1).sum())
        n_b = int((df['LABEL'] == 0).sum())
        print(f"   [{i}/3] {info['name']}: {len(df):,} samples "
              f"(P={n_p:,}, B={n_b:,})")

        ds = DualSeqDataset(
            df, genome, tokenizer, has_chr,
            seq_len=SEQ_LENGTH,
            max_tokens=max_tokens,
            seed=SEED + 100 + i,
        )
        test_loaders[key] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=pin_memory,
        )

    # --- Optionally evaluate dense baseline first ---
    all_results = {}

    if args.dense_weights and os.path.isfile(args.dense_weights):
        print(f"\n{'═'*78}")
        print(f"  DENSE BASELINE: {args.dense_weights}")
        print(f"{'═'*78}")
        dense_res = evaluate_single_pth(
            args.dense_weights, model, test_loaders, device, use_amp)
        all_results['dense_100'] = dense_res
        for key, m in dense_res.items():
            print(f"   Dense → {TEST_FILES[key]['name']:30s}  "
                  f"AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%  "
                  f"F1={m['f1']:.2f}%")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # --- Evaluate each compressed .pth ---
    for idx, (scorer, sp, fp) in enumerate(pth_list, 1):
        label = f"{scorer}@{sp}%"
        result_key = f"{scorer}_{sp:02d}"

        print(f"\n{'═'*78}")
        print(f"  [{idx}/{len(pth_list)}] {label.upper()}: {fp}")
        print(f"{'═'*78}")

        t0 = time.time()
        try:
            res = evaluate_single_pth(fp, model, test_loaders, device, use_amp)
            all_results[result_key] = res

            for key, m in res.items():
                print(f"   {label} → {TEST_FILES[key]['name']:30s}  "
                      f"AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%  "
                      f"F1={m['f1']:.2f}%  MCC={m['mcc']:.4f}")

            elapsed = time.time() - t0
            print(f"   ✅ {label} done in {elapsed:.0f}s")

        except Exception as e:
            print(f"   ❌ {label} FAILED: {e}")
            all_results[result_key] = {'error': str(e)}

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # =====================================================================
    # FINAL COMPARISON MATRIX
    # =====================================================================
    print(f"\n\n{'═'*100}")
    print(f"  FINAL COMPARISON — UNSEEN TEST SET AUROC (%) × SCORER × SPARSITY")
    print(f"{'═'*100}")

    for test_key, test_info in TEST_FILES.items():
        print(f"\n  ── {test_info['name']} ──")
        header = f"  {'Sparsity':>9}"
        for scorer in SCORERS:
            header += f" | {scorer.upper():>12}"
        print(header)
        print(f"  {'─'*9}" + "─┼─".join(["─"*12] * len(SCORERS)))

        for sp in SPARSITIES:
            row = f"  {sp:>8}%"
            for scorer in SCORERS:
                rk = f"{scorer}_{sp:02d}"
                if rk in all_results and test_key in all_results[rk]:
                    v = all_results[rk][test_key].get('auroc', float('nan'))
                    row += f" | {v:>12.4f}"
                else:
                    row += f" | {'N/A':>12}"
            print(row)

    # --- Per-test-set mean AUROC across sparsities ---
    print(f"\n\n{'═'*100}")
    print(f"  MEAN AUROC ACROSS ALL 3 UNSEEN TEST SETS")
    print(f"{'═'*100}")
    header = f"  {'Sparsity':>9}"
    for scorer in SCORERS:
        header += f" | {scorer.upper():>12}"
    print(header)
    print(f"  {'─'*9}" + "─┼─".join(["─"*12] * len(SCORERS)))

    for sp in SPARSITIES:
        row = f"  {sp:>8}%"
        for scorer in SCORERS:
            rk = f"{scorer}_{sp:02d}"
            if rk in all_results:
                aurocs = []
                for tk in TEST_FILES:
                    if tk in all_results[rk]:
                        aurocs.append(all_results[rk][tk].get('auroc', 0))
                if aurocs:
                    mean_a = sum(aurocs) / len(aurocs)
                    row += f" | {mean_a:>12.4f}"
                else:
                    row += f" | {'N/A':>12}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_time_seconds': time.time() - start_time,
        'models_evaluated': len(pth_list),
        'test_sets': list(TEST_FILES.keys()),
        'results': {},
    }

    for rk, res in all_results.items():
        summary['results'][rk] = {}
        for tk in TEST_FILES:
            if tk in res:
                summary['results'][rk][tk] = {
                    k: (float(v) if isinstance(v, (float, int)) else v)
                    for k, v in res[tk].items()
                }

    summary_path = os.path.join(args.output_root, 'unseen_test_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✅ Results saved: {summary_path}")

    # --- Final banner ---
    total_time = time.time() - start_time
    total_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\n{'═'*78}")
    print(f"  🎯 COMPRESSED MODEL UNSEEN TEST EVALUATION COMPLETE")
    print(f"     Models tested: {len(pth_list)}")
    print(f"     Test sets: 3 (07/08/09)")
    print(f"     Time: {total_str}")
    print(f"{'═'*78}\n")


if __name__ == '__main__':
    main()
