# =====================================================================
# data/loader.py — Unified dataset loading, balancing, and splitting
#
# Handles all three datasets through a single interface:
#   - Single-file datasets (ClinVar, dbSNP): load → balance → split
#   - Combined datasets (cBioPortal + gnomAD): load both → merge → balance → split
# =====================================================================

import os
import glob
import pandas as pd
import numpy as np
from .registry import get_dataset_info


def _find_data_dir(filenames: list) -> str:
    """
    Auto-discover the data directory by searching Kaggle input paths
    and the local 'crct dataset' folder.

    Args:
        filenames: List of CSV filenames to search for.

    Returns:
        Path to the directory containing the dataset files.

    Raises:
        FileNotFoundError if no valid directory is found.
    """
    # Priority search paths for Kaggle and local environments
    search_roots = [
        "/kaggle/input/",
        "/kaggle/working/",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "crct dataset"),
        os.path.dirname(os.path.dirname(__file__)),
        "./crct dataset",
        ".",
    ]

    for root in search_roots:
        if not os.path.exists(root):
            continue
        # Walk subdirectories to find the first match
        for dirpath, _, files in os.walk(root):
            if filenames[0] in files:
                # Verify ALL files exist in this directory
                if all(f in files for f in filenames):
                    return dirpath

    # Last resort: glob search in Kaggle input
    for pattern in ["/kaggle/input/**/" + filenames[0]]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    raise FileNotFoundError(
        f"Could not find dataset files {filenames} in any known location. "
        f"Searched: {search_roots}"
    )


def _load_single(data_dir: str, info: dict, max_per_class: int,
                  seed: int) -> tuple:
    """
    Load a single-file dataset (ClinVar or dbSNP).
    Both classes exist in the same file.

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    filepath = os.path.join(data_dir, info['files'][0])
    df = pd.read_csv(filepath)
    print(f"   Loaded: {info['files'][0]} — {len(df):,} rows")
    print(f"   Benign (0):     {(df[info['label_column']] == 0).sum():,}")
    print(f"   Pathogenic (1): {(df[info['label_column']] == 1).sum():,}")

    return _balance_and_split(df, info['label_column'], max_per_class, seed)


def _load_combined(data_dir: str, info: dict, max_per_class: int,
                   seed: int) -> tuple:
    """
    Load and combine two single-class files (cBioPortal + gnomAD).
    cBioPortal provides pathogenic variants, gnomAD provides benign.

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    path_file = info['files'][info['pathogenic_file_idx']]
    benign_file = info['files'][info['benign_file_idx']]

    df_path = pd.read_csv(os.path.join(data_dir, path_file))
    df_benign = pd.read_csv(os.path.join(data_dir, benign_file))

    print(f"   Loaded: {path_file} — {len(df_path):,} pathogenic")
    print(f"   Loaded: {benign_file} — {len(df_benign):,} benign")

    # Standardize columns before concat — keep only what we need + SPLIT
    keep_cols = ['CHROM', 'POS', 'REF', 'ALT', info['label_column'], 'SPLIT']
    df_path = df_path[keep_cols]
    df_benign = df_benign[keep_cols]

    df = pd.concat([df_path, df_benign], ignore_index=True)
    print(f"   Combined: {len(df):,} total")
    print(f"   Benign (0):     {(df[info['label_column']] == 0).sum():,}")
    print(f"   Pathogenic (1): {(df[info['label_column']] == 1).sum():,}")

    return _balance_and_split(df, info['label_column'], max_per_class, seed)


def _balance_and_split(df: pd.DataFrame, label_col: str,
                       max_per_class: int, seed: int) -> tuple:
    """
    Balance class counts and split into train/test DataFrames.

    Uses the pre-existing SPLIT column from the CSV if available,
    otherwise falls back to 85/15 random split.

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    n_benign = (df[label_col] == 0).sum()
    n_pathogenic = (df[label_col] == 1).sum()

    # Balance to minority class, capped at max_per_class
    n_per_class = min(n_benign, n_pathogenic, max_per_class)

    df_b = df[df[label_col] == 0].sample(n=n_per_class, random_state=seed)
    df_p = df[df[label_col] == 1].sample(n=n_per_class, random_state=seed)

    balanced = pd.concat([df_b, df_p])
    print(f"\n   Balanced: {len(balanced):,} ({n_per_class:,} per class)")

    # Split using pre-defined SPLIT column
    if 'SPLIT' in balanced.columns:
        train_df = balanced[balanced['SPLIT'] == 'train']
        test_df = balanced[balanced['SPLIT'] == 'test']

        # If the pre-defined test set is too small after balancing,
        # carve out 15% from train for a meaningful evaluation
        if len(test_df) < 0.05 * len(balanced):
            print("   ⚠️ Pre-defined test set too small after balancing, "
                  "using 85/15 random split instead")
            balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
            split_idx = int(len(balanced) * 0.85)
            train_df = balanced.iloc[:split_idx]
            test_df = balanced.iloc[split_idx:]
        else:
            # Shuffle within splits
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(balanced) * 0.85)
        train_df = balanced.iloc[:split_idx]
        test_df = balanced.iloc[split_idx:]

    # Standardize output columns
    rename_map = {label_col: 'LABEL'}
    train_out = train_df[['CHROM', 'POS', 'REF', 'ALT', label_col]].rename(columns=rename_map)
    test_out = test_df[['CHROM', 'POS', 'REF', 'ALT', label_col]].rename(columns=rename_map)

    train_out = train_out.reset_index(drop=True)
    test_out = test_out.reset_index(drop=True)

    print(f"   Train: {len(train_out):,} | Test: {len(test_out):,}")
    print(f"   Train — Benign: {(train_out['LABEL']==0).sum():,}, "
          f"Pathogenic: {(train_out['LABEL']==1).sum():,}")
    print(f"   Test  — Benign: {(test_out['LABEL']==0).sum():,}, "
          f"Pathogenic: {(test_out['LABEL']==1).sum():,}")

    return train_out, test_out


def load_dataset(dataset_name: str, max_per_class: int = 50_000,
                 seed: int = 42) -> tuple:
    """
    Main entry point: load any registered dataset by name.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal'
        max_per_class: Maximum samples per class (for memory management)
        seed: Random seed for reproducibility

    Returns:
        (train_df, test_df) — each with columns [CHROM, POS, REF, ALT, LABEL]
    """
    info = get_dataset_info(dataset_name)
    print(f"\n   Dataset: {info['description']}")
    print(f"   Strategy: {info['strategy']}")

    data_dir = _find_data_dir(info['files'])
    print(f"   Data dir: {data_dir}")

    if info['strategy'] == 'single':
        return _load_single(data_dir, info, max_per_class, seed)
    elif info['strategy'] == 'combine':
        return _load_combined(data_dir, info, max_per_class, seed)
    else:
        raise ValueError(f"Unknown loading strategy: {info['strategy']}")
