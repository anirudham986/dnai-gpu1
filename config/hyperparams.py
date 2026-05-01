# =====================================================================
# config/hyperparams.py — Central configuration for NTv2 multi-dataset
# training pipeline. All hyperparameters and per-dataset overrides live
# here so nothing is hard-coded elsewhere.
# =====================================================================

from copy import deepcopy

# ---------------------------------------------------------------------
# Base configuration (proven on ClinVar+gnomAD → 84% AUROC, 76% Acc)
# These defaults are shared across all datasets and only overridden
# where a dataset has specific needs.
# ---------------------------------------------------------------------
BASE_CFG = {
    # Model
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'seq_length': 1000,               # bp context window around variant

    # Data
    'max_per_class': 50_000,           # Cap per class for memory
    'use_predefined_split': True,      # Use SPLIT column from CSV

    # Training — full fine-tuning with differential LR
    'epochs': 20,
    'batch_size': 16,
    'grad_accum_steps': 4,             # Effective batch = 64
    'backbone_lr': 5e-6,              # Very low — preserve pre-training
    'head_lr': 5e-4,                  # Higher — learn task fast
    'weight_decay': 0.01,
    'warmup_fraction': 0.15,
    'label_smoothing': 0.05,
    'focal_gamma': 1.5,               # Moderate focal loss
    'max_grad_norm': 1.0,
    'dropout': 0.2,
    'num_layers_to_unfreeze': 22,      # ALL layers (full fine-tuning)
    'patience': 3,
    'seed': 42,

    # Paths (Kaggle default, overridable)
    'save_dir': '/kaggle/working/ntv2_trained',
    'num_workers': 2,
}

# Human-readable dataset choices for the interactive prompt
DATASET_CHOICES = {
    '1': 'clinvar',
    '2': 'dbsnp',
    '3': 'cbioportal',
    'clinvar': 'clinvar',
    'dbsnp': 'dbsnp',
    'cbioportal': 'cbioportal',
}

# ---------------------------------------------------------------------
# Per-dataset config overrides
# Each entry patches BASE_CFG for dataset-specific needs.
# Empty dict = use all base defaults.
# ---------------------------------------------------------------------
DATASET_OVERRIDES = {
    'clinvar': {
        # ClinVar 75k — already balanced ~37.5k/class
        # Well-curated clinical significance → model learns cleanly
        'max_per_class': 37_500,
    },
    'dbsnp': {
        # dbSNP 62k — already balanced ~31.4k/class
        # Mix of dbSNP_common (benign) and dbSNP_ClinVar (pathogenic)
        'max_per_class': 31_398,
    },
    'cbioportal': {
        # cBioPortal 63k pathogenic + gnomAD 55k benign → combined
        # Larger dataset, balance to gnomAD minority (55k)
        'max_per_class': 50_000,
    },
}


def get_config(dataset_name: str) -> dict:
    """
    Build the full configuration for a given dataset by merging
    the base config with dataset-specific overrides.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal'

    Returns:
        Complete configuration dictionary.
    """
    if dataset_name not in DATASET_OVERRIDES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Must be one of: {list(DATASET_OVERRIDES.keys())}"
        )

    cfg = deepcopy(BASE_CFG)
    cfg.update(DATASET_OVERRIDES[dataset_name])
    cfg['dataset_name'] = dataset_name
    cfg['save_dir'] = f"/kaggle/working/ntv2_{dataset_name}_trained"
    return cfg
