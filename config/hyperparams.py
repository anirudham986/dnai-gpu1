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
    'val_fold': 0,                     # Validation fold for kfold strategy

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
    'patience': 4,                     # Early stopping patience (epochs)
    'seed': 42,

    # Checkpoint behaviour
    'checkpoint_interval_hours': 8.5,  # Save timed checkpoint every 8.5 hrs
    'save_every_n_epochs': 2,          # Also save every N epochs

    # Paths (Kaggle default, overridable via CLI)
    'save_dir': '/kaggle/working/ntv2_trained',
    'num_workers': 2,
}

# Human-readable dataset choices for the interactive prompt
DATASET_CHOICES = {
    '1': 'clinvar',
    '2': 'dbsnp',
    '3': 'cbioportal',
    '4': 'consolidated',
    '5': 'consolidated_full',
    'clinvar':            'clinvar',
    'dbsnp':              'dbsnp',
    'cbioportal':         'cbioportal',
    'consolidated':       'consolidated',
    'consolidated_full':  'consolidated_full',
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
        'max_per_class': 37_498,
    },
    'dbsnp': {
        # dbSNP 62k — balanced 31.4k/class
        # NOTE: dbSNP_common (benign) vs dbSNP_ClinVar (pathogenic) — results
        # may be high because these are source-discriminable. For genuine
        # pathogenicity learning, use the 'consolidated' dataset.
        'max_per_class': 31_398,
    },
    'cbioportal': {
        # cBioPortal 63k pathogenic + gnomAD 55k benign → combined
        'max_per_class': 50_000,
    },
    'consolidated': {
        # All 4 datasets merged, deduplicated, 5-fold stratified
        # This is the primary dataset for all final model evaluation
        'max_per_class': 50_000,     # Per-class cap (handled by build script)
        'val_fold': 0,               # Default validation fold (override with --fold)
        'n_folds': 5,
        'patience': 3,               # Tight early stopping for Kaggle commit sessions
        'checkpoint_interval_hours': 8.5,
        'save_every_n_epochs': 2,
    },
    'consolidated_full': {
        # Full 100k training + 25k unseen holdout test
        # Used AFTER K-fold validation to produce the final compression-ready model
        'max_per_class': 50_000,
        'patience': 4,               # Slightly more patience for full training
        'checkpoint_interval_hours': 8.5,
        'save_every_n_epochs': 2,
    },
}


def get_config(dataset_name: str, val_fold: int = None) -> dict:
    """
    Build the full configuration for a given dataset by merging
    the base config with dataset-specific overrides.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal', 'consolidated',
                      'consolidated_full'
        val_fold:     Override validation fold (0–4) for 'consolidated' only

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
    cfg['save_dir']     = f"/kaggle/working/ntv2_{dataset_name}_trained"

    # Apply val_fold override
    if val_fold is not None:
        cfg['val_fold'] = val_fold

    return cfg
