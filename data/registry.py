# =====================================================================
# data/registry.py — Dataset registry
#
# Maps each dataset name to its CSV file(s), loading strategy, and
# description. This is the single source of truth for what datasets
# exist and how they are composed.
# =====================================================================

DATASET_REGISTRY = {
    'clinvar': {
        'description': 'ClinVar 75k — Pathogenic + Benign (clinical significance)',
        'files': ['01_clinvar_75k_P+B.csv'],
        'strategy': 'single',      # Single file, already has both classes
        'label_column': 'INT_LABEL',
        'has_both_classes': True,
    },
    'dbsnp': {
        'description': 'dbSNP 62k — Pathogenic + Benign (common variants + ClinVar)',
        'files': ['03_dbsnp_62k_P+B.csv'],
        'strategy': 'single',
        'label_column': 'INT_LABEL',
        'has_both_classes': True,
    },
    'cbioportal': {
        'description': 'cBioPortal 63k (Pathogenic) + gnomAD 55k (Benign) — combined',
        'files': ['04_cbioportal_63k_P.csv', '02_gnomad_55k_B.csv'],
        'strategy': 'combine',     # Two files: first=pathogenic, second=benign
        'label_column': 'INT_LABEL',
        'has_both_classes': False,  # Each file has one class; combined = both
        'pathogenic_file_idx': 0,   # Index of the pathogenic file in 'files'
        'benign_file_idx': 1,       # Index of the benign file in 'files'
    },
}


def get_dataset_info(dataset_name: str) -> dict:
    """
    Retrieve the registry entry for a dataset.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal'

    Returns:
        Registry dictionary with file paths, strategy, etc.

    Raises:
        ValueError if dataset_name is not in the registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]
