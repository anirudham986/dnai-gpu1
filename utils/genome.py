# =====================================================================
# utils/genome.py — hg38 reference genome loader
#
# Auto-discovers hg38.fa from Kaggle input directories or downloads
# it from UCSC if not found. Uses pyfaidx for efficient random-access
# sequence retrieval (memory-mapped, no full genome in RAM).
# =====================================================================

import os
import glob
from pyfaidx import Fasta


def load_hg38() -> tuple:
    """
    Load the hg38 reference genome.

    Search order:
        1. /kaggle/working/hg38.fa or hg38.fasta (cached from prior run)
        2. /kaggle/input/**/*.fa (Kaggle dataset input)
        3. Download from UCSC if nothing found

    Returns:
        (genome, has_chr) — pyfaidx.Fasta object and whether contigs
                            use 'chr' prefix (True for hg38)
    """
    fa_path = _find_hg38()
    genome = Fasta(fa_path, as_raw=True, sequence_always_upper=True)
    has_chr = 'chr1' in set(genome.keys())
    print(f"   ✅ hg38 loaded — {len(genome.keys())} contigs")
    return genome, has_chr


def _find_hg38() -> str:
    """Locate or download hg38.fa."""
    # Check cached locations
    for path in ["/kaggle/working/hg38.fa", "/kaggle/working/hg38.fasta"]:
        if os.path.exists(path):
            print(f"   ✅ Cached: {path}")
            return path

    # Search Kaggle input directories
    for pattern in ["/kaggle/input/**/hg38.fa", "/kaggle/input/**/*.fa"]:
        for match in glob.glob(pattern, recursive=True):
            if os.path.getsize(match) > 1e8:  # > 100MB = likely real genome
                print(f"   ✅ Found: {match}")
                return match

    # Download from UCSC
    print("   ⬇️ Downloading hg38 from UCSC...")
    return _download_hg38()


def _download_hg38() -> str:
    """Download hg38.fa.gz from UCSC and decompress."""
    import urllib.request
    import gzip
    import shutil

    gz_path = "/kaggle/working/hg38.fa.gz"
    fa_path = "/kaggle/working/hg38.fa"
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

    def _progress(count, block_size, total_size):
        if count % 500 == 0:
            mb = count * block_size / 1e6
            print(f"\r     {mb:.0f}MB...", end="", flush=True)

    urllib.request.urlretrieve(url, gz_path, reporthook=_progress)
    print()

    print("   Decompressing...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(fa_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print(f"   ✅ Saved: {fa_path}")
    return fa_path
