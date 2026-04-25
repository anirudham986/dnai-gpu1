#!/usr/bin/env python3
"""
================================================================================
SCRIPT 6 OF 6  —  cBioPortal REST API  |  75,000 variants  |  Both P and B
================================================================================
SOURCE   : cBioPortal public instance  (free, no auth for public studies)
API      : https://www.cbioportal.org/api
LABELS   : Pathogenic (INT_LABEL=1) — somatic missense / nonsense / splice
           Benign     (INT_LABEL=0) — silent/synonymous somatic mutations
TARGET   : 75,000 total  (37,500 P  +  37,500 B)
           EQUAL SAMPLING PER CANCER STUDY so no cancer type dominates.

WHY cBIOPORTAL IS DIFFERENT FROM TCGA
---------------------------------------
cBioPortal hosts data from BOTH TCGA and many additional studies:
  - MSK-IMPACT (Memorial Sloan Kettering — >100k patients)
  - GENIE (AACR — multi-institution)
  - Individual published cohort studies
  - Paediatric cancer studies
  - Rare cancer types not in TCGA
This gives access to cancer types and mutations NOT in TCGA.

WHY SYNONYMOUS = BENIGN (same rationale as Script 5)
------------------------------------------------------
cBioPortal stores somatic mutations only. Silent (synonymous) mutations:
  1. Do not change amino acid sequence
  2. Are under neutral selection — no cancer fitness effect
  3. Arise from the same mutational processes (same sequencing context)
  4. Are used as the within-dataset benign class in published benchmarks
     (COSMIC, OncoKB, CADD cancer panels all use this convention)

CANCER STUDIES USED (mix of TCGA + non-TCGA for maximum diversity)
  TCGA pan-cancer studies  (via cBioPortal's aggregated endpoints)
  MSK-IMPACT 2017          (10,000+ patients, many cancer types)
  GENIE 13.0               (pan-cancer, multi-institution)
  BRCA (METABRIC)          (breast cancer, published cohort)
  Lung (TRACERx)           (lung, longitudinal)
  Prostate (SU2C)          (metastatic prostate)
  Colorectal (DFCI)        (colorectal)
  Glioma (GLASS)           (brain tumour, longitudinal)
  Melanoma (DFCI)          (skin)
  Pancreatic (QCMG)        (pancreatic)
  AML (TCGA + Beat AML)    (leukaemia)
  Bladder (DFCI/BGI)       (bladder)

API RATE LIMIT: 5 requests/second (public instance)
Script enforces 0.25s sleep between requests automatically.

HOW TO RUN
  pip install pandas tqdm requests
  python 06_cbioportal_75k.py
  OUTDIR=/path python 06_cbioportal_75k.py

OUTPUT COLUMNS
  CHROM | POS | REF | ALT | LABEL | SOURCE | CONSEQUENCE | INT_LABEL
  CANCER_TYPE | STUDY_ID | GENE | PROTEIN_CHANGE | MUTATION_TYPE
  SAMPLE_ID | SPLIT
================================================================================
"""

import os, json, time, logging, warnings
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CFG = {
    "outdir":       os.environ.get("OUTDIR", "./variant_data"),
    "outfile":      "06_cbioportal_75k.csv",
    "target_total": 75_000,
    "test_chroms":  {"chr8"},
    "seed":         42,
    "api_base":     "https://www.cbioportal.org/api",
    "sleep":        0.25,    # 4 req/s — safely under 5 req/s limit
    "max_retry":    5,
    "page_size":    10_000,  # mutations per API page
}
os.makedirs(CFG["outdir"], exist_ok=True)
CHECKPOINT = os.path.join(CFG["outdir"], "cbioportal_checkpoint.csv")

VALID_BASES  = set("ACGT")
VALID_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])

# ── Studies to query — diverse mix of cancer types ────────────────────────────
# Format: (study_id, cancer_label, description)
# study_id must match cBioPortal's studyId field exactly.
STUDIES = [
    # TCGA studies (via cBioPortal) — broad cancer coverage
    ("luad_tcga",        "LUAD",       "Lung adenocarcinoma (TCGA)"),
    ("lusc_tcga",        "LUSC",       "Lung squamous cell (TCGA)"),
    ("brca_tcga",        "BRCA",       "Breast invasive carcinoma (TCGA)"),
    ("prad_tcga",        "PRAD",       "Prostate adenocarcinoma (TCGA)"),
    ("coadread_tcga",    "COADREAD",   "Colorectal adenocarcinoma (TCGA)"),
    ("gbm_tcga",         "GBM",        "Glioblastoma (TCGA)"),
    ("lgg_tcga",         "LGG",        "Brain lower grade glioma (TCGA)"),
    ("skcm_tcga",        "SKCM",       "Skin cutaneous melanoma (TCGA)"),
    ("blca_tcga",        "BLCA",       "Bladder urothelial carcinoma (TCGA)"),
    ("kirc_tcga",        "KIRC",       "Kidney clear cell (TCGA)"),
    ("hnsc_tcga",        "HNSC",       "Head and neck squamous (TCGA)"),
    ("ov_tcga",          "OV",         "Ovarian serous (TCGA)"),
    ("stad_tcga",        "STAD",       "Stomach adenocarcinoma (TCGA)"),
    ("ucec_tcga",        "UCEC",       "Uterine endometrial (TCGA)"),
    ("lihc_tcga",        "LIHC",       "Liver hepatocellular (TCGA)"),
    ("paad_tcga",        "PAAD",       "Pancreatic adenocarcinoma (TCGA)"),
    ("thca_tcga",        "THCA",       "Thyroid carcinoma (TCGA)"),
    ("laml_tcga",        "LAML",       "Acute myeloid leukaemia (TCGA)"),
    ("sarc_tcga",        "SARC",       "Sarcoma (TCGA)"),
    ("esca_tcga",        "ESCA",       "Oesophageal carcinoma (TCGA)"),
    ("meso_tcga",        "MESO",       "Mesothelioma (TCGA)"),
    ("uvm_tcga",         "UVM",        "Uveal melanoma (TCGA)"),
    ("acc_tcga",         "ACC",        "Adrenocortical carcinoma (TCGA)"),
    ("pcpg_tcga",        "PCPG",       "Phaeochromocytoma (TCGA)"),
    ("tgct_tcga",        "TGCT",       "Testicular germ cell (TCGA)"),
    ("dlbc_tcga",        "DLBC",       "Diffuse large B-cell lymphoma (TCGA)"),
    ("chol_tcga",        "CHOL",       "Cholangiocarcinoma (TCGA)"),
    # Non-TCGA studies — additional cancer types and larger cohorts
    ("msk_impact_2017",  "MSK_PAN",    "MSK-IMPACT pan-cancer 2017"),
    ("brca_metabric",    "BRCA_MB",    "Breast cancer METABRIC"),
    ("prad_su2c_2019",   "PRAD_SU2C",  "Metastatic prostate SU2C 2019"),
    ("crc_dfci_2016",    "CRC_DFCI",   "Colorectal DFCI 2016"),
    ("mel_dfci_2019",    "MEL_DFCI",   "Melanoma DFCI 2019"),
    ("paad_qcmg_2016",   "PAAD_QCMG",  "Pancreatic QCMG 2016"),
    ("aml_ohsu_2022",    "AML_OHSU",   "AML Beat AML 2022"),
    ("blca_bgi",         "BLCA_BGI",   "Bladder BGI"),
    ("glioma_glass_2019","GLIOMA",     "Glioma GLASS longitudinal 2019"),
]

# Mutation type → INT_LABEL mapping
PATHOGENIC_TYPES = frozenset({
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
    "In_Frame_Del", "In_Frame_Ins",
    "Frame_Shift_Del", "Frame_Shift_Ins",
})
BENIGN_TYPES = frozenset({"Silent"})

MUT_TO_SO = {
    "Missense_Mutation":      "missense_variant",
    "Nonsense_Mutation":      "stop_gained",
    "Splice_Site":            "splice_region_variant",
    "Translation_Start_Site": "start_lost",
    "Nonstop_Mutation":       "stop_lost",
    "In_Frame_Del":           "inframe_deletion",
    "In_Frame_Ins":           "inframe_insertion",
    "Frame_Shift_Del":        "frameshift_variant",
    "Frame_Shift_Ins":        "frameshift_variant",
    "Silent":                 "synonymous_variant",
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def chrom_norm(c):
    c = str(c).strip()
    if not c: return None
    return c if c.startswith("chr") else "chr" + c

def is_snv(r, a):
    return (len(r) == 1 and len(a) == 1
            and r in VALID_BASES and a in VALID_BASES and r != a)

def api_get(endpoint, params=None):
    """GET request to cBioPortal API with retry and rate limiting."""
    url = f"{CFG['api_base']}/{endpoint}"
    for attempt in range(CFG["max_retry"]):
        try:
            r = requests.get(url, params=params, timeout=90,
                             headers={"Accept": "application/json"})
            if r.status_code == 429:
                wait = 2 ** attempt
                log.info(f"\n    Rate limited — waiting {wait}s...")
                time.sleep(wait); continue
            if r.status_code == 404:
                return None   # Study not found — skip silently
            r.raise_for_status()
            time.sleep(CFG["sleep"])
            return r.json()
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == CFG["max_retry"] - 1:
                log.warning(f"    API error {endpoint}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

def api_post(endpoint, payload):
    """POST request to cBioPortal API."""
    url = f"{CFG['api_base']}/{endpoint}"
    for attempt in range(CFG["max_retry"]):
        try:
            r = requests.post(url, json=payload, timeout=120,
                              headers={"Accept": "application/json",
                                       "Content-Type": "application/json"})
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(CFG["sleep"])
            return r.json()
        except Exception as e:
            if attempt == CFG["max_retry"] - 1:
                log.warning(f"    POST error {endpoint}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

# ── Verify study exists ───────────────────────────────────────────────────────
def verify_study(study_id):
    """Return True if study exists and has mutation data."""
    data = api_get(f"studies/{study_id}")
    if not data:
        return False
    # Check for mutation molecular profile
    profiles = api_get(f"studies/{study_id}/molecular-profiles")
    if not profiles:
        return False
    has_mut = any("mutation" in p.get("molecularProfileId","").lower()
                  for p in profiles)
    return has_mut

# ── Get mutation molecular profile ID ────────────────────────────────────────
def get_mutation_profile(study_id):
    profiles = api_get(f"studies/{study_id}/molecular-profiles")
    if not profiles:
        return None
    # Prefer masked somatic or just mutations profile
    for p in profiles:
        pid = p.get("molecularProfileId","")
        if "mutation" in pid.lower():
            return pid
    return None

# ── Get all sample IDs for a study ───────────────────────────────────────────
def get_sample_list(study_id):
    """Return list of all sample IDs in the study."""
    data = api_get(f"studies/{study_id}/samples",
                   params={"pageSize": 50_000, "pageNumber": 0})
    if not data:
        return []
    return [s["sampleId"] for s in data]

# ── Fetch mutations for a study ───────────────────────────────────────────────
def fetch_study_mutations(study_id, cancer_label, cap_per_class):
    """
    Fetch mutations for a single cBioPortal study.
    Uses POST /mutations/fetch for efficiency.
    Returns (path_records, benign_records).
    """
    # Verify study exists
    if not verify_study(study_id):
        log.info(f"    Study {study_id} not found or no mutations — skipping")
        return [], []

    mut_profile = get_mutation_profile(study_id)
    if not mut_profile:
        log.info(f"    No mutation profile for {study_id} — skipping")
        return [], []

    samples = get_sample_list(study_id)
    if not samples:
        log.info(f"    No samples for {study_id} — skipping")
        return [], []

    log.info(f"    Profile: {mut_profile}  |  Samples: {len(samples):,}")

    P, B = [], []
    page = 0
    page_size = CFG["page_size"]

    while len(P) < cap_per_class or len(B) < cap_per_class:
        payload = {
            "sampleIds": samples,
            "molecularProfileId": mut_profile,
        }
        # Use paginated fetch
        params = {
            "pageSize":   page_size,
            "pageNumber": page,
            "projection": "DETAILED",
        }
        # POST to fetch mutations
        endpoint = f"mutations/fetch"
        data = api_post(f"{endpoint}?pageSize={page_size}&pageNumber={page}",
                        payload)

        if not data:
            break
        if len(data) == 0:
            break   # No more pages

        for mut in data:
            # Extract chromosome + position
            chrom = chrom_norm(str(mut.get("chr") or
                                   mut.get("chromosome") or ""))
            if not chrom or chrom not in VALID_CHROMS:
                continue

            ref = str(mut.get("referenceAllele","")).upper().strip()
            alt = str(mut.get("variantAllele","") or
                      mut.get("tumorSeqAllele2","")).upper().strip()

            if not is_snv(ref, alt):
                continue

            try:
                pos = int(mut.get("startPosition",0) or
                          mut.get("start",0))
            except (ValueError, TypeError):
                continue

            if pos <= 0:
                continue

            mut_type = mut.get("mutationType","")

            gene = ""
            if "gene" in mut and mut["gene"]:
                gene = mut["gene"].get("hugoGeneSymbol","")
            elif "hugoGeneSymbol" in mut:
                gene = mut.get("hugoGeneSymbol","")

            prot = mut.get("proteinChange","") or mut.get("aminoAcidChange","")

            base_rec = dict(
                CHROM=chrom, POS=pos, REF=ref, ALT=alt,
                CANCER_TYPE=cancer_label,
                STUDY_ID=study_id,
                GENE=gene,
                PROTEIN_CHANGE=prot,
                MUTATION_TYPE=mut_type,
                SAMPLE_ID=mut.get("sampleId",""),
            )

            if mut_type in PATHOGENIC_TYPES and len(P) < cap_per_class:
                P.append({**base_rec,
                          "LABEL":       "Pathogenic",
                          "SOURCE":      f"cBioPortal_{cancer_label}",
                          "CONSEQUENCE": MUT_TO_SO.get(mut_type,"missense_variant"),
                          "INT_LABEL":   1})

            elif mut_type in BENIGN_TYPES and len(B) < cap_per_class:
                B.append({**base_rec,
                          "LABEL":       "Benign",
                          "SOURCE":      f"cBioPortal_{cancer_label}",
                          "CONSEQUENCE": "synonymous_variant",
                          "INT_LABEL":   0})

            if len(P) >= cap_per_class and len(B) >= cap_per_class:
                break

        if len(data) < page_size:
            break   # Last page
        page += 1

    log.info(f"    {cancer_label}: P={len(P):,}  B={len(B):,}")
    return P, B

# ── Main fetch loop ───────────────────────────────────────────────────────────
def fetch_all():
    # Resume from checkpoint
    if os.path.exists(CHECKPOINT):
        df_ck = pd.read_csv(CHECKPOINT)
        all_P = df_ck[df_ck.INT_LABEL==1].to_dict("records")
        all_B = df_ck[df_ck.INT_LABEL==0].to_dict("records")
        done  = set(df_ck["STUDY_ID"].unique())
        log.info(f"  Resuming: P={len(all_P):,}  B={len(all_B):,}  "
                 f"done studies={len(done)}")
    else:
        all_P, all_B, done = [], [], set()

    target     = CFG["target_total"] // 2
    n_studies  = len(STUDIES)
    # Equal allocation per study
    remaining_studies = [s for s in STUDIES if s[0] not in done]
    per_study  = max(50, target // max(1, n_studies))
    log.info(f"  Per-study cap: {per_study:,} per class  |  "
             f"Studies remaining: {len(remaining_studies)}")

    for study_id, cancer_label, desc in remaining_studies:
        if len(all_P) >= target and len(all_B) >= target:
            log.info("  Both classes at target — stopping"); break

        log.info(f"\n  [{cancer_label}] {desc}")
        try:
            P, B = fetch_study_mutations(study_id, cancer_label, per_study)
            all_P.extend(P)
            all_B.extend(B)
            done.add(study_id)
        except Exception as e:
            log.warning(f"  {cancer_label} failed: {e}")
            continue

        log.info(f"  Running: P={len(all_P):,}  B={len(all_B):,}")
        # Save checkpoint
        pd.DataFrame(all_P + all_B).to_csv(CHECKPOINT, index=False)

    return pd.DataFrame(all_P), pd.DataFrame(all_B)

# ── Finalise ──────────────────────────────────────────────────────────────────
def finalise(dfp, dfb):
    if dfp.empty or dfb.empty:
        log.error("One or both classes empty — check API connectivity")
        return pd.DataFrame()

    dfp = dfp.drop_duplicates(["CHROM","POS","REF","ALT"]).reset_index(drop=True)
    dfb = dfb.drop_duplicates(["CHROM","POS","REF","ALT"]).reset_index(drop=True)

    target = CFG["target_total"] // 2
    n = min(target, len(dfp), len(dfb))
    log.info(f"  Balancing: {n:,} per class  "
             f"(P avail={len(dfp):,}  B avail={len(dfb):,})")

    def equal_study_sample(df, n_total):
        """Sample equally per study/cancer type."""
        studies = df["STUDY_ID"].unique()
        per_s = max(1, n_total // len(studies))
        parts = []
        for sid in studies:
            sub = df[df.STUDY_ID == sid]
            take = min(len(sub), per_s)
            parts.append(sub.sample(n=take, random_state=CFG["seed"]))
        out = pd.concat(parts, ignore_index=True)
        if len(out) > n_total:
            out = out.sample(n=n_total, random_state=CFG["seed"])
        return out

    dfp = equal_study_sample(dfp, n)
    dfb = equal_study_sample(dfb, n)
    log.info(f"  After equal-study sampling: P={len(dfp):,}  B={len(dfb):,}")

    df = pd.concat([dfp, dfb], ignore_index=True)\
         .sample(frac=1, random_state=CFG["seed"]).reset_index(drop=True)

    # Remove conflicts (same position labelled both P and B)
    key = ["CHROM","POS","REF","ALT"]
    lc  = df.groupby(key)["INT_LABEL"].nunique()
    conf = lc[lc > 1].reset_index()[key]
    if len(conf):
        conf["_c"] = True
        df = df.merge(conf, on=key, how="left")
        n_conf = df["_c"].sum()
        df = df[df["_c"].isna()].drop(columns=["_c"]).reset_index(drop=True)
        log.info(f"  Removed {n_conf:,} conflict positions")

    df["SPLIT"] = df["CHROM"].apply(
        lambda c: "test" if c in CFG["test_chroms"] else "train")
    return df

# ── Validate ──────────────────────────────────────────────────────────────────
def validate(df):
    log.info("  Validation:")
    for name, ok in {
        "No nulls":             df[["CHROM","POS","REF","ALT"]].notna().all().all(),
        "INT_LABEL in {0,1}":  df.INT_LABEL.isin([0,1]).all(),
        "All SNVs":            ((df.REF.str.len()==1)&(df.ALT.str.len()==1)).all(),
        "No dup positions":    not df.duplicated(["CHROM","POS","REF","ALT"]).any(),
        "Both classes":        df.INT_LABEL.nunique() == 2,
        "Multiple cancers":    df.CANCER_TYPE.nunique() > 5,
        "Split correct":       df.SPLIT.isin(["train","test"]).all(),
    }.items():
        log.info(f"    [{'PASS' if ok else 'FAIL'}] {name}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("="*60)
    log.info("Script 6/6 — cBioPortal API  |  75k  |  Pathogenic + Benign")
    log.info("Pathogenic: somatic missense / nonsense / splice")
    log.info("Benign:     somatic synonymous (silent) mutations")
    log.info("Balanced equally across all studies / cancer types")
    log.info("="*60)

    # Quick connectivity check
    log.info("\n  Checking cBioPortal API connectivity...")
    info = api_get("info")
    if info:
        log.info(f"  API version: {info.get('portalVersion','unknown')}")
    else:
        log.warning("  API check failed — will attempt anyway")

    dfp, dfb = fetch_all()
    df = finalise(dfp, dfb)
    if df.empty:
        log.error("No data collected. Check API connectivity and study IDs.")
        return

    validate(df)

    cols = ["CHROM","POS","REF","ALT","LABEL","SOURCE","CONSEQUENCE","INT_LABEL",
            "CANCER_TYPE","STUDY_ID","GENE","PROTEIN_CHANGE","MUTATION_TYPE",
            "SAMPLE_ID","SPLIT"]
    df = df[[c for c in cols if c in df.columns]]

    log.info(f"\n  Total: {len(df):,}  "
             f"P={(df.INT_LABEL==1).sum():,}  B={(df.INT_LABEL==0).sum():,}")
    log.info(f"  Train: {(df.SPLIT=='train').sum():,}  "
             f"Test(chr8): {(df.SPLIT=='test').sum():,}")
    log.info(f"  Cancer types: {df.CANCER_TYPE.nunique()}")
    log.info(f"\n  Per-cancer breakdown:")
    for ct in sorted(df.CANCER_TYPE.unique()):
        p = len(df[(df.CANCER_TYPE==ct)&(df.INT_LABEL==1)])
        b = len(df[(df.CANCER_TYPE==ct)&(df.INT_LABEL==0)])
        log.info(f"    {ct:<15} P={p:>5,}  B={b:>5,}")
    log.info(f"\n  Consequence breakdown:")
    for csq, cnt in df.CONSEQUENCE.value_counts().items():
        log.info(f"    {csq:<40} {cnt:>7,}")

    out = os.path.join(CFG["outdir"], CFG["outfile"])
    df.to_csv(out, index=False)
    log.info(f"\n  Saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    log.info("="*60)

if __name__ == "__main__":
    main()