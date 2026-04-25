#!/usr/bin/env python3
"""
================================================================================
SCRIPT 5 OF 6  —  TCGA via GDC API  |  75,000 variants  |  Both P and B
================================================================================
SOURCE   : TCGA Masked Somatic Mutation MAF files
           via NCI GDC REST API  (free, no auth for open-access MAFs)
API      : https://api.gdc.cancer.gov/
LABELS   : Pathogenic (INT_LABEL=1) — somatic missense / nonsense / splice
           Benign     (INT_LABEL=0) — silent/synonymous somatic mutations
                                      (same MAF files, within-dataset benign)
TARGET   : 75,000 total  (37,500 P  +  37,500 B)
           EQUAL SAMPLING PER CANCER TYPE — each of 33 TCGA cancer types
           contributes equally so no single cancer dominates the dataset.

WHY SILENT MUTATIONS AS BENIGN
--------------------------------
TCGA contains somatic mutations only — there is no matched germline benign
cohort in the open-access MAFs. The standard approach used in the cancer
genomics community is to treat silent (synonymous) somatic mutations as the
within-dataset benign class. Silent mutations:
  1. Do not change amino acid sequence
  2. Are under neutral selection in cancer (no fitness effect)
  3. Are found in equal proportion across tumour types
  4. Arise from the same mutational processes as missense mutations
  5. Have the same sequencing context and coverage as pathogenic mutations

This is the labelling scheme used in the COSMIC, OncoKB, and CADD cancer
variant benchmarks.

CANCER TYPES COVERED (all 33 TCGA projects)
  LUAD  Lung adenocarcinoma           LUSC  Lung squamous cell carcinoma
  BRCA  Breast invasive carcinoma     PRAD  Prostate adenocarcinoma
  COAD  Colon adenocarcinoma          READ  Rectal adenocarcinoma
  GBM   Glioblastoma multiforme       LGG   Brain lower grade glioma
  SKCM  Skin cutaneous melanoma       BLCA  Bladder urothelial carcinoma
  KIRC  Kidney renal clear cell       KIRP  Kidney renal papillary cell
  HNSC  Head and neck squamous cell   OV    Ovarian serous cystadenocarcinoma
  STAD  Stomach adenocarcinoma        UCEC  Uterine corpus endometrial carc.
  LIHC  Liver hepatocellular carc.    THCA  Thyroid carcinoma
  LAML  Acute myeloid leukaemia       CESC  Cervical squamous cell carcinoma
  SARC  Sarcoma                       ESCA  Oesophageal carcinoma
  PCPG  Phaeo- & paraganglioma        PAAD  Pancreatic adenocarcinoma
  TGCT  Testicular germ cell tumour   THYM  Thymoma
  DLBC  Diffuse large B-cell lymphoma MESO  Mesothelioma
  UVM   Uveal melanoma                ACC   Adrenocortical carcinoma
  UCS   Uterine carcinosarcoma        KICH  Kidney chromophobe
  CHOL  Cholangiocarcinoma

GDC API STRATEGY
----------------
1. Query /files endpoint for each TCGA project's Masked Somatic Mutation MAF
2. Stream each MAF (gzip TSV) via /data endpoint
3. Collect missense/nonsense/splice → pathogenic; synonymous → benign
4. Cap each cancer type at per_cancer_cap to ensure equal representation
5. Checkpoint after each cancer type

HOW TO RUN
  pip install pandas tqdm requests
  python 05_tcga_75k.py
  OUTDIR=/path python 05_tcga_75k.py

OUTPUT COLUMNS
  CHROM | POS | REF | ALT | LABEL | SOURCE | CONSEQUENCE | INT_LABEL
  CANCER_TYPE | TCGA_SAMPLE | GENE | HGVSc | HGVSp | T_ALT_COUNT
  T_REF_COUNT | TUMOR_DEPTH | SPLIT
================================================================================
"""

import os, io, gzip, json, time, logging, warnings
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CFG = {
    "outdir":        os.environ.get("OUTDIR","./variant_data"),
    "outfile":       "05_tcga_75k.csv",
    "target_total":  75_000,
    "test_chroms":   {"chr8"},
    "seed":          42,
    "gdc_api":       "https://api.gdc.cancer.gov",
    "sleep":         0.5,    # seconds between GDC API requests
    "max_retry":     5,
}
os.makedirs(CFG["outdir"], exist_ok=True)
CHECKPOINT = os.path.join(CFG["outdir"],"tcga_checkpoint.csv")

VALID_BASES  = set("ACGT")
VALID_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX","chrY"])

# All 33 TCGA projects
TCGA_PROJECTS = [
    "TCGA-LUAD","TCGA-LUSC","TCGA-BRCA","TCGA-PRAD","TCGA-COAD",
    "TCGA-READ","TCGA-GBM", "TCGA-LGG", "TCGA-SKCM","TCGA-BLCA",
    "TCGA-KIRC","TCGA-KIRP","TCGA-HNSC","TCGA-OV",  "TCGA-STAD",
    "TCGA-UCEC","TCGA-LIHC","TCGA-THCA","TCGA-LAML","TCGA-CESC",
    "TCGA-SARC","TCGA-ESCA","TCGA-PCPG","TCGA-PAAD","TCGA-TGCT",
    "TCGA-THYM","TCGA-DLBC","TCGA-MESO","TCGA-UVM", "TCGA-ACC",
    "TCGA-UCS", "TCGA-KICH","TCGA-CHOL",
]

# Short cancer-type label for readable SOURCE column
PROJECT_TO_CANCER = {p: p.replace("TCGA-","") for p in TCGA_PROJECTS}

# Pathogenic variant classifications in GDC/TCGA MAF files
PATHOGENIC_CLASSES = frozenset({
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
    "In_Frame_Del", "In_Frame_Ins",
    "Frame_Shift_Del", "Frame_Shift_Ins",
})
BENIGN_CLASSES = frozenset({
    "Silent",   # synonymous — the within-TCGA benign class
})

# SO term mapping for the CONSEQUENCE column
CLASS_TO_SO = {
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

def chrom_norm(c):
    c=str(c).strip(); return c if c.startswith("chr") else "chr"+c

def is_snv(r,a):
    return len(r)==1 and len(a)==1 and r in VALID_BASES and a in VALID_BASES and r!=a

def gdc_get(endpoint, params=None, retries=CFG["max_retry"]):
    """GET request to GDC API with retry."""
    url = f"{CFG['gdc_api']}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2**attempt); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt)
    return {}

def gdc_stream(file_id, retries=CFG["max_retry"]):
    """Stream a GDC file by UUID."""
    url = f"{CFG['gdc_api']}/data/{file_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=300)
            if r.status_code == 429:
                time.sleep(2**attempt); continue
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt)

def get_maf_file_id(project_id):
    """
    Query GDC /files endpoint to get the UUID of the Masked Somatic Mutation
    MAF for a given TCGA project.
    """
    params = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op":"=","content":{"field":"cases.project.project_id",
                                      "value":project_id}},
                {"op":"=","content":{"field":"data_type",
                                      "value":"Masked Somatic Mutation"}},
                {"op":"=","content":{"field":"data_format","value":"MAF"}},
                {"op":"=","content":{"field":"access","value":"open"}},
            ]
        }),
        "fields": "file_id,file_name,data_type",
        "size":   "10",
        "format": "json",
    }
    data = gdc_get("files", params=params)
    hits = data.get("data",{}).get("hits",[])
    if not hits:
        log.warning(f"  No MAF file found for {project_id}")
        return None
    # Prefer the file with 'somatic.maf' in the name
    for h in hits:
        if "somatic" in h.get("file_name","").lower():
            return h["file_id"]
    return hits[0]["file_id"]

def parse_maf_stream(response, project_id, cap_per_class):
    """
    Parse a streaming GDC MAF (gzipped TSV).
    Returns (path_records, benign_records).
    """
    cancer = PROJECT_TO_CANCER[project_id]
    P, B = [], []

    # Collect raw bytes
    raw = b"".join(response.iter_content(1 << 20))

    try:
        content = gzip.decompress(raw).decode("utf-8", errors="replace")
    except Exception:
        # Not gzip — try plain text
        content = raw.decode("utf-8", errors="replace")

    lines = content.split("\n")
    header = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if header is None:
            header = line.split("\t")
            continue

        row = dict(zip(header, line.split("\t")))

        chrom = chrom_norm(row.get("Chromosome",""))
        if chrom not in VALID_CHROMS: continue

        ref = row.get("Reference_Allele","").upper()
        alt = row.get("Tumor_Seq_Allele2","").upper()

        # SNV filter
        if not is_snv(ref, alt): continue

        try:
            pos = int(row.get("Start_Position",0))
        except ValueError:
            continue

        var_class = row.get("Variant_Classification","")

        if var_class in PATHOGENIC_CLASSES and len(P) < cap_per_class:
            csq = CLASS_TO_SO.get(var_class, "missense_variant")
            P.append(dict(
                CHROM=chrom, POS=pos, REF=ref, ALT=alt,
                LABEL="Pathogenic",
                SOURCE=f"TCGA_{cancer}",
                CONSEQUENCE=csq,
                INT_LABEL=1,
                CANCER_TYPE=cancer,
                TCGA_SAMPLE=row.get("Tumor_Sample_Barcode",""),
                GENE=row.get("Hugo_Symbol",""),
                HGVSc=row.get("HGVSc",""),
                HGVSp=row.get("HGVSp_Short",""),
                T_ALT_COUNT=row.get("t_alt_count",""),
                T_REF_COUNT=row.get("t_ref_count",""),
                TUMOR_DEPTH=row.get("t_depth",""),
            ))

        elif var_class in BENIGN_CLASSES and len(B) < cap_per_class:
            B.append(dict(
                CHROM=chrom, POS=pos, REF=ref, ALT=alt,
                LABEL="Benign",
                SOURCE=f"TCGA_{cancer}",
                CONSEQUENCE="synonymous_variant",
                INT_LABEL=0,
                CANCER_TYPE=cancer,
                TCGA_SAMPLE=row.get("Tumor_Sample_Barcode",""),
                GENE=row.get("Hugo_Symbol",""),
                HGVSc=row.get("HGVSc",""),
                HGVSp="",
                T_ALT_COUNT=row.get("t_alt_count",""),
                T_REF_COUNT=row.get("t_ref_count",""),
                TUMOR_DEPTH=row.get("t_depth",""),
            ))

        if len(P) >= cap_per_class and len(B) >= cap_per_class:
            break

    log.info(f"    {cancer}: P={len(P):,}  B={len(B):,}")
    return P, B

def fetch_all():
    # Resume from checkpoint
    if os.path.exists(CHECKPOINT):
        df_ck = pd.read_csv(CHECKPOINT)
        all_P = df_ck[df_ck.INT_LABEL==1].to_dict("records")
        all_B = df_ck[df_ck.INT_LABEL==0].to_dict("records")
        done_cancers = set(df_ck["CANCER_TYPE"].unique())
        log.info(f"  Checkpoint: P={len(all_P):,}  B={len(all_B):,}  "
                 f"done={done_cancers}")
    else:
        all_P, all_B, done_cancers = [], [], set()

    target = CFG["target_total"] // 2
    n_projects = len(TCGA_PROJECTS)
    # Equal cap per cancer type
    per_cancer = max(50, (target - len(all_P)) // max(1, n_projects - len(done_cancers)))
    log.info(f"  Per-cancer cap: {per_cancer:,} per class per cancer type")

    for project in TCGA_PROJECTS:
        cancer = PROJECT_TO_CANCER[project]
        if cancer in done_cancers:
            log.info(f"  Skipping {cancer} (already done)")
            continue
        if len(all_P) >= target and len(all_B) >= target:
            log.info("  Both classes at target — stopping")
            break

        log.info(f"\n  [{cancer}] Fetching MAF...")
        try:
            file_id = get_maf_file_id(project)
            if not file_id:
                continue
            log.info(f"    File UUID: {file_id}")
            response = gdc_stream(file_id)
            P, B = parse_maf_stream(response, project, per_cancer)
            all_P.extend(P)
            all_B.extend(B)
            done_cancers.add(cancer)
        except Exception as e:
            log.warning(f"  {cancer} failed: {e}")
            continue

        log.info(f"  Running: P={len(all_P):,}  B={len(all_B):,}")
        # Checkpoint
        pd.DataFrame(all_P + all_B).to_csv(CHECKPOINT, index=False)
        time.sleep(CFG["sleep"])

    return pd.DataFrame(all_P), pd.DataFrame(all_B)

def finalise(dfp, dfb):
    if dfp.empty or dfb.empty:
        log.error("One or both classes are empty!")
        return pd.DataFrame()

    dfp = dfp.drop_duplicates(["CHROM","POS","REF","ALT"]).reset_index(drop=True)
    dfb = dfb.drop_duplicates(["CHROM","POS","REF","ALT"]).reset_index(drop=True)

    target = CFG["target_total"] // 2
    n = min(target, len(dfp), len(dfb))
    log.info(f"  Balancing: {n:,} per class from "
             f"P={len(dfp):,} B={len(dfb):,} available")

    # Sample equally per cancer type within each class
    def equal_cancer_sample(df, n_total):
        n_cancers = df["CANCER_TYPE"].nunique()
        per_c = max(1, n_total // n_cancers)
        parts = []
        for ct in df["CANCER_TYPE"].unique():
            sub = df[df.CANCER_TYPE==ct]
            take = min(len(sub), per_c)
            parts.append(sub.sample(n=take, random_state=CFG["seed"]))
        out = pd.concat(parts)
        # Trim/fill to n_total
        if len(out) > n_total:
            out = out.sample(n=n_total, random_state=CFG["seed"])
        return out

    dfp = equal_cancer_sample(dfp, n)
    dfb = equal_cancer_sample(dfb, n)
    log.info(f"  After equal-cancer sampling: P={len(dfp):,}  B={len(dfb):,}")

    df = pd.concat([dfp, dfb], ignore_index=True)\
         .sample(frac=1, random_state=CFG["seed"]).reset_index(drop=True)
    df["SPLIT"] = df["CHROM"].apply(
        lambda c: "test" if c in CFG["test_chroms"] else "train")
    return df

def validate(df):
    log.info("  Validation:")
    for name,ok in {
        "No nulls":           df[["CHROM","POS","REF","ALT"]].notna().all().all(),
        "INT_LABEL in {0,1}": df.INT_LABEL.isin([0,1]).all(),
        "All SNVs":           ((df.REF.str.len()==1)&(df.ALT.str.len()==1)).all(),
        "No dup positions":   not df.duplicated(["CHROM","POS","REF","ALT"]).any(),
        "Both classes":       df.INT_LABEL.nunique()==2,
        "Multiple cancers":   df.CANCER_TYPE.nunique()>5,
        "Split correct":      df.SPLIT.isin(["train","test"]).all(),
    }.items():
        log.info(f"    [{'PASS' if ok else 'FAIL'}] {name}")

def main():
    log.info("="*60)
    log.info("Script 5/6 — TCGA via GDC API  |  75k  |  Pathogenic + Benign")
    log.info("Pathogenic: somatic missense/nonsense/splice mutations")
    log.info("Benign:     somatic synonymous (silent) mutations")
    log.info("Balanced equally across all 33 TCGA cancer types")
    log.info("="*60)

    dfp, dfb = fetch_all()
    df = finalise(dfp, dfb)
    if df.empty: return
    validate(df)

    cols = ["CHROM","POS","REF","ALT","LABEL","SOURCE","CONSEQUENCE","INT_LABEL",
            "CANCER_TYPE","TCGA_SAMPLE","GENE","HGVSc","HGVSp",
            "T_ALT_COUNT","T_REF_COUNT","TUMOR_DEPTH","SPLIT"]
    df = df[[c for c in cols if c in df.columns]]

    log.info(f"\n  Total: {len(df):,}  "
             f"P={(df.INT_LABEL==1).sum():,}  B={(df.INT_LABEL==0).sum():,}")
    log.info(f"  Train: {(df.SPLIT=='train').sum():,}  "
             f"Test(chr8): {(df.SPLIT=='test').sum():,}")
    log.info(f"\n  Cancer-type distribution (P per cancer):")
    for ct, cnt in (df[df.INT_LABEL==1].CANCER_TYPE.value_counts()).items():
        b_cnt = len(df[(df.INT_LABEL==0)&(df.CANCER_TYPE==ct)])
        log.info(f"    {ct:<8} P={cnt:>5,}  B={b_cnt:>5,}")

    out = os.path.join(CFG["outdir"], CFG["outfile"])
    df.to_csv(out, index=False)
    log.info(f"\n  Saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()