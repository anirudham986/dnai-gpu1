#!/usr/bin/env python3
"""
================================================================================
SCRIPT 1 OF 6  —  ClinVar  |  75,000 variants  |  Pathogenic + Benign
================================================================================
SOURCE   : ClinVar GRCh38 VCF  (NCBI FTP — free, no auth)
URL      : https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
SIZE     : ~200 MB compressed
LABELS   : Pathogenic/Likely_pathogenic  →  INT_LABEL = 1   (~96k available)
           Benign/Likely_benign          →  INT_LABEL = 0   (~651k available)
TARGET   : 75,000 total  (37,500 P  +  37,500 B)

STAR TIERS (quality of clinical evidence)
  4★  practice_guideline                                     ~13 variants
  3★  reviewed_by_expert_panel                              ~3,700
  2★  criteria_provided,_multiple_submitters,_no_conflicts  ~180,000
  1★  criteria_provided,_single_submitter                   ~560,000

HOW TO RUN
  pip install pandas tqdm
  python 01_clinvar_75k.py
  OUTDIR=/custom/path python 01_clinvar_75k.py

OUTPUT COLUMNS
  CHROM | POS | REF | ALT | LABEL | SOURCE | CONSEQUENCE | INT_LABEL
  CLINVAR_ID | CLINVAR_SIG | REVIEW_STARS | REVIEW_STATUS | SPLIT
================================================================================
"""

import os, gzip, logging, warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CFG = {
    "outdir":   os.environ.get("OUTDIR", "./variant_data"),
    "outfile":  "01_clinvar_75k.csv",
    "target":   75_000,
    "min_stars": 1,
    "test_chroms": {"chr8"},
    "seed": 42,
}
os.makedirs(CFG["outdir"], exist_ok=True)

VALID_BASES  = set("ACGT")
VALID_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
PATH_SIGS    = frozenset({"Pathogenic","Likely_pathogenic","Pathogenic/Likely_pathogenic",
                           "Pathogenic,_low_penetrance","Likely_pathogenic,_low_penetrance",
                           "Pathogenic,_other","Pathogenic,_risk_factor"})
BEN_SIGS     = frozenset({"Benign","Likely_benign","Benign/Likely_benign"})
STAR_MAP     = {"practice_guideline":4,
                "reviewed_by_expert_panel":3,
                "criteria_provided,_multiple_submitters,_no_conflicts":2,
                "criteria_provided,_single_submitter":1,
                "criteria_provided,_conflicting_interpretations":1}
CODING_P = frozenset({"missense_variant","stop_gained","stop_lost","start_lost",
                       "splice_donor_variant","splice_acceptor_variant",
                       "splice_region_variant","frameshift_variant",
                       "inframe_insertion","inframe_deletion"})
CODING_B = frozenset({"missense_variant","synonymous_variant",
                       "stop_retained_variant","splice_region_variant"})

def chrom_norm(c):
    c = str(c).strip(); return c if c.startswith("chr") else "chr"+c

def is_snv(r, a):
    return len(r)==1 and len(a)==1 and r in VALID_BASES and a in VALID_BASES and r!=a

def parse_info(s):
    d={}
    for tok in s.split(";"):
        if "=" in tok: k,v=tok.split("=",1); d[k]=v
        else: d[tok]=True
    return d

def clnsig(raw):
    sigs={s.strip() for s in raw.replace("/",",").split(",")}
    if sigs & PATH_SIGS: return "Pathogenic",1
    if sigs & BEN_SIGS:  return "Benign",0
    return None,None

def stars(rs):
    rs=rs.lower()
    for p,s in STAR_MAP.items():
        if p in rs: return s
    return 0

def parse_mc(mc):
    if not mc or mc==".": return None
    f=mc.split(",")[0]; return f.split("|",1)[1] if "|" in f else f

def download():
    import urllib.request
    url ="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    dest=os.path.join(CFG["outdir"],"clinvar_grch38.vcf.gz")
    if os.path.exists(dest) and os.path.getsize(dest)>10_000_000:
        log.info(f"  Cached: {dest}"); return dest
    log.info("  Downloading ClinVar GRCh38 VCF (~200 MB)...")
    def p(c,bs,ts):
        if c%200==0: print(f"\r    {c*bs/1e6:.0f} MB...",end="",flush=True)
    urllib.request.urlretrieve(url,dest,reporthook=p); print()
    return dest

def parse(vcf_gz):
    log.info("  Parsing ClinVar VCF...")
    P,B=[],[]
    sk=dict(snv=0,sig=0,star=0,csq=0)
    with gzip.open(vcf_gz,"rt",errors="replace") as fh:
        for line in tqdm(fh,desc="  ClinVar",unit="L",mininterval=5):
            if line.startswith("#"): continue
            cols=line.rstrip("\n").split("\t")
            if len(cols)<8: continue
            chrom,pos,vid,ref,alt=cols[0],cols[1],cols[2],cols[3],cols[4]
            if ","in alt or not is_snv(ref,alt): sk["snv"]+=1; continue
            chrom=chrom_norm(chrom)
            if chrom not in VALID_CHROMS: continue
            info=parse_info(cols[7])
            if info.get("CLNVC","")!="single_nucleotide_variant": sk["snv"]+=1; continue
            ls,il=clnsig(info.get("CLNSIG",""))
            if ls is None: sk["sig"]+=1; continue
            rs=info.get("CLNREVSTAT",""); st=stars(rs)
            if st<CFG["min_stars"]: sk["star"]+=1; continue
            mc=parse_mc(info.get("MC",""))
            if mc is None: sk["csq"]+=1; continue
            allow=CODING_P if il==1 else CODING_B
            if mc not in allow: sk["csq"]+=1; continue
            rec=dict(CHROM=chrom,POS=int(pos),REF=ref.upper(),ALT=alt.upper(),
                     LABEL=ls,SOURCE=f"ClinVar_{st}star",CONSEQUENCE=mc,INT_LABEL=il,
                     CLINVAR_ID=vid if vid!="." else None,
                     CLINVAR_SIG=info.get("CLNSIG",""),
                     REVIEW_STARS=st,REVIEW_STATUS=rs)
            (P if il==1 else B).append(rec)
    log.info(f"  Raw — P:{len(P):,}  B:{len(B):,}  "
             f"skipped snv={sk['snv']:,} sig={sk['sig']:,} "
             f"star={sk['star']:,} csq={sk['csq']:,}")
    return pd.DataFrame(P),pd.DataFrame(B)

def balance(dfp,dfb):
    n=CFG["target"]//2
    # Pathogenic: prefer higher stars
    if len(dfp)>n:
        dfp=dfp.sort_values("REVIEW_STARS",ascending=False).head(n*4)\
               .sample(n=n,random_state=CFG["seed"],weights="REVIEW_STARS")
    # Benign: stratified by star tier
    if len(dfb)>n:
        parts=[]
        for st,cnt in dfb["REVIEW_STARS"].value_counts().items():
            take=min(cnt,max(1,int(n*cnt/len(dfb))))
            parts.append(dfb[dfb.REVIEW_STARS==st].sample(n=take,random_state=CFG["seed"]))
        dfb=pd.concat(parts)
        if len(dfb)>n: dfb=dfb.sample(n=n,random_state=CFG["seed"])
        elif len(dfb)<n:
            shortfall=n-len(dfb); used=set(dfb.index)
            extra=(dfb_all:=dfb)[~dfb_all.index.isin(used)]  # no-op, already subset
            # just trim to what we have — still valid
    log.info(f"  Selected — P:{len(dfp):,}  B:{len(dfb):,}")
    df=pd.concat([dfp,dfb],ignore_index=True)\
       .sample(frac=1,random_state=CFG["seed"]).reset_index(drop=True)
    df["SPLIT"]=df["CHROM"].apply(lambda c:"test" if c in CFG["test_chroms"] else "train")
    return df

def validate(df):
    log.info("  Validation:")
    for name,ok in {
        "No nulls":             df[["CHROM","POS","REF","ALT"]].notna().all().all(),
        "INT_LABEL in {0,1}":  df["INT_LABEL"].isin([0,1]).all(),
        "All SNVs":            ((df.REF.str.len()==1)&(df.ALT.str.len()==1)).all(),
        "No dup positions":    not df.duplicated(["CHROM","POS","REF","ALT"]).any(),
        "Both classes":        df["INT_LABEL"].nunique()==2,
        "Split correct":       df["SPLIT"].isin(["train","test"]).all(),
        "≥ 70k rows":          len(df)>=70_000,
    }.items():
        log.info(f"    [{'PASS' if ok else 'FAIL'}] {name}")

def main():
    log.info("="*60)
    log.info("Script 1/6 — ClinVar 75k  |  Pathogenic + Benign")
    log.info("="*60)
    vcf=download()
    dfp,dfb=parse(vcf)
    df=balance(dfp,dfb)
    validate(df)
    cols=["CHROM","POS","REF","ALT","LABEL","SOURCE","CONSEQUENCE","INT_LABEL",
          "CLINVAR_ID","CLINVAR_SIG","REVIEW_STARS","REVIEW_STATUS","SPLIT"]
    df=df[[c for c in cols if c in df.columns]]
    log.info(f"\n  Total: {len(df):,}  P={(df.INT_LABEL==1).sum():,}  "
             f"B={(df.INT_LABEL==0).sum():,}")
    log.info(f"  Train: {(df.SPLIT=='train').sum():,}  "
             f"Test(chr8): {(df.SPLIT=='test').sum():,}")
    log.info("\n  Star-tier breakdown:")
    for src,cnt in df.SOURCE.value_counts().items():
        p=len(df[(df.SOURCE==src)&(df.INT_LABEL==1)])
        b=len(df[(df.SOURCE==src)&(df.INT_LABEL==0)])
        log.info(f"    {src:<28} {cnt:>7,}  P={p:>6,}  B={b:>6,}")
    out=os.path.join(CFG["outdir"],CFG["outfile"])
    df.to_csv(out,index=False)
    log.info(f"\n  Saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

if __name__=="__main__":
    main()
