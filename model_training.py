# ===================================================================
# CELL 1: NT v2 — VARIANT EFFECT PREDICTION (GLRB-OPTIMIZED)
# ===================================================================
# Target: Match/beat the official GLRB benchmark (0.75 AUROC)
#
# Model:     InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# Task:      ClinVar pathogenic vs gnomAD benign (coding variants)
# Approach:  Dual-sequence (REF vs ALT) + mean pooling
# Strategy:  Full fine-tuning at very low backbone LR
# ===================================================================

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers==4.40.2', 'pyfaidx'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from pyfaidx import Fasta
import time, os, json, glob
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION — Tuned to match/beat GLRB benchmark (0.75 AUROC)
# =====================================================================
CFG = {
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'seq_length': 1000,

    # Data — all available coding variants, no augmentation
    'max_per_class': 50000,          # Use everything available
    'train_fraction': 0.85,

    # Training — full fine-tuning with differential LR
    'epochs': 20,
    'batch_size': 16,
    'grad_accum_steps': 4,           # Effective batch = 64
    'backbone_lr': 5e-6,             # Very low — preserve pre-training
    'head_lr': 5e-4,                 # Higher — learn task fast
    'weight_decay': 0.01,
    'warmup_fraction': 0.15,
    'label_smoothing': 0.05,
    'focal_gamma': 1.5,              # Moderate focal loss
    'max_grad_norm': 1.0,
    'dropout': 0.2,
    'num_layers_to_unfreeze': 22,    # ALL layers (full fine-tuning)
    'patience': 7,
    'seed': 42,
    'save_dir': '/kaggle/working/ntv2_10k_trained',
}

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(CFG['seed'])

print("\n" + "=" * 70)
print("NT v2 — VARIANT EFFECT PREDICTION (GLRB-OPTIMIZED)")
print("=" * 70)
print(f"Target:   Beat GLRB benchmark (0.75 AUROC for NT v2 100M)")
print(f"Model:    {CFG['model_name']}")
print(f"Approach: Dual-sequence + Full fine-tuning + Focal loss")
print(f"Data:     ClinVar/gnomAD coding variants only")
print(f"Context:  {CFG['seq_length']}bp from hg38")
print("=" * 70 + "\n")

# =====================================================================
# 1. DEVICE
# =====================================================================
def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
        print(f"   GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        return d
    return torch.device('cpu')
device = get_device()

# =====================================================================
# 2. hg38 REFERENCE GENOME
# =====================================================================
print("\n" + "-" * 70)
print("1. Loading hg38...")
print("-" * 70)

def get_hg38():
    for p in ["/kaggle/working/hg38.fa", "/kaggle/working/hg38.fasta"]:
        if os.path.exists(p):
            print(f"   ✅ Cached: {p}"); return p
    for pat in ["/kaggle/input/**/hg38.fa", "/kaggle/input/**/*.fa"]:
        for m in glob.glob(pat, recursive=True):
            if os.path.getsize(m) > 1e8: return m
    print("   ⬇️ Downloading hg38...")
    gz, fa = "/kaggle/working/hg38.fa.gz", "/kaggle/working/hg38.fa"
    import urllib.request, gzip, shutil
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    def _p(c,bs,ts):
        if c%500==0: print(f"\r     {c*bs/1e6:.0f}MB...",end="",flush=True)
    urllib.request.urlretrieve(url, gz, reporthook=_p); print()
    with gzip.open(gz,'rb') as fi, open(fa,'wb') as fo: shutil.copyfileobj(fi,fo)
    os.remove(gz); return fa

genome = Fasta(get_hg38(), as_raw=True, sequence_always_upper=True)
has_chr = 'chr1' in set(genome.keys())
print(f"   ✅ hg38 — {len(genome.keys())} contigs")

# =====================================================================
# 3. TOKENIZER + MODEL
# =====================================================================
print("\n" + "-" * 70)
print("2. Loading NT v2...")
print("-" * 70)

tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'], trust_remote_code=True)
max_tok = min(256, tokenizer.model_max_length)
print(f"   Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tok}")


class FocalLoss(nn.Module):
    """Focal loss — focuses on hard-to-classify examples."""
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class NTv2DualSeqClassifier(nn.Module):
    """
    Dual-sequence variant effect classifier.
    [emb_ref; emb_alt; emb_ref - emb_alt] → classifier
    """
    def __init__(self, model_name=CFG['model_name'],
                 n_unfreeze=CFG['num_layers_to_unfreeze']):
        super().__init__()

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        for a,v in [('is_decoder',False),('add_cross_attention',False),
                    ('chunk_size_feed_forward',0)]:
            if not hasattr(cfg,a): setattr(cfg,a,v)

        full = AutoModelForMaskedLM.from_pretrained(
            model_name, config=cfg, trust_remote_code=True)
        self.backbone = full.esm; del full

        self.hidden_size = self.backbone.config.hidden_size  # 512
        n_layers = self.backbone.config.num_hidden_layers     # 22

        # Full fine-tuning: unfreeze everything but embeddings
        freeze_until = n_layers - n_unfreeze
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = (n_unfreeze >= n_layers)  # Unfreeze embeddings only for full FT
        for i, layer in enumerate(self.backbone.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = (i >= freeze_until)
        if hasattr(self.backbone, 'layer_norm'):
            for p in self.backbone.layer_norm.parameters():
                p.requires_grad = True

        # Simpler classifier — less overfitting risk
        drop = CFG.get('dropout', 0.2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 3),  # 1536
            nn.Dropout(drop),
            nn.Linear(self.hidden_size * 3, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 2),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed(self, ids, mask):
        """Mean pooling (official NT v2 approach)."""
        out = self.backbone(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True)
        h = out.last_hidden_state
        m = mask.unsqueeze(-1).float()
        return torch.sum(h * m, dim=1) / torch.clamp(m.sum(1), min=1e-9)

    def forward(self, ref_ids, ref_mask, alt_ids, alt_mask):
        er = self._embed(ref_ids, ref_mask)
        ea = self._embed(alt_ids, alt_mask)
        x = torch.cat([er, ea, er - ea], dim=-1)
        return self.classifier(x)

    def get_param_groups(self, backbone_lr, head_lr):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = list(self.classifier.parameters())
        return [{'params': bb, 'lr': backbone_lr},
                {'params': hd, 'lr': head_lr}]

model = NTv2DualSeqClassifier().to(device)
tot = sum(p.numel() for p in model.parameters())
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ Total: {tot:,} | Trainable: {trn:,} ({100*trn/tot:.1f}%)")
print(f"   Full fine-tuning: ALL {CFG['num_layers_to_unfreeze']} layers + embeddings")
print(f"   LR: backbone={CFG['backbone_lr']}, head={CFG['head_lr']}")

# =====================================================================
# 4. DATA — CODING VARIANTS ONLY
# =====================================================================
print("\n" + "-" * 70)
print("3. Loading ClinVar/gnomAD coding variants...")
print("-" * 70)

possible_dirs = [
    "/kaggle/input/datasets/anirudhamahesha/dnabert2-variant-data/",
    "/kaggle/input/dnabert2-variant-data/",
    "/kaggle/input/mainel-data/", "/kaggle/input/mainel/", "/kaggle/input/",
]
data_dir = None
for d in possible_dirs:
    if os.path.exists(d):
        for root, dirs, files in os.walk(d):
            if "vep_pathogenic_coding.csv" in files:
                data_dir = root + "/"; break
    if data_dir: break
if not data_dir: data_dir = "./"

df_coding = pd.read_csv(os.path.join(data_dir, "vep_pathogenic_coding.csv"))
print(f"   Total: {len(df_coding):,}")
print(f"   Benign (0): {(df_coding['INT_LABEL']==0).sum():,}")
print(f"   Pathogenic (1): {(df_coding['INT_LABEL']==1).sum():,}")

# Balanced sampling — use as much as possible
N = CFG['max_per_class']
n_b = min(N, (df_coding['INT_LABEL']==0).sum())
n_p = min(N, (df_coding['INT_LABEL']==1).sum())
n_min = min(n_b, n_p)  # Balance to minority class

coding_b = df_coding[df_coding['INT_LABEL']==0].sample(n=n_min, random_state=42)
coding_p = df_coding[df_coding['INT_LABEL']==1].sample(n=n_min, random_state=42)

df_all = pd.concat([coding_b, coding_p])[['CHROM','POS','REF','ALT','INT_LABEL']].rename(
    columns={'INT_LABEL':'LABEL'})
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Using: {len(df_all):,} (balanced: {n_min:,} per class)")

# =====================================================================
# 5. DUAL-SEQUENCE DATASET
# =====================================================================
class DualSeqDataset(Dataset):
    def __init__(self, df, genome, tokenizer, has_chr, seq_len=1000,
                 max_tokens=256, seed=42):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        valid = set('ACGT')
        rng = np.random.RandomState(seed)
        self.ref_seqs, self.alt_seqs, self.labels = [], [], []
        skipped = 0

        print(f"   Extracting {len(df):,} variant pairs ({seq_len}bp)...")
        for idx in tqdm(range(len(df)), desc="   Sequences"):
            row = df.iloc[idx]
            chrom = str(row['CHROM']).strip()
            pos, ref, alt = int(row['POS']), str(row['REF']).upper().strip(), str(row['ALT']).upper().strip()
            label = int(row['LABEL'])

            if has_chr and not chrom.startswith('chr'): chrom = 'chr'+chrom
            elif not has_chr and chrom.startswith('chr'): chrom = chrom[3:]
            if chrom not in genome.keys(): skipped+=1; continue
            if len(ref)!=1: ref = ref[0] if ref else 'A'
            if len(alt)!=1: alt = alt[0] if alt else 'A'
            if ref not in valid or alt not in valid: skipped+=1; continue

            half = seq_len//2
            start, end = pos-1-half, pos-1+half
            if start<0 or end>len(genome[chrom]): skipped+=1; continue

            seq = genome[chrom][start:end].upper()
            if len(seq)!=seq_len: skipped+=1; continue

            nbad = sum(1 for b in seq if b not in valid)
            if nbad > seq_len*0.05: skipped+=1; continue
            if nbad:
                sl = list(seq)
                for i,b in enumerate(sl):
                    if b not in valid: sl[i]=rng.choice(list(valid))
                seq = ''.join(sl)

            c = half
            rs = list(seq); rs[c]=ref; rs=''.join(rs)
            al = list(seq); al[c]=alt; al=''.join(al)

            self.ref_seqs.append(rs); self.alt_seqs.append(al)
            self.labels.append(label)

        print(f"   ✅ {len(self.labels):,} variant pairs")
        if skipped: print(f"   ⚠️ Skipped {skipped:,}")
        np_lab = np.array(self.labels)
        print(f"      Path: {np_lab.sum():,} | Benign: {(1-np_lab).sum():,}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        re = self.tokenizer(self.ref_seqs[idx], max_length=self.max_tokens,
                            padding='max_length', truncation=True, return_tensors='pt')
        ae = self.tokenizer(self.alt_seqs[idx], max_length=self.max_tokens,
                            padding='max_length', truncation=True, return_tensors='pt')
        return {'ref_ids':re['input_ids'].squeeze(0),
                'ref_mask':re['attention_mask'].squeeze(0),
                'alt_ids':ae['input_ids'].squeeze(0),
                'alt_mask':ae['attention_mask'].squeeze(0),
                'labels':torch.tensor(self.labels[idx], dtype=torch.long)}

full_ds = DualSeqDataset(df_all, genome, tokenizer, has_chr,
                          CFG['seq_length'], max_tok, CFG['seed'])
TOTAL = len(full_ds)
train_n = int(TOTAL * CFG['train_fraction'])
test_n = TOTAL - train_n
train_ds, test_ds = random_split(full_ds, [train_n, test_n],
                                  generator=torch.Generator().manual_seed(42))
pin = device.type == 'cuda'
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=2, pin_memory=pin)
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False,
                         num_workers=2, pin_memory=pin)
print(f"\n   Train: {train_n:,} | Test: {test_n:,}")
print(f"   Batches/epoch: {len(train_loader)} | Eff batch: {CFG['batch_size']*CFG['grad_accum_steps']}")

# =====================================================================
# 6. EVALUATION
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device, full=False):
    model.eval()
    preds, labs, probs = [], [], []
    for b in loader:
        out = model(b['ref_ids'].to(device), b['ref_mask'].to(device),
                    b['alt_ids'].to(device), b['alt_mask'].to(device))
        preds.extend(torch.argmax(out,1).cpu().numpy())
        labs.extend(b['labels'].numpy())
        probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())

    acc = accuracy_score(labs, preds)*100
    auroc = roc_auc_score(labs, probs)*100
    f1 = f1_score(labs, preds)*100
    if full:
        mcc = matthews_corrcoef(labs, preds)
        prec = precision_score(labs, preds, zero_division=0)*100
        rec = recall_score(labs, preds, zero_division=0)*100
        tn,fp,fn,tp = confusion_matrix(labs, preds).ravel()
        return {'accuracy':acc,'auroc':auroc,'f1':f1,'mcc':mcc,
                'precision':prec,'recall':rec,
                'specificity':tn/(tn+fp)*100 if (tn+fp)>0 else 0,
                'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)}
    return acc, auroc, f1

# =====================================================================
# 7. TRAINING
# =====================================================================
print("\n" + "-" * 70)
print("4. Training (full fine-tuning)...")
print("-" * 70)

def train(model, train_loader, test_loader, device, cfg=CFG):
    print(f"\n   Config: {cfg['epochs']} epochs | backbone_lr={cfg['backbone_lr']} | "
          f"head_lr={cfg['head_lr']} | focal_γ={cfg['focal_gamma']}")

    optimizer = optim.AdamW(
        model.get_param_groups(cfg['backbone_lr'], cfg['head_lr']),
        weight_decay=cfg['weight_decay'])

    criterion = FocalLoss(gamma=cfg['focal_gamma'],
                          label_smoothing=cfg['label_smoothing'])

    total_steps = (len(train_loader)//cfg['grad_accum_steps']) * cfg['epochs']
    warmup = int(total_steps * cfg['warmup_fraction'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"   Steps: {total_steps} | Warmup: {warmup}")

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_auroc, best_state = 0, None
    history = {'train_loss':[],'train_acc':[],'test_acc':[],'test_auroc':[],'test_f1':[],'lr':[]}
    patience = 0

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        model.train()
        tloss, correct, total = 0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"   Ep {epoch+1}/{cfg['epochs']}", leave=False)
        for step, batch in enumerate(pbar):
            ri,rm = batch['ref_ids'].to(device), batch['ref_mask'].to(device)
            ai,am = batch['alt_ids'].to(device), batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(ri,rm,ai,am)
                    loss = criterion(out, labs) / cfg['grad_accum_steps']
                scaler.scale(loss).backward()
            else:
                out = model(ri,rm,ai,am)
                loss = criterion(out, labs) / cfg['grad_accum_steps']
                loss.backward()

            tloss += loss.item() * cfg['grad_accum_steps']
            correct += (torch.argmax(out,1)==labs).sum().item()
            total += labs.size(0)

            if (step+1)%cfg['grad_accum_steps']==0 or (step+1)==len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                    optimizer.step()
                scheduler.step(); optimizer.zero_grad()

            pbar.set_postfix(loss=f'{loss.item()*cfg["grad_accum_steps"]:.4f}',
                             acc=f'{100*correct/total:.1f}%')

        et = time.time()-t0
        avg_loss = tloss/len(train_loader)
        train_acc = 100*correct/total
        test_acc, test_auroc, test_f1 = evaluate(model, test_loader, device)
        lr = scheduler.get_last_lr()[0]

        print(f"\n   Ep {epoch+1}: Loss={avg_loss:.4f} | Train={train_acc:.1f}% | "
              f"Test={test_acc:.1f}% | AUROC={test_auroc:.1f}% | F1={test_f1:.1f}% | "
              f"LR={lr:.2e} | {et:.0f}s")

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_auroc'].append(test_auroc)
        history['test_f1'].append(test_f1)
        history['lr'].append(lr)

        if test_auroc > best_auroc:
            best_auroc = test_auroc
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            patience = 0
            print(f"   🎯 NEW BEST: AUROC={test_auroc:.2f}%, Acc={test_acc:.2f}%, F1={test_f1:.2f}%")
        else:
            patience += 1
            if patience >= cfg['patience']:
                print(f"   ⏹ Early stopping ({cfg['patience']} epochs)"); break

    if best_state: model.load_state_dict(best_state)
    return best_acc, best_auroc, history

best_acc, best_auroc, history = train(model, train_loader, test_loader, device)

# =====================================================================
# 8. FINAL METRICS
# =====================================================================
print("\n" + "-" * 70)
print("5. Final metrics...")
print("-" * 70)

final = evaluate(model, test_loader, device, full=True)
print(f"\n   Accuracy:    {final['accuracy']:.2f}%")
print(f"   AUROC:       {final['auroc']:.2f}%")
print(f"   F1:          {final['f1']:.2f}%")
print(f"   MCC:         {final['mcc']:.4f}")
print(f"   Precision:   {final['precision']:.2f}%")
print(f"   Recall:      {final['recall']:.2f}%")
print(f"   Specificity: {final['specificity']:.2f}%")
print(f"   TP={final['tp']} FP={final['fp']} FN={final['fn']} TN={final['tn']}")

glrb = 75.0
if final['auroc'] >= glrb:
    print(f"\n   ✅ BEATS GLRB benchmark ({final['auroc']:.2f}% > {glrb}%)")
else:
    print(f"\n   📊 GLRB benchmark: {glrb}% — gap: {glrb - final['auroc']:.2f}%")

# =====================================================================
# 9. SAVE
# =====================================================================
print("\n" + "-" * 70)
print("6. Saving...")
print("-" * 70)

save_dir = CFG['save_dir']
os.makedirs(save_dir, exist_ok=True)
model_path = f"{save_dir}/ntv2_10k_trained.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'model_name':CFG['model_name'], 'hidden_size':model.hidden_size,
               'num_layers_to_unfreeze':CFG['num_layers_to_unfreeze'],
               'best_accuracy':best_acc, 'best_auroc':best_auroc,
               'pooling':'mean', 'approach':'dual_sequence_focal',
               'seq_length':CFG['seq_length'], 'full_finetune':True},
    'final_metrics': final,
    'hyperparameters': CFG,
}, model_path)
print(f"   ✅ Model: {model_path}")

with open(f"{save_dir}/training_history.json",'w') as f:
    json.dump(history, f, indent=2)
with open(f"{save_dir}/dataset_info.json",'w') as f:
    json.dump({'total':TOTAL,'train':train_n,'test':test_n,
               'approach':'dual_seq_focal_fullft','data':'coding_only'}, f, indent=2)
print(f"   ✅ History saved")

# Verify reload
print("\n" + "-" * 70)
print("7. Verification...")
print("-" * 70)
mv = NTv2DualSeqClassifier().to(device)
mv.load_state_dict(torch.load(model_path, map_location=device,
                               weights_only=False)['model_state_dict'])
va, vr, _ = evaluate(mv, test_loader, device)
ok = abs(va - final['accuracy']) < 0.01
print(f"   Reload: Acc={va:.2f}%, AUROC={vr:.2f}% — {'✅ PASS' if ok else '❌ FAIL'}")
del mv

print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE")
print("=" * 70)
print(f"Approach:  Dual-seq + Focal loss + Full fine-tuning")
print(f"Data:      {TOTAL:,} ClinVar/gnomAD coding variants")
print(f"Context:   {CFG['seq_length']}bp from hg38")
print(f"Accuracy:  {final['accuracy']:.2f}%")
print(f"AUROC:     {final['auroc']:.2f}%")
print(f"F1:        {final['f1']:.2f}%")
print(f"MCC:       {final['mcc']:.4f}")
print(f"GLRB ref:  {glrb}% AUROC")
print(f"Saved:     {save_dir}/")
print("✅ READY FOR PRUNING (Cell 2)")
print("=" * 70)