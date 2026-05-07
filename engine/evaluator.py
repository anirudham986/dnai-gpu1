# =====================================================================
# engine/evaluator.py — Model evaluation with comprehensive metrics
#
# Always returns a full metrics dictionary. The caller decides what
# to print based on verbosity settings.
# =====================================================================

import torch
from torch.amp import autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = True,
             desc: str = "Evaluating") -> dict:
    """
    Evaluate the model on a DataLoader.

    Always returns a comprehensive metrics dictionary including:
    accuracy, auroc, f1, mcc, precision, recall, specificity,
    and confusion matrix components (tp, fp, tn, fn).

    Args:
        model:   NTv2DualSeqClassifier
        loader:  DataLoader with DualSeqDataset
        device:  torch.device
        use_amp: Whether to use mixed precision (GPU only)
        desc:    Description for tqdm progress bar

    Returns:
        dict with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"   {desc}", leave=False)
    for batch in pbar:
        ref_ids = batch['ref_ids'].to(device)
        ref_mask = batch['ref_mask'].to(device)
        alt_ids = batch['alt_ids'].to(device)
        alt_mask = batch['alt_mask'].to(device)

        if use_amp:
            with autocast('cuda'):
                logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
        else:
            logits = model(ref_ids, ref_mask, alt_ids, alt_mask)

        probs = torch.softmax(logits, 1)
        preds = torch.argmax(logits, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds) * 100
    auroc = roc_auc_score(all_labels, all_probs) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    mcc = matthews_corrcoef(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, zero_division=0) * 100
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0

    return {
        'accuracy': acc,
        'auroc': auroc,
        'f1': f1,
        'mcc': mcc,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'n_samples': len(all_labels),
    }


def print_metrics(metrics: dict, prefix: str = "   ", compact: bool = False):
    """
    Pretty-print a metrics dictionary.

    Args:
        metrics: Dict returned by evaluate()
        prefix:  Indentation prefix
        compact: If True, print single line; if False, full breakdown
    """
    if compact:
        print(f"{prefix}Acc={metrics['accuracy']:.2f}% | "
              f"AUROC={metrics['auroc']:.2f}% | "
              f"F1={metrics['f1']:.2f}% | "
              f"MCC={metrics['mcc']:.4f}")
    else:
        print(f"{prefix}Accuracy:    {metrics['accuracy']:.2f}%")
        print(f"{prefix}AUROC:       {metrics['auroc']:.2f}%")
        print(f"{prefix}F1:          {metrics['f1']:.2f}%")
        print(f"{prefix}MCC:         {metrics['mcc']:.4f}")
        print(f"{prefix}Precision:   {metrics['precision']:.2f}%")
        print(f"{prefix}Recall:      {metrics['recall']:.2f}%")
        print(f"{prefix}Specificity: {metrics['specificity']:.2f}%")
        print(f"{prefix}Confusion:   TP={metrics['tp']}  FP={metrics['fp']}  "
              f"FN={metrics['fn']}  TN={metrics['tn']}")
        print(f"{prefix}Samples:     {metrics['n_samples']:,}")
