# =====================================================================
# engine/evaluator.py — Model evaluation with comprehensive metrics
#
# Supports both quick evaluation (acc, auroc, f1) during training and
# full evaluation (all metrics + confusion matrix) for final reporting.
# =====================================================================

import torch
from torch.amp import autocast
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
)


@torch.no_grad()
def evaluate(model, loader, device, full: bool = False,
             use_amp: bool = True):
    """
    Evaluate the model on a DataLoader.

    Args:
        model: NTv2DualSeqClassifier
        loader: DataLoader with DualSeqDataset
        device: torch.device
        full: If True, return comprehensive metrics dict.
              If False, return (accuracy, auroc, f1) tuple.

    Returns:
        If full=False: (accuracy%, auroc%, f1%)
        If full=True:  dict with all metrics + confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        ref_ids = batch['ref_ids'].to(device)
        ref_mask = batch['ref_mask'].to(device)
        alt_ids = batch['alt_ids'].to(device)
        alt_mask = batch['alt_mask'].to(device)

        if use_amp:
            with autocast('cuda'):
                logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
        else:
            logits = model(ref_ids, ref_mask, alt_ids, alt_mask)

        all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
        all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds) * 100
    auroc = roc_auc_score(all_labels, all_probs) * 100
    f1 = f1_score(all_labels, all_preds) * 100

    if not full:
        return acc, auroc, f1

    # Full metrics
    mcc = matthews_corrcoef(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, zero_division=0) * 100
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    return {
        'accuracy': acc,
        'auroc': auroc,
        'f1': f1,
        'mcc': mcc,
        'precision': prec,
        'recall': rec,
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }
