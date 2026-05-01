# =====================================================================
# engine/trainer.py — Training loop with mixed precision, gradient
# accumulation, cosine LR schedule, and early stopping.
#
# Preserves the exact training strategy that achieved 84% AUROC:
#   - AdamW with differential LR (backbone 5e-6, head 5e-4)
#   - Focal loss with label smoothing
#   - Cosine schedule with warmup
#   - Mixed precision (AMP) for GPU efficiency
#   - Early stopping on test AUROC
# =====================================================================

import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from copy import deepcopy

from .evaluator import evaluate


def train(model, train_loader, test_loader, device, criterion, cfg: dict,
          use_amp: bool = True):
    """
    Full training loop for the NTv2 dual-sequence classifier.

    Args:
        model: NTv2DualSeqClassifier (already on device)
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        device: torch.device
        criterion: Loss function (FocalLoss)
        cfg: Configuration dictionary from config/hyperparams.py

    Returns:
        (best_accuracy, best_auroc, history_dict)
    """
    print(f"\n   Config: {cfg['epochs']} epochs | backbone_lr={cfg['backbone_lr']} | "
          f"head_lr={cfg['head_lr']} | focal_γ={cfg['focal_gamma']}")

    # Optimizer with differential learning rates
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['backbone_lr'], cfg['head_lr']),
        weight_decay=cfg['weight_decay']
    )

    # Cosine schedule with warmup
    total_steps = (len(train_loader) // cfg['grad_accum_steps']) * cfg['epochs']
    warmup_steps = int(total_steps * cfg['warmup_fraction'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    print(f"   Steps: {total_steps} | Warmup: {warmup_steps}")

    # Mixed precision — only if GPU supports it (sm_70+)
    scaler = GradScaler('cuda') if use_amp else None

    # Tracking
    best_auroc = 0.0
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_acc': [], 'test_auroc': [], 'test_f1': [],
        'lr': [],
    }

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"   Ep {epoch + 1}/{cfg['epochs']}",
            leave=False
        )

        for step, batch in enumerate(pbar):
            ref_ids = batch['ref_ids'].to(device)
            ref_mask = batch['ref_mask'].to(device)
            alt_ids = batch['alt_ids'].to(device)
            alt_mask = batch['alt_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            if use_amp:
                with autocast('cuda'):
                    logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                    loss = criterion(logits, labels) / cfg['grad_accum_steps']
                scaler.scale(loss).backward()
            else:
                logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                loss = criterion(logits, labels) / cfg['grad_accum_steps']
                loss.backward()

            running_loss += loss.item() * cfg['grad_accum_steps']
            correct += (torch.argmax(logits, 1) == labels).sum().item()
            total += labels.size(0)

            # Gradient accumulation step
            if (step + 1) % cfg['grad_accum_steps'] == 0 or \
               (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg['max_grad_norm']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg['max_grad_norm']
                    )
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                loss=f'{loss.item() * cfg["grad_accum_steps"]:.4f}',
                acc=f'{100 * correct / total:.1f}%'
            )

        # Epoch metrics
        elapsed = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        test_acc, test_auroc, test_f1 = evaluate(
            model, test_loader, device
        )
        lr = scheduler.get_last_lr()[0]

        print(f"\n   Ep {epoch + 1}: Loss={avg_loss:.4f} | "
              f"Train={train_acc:.1f}% | Test={test_acc:.1f}% | "
              f"AUROC={test_auroc:.1f}% | F1={test_f1:.1f}% | "
              f"LR={lr:.2e} | {elapsed:.0f}s")

        # Track history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_auroc'].append(test_auroc)
        history['test_f1'].append(test_f1)
        history['lr'].append(lr)

        # Best model tracking + early stopping
        if test_auroc > best_auroc:
            best_auroc = test_auroc
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            print(f"   🎯 NEW BEST: AUROC={test_auroc:.2f}%, "
                  f"Acc={test_acc:.2f}%, F1={test_f1:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"   ⏹ Early stopping after {cfg['patience']} "
                      f"epochs without improvement")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, best_auroc, history
