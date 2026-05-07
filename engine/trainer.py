# =====================================================================
# engine/trainer.py — Training loop with:
#   - Timed checkpoints (every ~8.5 hours — for Kaggle 9hr limit)
#   - Epoch checkpoints (every N epochs)
#   - Best-model checkpoints (on every new best AUROC)
#   - Full resume capability (restores model + optimizer + scheduler
#     + scaler + RNG states + history + epoch counter)
#   - Mixed precision, gradient accumulation, cosine LR, early stopping
# =====================================================================

import os
import glob
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from copy import deepcopy

from .evaluator import evaluate


# =====================================================================
# Checkpoint helpers
# =====================================================================

def _save_checkpoint(path: str, epoch: int, model, optimizer, scheduler,
                     scaler, best_auroc: float, best_acc: float,
                     history: dict, cfg: dict, session_start: float,
                     tag: str = ''):
    """
    Save a full training checkpoint — everything needed to resume exactly.

    Saved state:
      - epoch index (0-based, the epoch just COMPLETED)
      - model, optimizer, scheduler, scaler state dicts
      - RNG states (torch + numpy) for reproducibility
      - training history up to this point
      - config dict
      - best metrics seen so far
      - wall-clock session start time (for timed checkpoint accounting)
    """
    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict() if scaler is not None else None,
        'best_auroc':           best_auroc,
        'best_acc':             best_acc,
        'history':              history,
        'cfg':                  cfg,
        'val_fold':             cfg.get('val_fold', -1),
        'rng_torch':            torch.get_rng_state(),
        'rng_numpy':            np.random.get_state(),
        'session_start':        session_start,
        'tag':                  tag,
        'saved_at':             time.time(),
    }
    if torch.cuda.is_available():
        checkpoint['rng_cuda'] = torch.cuda.get_rng_state()

    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    scaler=None, device=None):
    """
    Load a checkpoint and restore all states.

    Args:
        path:      Path to the .pth checkpoint file
        model:     Model to restore weights into
        optimizer: Optional — restore optimizer state
        scheduler: Optional — restore scheduler state
        scaler:    Optional — restore GradScaler state
        device:    torch.device for map_location

    Returns:
        dict with 'epoch', 'best_auroc', 'best_acc', 'history', 'cfg'
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    # Restore RNG states for full reproducibility
    if 'rng_torch' in ckpt:
        torch.set_rng_state(ckpt['rng_torch'])
    if 'rng_numpy' in ckpt:
        np.random.set_state(ckpt['rng_numpy'])
    if 'rng_cuda' in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt['rng_cuda'])

    return {
        'epoch':      ckpt['epoch'],        # Last completed epoch (0-based)
        'best_auroc': ckpt['best_auroc'],
        'best_acc':   ckpt['best_acc'],
        'history':    ckpt['history'],
        'cfg':        ckpt.get('cfg', {}),
        'val_fold':   ckpt.get('val_fold', -1),
        'saved_at':   ckpt.get('saved_at', 0),
    }


def _rotate_checkpoints(save_dir: str, prefix: str, keep: int = 3):
    """
    Delete old epoch checkpoints, keeping only the newest `keep` files.
    Does NOT delete 'best' or 'timed' checkpoints.
    """
    pattern = os.path.join(save_dir, f'{prefix}_ep*.pth')
    files   = sorted(glob.glob(pattern), key=os.path.getmtime)
    while len(files) > keep:
        try:
            os.remove(files.pop(0))
        except OSError:
            break


def find_latest_checkpoint(save_dir: str) -> str | None:
    """
    Scan save_dir and return the path of the most recently saved checkpoint.
    Looks for: checkpoint_best.pth, timed checkpoints, epoch checkpoints.
    Returns None if no checkpoint found.
    """
    if not os.path.isdir(save_dir):
        return None
    patterns = [
        'checkpoint_timed_*.pth',
        'checkpoint_ep*.pth',
        'checkpoint_best.pth',
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(os.path.join(save_dir, pat)))
    if not all_files:
        return None
    # Return the most recently saved file
    return max(all_files, key=os.path.getmtime)


# =====================================================================
# Main training function
# =====================================================================

def train(model, train_loader, val_loader, device, criterion, cfg: dict,
          use_amp: bool = True, resume_from: str = None):
    """
    Full training loop for the NTv2 dual-sequence classifier.

    Args:
        model:        NTv2DualSeqClassifier (already on device)
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        device:       torch.device
        criterion:    Loss function (FocalLoss)
        cfg:          Configuration dictionary
        use_amp:      Whether to use mixed precision (GPU only)
        resume_from:  Optional path to a checkpoint to resume from

    Returns:
        (best_accuracy, best_auroc, history_dict)
    """
    save_dir   = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    ckpt_prefix = f"ntv2_{cfg.get('dataset_name', 'model')}_fold{cfg.get('val_fold', 0)}"

    print(f"\n   Config: {cfg['epochs']} epochs | "
          f"backbone_lr={cfg['backbone_lr']} | "
          f"head_lr={cfg['head_lr']} | focal_γ={cfg['focal_gamma']}")

    # ------------------------------------------------------------------
    # Optimizer — differential LR (backbone very low, head higher)
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['backbone_lr'], cfg['head_lr']),
        weight_decay=cfg['weight_decay']
    )

    # ------------------------------------------------------------------
    # Cosine schedule with warmup
    # ------------------------------------------------------------------
    total_steps  = (len(train_loader) // cfg['grad_accum_steps']) * cfg['epochs']
    warmup_steps = int(total_steps * cfg['warmup_fraction'])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"   Steps: {total_steps} | Warmup: {warmup_steps}")

    # ------------------------------------------------------------------
    # Mixed precision scaler (AMP — GPU sm_70+ only)
    # ------------------------------------------------------------------
    scaler = GradScaler('cuda') if use_amp else None

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    best_auroc       = 0.0
    best_acc         = 0.0
    best_state       = None
    patience_counter = 0
    start_epoch      = 0
    session_start    = time.time()
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_auroc': [], 'val_f1': [],
        'lr': [],
    }

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if resume_from is not None:
        print(f"\n   ♻️  RESUMING from: {resume_from}")
        ckpt_meta = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device
        )
        start_epoch      = ckpt_meta['epoch'] + 1   # Resume AFTER last completed epoch
        best_auroc       = ckpt_meta['best_auroc']
        best_acc         = ckpt_meta['best_acc']
        history          = ckpt_meta['history']
        best_state       = deepcopy(model.state_dict())
        print(f"   ♻️  Resuming from epoch {start_epoch} "
              f"(best so far: AUROC={best_auroc:.2f}% Acc={best_acc:.2f}%)")
    else:
        print(f"\n   🚀  Starting fresh training from epoch 1")

    # ------------------------------------------------------------------
    # Checkpoint interval (timed — for Kaggle 9hr sessions)
    # ------------------------------------------------------------------
    CHECKPOINT_INTERVAL_SECS = cfg.get('checkpoint_interval_hours', 8.5) * 3600
    last_timed_checkpoint_t  = session_start
    SAVE_EVERY_N_EPOCHS      = cfg.get('save_every_n_epochs', 2)

    print(f"\n   ⏱  Timed checkpoints every "
          f"{cfg.get('checkpoint_interval_hours', 8.5):.1f} hours")
    print(f"   💾  Epoch checkpoints every {SAVE_EVERY_N_EPOCHS} epochs")
    print(f"   📁  Save dir: {save_dir}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"   Ep {epoch + 1}/{cfg['epochs']}",
            leave=False
        )

        for step, batch in enumerate(pbar):
            ref_ids  = batch['ref_ids'].to(device)
            ref_mask = batch['ref_mask'].to(device)
            alt_ids  = batch['alt_ids'].to(device)
            alt_mask = batch['alt_mask'].to(device)
            labels   = batch['labels'].to(device)

            if use_amp:
                with autocast('cuda'):
                    logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                    loss   = criterion(logits, labels) / cfg['grad_accum_steps']
                scaler.scale(loss).backward()
            else:
                logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                loss   = criterion(logits, labels) / cfg['grad_accum_steps']
                loss.backward()

            running_loss += loss.item() * cfg['grad_accum_steps']
            correct      += (torch.argmax(logits, 1) == labels).sum().item()
            total        += labels.size(0)

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

        # ---- Epoch metrics ----------------------------------------
        elapsed    = time.time() - epoch_start
        avg_loss   = running_loss / len(train_loader)
        train_acc  = 100 * correct / total
        val_acc, val_auroc, val_f1 = evaluate(model, val_loader, device, use_amp=use_amp)
        lr = scheduler.get_last_lr()[0]

        print(f"\n   Ep {epoch + 1}: Loss={avg_loss:.4f} | "
              f"Train={train_acc:.1f}% | Val={val_acc:.1f}% | "
              f"AUROC={val_auroc:.1f}% | F1={val_f1:.1f}% | "
              f"LR={lr:.2e} | {elapsed:.0f}s")

        # Track history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        history['val_f1'].append(val_f1)
        history['lr'].append(lr)

        # ---- Best model tracking + early stopping -----------------
        if val_auroc > best_auroc:
            best_auroc       = val_auroc
            best_acc         = val_acc
            best_state       = deepcopy(model.state_dict())
            patience_counter = 0

            # Save best checkpoint immediately
            best_path = os.path.join(save_dir, f'{ckpt_prefix}_best.pth')
            _save_checkpoint(
                best_path, epoch, model, optimizer, scheduler,
                scaler, best_auroc, best_acc, history, cfg,
                session_start, tag='best'
            )
            print(f"   🎯 NEW BEST: AUROC={val_auroc:.2f}% "
                  f"Acc={val_acc:.2f}% F1={val_f1:.2f}% → saved ✅")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"   ⏹ Early stopping after {cfg['patience']} "
                      f"epochs without improvement")
                break

        # ---- Epoch checkpoint (every N epochs) --------------------
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            ep_path = os.path.join(
                save_dir, f'{ckpt_prefix}_ep{epoch + 1:03d}.pth'
            )
            _save_checkpoint(
                ep_path, epoch, model, optimizer, scheduler,
                scaler, best_auroc, best_acc, history, cfg,
                session_start, tag=f'epoch_{epoch+1}'
            )
            print(f"   💾 Epoch checkpoint saved → {os.path.basename(ep_path)}")
            _rotate_checkpoints(save_dir, ckpt_prefix, keep=3)

        # ---- Timed checkpoint (every ~8.5 hours) ------------------
        time_since_last = time.time() - last_timed_checkpoint_t
        total_elapsed   = time.time() - session_start

        if time_since_last >= CHECKPOINT_INTERVAL_SECS:
            ts = int(time.time())
            timed_path = os.path.join(
                save_dir,
                f'{ckpt_prefix}_timed_ep{epoch + 1:03d}_{ts}.pth'
            )
            _save_checkpoint(
                timed_path, epoch, model, optimizer, scheduler,
                scaler, best_auroc, best_acc, history, cfg,
                session_start, tag=f'timed_ep{epoch+1}'
            )
            last_timed_checkpoint_t = time.time()
            hours_elapsed = total_elapsed / 3600
            print(f"\n   {'='*60}")
            print(f"   ⏰ TIMED CHECKPOINT SAVED (session: {hours_elapsed:.1f}h)")
            print(f"   📁 {os.path.basename(timed_path)}")
            print(f"   ♻️  To resume in next cell:")
            print(f"      --resume {timed_path}")
            print(f"   {'='*60}\n")

        # Save history JSON after every epoch (cheap, always useful)
        hist_path = os.path.join(save_dir, 'training_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # Restore best model weights before returning
    # ------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n   ✅ Restored best model (AUROC={best_auroc:.2f}%)")

    return best_acc, best_auroc, history
