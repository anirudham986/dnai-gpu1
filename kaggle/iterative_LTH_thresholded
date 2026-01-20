# ===================================================
# STEP 2: LAYER-BY-LAYER PRUNING WITH LOTTERY TICKET HYPOTHESIS
# ===================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
import time
import json
from tqdm import tqdm
import os
import psutil
from copy import deepcopy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import logging
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 STEP 2: LAYER-BY-LAYER PRUNING WITH LOTTERY TICKET HYPOTHESIS")
print("="*70)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# ===================================================
# 1. LOAD THE TRAINED MODEL FROM STEP 1
# ===================================================
print("\n1. Loading trained DNABERT-2 model from Step 1...")

# Define the model class (same as before)
class DNABERT2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits, outputs.hidden_states

# Load the trained model from Step 1
model_path = "/kaggle/working/dnabert2_10k_trained/dnabert2_10k_trained.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = DNABERT2Classifier().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"✅ Model loaded from: {model_path}")
print(f"   Initial accuracy: {checkpoint['config']['best_accuracy']:.2f}%")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===================================================
# 2. LOAD OR CREATE DATASET (REUSE FROM STEP 1)
# ===================================================
print("\n2. Loading dataset...")

# Create the same dataset class from Step 1
class SimpleDNADataset(Dataset):
    def __init__(self, num_samples, seq_length=256):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.bases = ['A', 'C', 'G', 'T']
        
        print(f"   Generating {num_samples:,} DNA sequences...")
        self.sequences = []
        self.labels = []
        
        for i in range(num_samples):
            if i < num_samples // 2:  # Benign
                label = 0
                seq = ''.join(np.random.choice(self.bases, seq_length))
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "AAAAAA" + seq[pos+6:]
                for _ in range(3):
                    pos = np.random.randint(0, seq_length - 4)
                    seq = seq[:pos] + "AAAA" + seq[pos+4:]
            else:  # Pathogenic
                label = 1
                seq = ''.join(np.random.choice(self.bases, seq_length))
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "TATAAA" + seq[pos+6:]
                variations = ["TATATA", "TAAAAA", "TTTAAA"]
                for _ in range(2):
                    pos = np.random.randint(0, seq_length - 6)
                    seq = seq[:pos] + np.random.choice(variations) + seq[pos+6:]
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        print(f"   Classes: {sum(self.labels):,} pathogenic, {num_samples - sum(self.labels):,} benign")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = " " + self.sequences[idx]
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        encoding = tokenizer(
            sequence,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create smaller dataset for pruning experiments
prune_train_size = 4000
prune_val_size = 1000
prune_test_size = 1000
TOTAL_PRUNE_SAMPLES = prune_train_size + prune_val_size + prune_test_size

print(f"Creating {TOTAL_PRUNE_SAMPLES:,} samples for pruning experiments...")
prune_dataset = SimpleDNADataset(num_samples=TOTAL_PRUNE_SAMPLES, seq_length=256)

# Split into train/val/test
prune_train, prune_val, prune_test = random_split(
    prune_dataset,
    [prune_train_size, prune_val_size, prune_test_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
prune_train_loader = DataLoader(prune_train, batch_size=16, shuffle=True)
prune_val_loader = DataLoader(prune_val, batch_size=16, shuffle=False)
prune_test_loader = DataLoader(prune_test, batch_size=16, shuffle=False)

print(f"✅ Dataset created for pruning")
print(f"   Train: {len(prune_train):,} samples")
print(f"   Validation: {len(prune_val):,} samples")
print(f"   Test: {len(prune_test):,} samples")

# ===================================================
# 3. LOTTERY TICKET HYPOTHESIS PRUNER (OPTIMIZED)
# ===================================================
print("\n3. Initializing Lottery Ticket Hypothesis Pruner...")

class LotteryTicketPruner:
    def __init__(self, model, device, save_path="/kaggle/working/original_weights.pt"):
        """
        OPTIMIZED FOR KAGGLE CPU:
        Uses lazy loading to avoid memory issues with 89M parameters
        """
        self.model = model.to(device)
        self.device = device
        
        print("   Using lazy loading to save memory...")
        start_time = time.time()
        
        # LAZY LOADING APPROACH: Save only linear layer weights to disk
        # This prevents deepcopy() memory explosion on 89M params
        linear_weights = {}
        
        # Collect only linear layer weights (these are what we prune)
        for name, param in model.named_parameters():
            if 'weight' in name and ('linear' in name.lower() or 'classifier' in name.lower() or 'attention' in name.lower()):
                linear_weights[name] = param.data.clone().cpu()
        
        # Save to disk (much faster than keeping in RAM)
        torch.save(linear_weights, save_path)
        self.original_weights_path = save_path
        
        # Store which layers we'll prune
        self.prunable_layers = list(linear_weights.keys())
        print(f"   Found {len(self.prunable_layers)} prunable layers")
        
        # Track masks for each layer
        self.masks = {}
        
        init_time = time.time() - start_time
        print(f"   ✅ Pruner initialized in {init_time:.2f} seconds")
    
    def _load_original_weight(self, layer_name):
        """Lazy load original weight from disk when needed"""
        if not hasattr(self, '_cached_weights'):
            self._cached_weights = torch.load(self.original_weights_path, map_location='cpu')
        
        if layer_name in self._cached_weights:
            return self._cached_weights[layer_name].to(self.device)
        return None
    
    def evaluate_model(self, data_loader):
        """Evaluate model accuracy"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs, _ = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        return accuracy
    
    def create_layer_mask(self, layer_name, target_sparsity):
        """Create mask for a specific layer with target sparsity"""
        for name, module in self.model.named_modules():
            if name in layer_name and isinstance(module, nn.Linear):
                weights = module.weight.data.abs().flatten()
                
                # OPTIMIZED: Use percentile instead of sort for large tensors
                k = int(len(weights) * target_sparsity)
                if k > 0:
                    # Use torch.kthvalue which is faster than sort for percentiles
                    threshold = torch.kthvalue(weights, k).values.item()
                    
                    # Create mask (1 = keep, 0 = prune)
                    mask = (module.weight.data.abs() > threshold).float()
                    
                    # Store mask
                    self.masks[layer_name] = mask
                    
                    # Apply mask
                    module.weight.data *= mask
                    
                    # Calculate actual sparsity
                    actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
                    
                    logger.info(f"  Layer {layer_name}: Target {target_sparsity*100:.1f}%, "
                                f"Actual {actual_sparsity*100:.1f}% sparsity")
                    return actual_sparsity
        return 0.0
    
    def create_global_mask(self, target_sparsity):
        """Create global mask across ALL linear layers - OPTIMIZED"""
        logger.info(f"Creating GLOBAL mask with {target_sparsity*100:.1f}% sparsity target")
        
        all_weights = []
        layer_info = {}
        
        # Collect all weights (only for linear layers)
        print("   Collecting weights for global pruning...")
        for name, module in tqdm(list(self.model.named_modules()), desc="Collecting weights"):
            if isinstance(module, nn.Linear):
                weights = module.weight.data.abs().flatten().cpu()
                all_weights.append(weights)
                layer_info[name] = {
                    'module': module,
                    'weight_shape': module.weight.shape,
                    'start_idx': sum([len(w) for w in all_weights]) - len(weights),
                    'end_idx': sum([len(w) for w in all_weights])
                }
        
        if not all_weights:
            logger.warning("No Linear layers found!")
            return 0.0
        
        all_weights = torch.cat(all_weights)
        print(f"   Total weights to consider: {len(all_weights):,}")
        
        # Calculate global threshold using percentile (faster)
        k = int(len(all_weights) * target_sparsity)
        
        # ============ FIX 1: VALIDATE k VALUE ============
        # Fix: Ensure k is valid
        if k <= 0:
            print(f"   WARNING: k={k} is too small. Setting to minimum 1.")
            k = 1  # At least prune 1 weight
        if k >= len(all_weights):
            print(f"   WARNING: k={k} >= total weights. Setting to len-1.")
            k = len(all_weights) - 1
        
        print(f"   Finding threshold to prune {k:,} weights...")
        
        # Use kthvalue for speed - O(n) instead of O(n log n) for sort
        start_time = time.time()
        
        try:
            # kthvalue is 1-indexed, so we need k+1 for k-th smallest
            global_threshold = torch.kthvalue(all_weights, k).values.item()
        except Exception as e:
            print(f"   Error in kthvalue: {e}. Using median-based fallback.")
            # Fallback to median-based threshold
            median_val = torch.median(all_weights).item()
            global_threshold = median_val * target_sparsity
        
        threshold_time = time.time() - start_time
        
        # ============ FIX 2: CHECK THRESHOLD VALIDITY ============
        print(f"   Threshold found in {threshold_time:.2f}s: {global_threshold:.10f}")
        
        # Check if threshold is too small or zero
        if global_threshold < 1e-8:
            print(f"   ⚠️ Threshold too small ({global_threshold:.10f})! Using percentile fallback...")
            # Use numpy percentile which is more stable for small values
            try:
                import numpy as np
                weights_np = all_weights.numpy()
                global_threshold = np.percentile(weights_np, target_sparsity * 100)
                print(f"   Adjusted threshold via percentile: {global_threshold:.10f}")
                
                # If still too small, use a minimum threshold
                if global_threshold < 1e-8:
                    print(f"   Still too small. Using minimum threshold 1e-6.")
                    global_threshold = 1e-6
            except Exception as e:
                print(f"   Error in percentile: {e}. Using fixed threshold 1e-6.")
                global_threshold = 1e-6
        
        total_pruned = 0
        total_weights = 0
        
        print("   Applying masks to layers...")
        for name, info in tqdm(layer_info.items(), desc="Applying masks"):
            module = info['module']
            mask = (module.weight.data.abs() > global_threshold).float()
            self.masks[name] = mask
            module.weight.data *= mask
            
            layer_pruned = (mask == 0).sum().item()
            layer_total = mask.numel()
            total_pruned += layer_pruned
            total_weights += layer_total
        
        global_sparsity = total_pruned / total_weights if total_weights > 0 else 0
        logger.info(f"Global sparsity achieved: {global_sparsity*100:.1f}% ({total_pruned:,}/{total_weights:,} weights)")
        
        # Sanity check
        if global_sparsity < target_sparsity * 0.5:  # Less than half of target
            print(f"   ⚠️ Warning: Achieved sparsity ({global_sparsity*100:.1f}%) much lower than target ({target_sparsity*100:.1f}%)")
        
        return global_sparsity
    
    def reset_to_original_weights(self):
        """LOTTERY TICKET: Reset surviving weights to original values - OPTIMIZED"""
        logger.info("Applying Lottery Ticket Hypothesis: Resetting to original weights...")
        
        # Lazy load weights from disk
        original_weights = torch.load(self.original_weights_path, map_location=self.device)
        
        reset_count = 0
        for name, module in tqdm(list(self.model.named_modules()), desc="Resetting weights"):
            if isinstance(module, nn.Linear) and name in self.masks:
                mask = self.masks[name]
                
                # Find corresponding original weight in our saved dict
                # Try different naming patterns
                weight_key = None
                for key in original_weights.keys():
                    if name in key or key.endswith(name + '.weight'):
                        weight_key = key
                        break
                
                if weight_key and weight_key in original_weights:
                    original_weight = original_weights[weight_key]
                    
                    # Ensure shapes match
                    if original_weight.shape == module.weight.shape:
                        # Only reset weights that survived pruning (mask == 1)
                        module.weight.data[mask.bool()] = original_weight[mask.bool()]
                        reset_count += 1
                
                # Reset bias if it exists
                if module.bias is not None:
                    bias_key = weight_key.replace('.weight', '.bias') if weight_key else None
                    if bias_key and bias_key in original_weights:
                        original_bias = original_weights[bias_key]
                        if original_bias.shape == module.bias.shape:
                            module.bias.data.copy_(original_bias)
        
        logger.info(f"✅ Lottery Ticket reset complete ({reset_count} layers reset)")
    
    def fine_tune_with_mask(self, train_loader, val_loader, epochs=5, lr=1e-4):
        """Fine-tune while respecting the mask (LTH training)"""
        logger.info(f"Fine-tuning for {epochs} epochs with LTH...")
        
        # Only optimize parameters that are not masked
        params_to_optimize = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.masks:
                mask = self.masks[name]
                
                # Create parameter group with masked gradients
                params_to_optimize.append({
                    'params': module.parameters(),
                    'lr': lr,
                    'mask': mask if name + '.weight' in self.masks else None
                })
            elif isinstance(module, nn.Linear):
                params_to_optimize.append({
                    'params': module.parameters(),
                    'lr': lr
                })
        
        optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient masking (LTH)
                self._apply_gradient_masking()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            
            # Validate
            val_acc = self.evaluate_model(val_loader)
            scheduler.step()
            
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                       f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = deepcopy(self.model.state_dict())
                logger.info(f"    🎯 NEW BEST: {val_acc:.2f}%")
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        return best_acc
    
    def _apply_gradient_masking(self):
        """Apply gradient masking for LTH (zero out gradients for pruned weights)"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.masks:
                if module.weight.grad is not None:
                    # Zero out gradients for pruned weights
                    mask = self.masks[name]
                    module.weight.grad.data *= mask
    
    def iterative_pruning_lth(self, train_loader, val_loader, test_loader, 
                             target_sparsities=[0.2, 0.36, 0.488, 0.5904]):
        """
        Iterative pruning with Lottery Ticket Hypothesis
        Each step: Prune → LTH reset → Fine-tune
        target_sparsities: Cumulative sparsity at each step
        """
        logger.info("Starting iterative pruning with Lottery Ticket Hypothesis...")
        
        # Initial evaluation
        print("Evaluating initial model...")
        initial_acc = self.evaluate_model(test_loader)
        logger.info(f"Initial accuracy: {initial_acc:.2f}%")
        
        results = []
        previous_sparsity = 0.0
        
        for step, target_cumulative_sparsity in enumerate(target_sparsities):
            step_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP {step+1}/{len(target_sparsities)}")
            logger.info(f"Target cumulative sparsity: {target_cumulative_sparsity*100:.1f}%")
            
            # ============ FIX 3: ADD ERROR HANDLING ============
            try:
                # Calculate step sparsity (prune this much more)
                if step == 0:
                    step_sparsity = target_cumulative_sparsity
                else:
                    # Calculate how much more to prune
                    current_density = 1 - previous_sparsity
                    target_density = 1 - target_cumulative_sparsity
                    step_sparsity = 1 - (target_density / current_density)
                
                logger.info(f"Step sparsity: {step_sparsity*100:.1f}%")
                
                # 1. Create global mask for this step
                print(f"Step {step+1}: Creating global mask...")
                step_sparsity_achieved = self.create_global_mask(step_sparsity)
                
                # Check if pruning actually happened
                if step_sparsity_achieved < 0.01:  # Less than 1% pruned
                    print(f"   ⚠️ Very little pruning achieved ({step_sparsity_achieved*100:.2f}%)")
                    print(f"   Adjusting step sparsity to be more aggressive...")
                    step_sparsity = min(step_sparsity * 2, 0.5)  # Try harder
                    step_sparsity_achieved = self.create_global_mask(step_sparsity)
                
                cumulative_sparsity = 1 - ((1 - previous_sparsity) * (1 - step_sparsity_achieved))
                
                # 2. Apply Lottery Ticket Hypothesis: Reset to original weights
                print(f"Step {step+1}: Applying Lottery Ticket reset...")
                self.reset_to_original_weights()
                
                # 3. Fine-tune with masks
                logger.info(f"Step {step+1}: Fine-tuning at {cumulative_sparsity*100:.1f}% sparsity...")
                fine_tuned_acc = self.fine_tune_with_mask(
                    train_loader, val_loader, 
                    epochs=3,  # Short fine-tuning for speed
                    lr=1e-4
                )
                
                # 4. Evaluate
                print(f"Step {step+1}: Evaluating...")
                final_acc = self.evaluate_model(test_loader)
                
                step_time = time.time() - step_start
                results.append({
                    'step': step + 1,
                    'target_cumulative_sparsity': target_cumulative_sparsity,
                    'step_sparsity': step_sparsity,
                    'achieved_cumulative_sparsity': cumulative_sparsity,
                    'fine_tuned_acc': fine_tuned_acc,
                    'final_acc': final_acc,
                    'step_time_seconds': step_time
                })
                
                logger.info(f"Step {step+1} Results:")
                logger.info(f"  Cumulative sparsity: {cumulative_sparsity*100:.1f}%")
                logger.info(f"  Fine-tuned accuracy: {fine_tuned_acc:.2f}%")
                logger.info(f"  Final accuracy: {final_acc:.2f}%")
                logger.info(f"  Step time: {step_time:.1f}s")
                
                previous_sparsity = cumulative_sparsity
                
            except Exception as e:
                # ============ ERROR HANDLING ============
                print(f"\n❌ ERROR in Step {step+1}: {str(e)}")
                print(f"Continuing with next step...")
                import traceback
                traceback.print_exc()
                
                # Log failed step
                results.append({
                    'step': step + 1,
                    'error': str(e),
                    'failed': True
                })
                continue  # Skip to next step
            # ============ END OF FIX 3 ============
        
        return results
    
    def analyze_sparsity(self):
        """Analyze current sparsity of the model"""
        total_weights = 0
        zero_weights = 0
        
        print(f"\n{'='*60}")
        print("SPARSITY ANALYSIS")
        print('='*60)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                layer_zeros = (weights == 0).sum().item()
                layer_total = weights.numel()
                
                total_weights += layer_total
                zero_weights += layer_zeros
                
                sparsity = layer_zeros / layer_total
                print(f"  {name:30}: {sparsity*100:6.1f}% sparse "
                      f"({layer_zeros:,}/{layer_total:,})")
        
        if total_weights > 0:
            total_sparsity = zero_weights / total_weights
            print(f"\n  TOTAL SPARSITY: {total_sparsity*100:.1f}%")
            print(f"  Zero weights: {zero_weights:,}/{total_weights:,}")
            print(f"  Effective parameters: {total_weights - zero_weights:,}")
        
        return total_sparsity

# ===================================================
# 4. RUN LOTTERY TICKET PRUNING (WITH TIMING)
# ===================================================
print("\n4. Running Lottery Ticket Hypothesis Pruning...")

# Start overall timer
overall_start = time.time()

# Initialize pruner with optimized lazy loading
print("Initializing pruner (optimized for Kaggle CPU)...")
pruner_start = time.time()
pruner = LotteryTicketPruner(model, device, save_path="/kaggle/working/original_weights.pt")
pruner_time = time.time() - pruner_start
print(f"✅ Pruner initialized in {pruner_time:.2f} seconds")

# ============ FIX 4: ADD WEIGHT DIAGNOSTIC ============
print("\n🔍 Weight Distribution Analysis (Before Pruning):")
all_weights = []
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() > 1:
        weights = param.data.abs().flatten().cpu()
        all_weights.append(weights)

if all_weights:
    all_weights = torch.cat(all_weights)
    print(f"  Total weights: {len(all_weights):,}")
    print(f"  Min: {all_weights.min().item():.10f}")
    print(f"  Max: {all_weights.max().item():.6f}")
    print(f"  Mean: {all_weights.mean().item():.6f}")
    print(f"  Std: {all_weights.std().item():.6f}")
    
    # Check zeros
    exactly_zero = (all_weights == 0).sum().item()
    near_zero = (all_weights < 1e-8).sum().item()
    print(f"  Exactly zero: {exactly_zero:,} ({exactly_zero/len(all_weights)*100:.2f}%)")
    print(f"  Near-zero (<1e-8): {near_zero:,} ({near_zero/len(all_weights)*100:.2f}%)")
    
    # Check what threshold would be for 33% pruning
    if len(all_weights) > 100:
        k_33 = int(len(all_weights) * 0.33)
        if k_33 > 0 and k_33 < len(all_weights):
            threshold_33 = torch.kthvalue(all_weights, k_33).values.item()
            print(f"  Threshold for 33% sparsity: {threshold_33:.10f}")
            print(f"  Weights <= threshold: {(all_weights <= threshold_33).sum().item():,}")
# ============ END OF FIX 4 ============

# Target sparsities for iterative pruning
target_sparsities = [
    0.33,        # Step 1: 33% sparse
    0.55,        # Step 2: 55% sparse cumulative
    0.70,        # Step 3: 70% sparse cumulative  
    0.80         # Step 4: 80% sparse cumulative
]

print(f"\nPruning schedule for 80% total sparsity:")
for i, sparsity in enumerate(target_sparsities):
    print(f"  Step {i+1}: Target {sparsity*100:.1f}% cumulative sparsity")

# Run iterative pruning with LTH
print("\nStarting iterative pruning process...")
results_start = time.time()
results = pruner.iterative_pruning_lth(
    prune_train_loader,
    prune_val_loader,
    prune_test_loader,
    target_sparsities=target_sparsities
)
pruning_time = time.time() - results_start
print(f"✅ Iterative pruning completed in {pruning_time:.1f} seconds")

# ============ FIX 5: ADD CONTINUATION LOGIC ============
print(f"\n📊 Pruning Diagnostics:")
print(f"  Target steps: {len(target_sparsities)}")
completed_steps = [r for r in results if 'failed' not in r]
failed_steps = [r for r in results if 'failed' in r]
print(f"  Completed steps: {len(completed_steps)}")
print(f"  Failed steps: {len(failed_steps)}")

# Check if we completed all steps
if len(completed_steps) < len(target_sparsities):
    print(f"\n⚠️ WARNING: Only completed {len(completed_steps)}/{len(target_sparsities)} steps!")
    
    # Continue from current sparsity
    current_sparsity = pruner.analyze_sparsity()
    print(f"Current sparsity: {current_sparsity*100:.1f}%")
    
    # Adjust remaining targets
    remaining_steps = len(target_sparsities) - len(completed_steps)
    if remaining_steps > 0 and current_sparsity < 0.8:
        print(f"\nAttempting to complete pruning with adjusted targets...")
        # Calculate new targets to reach 80% from current
        remaining_density = 1 - current_sparsity
        target_final_density = 0.2  # 80% sparse = 20% density
        
        # Spread remaining pruning over remaining steps
        keep_ratio_per_step = (target_final_density / remaining_density) ** (1/remaining_steps)
        
        adjusted_targets = []
        current_density = remaining_density
        for i in range(remaining_steps):
            current_density *= keep_ratio_per_step
            adjusted_targets.append(1 - current_density)
        
        print(f"Adjusted targets: {[f'{t*100:.1f}%' for t in adjusted_targets]}")
        
        # Continue pruning with adjusted targets
        for i, target in enumerate(adjusted_targets):
            step_num = len(completed_steps) + i + 1
            print(f"\n📌 Continuing with Step {step_num}: Target {target*100:.1f}%")
            try:
                # Calculate step sparsity
                step_sparsity = 1 - ((1 - target) / (1 - current_sparsity))
                print(f"  Step sparsity needed: {step_sparsity*100:.1f}%")
                
                # Prune
                step_sparsity_achieved = pruner.create_global_mask(step_sparsity)
                current_sparsity = 1 - ((1 - current_sparsity) * (1 - step_sparsity_achieved))
                
                # LTH reset and fine-tune
                pruner.reset_to_original_weights()
                fine_tuned_acc = pruner.fine_tune_with_mask(
                    prune_train_loader, prune_val_loader, epochs=3, lr=1e-4
                )
                final_acc = pruner.evaluate_model(prune_test_loader)
                
                results.append({
                    'step': step_num,
                    'target_cumulative_sparsity': target,
                    'step_sparsity': step_sparsity,
                    'achieved_cumulative_sparsity': current_sparsity,
                    'fine_tuned_acc': fine_tuned_acc,
                    'final_acc': final_acc,
                    'continued': True
                })
                
                print(f"  Step {step_num} completed: {current_sparsity*100:.1f}% sparse, {final_acc:.2f}% accuracy")
                
            except Exception as e:
                print(f"  ❌ Failed to continue: {e}")
                break
# ============ END OF FIX 5 ============

# ===================================================
# 5. FINAL ANALYSIS AND SAVING
# ===================================================
print("\n5. Final analysis and saving results...")

# Analyze final sparsity
final_sparsity = pruner.analyze_sparsity()

# Final evaluation - VERIFY ACCURACY
print("\n🔍 Verifying final accuracy...")

# Method 1: Use pruner's evaluate
pruner_final_acc = pruner.evaluate_model(prune_test_loader)

# Method 2: Manual verification
def manual_evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, _ = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

manual_final_acc = manual_evaluate(pruner.model, prune_test_loader, device)

print(f"Pruner evaluation: {pruner_final_acc:.2f}%")
print(f"Manual evaluation: {manual_final_acc:.2f}%")

# Use the average for final report
final_acc = (pruner_final_acc + manual_final_acc) / 2
initial_acc = checkpoint['config']['best_accuracy']

print(f"\n{'='*60}")
print("FINAL RESULTS")
print('='*60)
print(f"Initial accuracy: {initial_acc:.2f}%")
print(f"Final accuracy: {final_acc:.2f}%")
print(f"Accuracy change: {final_acc - initial_acc:+.2f}%")
print(f"Final sparsity: {final_sparsity*100:.1f}%")

# Check if results are reasonable
if final_acc > initial_acc + 5:  # More than 5% improvement
    print(f"\n⚠️ WARNING: Accuracy improved by {final_acc - initial_acc:+.2f}%")
    print("This might indicate data leakage or evaluation bug!")
elif final_acc < initial_acc - 10:  # More than 10% drop
    print(f"\n⚠️ WARNING: Accuracy dropped by {initial_acc - final_acc:.2f}%")
    print("Pruning might be too aggressive!")

# Save pruned model
save_dir = "/kaggle/working/dnabert2_pruned_lth"
os.makedirs(save_dir, exist_ok=True)

model_path = f"{save_dir}/dnabert2_80percent_pruned_lth.pth"
torch.save({
    'model_state_dict': pruner.model.state_dict(),
    'masks': pruner.masks,
    'original_weights_path': pruner.original_weights_path,
    'pruning_results': results,
    'config': {
        'initial_accuracy': initial_acc,
        'final_accuracy': final_acc,
        'final_sparsity': final_sparsity,
        'total_steps': len(target_sparsities),
        'target_sparsities': target_sparsities
    }
}, model_path)

print(f"\n✅ Pruned model saved to: {model_path}")

# Save detailed results
results_path = f"{save_dir}/pruning_results.json"
with open(results_path, 'w') as f:
    # Convert tensors to lists for JSON
    json_results = []
    for r in results:
        json_r = {}
        for k, v in r.items():
            if torch.is_tensor(v):
                json_r[k] = v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, np.ndarray):
                json_r[k] = v.tolist()
            else:
                json_r[k] = v
        json_results.append(json_r)
    json.dump(json_results, f, indent=2)

print(f"✅ Pruning results saved to: {results_path}")

# ===================================================
# 6. COMPREHENSIVE COMPARISON
# ===================================================
print("\n6. Comprehensive comparison...")

# Load original model for comparison
original_model = DNABERT2Classifier().to(device)
original_model.load_state_dict(checkpoint['model_state_dict'])

# Compare inference time
def measure_inference_time(model, data_loader):
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids, attention_mask)
    
    return time.time() - start_time

# Warm-up
print("Warming up for inference speed test...")
_ = measure_inference_time(original_model, prune_test_loader)
_ = measure_inference_time(pruner.model, prune_test_loader)

# Measure
print("Measuring inference speed...")
original_time = measure_inference_time(original_model, prune_test_loader)
pruned_time = measure_inference_time(pruner.model, prune_test_loader)

speedup = ((original_time - pruned_time) / original_time) * 100 if original_time > 0 else 0

print(f"\n{'='*60}")
print("PERFORMANCE COMPARISON")
print('='*60)
print(f"Inference time:")
print(f"  Original model: {original_time:.2f}s")
print(f"  Pruned model ({final_sparsity*100:.1f}% sparse): {pruned_time:.2f}s")
print(f"  Speed improvement: {speedup:.1f}%")

# Count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero

orig_total, orig_nonzero = count_parameters(original_model)
pruned_total, pruned_nonzero = count_parameters(pruner.model)

print(f"\nParameter comparison:")
print(f"  Original: {orig_total:,} total, {orig_nonzero:,} non-zero")
print(f"  Pruned: {pruned_total:,} total, {pruned_nonzero:,} non-zero")
print(f"  Non-zero reduction: {(1 - pruned_nonzero/orig_nonzero)*100:.1f}%")

# Total time
overall_time = time.time() - overall_start
print(f"\n{'='*60}")
print("🎉 LOTTERY TICKET PRUNING COMPLETE!")
print('='*60)
print(f"✓ Total execution time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
print(f"✓ Achieved {final_sparsity*100:.1f}% sparsity")
print(f"✓ Final accuracy: {final_acc:.2f}% (Initial: {initial_acc:.2f}%)")
print(f"✓ Speed improvement: {speedup:.1f}%")
print(f"✓ Model saved to: {save_dir}/")
print('='*60)

# Memory usage
process = psutil.Process(os.getpid())
end_ram = process.memory_info().rss / 1024**2
print(f"\nRAM usage: {end_ram:.2f} MB")
print(f"Execution complete!")
