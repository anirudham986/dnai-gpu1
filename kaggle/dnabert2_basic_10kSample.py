import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
import time
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

print("\nFAST TRAINING: DNABERT-2 ON 10K SAMPLES\n")

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
# 1. LOAD DNABERT-2 PRE-TRAINED MODEL
# ===================================================
print("\n1. Loading DNABERT-2 from HuggingFace...")

class DNABERT2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained DNABERT-2 (EXACT model from paper)
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        
        # Freeze MOST layers for faster training - only train last 3 layers
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True  # Train last 3 layers
            else:
                param.requires_grad = False  # Freeze others
                
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # Keep for later pruning analysis
        )
        # Get [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits, outputs.hidden_states

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Special tokens for DNA
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

# Create model
model = DNABERT2Classifier().to(device)
print(f"✅ DNABERT-2 model loaded")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===================================================
# 2. CREATE SYNTHETIC DATASET (10K SAMPLES - CLEAR PATTERNS)
# ===================================================
print("\n2. Creating synthetic dataset with 10K samples (CLEAR patterns)...")

class SimpleDNADataset(Dataset):
    def __init__(self, num_samples, seq_length=256):  # SHORTER sequences!
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.bases = ['A', 'C', 'G', 'T']
        
        print(f"   Generating {num_samples:,} DNA sequences with CLEAR patterns...")
        self.sequences = []
        self.labels = []
        
        # SIMPLE and CLEAR patterns for easy learning
        # Pathogenic: Has "TATAAA" motif (TATA box)
        # Benign: Has "AAAAAA" motif (simple poly-A)
        
        for i in tqdm(range(num_samples), desc="Generating"):
            if i < num_samples // 2:  # First half: BENIGN
                label = 0
                # Create sequence with "AAAAAA" pattern
                seq = ''.join(np.random.choice(self.bases, seq_length))
                # Insert "AAAAAA" at random position
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "AAAAAA" + seq[pos+6:]
                # Also add some other A-rich regions
                for _ in range(3):
                    pos = np.random.randint(0, seq_length - 4)
                    seq = seq[:pos] + "AAAA" + seq[pos+4:]
                    
            else:  # Second half: PATHOGENIC
                label = 1
                # Create sequence with "TATAAA" pattern
                seq = ''.join(np.random.choice(self.bases, seq_length))
                # Insert "TATAAA" at random position
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "TATAAA" + seq[pos+6:]
                # Also add some TATA-like variations
                variations = ["TATATA", "TAAAAA", "TTTAAA"]
                for _ in range(2):
                    pos = np.random.randint(0, seq_length - 6)
                    seq = seq[:pos] + np.random.choice(variations) + seq[pos+6:]
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        print(f"   Classes: {sum(self.labels):,} pathogenic, {num_samples - sum(self.labels):,} benign")
        print(f"   Pattern: Benign = 'AAAAAA', Pathogenic = 'TATAAA'")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # DNABERT expects " " at beginning
        sequence = " " + self.sequences[idx]
        
        # Tokenize with SHORTER max length for speed
        encoding = tokenizer(
            sequence,
            max_length=256,  # SHORTER for speed!
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create dataset with 10K samples (MUCH SMALLER!)
train_size = 8000  # 8K for training
test_size = 2000   # 2K for testing
TOTAL_SAMPLES = train_size + test_size

print(f"\nCreating {TOTAL_SAMPLES:,} total samples...")
full_dataset = SimpleDNADataset(num_samples=TOTAL_SAMPLES, seq_length=256)  # 256bp sequences

# Split
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n✅ Dataset created")
print(f"   Train: {len(train_dataset):,} samples")
print(f"   Test: {len(test_dataset):,} samples")
print(f"   Sequence length: 256bp (for speed)")

# Create data loaders with LARGER batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Larger batch
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)   # Larger batch

# ===================================================
# 3. OPTIMIZED TRAINING FUNCTION (2 EPOCHS ONLY)
# ===================================================
def train_model_fast(model, train_loader, test_loader, epochs=2, lr=3e-5):
    """FAST training - only 2 epochs with high LR"""
    model.train()
    
    # Only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Simple scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    best_acc = 0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_auroc': []}
    
    for epoch in range(epochs):
        print(f"\n⚡ Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # FAST training - no tqdm for inner loop to reduce overhead
        start_epoch_time = time.time()
        batch_count = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Light gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batch_count += 1
            
            # Show progress every 50 batches
            if batch_count % 50 == 0:
                batch_acc = 100.0 * (preds == labels).sum().item() / labels.size(0)
                print(f"  Batch {batch_count}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%")
        
        epoch_time = time.time() - start_epoch_time
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Quick evaluation
        test_acc, test_auroc, test_f1 = evaluate_model_fast(model, test_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%, Test AUROC: {test_auroc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_auroc'].append(test_auroc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            print(f"  🎯 NEW BEST MODEL: {test_acc:.2f}% accuracy")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history, best_acc

def evaluate_model_fast(model, data_loader):
    """FAST evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, _ = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    auroc = roc_auc_score(all_labels, all_probs) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    
    return accuracy, auroc, f1

# ===================================================
# 4. TRAIN THE MODEL (2 EPOCHS ONLY!)
# ===================================================
print("\n3. Training DNABERT-2 on 10K samples (2 EPOCHS ONLY)...")
print("-" * 50)

start_time = time.time()

# Train for ONLY 2 epochs with higher learning rate
trained_model, history, best_acc = train_model_fast(
    model, 
    train_loader, 
    test_loader, 
    epochs=2,  # ONLY 2 EPOCHS!
    lr=3e-5    # Slightly higher LR
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds")
print(f"   Best accuracy achieved: {best_acc:.2f}%")

# Final evaluation
print("\n4. Final evaluation...")
final_acc, final_auroc, final_f1 = evaluate_model_fast(trained_model, test_loader)
print(f"   Final Test Accuracy: {final_acc:.2f}%")
print(f"   Final Test AUROC: {final_auroc:.2f}%")
print(f"   Final Test F1: {final_f1:.2f}%")

# ===================================================
# 5. SAVE EVERYTHING
# ===================================================
print("\n5. Saving model and data...")

# Create directory
save_dir = "/kaggle/working/dnabert2_10k_trained"
os.makedirs(save_dir, exist_ok=True)

# Save model
model_path = f"{save_dir}/dnabert2_10k_trained.pth"
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'config': {
        'model_name': 'DNABERT-2-117M',
        'training_epochs': 2,
        'best_accuracy': best_acc,
        'final_accuracy': final_acc,
        'final_auroc': final_auroc,
        'training_time': training_time,
        'train_samples': train_size,
        'test_samples': test_size,
        'sequence_length': 256,
        'batch_size': 32,
        'learning_rate': 3e-5
    },
    'history': history,
    'tokenizer_info': {
        'name': 'zhihan1996/DNABERT-2-117M',
        'vocab_size': tokenizer.vocab_size
    }
}, model_path)

print(f"✅ Model saved to: {model_path}")

# Save dataset info
dataset_info = {
    'total_samples': TOTAL_SAMPLES,
    'train_samples': train_size,
    'test_samples': test_size,
    'sequence_length': 256,
    'class_distribution': {
        'train_pathogenic': sum(full_dataset.labels[:train_size]),
        'train_benign': train_size - sum(full_dataset.labels[:train_size]),
        'test_pathogenic': sum(full_dataset.labels[train_size:]),
        'test_benign': test_size - sum(full_dataset.labels[train_size:])
    },
    'patterns': {
        'benign': 'Contains "AAAAAA" motif',
        'pathogenic': 'Contains "TATAAA" motif'
    }
}

with open(f"{save_dir}/dataset_info.json", 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"✅ Dataset info saved")

# Save training history
history_path = f"{save_dir}/training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"✅ Training history saved")

# ===================================================
# 6. VERIFICATION
# ===================================================
print("\n6. Final verification...")

# Load and test saved model
checkpoint = torch.load(model_path, map_location=device,weights_only=False)
loaded_model = DNABERT2Classifier().to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# Quick test
test_acc, test_auroc, test_f1 = evaluate_model_fast(loaded_model, test_loader)
print(f"✅ Loaded model test:")
print(f"   Accuracy: {test_acc:.2f}% (Expected: {final_acc:.2f}%)")
print(f"   AUROC: {test_auroc:.2f}%")
print(f"   F1: {test_f1:.2f}%")

# Summary
print("\n" + "="*70)
print("🎯 TRAINING COMPLETE - SUMMARY")
print("="*70)
print(f"Model: DNABERT-2 (zhihan1996/DNABERT-2-117M)")
print(f"Training samples: {train_size:,}")
print(f"Test samples: {test_size:,}")
print(f"Sequence length: 256bp")
print(f"Epochs: 2")
print(f"Best accuracy: {best_acc:.2f}%")
print(f"Final accuracy: {final_acc:.2f}%")
print(f"Training time: {training_time:.2f}s ({training_time/60:.1f} minutes)")
print(f"\nSaved to: {save_dir}/")
print(f"Files created:")
print(f"  - dnabert2_10k_trained.pth (trained model weights)")
print(f"  - dataset_info.json (dataset details)")
print(f"  - training_history.json (training metrics)")
print("\n✅ READY FOR PRUNING EXPERIMENTS!")
print("="*70)
