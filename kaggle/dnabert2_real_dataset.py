import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# 1. Load DNABERT-2
print("\n" + "="*70)
print("STEP 1: LOADING DNABERT-2")
print("="*70)

class DNABERT2ClassifierLayerwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        
        self.param_groups = {
            'embeddings': [],
            'early_layers': [],
            'middle_layers': [],
            'late_layers': [],
            'classifier': []
        }
        
        # Categorize parameters
        for name, param in self.bert.named_parameters():
            if 'embeddings' in name:
                self.param_groups['embeddings'].append(param)
            elif any(f'layer.{i}.' in name for i in range(0, 6)):
                self.param_groups['early_layers'].append(param)
            elif any(f'layer.{i}.' in name for i in range(6, 10)):
                self.param_groups['middle_layers'].append(param)
            elif any(f'layer.{i}.' in name for i in range(10, 12)):
                self.param_groups['late_layers'].append(param)
        
        # Classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.param_groups['classifier'].extend(list(self.classifier.parameters()))
        
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

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

# Create model
model = DNABERT2ClassifierLayerwise().to(device)
print(f"DNABERT-2 model loaded")

# 2. Create dataset class for your CSV
print("\n2. Creating dataset loader for VEP data...")

class VEPDNADataset(Dataset):
    def __init__(self, csv_path, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        print(f"Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Display dataset info
        print(f"Dataset shape: {self.df.shape}")
        print(f"First few rows:")
        print(self.df.head())
        
        # Check for required columns
        if 'CONSEQUENCE' not in self.df.columns:
            print("Warning: No 'CONSEQUENCE' column found")
        
        # Create sequences from genomic data
        # Since you don't have actual sequences, we'll create mock ones
        # IMPORTANT: Replace this with real sequence fetching if available
        self.sequences = []
        self.labels = []
        
        print("Generating mock sequences from genomic data...")
        bases = ['A', 'C', 'G', 'T']
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            # Create a mock sequence based on genomic position
            # This is a TEMPORARY solution - you should replace with real sequences
            seq = ''.join(np.random.choice(bases, seq_length))
            
            # Add mutation at position (mock)
            if 'POS' in row and 'REF' in row and 'ALT' in row:
                pos_in_seq = min(int(row['POS']) % seq_length, seq_length - 5)
                seq = seq[:pos_in_seq] + row['ALT'] + seq[pos_in_seq+1:]
            
            self.sequences.append(seq)
            
            # Determine label
            if 'LABEL' in row:
                if row['LABEL'] == 'Pathogenic':
                    self.labels.append(1)
                elif row['LABEL'] == 'Common':
                    self.labels.append(0)
                else:
                    # Try to infer from other columns
                    if 'PATHOGENIC' in str(row).upper():
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
            else:
                # Default to benign if no label
                self.labels.append(0)
        
        print(f"Created {len(self.sequences)} sequences")
        print(f"Labels - Pathogenic(1): {sum(self.labels)}, Benign(0): {len(self.labels)-sum(self.labels)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = " " + self.sequences[idx]
        
        encoding = self.tokenizer(
            sequence,
            max_length=self.seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 3. Load your dataset
# Update this path to your actual CSV file location
CSV_PATH = "/kaggle/input/vep-pathogenic-non-coding/vep_pathogenic_non_coding_subset.csv"

try:
    dataset = VEPDNADataset(CSV_PATH, tokenizer, seq_length=512)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nCreating small mock dataset for testing...")
    # Create a small mock dataset for testing
    class MockDataset(Dataset):
        def __init__(self, num_samples=1000, seq_length=512):
            self.sequences = []
            self.labels = []
            bases = ['A', 'C', 'G', 'T']
            
            for i in range(num_samples):
                seq = ''.join(np.random.choice(bases, seq_length))
                self.sequences.append(seq)
                self.labels.append(1 if i < num_samples//2 else 0)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            sequence = " " + self.sequences[idx]
            encoding = tokenizer(
                sequence,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    dataset = MockDataset(num_samples=1000, seq_length=512)

# Split dataset
TOTAL_SAMPLES = len(dataset)
train_size = int(0.8 * TOTAL_SAMPLES)
test_size = TOTAL_SAMPLES - train_size

print(f"\nDataset size: {TOTAL_SAMPLES}")
print(f"Train: {train_size}, Test: {test_size}")

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4. Training functions (keep as is)
def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
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

def train_model_layerwise(model, train_loader, test_loader, epochs=3, base_lr=3e-5):
    print(f"\nTraining with layer-wise learning rates...")
    
    optimizer_grouped_parameters = [
        {'params': model.param_groups['embeddings'], 'lr': base_lr * 0.001},
        {'params': model.param_groups['early_layers'], 'lr': base_lr * 0.01},
        {'params': model.param_groups['middle_layers'], 'lr': base_lr * 0.1},
        {'params': model.param_groups['late_layers'], 'lr': base_lr},
        {'params': model.param_groups['classifier'], 'lr': base_lr * 2}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    best_acc = 0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_auroc': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.param_groups['embeddings'] + model.param_groups['early_layers'],
                max_norm=0.1
            )
            torch.nn.utils.clip_grad_norm_(
                model.param_groups['classifier'],
                max_norm=1.0
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        test_acc, test_auroc, test_f1 = evaluate_model(model, test_loader)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%, Test AUROC: {test_auroc:.2f}%")
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_auroc'].append(test_auroc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            print(f"  ✓ NEW BEST MODEL: {test_acc:.2f}%")
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with accuracy: {best_acc:.2f}%")
    
    return model, history, best_acc

# 5. Train the model
print("\n3. Starting training...")
start_time = time.time()

trained_model, history, best_acc = train_model_layerwise(
    model, 
    train_loader, 
    test_loader, 
    epochs=3,
    base_lr=3e-5
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# 6. Final evaluation
print("\n4. Final evaluation...")
final_acc, final_auroc, final_f1 = evaluate_model(trained_model, test_loader)
print(f"   Final Test Accuracy: {final_acc:.2f}%")
print(f"   Final Test AUROC: {final_auroc:.2f}%")

# 7. Save model
print("\n5. Saving model...")
save_dir = "/kaggle/working/dnabert2_trained"
os.makedirs(save_dir, exist_ok=True)

model_path = f"{save_dir}/dnabert2_vep_trained.pth"
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'best_accuracy': best_acc,
    'final_accuracy': final_acc,
    'final_auroc': final_auroc,
    'history': history
}, model_path)

print(f"✅ Model saved to: {model_path}")
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
