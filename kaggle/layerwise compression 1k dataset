import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
import time
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*70)
print("🎯 LAYER ABLATION & COMPRESSION STUDY")
print("="*70)

# ===================================================
# 1. CREATE PRUNABLE MODEL
# ===================================================
print("\n1. Creating DNABERT-2 model with layer removal capability...")

class DNABERT2Prunable(nn.Module):
    """Model that allows layer removal for ablation study"""
    def __init__(self):
        super().__init__()
        # Load base model WITHOUT trust_remote_code
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
    
    def remove_layers(self, layers_to_remove):
        """Remove specific encoder layers"""
        if not layers_to_remove:
            return self
        
        # Get current layers
        current_layers = list(self.bert.encoder.layer)
        
        # Remove in reverse order
        for layer_idx in sorted(layers_to_remove, reverse=True):
            if 0 <= layer_idx < len(current_layers):
                current_layers.pop(layer_idx)
        
        # Update with fewer layers
        self.bert.encoder.layer = nn.ModuleList(current_layers)
        print(f"  Removed layers {layers_to_remove}. Now {len(current_layers)} layers.")
        return self

# ===================================================
# 2. LOAD YOUR TRAINED WEIGHTS
# ===================================================
print("\n2. Loading your trained model weights...")

# Try to load the saved weights
try:
    checkpoint = torch.load("/kaggle/working/dnabert2_2epochs/model.pth", map_location=device)
    print(f"✅ Weights loaded successfully")
    
    # Create model and try to load weights
    model = DNABERT2Prunable().to(device)
    
    # Load with strict=False to handle any mismatches
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Model weights loaded (some mismatches ignored)")
    
except Exception as e:
    print(f"❌ Error loading: {e}")
    print("\nCreating fresh model for ablation study...")
    model = DNABERT2Prunable().to(device)
    print(f"✅ Created fresh model")

# ===================================================
# 3. CREATE DATASET
# ===================================================
print("\n3. Creating test dataset...")

# Load tokenizer WITHOUT trust_remote_code
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

class DNADataset(Dataset):
    def __init__(self, num_samples=500, seq_length=256):
        self.sequences = []
        self.labels = []
        bases = ['A', 'C', 'G', 'T']
        
        for i in range(num_samples):
            if i < num_samples // 2:  # Benign
                seq = ''.join(np.random.choice(bases, seq_length))
                # Add benign pattern
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "AAAAAA" + seq[pos+6:]
                label = 0
            else:  # Pathogenic
                seq = ''.join(np.random.choice(bases, seq_length))
                # Add pathogenic pattern
                pos = np.random.randint(0, seq_length - 6)
                seq = seq[:pos] + "TATAAA" + seq[pos+6:]
                label = 1
            
            self.sequences.append(seq)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = " " + self.sequences[idx]
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

# Create datasets
train_dataset = DNADataset(num_samples=400, seq_length=256)
val_dataset = DNADataset(num_samples=100, seq_length=256)
test_dataset = DNADataset(num_samples=200, seq_length=256)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# ===================================================
# 4. EVALUATION FUNCTIONS
# ===================================================
def evaluate_model(model, data_loader):
    """Evaluate model performance"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    auroc = roc_auc_score(all_labels, all_probs) * 100 if len(set(all_labels)) > 1 else 50.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    return accuracy, auroc, f1

def quick_finetune(model, train_loader, val_loader, epochs=2, lr=1e-5):
    """Quick fine-tuning after layer removal"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        # Training
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        model.train()
    
    return model

# ===================================================
# 5. RUN LAYER ABLATION STUDY
# ===================================================
print("\n4. Running layer ablation study...")

# First, get baseline performance
print("\nEvaluating baseline model (12 layers)...")
baseline_acc, baseline_auroc, baseline_f1 = evaluate_model(model, test_loader)
print(f"Baseline: Acc={baseline_acc:.2f}%, AUROC={baseline_auroc:.2f}%")

# Test removing each layer
print("\nTesting layer importance...")
layer_results = []

for layer_idx in range(12):  # DNABERT-2 has 12 layers
    print(f"\n🔬 Testing without layer {layer_idx}...")
    
    # Create fresh model
    test_model = DNABERT2Prunable().to(device)
    test_model.load_state_dict(model.state_dict(), strict=False)
    
    # Remove the layer
    test_model.remove_layers([layer_idx])
    
    # Quick fine-tune
    test_model = quick_finetune(test_model, train_loader, val_loader, epochs=1, lr=1e-5)
    
    # Evaluate
    acc, auroc, f1 = evaluate_model(test_model, test_loader)
    
    # Calculate performance change
    acc_change = acc - baseline_acc
    auroc_change = auroc - baseline_auroc
    
    layer_results.append({
        'layer': layer_idx,
        'accuracy': acc,
        'auroc': auroc,
        'acc_change': acc_change,
        'auroc_change': auroc_change
    })
    
    print(f"  Acc={acc:.2f}% (Δ={acc_change:+.2f}%), AUROC={auroc:.2f}%")

# ===================================================
# 6. IDENTIFY LAYER TYPES
# ===================================================
print("\n" + "="*60)
print("Layer Classification Results")
print("="*60)

cornerstone_layers = []
unfavourable_layers = []
neutral_layers = []

print("\nLayer analysis:")
for result in layer_results:
    layer_idx = result['layer']
    acc_change = result['acc_change']
    
    # Paper's criteria
    if acc_change <= -5.0:  # More than 5% drop
        cornerstone_layers.append(layer_idx)
        print(f"Layer {layer_idx}: CORNERSTONE (Δ={acc_change:.2f}%)")
    elif acc_change >= 0:  # No drop or improvement
        unfavourable_layers.append(layer_idx)
        print(f"Layer {layer_idx}: UNFAVOURABLE (Δ={acc_change:+.2f}%)")
    else:  # Small drop (0 to -5%)
        neutral_layers.append(layer_idx)
        print(f"Layer {layer_idx}: NEUTRAL (Δ={acc_change:.2f}%)")

print(f"\n📊 Summary:")
print(f"Cornerstone layers: {sorted(cornerstone_layers)}")
print(f"Unfavourable layers: {sorted(unfavourable_layers)}")
print(f"Neutral layers: {sorted(neutral_layers)}")

# ===================================================
# 7. CREATE PRUNED MODELS
# ===================================================
print("\n" + "="*60)
print("Creating Pruned Models")
print("="*60)

# Model 1: Remove unfavourable layers
if unfavourable_layers:
    print(f"\n📦 Creating model without unfavourable layers: {unfavourable_layers}")
    
    model1 = DNABERT2Prunable().to(device)
    model1.load_state_dict(model.state_dict(), strict=False)
    model1.remove_layers(unfavourable_layers)
    
    # Fine-tune
    model1 = quick_finetune(model1, train_loader, val_loader, epochs=2, lr=2e-5)
    
    # Evaluate
    acc1, auroc1, f1_1 = evaluate_model(model1, test_loader)
    
    print(f"  Results: Acc={acc1:.2f}%, AUROC={auroc1:.2f}%")
    print(f"  Layers: {12 - len(unfavourable_layers)} remaining")

# Model 2: Aggressive pruning - keep only essential
print(f"\n📦 Creating aggressively pruned model...")
# Keep layers 0-3, 10-11 (common pattern from papers)
keep_layers = [0, 1, 2, 3, 10, 11]
remove_layers = [i for i in range(12) if i not in keep_layers]

model2 = DNABERT2Prunable().to(device)
model2.load_state_dict(model.state_dict(), strict=False)
model2.remove_layers(remove_layers)

# Fine-tune
model2 = quick_finetune(model2, train_loader, val_loader, epochs=3, lr=3e-5)

# Evaluate
acc2, auroc2, f1_2 = evaluate_model(model2, test_loader)

print(f"  Results: Acc={acc2:.2f}%, AUROC={auroc2:.2f}%")
print(f"  Layers: {len(keep_layers)} remaining (reduced from 12)")

# ===================================================
# 8. COMPRESSION ANALYSIS
# ===================================================
print("\n" + "="*60)
print("Compression Performance")
print("="*60)

print(f"\nBaseline (12 layers):")
print(f"  Accuracy: {baseline_acc:.2f}%")
print(f"  AUROC: {baseline_auroc:.2f}%")

print(f"\nPruned Model ({len(keep_layers)} layers):")
print(f"  Accuracy: {acc2:.2f}% (Δ={acc2 - baseline_acc:+.2f}%)")
print(f"  AUROC: {auroc2:.2f}% (Δ={auroc2 - baseline_auroc:+.2f}%)")

# Calculate compression ratio
compression_ratio = (12 - len(keep_layers)) / 12 * 100
print(f"\nCompression: {compression_ratio:.1f}% fewer layers")

# ===================================================
# 9. SAVE RESULTS
# ===================================================
print("\n" + "="*60)
print("Saving Results")
print("="*60)

save_dir = "/kaggle/working/ablation_results"
os.makedirs(save_dir, exist_ok=True)

# Save model
model_path = f"{save_dir}/pruned_model_{len(keep_layers)}layers.pth"
torch.save(model2.state_dict(), model_path)
print(f"✅ Saved pruned model to {model_path}")

# Save results
results = {
    'baseline': {
        'accuracy': float(baseline_acc),
        'auroc': float(baseline_auroc),
        'layers': 12
    },
    'layer_analysis': layer_results,
    'layer_classification': {
        'cornerstone': cornerstone_layers,
        'unfavourable': unfavourable_layers,
        'neutral': neutral_layers
    },
    'pruned_model': {
        'layers_kept': keep_layers,
        'layers_removed': remove_layers,
        'total_layers': len(keep_layers),
        'accuracy': float(acc2),
        'auroc': float(auroc2),
        'compression_ratio': float(compression_ratio)
    }
}

results_path = f"{save_dir}/ablation_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved results to {results_path}")

print(f"\n" + "="*70)
print("🎉 ABLATION STUDY COMPLETE!")
print("="*70)
print(f"✅ Tested all 12 layers individually")
print(f"✅ Identified {len(cornerstone_layers)} cornerstone layers")
print(f"✅ Created pruned model with {len(keep_layers)} layers")
print(f"✅ Compression: {compression_ratio:.1f}% fewer layers")
print(f"✅ Results saved to: {save_dir}/")
print("="*70)
