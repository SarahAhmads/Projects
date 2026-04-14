import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import ASTForAudioClassification, ASTConfig
import onnx
import onnxruntime as ort

class AudioDataset(Dataset):
    def __init__(self, df, label_to_idx):
        self.specs = df['spec'].tolist()
        self.labels = df['label'].tolist()
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        # Normalize spec
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        
        # AST expects (time_frames, n_mels)
        # librosa output is (n_mels, time_frames), let's transpose
        spec = spec.T
        
        label = self.label_to_idx[self.labels[idx]]
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(specs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs).logits
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def export_to_onnx(model, dummy_input, path):
    print(f"Exporting model to ONNX at {path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path("data/processed/dataset.pkl")
    
    if not data_path.exists():
        print("Processed dataset not found. Run preprocess_data.py first.")
        # For demonstration, we'll assume it exists or exit gracefully
        return

    df = pd.read_pickle(data_path)
    labels = sorted(df['label'].unique())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    num_labels = len(labels)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_ds = AudioDataset(train_df, label_to_idx)
    val_ds = AudioDataset(val_df, label_to_idx)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Initialize AST Model (Fine-tuning)
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593", 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    epochs = 10
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # Export to ONNX
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    dummy_input = torch.randn(1, 1024, 128).to(device) # (batch, time, freq)
    export_to_onnx(model, dummy_input, "model.onnx")

if __name__ == "__main__":
    main()
