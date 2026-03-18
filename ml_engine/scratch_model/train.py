import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics import precision_score, recall_score

from ml_engine.scratch_model.dataset import create_dataloaders
from ml_engine.scratch_model.model import BiLSTMCRF

# ==========================================
# 1. SETUP AND REPRODUCIBILITY
# ==========================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ==========================================
# 2. VALIDATION FUNCTION
# ==========================================
def validate(model, val_loader, device, idx2label):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in val_loader:
            word_tensor, char_tensor, label_tensor, orig_tokens = batch
            
            word_tensor = word_tensor.to(device)
            char_tensor = char_tensor.to(device)
            label_tensor = label_tensor.to(device)
            
            mask = (label_tensor != 0)
            
            # Compute Loss
            emissions = model(word_tensor, char_tensor, mask)
            loss = model.neg_log_likelihood(emissions, label_tensor, mask)
            total_loss += loss.item()
            
            # Compute Predictions
            batch_preds = model.predict(word_tensor, char_tensor, mask)
            
            # Align Predictions with True Labels (excluding PAD)
            batch_size = label_tensor.size(0)
            seq_lens = mask.sum(dim=1).long().tolist()
            
            for i in range(batch_size):
                length = seq_lens[i]
                
                # Get true label indices for this sequence
                true_idx = label_tensor[i, :length].tolist()
                pred_idx = batch_preds[i]
                
                # Ensure lengths match
                min_len = min(len(true_idx), len(pred_idx))
                
                # Convert to string labels
                true_tags = [idx2label[idx] for idx in true_idx[:min_len]]
                pred_tags = [idx2label[idx] for idx in pred_idx[:min_len]]
                
                all_trues.append(true_tags)
                all_preds.append(pred_tags)

    avg_loss = total_loss / len(val_loader)
    
    # Calculate Metrics using seqeval
    # seqeval expects lists of lists of strings
    try:
        overall_p = precision_score(all_trues, all_preds)
        overall_r = recall_score(all_trues, all_preds)
        overall_f1 = f1_score(all_trues, all_preds)
        
        # Parse classification report for per-entity metrics
        report_str = classification_report(all_trues, all_preds)
        
        per_entity_f1 = {}
        # Simple parsing of the classification report string
        lines = report_str.split('\n')
        for line in lines[2:-4]: # Skip headers and summaries
            if not line.strip(): continue
            parts = line.split()
            if len(parts) >= 4:
                ent_type = parts[0]
                try:
                    ent_f1 = float(parts[3])
                    per_entity_f1[ent_type] = ent_f1
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"Warning: seqeval metric calculation failed ({e}). Returning 0s.")
        overall_p, overall_r, overall_f1 = 0.0, 0.0, 0.0
        per_entity_f1 = {}

    model.train()
    return avg_loss, overall_p, overall_r, overall_f1, per_entity_f1

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================
def plot_curves(history, output_dir="models"):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    
    # Plot 1: Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', color='orange', marker='s')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    
    # Plot 2: F1 Curve
    plt.figure(figsize=(12, 8))
    
    # Track overall F1
    overall_f1 = [h["val_f1"] for h in history]
    plt.plot(epochs, overall_f1, label='Overall Macro F1', color='black', linewidth=3, zorder=10)
    
    # Collect all unique entities seen across history
    all_entities = set()
    for h in history:
        all_entities.update(h.get("per_entity_f1", {}).keys())
        
    for ent in all_entities:
        ent_f1_scores = [h.get("per_entity_f1", {}).get(ent, 0.0) for h in history]
        plt.plot(epochs, ent_f1_scores, label=ent, alpha=0.7, marker='.')
        
    plt.title("Validation F1 Per Entity Type Over Training")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.ylim(-0.05, 1.05)
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"))
    plt.close()

# ==========================================
# 4. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("\nInitializing Training Pipeline...")
    
    # 1. Setup
    set_seeds(42)
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs("models", exist_ok=True)
    
    # 2. Hyperparameters
    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "max_epochs": 50,
        "patience": 7,
        "batch_size": 32,
        "hidden_size": 256,
        "dropout": 0.3,
        "grad_clip": 5.0,
        "device": str(device)
    }
    with open("models/training_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # 3. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, full_dataset, word2idx, label2idx = create_dataloaders(batch_size=config["batch_size"])
    
    # Load character vocab generated by dataset.py
    with open("models/vocab.json", "r") as f:
        vocab_data = json.load(f)
        char2idx = vocab_data["char2idx"]
        
    # Reverse label mapping
    idx2label = {v: k for k, v in label2idx.items()}
    
    # 4. Initialize Model
    print("Building model...")
    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        char_vocab_size=len(char2idx),
        num_labels=len(label2idx),
        label2idx=label2idx,
        hidden_size=config["hidden_size"],
        dropout=config["dropout"]
    )
    
    model.count_parameters()
    model = model.to(device)
    
    # 5. Optimizers and Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)    
    # 6. Training Loop Variables
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    
    print("\nStarting Training Loop...\n")
    
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            word_tensor, char_tensor, label_tensor, _ = batch
            
            word_tensor = word_tensor.to(device)
            char_tensor = char_tensor.to(device)
            label_tensor = label_tensor.to(device)
            
            mask = (label_tensor != 0)
            
            optimizer.zero_grad()
            
            emissions = model(word_tensor, char_tensor, mask)
            loss = model.neg_log_likelihood(emissions, label_tensor, mask)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATE ---
        val_loss, val_p, val_r, val_f1, per_ent_f1 = validate(model, val_loader, device, idx2label)
        
        # Update Scheduler
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        # Save History
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_precision": val_p,
            "val_recall": val_r,
            "per_entity_f1": per_ent_f1,
            "lr": current_lr,
            "epoch_time_seconds": epoch_time
        })
        
        with open("models/training_history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        # --- PRINT EPOCH SUMMARY ---
        print("╔" + "═" * 54 + "╗")
        print(f"║ Epoch {epoch:>2}/{config['max_epochs']} | Time: {epoch_time:>5.1f}s | LR: {current_lr:.6f}          ║")
        print("╠" + "═" * 54 + "╣")
        print(f"║ Train Loss: {avg_train_loss:.4f}    Val Loss: {val_loss:.4f}               ║")
        print(f"║ Val Precision: {val_p:.3f}  Val Recall: {val_r:.3f}  F1: {val_f1:.3f} ║")
        print("╠" + "═" * 54 + "╣")
        print("║ Per-Entity F1:                                       ║")
        
        for ent, f1 in sorted(per_ent_f1.items()):
            print(f"║   {ent:<15}: {f1:.3f}                               ║")
        print("╚" + "═" * 54 + "╝")
        
        # --- CHECKPOINTING & EARLY STOPPING ---
        if val_f1 > best_val_f1:
            print("★ New best model saved!")
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_val_f1,
                'vocab_size': len(word2idx),
                'char_vocab_size': len(char2idx),
                'num_labels': len(label2idx),
                'label2idx': label2idx,
                'idx2label': idx2label
            }, "models/scratch_ner.pt")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= config["patience"]:
            print(f"\nEarly stopping triggered after {epoch} epochs (Patience: {config['patience']}).")
            break
            
    # --- END OF TRAINING ---
    print("\nGenerating training curves...")
    plot_curves(history, output_dir="models")
    
    print("╔" + "═" * 42 + "╗")
    print("║          TRAINING COMPLETE               ║")
    print("╠" + "═" * 42 + "╣")
    print(f"║ Best Epoch     : {best_epoch:<24}║")
    print(f"║ Best Val F1    : {best_val_f1:<.3f}                     ║")
    print("║ Model saved to : models/scratch_ner.pt   ║")
    print("║ Loss curve     : models/loss_curve.png   ║")
    print("║ F1 curve       : models/f1_curve.png     ║")
    print("╚" + "═" * 42 + "╝\n")

if __name__ == "__main__":
    main()