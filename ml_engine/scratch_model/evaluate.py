import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics import precision_score, recall_score

from ml_engine.scratch_model.dataset import create_dataloaders
from ml_engine.scratch_model.model import BiLSTMCRF

# ==========================================
# 1. LOAD MODEL FUNCTION
# ==========================================
def load_model(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    vocab_size = checkpoint['vocab_size']
    char_vocab_size = checkpoint['char_vocab_size']
    num_labels = checkpoint['num_labels']
    label2idx = checkpoint['label2idx']
    
    # Ensure keys are integers in idx2label
    idx2label = {int(k): v for k, v in checkpoint['idx2label'].items()}
    
    model = BiLSTMCRF(
        vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_labels=num_labels,
        label2idx=label2idx
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully (Epoch {checkpoint['epoch']}, Best F1: {checkpoint['best_f1']:.4f})")
    return model, idx2label, label2idx

# ==========================================
# 2. RUN INFERENCE ON TEST SET
# ==========================================
def get_predictions(model, test_loader, device, idx2label):
    all_true_tags = []
    all_pred_tags = []
    all_tokens = []
    
    with torch.no_grad():
        for batch in test_loader:
            word_tensor, char_tensor, label_tensor, orig_tokens = batch
            
            word_tensor = word_tensor.to(device)
            char_tensor = char_tensor.to(device)
            label_tensor = label_tensor.to(device)
            
            mask = (label_tensor != 0)
            
            # Get Viterbi predictions
            batch_preds = model.predict(word_tensor, char_tensor, mask)
            
            batch_size = label_tensor.size(0)
            seq_lens = mask.sum(dim=1).long().tolist()
            
            for i in range(batch_size):
                length = seq_lens[i]
                
                true_idx = label_tensor[i, :length].tolist()
                pred_idx = batch_preds[i]
                
                # Truncate to min length just in case
                min_len = min(len(true_idx), len(pred_idx))
                
                true_tags = [idx2label[idx] for idx in true_idx[:min_len]]
                pred_tags = [idx2label[idx] for idx in pred_idx[:min_len]]
                
                # orig_tokens contains lists of strings per batch
                tokens = orig_tokens[i][:min_len]
                
                all_true_tags.append(true_tags)
                all_pred_tags.append(pred_tags)
                all_tokens.append(tokens)
                
    return all_true_tags, all_pred_tags, all_tokens

# ==========================================
# 3. FULL METRICS REPORT
# ==========================================
def compute_metrics(all_true_tags, all_pred_tags):
    print("\n" + "="*50)
    print(" SEQEVAL CLASSIFICATION REPORT")
    print("="*50)
    
    # 1. Seqeval Metrics
    report_str = classification_report(all_true_tags, all_pred_tags)
    print(report_str)
    
    macro_f1 = f1_score(all_true_tags, all_pred_tags, average='macro')
    micro_f1 = f1_score(all_true_tags, all_pred_tags, average='micro')
    weighted_f1 = f1_score(all_true_tags, all_pred_tags, average='weighted')
    
    # Parse entity-level metrics from the report string
    entity_metrics = {}
    lines = report_str.split('\n')
    for line in lines[2:-4]:
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 5:
            ent_type = parts[0]
            entity_metrics[ent_type] = {
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1": float(parts[3]),
                "support": int(parts[4])
            }
            
    # 2. Token-level Sklearn Metrics
    flat_true = [tag for seq in all_true_tags for tag in seq]
    flat_pred = [tag for seq in all_pred_tags for tag in seq]
    
    token_acc = accuracy_score(flat_true, flat_pred)
    conf_matrix = confusion_matrix(flat_true, flat_pred)
    
    print(f"\nToken-level Accuracy: {token_acc:.4f}")
    
    return {
        "entity_metrics": entity_metrics,
        "overall_metrics": {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "token_accuracy": token_acc
        },
        "confusion_matrix": conf_matrix.tolist()
    }

# ==========================================
# 4. ERROR ANALYSIS
# ==========================================
def analyze_errors(all_true_tags, all_pred_tags, all_tokens):
    fp_examples = []
    fn_examples = []
    wt_examples = []
    
    total_fp = 0
    total_fn = 0
    total_wt = 0
    
    for tokens, true_seq, pred_seq in zip(all_tokens, all_true_tags, all_pred_tags):
        has_fp = False
        has_fn = False
        has_wt = False
        
        for t, p in zip(true_seq, pred_seq):
            if t == 'O' and p != 'O': 
                has_fp = True
                total_fp += 1
            elif t != 'O' and p == 'O': 
                has_fn = True
                total_fn += 1
            elif t != 'O' and p != 'O' and t[2:] != p[2:]: 
                has_wt = True
                total_wt += 1
                
        if has_fp and len(fp_examples) < 5:
            fp_examples.append((tokens, pred_seq, true_seq))
        if has_fn and len(fn_examples) < 5:
            fn_examples.append((tokens, pred_seq, true_seq))
        if has_wt and len(wt_examples) < 5:
            wt_examples.append((tokens, pred_seq, true_seq))
            
    print("\n" + "="*50)
    print(" ERROR ANALYSIS EXAMPLES")
    print("="*50)
    
    def print_examples(title, examples):
        print(f"\n--- {title} ---")
        for i, (toks, preds, trues) in enumerate(examples, 1):
            print(f"Example {i}:")
            print(f"  Tokens : {' '.join(toks)}")
            print(f"  Preds  : {' '.join(preds)}")
            print(f"  Trues  : {' '.join(trues)}\n")

    print_examples("Type 1 — False Positives (Predicted Entity, True is O)", fp_examples)
    print_examples("Type 2 — False Negatives (Predicted O, True is Entity)", fn_examples)
    print_examples("Type 3 — Wrong Entity Type (Predicted X, True is Y)", wt_examples)
    
    return {
        "false_positives_count": total_fp,
        "false_negatives_count": total_fn,
        "wrong_type_count": total_wt,
        "examples": {
            "false_positives": [{"tokens": t, "preds": p, "trues": tr} for t, p, tr in fp_examples],
            "false_negatives": [{"tokens": t, "preds": p, "trues": tr} for t, p, tr in fn_examples],
            "wrong_type": [{"tokens": t, "preds": p, "trues": tr} for t, p, tr in wt_examples]
        }
    }

# ==========================================
# 5. CONFIDENCE ANALYSIS
# ==========================================
def analyze_confidence(model, test_loader, device):
    correct_confidences = []
    incorrect_confidences = []
    
    all_true_tags_idx = []
    all_pred_probs = []
    all_masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            word_tensor, char_tensor, label_tensor, _ = batch
            word_tensor, char_tensor, label_tensor = word_tensor.to(device), char_tensor.to(device), label_tensor.to(device)
            mask = (label_tensor != 0)
            
            # Raw emission scores
            emissions = model.forward(word_tensor, char_tensor, mask)
            
            # Compute token-level confidence using softmax
            probs = torch.softmax(emissions, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
            # Flatten arrays for mask filtering
            flat_mask = mask.view(-1)
            flat_labels = label_tensor.view(-1)[flat_mask]
            flat_preds = preds.view(-1)[flat_mask]
            flat_probs = max_probs.view(-1)[flat_mask]
            
            # Collect correct vs incorrect
            correct_mask = (flat_preds == flat_labels)
            incorrect_mask = (flat_preds != flat_labels)
            
            correct_confidences.extend(flat_probs[correct_mask].cpu().numpy().tolist())
            incorrect_confidences.extend(flat_probs[incorrect_mask].cpu().numpy().tolist())
            
            all_true_tags_idx.append(label_tensor)
            all_pred_probs.append(probs)
            all_masks.append(mask)

    
    # Plot Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, bins=50, alpha=0.5, color='blue', label='Correct Predictions')
    plt.hist(incorrect_confidences, bins=50, alpha=0.5, color='red', label='Incorrect Predictions')
    plt.title("Confidence Score Distribution: Correct vs Incorrect")
    plt.xlabel("Confidence Score (0.0 to 1.0)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/confidence_analysis.png")
    plt.close()
    
    print("\n" + "="*50)
    print(" CONFIDENCE THRESHOLD SEARCH")
    print("="*50)
    
    best_f1 = 0.0
    optimal_threshold = 0.5
    
    # We'll use the pre-loaded idx2label directly from the model inside a local search
    checkpoint = torch.load("models/scratch_ner.pt", map_location=device, weights_only=False)
    idx2label = {int(k): v for k, v in checkpoint['idx2label'].items()}
    o_index = checkpoint['label2idx']["O"]
    
    # Evaluate thresholds from 0.5 to 0.99
    thresholds = np.arange(0.5, 1.0, 0.01)
    for t in thresholds:
        t_true_tags = []
        t_pred_tags = []
        
        for labels, probs, mask in zip(all_true_tags_idx, all_pred_probs, all_masks):
            max_p, preds = torch.max(probs, dim=-1)
            
            # Apply threshold: If confidence < T, predict 'O'
            preds = torch.where(max_p < t, torch.tensor(o_index, device=device), preds)
            
            batch_size = labels.size(0)
            seq_lens = mask.sum(dim=1).long().tolist()
            
            for i in range(batch_size):
                length = seq_lens[i]
                true_idx = labels[i, :length].tolist()
                pred_idx = preds[i, :length].tolist()
                
                t_true_tags.append([idx2label[idx] for idx in true_idx])
                t_pred_tags.append([idx2label[idx] for idx in pred_idx])
                
        current_f1 = f1_score(t_true_tags, t_pred_tags, average='macro')
        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = float(t)
            
    print(f"Optimal Confidence Threshold: {optimal_threshold:.2f} (Macro F1: {best_f1:.4f})")
    
    return optimal_threshold

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\nStarting Evaluation Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = "models/scratch_ner.pt"
    
    # 1. Load Model
    model, idx2label, label2idx = load_model(checkpoint_path, device)
    
    # 2. Load Test Data
    print("Loading test dataset...")
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    _, _, test_loader, _, _, _ = create_dataloaders(batch_size=32)
    test_set_size = len(test_loader.dataset)
    
    # 3. Get Predictions
    print("Running inference on test set...")
    all_true_tags, all_pred_tags, all_tokens = get_predictions(model, test_loader, device, idx2label)
    
    # 4. Compute Metrics
    metrics_report = compute_metrics(all_true_tags, all_pred_tags)
    
    # 5. Analyze Errors
    error_report = analyze_errors(all_true_tags, all_pred_tags, all_tokens)
    
    # 6. Analyze Confidence
    optimal_threshold = analyze_confidence(model, test_loader, device)
    
    # 7. Save JSON Report
    evaluation_report = {
        "model": "BiLSTM-CRF",
        "checkpoint": checkpoint_path,
        "test_set_size": test_set_size,
        "entity_metrics": metrics_report["entity_metrics"],
        "overall_metrics": metrics_report["overall_metrics"],
        "optimal_confidence_threshold": optimal_threshold,
        "error_analysis": {
            "false_positives_count": error_report["false_positives_count"],
            "false_negatives_count": error_report["false_negatives_count"],
            "wrong_type_count": error_report["wrong_type_count"]
        }
    }
    
    os.makedirs("models", exist_ok=True)
    report_path = "models/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)
        
    print("\n" + "="*50)
    print(f"Evaluation complete. Report saved to {report_path}")
    print("="*50 + "\n")