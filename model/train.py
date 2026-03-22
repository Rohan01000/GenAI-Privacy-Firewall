"""
train.py  —  Train the BiLSTM-CRF PII detector
================================================
Usage:
  python model/train.py

Reads  : data/pii_ner_dataset.csv
Writes : model/saved/  (model weights + vocabs + config)

Training takes ~5–10 min on CPU for 5 000 sentences / 10 epochs.
"""

import json
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.bilstm_crf import BiLSTMCRF
from model.data_loader import (
    PIIDataset, build_vocabs, collate_fn,
    load_sentences, save_vocabs,
)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    csv_path        = "data/pii_ner_dataset.csv",
    model_dir       = "model/saved",
    word_embed_dim  = 100,
    char_embed_dim  = 30,
    char_cnn_out    = 50,
    hidden_size     = 256,
    num_lstm_layers = 2,
    dropout         = 0.3,
    batch_size      = 32,
    epochs          = 10,
    lr              = 1e-3,
    val_split       = 0.1,
    seed            = 42,
)

torch.manual_seed(CFG["seed"])

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, label_vocab):
    id2label = {v: k for k, v in label_vocab.items()}
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for words, chars, labels, mask in loader:
            words, chars, mask = words.to(device), chars.to(device), mask.to(device)
            preds = model.predict(words, chars, mask)
            for i, pred_seq in enumerate(preds):
                L = int(mask[i].sum().item())
                gold = labels[i, :L].tolist()
                pred = pred_seq[:L]
                for g, p in zip(gold, pred):
                    g_lbl = id2label.get(g, "O")
                    p_lbl = id2label.get(p, "O")
                    if g_lbl != "O" and p_lbl != "O" and g_lbl == p_lbl:
                        tp += 1
                    elif g_lbl != "O" and p_lbl == "O":
                        fn += 1
                    elif g_lbl == "O" and p_lbl != "O":
                        fp += 1

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(CFG["model_dir"], exist_ok=True)

    print("📂  Loading dataset …")
    sentences = load_sentences(CFG["csv_path"])
    print(f"   {len(sentences)} sentences loaded")

    word_vocab, char_vocab, label_vocab = build_vocabs(sentences)
    save_vocabs(word_vocab, char_vocab, label_vocab, CFG["model_dir"])

    # Save model config
    model_cfg = {k: v for k, v in CFG.items()
                 if k not in ("csv_path", "model_dir", "batch_size",
                              "epochs", "lr", "val_split", "seed")}
    model_cfg.update({
        "vocab_size":      len(word_vocab),
        "char_vocab_size": len(char_vocab),
        "num_tags":        len(label_vocab),
    })
    with open(os.path.join(CFG["model_dir"], "model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    dataset = PIIDataset(sentences, word_vocab, char_vocab, label_vocab)
    val_size   = int(len(dataset) * CFG["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀  Training on {device}")

    model = BiLSTMCRF(
        vocab_size      = len(word_vocab),
        char_vocab_size = len(char_vocab),
        num_tags        = len(label_vocab),
        word_embed_dim  = CFG["word_embed_dim"],
        char_embed_dim  = CFG["char_embed_dim"],
        char_cnn_out    = CFG["char_cnn_out"],
        hidden_size     = CFG["hidden_size"],
        num_lstm_layers = CFG["num_lstm_layers"],
        dropout         = CFG["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_f1   = 0.0
    best_path = os.path.join(CFG["model_dir"], "best_model.pt")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for words, chars, labels, mask in train_loader:
            words, chars = words.to(device), chars.to(device)
            labels, mask = labels.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = model.loss(words, chars, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        prec, rec, f1 = evaluate(model, val_loader, device, label_vocab)
        scheduler.step(1 - f1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"loss={avg_loss:.4f}  "
            f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
            f"({elapsed:.1f}s)"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"   💾  Saved best model  (F1={f1:.4f})")

    print(f"\n✅  Training complete.  Best val F1 = {best_f1:.4f}")
    print(f"   Model saved → {best_path}")


if __name__ == "__main__":
    main()
