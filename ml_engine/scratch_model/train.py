import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os

from .model import ScratchNERModel
from .dataset import NERDataset

def train_scratch():
    # Dummy data for demonstration
    texts = ["John lives in Boston", "Alice works at Google"]
    tags = [["B-PER", "O", "O", "B-LOC"], ["B-PER", "O", "O", "B-ORG"]]
    
    vocab = {"<PAD>": 0, "<UNK>": 1, "john": 2, "lives": 3, "in": 4, "boston": 5, "alice": 6, "works": 7, "at": 8, "google": 9}
    tag2idx = {"O": 0, "B-PER": 1, "B-LOC": 2, "B-ORG": 3}
    
    dataset = NERDataset(texts, tags, vocab, tag2idx)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = ScratchNERModel(vocab_size=len(vocab), num_classes=len(tag2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            # Flatten to compute loss
            active_loss = batch["mask"].view(-1)
            active_logits = logits.view(-1, len(tag2idx))[active_loss]
            active_labels = batch["labels"].view(-1)[active_loss]
            
            loss = criterion(active_logits, active_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss}")

    # Save logic
    os.makedirs("models/scratch", exist_ok=True)
    torch.save(model.state_dict(), "models/scratch/weights.pt")
    with open("models/scratch/vocab.json", "w") as f:
        json.dump(vocab, f)
    with open("models/scratch/tag2idx.json", "w") as f:
        json.dump(tag2idx, f)
    print("Scratch model saved to models/scratch/")

if __name__ == "__main__":
    train_scratch()