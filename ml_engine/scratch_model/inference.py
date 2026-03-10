import torch
import json
import os
import re
from .model import ScratchNERModel

class ScratchDetector:
    def __init__(self, model_dir="models/scratch"):
        self.model_dir = model_dir
        self.model = None
        self.vocab = {}
        self.idx2tag = {}
        
    def load(self):
        if not os.path.exists(os.path.join(self.model_dir, "weights.pt")):
            raise FileNotFoundError(f"Scratch model weights not found in {self.model_dir}")
            
        with open(os.path.join(self.model_dir, "vocab.json"), "r") as f:
            self.vocab = json.load(f)
        with open(os.path.join(self.model_dir, "tag2idx.json"), "r") as f:
            tag2idx = json.load(f)
            self.idx2tag = {v: k for k, v in tag2idx.items()}
            
        self.model = ScratchNERModel(vocab_size=len(self.vocab), num_classes=len(tag2idx))
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "weights.pt")))
        self.model.eval()
        
    def detect(self, text: str, threshold: float = 0.5):
        if not self.model:
            return []
            
        # Basic tokenization
        words = re.findall(r'\b\w+\b|\S', text)
        word_indices = [self.vocab.get(w.lower(), self.vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor([word_indices], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
        entities = []
        current_idx = 0
        
        for i, word in enumerate(words):
            tag = self.idx2tag[preds[0][i].item()]
            score = max_probs[0][i].item()
            
            # Find word span in original text
            start = text.find(word, current_idx)
            end = start + len(word)
            current_idx = end
            
            if tag != "O" and score >= threshold:
                # Strip B- or I- prefix
                entity_group = tag[2:] if len(tag) > 2 else tag
                entities.append({
                    "entity_group": entity_group,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end
                })
                
        return entities