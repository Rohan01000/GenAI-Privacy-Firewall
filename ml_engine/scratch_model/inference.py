import re
import json
import torch
import numpy as np
from typing import List, Dict, Tuple

from ml_engine.scratch_model.model import BiLSTMCRF

class ScratchNERInference:
    def __init__(self, checkpoint_path="models/scratch_ner.pt", confidence_threshold=0.85):
        self._is_ready = False
        try:
            # 1. Load Vocabularies
            with open("models/vocab.json", "r") as f:
                vocab = json.load(f)
                self.word2idx = vocab["word2idx"]
                self.char2idx = vocab["char2idx"]

            with open("models/label_map.json", "r") as f:
                labels = json.load(f)
                self.label2idx = labels["label2idx"]
                self.idx2label = {int(k): v for k, v in labels["idx2label"].items()}

            # 2. Setup Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 3. Load Checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 4. Reconstruct Model
            self.model = BiLSTMCRF(
                vocab_size=checkpoint["vocab_size"],
                char_vocab_size=checkpoint["char_vocab_size"],
                num_labels=checkpoint["num_labels"],
                label2idx=self.label2idx
            )
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)

            self.confidence_threshold = confidence_threshold

            # 5. Warmup Pass
            word_ids = torch.zeros(1, 5, dtype=torch.long, device=self.device)
            char_ids = torch.zeros(1, 5, 10, dtype=torch.long, device=self.device)
            mask = torch.ones(1, 5, dtype=torch.bool, device=self.device)
            
            with torch.no_grad():
                self.model.predict(word_ids, char_ids, mask)
                
            print("Model warmed up and ready")
            self._is_ready = True
            
        except Exception as e:
            print(f"Failed to initialize ScratchNERInference: {e}")

    def tokenize(self, text: str) -> List[str]:
        # Exact tokenization match with dataset.py
        return re.findall(r"\w+|[^\w\s]", text)

    def text_to_indices(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unk_word = self.word2idx.get("<UNK>", 1)
        unk_char = self.char2idx.get("<UNK>", 1)

        # Word to index
        word_indices = [self.word2idx.get(t, unk_word) for t in tokens]
        
        # Char to index
        char_indices_list = [[self.char2idx.get(c, unk_char) for c in t] for t in tokens]
        
        # Pad chars to local batch max
        max_char_len = max([len(chars) for chars in char_indices_list]) if char_indices_list else 1
        
        char_padded = []
        for chars in char_indices_list:
            pad_len = max_char_len - len(chars)
            char_padded.append(chars + [0] * pad_len)

        # Convert to tensors and add batch dimension (unsqueeze)
        word_ids = torch.tensor([word_indices], dtype=torch.long, device=self.device)
        char_ids = torch.tensor([char_padded], dtype=torch.long, device=self.device)
        mask = torch.ones(1, len(tokens), dtype=torch.bool, device=self.device)

        return word_ids, char_ids, mask

    def detect_entities(self, text: str) -> List[dict]:
        if not text.strip():
            return []

        tokens = self.tokenize(text)
        if not tokens:
            return []

        word_ids, char_ids, mask = self.text_to_indices(tokens)

        with torch.no_grad():
            emissions = self.model.forward(word_ids, char_ids, mask)
            predictions = self.model.predict(word_ids, char_ids, mask)
            
            # Compute confidence scores
            probs = torch.softmax(emissions, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)

        pred_tags = [self.idx2label[idx] for idx in predictions[0]]
        confidences = max_probs[0].cpu().numpy().tolist()

        # Reconstruct original character positions
        token_spans = []
        current_char_idx = 0
        for token in tokens:
            start_idx = text.find(token, current_char_idx)
            end_idx = start_idx + len(token)
            token_spans.append((start_idx, end_idx))
            current_char_idx = end_idx

        # Parse BIO tags
        entities = []
        current_entity = None

        for i, (tag, conf) in enumerate(zip(pred_tags, confidences)):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                ent_type = tag[2:]
                start_char, end_char = token_spans[i]
                current_entity = {
                    "entity_type": ent_type,
                    "value": text[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "confidences": [conf]
                }
            elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag[2:]:
                _, end_char = token_spans[i]
                current_entity["end"] = end_char
                current_entity["value"] = text[current_entity["start"]:end_char]
                current_entity["confidences"].append(conf)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        # Filter by confidence and format output
        final_entities = []
        for ent in entities:
            avg_conf = sum(ent["confidences"]) / len(ent["confidences"])
            if avg_conf >= self.confidence_threshold:
                final_entities.append({
                    "entity_type": ent["entity_type"],
                    "value": ent["value"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "confidence": round(avg_conf, 4)
                })

        final_entities.sort(key=lambda x: x["start"])
        return final_entities

    def detect_entities_batch(self, texts: List[str]) -> List[List[dict]]:
        if not texts:
            return []

        all_tokens = [self.tokenize(text) for text in texts]
        
        # Determine batch dimensions
        max_seq_len = max((len(t) for t in all_tokens), default=1)
        if max_seq_len == 0: 
            max_seq_len = 1
            
        max_char_len = 1
        for tokens in all_tokens:
            for token in tokens:
                if len(token) > max_char_len:
                    max_char_len = len(token)

        unk_word = self.word2idx.get("<UNK>", 1)
        unk_char = self.char2idx.get("<UNK>", 1)

        batch_word_ids = []
        batch_char_ids = []
        batch_mask = []

        for tokens in all_tokens:
            word_indices = [self.word2idx.get(t, unk_word) for t in tokens]
            char_indices = [[self.char2idx.get(c, unk_char) for c in t] for t in tokens]
            
            # Pad chars
            char_padded = []
            for chars in char_indices:
                char_padded.append(chars + [0] * (max_char_len - len(chars)))
                
            # Pad sequences
            seq_pad_len = max_seq_len - len(tokens)
            word_indices.extend([0] * seq_pad_len)
            
            for _ in range(seq_pad_len):
                char_padded.append([0] * max_char_len)
                
            mask = [True] * len(tokens) + [False] * seq_pad_len
            
            batch_word_ids.append(word_indices)
            batch_char_ids.append(char_padded)
            batch_mask.append(mask)

        word_tensor = torch.tensor(batch_word_ids, dtype=torch.long, device=self.device)
        char_tensor = torch.tensor(batch_char_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(batch_mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            emissions = self.model.forward(word_tensor, char_tensor, mask_tensor)
            predictions = self.model.predict(word_tensor, char_tensor, mask_tensor)
            probs = torch.softmax(emissions, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)

        # Parse batch results back to texts
        results = []
        for b_idx, text in enumerate(texts):
            tokens = all_tokens[b_idx]
            if not tokens:
                results.append([])
                continue

            pred_tags = [self.idx2label[idx] for idx in predictions[b_idx]]
            confidences = max_probs[b_idx][:len(tokens)].cpu().numpy().tolist()

            entities = []
            current_entity = None
            token_spans = []
            current_char_idx = 0
            
            for token in tokens:
                start_idx = text.find(token, current_char_idx)
                end_idx = start_idx + len(token)
                token_spans.append((start_idx, end_idx))
                current_char_idx = end_idx

            for i, (tag, conf) in enumerate(zip(pred_tags, confidences)):
                if tag.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    start_char, end_char = token_spans[i]
                    current_entity = {
                        "entity_type": tag[2:],
                        "value": text[start_char:end_char],
                        "start": start_char,
                        "end": end_char,
                        "confidences": [conf]
                    }
                elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag[2:]:
                    _, end_char = token_spans[i]
                    current_entity["end"] = end_char
                    current_entity["value"] = text[current_entity["start"]:end_char]
                    current_entity["confidences"].append(conf)
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            final_entities = []
            for ent in entities:
                avg_conf = sum(ent["confidences"]) / len(ent["confidences"])
                if avg_conf >= self.confidence_threshold:
                    final_entities.append({
                        "entity_type": ent["entity_type"],
                        "value": ent["value"],
                        "start": ent["start"],
                        "end": ent["end"],
                        "confidence": round(avg_conf, 4)
                    })
                    
            final_entities.sort(key=lambda x: x["start"])
            results.append(final_entities)

        return results

    def is_ready(self) -> bool:
        return self._is_ready

if __name__ == "__main__":
    # Test Block
    print("Initializing inference module...")
    inference = ScratchNERInference(confidence_threshold=0.0) # Lowered for testing untrained dummy outputs
    
    test_sentences = [
        "Can you help debug this?",
        "Contact John Smith at john@acme.com about the project",
        "My AWS key is AKIAIOSFODNN7EXAMPLE and it keeps failing",
        "The account SSN 123-45-6789 belongs to Sarah Johnson"
    ]
    
    for sentence in test_sentences:
        print(f"\nINPUT: {sentence}")
        print("ENTITIES FOUND:")
        
        entities = inference.detect_entities(sentence)
        if not entities:
            print("  (None)")
        else:
            for ent in entities:
                print(f"  - {ent['entity_type']}: \"{ent['value']}\" (pos {ent['start']}-{ent['end']}, confidence: {ent['confidence']:.2f})")
                
    print("\ninference.py ready for integration")