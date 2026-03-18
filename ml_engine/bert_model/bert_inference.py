import torch
import json
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BertNERInference:
    def __init__(self, model_path="models/bert_ner", confidence_threshold=0.85):
        self._is_ready = False
        try:
            # 1. Load Vocabularies
            map_path = os.path.join(model_path, "label_map.json")
            with open(map_path, "r") as f:
                labels = json.load(f)
                self.label2idx = labels["label2idx"]
                self.idx2label = {int(k): v for k, v in labels["idx2label"].items()}

            # 2. Setup Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 3. Load Model and Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            self.model.eval()
            self.model.to(self.device)

            self.confidence_threshold = confidence_threshold

            # 4. Warmup Pass
            with torch.no_grad():
                dummy_input = self.tokenizer("warmup", return_tensors="pt").to(self.device)
                self.model(**dummy_input)
                
            print("BERT model warmed up and ready")
            self._is_ready = True
            
        except Exception as e:
            print(f"Failed to initialize BertNERInference: {e}")

    def detect_entities(self, text: str) -> List[dict]:
        if not text.strip():
            return []

        # Tokenize with offsets to map back to original characters
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        
        confidences = max_probs[0].cpu().numpy().tolist()
        pred_indices = preds[0].cpu().numpy().tolist()
        
        # WordPiece aligns to word_ids to ignore subword continuations
        word_ids = encoding.word_ids()

        entities = []
        current_entity = None
        previous_word_idx = None

        for i, (word_idx, offset) in enumerate(zip(word_ids, offsets)):
            # Skip special tokens ([CLS], [SEP]) where word_idx is None
            if word_idx is None:
                continue
                
            # Skip subword tokens (only process the first piece of a word)
            if word_idx == previous_word_idx:
                # But we do need to update the end offset of the current entity if it continues
                if current_entity and offset[1] > 0:
                    current_entity["end"] = offset[1]
                    current_entity["value"] = text[current_entity["start"]:offset[1]]
                continue
                
            previous_word_idx = word_idx
            
            tag = self.idx2label[pred_indices[i]]
            conf = confidences[i]
            
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                ent_type = tag[2:]
                start_char, end_char = offset
                current_entity = {
                    "entity_type": ent_type,
                    "value": text[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "confidences": [conf]
                }
            elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag[2:]:
                _, end_char = offset
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

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=512
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        batch_results = []

        for b_idx, text in enumerate(texts):
            offsets = encodings["offset_mapping"][b_idx].tolist()
            confidences = max_probs[b_idx].cpu().numpy().tolist()
            pred_indices = preds[b_idx].cpu().numpy().tolist()
            word_ids = encodings.word_ids(batch_index=b_idx)

            entities = []
            current_entity = None
            previous_word_idx = None

            for i, (word_idx, offset) in enumerate(zip(word_ids, offsets)):
                if word_idx is None:
                    continue
                    
                if word_idx == previous_word_idx:
                    if current_entity and offset[1] > 0:
                        current_entity["end"] = offset[1]
                        current_entity["value"] = text[current_entity["start"]:offset[1]]
                    continue
                    
                previous_word_idx = word_idx
                
                tag = self.idx2label[pred_indices[i]]
                conf = confidences[i]
                
                if tag.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    start_char, end_char = offset
                    current_entity = {
                        "entity_type": tag[2:],
                        "value": text[start_char:end_char],
                        "start": start_char,
                        "end": end_char,
                        "confidences": [conf]
                    }
                elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag[2:]:
                    _, end_char = offset
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
            batch_results.append(final_entities)

        return batch_results

    def is_ready(self) -> bool:
        return self._is_ready


if __name__ == "__main__":
    # Test Block
    print("Initializing BERT inference module...")
    
    # Using 0.0 threshold to see outputs even if untrained weights are loaded
    inference = BertNERInference(confidence_threshold=0.0) 
    
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
                
    print("\nbert_inference.py ready for integration")