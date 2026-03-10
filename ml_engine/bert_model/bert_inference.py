from transformers import pipeline
import os

class BertDetector:
    def __init__(self, model_path="models/bert_ner"):
        self.model_path = model_path
        self.nlp = None

    def load(self):
        # Fallback to huggingface hub if not downloaded locally
        path = self.model_path if os.path.exists(self.model_path) else "dslim/bert-base-NER"
        self.nlp = pipeline("ner", model=path, tokenizer=path, aggregation_strategy="simple")

    def detect(self, text: str, threshold: float = 0.5):
        if not self.nlp:
            return []
            
        results = self.nlp(text)
        
        filtered = []
        for res in results:
            if res["score"] >= threshold:
                filtered.append({
                    "entity_group": res["entity_group"],
                    "score": float(res["score"]),
                    "word": res["word"],
                    "start": res["start"],
                    "end": res["end"]
                })
        return filtered