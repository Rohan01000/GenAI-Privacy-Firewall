from config.settings import settings
from .rule_based_detector import RuleBasedDetector
from .bert_model.bert_inference import BertDetector
from .scratch_model.inference import ScratchDetector

class CombinedDetector:
    def __init__(self):
        self.rule_based = RuleBasedDetector()
        self.ml_model = None
        self._ready = False

    def load_models(self):
        # Load rule-based
        self.rule_based.load()
        
        # Load ML based on config
        if settings.model_type == "bert":
            self.ml_model = BertDetector()
        elif settings.model_type == "scratch":
            self.ml_model = ScratchDetector()
        else:
            raise ValueError(f"Unknown model_type: {settings.model_type}")
            
        self.ml_model.load()
        self._ready = True

    def unload_models(self):
        self.ml_model = None
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def detect_entities(self, text: str):
        if not self._ready:
            return []

        # 1. Get rule-based entities
        rule_entities = self.rule_based.detect(text, threshold=0.5)
        
        # 2. Get ML-based entities
        ml_entities = self.ml_model.detect(text, threshold=settings.confidence_threshold)
        
        # 3. Merge strategies (simple union with deduplication by spans)
        merged = []
        seen_spans = set()
        
        for ent in rule_entities + ml_entities:
            span = (ent["start"], ent["end"])
            # Simple conflict resolution: keep the first detected (prioritizes rule-based if listed first)
            # A more advanced logic would check for overlaps and pick the highest confidence
            overlap = any(s[0] < span[1] and span[0] < s[1] for s in seen_spans)
            if not overlap:
                seen_spans.add(span)
                merged.append(ent)
                
        # Sort by start index
        merged.sort(key=lambda x: x["start"])
        return merged

# Global singleton instance
detector_instance = CombinedDetector()