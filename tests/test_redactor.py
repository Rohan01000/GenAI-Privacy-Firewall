import pytest
from ml_engine.redactor import Redactor

class MockDetector:
    def detect_entities(self, text):
        if "John" in text:
            return [{"start": 11, "end": 15, "entity_group": "PER"}]
        return []

def test_redactor_basic():
    detector = MockDetector()
    redactor = Redactor(detector)
    
    original = "My name is John."
    redacted, entities = redactor.redact(original)
    
    assert "John" not in redacted
    assert "[REDACTED_PER]" in redacted
    assert len(entities) == 1
    assert redacted == "My name is [REDACTED_PER]."

def test_redactor_no_entities():
    detector = MockDetector()
    redactor = Redactor(detector)
    
    original = "I love programming."
    redacted, entities = redactor.redact(original)
    
    assert redacted == original
    assert len(entities) == 0