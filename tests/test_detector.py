import pytest
from ml_engine.rule_based_detector import RuleBasedDetector

def test_rule_based_detector_email():
    detector = RuleBasedDetector()
    detector.load()  # Loads presidio
    
    text = "Send an email to test@example.com."
    entities = detector.detect(text, threshold=0.4)
    
    assert len(entities) > 0
    types = [e["entity_group"] for e in entities]
    assert "EMAIL_ADDRESS" in types