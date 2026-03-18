import pytest
from ml_engine.redactor import RedactionEngine

@pytest.fixture
def engine():
    return RedactionEngine(confidence_threshold=0.85)

# Test 1: Single entity redacted correctly
def test_single_entity_redacted(engine):
    text = "Contact John Smith about the project"
    entities = [{"entity_type": "PERSON", "start": 8, "end": 18, "confidence": 0.95}]
    
    redacted, mapping, session_id = engine.redact(text, entities)
    
    assert "[PERSON_1]" in redacted
    assert "John Smith" not in redacted
    assert redacted == "Contact [PERSON_1] about the project"
    assert mapping["[PERSON_1]"] == "John Smith"

# Test 2: Multiple same-type entities get incrementing counters
def test_multiple_same_type_entities(engine):
    text = "John and Mary went to the store"
    entities = [
        {"entity_type": "PERSON", "start": 0, "end": 4, "confidence": 0.99},
        {"entity_type": "PERSON", "start": 9, "end": 13, "confidence": 0.99}
    ]
    
    redacted, mapping, _ = engine.redact(text, entities)
    
    assert "[PERSON_1]" in redacted
    assert "[PERSON_2]" in redacted
    assert redacted == "[PERSON_1] and [PERSON_2] went to the store"
    assert mapping["[PERSON_1]"] == "John"
    assert mapping["[PERSON_2]"] == "Mary"

# Test 3: Multiple different entity types
def test_multiple_different_entity_types(engine):
    text = "Alice's email is a@b.com and SSN is 123"
    entities = [
        {"entity_type": "PERSON", "start": 0, "end": 5, "confidence": 0.90},
        {"entity_type": "EMAIL", "start": 17, "end": 24, "confidence": 0.95},
        {"entity_type": "SSN", "start": 36, "end": 39, "confidence": 0.98}
    ]
    
    redacted, mapping, _ = engine.redact(text, entities)
    
    assert "[PERSON_1]" in redacted
    assert "[EMAIL_1]" in redacted
    assert "[SSN_1]" in redacted
    assert redacted == "[PERSON_1]'s email is [EMAIL_1] and SSN is [SSN_1]"

# Test 4: Overlapping spans — higher confidence wins
def test_overlapping_spans(engine):
    text = "Call 555-123-4567 today"
    entities = [
        {"entity_type": "PHONE", "start": 5, "end": 17, "confidence": 0.95},
        {"entity_type": "SSN", "start": 5, "end": 17, "confidence": 0.87}  # Overlap, lower confidence
    ]
    
    redacted, mapping, _ = engine.redact(text, entities)
    
    assert "[PHONE_1]" in redacted
    assert "[SSN_1]" not in redacted
    assert redacted == "Call [PHONE_1] today"

# Test 5: Entity below threshold is NOT redacted
def test_entity_below_threshold(engine):
    text = "Find John"
    entities = [{"entity_type": "PERSON", "start": 5, "end": 9, "confidence": 0.75}]
    
    redacted, mapping, _ = engine.redact(text, entities)
    
    assert redacted == "Find John"
    assert len(mapping) == 0

# Test 6: reinsert restores original values
def test_reinsert_restores_original(engine):
    original_text = "My SSN is 123-45-6789"
    entities = [{"entity_type": "SSN", "start": 10, "end": 21, "confidence": 0.99}]
    
    redacted, mapping, _ = engine.redact(original_text, entities)
    restored = engine.reinsert(redacted, mapping)
    
    assert restored == original_text

# Test 7: reinsert handles lowercased placeholder
def test_reinsert_lowercased_placeholder(engine):
    text = "Call John"
    entities = [{"entity_type": "PERSON", "start": 5, "end": 9, "confidence": 0.99}]
    
    _, mapping, _ = engine.redact(text, entities)
    llm_response = "I will call [person_1] right away."
    
    restored = engine.reinsert(llm_response, mapping)
    assert restored == "I will call John right away."

# Test 8: reinsert handles placeholder with punctuation
def test_reinsert_with_punctuation(engine):
    text = "Call John"
    entities = [{"entity_type": "PERSON", "start": 5, "end": 9, "confidence": 0.99}]
    
    _, mapping, _ = engine.redact(text, entities)
    llm_response = "Here is the info for [PERSON_1]."
    
    restored = engine.reinsert(llm_response, mapping)
    assert restored == "Here is the info for John."

# Test 9: reinsert handles duplicated placeholder
def test_reinsert_duplicated_placeholder(engine):
    text = "SSN is 123"
    entities = [{"entity_type": "SSN", "start": 7, "end": 10, "confidence": 0.99}]
    
    _, mapping, _ = engine.redact(text, entities)
    llm_response = "[SSN_1] is recorded. Verifying [SSN_1]."
    
    restored = engine.reinsert(llm_response, mapping)
    assert restored == "123 is recorded. Verifying 123."

# Test 10: reinsert with no placeholders in response
def test_reinsert_no_placeholders(engine):
    text = "John is here"
    entities = [{"entity_type": "PERSON", "start": 0, "end": 4, "confidence": 0.99}]
    
    _, mapping, _ = engine.redact(text, entities)
    llm_response = "I understand the request."
    
    restored = engine.reinsert(llm_response, mapping)
    assert restored == "I understand the request."

# Test 11: validate_redaction returns True when clean
def test_validate_redaction_clean(engine):
    text = "My name is John"
    mapping = {"[PERSON_1]": "John"}
    redacted = "My name is [PERSON_1]"
    
    assert engine.validate_redaction(text, redacted, mapping) is True

# Test 12: validate_redaction returns False when leak detected
def test_validate_redaction_leak(engine):
    text = "My name is John"
    mapping = {"[PERSON_1]": "John"}
    # Simulate a broken redaction where the original value is still present
    redacted = "My name is [PERSON_1] (aka john)"
    
    assert engine.validate_redaction(text, redacted, mapping) is False

# Test 13: Session isolation
def test_session_isolation(engine):
    text1 = "Alice is here"
    ent1 = [{"entity_type": "PERSON", "start": 0, "end": 5, "confidence": 0.99}]
    
    text2 = "Bob is there"
    ent2 = [{"entity_type": "PERSON", "start": 0, "end": 3, "confidence": 0.99}]
    
    _, map1, _ = engine.redact(text1, ent1)
    _, map2, _ = engine.redact(text2, ent2)
    
    assert map1 is not map2
    assert map1["[PERSON_1]"] == "Alice"
    assert map2["[PERSON_1]"] == "Bob"

# Test 14: cleanup removes session
def test_cleanup_removes_session(engine):
    text = "Find John"
    entities = [{"entity_type": "PERSON", "start": 5, "end": 9, "confidence": 0.99}]
    
    _, _, session_id = engine.redact(text, entities)
    
    assert session_id in engine.sessions
    engine.cleanup(session_id)
    assert session_id not in engine.sessions

# Test 15: Empty input
def test_empty_input(engine):
    text = ""
    entities = []
    
    redacted, mapping, session_id = engine.redact(text, entities)
    
    assert redacted == ""
    assert isinstance(mapping, dict)
    assert len(mapping) == 0
    assert isinstance(session_id, str)
    assert len(session_id) > 0