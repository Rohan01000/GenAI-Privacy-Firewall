import pytest
from unittest.mock import MagicMock, patch
from ml_engine.rule_based_detector import RuleBasedDetector
from ml_engine.combined_detector import CombinedDetector

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def rule_detector():
    """Provides a fresh instance of RuleBasedDetector for each test."""
    return RuleBasedDetector()

@pytest.fixture
@patch('ml_engine.scratch_model.inference.ScratchNERInference')
def combined_detector(mock_ml_inference):
    """
    Provides a CombinedDetector with the ML model mocked out 
    to avoid loading heavy neural network weights during unit tests.
    """
    return CombinedDetector(model_type="scratch", confidence_threshold=0.85)


# ==========================================
# RULE-BASED DETECTOR TESTS (10 TESTS)
# ==========================================

def test_openai_api_key_detected(rule_detector):
    """Test 1: OpenAI API key detected"""
    key = "sk-" + "a" * 48
    text = f"My key is {key} do not share it."
    results = rule_detector.detect(text)
    
    assert len(results) > 0
    # Ensure at least one result is the API_KEY with 1.0 confidence
    api_key_results = [r for r in results if r["entity_type"] == "API_KEY"]
    assert len(api_key_results) > 0
    assert api_key_results[0]["value"] == key
    assert api_key_results[0]["confidence"] == 1.0

def test_aws_access_key_detected(rule_detector):
    """Test 2: AWS access key detected"""
    key = "AKIAIOSFODNN7EXAMPLE"
    text = f"AWS key: {key}"
    results = rule_detector.detect(text)
    
    api_key_results = [r for r in results if r["entity_type"] == "API_KEY"]
    assert len(api_key_results) > 0
    assert api_key_results[0]["value"] == key

def test_private_ip_address_detected(rule_detector):
    """Test 3: Private IP address detected"""
    ip = "192.168.1.100"
    text = f"Server is at {ip}"
    results = rule_detector.detect(text)
    
    ip_results = [r for r in results if r["entity_type"] == "IP_ADDRESS"]
    assert len(ip_results) > 0
    assert ip_results[0]["value"] == ip

def test_internal_network_10_detected(rule_detector):
    """Test 4: Internal network 10.x.x.x detected"""
    text = "Connect to 10.0.0.1 for access"
    results = rule_detector.detect(text)
    
    ip_results = [r for r in results if r["entity_type"] == "IP_ADDRESS"]
    assert len(ip_results) > 0
    assert ip_results[0]["value"] == "10.0.0.1"

def test_email_address_detected(rule_detector):
    """Test 5: Email address detected"""
    email = "john.doe@company.com"
    text = f"Send to {email} please"
    results = rule_detector.detect(text)
    
    email_results = [r for r in results if r["entity_type"] == "EMAIL"]
    assert len(email_results) > 0
    assert email_results[0]["value"] == email

def test_password_keyword_pattern_detected(rule_detector):
    """Test 6: Password keyword pattern detected"""
    secret = "SuperSecret123"
    text = f"db_password={secret}"
    results = rule_detector.detect(text)
    
    password_results = [r for r in results if r["entity_type"] == "PASSWORD"]
    assert len(password_results) > 0
    assert password_results[0]["value"] == secret

def test_credit_card_detected(rule_detector):
    """Test 7: Credit card detected"""
    cc = "4532-1234-5678-9012"
    text = f"Card number {cc} failed"
    results = rule_detector.detect(text)
    
    cc_results = [r for r in results if r["entity_type"] == "CREDIT_CARD"]
    assert len(cc_results) > 0
    assert cc_results[0]["value"] == cc

def test_ssn_detected(rule_detector):
    """Test 8: SSN detected"""
    ssn = "123-45-6789"
    text = f"SSN on file: {ssn}"
    results = rule_detector.detect(text)
    
    ssn_results = [r for r in results if r["entity_type"] == "SSN"]
    assert len(ssn_results) > 0
    assert ssn_results[0]["value"] == ssn

def test_clean_text_returns_empty(rule_detector):
    """Test 9: Clean text returns empty list"""
    text = "What is the capital of France?"
    results = rule_detector.detect(text)
    assert results == []

def test_multiple_entities_in_one_text(rule_detector):
    """Test 10: Multiple entities in one text"""
    text = "Contact test@example.com from 10.0.0.5 using key AKIAIOSFODNN7EXAMPLE."
    results = rule_detector.detect(text)
    
    assert len(results) >= 3
    entity_types = [r["entity_type"] for r in results]
    assert "EMAIL" in entity_types
    assert "IP_ADDRESS" in entity_types
    assert "API_KEY" in entity_types


# ==========================================
# COMBINED DETECTOR TESTS (5 TESTS)
# ==========================================

def test_rule_priority_on_overlap(combined_detector):
    """Test 11: Rule-based result takes priority over ML on overlap"""
    # Mock ML response
    combined_detector.ml_model.detect_entities.return_value = [
        {"entity_type": "PERSON", "value": "overlap", "start": 5, "end": 15, "confidence": 0.95}
    ]
    # Mock Rule response
    combined_detector.rule_detector = MagicMock()
    combined_detector.rule_detector.detect.return_value = [
        {"entity_type": "API_KEY", "value": "overlap", "start": 5, "end": 15, "confidence": 1.0}
    ]
    
    results = combined_detector.detect_sync("test overlap test")
    
    assert len(results) == 1
    assert results[0]["entity_type"] == "API_KEY"

def test_non_overlapping_both_included(combined_detector):
    """Test 12: Non-overlapping ML and rule results both included"""
    # Mock ML response
    combined_detector.ml_model.detect_entities.return_value = [
        {"entity_type": "PERSON", "value": "John", "start": 0, "end": 4, "confidence": 0.95}
    ]
    # Mock Rule response
    combined_detector.rule_detector = MagicMock()
    combined_detector.rule_detector.detect.return_value = [
        {"entity_type": "API_KEY", "value": "sk-12345", "start": 10, "end": 18, "confidence": 1.0}
    ]
    
    results = combined_detector.detect_sync("John used sk-12345")
    
    assert len(results) == 2
    entity_types = [r["entity_type"] for r in results]
    assert "PERSON" in entity_types
    assert "API_KEY" in entity_types

def test_empty_text_returns_empty_list(combined_detector):
    """Test 13: Empty text returns empty list"""
    results = combined_detector.detect_sync("")
    assert results == []

def test_detect_batch(combined_detector):
    """Test 14: detect_batch returns one list per input text"""
    # Mock the internal sync call for simplicity
    with patch.object(combined_detector, 'detect_sync', return_value=[{"entity_type": "PERSON", "start": 0, "end": 4}]) as mock_sync:
        texts = ["text one", "text two", "text three"]
        results = combined_detector.detect_batch(texts)
        
        assert len(results) == 3
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)
        assert isinstance(results[2], list)

def test_results_sorted_by_start_position(combined_detector):
    """Test 15: Results sorted by start position"""
    # Simulate unordered returns from the underlying detectors
    combined_detector.ml_model.detect_entities.return_value = [
        {"entity_type": "PERSON", "value": "Second", "start": 20, "end": 26, "confidence": 0.9}
    ]
    combined_detector.rule_detector = MagicMock()
    combined_detector.rule_detector.detect.return_value = [
        {"entity_type": "EMAIL", "value": "first@test.com", "start": 0, "end": 14, "confidence": 1.0}
    ]
    
    # Combined detector logic should sort them by start index ascending
    results = combined_detector.detect_sync("first@test.com and Second")
    
    assert len(results) == 2
    assert results[0]["start"] == 0
    assert results[0]["entity_type"] == "EMAIL"
    assert results[1]["start"] == 20
    assert results[1]["entity_type"] == "PERSON"