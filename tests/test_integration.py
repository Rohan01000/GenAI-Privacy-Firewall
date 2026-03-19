import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from main import app
from config.settings import settings

class MockResponse:
    def __init__(self, content):
        self.status_code = 200
        self._content = content
        
    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}
        
    def raise_for_status(self):
        pass

@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(autouse=True)
def reset_daily_counts():
    from proxy.proxy_handler import api_key_daily_counts, limiter
    api_key_daily_counts.clear()
    limiter._storage.reset()
    yield
    api_key_daily_counts.clear()
    limiter._storage.reset()

@pytest.fixture
def mock_llm():
    """Mocks the external LLM API post request to echo back the received content."""
    async def mock_post_impl(*args, **kwargs):
        req_json = kwargs.get('json', {})
        messages = req_json.get('messages', [])
        content = messages[0]['content'] if messages else ""
        return MockResponse(content)

    with patch('proxy.proxy_handler.http_client.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = mock_post_impl
        yield mock_post

def test_clean_prompt_passes_unchanged(client, mock_llm):
    """Test 1: Clean prompt passes through unchanged."""
    headers = {"Authorization": "Bearer test-key-1"}
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is the capital of France?"}]}
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    
    assert response.status_code == 200
    resp_data = response.json()
    final_content = resp_data["choices"][0]["message"]["content"]
    
    assert "France" in final_content
    assert "[" not in final_content and "]" not in final_content  # No placeholders

def test_ssn_redacted_then_reinserted(client, mock_llm):
    """Test 2: SSN is redacted then reinserted correctly."""
    headers = {"Authorization": "Bearer test-key-2"}
    secret_ssn = "123-45-6789"
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"Analyze account for SSN {secret_ssn}"}]}
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    
    assert response.status_code == 200
    
    # Check outgoing request to mock LLM
    outgoing_payload = mock_llm.call_args.kwargs.get('json', {})
    outgoing_content = outgoing_payload["messages"][0]["content"]
    assert secret_ssn not in outgoing_content
    assert "[SSN_1]" in outgoing_content
    
    # Check final response
    resp_data = response.json()
    final_content = resp_data["choices"][0]["message"]["content"]
    assert secret_ssn in final_content

def test_openai_api_key_caught(client, mock_llm):
    """Test 3: OpenAI API key caught by rule-based detector."""
    headers = {"Authorization": "Bearer test-key-3"}
    secret_key = "sk-" + "a" * 48
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"My key is {secret_key}"}]}
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    
    assert response.status_code == 200
    
    outgoing_content = mock_llm.call_args.kwargs.get('json', {})["messages"][0]["content"]
    assert secret_key not in outgoing_content
    
    final_content = response.json()["choices"][0]["message"]["content"]
    assert secret_key in final_content

def test_multiple_entity_types_simultaneously(client, mock_llm):
    """Test 4: Multiple entity types simultaneously."""
    headers = {"Authorization": "Bearer test-key-4"}
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "User John Smith at john@acme.com card 4532-1234-5678-9012"}]}
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    
    outgoing_content = mock_llm.call_args.kwargs.get('json', {})["messages"][0]["content"]
    assert "John Smith" not in outgoing_content
    assert "john@acme.com" not in outgoing_content
    assert "4532-1234-5678-9012" not in outgoing_content
    
    final_content = response.json()["choices"][0]["message"]["content"]
    assert "John Smith" in final_content
    assert "john@acme.com" in final_content
    assert "4532-1234-5678-9012" in final_content

def test_obfuscated_ssn(client, mock_llm):
    """Test 5: Obfuscated SSN with dots instead of dashes."""
    headers = {"Authorization": "Bearer test-key-5"}
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "SSN is 123.45.6789"}]}
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200

def test_oversized_request_rejected(client, mock_llm):
    """Test 6: Oversized request rejected."""
    headers = {"Authorization": "Bearer test-key-6"}
    huge_text = "a" * 60000
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": huge_text}]}
    
    # Mock content length header to trigger size check
    headers["Content-Length"] = "60000"
    
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 400

def test_rate_limit_enforced(client, mock_llm):
    """Test 7: Rate limit enforced."""
    headers = {"Authorization": "Bearer test-rate-limit-key"}
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hello"}]}
    
    # Send 60 requests
    for _ in range(60):
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        
    # 61st request should hit the 60/minute limiter
    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 429

def test_admin_stats_endpoint(client, mock_llm):
    """Test 8: Admin stats endpoint works."""
    headers = {"Authorization": "Bearer test-key-8"}
    admin_headers = {"X-Admin-Key": settings.admin_secret_key}
    
    # Generate 3 requests with sensitive data
    for _ in range(3):
        payload = {"model": "gpt", "messages": [{"role": "user", "content": "My email is test@example.com"}]}
        client.post("/v1/chat/completions", json=payload, headers=headers)
        
    response = client.get("/admin/stats", headers=admin_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_requests"] >= 3
    assert data["total_entities_redacted"] > 0

def test_model_fallback_on_failure(client, mock_llm):
    """Test 9: Model fallback when ML model fails."""
    headers = {"Authorization": "Bearer test-key-9"}
    secret_key = "sk-" + "b" * 48
    payload = {"model": "gpt", "messages": [{"role": "user", "content": f"Key: {secret_key}"}]}
    
    with patch('proxy.proxy_handler.detector.detect_sync', side_effect=RuntimeError("Simulated Model Failure")):
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        
        assert response.status_code == 200
        assert "X-Firewall-Mode" in response.headers
        assert response.headers["X-Firewall-Mode"] == "fallback"
        
        # Ensure rule-based still caught it
        outgoing_content = mock_llm.call_args.kwargs.get('json', {})["messages"][0]["content"]
        assert secret_key not in outgoing_content

def test_redaction_validation_failure(client, mock_llm):
    """Test 10: Redaction validation failure blocks request."""
    headers = {"Authorization": "Bearer test-key-10"}
    payload = {"model": "gpt", "messages": [{"role": "user", "content": "Test validation failure."}]}
    
    with patch('proxy.proxy_handler.redactor.validate_redaction', return_value=False):
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 400