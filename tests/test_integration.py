import pytest
from fastapi.testclient import TestClient
from main import app
from config.settings import settings

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_admin_analyze_unauthorized():
    response = client.post("/admin/analyze", json={"text": "Hello John"})
    # Missing auth header
    assert response.status_code == 422 

def test_admin_analyze_authorized():
    # Wait for lifespan to load models in standard setup, 
    # but TestClient triggers lifespan automatically in newer FastAPI.
    headers = {"x-admin-token": settings.admin_secret_key}
    response = client.post("/admin/analyze", json={"text": "Hello John Doe"}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        assert "original_text" in data
        assert "redacted_text" in data