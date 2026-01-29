"""
Integration tests for Cricket Inference Service.

Tests the FastAPI endpoints for health checks and inference.
Validates response structure and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from service.app import app, InferRequest, InferResponse


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test /healthz endpoint returns 200 OK."""
        response = client.get('/healthz')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'service' in data
    
    def test_ready_endpoint(self, client):
        """Test /readyz endpoint returns readiness status."""
        response = client.get('/readyz')
        assert response.status_code == 200
        data = response.json()
        assert 'ready' in data
        assert isinstance(data['ready'], bool)
        assert 'model_loaded' in data
    
    def test_root_endpoint(self, client):
        """Test root / endpoint returns service info."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data


class TestInferenceEndpoint:
    """Test inference endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    def test_infer_basic_question(self, client):
        """Test basic inference with cricket question.
        
        This is a simple test that validates:
        1. Endpoint responds with 200 or 503 (model loading)
        2. Response structure is valid
        3. Required fields are present
        """
        payload = InferRequest(
            question="Who won IPL 2023?"
        )
        
        response = client.post('/infer', json=payload.dict())
        
        # Accept 503 if model not loaded, 200 if successful
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Validate response structure
            assert 'answer' in data
            assert 'latency_ms' in data
            assert 'tokens_generated' in data
            
            # Validate data types
            assert isinstance(data['answer'], str)
            assert isinstance(data['latency_ms'], (int, float))
            assert isinstance(data['tokens_generated'], int)
            
            # Validate non-empty answer
            assert len(data['answer']) > 0
            assert data['latency_ms'] >= 0
            assert data['tokens_generated'] >= 0
    
    def test_infer_with_context(self, client):
        """Test inference with additional context."""
        payload = {
            "question": "How many sixes were hit in IPL 2023?",
            "context": "IPL 2023 had many explosive batting performances."
        }
        
        response = client.post('/infer', json=payload)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'answer' in data
            assert 'latency_ms' in data
    
    def test_infer_empty_question_error(self, client):
        """Test that empty question returns 400 error."""
        payload = {
            "question": ""
        }
        
        response = client.post('/infer', json=payload)
        
        # Should be 400 (bad request) when model is loaded
        # or 503 (service unavailable) if model not loaded
        assert response.status_code in [400, 503]
        
        if response.status_code == 400:
            data = response.json()
            assert 'detail' in data
    
    def test_infer_missing_question_error(self, client):
        """Test that missing question field returns error."""
        payload = {}
        
        response = client.post('/infer', json=payload)
        
        # Should return error (422 for validation error or 503)
        assert response.status_code in [422, 503]
    
    def test_infer_response_schema(self, client):
        """Test that inference response matches InferResponse schema."""
        payload = {
            "question": "What was RCB's win record in IPL 2023?"
        }
        
        response = client.post('/infer', json=payload)
        
        if response.status_code == 200:
            # Try to parse as InferResponse
            data = response.json()
            response_model = InferResponse(**data)
            
            # Validate schema
            assert response_model.answer is not None
            assert response_model.latency_ms >= 0
            assert response_model.tokens_generated >= 0


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    def test_invalid_json_error(self, client):
        """Test that invalid JSON returns 422 error."""
        response = client.post('/infer', json="invalid")
        assert response.status_code in [422, 400]
    
    def test_unknown_endpoint_error(self, client):
        """Test that unknown endpoint returns 404."""
        response = client.get('/unknown-endpoint')
        assert response.status_code == 404


class TestRequestValidation:
    """Test request validation."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    def test_infer_question_type_validation(self, client):
        """Test that non-string question is rejected."""
        payload = {
            "question": 123  # Should be string
        }
        
        response = client.post('/infer', json=payload)
        
        # Should return validation error
        assert response.status_code in [422, 503]
    
    def test_infer_context_optional(self, client):
        """Test that context field is optional."""
        payload = {
            "question": "Which team won IPL 2023?"
        }
        
        response = client.post('/infer', json=payload)
        
        # Should work without context field
        assert response.status_code in [200, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
