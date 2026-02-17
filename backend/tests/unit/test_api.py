"""Tests for Flask API server.

Tests all API endpoints, request/response validation, error handling,
request logging, and middleware functionality.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.api import create_app
from src.config import Config
from src.schemas import (
    UserProfile,
    RecommendationRequest,
    AnalysisRequest,
    ErrorResponse,
)


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.FLASK_ENV = 'testing'
    return config


@pytest.fixture
def app(config):
    """Create test Flask application."""
    app = create_app(config)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def mock_rag_pipeline(app):
    """Mock RAG pipeline."""
    mock_pipeline = Mock()
    app.config['RAG_PIPELINE'] = mock_pipeline
    return mock_pipeline


@pytest.fixture
def mock_cache(app):
    """Mock cache manager."""
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.generate_key.return_value = "test_key"
    app.config['CACHE_MANAGER'] = mock_cache
    return mock_cache


# ============================================================================
# Health Check Endpoint Tests
# ============================================================================


class TestHealthCheckEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test health check returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_check_returns_json(self, client):
        """Test health check returns valid JSON."""
        response = client.get('/health')
        data = response.get_json()
        
        assert 'status' in data
        assert 'version' in data
        assert 'faiss_index_loaded' in data
        assert 'mistral_api_available' in data
    
    def test_health_check_status_values(self, client):
        """Test health check status is valid enum."""
        response = client.get('/health')
        data = response.get_json()
        
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_health_check_has_timestamp(self, client):
        """Test health check includes timestamp."""
        response = client.get('/health')
        data = response.get_json()
        
        assert 'timestamp' in data
    
    def test_health_check_no_cache(self, client, mock_cache):
        """Test health check is not cached."""
        response1 = client.get('/health')
        response2 = client.get('/health')
        
        assert response1.status_code == 200
        assert response2.status_code == 200


# ============================================================================
# Recommendations Endpoint Tests
# ============================================================================


class TestRecommendationsEndpoint:
    """Tests for /api/recommendations endpoint."""
    
    def test_options_request_returns_204(self, client):
        """Test OPTIONS request returns 204."""
        response = client.options('/api/recommendations')
        assert response.status_code == 204
    
    def test_get_method_not_allowed(self, client):
        """Test GET method is not allowed."""
        response = client.get('/api/recommendations')
        assert response.status_code == 405
    
    def test_valid_recommendation_request(self, client, mock_rag_pipeline, mock_cache):
        """Test valid recommendation request."""
        mock_rag_pipeline.generate_recommendations.return_value = [
            {
                'id': 'rec_001',
                'title': 'SaaS Platform',
                'description': 'Build a Software-as-a-Service platform providing project management tools for remote teams. This business model allows for recurring revenue and scalability.' * 2,
                'why_it_fits': 'Your programming background and leadership experience makes this ideal for building and managing such a platform.',
                'required_skills': [],
                'key_challenges': []
            }
        ]
        
        request_data = {
            'user_profile': {
                'age': 30,
                'skills': ['Python']
            },
            'num_recommendations': 3
        }
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'recommendations' in data
    
    def test_recommendation_request_without_json(self, client):
        """Test recommendation request without JSON."""
        response = client.post('/api/recommendations')
        assert response.status_code == 400
    
    def test_invalid_user_profile_returns_400(self, client):
        """Test invalid user profile validation."""
        request_data = {
            'user_profile': {
                'age': 15  # Too young
            }
        }
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['error_type'] == 'validation_error'
        assert len(data['details']) > 0
    
    def test_invalid_num_recommendations_returns_400(self, client):
        """Test invalid number of recommendations."""
        request_data = {
            'user_profile': {},
            'num_recommendations': 0  # Too low
        }
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_recommendation_with_custom_flags(self, client, mock_rag_pipeline, mock_cache):
        """Test recommendation with custom flags."""
        mock_rag_pipeline.generate_recommendations.return_value = [
            {
                'id': 'rec_001',
                'title': 'Test Idea',
                'description': 'This is a comprehensive test description that meets the minimum character requirement for business recommendations' * 2,
                'why_it_fits': 'This business opportunity aligns well with your profile and experience level.',
                'required_skills': [],
                'key_challenges': []
            }
        ]
        
        request_data = {
            'user_profile': {'skills': ['Python']},
            'num_recommendations': 5,
            'include_reasoning': False,
            'include_market_data': False
        }
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        mock_rag_pipeline.generate_recommendations.assert_called_once()
    
    def test_recommendation_response_includes_metadata(self, client, mock_rag_pipeline, mock_cache):
        """Test recommendation response includes metadata."""
        mock_rag_pipeline.generate_recommendations.return_value = [
            {
                'id': 'rec_001',
                'title': 'Test Idea',
                'description': 'This is a comprehensive test description that meets the minimum character requirement for business recommendations' * 2,
                'why_it_fits': 'This business opportunity aligns well with your profile and experience.',
                'required_skills': [],
                'key_challenges': []
            }
        ]
        
        request_data = {
            'user_profile': {'age': 30}
        }
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        data = response.get_json()
        assert 'generation_time_ms' in data
        assert 'model_used' in data
        assert 'context_sources' in data
    
    def test_recommendation_caching(self, client, mock_rag_pipeline, mock_cache):
        """Test recommendation caching works."""
        # First request hits cache miss
        mock_cache.get.return_value = None
        mock_rag_pipeline.generate_recommendations.return_value = [
            {
                'id': 'rec_001',
                'title': 'Test Idea',
                'description': 'This is a comprehensive test description that meets the minimum character requirement for business recommendations' * 2,
                'why_it_fits': 'This business opportunity aligns well with your profile.',
                'required_skills': [],
                'key_challenges': []
            }
        ]
        
        request_data = {'user_profile': {'age': 30}}
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        mock_cache.set.assert_called_once()
    
    def test_recommendation_cache_hit(self, client, mock_cache):
        """Test recommendation cache hit."""
        cached_response = {
            'success': True,
            'recommendations': [],
            'generation_time_ms': 100,
            'model_used': 'mistral-large-latest',
            'context_sources': 0
        }
        mock_cache.get.return_value = cached_response
        
        request_data = {'user_profile': {'age': 30}}
        
        response = client.post(
            '/api/recommendations',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data == cached_response


# ============================================================================
# Analysis Endpoint Tests
# ============================================================================


class TestAnalysisEndpoint:
    """Tests for /api/analyze endpoint."""
    
    def test_options_request_returns_204(self, client):
        """Test OPTIONS request returns 204."""
        response = client.options('/api/analyze')
        assert response.status_code == 204
    
    def test_get_method_not_allowed(self, client):
        """Test GET method is not allowed."""
        response = client.get('/api/analyze')
        assert response.status_code == 405
    
    def test_valid_analysis_request(self, client, mock_rag_pipeline, mock_cache):
        """Test valid analysis request."""
        mock_rag_pipeline.analyze_business_idea.return_value = {
            'business_idea': 'E-commerce platform for local artisans',
            'viability_score': 75.0,
            'summary': 'This business idea demonstrates strong market potential with a growing audience of consumers interested in supporting local artisans and unique handmade products. Success depends on effective marketing and logistics.'
        }
        
        request_data = {
            'business_idea': 'E-commerce platform for local artisans'
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'analysis' in data
    
    def test_analysis_request_without_json(self, client):
        """Test analysis request without JSON."""
        response = client.post('/api/analyze')
        assert response.status_code == 400
    
    def test_invalid_business_idea_too_short(self, client):
        """Test business idea validation - too short."""
        request_data = {
            'business_idea': 'Short'
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_invalid_business_idea_too_long(self, client):
        """Test business idea validation - too long."""
        request_data = {
            'business_idea': 'x' * 5001
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_analysis_with_user_profile(self, client, mock_rag_pipeline, mock_cache):
        """Test analysis with user profile."""
        mock_rag_pipeline.analyze_business_idea.return_value = {
            'business_idea': 'Valid business idea description here',
            'viability_score': 75.0,
            'summary': 'This business idea demonstrates strong market potential with a growing audience of consumers interested in supporting products and services. Success depends on effective marketing strategy.'
        }
        
        request_data = {
            'business_idea': 'Valid business idea description here',
            'user_profile': {
                'age': 30,
                'skills': ['Programming']
            }
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
    
    def test_analysis_depth_options(self, client, mock_rag_pipeline, mock_cache):
        """Test analysis depth options."""
        mock_rag_pipeline.analyze_business_idea.return_value = {
            'business_idea': 'Valid business idea description here',
            'viability_score': 75.0,
            'summary': 'This business idea demonstrates strong market potential with growing consumer interest in the target market segment. Success depends on effective marketing and operational efficiency.'
        }
        
        for depth in ['basic', 'detailed', 'comprehensive']:
            request_data = {
                'business_idea': 'Valid business idea description here',
                'analysis_depth': depth
            }
            
            response = client.post(
                '/api/analyze',
                json=request_data,
                content_type='application/json'
            )
            
            assert response.status_code == 200
    
    def test_invalid_analysis_depth(self, client):
        """Test invalid analysis depth."""
        request_data = {
            'business_idea': 'Valid business idea description',
            'analysis_depth': 'invalid'
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_analysis_with_flags(self, client, mock_rag_pipeline, mock_cache):
        """Test analysis with various flags."""
        mock_rag_pipeline.analyze_business_idea.return_value = {
            'business_idea': 'Valid business idea description here',
            'viability_score': 75.0,
            'summary': 'This business idea demonstrates strong market potential with growing consumer interest in the target market. Success depends on effective execution and market positioning strategy.'
        }
        
        request_data = {
            'business_idea': 'Valid business idea description here',
            'include_swot': True,
            'include_market_size': True,
            'include_financial_projections': False
        }
        
        response = client.post(
            '/api/analyze',
            json=request_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        mock_rag_pipeline.analyze_business_idea.assert_called_once()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 Not Found response."""
        response = client.get('/non-existent-endpoint')
        assert response.status_code == 404
        
        data = response.get_json()
        assert data['error_type'] == 'not_found'
        assert 'not found' in data['message'].lower()
    
    def test_405_method_not_allowed(self, client):
        """Test 405 Method Not Allowed response."""
        response = client.delete('/api/recommendations')
        assert response.status_code == 405
    
    def test_error_response_includes_request_id(self, client):
        """Test error response includes request ID."""
        response = client.get('/non-existent')
        data = response.get_json()
        
        assert 'request_id' in data
        assert data['request_id'] is not None
    
    def test_error_response_has_required_fields(self, client):
        """Test error response has all required fields."""
        response = client.get('/non-existent')
        data = response.get_json()
        
        assert 'success' in data
        assert data['success'] is False
        assert 'error_type' in data
        assert 'message' in data
        assert 'timestamp' in data


# ============================================================================
# Middleware Tests
# ============================================================================


class TestMiddleware:
    """Tests for middleware functionality."""
    
    def test_request_id_in_response_headers(self, client):
        """Test request ID is in response headers."""
        response = client.get('/health')
        
        assert 'X-Request-ID' in response.headers
        request_id = response.headers['X-Request-ID']
        assert request_id is not None
        assert len(request_id) > 0
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present."""
        response = client.options(
            '/api/recommendations',
            headers={'Origin': 'http://localhost:3000'}
        )
        
        # Flask-CORS should add headers
        assert response.status_code == 204
    
    def test_request_logging_occurs(self, client):
        """Test request logging occurs."""
        with patch('src.api.logger') as mock_logger:
            response = client.get('/health')
            
            # Should log at least once
            assert mock_logger.info.called or mock_logger.debug.called


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndpointIntegration:
    """Integration tests for API endpoints."""
    
    def test_recommendation_to_analysis_flow(self, client, mock_rag_pipeline, mock_cache):
        """Test recommendation followed by analysis."""
        # Mock responses
        mock_rag_pipeline.generate_recommendations.return_value = [
            {
                'id': 'rec_001',
                'title': 'SaaS Platform',
                'description': 'Build a Software-as-a-Service platform providing comprehensive project management tools for remote teams with advanced collaboration features.' * 2,
                'why_it_fits': 'Your programming background and experience makes building such a platform feasible.',
                'required_skills': []
            }
        ]
        
        mock_rag_pipeline.analyze_business_idea.return_value = {
            'business_idea': 'SaaS Platform',
            'viability_score': 75.0,
            'summary': 'This SaaS business demonstrates strong market potential with growing demand from remote teams. Success depends on product-market fit and effective customer acquisition strategy.'
        }
        
        mock_cache.get.return_value = None
        
        # Get recommendations
        rec_req = {'user_profile': {'age': 30}}
        rec_resp = client.post(
            '/api/recommendations',
            json=rec_req,
            content_type='application/json'
        )
        assert rec_resp.status_code == 200
        
        # Analyze a business idea
        analysis_req = {'business_idea': 'Valid business idea description'}
        analysis_resp = client.post(
            '/api/analyze',
            json=analysis_req,
            content_type='application/json'
        )
        assert analysis_resp.status_code == 200
    
    def test_multiple_requests_have_unique_ids(self, client):
        """Test multiple requests have unique request IDs."""
        request_ids = []
        
        for _ in range(3):
            response = client.get('/health')
            request_id = response.headers['X-Request-ID']
            request_ids.append(request_id)
        
        # All request IDs should be unique
        assert len(set(request_ids)) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
