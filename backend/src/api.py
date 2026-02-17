"""Flask API server for BizBot backend.

Provides REST endpoints for business recommendations and idea analysis
using the RAG pipeline with Mistral AI and FAISS retrieval.
"""

import logging
import json
import time
import uuid
import os
from datetime import datetime, timezone, UTC
from functools import wraps
from typing import Dict, Any, Tuple
from pathlib import Path

from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

from src.config import Config, ConfigurationError
from src.mistral_client import MistralClient, MistralAPIError
from src.faiss_retriever import FAISSRetriever, FAISSRetrieverError
from src.prompt_builder import PromptBuilder
from src.rag_pipeline import RAGPipeline, RAGPipelineError
from src.document_processor import DocumentProcessor
from src.cache_manager import CacheManager
from src.recommendation_engine import RecommendationEngine
from src.schemas import (
    UserProfile,
    RecommendationRequest,
    AnalysisRequest,
    RecommendationResponse,
    AnalysisResponse,
    ErrorResponse,
    ErrorDetail,
    HealthCheck,
)


logger = logging.getLogger(__name__)


def create_app(config: Config = None) -> Flask:
    """Create and configure Flask application.
    
    Args:
        config: Configuration object (creates default if None)
        
    Returns:
        Configured Flask application
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if config is None:
        config = Config()
    
    # Determine frontend directory path
    backend_dir = Path(__file__).resolve().parent.parent
    frontend_dir = backend_dir.parent / 'frontend'
    
    app = Flask(
        __name__,
        static_folder=str(frontend_dir),
        static_url_path=''
    )
    
    # Store config and dependencies in app context
    app.config['CONFIG'] = config
    app.config['FRONTEND_DIR'] = str(frontend_dir)
    
    # Configure CORS
    CORS(
        app,
        origins=config.CORS_ORIGINS,
        methods=['GET', 'POST', 'OPTIONS'],
        allow_headers=['Content-Type', 'Authorization'],
        supports_credentials=True,
        max_age=3600
    )
    
    # Initialize dependencies
    try:
        app.config['MISTRAL_CLIENT'] = MistralClient(
            api_key=config.MISTRAL_API_KEY,
            model=config.MISTRAL_MODEL
        )
        logger.info("Mistral AI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client: {str(e)}")
        raise ConfigurationError(f"Mistral initialization failed") from e
    
    try:
        app.config['FAISS_RETRIEVER'] = FAISSRetriever(
            mistral_client=app.config['MISTRAL_CLIENT'],
            index_path=config.FAISS_INDEX_PATH,
            embedding_dimension=config.EMBEDDING_DIMENSION
        )
        logger.info("FAISS retriever initialized")
    except Exception as e:
        logger.warning(f"FAISS retriever initialization warning: {str(e)}")
    
    # Calculate max context tokens from reserved tokens
    max_context_tokens = config.MAX_TOKENS - config.RESERVED_TOKENS
    
    app.config['PROMPT_BUILDER'] = PromptBuilder(
        max_context_tokens=max_context_tokens
    )
    logger.info("Prompt builder initialized")
    
    app.config['RAG_PIPELINE'] = RAGPipeline(
        mistral_client=app.config['MISTRAL_CLIENT'],
        faiss_retriever=app.config['FAISS_RETRIEVER'],
        prompt_builder=app.config['PROMPT_BUILDER'],
        top_k_documents=config.TOP_K_DOCUMENTS
    )
    logger.info("RAG pipeline initialized")
    
    app.config['CACHE_MANAGER'] = CacheManager(
        ttl=config.CACHE_TTL,
        max_size=config.CACHE_MAX_SIZE
    )
    logger.info("Cache manager initialized")
    
    # Initialize recommendation engine with enriched ideas
    try:
        enriched_ideas_path = "data/documents/ideas_enriched.csv"
        app.config['RECOMMENDATION_ENGINE'] = RecommendationEngine(enriched_ideas_path)
        logger.info("Recommendation engine initialized")
    except Exception as e:
        logger.warning(f"Recommendation engine initialization failed: {str(e)}")
        app.config['RECOMMENDATION_ENGINE'] = None
    
    # Register middleware
    _register_middleware(app)
    
    # Register routes
    _register_frontend_routes(app)  # Serve frontend HTML pages
    _register_health_check(app)
    _register_profile_recommendations(app)  # NEW: Profile-based matching
    _register_recommendations(app)  # OLD: Legacy RAG-based
    _register_analysis(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    logger.info("Flask application initialized successfully")
    return app


def _register_frontend_routes(app: Flask) -> None:
    """Register routes to serve frontend HTML pages.
    
    Serves the frontend directory as static files so the entire
    app can be deployed as a single service on Render.
    """
    frontend_dir = app.config.get('FRONTEND_DIR', '')
    
    @app.route('/')
    def serve_landing():
        """Serve landing page."""
        return send_from_directory(frontend_dir, 'land_page.html')
    
    @app.route('/<path:filename>')
    def serve_frontend(filename):
        """Serve frontend static files (HTML, CSS, JS, images)."""
        # Don't interfere with API routes
        if filename.startswith('api/') or filename == 'health':
            from flask import abort
            abort(404)
        file_path = os.path.join(frontend_dir, filename)
        if os.path.isfile(file_path):
            return send_from_directory(frontend_dir, filename)
        # Fall back to landing page for SPA-like routing
        return send_from_directory(frontend_dir, 'land_page.html')


def _register_middleware(app: Flask) -> None:
    """Register request/response middleware.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def before_request():
        """Log incoming request and set up context."""
        g.request_id = str(uuid.uuid4())
        g.request_start_time = time.time()
        
        # Log request
        logger.info(
            f"[{g.request_id}] {request.method} {request.path} - "
            f"Client: {request.remote_addr}"
        )
        
        if request.is_json and request.get_json():
            # Log request body (anonymized)
            try:
                body = request.get_json()
                logger.debug(f"[{g.request_id}] Request body keys: {list(body.keys())}")
            except Exception as e:
                logger.debug(f"[{g.request_id}] Could not log request body: {str(e)}")
    
    @app.after_request
    def after_request(response):
        """Log response and metrics."""
        elapsed_ms = (time.time() - g.request_start_time) * 1000
        
        logger.info(
            f"[{g.request_id}] {request.method} {request.path} - "
            f"Status: {response.status_code} - "
            f"Time: {elapsed_ms:.2f}ms"
        )
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = g.request_id
        
        return response


def _register_health_check(app: Flask) -> None:
    """Register health check endpoint.
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Check API health status.
        
        Returns:
            JSON health status response
        """
        config = app.config['CONFIG']
        mistral_client = app.config['MISTRAL_CLIENT']
        faiss_retriever = app.config['FAISS_RETRIEVER']
        
        # Determine health status
        status = "healthy"
        faiss_loaded = not faiss_retriever.is_empty if faiss_retriever else False
        
        if not faiss_loaded:
            status = "degraded"
        
        health = HealthCheck(
            status=status,
            version="1.0.0",
            faiss_index_loaded=faiss_loaded,
            mistral_api_available=True  # Would check API in production
        )
        
        return jsonify(health.model_dump()), 200

def _register_profile_recommendations(app: Flask) -> None:
    """Register profile-based recommendation endpoint (structured matching).
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/api/match', methods=['POST', 'OPTIONS'])
    def get_profile_matches():
        """Get startup idea matches based on structured user profile.
        
        This is a deterministic scoring engine, NOT an LLM-based chatbot.
        Returns ranked ideas with calculated match percentages.
        
        Request JSON:
            {
                "skills": ["Python", "Marketing", "Sales"],
                "experience_level": "Intermediate",
                "industry_interest": ["AI/ML", "E-commerce"],
                "business_model_preference":"B2B",
                "starting_capital": 5000,
                "desired_income": 10000,
                "time_commitment": "20-40 hours",
                "network_strength": "Moderate",
                "existing_assets": ["website", "social media"]
            }
            
        Returns:
            JSON with ranked ideas and match scores:
            {
                "success": true,
                "matches": [{
                    "match_percentage": 94,
                    "match_score": 0.94,
                    "idea_text": "...",
                    "short_summary": "...",
                    "domain": "AI/ML",
                    "business_model": "B2B",
                    "difficulty": "Medium",
                    "scalability": "High",
                    "estimated_cost_bucket": "1000-10000",
                    "required_skills": "Python, ML, API development",
                    "target_customer": "Small businesses",
                    "explanation": "Why this matches your profile",
                    "score_breakdown": {...}
                }],
                "total_candidates": 50,
                "filtered_candidates": 25
            }
        """
        try:
            if request.method == 'OPTIONS':
                return '', 204
            
            # Validate request
            if not request.is_json:
                error_response = ErrorResponse(
                    error_type="bad_request",
                    message="Request must be JSON",
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            request_data = request.get_json()
            
            # Parse and validate user profile
            try:
                user_profile = UserProfile(**request_data)
                logger.debug(f"[{g.request_id}] Validated user profile")
            except ValidationError as e:
                logger.warning(f"[{g.request_id}] Validation error: {e}")
                errors = [
                    ErrorDetail(
                        field=str(err.get('loc', ['unknown'])[0]),
                        message=err['msg'],
                        error_code=err.get('type')
                    )
                    for err in e.errors()
                ]
                error_response = ErrorResponse(
                    error_type="validation_error",
                    message="Profile validation failed",
                    details=errors,
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            # Check if recommendation engine is available
            rec_engine = app.config.get('RECOMMENDATION_ENGINE')
            if not rec_engine:
                error_response = ErrorResponse(
                    error_type="service_unavailable",
                    message="Recommendation engine not available. Ensure ideas_enriched.csv exists.",
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 503
            
            # Check cache
            cache_manager = app.config['CACHE_MANAGER']
            cache_key = cache_manager._generate_key({
                'endpoint': 'profile_match',
                'profile': user_profile.model_dump()
            })
            
            cached_response = cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"[{g.request_id}] Cache hit for profile matches")
                return jsonify(cached_response), 200
            
            # Get recommendations from scoring engine
            logger.info(f"[{g.request_id}] Computing profile matches")
            gen_start = time.time()
            
            try:
                # Get top 10 matches
                top_n = request_data.get('top_n', 10)
                recommendations = rec_engine.get_recommendations(user_profile, top_n=top_n)
                
                # Add explanations to each recommendation
                for rec in recommendations:
                    rec['explanation'] = rec_engine.generate_match_explanation(rec, user_profile)
                
                generation_time_ms = (time.time() - gen_start) * 1000
                
                # Build response
                response_data = {
                    "success": True,
                    "matches": recommendations,
                    "total_candidates": len(rec_engine.ideas_df),
                    "scored_candidates": len(rec_engine.ideas_df),
                    "generation_time_ms": generation_time_ms,
                    "request_id": g.request_id
                }
                
                # Cache response
                cache_manager.set(cache_key, response_data)
                
                logger.info(
                    f"[{g.request_id}] Generated {len(recommendations)} matches "
                    f"in {generation_time_ms:.2f}ms"
                )
                
                return jsonify(response_data), 200
                
            except Exception as e:
                logger.error(f"[{g.request_id}] Scoring error: {str(e)}", exc_info=True)
                error_response = ErrorResponse(
                    error_type="server_error",
                    message=f"Failed to compute matches: {str(e)}",
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 500
            
        except Exception as e:
            logger.error(f"[{g.request_id}] Match endpoint error: {str(e)}", exc_info=True)
            error_response = ErrorResponse(
                error_type="server_error",
                message="Failed to process match request",
                request_id=g.request_id
            )
            return jsonify(error_response.model_dump()), 500

def _register_recommendations(app: Flask) -> None:
    """Register business recommendations endpoint.
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/api/recommendations', methods=['POST', 'OPTIONS'])
    def get_recommendations():
        """Generate personalized business recommendations.
        
        Request JSON:
            {
                "user_profile": {...},
                "num_recommendations": 3,
                "include_reasoning": true,
                "include_market_data": true
            }
            
        Returns:
            JSON response with recommendations list
        """
        try:
            if request.method == 'OPTIONS':
                return '', 204
            
            # Validate request
            if not request.is_json:
                error_response = ErrorResponse(
                    error_type="bad_request",
                    message="Request must be JSON with Content-Type: application/json",
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            request_data = request.get_json()
            
            # Parse and validate request
            try:
                api_request = RecommendationRequest(**request_data)
                logger.debug(f"[{g.request_id}] Validated recommendation request")
            except ValidationError as e:
                logger.warning(f"[{g.request_id}] Validation error: {e}")
                errors = [
                    ErrorDetail(
                        field=err.get('loc', ['unknown'])[0],
                        message=err['msg'],
                        error_code=err.get('type')
                    )
                    for err in e.errors()
                ]
                error_response = ErrorResponse(
                    error_type="validation_error",
                    message="Request validation failed",
                    details=errors,
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            # Check cache
            cache_manager = app.config['CACHE_MANAGER']
            cache_key = cache_manager._generate_key({
                'endpoint': 'recommendations',
                'profile': api_request.user_profile.model_dump(),
                'num': api_request.num_recommendations
            })
            
            cached_response = cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"[{g.request_id}] Cache hit for recommendations")
                return jsonify(cached_response), 200
            
            # Generate recommendations
            logger.info(f"[{g.request_id}] Generating recommendations")
            gen_start = time.time()
            
            rag_pipeline = app.config['RAG_PIPELINE']
            recommendations = rag_pipeline.generate_recommendations(
                user_profile=api_request.user_profile.model_dump(),
                num_recommendations=api_request.num_recommendations,
                include_reasoning=api_request.include_reasoning,
                include_market_data=api_request.include_market_data
            )
            
            generation_time_ms = (time.time() - gen_start) * 1000
            
            # Build response
            response = RecommendationResponse(
                success=True,
                recommendations=recommendations,
                generation_time_ms=generation_time_ms,
                model_used=app.config['CONFIG'].MISTRAL_MODEL,
                context_sources=len(recommendations)  # Would be actual count
            )
            
            # Cache response
            response_dict = response.model_dump()
            cache_manager.set(cache_key, response_dict)
            
            logger.info(
                f"[{g.request_id}] Generated {len(recommendations)} recommendations "
                f"in {generation_time_ms:.2f}ms"
            )
            
            return jsonify(response_dict), 200
            
        except Exception as e:
            logger.error(f"[{g.request_id}] Recommendations error: {str(e)}", exc_info=True)
            error_response = ErrorResponse(
                error_type="server_error",
                message="Failed to generate recommendations",
                request_id=g.request_id
            )
            return jsonify(error_response.model_dump()), 500


def _register_analysis(app: Flask) -> None:
    """Register business analysis endpoint.
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/api/analyze', methods=['POST', 'OPTIONS'])
    def analyze_business():
        """Analyze a business idea.
        
        Request JSON:
            {
                "business_idea": "...",
                "user_profile": {...} (optional),
                "analysis_depth": "detailed",
                "include_swot": true,
                "include_market_size": true,
                "include_financial_projections": false
            }
            
        Returns:
            JSON response with analysis
        """
        try:
            if request.method == 'OPTIONS':
                return '', 204
            
            # Validate request
            if not request.is_json:
                error_response = ErrorResponse(
                    error_type="bad_request",
                    message="Request must be JSON with Content-Type: application/json",
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            request_data = request.get_json()
            
            # Parse and validate request
            try:
                api_request = AnalysisRequest(**request_data)
                logger.debug(f"[{g.request_id}] Validated analysis request")
            except ValidationError as e:
                logger.warning(f"[{g.request_id}] Validation error: {e}")
                errors = [
                    ErrorDetail(
                        field=err.get('loc', ['unknown'])[0],
                        message=err['msg'],
                        error_code=err.get('type')
                    )
                    for err in e.errors()
                ]
                error_response = ErrorResponse(
                    error_type="validation_error",
                    message="Request validation failed",
                    details=errors,
                    request_id=g.request_id
                )
                return jsonify(error_response.model_dump()), 400
            
            # Check cache
            cache_manager = app.config['CACHE_MANAGER']
            profile_dict = (
                api_request.user_profile.model_dump()
                if api_request.user_profile else None
            )
            cache_key = cache_manager._generate_key({
                'endpoint': 'analyze',
                'idea': api_request.business_idea,
                'depth': api_request.analysis_depth,
                'profile': profile_dict
            })
            
            cached_response = cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"[{g.request_id}] Cache hit for analysis")
                return jsonify(cached_response), 200
            
            # Generate analysis
            logger.info(f"[{g.request_id}] Analyzing business idea")
            analysis_start = time.time()
            
            rag_pipeline = app.config['RAG_PIPELINE']
            analysis = rag_pipeline.analyze_business_idea(
                business_idea=api_request.business_idea,
                user_profile=api_request.user_profile.model_dump() if api_request.user_profile else None
            )
            
            generation_time_ms = (time.time() - analysis_start) * 1000
            
            # Build response — pass raw analysis dict from Mistral
            response_data = {
                "success": True,
                "analysis": analysis,
                "business_idea": api_request.business_idea,
                "generation_time_ms": generation_time_ms,
                "model_used": app.config['CONFIG'].MISTRAL_MODEL,
                "request_id": g.request_id
            }
            
            # Cache response
            cache_manager.set(cache_key, response_data)
            
            logger.info(
                f"[{g.request_id}] Analysis completed "
                f"in {generation_time_ms:.2f}ms"
            )
            
            return jsonify(response_data), 200
            
        except RAGPipelineError as e:
            logger.error(f"[{g.request_id}] RAG pipeline error: {str(e)}")
            error_response = ErrorResponse(
                error_type="model_error",
                message=str(e),
                request_id=g.request_id
            )
            return jsonify(error_response.model_dump()), 500
        except Exception as e:
            logger.error(f"[{g.request_id}] Analysis error: {str(e)}", exc_info=True)
            error_response = ErrorResponse(
                error_type="server_error",
                message="Failed to analyze business idea",
                request_id=g.request_id
            )
            return jsonify(error_response.model_dump()), 500


def _register_error_handlers(app: Flask) -> None:
    """Register global error handlers.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        error_response = ErrorResponse(
            error_type="bad_request",
            message="Bad request: " + str(error),
            request_id=getattr(g, 'request_id', 'unknown')
        )
        return jsonify(error_response.model_dump()), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        error_response = ErrorResponse(
            error_type="not_found",
            message=f"Endpoint not found: {request.path}",
            request_id=getattr(g, 'request_id', 'unknown')
        )
        return jsonify(error_response.model_dump()), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        error_response = ErrorResponse(
            error_type="bad_request",
            message=f"Method {request.method} not allowed for {request.path}",
            request_id=getattr(g, 'request_id', 'unknown')
        )
        return jsonify(error_response.model_dump()), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors."""
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f"[{request_id}] Internal server error: {str(error)}", exc_info=True)
        
        error_response = ErrorResponse(
            error_type="server_error",
            message="Internal server error",
            request_id=request_id
        )
        return jsonify(error_response.model_dump()), 500


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run app
    config = Config()
    app = create_app(config)
    
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=(config.FLASK_ENV == 'development')
    )
