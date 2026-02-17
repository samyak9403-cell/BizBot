"""Unit tests for API validation schemas.

Tests all Pydantic models for request and response validation,
including edge cases and error handling.
"""

import pytest
import json
from datetime import datetime
from pydantic import ValidationError

from src.schemas import (
    UserProfile,
    RecommendationRequest,
    AnalysisRequest,
    BusinessRecommendation,
    RecommendationResponse,
    SWOTAnalysis,
    BusinessAnalysis,
    AnalysisResponse,
    ErrorDetail,
    ErrorResponse,
    PaginationParams,
    HealthCheck,
)


# ============================================================================
# UserProfile Tests
# ============================================================================


class TestUserProfile:
    """Tests for UserProfile schema."""
    
    def test_create_minimal_profile(self):
        """Test creating a valid minimal user profile."""
        profile = UserProfile()
        assert profile.age is None
        assert profile.skills == []
        assert profile.interests == []
    
    def test_create_full_profile(self):
        """Test creating a full user profile."""
        profile = UserProfile(
            age=35,
            education_level="masters",
            work_experience=10,
            skills=["Python", "Leadership", "Marketing"],
            interests=["Tech", "Business", "Innovation"],
            business_type="product",
            industry_preference="SaaS",
            startup_capital=50000,
            time_commitment="full_time",
            location="San Francisco, CA",
            risk_tolerance="medium",
            goals=["profit", "impact"],
            target_revenue=500000,
            growth_timeline="3_years"
        )
        
        assert profile.age == 35
        assert profile.education_level == "masters"
        assert len(profile.skills) == 3
        assert profile.startup_capital == 50000
    
    def test_age_validation(self):
        """Test age validation."""
        # Valid age
        profile = UserProfile(age=25)
        assert profile.age == 25
        
        # Too young
        with pytest.raises(ValidationError) as exc:
            UserProfile(age=17)
        assert "greater than or equal to 18" in str(exc.value)
        
        # Too old
        with pytest.raises(ValidationError) as exc:
            UserProfile(age=121)
        assert "less than or equal to 120" in str(exc.value)
    
    def test_education_level_validation(self):
        """Test education level enum validation."""
        valid_levels = ["high_school", "bachelors", "masters", "phd", "other"]
        
        for level in valid_levels:
            profile = UserProfile(education_level=level)
            assert profile.education_level == level
        
        # Invalid level
        with pytest.raises(ValidationError):
            UserProfile(education_level="invalid_level")
    
    def test_business_type_validation(self):
        """Test business type enum validation."""
        valid_types = ["service", "product", "hybrid", "online", "offline"]
        
        for btype in valid_types:
            profile = UserProfile(business_type=btype)
            assert profile.business_type == btype
        
        # Invalid type
        with pytest.raises(ValidationError):
            UserProfile(business_type="invalid_type")
    
    def test_time_commitment_validation(self):
        """Test time commitment enum validation."""
        valid_commitments = ["full_time", "part_time", "flexible"]
        
        for commitment in valid_commitments:
            profile = UserProfile(time_commitment=commitment)
            assert profile.time_commitment == commitment
        
        # Invalid commitment
        with pytest.raises(ValidationError):
            UserProfile(time_commitment="invalid")
    
    def test_risk_tolerance_validation(self):
        """Test risk tolerance enum validation."""
        valid_tolerances = ["low", "medium", "high"]
        
        for tolerance in valid_tolerances:
            profile = UserProfile(risk_tolerance=tolerance)
            assert profile.risk_tolerance == tolerance
        
        # Invalid tolerance
        with pytest.raises(ValidationError):
            UserProfile(risk_tolerance="extreme")
    
    def test_skills_validation(self):
        """Test skills list validation."""
        # Valid skills
        profile = UserProfile(skills=["Python", "JavaScript", "Leadership"])
        assert len(profile.skills) == 3
        
        # Empty string in skills
        with pytest.raises(ValidationError):
            UserProfile(skills=["Python", "", "JavaScript"])
    
    def test_capital_validation(self):
        """Test startup capital validation."""
        # Valid capital
        profile = UserProfile(startup_capital=100000)
        assert profile.startup_capital == 100000
        
        # Negative capital
        with pytest.raises(ValidationError):
            UserProfile(startup_capital=-1000)
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in UserProfile."""
        profile = UserProfile(
            age=30,
            custom_field="custom_value"
        )
        assert profile.age == 30
        assert profile.custom_field == "custom_value"


# ============================================================================
# RecommendationRequest Tests
# ============================================================================


class TestRecommendationRequest:
    """Tests for RecommendationRequest schema."""
    
    def test_create_valid_request(self):
        """Test creating a valid recommendation request."""
        profile = UserProfile(skills=["Python", "Data Analysis"])
        request = RecommendationRequest(user_profile=profile)
        
        assert request.user_profile == profile
        assert request.num_recommendations == 3
        assert request.include_reasoning is True
        assert request.include_market_data is True
    
    def test_custom_num_recommendations(self):
        """Test custom number of recommendations."""
        profile = UserProfile()
        request = RecommendationRequest(
            user_profile=profile,
            num_recommendations=5
        )
        
        assert request.num_recommendations == 5
    
    def test_num_recommendations_validation(self):
        """Test num_recommendations validation."""
        profile = UserProfile()
        
        # Too low
        with pytest.raises(ValidationError):
            RecommendationRequest(user_profile=profile, num_recommendations=0)
        
        # Too high
        with pytest.raises(ValidationError):
            RecommendationRequest(user_profile=profile, num_recommendations=11)
    
    def test_flags_configuration(self):
        """Test request flags configuration."""
        profile = UserProfile()
        request = RecommendationRequest(
            user_profile=profile,
            include_reasoning=False,
            include_market_data=False
        )
        
        assert request.include_reasoning is False
        assert request.include_market_data is False


# ============================================================================
# AnalysisRequest Tests
# ============================================================================


class TestAnalysisRequest:
    """Tests for AnalysisRequest schema."""
    
    def test_create_valid_analysis_request(self):
        """Test creating a valid analysis request."""
        request = AnalysisRequest(
            business_idea="A platform that connects freelance developers with startups looking for technical co-founders"
        )
        
        assert len(request.business_idea) > 10
        assert request.analysis_depth == "detailed"
        assert request.include_swot is True
    
    def test_business_idea_validation(self):
        """Test business idea validation."""
        # Too short
        with pytest.raises(ValidationError):
            AnalysisRequest(business_idea="Too short")
        
        # Too long
        with pytest.raises(ValidationError):
            AnalysisRequest(business_idea="x" * 5001)
        
        # Empty
        with pytest.raises(ValidationError):
            AnalysisRequest(business_idea="")
    
    def test_analysis_depth_validation(self):
        """Test analysis depth enum validation."""
        valid_depths = ["basic", "detailed", "comprehensive"]
        
        for depth in valid_depths:
            request = AnalysisRequest(
                business_idea="Valid business idea description here",
                analysis_depth=depth
            )
            assert request.analysis_depth == depth
        
        # Invalid depth
        with pytest.raises(ValidationError):
            AnalysisRequest(
                business_idea="Valid business idea",
                analysis_depth="invalid"
            )
    
    def test_with_user_profile(self):
        """Test analysis request with user profile."""
        profile = UserProfile(age=30, skills=["Marketing"])
        request = AnalysisRequest(
            business_idea="E-commerce platform for local artisans",
            user_profile=profile
        )
        
        assert request.user_profile == profile


# ============================================================================
# BusinessRecommendation Tests
# ============================================================================


class TestBusinessRecommendation:
    """Tests for BusinessRecommendation schema."""
    
    def test_create_valid_recommendation(self):
        """Test creating a valid business recommendation."""
        rec = BusinessRecommendation(
            id="rec_001",
            title="SaaS Platform",
            description="Build a project management SaaS tool for remote teams " * 5,
            why_it_fits="Your programming background and leadership experience makes this ideal",
            required_skills=["Software Development", "Product Management"],
            startup_capital_estimate=50000,
            time_to_profitability="18_months",
            profit_potential="high",
            market_size="$15B annually",
            key_challenges=["Competition", "Customer acquisition"],
            next_steps=["Market research", "Build MVP"],
            relevance_score=0.85
        )
        
        assert rec.id == "rec_001"
        assert rec.relevance_score == 0.85
        assert len(rec.required_skills) == 2
    
    def test_title_validation(self):
        """Test title validation."""
        # Too short (less than 5 chars)
        with pytest.raises(ValidationError):
            BusinessRecommendation(
                id="rec_001",
                title="Bad",
                description="x" * 100,
                why_it_fits="x" * 50
            )
        
        # Too long
        with pytest.raises(ValidationError):
            BusinessRecommendation(
                id="rec_001",
                title="x" * 201,
                description="x" * 100,
                why_it_fits="x" * 50
            )
    
    def test_description_validation(self):
        """Test description validation."""
        # Too short
        with pytest.raises(ValidationError):
            BusinessRecommendation(
                id="rec_001",
                title="Valid Title",
                description="Too short",
                why_it_fits="x" * 50
            )
    
    def test_relevance_score_validation(self):
        """Test relevance score validation."""
        # Valid score
        rec = BusinessRecommendation(
            id="rec_001",
            title="SaaS Platform",
            description="x" * 100,
            why_it_fits="x" * 50,
            relevance_score=0.75
        )
        assert rec.relevance_score == 0.75
        
        # Score too high
        with pytest.raises(ValidationError):
            BusinessRecommendation(
                id="rec_001",
                title="SaaS Platform",
                description="x" * 100,
                why_it_fits="x" * 50,
                relevance_score=1.5
            )
        
        # Score too low
        with pytest.raises(ValidationError):
            BusinessRecommendation(
                id="rec_001",
                title="SaaS Platform",
                description="x" * 100,
                why_it_fits="x" * 50,
                relevance_score=-0.1
            )


# ============================================================================
# RecommendationResponse Tests
# ============================================================================


class TestRecommendationResponse:
    """Tests for RecommendationResponse schema."""
    
    def test_create_valid_response(self):
        """Test creating a valid recommendation response."""
        rec = BusinessRecommendation(
            id="rec_001",
            title="SaaS Platform",
            description="x" * 100,
            why_it_fits="x" * 50
        )
        
        response = RecommendationResponse(
            success=True,
            recommendations=[rec],
            generation_time_ms=1234.5,
            model_used="mistral-large-latest",
            context_sources=5
        )
        
        assert response.success is True
        assert len(response.recommendations) == 1
        assert response.generation_time_ms == 1234.5
    
    def test_empty_recommendations_invalid(self):
        """Test that empty recommendations list is invalid."""
        with pytest.raises(ValidationError):
            RecommendationResponse(
                success=True,
                recommendations=[],
                generation_time_ms=100
            )
    
    def test_unique_ids_required(self):
        """Test that recommendation IDs must be unique."""
        rec1 = BusinessRecommendation(
            id="rec_001",
            title="Title 1",
            description="x" * 100,
            why_it_fits="x" * 50
        )
        rec2 = BusinessRecommendation(
            id="rec_001",
            title="Title 2",
            description="x" * 100,
            why_it_fits="x" * 50
        )
        
        with pytest.raises(ValidationError):
            RecommendationResponse(
                success=True,
                recommendations=[rec1, rec2],
                generation_time_ms=100
            )


# ============================================================================
# BusinessAnalysis Tests
# ============================================================================


class TestBusinessAnalysis:
    """Tests for BusinessAnalysis schema."""
    
    def test_create_valid_analysis(self):
        """Test creating a valid business analysis."""
        analysis = BusinessAnalysis(
            business_idea="E-commerce platform for artisans",
            viability_score=75.5,
            summary="This business idea has good potential with moderate competition",
            market_analysis="Market is growing at 15% annually",
            market_size="$5B market",
            target_audience=["Small businesses", "Individual artisans"],
            revenue_model="Commission per transaction",
            startup_capital_estimate=100000,
            profitability_timeline="12 months",
            required_skills=["E-commerce", "Web Development"],
            key_risks=["High competition", "Payment processing"]
        )
        
        assert analysis.viability_score == 75.5
        assert len(analysis.target_audience) == 2
    
    def test_viability_score_validation(self):
        """Test viability score validation."""
        # Valid score
        analysis = BusinessAnalysis(
            business_idea="Valid idea" * 10,
            viability_score=50.0,
            summary="x" * 100
        )
        assert analysis.viability_score == 50.0
        
        # Invalid score
        with pytest.raises(ValidationError):
            BusinessAnalysis(
                business_idea="Valid idea" * 10,
                viability_score=150.0,
                summary="x" * 100
            )
    
    def test_swot_analysis_optional(self):
        """Test that SWOT analysis is optional."""
        analysis = BusinessAnalysis(
            business_idea="Valid idea" * 10,
            viability_score=50.0,
            summary="x" * 100,
            swot=None
        )
        
        assert analysis.swot is None
    
    def test_swot_analysis_included(self):
        """Test SWOT analysis when included."""
        swot = SWOTAnalysis(
            strengths=["Strong team", "Unique value prop"],
            weaknesses=["Limited funding", "New market entrants"],
            opportunities=["Growing market", "International expansion"],
            threats=["Large competitors", "Economic downturn"]
        )
        
        analysis = BusinessAnalysis(
            business_idea="Valid idea" * 10,
            viability_score=75.0,
            summary="x" * 100,
            swot=swot
        )
        
        assert analysis.swot is not None
        assert len(analysis.swot.strengths) == 2


# ============================================================================
# AnalysisResponse Tests
# ============================================================================


class TestAnalysisResponse:
    """Tests for AnalysisResponse schema."""
    
    def test_create_valid_analysis_response(self):
        """Test creating a valid analysis response."""
        analysis = BusinessAnalysis(
            business_idea="Valid idea" * 10,
            viability_score=75.0,
            summary="x" * 100
        )
        
        response = AnalysisResponse(
            success=True,
            analysis=analysis,
            generation_time_ms=2500.0,
            context_sources=3
        )
        
        assert response.success is True
        assert response.generation_time_ms == 2500.0


# ============================================================================
# ErrorResponse Tests
# ============================================================================


class TestErrorResponse:
    """Tests for ErrorResponse schema."""
    
    def test_create_simple_error(self):
        """Test creating a simple error response."""
        error = ErrorResponse(
            error_type="validation_error",
            message="Invalid request format"
        )
        
        assert error.success is False
        assert error.error_type == "validation_error"
        assert error.message == "Invalid request format"
    
    def test_create_detailed_error(self):
        """Test creating an error response with details."""
        details = [
            ErrorDetail(field="age", message="Must be at least 18"),
            ErrorDetail(field="skills", message="At least one skill required")
        ]
        
        error = ErrorResponse(
            error_type="validation_error",
            message="User profile validation failed",
            details=details,
            request_id="req_12345"
        )
        
        assert len(error.details) == 2
        assert error.request_id == "req_12345"
    
    def test_error_type_normalization(self):
        """Test that error types are normalized to lowercase."""
        error = ErrorResponse(
            error_type="SERVER_ERROR",
            message="Internal error"
        )
        
        assert error.error_type == "server_error"
    
    def test_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        error = ErrorResponse(
            error_type="server_error",
            message="Test error"
        )
        
        assert error.timestamp is not None
        assert isinstance(error.timestamp, datetime)


# ============================================================================
# PaginationParams Tests
# ============================================================================


class TestPaginationParams:
    """Tests for PaginationParams schema."""
    
    def test_create_default_pagination(self):
        """Test creating pagination with defaults."""
        params = PaginationParams()
        
        assert params.skip == 0
        assert params.limit == 10
    
    def test_create_custom_pagination(self):
        """Test creating custom pagination."""
        params = PaginationParams(skip=20, limit=50)
        
        assert params.skip == 20
        assert params.limit == 50
    
    def test_skip_validation(self):
        """Test skip parameter validation."""
        # Negative skip
        with pytest.raises(ValidationError):
            PaginationParams(skip=-1)
    
    def test_limit_validation(self):
        """Test limit parameter validation."""
        # Zero limit
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)
        
        # Limit too high
        with pytest.raises(ValidationError):
            PaginationParams(limit=101)


# ============================================================================
# HealthCheck Tests
# ============================================================================


class TestHealthCheck:
    """Tests for HealthCheck schema."""
    
    def test_create_healthy_status(self):
        """Test creating health check with healthy status."""
        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            faiss_index_loaded=True,
            mistral_api_available=True
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
    
    def test_create_degraded_status(self):
        """Test creating health check with degraded status."""
        health = HealthCheck(
            status="degraded",
            version="1.0.0",
            faiss_index_loaded=True,
            mistral_api_available=False
        )
        
        assert health.status == "degraded"
    
    def test_status_validation(self):
        """Test status validation."""
        # Invalid status
        with pytest.raises(ValidationError):
            HealthCheck(status="invalid", version="1.0.0")


# ============================================================================
# SWOTAnalysis Tests
# ============================================================================


class TestSWOTAnalysis:
    """Tests for SWOTAnalysis schema."""
    
    def test_create_valid_swot(self):
        """Test creating valid SWOT analysis."""
        swot = SWOTAnalysis(
            strengths=["Strong team", "Unique tech"],
            weaknesses=["Limited funding"],
            opportunities=["Growing market"],
            threats=["New competitors"]
        )
        
        assert len(swot.strengths) == 2
        assert len(swot.weaknesses) == 1
    
    def test_swot_empty_lists_allowed(self):
        """Test that empty lists are allowed in SWOT."""
        swot = SWOTAnalysis()
        
        assert swot.strengths == []
        assert swot.weaknesses == []
        assert swot.opportunities == []
        assert swot.threats == []
    
    def test_swot_empty_string_validation(self):
        """Test that empty strings are not allowed in SWOT items."""
        with pytest.raises(ValidationError):
            SWOTAnalysis(strengths=["Valid", ""])


# ============================================================================
# Integration Tests
# ============================================================================


class TestSchemaIntegration:
    """Integration tests for multiple schemas."""
    
    def test_request_response_workflow(self):
        """Test a complete request-response workflow."""
        # Create request
        profile = UserProfile(
            age=30,
            skills=["Python", "Business Analysis"],
            business_type="product",
            startup_capital=100000
        )
        
        request = RecommendationRequest(
            user_profile=profile,
            num_recommendations=3
        )
        
        assert request.user_profile.age == 30
        
        # Create response
        recommendations = [
            BusinessRecommendation(
                id=f"rec_{i:03d}",
                title=f"Idea {i}",
                description="x" * 100,
                why_it_fits="y" * 50,
                relevance_score=0.8 - (i * 0.1)
            )
            for i in range(1, 4)
        ]
        
        response = RecommendationResponse(
            recommendations=recommendations,
            generation_time_ms=1500
        )
        
        assert len(response.recommendations) == 3
    
    def test_analysis_request_response_workflow(self):
        """Test complete analysis workflow."""
        # Create request
        request = AnalysisRequest(
            business_idea="AI-powered customer service platform for SMBs",
            analysis_depth="comprehensive",
            include_swot=True
        )
        
        # Create response
        swot = SWOTAnalysis(
            strengths=["Growing AI adoption"],
            weaknesses=["High development cost"],
            opportunities=["Global market"],
            threats=["Major tech companies entering space"]
        )
        
        analysis = BusinessAnalysis(
            business_idea=request.business_idea,
            viability_score=85.0,
            summary="This business idea has strong market potential and good scalability with a focused go-to-market strategy.",
            swot=swot
        )
        
        response = AnalysisResponse(
            analysis=analysis,
            generation_time_ms=3000
        )
        
        assert response.analysis.viability_score == 85.0
        assert response.analysis.swot is not None
    
    def test_json_serialization(self):
        """Test that all schemas can be serialized to JSON."""
        profile = UserProfile(age=30, skills=["Python"])
        request = RecommendationRequest(user_profile=profile)
        
        # Should be JSON serializable
        request_json = request.model_dump_json()
        assert isinstance(request_json, str)
        
        # Should be deserializable
        request_restored = RecommendationRequest(**json.loads(request_json))
        assert request_restored.user_profile.age == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
