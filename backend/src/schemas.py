"""Data validation schemas for BizBot API.

Defines Pydantic models for request and response validation,
ensuring type safety and data consistency across the API.
"""

import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone, UTC

import numpy as np
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict


logger = logging.getLogger(__name__)


# ============================================================================
# API REQUEST SCHEMAS
# ============================================================================


class UserProfile(BaseModel):
    """User profile from questionnaire responses.
    
    Contains user answers from the structured questionnaire that guide
    startup idea matching. This is used for deterministic scoring, not LLM chat.
    """
    
    model_config = ConfigDict(extra='allow')
    
    # Skills & Experience
    skills: List[str] = Field(
        default_factory=list, 
        description="User's skill profile (e.g., Tech, Marketing, Sales, Design, Finance)",
        min_length=0
    )
    experience_level: str = Field(
        default="Intermediate",
        description="Experience level: Beginner, Intermediate, Expert"
    )
    
    # Industry & Domain Preferences
    industry_interest: List[str] = Field(
        default_factory=list,
        description="Industries of interest (FinTech, HealthTech, EdTech, AI/ML, E-commerce, etc.)",
        min_length=0
    )
    
    # Business Model Preferences
    business_model_preference: Optional[str] = Field(
        None,
        description="Preferred business model: B2B, B2C, or Both"
    )
    
    # Financial Capacity
    starting_capital: Optional[float] = Field(
        None,
        description="Available startup capital in USD",
        ge=0
    )
    desired_income: Optional[float] = Field(
        None,
        description="Desired monthly income in USD",
        ge=0
    )
    
    # Time & Commitment
    time_commitment: str = Field(
        default="flexible",
        description="Weekly time availability: part_time, full_time, flexible"
    )
    
    # Network & Resources
    network_strength: str = Field(
        default="Moderate",
        description="Professional network strength: Weak, Moderate, Strong"
    )
    existing_assets: List[str] = Field(
        default_factory=list,
        description="Assets user already owns (e.g., website, social media following, equipment)",
        min_length=0
    )
    
    # Legacy fields for backward compatibility
    age: Optional[int] = Field(None, description="User age", ge=18, le=120)
    education_level: Optional[str] = Field(
        None, 
        description="Education level (high_school, bachelors, masters, phd, other)"
    )
    work_experience: Optional[int] = Field(
        None, 
        description="Years of work experience", 
        ge=0, 
        le=70
    )
    interests: List[str] = Field(
        default_factory=list,
        description="General interests", 
        min_length=0
    )
    business_type: Optional[str] = Field(
        None,
        description="Preferred business type (service, product, hybrid)"
    )
    industry_preference: Optional[str] = Field(
        None,
        description="Preferred industry (legacy field, use industry_interest instead)"
    )
    startup_capital: Optional[int] = Field(
        None,
        description="Legacy field for starting_capital",
        ge=0
    )
    location: Optional[str] = Field(
        None,
        description="Geographic location"
    )
    risk_tolerance: Optional[str] = Field(
        None,
        description="Risk tolerance level (low, medium, high)"
    )
    goals: List[str] = Field(
        default_factory=list,
        description="Business goals"
    )
    target_revenue: Optional[int] = Field(
        None,
        description="Target annual revenue in USD",
        ge=0
    )
    growth_timeline: Optional[str] = Field(
        None,
        description="Target growth timeline"
    )
    
    @field_validator('skills', 'interests', 'goals')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Ensure all list items are non-empty strings."""
        if v is not None:
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All list items must be non-empty strings")
        return v
    
    @field_validator('education_level')
    @classmethod
    def validate_education_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate education level enum."""
        valid_options = {'high_school', 'bachelors', 'masters', 'phd', 'other'}
        if v is not None and v.lower() not in valid_options:
            raise ValueError(
                f"education_level must be one of {valid_options}, got: {v}"
            )
        return v.lower() if v is not None else None
    
    @field_validator('business_type')
    @classmethod
    def validate_business_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate business type enum."""
        valid_options = {'service', 'product', 'hybrid', 'online', 'offline'}
        if v is not None and v.lower() not in valid_options:
            raise ValueError(
                f"business_type must be one of {valid_options}, got: {v}"
            )
        return v.lower() if v is not None else None
    
    @field_validator('time_commitment')
    @classmethod
    def validate_time_commitment(cls, v: Optional[str]) -> Optional[str]:
        """Validate time commitment enum."""
        valid_options = {'full_time', 'part_time', 'flexible'}
        if v is not None and v.lower() not in valid_options:
            raise ValueError(
                f"time_commitment must be one of {valid_options}, got: {v}"
            )
        return v.lower() if v is not None else None
    
    @field_validator('risk_tolerance')
    @classmethod
    def validate_risk_tolerance(cls, v: Optional[str]) -> Optional[str]:
        """Validate risk tolerance enum."""
        valid_options = {'low', 'medium', 'high'}
        if v is not None and v.lower() not in valid_options:
            raise ValueError(
                f"risk_tolerance must be one of {valid_options}, got: {v}"
            )
        return v.lower() if v is not None else None


class RecommendationRequest(BaseModel):
    """Request for business idea recommendations.
    
    Combines user profile with generation parameters to request
    personalized business recommendations.
    """
    
    user_profile: UserProfile = Field(
        ...,
        description="User profile information"
    )
    num_recommendations: int = Field(
        default=3,
        description="Number of recommendations to generate",
        ge=1,
        le=10
    )
    include_reasoning: bool = Field(
        default=True,
        description="Include reasoning for each recommendation"
    )
    include_market_data: bool = Field(
        default=True,
        description="Include market data and statistics"
    )
    
    @field_validator('user_profile')
    @classmethod
    def validate_user_profile_not_empty(cls, v: UserProfile) -> UserProfile:
        """Ensure user profile has at least some information."""
        # Get all fields that might be non-None and non-empty
        has_data = any([
            v.age is not None,
            v.education_level is not None,
            v.skills,
            v.interests,
            v.business_type is not None,
            v.startup_capital is not None,
            v.risk_tolerance is not None,
        ])
        
        if not has_data:
            logger.warning("User profile has minimal data")
        
        return v


class AnalysisRequest(BaseModel):
    """Request for business idea analysis.
    
    Contains a business idea description and optional user profile
    for contextual analysis.
    """
    
    business_idea: str = Field(
        ...,
        description="Description of the business idea to analyze",
        min_length=10,
        max_length=5000
    )
    user_profile: Optional[UserProfile] = Field(
        None,
        description="Optional user profile for contextual analysis"
    )
    analysis_depth: Literal['basic', 'detailed', 'comprehensive'] = Field(
        default='detailed',
        description="Depth of analysis (basic, detailed, comprehensive)"
    )
    include_swot: bool = Field(
        default=True,
        description="Include SWOT analysis"
    )
    include_market_size: bool = Field(
        default=True,
        description="Include market size estimation"
    )
    include_financial_projections: bool = Field(
        default=False,
        description="Include financial projections if user profile available"
    )
    
    @field_validator('business_idea')
    @classmethod
    def validate_business_idea(cls, v: str) -> str:
        """Ensure business idea has meaningful content."""
        if not v.strip():
            raise ValueError("Business idea description cannot be empty or whitespace only")
        return v.strip()


# ============================================================================
# API RESPONSE SCHEMAS
# ============================================================================


class BusinessRecommendation(BaseModel):
    """Single business recommendation.
    
    Represents one recommended business idea with detailed information
    about why it matches the user's profile.
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for recommendation",
        min_length=1
    )
    title: str = Field(
        ...,
        description="Business idea title",
        min_length=5,
        max_length=200
    )
    description: str = Field(
        ...,
        description="Detailed description of the business idea",
        min_length=50,
        max_length=2000
    )
    why_it_fits: str = Field(
        ...,
        description="Explanation of why this fits the user",
        min_length=30,
        max_length=1000
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills required for this business"
    )
    startup_capital_estimate: Optional[int] = Field(
        None,
        description="Estimated startup capital needed (USD)",
        ge=0
    )
    time_to_profitability: Optional[str] = Field(
        None,
        description="Estimated time to profitability (e.g., '6_months', '1_year')"
    )
    profit_potential: Optional[str] = Field(
        None,
        description="Profit potential level (low, medium, high)"
    )
    market_size: Optional[str] = Field(
        None,
        description="Market size assessment"
    )
    key_challenges: List[str] = Field(
        default_factory=list,
        description="Key challenges for this business"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Relevance score to user profile (0-1)",
        ge=0.0,
        le=1.0
    )
    
    @field_validator('required_skills', 'key_challenges', 'next_steps')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Ensure all list items are non-empty strings."""
        if v is not None:
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All list items must be non-empty strings")
        return v


class RecommendationResponse(BaseModel):
    """Response containing business recommendations.
    
    Returned when requesting business idea recommendations.
    """
    
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(
        default=True,
        description="Whether recommendations were generated successfully"
    )
    recommendations: List[BusinessRecommendation] = Field(
        ...,
        description="List of business recommendations",
        min_length=1
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate recommendations (milliseconds)",
        ge=0
    )
    model_used: str = Field(
        default="mistral-large-latest",
        description="LLM model used for generation"
    )
    context_sources: int = Field(
        default=0,
        description="Number of context documents used"
    )
    
    @field_validator('recommendations')
    @classmethod
    def validate_recommendations(cls, v: List[BusinessRecommendation]) -> List[BusinessRecommendation]:
        """Ensure recommendations have unique IDs."""
        if v:
            ids = [rec.id for rec in v]
            if len(ids) != len(set(ids)):
                raise ValueError("All recommendations must have unique IDs")
        return v


class SWOTAnalysis(BaseModel):
    """SWOT analysis for a business idea.
    
    Contains Strengths, Weaknesses, Opportunities, and Threats.
    """
    
    strengths: List[str] = Field(
        default_factory=list,
        description="Business strengths"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Business weaknesses"
    )
    opportunities: List[str] = Field(
        default_factory=list,
        description="Market opportunities"
    )
    threats: List[str] = Field(
        default_factory=list,
        description="Market threats"
    )
    
    @field_validator('strengths', 'weaknesses', 'opportunities', 'threats')
    @classmethod
    def validate_swot_items(cls, v: List[str]) -> List[str]:
        """Ensure all items are non-empty strings."""
        if v is not None:
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All SWOT items must be non-empty strings")
        return v


class BusinessAnalysis(BaseModel):
    """Comprehensive business idea analysis.
    
    Contains detailed analysis of a business idea including market,
    financial, and operational aspects.
    """
    
    business_idea: str = Field(
        ...,
        description="The analyzed business idea",
        min_length=10
    )
    viability_score: float = Field(
        ...,
        description="Business viability score (0-100)",
        ge=0.0,
        le=100.0
    )
    summary: str = Field(
        ...,
        description="Executive summary of the analysis",
        min_length=50,
        max_length=2000
    )
    market_analysis: Optional[str] = Field(
        None,
        description="Market analysis and opportunity assessment"
    )
    market_size: Optional[str] = Field(
        None,
        description="Estimated market size"
    )
    target_audience: List[str] = Field(
        default_factory=list,
        description="Target audience segments"
    )
    revenue_model: Optional[str] = Field(
        None,
        description="Suggested revenue model"
    )
    startup_capital_estimate: Optional[int] = Field(
        None,
        description="Estimated startup capital needed (USD)",
        ge=0
    )
    profitability_timeline: Optional[str] = Field(
        None,
        description="Estimated timeline to profitability"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills required"
    )
    required_resources: List[str] = Field(
        default_factory=list,
        description="Resources needed"
    )
    swot: Optional[SWOTAnalysis] = Field(
        None,
        description="SWOT analysis if requested"
    )
    key_risks: List[str] = Field(
        default_factory=list,
        description="Key risks and mitigation strategies"
    )
    competitive_landscape: Optional[str] = Field(
        None,
        description="Analysis of competitive landscape"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for moving forward"
    )
    
    @field_validator('target_audience', 'required_skills', 'required_resources', 'key_risks', 'recommendations')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Ensure all list items are non-empty strings."""
        if v is not None:
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All list items must be non-empty strings")
        return v


class AnalysisResponse(BaseModel):
    """Response containing business idea analysis.
    
    Returned when requesting analysis of a business idea.
    """
    
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(
        default=True,
        description="Whether analysis was completed successfully"
    )
    analysis: BusinessAnalysis = Field(
        ...,
        description="Business idea analysis"
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate analysis (milliseconds)",
        ge=0
    )
    model_used: str = Field(
        default="mistral-large-latest",
        description="LLM model used for generation"
    )
    context_sources: int = Field(
        default=0,
        description="Number of context documents used"
    )


# ============================================================================
# ERROR RESPONSE SCHEMA
# ============================================================================


class ErrorDetail(BaseModel):
    """Details about a validation or processing error."""
    
    field: Optional[str] = Field(
        None,
        description="Field that caused the error (if validation error)"
    )
    message: str = Field(
        ...,
        description="Error message",
        min_length=1
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )


class ErrorResponse(BaseModel):
    """Standard error response format.
    
    Used for all error responses from the API to maintain consistency.
    """
    
    success: bool = Field(
        default=False,
        description="Always False for error responses"
    )
    error_type: str = Field(
        ...,
        description="Type of error (validation_error, not_found, server_error, etc.)",
        min_length=1
    )
    message: str = Field(
        ...,
        description="User-friendly error message",
        min_length=1
    )
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the error occurred"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracking"
    )
    
    @field_validator('error_type')
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        """Ensure error_type is valid."""
        valid_types = {
            'validation_error',
            'not_found',
            'server_error',
            'unauthorized',
            'forbidden',
            'rate_limit_error',
            'model_error',
            'retrieval_error',
            'bad_request'
        }
        
        if v.lower() not in valid_types:
            logger.warning(f"Invalid error_type: {v}")
        
        return v.lower()


# ============================================================================
# UTILITY CLASSES
# ============================================================================


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    
    skip: int = Field(
        default=0,
        description="Number of items to skip",
        ge=0
    )
    limit: int = Field(
        default=10,
        description="Number of items to return",
        ge=1,
        le=100
    )


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: Literal['healthy', 'degraded', 'unhealthy'] = Field(
        default='healthy',
        description="System health status"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp"
    )
    faiss_index_loaded: bool = Field(
        default=False,
        description="Whether FAISS index is loaded"
    )
    mistral_api_available: bool = Field(
        default=False,
        description="Whether Mistral API is accessible"
    )
