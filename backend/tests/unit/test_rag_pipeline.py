"""Unit tests for RAG pipeline."""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.rag_pipeline import RAGPipeline, RAGPipelineError
from src.mistral_client import MistralAPIError
from src.faiss_retriever import FAISSRetrieverError
from src.document_processor import Document, DocumentMetadata


@pytest.fixture
def mock_mistral_client():
    """Create mock Mistral client."""
    client = Mock()
    client.complete_with_retry = Mock()
    return client


@pytest.fixture
def mock_faiss_retriever():
    """Create mock FAISS retriever."""
    retriever = Mock()
    retriever.is_empty = False
    retriever.search = Mock()
    return retriever


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    builder = Mock()
    builder.build_recommendation_prompt = Mock()
    builder.build_analysis_prompt = Mock()
    builder._format_context = Mock(return_value="formatted context")
    return builder


@pytest.fixture
def rag_pipeline(mock_mistral_client, mock_faiss_retriever, mock_prompt_builder):
    """Create RAG pipeline with mocked dependencies."""
    return RAGPipeline(
        mistral_client=mock_mistral_client,
        faiss_retriever=mock_faiss_retriever,
        prompt_builder=mock_prompt_builder,
        top_k_documents=5
    )


@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing."""
    return {
        "professional_status": "employed_fulltime",
        "timeCommitment": 10,
        "budget": "1000_5000",
        "skills": ["programming", "marketing"],
        "industries": ["technology", "education"],
        "business_model": "online",
        "primary_goal": "passive_income"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            content="Sample business document about tech startups",
            metadata=DocumentMetadata(
                source="tech_startups.txt",
                category="technology"
            )
        ),
        Document(
            content="Sample business document about online education",
            metadata=DocumentMetadata(
                source="online_education.txt",
                category="education"
            )
        )
    ]


class TestRAGPipelineInit:
    """Test RAG pipeline initialization."""
    
    def test_init_success(self, mock_mistral_client, mock_faiss_retriever, mock_prompt_builder):
        """Test successful initialization."""
        pipeline = RAGPipeline(
            mistral_client=mock_mistral_client,
            faiss_retriever=mock_faiss_retriever,
            prompt_builder=mock_prompt_builder,
            top_k_documents=5
        )
        
        assert pipeline.mistral_client == mock_mistral_client
        assert pipeline.faiss_retriever == mock_faiss_retriever
        assert pipeline.prompt_builder == mock_prompt_builder
        assert pipeline.top_k_documents == 5
    
    def test_init_invalid_top_k(self, mock_mistral_client, mock_faiss_retriever, mock_prompt_builder):
        """Test initialization with invalid top_k_documents."""
        with pytest.raises(ValueError, match="top_k_documents must be positive"):
            RAGPipeline(
                mistral_client=mock_mistral_client,
                faiss_retriever=mock_faiss_retriever,
                prompt_builder=mock_prompt_builder,
                top_k_documents=0
            )


class TestRetrieveContext:
    """Test context retrieval."""
    
    def test_retrieve_context_success(self, rag_pipeline, sample_documents):
        """Test successful context retrieval."""
        # Mock FAISS search results
        rag_pipeline.faiss_retriever.search.return_value = [
            (sample_documents[0], 0.95),
            (sample_documents[1], 0.87)
        ]
        
        documents = rag_pipeline._retrieve_context("test query")
        
        assert len(documents) == 2
        assert documents[0] == sample_documents[0]
        assert documents[1] == sample_documents[1]
        
        rag_pipeline.faiss_retriever.search.assert_called_once_with(
            query="test query",
            top_k=5
        )
    
    def test_retrieve_context_empty_index(self, rag_pipeline):
        """Test retrieval with empty index."""
        rag_pipeline.faiss_retriever.is_empty = True
        
        documents = rag_pipeline._retrieve_context("test query")
        
        assert documents == []
        rag_pipeline.faiss_retriever.search.assert_not_called()
    
    def test_retrieve_context_custom_top_k(self, rag_pipeline, sample_documents):
        """Test retrieval with custom top_k."""
        rag_pipeline.faiss_retriever.search.return_value = [
            (sample_documents[0], 0.95)
        ]
        
        documents = rag_pipeline._retrieve_context("test query", top_k=1)
        
        assert len(documents) == 1
        rag_pipeline.faiss_retriever.search.assert_called_once_with(
            query="test query",
            top_k=1
        )
    
    def test_retrieve_context_faiss_error(self, rag_pipeline):
        """Test retrieval when FAISS raises error."""
        rag_pipeline.faiss_retriever.search.side_effect = FAISSRetrieverError("Search failed")
        
        with pytest.raises(RAGPipelineError, match="Context retrieval failed"):
            rag_pipeline._retrieve_context("test query")


class TestGenerateRecommendations:
    """Test recommendation generation."""
    
    def test_generate_recommendations_success(
        self, rag_pipeline, sample_user_profile, sample_documents
    ):
        """Test successful recommendation generation."""
        # Mock FAISS search
        rag_pipeline.faiss_retriever.search.return_value = [
            (sample_documents[0], 0.95)
        ]
        
        # Mock prompt builder
        rag_pipeline.prompt_builder.build_recommendation_prompt.return_value = [
            {"role": "system", "content": "You are a business consultant"},
            {"role": "user", "content": "Generate recommendations"}
        ]
        
        # Mock Mistral response
        mock_response = json.dumps({
            "recommendations": [
                {
                    "name": "Online Course Platform",
                    "description": "Create and sell online courses",
                    "fitReasons": ["Matches tech skills", "Fits budget"],
                    "firstSteps": ["Research market", "Build MVP"],
                    "startupCost": "$1,000-$3,000",
                    "timeToRevenue": "3-6 months",
                    "scalability": "High",
                    "competition": "Medium",
                    "matchScore": 85
                },
                {
                    "name": "Tech Consulting",
                    "description": "Provide technical consulting services",
                    "fitReasons": ["Uses programming skills"],
                    "firstSteps": ["Build portfolio"],
                    "startupCost": "$500-$1,000",
                    "timeToRevenue": "1-3 months",
                    "scalability": "Medium",
                    "competition": "High",
                    "matchScore": 78
                },
                {
                    "name": "SaaS Product",
                    "description": "Build a software as a service product",
                    "fitReasons": ["Tech background"],
                    "firstSteps": ["Validate idea"],
                    "startupCost": "$2,000-$5,000",
                    "timeToRevenue": "6-12 months",
                    "scalability": "High",
                    "competition": "High",
                    "matchScore": 72
                }
            ]
        })
        rag_pipeline.mistral_client.complete_with_retry.return_value = mock_response
        
        # Generate recommendations
        recommendations = rag_pipeline.generate_recommendations(
            user_profile=sample_user_profile,
            num_recommendations=3
        )
        
        # Verify results
        assert len(recommendations) == 3
        assert recommendations[0]["name"] == "Online Course Platform"
        assert recommendations[0]["matchScore"] == 85
        assert recommendations[1]["name"] == "Tech Consulting"
        assert recommendations[2]["name"] == "SaaS Product"
        
        # Verify calls
        rag_pipeline.faiss_retriever.search.assert_called_once()
        rag_pipeline.prompt_builder.build_recommendation_prompt.assert_called_once()
        rag_pipeline.mistral_client.complete_with_retry.assert_called_once()
    
    def test_generate_recommendations_empty_profile(self, rag_pipeline):
        """Test generation with empty user profile."""
        with pytest.raises(ValueError, match="user_profile cannot be empty"):
            rag_pipeline.generate_recommendations(user_profile={})
    
    def test_generate_recommendations_invalid_num(self, rag_pipeline, sample_user_profile):
        """Test generation with invalid num_recommendations."""
        with pytest.raises(ValueError, match="num_recommendations must be positive"):
            rag_pipeline.generate_recommendations(
                user_profile=sample_user_profile,
                num_recommendations=0
            )
    
    def test_generate_recommendations_no_context(
        self, rag_pipeline, sample_user_profile
    ):
        """Test generation when no context documents are retrieved."""
        # Mock empty FAISS index
        rag_pipeline.faiss_retriever.is_empty = True
        
        # Mock prompt builder
        rag_pipeline.prompt_builder.build_recommendation_prompt.return_value = [
            {"role": "system", "content": "You are a business consultant"},
            {"role": "user", "content": "Generate recommendations"}
        ]
        
        # Mock Mistral response
        mock_response = json.dumps({
            "recommendations": [
                {
                    "name": "Test Business",
                    "description": "Test description",
                    "fitReasons": ["reason"],
                    "firstSteps": ["step"],
                    "startupCost": "$1,000",
                    "timeToRevenue": "3 months",
                    "scalability": "Medium",
                    "competition": "Low",
                    "matchScore": 80
                }
            ]
        })
        rag_pipeline.mistral_client.complete_with_retry.return_value = mock_response
        
        # Should still work without context
        recommendations = rag_pipeline.generate_recommendations(
            user_profile=sample_user_profile,
            num_recommendations=1
        )
        
        assert len(recommendations) == 1
    
    def test_generate_recommendations_api_error(
        self, rag_pipeline, sample_user_profile
    ):
        """Test generation when Mistral API fails."""
        rag_pipeline.faiss_retriever.is_empty = True
        rag_pipeline.prompt_builder.build_recommendation_prompt.return_value = []
        rag_pipeline.mistral_client.complete_with_retry.side_effect = MistralAPIError(
            "API error"
        )
        
        with pytest.raises(RAGPipelineError, match="Failed to generate recommendations"):
            rag_pipeline.generate_recommendations(
                user_profile=sample_user_profile,
                num_recommendations=3
            )
    
    def test_generate_recommendations_invalid_json(
        self, rag_pipeline, sample_user_profile
    ):
        """Test generation with invalid JSON response."""
        rag_pipeline.faiss_retriever.is_empty = True
        rag_pipeline.prompt_builder.build_recommendation_prompt.return_value = []
        rag_pipeline.mistral_client.complete_with_retry.return_value = "Not valid JSON"
        
        with pytest.raises(RAGPipelineError, match="Failed to generate recommendations"):
            rag_pipeline.generate_recommendations(
                user_profile=sample_user_profile,
                num_recommendations=3
            )


class TestAnalyzeBusinessIdea:
    """Test business idea analysis."""
    
    def test_analyze_business_idea_success(self, rag_pipeline, sample_documents):
        """Test successful business idea analysis."""
        # Mock FAISS search
        rag_pipeline.faiss_retriever.search.return_value = [
            (sample_documents[0], 0.92)
        ]
        
        # Mock prompt builder
        rag_pipeline.prompt_builder.build_analysis_prompt.return_value = [
            {"role": "system", "content": "You are a business analyst"},
            {"role": "user", "content": "Analyze this idea"}
        ]
        
        # Mock Mistral response
        mock_response = json.dumps({
            "viabilityScore": 75,
            "marketFit": {
                "targetMarket": "Tech professionals",
                "demand": "High",
                "positioning": "Premium"
            },
            "risks": [
                {
                    "risk": "High competition",
                    "severity": "High",
                    "mitigation": "Focus on niche"
                }
            ],
            "suggestions": [
                {
                    "suggestion": "Start with MVP",
                    "impact": "Reduce risk",
                    "priority": "High"
                }
            ],
            "costStructure": {
                "startup": "$5,000",
                "monthly": "$500",
                "breakeven": "12 months"
            },
            "scalability": {
                "potential": "High",
                "challenges": "Technical complexity",
                "strategy": "Gradual expansion"
            }
        })
        rag_pipeline.mistral_client.complete_with_retry.return_value = mock_response
        
        # Analyze idea
        analysis = rag_pipeline.analyze_business_idea(
            business_idea="Create an online course platform for tech professionals"
        )
        
        # Verify results
        assert analysis["viabilityScore"] == 75
        assert "marketFit" in analysis
        assert "risks" in analysis
        assert "suggestions" in analysis
        assert "costStructure" in analysis
        assert "scalability" in analysis
        
        # Verify calls
        rag_pipeline.faiss_retriever.search.assert_called_once()
        rag_pipeline.prompt_builder.build_analysis_prompt.assert_called_once()
        rag_pipeline.mistral_client.complete_with_retry.assert_called_once()
    
    def test_analyze_business_idea_with_profile(
        self, rag_pipeline, sample_user_profile
    ):
        """Test analysis with user profile."""
        rag_pipeline.faiss_retriever.is_empty = True
        rag_pipeline.prompt_builder.build_analysis_prompt.return_value = []
        
        mock_response = json.dumps({
            "viabilityScore": 80,
            "marketFit": {},
            "risks": [],
            "suggestions": [],
            "costStructure": {},
            "scalability": {}
        })
        rag_pipeline.mistral_client.complete_with_retry.return_value = mock_response
        
        analysis = rag_pipeline.analyze_business_idea(
            business_idea="Test idea",
            user_profile=sample_user_profile
        )
        
        assert analysis["viabilityScore"] == 80
        
        # Verify profile was passed to prompt builder
        call_args = rag_pipeline.prompt_builder.build_analysis_prompt.call_args
        assert call_args[1]["user_profile"] == sample_user_profile
    
    def test_analyze_business_idea_empty_idea(self, rag_pipeline):
        """Test analysis with empty business idea."""
        with pytest.raises(ValueError, match="business_idea cannot be empty"):
            rag_pipeline.analyze_business_idea(business_idea="")
    
    def test_analyze_business_idea_api_error(self, rag_pipeline):
        """Test analysis when Mistral API fails."""
        rag_pipeline.faiss_retriever.is_empty = True
        rag_pipeline.prompt_builder.build_analysis_prompt.return_value = []
        rag_pipeline.mistral_client.complete_with_retry.side_effect = MistralAPIError(
            "API error"
        )
        
        with pytest.raises(RAGPipelineError, match="Failed to analyze business idea"):
            rag_pipeline.analyze_business_idea(business_idea="Test idea")
    
    def test_analyze_business_idea_invalid_json(self, rag_pipeline):
        """Test analysis with invalid JSON response."""
        rag_pipeline.faiss_retriever.is_empty = True
        rag_pipeline.prompt_builder.build_analysis_prompt.return_value = []
        rag_pipeline.mistral_client.complete_with_retry.return_value = "Invalid JSON"
        
        with pytest.raises(RAGPipelineError, match="Failed to analyze business idea"):
            rag_pipeline.analyze_business_idea(business_idea="Test idea")


class TestBuildQueryFromProfile:
    """Test query building from user profile."""
    
    def test_build_query_complete_profile(self, rag_pipeline, sample_user_profile):
        """Test query building with complete profile."""
        query = rag_pipeline._build_query_from_profile(sample_user_profile)
        
        assert "employed_fulltime" in query
        assert "10 hours/week" in query
        assert "1000_5000" in query
        assert "programming" in query
        assert "marketing" in query
        assert "technology" in query
        assert "education" in query
        assert "online" in query
        assert "passive_income" in query
    
    def test_build_query_minimal_profile(self, rag_pipeline):
        """Test query building with minimal profile."""
        minimal_profile = {
            "professional_status": "student",
            "timeCommitment": 5,
            "budget": "under_1000"
        }
        
        query = rag_pipeline._build_query_from_profile(minimal_profile)
        
        assert "student" in query
        assert "5 hours/week" in query
        assert "under_1000" in query


class TestParseResponses:
    """Test response parsing."""
    
    def test_parse_recommendations_valid(self, rag_pipeline):
        """Test parsing valid recommendations response."""
        response = json.dumps({
            "recommendations": [
                {
                    "name": "Test Business",
                    "description": "Description",
                    "fitReasons": ["reason1"],
                    "firstSteps": ["step1"],
                    "startupCost": "$1,000",
                    "timeToRevenue": "3 months",
                    "scalability": "High",
                    "competition": "Low",
                    "matchScore": 85
                }
            ]
        })
        
        recommendations = rag_pipeline._parse_recommendations_response(response)
        
        assert len(recommendations) == 1
        assert recommendations[0]["name"] == "Test Business"
        assert recommendations[0]["matchScore"] == 85
    
    def test_parse_recommendations_with_extra_text(self, rag_pipeline):
        """Test parsing response with extra text around JSON."""
        response = """
        Here are the recommendations:
        
        {
            "recommendations": [
                {
                    "name": "Test",
                    "description": "Desc",
                    "fitReasons": [],
                    "firstSteps": [],
                    "startupCost": "$1,000",
                    "timeToRevenue": "3 months",
                    "scalability": "High",
                    "competition": "Low",
                    "matchScore": 80
                }
            ]
        }
        
        I hope this helps!
        """
        
        recommendations = rag_pipeline._parse_recommendations_response(response)
        
        assert len(recommendations) == 1
        assert recommendations[0]["name"] == "Test"
    
    def test_parse_analysis_valid(self, rag_pipeline):
        """Test parsing valid analysis response."""
        response = json.dumps({
            "viabilityScore": 75,
            "marketFit": {"targetMarket": "Test"},
            "risks": [{"risk": "Test risk"}],
            "suggestions": [{"suggestion": "Test suggestion"}],
            "costStructure": {"startup": "$5,000"},
            "scalability": {"potential": "High"}
        })
        
        analysis = rag_pipeline._parse_analysis_response(response)
        
        assert analysis["viabilityScore"] == 75
        assert "marketFit" in analysis
        assert "risks" in analysis
    
    def test_parse_analysis_invalid_score(self, rag_pipeline):
        """Test parsing analysis with invalid viability score."""
        response = json.dumps({
            "viabilityScore": 150,  # Invalid: > 100
            "marketFit": {},
            "risks": [],
            "suggestions": [],
            "costStructure": {},
            "scalability": {}
        })
        
        analysis = rag_pipeline._parse_analysis_response(response)
        
        # Should be clamped to 100
        assert analysis["viabilityScore"] == 100
