"""Unit tests for PromptBuilder class."""

import pytest
from src.prompt_builder import PromptBuilder
from src.document_processor import Document, DocumentMetadata


class TestPromptBuilderInitialization:
    """Test PromptBuilder initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        builder = PromptBuilder()
        assert builder.max_context_tokens == 3000
    
    def test_init_with_custom_max_tokens(self):
        """Test initialization with custom max_context_tokens."""
        builder = PromptBuilder(max_context_tokens=5000)
        assert builder.max_context_tokens == 5000
    
    def test_init_with_invalid_max_tokens(self):
        """Test initialization fails with invalid max_context_tokens."""
        with pytest.raises(ValueError, match="max_context_tokens must be positive"):
            PromptBuilder(max_context_tokens=0)
        
        with pytest.raises(ValueError, match="max_context_tokens must be positive"):
            PromptBuilder(max_context_tokens=-100)


class TestBuildRecommendationPrompt:
    """Test recommendation prompt building."""
    
    def test_build_recommendation_prompt_basic(self):
        """Test building recommendation prompt with basic user profile."""
        builder = PromptBuilder()
        
        user_profile = {
            "professional_status": "employed_fulltime",
            "timeCommitment": 10,
            "budget": "1000_5000"
        }
        
        messages = builder.build_recommendation_prompt(
            user_profile=user_profile,
            context_documents=[],
            num_recommendations=3
        )
        
        # Should return list of messages
        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user message
        
        # Check message structure
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        # Check content includes key components
        user_content = messages[1]["content"]
        assert "Professional Status" in user_content
        assert "employed_fulltime" in user_content
        assert "Time Availability" in user_content
        assert "10" in user_content
        assert "Budget" in user_content
        assert "1000_5000" in user_content
    
    def test_build_recommendation_prompt_with_context(self):
        """Test building recommendation prompt with context documents."""
        builder = PromptBuilder()
        
        user_profile = {
            "professional_status": "freelancer",
            "timeCommitment": 20,
            "budget": "under_1000"
        }
        
        doc = Document(
            content="E-commerce businesses have low startup costs.",
            metadata=DocumentMetadata(
                source="ecommerce_guide.txt",
                category="business_models"
            )
        )
        
        messages = builder.build_recommendation_prompt(
            user_profile=user_profile,
            context_documents=[doc],
            num_recommendations=3
        )
        
        user_content = messages[1]["content"]
        
        # Check context is included
        assert "Relevant Business Knowledge" in user_content
        assert "ecommerce_guide.txt" in user_content
        assert "E-commerce businesses have low startup costs" in user_content
    
    def test_build_recommendation_prompt_with_all_profile_fields(self):
        """Test building prompt with complete user profile."""
        builder = PromptBuilder()
        
        user_profile = {
            "professional_status": "student",
            "timeCommitment": 15,
            "budget": "under_1000",
            "skills": ["programming", "design"],
            "industries": ["technology", "education"],
            "business_model": "online_service",
            "target_market": "students",
            "location": "urban",
            "riskTolerance": 7,
            "revenue_timeline": "6_12_months",
            "primary_goal": "side_income",
            "additionalContext": "Interested in EdTech"
        }
        
        messages = builder.build_recommendation_prompt(
            user_profile=user_profile,
            context_documents=[],
            num_recommendations=5
        )
        
        user_content = messages[1]["content"]
        
        # Check all fields are included
        assert "Skills: programming, design" in user_content
        assert "Industry Interests: technology, education" in user_content
        assert "Preferred Business Model: online_service" in user_content
        assert "Target Market: students" in user_content
        assert "Location: urban" in user_content
        assert "Risk Tolerance: 7" in user_content
        assert "Revenue Timeline: 6_12_months" in user_content
        assert "Primary Goal: side_income" in user_content
        assert "Additional Context: Interested in EdTech" in user_content
        
        # Check num_recommendations is reflected
        assert "Generate exactly 5 distinct business ideas" in user_content
    
    def test_build_recommendation_prompt_empty_profile_fails(self):
        """Test building prompt fails with empty user profile."""
        builder = PromptBuilder()
        
        with pytest.raises(ValueError, match="user_profile cannot be empty"):
            builder.build_recommendation_prompt(
                user_profile={},
                context_documents=[],
                num_recommendations=3
            )
    
    def test_build_recommendation_prompt_invalid_num_recommendations(self):
        """Test building prompt fails with invalid num_recommendations."""
        builder = PromptBuilder()
        
        user_profile = {"professional_status": "employed_fulltime"}
        
        with pytest.raises(ValueError, match="num_recommendations must be positive"):
            builder.build_recommendation_prompt(
                user_profile=user_profile,
                context_documents=[],
                num_recommendations=0
            )
        
        with pytest.raises(ValueError, match="num_recommendations must be positive"):
            builder.build_recommendation_prompt(
                user_profile=user_profile,
                context_documents=[],
                num_recommendations=-1
            )


class TestBuildAnalysisPrompt:
    """Test analysis prompt building."""
    
    def test_build_analysis_prompt_basic(self):
        """Test building analysis prompt with business idea only."""
        builder = PromptBuilder()
        
        business_idea = "A mobile app for connecting freelance tutors with students"
        
        messages = builder.build_analysis_prompt(
            business_idea=business_idea,
            context_documents=[]
        )
        
        # Should return list of messages
        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user message
        
        # Check message structure
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        # Check content includes the business idea
        user_content = messages[1]["content"]
        assert business_idea in user_content
        assert "Analyze the following business idea" in user_content
    
    def test_build_analysis_prompt_with_context(self):
        """Test building analysis prompt with context documents."""
        builder = PromptBuilder()
        
        business_idea = "Online tutoring platform"
        
        doc = Document(
            content="The online education market is growing at 15% annually.",
            metadata=DocumentMetadata(
                source="education_market_report.txt",
                category="market_data"
            )
        )
        
        messages = builder.build_analysis_prompt(
            business_idea=business_idea,
            context_documents=[doc]
        )
        
        user_content = messages[1]["content"]
        
        # Check context is included
        assert "Relevant Market Data" in user_content
        assert "education_market_report.txt" in user_content
        assert "online education market is growing" in user_content
    
    def test_build_analysis_prompt_with_user_profile(self):
        """Test building analysis prompt with user profile."""
        builder = PromptBuilder()
        
        business_idea = "Freelance consulting service"
        user_profile = {
            "professional_status": "employed_fulltime",
            "timeCommitment": 10,
            "budget": "under_1000",
            "skills": ["consulting", "business_analysis"]
        }
        
        messages = builder.build_analysis_prompt(
            business_idea=business_idea,
            context_documents=[],
            user_profile=user_profile
        )
        
        user_content = messages[1]["content"]
        
        # Check user profile is included
        assert "User Context" in user_content
        assert "Professional Status: employed_fulltime" in user_content
        assert "Skills: consulting, business_analysis" in user_content
    
    def test_build_analysis_prompt_empty_idea_fails(self):
        """Test building analysis prompt fails with empty business idea."""
        builder = PromptBuilder()
        
        with pytest.raises(ValueError, match="business_idea cannot be empty"):
            builder.build_analysis_prompt(
                business_idea="",
                context_documents=[]
            )
        
        with pytest.raises(ValueError, match="business_idea cannot be empty"):
            builder.build_analysis_prompt(
                business_idea="   ",
                context_documents=[]
            )


class TestFormatUserProfile:
    """Test user profile formatting."""
    
    def test_format_user_profile_basic_fields(self):
        """Test formatting basic profile fields."""
        builder = PromptBuilder()
        
        profile = {
            "professional_status": "freelancer",
            "timeCommitment": 20,
            "budget": "5000_25000"
        }
        
        formatted = builder._format_user_profile(profile)
        
        assert "Professional Status: freelancer" in formatted
        assert "Time Availability: 20" in formatted
        assert "Budget: 5000_25000" in formatted
    
    def test_format_user_profile_with_lists(self):
        """Test formatting profile fields that are lists."""
        builder = PromptBuilder()
        
        profile = {
            "skills": ["python", "javascript", "design"],
            "industries": ["technology", "finance"]
        }
        
        formatted = builder._format_user_profile(profile)
        
        assert "Skills: python, javascript, design" in formatted
        assert "Industry Interests: technology, finance" in formatted
    
    def test_format_user_profile_empty_lists_excluded(self):
        """Test that empty lists are not included in formatted output."""
        builder = PromptBuilder()
        
        profile = {
            "professional_status": "student",
            "skills": [],
            "industries": []
        }
        
        formatted = builder._format_user_profile(profile)
        
        assert "Professional Status: student" in formatted
        assert "Skills:" not in formatted
        assert "Industry Interests:" not in formatted
    
    def test_format_user_profile_empty_returns_message(self):
        """Test formatting empty profile returns default message."""
        builder = PromptBuilder()
        
        formatted = builder._format_user_profile({})
        
        assert formatted == "No profile information provided"


class TestFormatContext:
    """Test context document formatting."""
    
    def test_format_context_single_document(self):
        """Test formatting single context document."""
        builder = PromptBuilder()
        
        doc = Document(
            content="This is a test document about business strategies.",
            metadata=DocumentMetadata(
                source="strategies.txt",
                category="business_guides"
            )
        )
        
        formatted = builder._format_context([doc])
        
        assert "## Document 1: strategies.txt" in formatted
        assert "Category: business_guides" in formatted
        assert "This is a test document about business strategies" in formatted
    
    def test_format_context_multiple_documents(self):
        """Test formatting multiple context documents."""
        builder = PromptBuilder()
        
        doc1 = Document(
            content="First document content.",
            metadata=DocumentMetadata(source="doc1.txt", category="cat1")
        )
        
        doc2 = Document(
            content="Second document content.",
            metadata=DocumentMetadata(source="doc2.txt", category="cat2")
        )
        
        formatted = builder._format_context([doc1, doc2])
        
        assert "## Document 1: doc1.txt" in formatted
        assert "First document content" in formatted
        assert "## Document 2: doc2.txt" in formatted
        assert "Second document content" in formatted
    
    def test_format_context_empty_list(self):
        """Test formatting empty document list returns empty string."""
        builder = PromptBuilder()
        
        formatted = builder._format_context([])
        
        assert formatted == ""


class TestTokenLimitEnforcement:
    """Test token limit enforcement in context formatting."""
    
    def test_format_context_respects_token_limit(self):
        """Test that context is truncated when exceeding token limit."""
        # Use small token limit for testing
        builder = PromptBuilder(max_context_tokens=50)
        
        # Create documents that would exceed limit
        docs = [
            Document(
                content=" ".join(["word"] * 100),  # 100 words
                metadata=DocumentMetadata(source=f"doc{i}.txt", category="test")
            )
            for i in range(5)
        ]
        
        formatted = builder._format_context(docs)
        
        # Should be truncated
        assert "truncated due to token limit" in formatted.lower()
        
        # Estimate tokens (rough check)
        token_count = builder._count_tokens(formatted)
        assert token_count <= builder.max_context_tokens * 1.2  # Allow some margin
    
    def test_count_tokens_basic(self):
        """Test token counting approximation."""
        builder = PromptBuilder()
        
        text = "This is a simple test sentence."
        token_count = builder._count_tokens(text)
        
        # Should be approximately 6 words * 1.3 = ~8 tokens
        assert 6 <= token_count <= 10
    
    def test_truncate_to_tokens(self):
        """Test text truncation to token limit."""
        builder = PromptBuilder()
        
        text = " ".join(["word"] * 100)
        
        truncated = builder._truncate_to_tokens(text, max_tokens=20)
        
        # Should have approximately 20/1.3 = ~15 words
        word_count = len(truncated.split())
        assert 10 <= word_count <= 20
    
    def test_truncate_to_tokens_zero_returns_empty(self):
        """Test truncating to zero tokens returns empty string."""
        builder = PromptBuilder()
        
        text = "Some text here"
        truncated = builder._truncate_to_tokens(text, max_tokens=0)
        
        assert truncated == ""
    
    def test_truncate_to_tokens_no_truncation_needed(self):
        """Test truncation when text is already under limit."""
        builder = PromptBuilder()
        
        text = "Short text"
        truncated = builder._truncate_to_tokens(text, max_tokens=1000)
        
        assert truncated == text
