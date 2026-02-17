# Design Document: Mistral AI + RAG + FAISS Backend Integration

## Overview

This design document describes the architecture for integrating Mistral AI with Retrieval Augmented Generation (RAG) using FAISS vector database into the BizBot backend. The system will process user questionnaire responses, retrieve relevant business knowledge from a vector database, and generate personalized business recommendations using Mistral's large language model.

The backend will be implemented as a Python Flask REST API that serves two primary endpoints: one for generating business idea recommendations based on user profiles, and another for analyzing user-provided business ideas. The system uses FAISS for efficient semantic search over a knowledge base of business documents, and Mistral AI for intelligent text generation.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Frontend      │
│  (HTML/JS)      │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────────────────────────────┐
│         Flask REST API                  │
│  ┌─────────────────────────────────┐   │
│  │  /api/recommendations           │   │
│  │  /api/analyze                   │   │
│  └─────────────────────────────────┘   │
└────────┬────────────────────┬───────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────┐
│  RAG Pipeline   │  │  Mistral Client  │
│                 │  │                  │
│  ┌───────────┐  │  │  - Embeddings   │
│  │  Query    │  │  │  - Chat         │
│  │  Builder  │  │  │  - Completion   │
│  └─────┬─────┘  │  └──────────────────┘
│        │        │
│        ▼        │
│  ┌───────────┐  │
│  │  FAISS    │  │
│  │  Retriever│  │
│  └─────┬─────┘  │
│        │        │
│        ▼        │
│  ┌───────────┐  │
│  │  Context  │  │
│  │  Formatter│  │
│  └───────────┘  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
│  (Vector DB)    │
│                 │
│  - Embeddings   │
│  - Metadata     │
└─────────────────┘
```

### Component Interaction Flow

1. **Request Reception**: Flask API receives POST request with user data
2. **Query Construction**: User profile is converted into a semantic query
3. **Vector Search**: Query embedding is used to search FAISS index
4. **Context Retrieval**: Top-K relevant documents are retrieved
5. **Prompt Assembly**: User data + retrieved context + instructions → LLM prompt
6. **Generation**: Mistral AI generates recommendations or analysis
7. **Response Formatting**: Output is parsed and formatted as JSON
8. **Response Return**: JSON response sent back to frontend

## Components and Interfaces

### 1. Flask API Server (`app.py`)

**Responsibility**: Handle HTTP requests, route to appropriate handlers, manage CORS

**Interface**:
```python
class FlaskApp:
    def __init__(self, rag_pipeline: RAGPipeline, config: Config):
        """Initialize Flask app with RAG pipeline and configuration"""
        
    def create_app(self) -> Flask:
        """Create and configure Flask application"""
        
    @app.route('/api/recommendations', methods=['POST'])
    def generate_recommendations() -> Response:
        """
        Generate business idea recommendations
        
        Request Body:
        {
            "professional_status": str,
            "timeCommitment": int,
            "budget": str,
            "skills": List[str],
            "industries": List[str],
            ...
        }
        
        Response:
        {
            "recommendations": [
                {
                    "name": str,
                    "description": str,
                    "fitReasons": List[str],
                    "firstSteps": List[str],
                    "startupCost": str,
                    "timeToRevenue": str,
                    "scalability": str,
                    "competition": str,
                    "matchScore": int
                }
            ]
        }
        """
        
    @app.route('/api/analyze', methods=['POST'])
    def analyze_idea() -> Response:
        """
        Analyze a user-provided business idea
        
        Request Body:
        {
            "businessIdea": str,
            "userProfile": {...}  # Optional
        }
        
        Response:
        {
            "viabilityScore": int,
            "marketFit": {...},
            "risks": [...],
            "suggestions": [...],
            ...
        }
        """
```

### 2. RAG Pipeline (`rag_pipeline.py`)

**Responsibility**: Orchestrate retrieval and generation process

**Interface**:
```python
class RAGPipeline:
    def __init__(
        self,
        mistral_client: MistralClient,
        faiss_retriever: FAISSRetriever,
        prompt_builder: PromptBuilder
    ):
        """Initialize RAG pipeline with dependencies"""
        
    def generate_recommendations(
        self,
        user_profile: Dict[str, Any],
        num_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate business recommendations using RAG
        
        Args:
            user_profile: User questionnaire responses
            num_recommendations: Number of ideas to generate
            
        Returns:
            List of business idea recommendations
        """
        
    def analyze_business_idea(
        self,
        business_idea: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a user-provided business idea
        
        Args:
            business_idea: The business concept to analyze
            user_profile: Optional user context
            
        Returns:
            Analysis results with scores and suggestions
        """
        
    def _retrieve_context(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents from FAISS"""
        
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents for LLM prompt"""
```

### 3. FAISS Retriever (`faiss_retriever.py`)

**Responsibility**: Manage FAISS index, perform semantic search

**Interface**:
```python
class FAISSRetriever:
    def __init__(
        self,
        mistral_client: MistralClient,
        index_path: str = "data/faiss_index"
    ):
        """Initialize FAISS retriever with embedding model"""
        
    def build_index(self, documents: List[Document]) -> None:
        """
        Build FAISS index from documents
        
        Args:
            documents: List of business documents to index
        """
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing index"""
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        
    def save_index(self, path: str) -> None:
        """Persist FAISS index to disk"""
        
    def load_index(self, path: str) -> None:
        """Load FAISS index from disk"""
        
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using Mistral"""
```

### 4. Mistral Client Wrapper (`mistral_client.py`)

**Responsibility**: Interface with Mistral AI API

**Interface**:
```python
class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """Initialize Mistral client"""
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embedding vector(s) as numpy array
        """
        
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """
        Generate chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        
    def complete_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3
    ) -> str:
        """Chat completion with exponential backoff retry"""
```

### 5. Document Processor (`document_processor.py`)

**Responsibility**: Load, chunk, and prepare documents for indexing

**Interface**:
```python
class Document:
    """Represents a document with content and metadata"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document processor"""
        
    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from directory
        
        Supports: .txt, .pdf, .json, .md
        """
        
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split document into chunks with overlap
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from file"""
```

### 6. Prompt Builder (`prompt_builder.py`)

**Responsibility**: Construct prompts for Mistral AI

**Interface**:
```python
class PromptBuilder:
    def build_recommendation_prompt(
        self,
        user_profile: Dict[str, Any],
        context_documents: List[Document],
        num_recommendations: int = 3
    ) -> List[Dict[str, str]]:
        """
        Build prompt for generating recommendations
        
        Returns:
            List of message dicts for chat completion
        """
        
    def build_analysis_prompt(
        self,
        business_idea: str,
        context_documents: List[Document],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for analyzing business idea"""
        
    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """Format user profile for prompt"""
        
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context"""
```

### 7. Cache Manager (`cache_manager.py`)

**Responsibility**: Manage caching for performance

**Interface**:
```python
class CacheManager:
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """Initialize cache with TTL and size limit"""
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value"""
        
    def set(self, key: str, value: Any) -> None:
        """Store value in cache"""
        
    def invalidate(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern"""
        
    def _generate_key(self, data: Dict) -> str:
        """Generate cache key from data"""
```

### 8. Configuration Manager (`config.py`)

**Responsibility**: Load and validate configuration

**Interface**:
```python
class Config:
    # API Configuration
    MISTRAL_API_KEY: str
    MISTRAL_MODEL: str = "mistral-large-latest"
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = "data/faiss_index"
    EMBEDDING_DIMENSION: int = 1024
    
    # RAG Configuration
    TOP_K_DOCUMENTS: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # API Configuration
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    CORS_ORIGINS: List[str] = ["*"]
    
    # Cache Configuration
    CACHE_TTL: int = 3600
    CACHE_MAX_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/bizbot.log"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        
    def validate(self) -> None:
        """Validate configuration values"""
```

## Data Models

### User Profile Schema

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """User questionnaire response data"""
    professional_status: str = Field(..., description="Employment status")
    timeCommitment: int = Field(..., ge=5, le=60, description="Hours per week")
    budget: str = Field(..., description="Available capital range")
    skills: Optional[List[str]] = Field(default=[], description="User skills")
    industries: Optional[List[str]] = Field(default=[], description="Industry interests")
    business_model: Optional[str] = Field(None, description="Preferred business model")
    target_market: Optional[str] = Field(None, description="Target customer segment")
    location: Optional[str] = Field(None, description="Business location preference")
    riskTolerance: Optional[int] = Field(None, ge=1, le=10, description="Risk tolerance")
    revenue_timeline: Optional[str] = Field(None, description="Expected time to revenue")
    primary_goal: Optional[str] = Field(None, description="Primary business goal")
    additionalContext: Optional[str] = Field(None, description="Additional information")
```

### Business Recommendation Schema

```python
class BusinessRecommendation(BaseModel):
    """Generated business idea recommendation"""
    name: str = Field(..., description="Business name/concept")
    description: str = Field(..., description="Detailed description")
    fitReasons: List[str] = Field(..., description="Why it fits the user")
    firstSteps: List[str] = Field(..., description="Initial action items")
    startupCost: str = Field(..., description="Estimated startup cost range")
    timeToRevenue: str = Field(..., description="Expected time to first revenue")
    scalability: str = Field(..., description="Scalability potential: Low/Medium/High")
    competition: str = Field(..., description="Competition level: Low/Medium/High")
    matchScore: int = Field(..., ge=0, le=100, description="Match percentage")
    sources: Optional[List[str]] = Field(default=[], description="Knowledge base sources")
```

### Business Analysis Schema

```python
class BusinessAnalysis(BaseModel):
    """Analysis of user-provided business idea"""
    viabilityScore: int = Field(..., ge=0, le=100, description="Overall viability score")
    marketFit: Dict[str, Any] = Field(..., description="Market fit metrics")
    risks: List[Dict[str, str]] = Field(..., description="Identified risks")
    suggestions: List[Dict[str, str]] = Field(..., description="Improvement suggestions")
    costStructure: Dict[str, str] = Field(..., description="Cost breakdown")
    scalability: Dict[str, Any] = Field(..., description="Scalability analysis")
    sources: Optional[List[str]] = Field(default=[], description="Knowledge base sources")
```

### Document Schema

```python
class DocumentMetadata(BaseModel):
    """Metadata for knowledge base documents"""
    source: str = Field(..., description="Document source/filename")
    category: str = Field(..., description="Document category")
    date: Optional[str] = Field(None, description="Document date")
    industry: Optional[str] = Field(None, description="Related industry")
    tags: Optional[List[str]] = Field(default=[], description="Document tags")

class Document(BaseModel):
    """Knowledge base document"""
    content: str = Field(..., description="Document text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    chunk_id: Optional[int] = Field(None, description="Chunk identifier")
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*



### Property 1: User Profile Parsing Completeness
*For any* valid user profile JSON, parsing should extract all present fields without data loss
**Validates: Requirements 1.1**

### Property 2: Required Field Validation
*For any* user profile missing required fields (professional_status, timeCommitment, budget), validation should reject it
**Validates: Requirements 1.2**

### Property 3: Validation Error Messages Include Field Names
*For any* validation failure, the error message should contain the names of all missing or invalid fields
**Validates: Requirements 1.3**

### Property 4: Valid Profile Produces Well-Formed Query
*For any* valid user profile, the query conversion should produce a non-empty string suitable for embedding
**Validates: Requirements 1.4**

### Property 5: API Configuration Error on Missing Key
*For any* initialization attempt with missing or empty MISTRAL_API_KEY, the system should raise a configuration error
**Validates: Requirements 2.2**

### Property 6: Exponential Backoff Timing
*For any* sequence of rate limit errors, retry delays should follow exponential backoff pattern (delay_n = base * 2^n)
**Validates: Requirements 2.4**

### Property 7: API Errors Are Logged and User-Friendly
*For any* Mistral API error, the system should log the full error and return a message without technical details
**Validates: Requirements 2.5**

### Property 8: Document Embeddings Have Correct Dimension
*For any* document added to the knowledge base, its embedding should have dimension matching EMBEDDING_DIMENSION config
**Validates: Requirements 3.2**

### Property 9: Search Returns Exactly K Results
*For any* query when the index contains >= K documents, search should return exactly K results ordered by similarity score (descending)
**Validates: Requirements 3.4**

### Property 10: FAISS Index Persistence Round-Trip
*For any* FAISS index, saving to disk and loading should produce an index that returns identical search results for the same query
**Validates: Requirements 3.5**

### Property 11: Document Chunks Respect Token Limit
*For any* document chunked by the system, all chunks should have token count <= CHUNK_SIZE
**Validates: Requirements 4.2**

### Property 12: Consecutive Chunks Have Overlap
*For any* two consecutive chunks from the same document, they should share at least CHUNK_OVERLAP tokens of content
**Validates: Requirements 4.3**

### Property 13: Metadata Preservation
*For any* document with metadata added to the index, retrieving that document should return the same metadata
**Validates: Requirements 4.4**

### Property 14: Retrieved Documents Appear in Prompt
*For any* RAG query, all retrieved documents should appear in the formatted prompt sent to the LLM
**Validates: Requirements 5.2**

### Property 15: Prompt Contains All Required Components
*For any* recommendation request, the final prompt should contain user profile data, retrieved context, and generation instructions
**Validates: Requirements 5.3**

### Property 16: Context Token Limit Enforcement
*For any* retrieved context, the total token count should not exceed (MAX_TOKENS - RESERVED_TOKENS) where RESERVED_TOKENS accounts for prompt structure and generation space
**Validates: Requirements 5.5**

### Property 17: Minimum Recommendation Count
*For any* valid recommendation request, the response should contain at least 3 distinct business ideas
**Validates: Requirements 6.1**

### Property 18: Recommendation Schema Completeness
*For any* generated recommendation, it should contain all required fields: name, description, fitReasons, firstSteps, startupCost, timeToRevenue, scalability, competition, matchScore
**Validates: Requirements 6.2**

### Property 19: Recommendations Are Valid JSON
*For any* recommendation response, it should be parseable as valid JSON and validate against the BusinessRecommendation schema
**Validates: Requirements 6.3**

### Property 20: Manual Idea Triggers Analysis
*For any* request containing a businessIdea field, the system should call analyze_business_idea instead of generate_recommendations
**Validates: Requirements 6.4**

### Property 21: Match Scores in Valid Range
*For any* generated recommendation, the matchScore should be an integer between 0 and 100 inclusive
**Validates: Requirements 6.5**

### Property 22: Analysis Schema Completeness
*For any* business idea analysis, the response should contain all required fields: viabilityScore, marketFit, risks, suggestions, costStructure, scalability
**Validates: Requirements 7.2**

### Property 23: Analysis Is Valid JSON
*For any* analysis response, it should be parseable as valid JSON and validate against the BusinessAnalysis schema
**Validates: Requirements 7.4**

### Property 24: Invalid Request Returns 400
*For any* request with invalid schema (missing required fields or wrong types), the API should return HTTP status 400
**Validates: Requirements 8.4**

### Property 25: Valid Request Returns 200
*For any* request with valid schema and successful processing, the API should return HTTP status 200
**Validates: Requirements 8.5**

### Property 26: Cache Hit Returns Same Result
*For any* query executed twice within the cache TTL period, the second execution should return identical results to the first
**Validates: Requirements 9.2**

### Property 27: Embeddings Not Recomputed
*For any* document embedded once, subsequent operations should use the cached embedding without calling the embedding API again
**Validates: Requirements 9.3**

### Property 28: Cache Invalidation on Update
*For any* knowledge base update, all cache entries related to the updated documents should be cleared
**Validates: Requirements 9.4**

### Property 29: Cache Size Bounded
*For any* sequence of cache operations, the cache size should never exceed CACHE_MAX_SIZE entries
**Validates: Requirements 9.5**

### Property 30: API Request Logging Completeness
*For any* API request, the log entry should contain timestamp, anonymized user data, endpoint, and response time
**Validates: Requirements 10.1**

### Property 31: Error Logging Includes Stack Trace
*For any* exception raised during processing, the log should contain the full stack trace and error context
**Validates: Requirements 10.2**

### Property 32: FAISS Metrics Logged
*For any* FAISS search operation, the log should contain query time, number of results, and relevance scores
**Validates: Requirements 10.3**

### Property 33: Mistral API Usage Logged
*For any* Mistral API call, the log should contain tokens consumed, model name, and latency
**Validates: Requirements 10.4**

### Property 34: API Keys Never Logged
*For any* log entry or API response, it should not contain the MISTRAL_API_KEY or any substring of it
**Validates: Requirements 11.2**

### Property 35: API Key Format Validation
*For any* API key that doesn't match the expected format pattern, the system should reject it before making API calls
**Validates: Requirements 11.4**

## Error Handling

### Error Categories

1. **Configuration Errors**
   - Missing API keys
   - Invalid configuration values
   - Missing required files
   - **Handling**: Fail fast at startup with clear error messages

2. **API Errors**
   - Rate limiting (429)
   - Authentication failures (401)
   - Service unavailable (503)
   - **Handling**: Exponential backoff retry for transient errors, clear error messages for permanent failures

3. **Validation Errors**
   - Invalid request schema
   - Missing required fields
   - Type mismatches
   - **Handling**: Return HTTP 400 with detailed field-level errors

4. **Processing Errors**
   - FAISS index corruption
   - Embedding generation failures
   - JSON parsing errors
   - **Handling**: Log full error, return user-friendly message, attempt recovery where possible

5. **Resource Errors**
   - Out of memory
   - Disk space exhausted
   - Cache overflow
   - **Handling**: Graceful degradation, clear resource limit messages

### Error Response Format

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid request: missing required field 'professional_status'",
        "details": {
            "field": "professional_status",
            "expected": "string",
            "received": "null"
        }
    }
}
```

### Retry Strategy

- **Transient Errors**: Retry with exponential backoff (base delay: 1s, max retries: 3)
- **Rate Limits**: Respect Retry-After header, exponential backoff otherwise
- **Permanent Errors**: No retry, immediate failure

## Testing Strategy

### Unit Testing

The system will use **pytest** for unit testing with the following focus areas:

1. **Component Isolation**: Test each component independently with mocked dependencies
2. **Edge Cases**: Empty inputs, maximum sizes, boundary values
3. **Error Conditions**: Invalid inputs, API failures, resource exhaustion
4. **Data Validation**: Schema compliance, type checking, range validation

Example unit tests:
- `test_user_profile_parsing_with_all_fields()`
- `test_user_profile_validation_missing_required_field()`
- `test_faiss_search_returns_k_results()`
- `test_document_chunking_respects_token_limit()`
- `test_api_error_returns_user_friendly_message()`

### Property-Based Testing

The system will use **Hypothesis** for property-based testing with minimum 100 iterations per test. Each property test will be tagged with its corresponding design property number.

**Property Test Configuration**:
```python
from hypothesis import given, settings
import hypothesis.strategies as st

@settings(max_examples=100)
@given(user_profile=st.fixed_dictionaries({
    'professional_status': st.sampled_from(['employed_fulltime', 'freelancer', 'student']),
    'timeCommitment': st.integers(min_value=5, max_value=60),
    'budget': st.sampled_from(['under_1000', '1000_5000', '5000_25000'])
}))
def test_property_1_user_profile_parsing_completeness(user_profile):
    """
    Feature: mistral-rag-backend, Property 1: User Profile Parsing Completeness
    For any valid user profile JSON, parsing should extract all present fields without data loss
    """
    parsed = parse_user_profile(user_profile)
    assert all(parsed[key] == user_profile[key] for key in user_profile.keys())
```

**Key Property Tests**:
- Property 1: User profile parsing completeness
- Property 9: Search returns exactly K results
- Property 10: FAISS index persistence round-trip
- Property 11: Document chunks respect token limit
- Property 12: Consecutive chunks have overlap
- Property 16: Context token limit enforcement
- Property 17: Minimum recommendation count
- Property 18: Recommendation schema completeness
- Property 21: Match scores in valid range
- Property 26: Cache hit returns same result
- Property 34: API keys never logged

### Integration Testing

Integration tests will verify component interactions:

1. **End-to-End RAG Pipeline**: User profile → retrieval → generation → response
2. **API Endpoints**: HTTP request → processing → JSON response
3. **FAISS Operations**: Document ingestion → indexing → search → retrieval
4. **Cache Integration**: Request → cache check → computation → cache store

### Performance Testing

Performance benchmarks:
- **Recommendation Generation**: < 5 seconds for 3 recommendations
- **Business Idea Analysis**: < 3 seconds for analysis
- **FAISS Search**: < 100ms for top-5 retrieval
- **Cache Hit**: < 10ms response time
- **Index Load**: < 2 seconds for 10,000 documents

## Deployment Considerations

### Environment Setup

```bash
# Required environment variables
MISTRAL_API_KEY=your_api_key_here
FLASK_ENV=production
LOG_LEVEL=INFO

# Optional configuration
FAISS_INDEX_PATH=data/faiss_index
CACHE_TTL=3600
TOP_K_DOCUMENTS=5
```

### Directory Structure

```
backend/
├── app.py                 # Flask application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── data/
│   ├── faiss_index/      # FAISS index files
│   ├── documents/        # Source documents for knowledge base
│   └── metadata.json     # Document metadata
├── logs/
│   └── bizbot.log        # Application logs
├── src/
│   ├── rag_pipeline.py   # RAG orchestration
│   ├── faiss_retriever.py # FAISS operations
│   ├── mistral_client.py  # Mistral API wrapper
│   ├── document_processor.py # Document loading and chunking
│   ├── prompt_builder.py  # Prompt construction
│   └── cache_manager.py   # Caching logic
└── tests/
    ├── unit/             # Unit tests
    ├── integration/      # Integration tests
    └── property/         # Property-based tests
```

### Dependencies

```
# requirements.txt
flask==3.0.0
flask-cors==4.0.0
mistralai==0.1.0
faiss-cpu==1.7.4  # or faiss-gpu for GPU support
numpy==1.24.0
pydantic==2.5.0
python-dotenv==1.0.0
hypothesis==6.92.0  # for property-based testing
pytest==7.4.0
PyPDF2==3.0.0  # for PDF processing
```

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple Flask instances behind a load balancer
2. **FAISS Optimization**: Use GPU-accelerated FAISS for large knowledge bases (>100K documents)
3. **Cache Distribution**: Use Redis for shared caching across instances
4. **Async Processing**: Consider Celery for long-running analysis tasks
5. **Rate Limiting**: Implement per-user rate limits to prevent abuse

## Security Considerations

1. **API Key Protection**: Never log or expose Mistral API keys
2. **Input Sanitization**: Validate and sanitize all user inputs
3. **CORS Configuration**: Restrict CORS to specific frontend domains in production
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **Data Privacy**: Anonymize user data in logs
6. **HTTPS Only**: Enforce HTTPS in production
7. **Dependency Scanning**: Regularly scan dependencies for vulnerabilities

## Monitoring and Observability

### Metrics to Track

1. **Request Metrics**
   - Requests per minute
   - Average response time
   - Error rate by endpoint

2. **RAG Metrics**
   - FAISS search latency
   - Number of documents retrieved
   - Average relevance scores

3. **LLM Metrics**
   - Mistral API latency
   - Tokens consumed per request
   - API error rate

4. **Cache Metrics**
   - Cache hit rate
   - Cache size
   - Eviction rate

### Logging Strategy

- **Structured Logging**: Use JSON format for easy parsing
- **Log Levels**: DEBUG for development, INFO for production, ERROR for failures
- **Log Rotation**: Rotate logs daily, keep 30 days of history
- **Sensitive Data**: Never log API keys, PII, or full user profiles

## Future Enhancements

1. **Multi-Model Support**: Support for other LLM providers (OpenAI, Anthropic)
2. **Advanced RAG**: Implement hybrid search (dense + sparse), re-ranking
3. **User Feedback Loop**: Collect user ratings to improve recommendations
4. **Personalization**: Learn from user interactions to refine suggestions
5. **Real-Time Updates**: Stream recommendations as they're generated
6. **Multi-Language**: Support for non-English business ideas
7. **Industry-Specific Models**: Fine-tuned models for specific industries
