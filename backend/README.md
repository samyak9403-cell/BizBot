# BizBot Backend - RAG-Powered Business Recommendation System

A Flask-based REST API that uses Retrieval-Augmented Generation (RAG) with Mistral AI and FAISS vector search to provide intelligent business recommendations and analysis.

## 🚀 Quick Start

```bash
# 1. Set up environment
export MISTRAL_API_KEY="your-mistral-api-key-here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build knowledge base (one-time)
python build_knowledge_base.py

# 4. Run application
python app.py

# 5. Application runs on http://localhost:5000
```

## 📋 Overview

BizBot is a backend system that helps entrepreneurs and business enthusiasts get personalized recommendations and analysis. It combines:

- **Mistral AI**: Large language model for generating intelligent recommendations
- **FAISS**: Vector database for fast semantic search over business documentation
- **RAG Pipeline**: Retrieves relevant context before generating responses (higher quality, more accurate answers)
- **Flask**: RESTful API for easy integration with frontend applications

### Key Features

✅ **Smart Recommendations** - Get personalized business recommendations based on your profile  
✅ **Business Analysis** - Analyze business ideas with market insights  
✅ **Semantic Search** - Find relevant business knowledge using FAISS  
✅ **Response Caching** - Fast cached responses for repeated queries  
✅ **Graceful Shutdown** - Clean resource cleanup on shutdown  
✅ **Health Monitoring** - Check application health and component status  
✅ **Error Handling** - Informative error messages and proper HTTP status codes  

---

## 📦 Architecture

```
┌─────────────────────────────────────────┐
│         Frontend (HTML/JS)              │
└────────────────┬────────────────────────┘
                 │ HTTP
                 ▼
┌─────────────────────────────────────────┐
│    Flask REST API (app.py)              │
│  ├─ /api/recommendations                │
│  ├─ /api/analyze                        │
│  └─ /health                             │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌────────────┐
   │   RAG   │      │   Cache    │
   │Pipeline │      │  Manager   │
   └────┬────┘      └────────────┘
        │
   ┌────┴──────────────────┬──────────────┐
   ▼                       ▼              ▼
┌──────────┐         ┌────────────┐  ┌─────────┐
│  FAISS   │         │  Mistral   │  │Document │
│ Retriever│         │   Client   │  │Processor│
└──────────┘         └────────────┘  └─────────┘
      │                    │
      ▼                    ▼
┌──────────────┐    ┌─────────────────┐
│ Vector Index │    │ Mistral API     │
│ (business    │    │ (embeddings &   │
│  documents)  │    │  generation)    │
└──────────────┘    └─────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Mistral AI API key (get one at https://console.mistral.ai)
- 500+ MB disk space for FAISS index

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd BizBot
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

**Key Dependencies:**
- Flask 3.0.0 - Web framework
- Pydantic 2.8.2 - Data validation
- FAISS (faiss-cpu 1.9.0) - Vector search
- MistralAI 1.0.0 - LLM client
- PyYAML, python-dotenv, requests

### Step 4: Configure Environment
Create a `.env` file in the backend directory:

```bash
# Required
MISTRAL_API_KEY=your-mistral-api-key-here

# Optional (defaults shown)
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
MISTRAL_MODEL=mistral-large-latest
FAISS_INDEX_PATH=data/faiss_index
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_DOCUMENTS=5
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
```

### Step 5: Build Knowledge Base
```bash
python build_knowledge_base.py

# Options:
python build_knowledge_base.py --rebuild        # Force rebuild
python build_knowledge_base.py --verbose        # Detailed output
python build_knowledge_base.py --log-level DEBUG  # Debug logging
```

### Step 6: Run Application
```bash
python app.py
```

Server starts on `http://localhost:5000`

---

## 🎯 Usage

### Starting the Server
```bash
python app.py

# Output:
# ================================================================================
# BizBot Backend Initialization Starting
# ================================================================================
# Step 1/7: Loading configuration...
# ✓ Configuration loaded (Flask env: development)
# ...
# ✅ BizBot Backend Initialization Complete
# Server will run on http://0.0.0.0:5000
# Press Ctrl+C to shut down gracefully
```

### API Endpoints

#### Check Server Health
```bash
curl http://localhost:5000/health
```

#### Generate Business Recommendations
```bash
curl -X POST http://localhost:5000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "education_level": "bachelor",
      "years_experience": 5,
      "industry": "technology",
      "business_type": "b2c",
      "time_commitment": "full_time",
      "risk_tolerance": "moderate",
      "investment_capital": 50000
    },
    "number_recommendations": 5,
    "diversity_preference": "high"
  }'
```

#### Analyze Business Idea
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "business_idea": "Online marketplace for handmade crafts",
    "user_profile": {
      "education_level": "bachelor",
      "years_experience": 3,
      "business_type": "b2c"
    },
    "analysis_depth": "comprehensive"
  }'
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

---

## 📁 Project Structure

```
backend/
├── app.py                    # Application entry point
├── build_knowledge_base.py   # CLI for building FAISS index
├── requirements.txt          # Python dependencies
├── .env.example              # Example configuration
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── api.py               # Flask REST API
│   ├── config.py            # Configuration management
│   ├── mistral_client.py    # Mistral AI wrapper
│   ├── document_processor.py # Document loading & chunking
│   ├── faiss_retriever.py   # Vector search
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── prompt_builder.py    # LLM prompt construction
│   ├── cache_manager.py     # Response caching
│   └── schemas.py           # Pydantic models
│
├── data/
│   ├── documents/           # Business documents
│   │   ├── saas_business_model.txt
│   │   ├── ecommerce_strategy.md
│   │   ├── service_business_guide.json
│   │   ├── digital_marketing_strategy.txt
│   │   └── startup_funding_guide.md
│   └── faiss_index/         # FAISS vector index (generated)
│
├── tests/
│   ├── unit/                # Unit tests (255 tests)
│   └── integration/         # Integration tests (12 tests)
│
└── logs/                     # Application logs
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Coverage
- **255 unit tests** - Component-level testing
- **12 app integration tests** - Full app lifecycle
- **Total: 272 tests** - All passing ✅

---

## 📖 Documentation

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions
- **Requirements: Python 3.10+, Mistral API key**

---

## 📚 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | Flask | 3.0.0 |
| Validation | Pydantic | 2.8.2 |
| LLM | Mistral AI | 1.0.0 |
| Vector Search | FAISS | 1.9.0 |
| Testing | Pytest | 7.4.0 |
| Python | - | 3.10+ |

---

## Status

**Production Ready** ✅  
**Test Coverage**: 272/272 tests passing  
**College Project** - Educational purposes

### POST /api/analyze
Analyze a user-provided business idea.
