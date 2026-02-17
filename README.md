# 🚀 BizBot — AI-Powered Startup Idea Recommender

BizBot is an intelligent startup recommendation engine that matches aspiring entrepreneurs with the perfect business idea based on their skills, experience, budget, and preferences.

Unlike generic chatbots, BizBot uses a **deterministic scoring engine** powered by **999 enriched startup ideas** and **Mistral AI** to deliver personalized, data-driven recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![Mistral AI](https://img.shields.io/badge/Mistral-AI-purple)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)

---

## ✨ Features

### 🎯 Smart Questionnaire → Personalized Matches
Answer 7 simple questions about your skills, budget, and interests. BizBot scores all 999 ideas against your profile and returns the best matches with a **match percentage**.

### 🔍 Idea Analyzer
Already have a business idea? Enter it and get an AI-powered analysis including viability score, SWOT analysis, market fit, risks, and actionable suggestions.

### 📊 Deterministic Scoring (Not Random)
Match percentages come from a **weighted scoring algorithm**, not LLM guessing. Same profile always returns the same results.

### 🧠 Mistral AI + RAG Pipeline
Uses Retrieval-Augmented Generation for context-aware business analysis with real knowledge base documents.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                  Frontend                    │
│  Landing Page → Questionnaire → Results      │
│  Landing Page → Idea Input → AI Analysis     │
└──────────────────┬──────────────────────────┘
                   │ HTTP POST
┌──────────────────▼──────────────────────────┐
│               Flask API Server               │
│  /api/match    → Scoring Engine              │
│  /api/analyze  → Mistral AI + RAG Pipeline   │
│  /health       → System Status               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            Data & AI Layer                   │
│  999 Enriched Startup Ideas (CSV)            │
│  FAISS Vector Index (Semantic Search)        │
│  Mistral AI (Embeddings + Chat)              │
│  Weighted Scoring Algorithm                  │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
BizBot/
├── frontend/
│   ├── land_page.html          # Landing page
│   ├── questionnaire.html      # 7-step questionnaire
│   ├── recommendations.html    # Match results display
│   ├── idea-input.html         # Business idea input
│   ├── analysis.html           # AI analysis results
│   ├── learn.html              # Learning resources
│   ├── about.html              # About page
│   └── settings.html           # Settings
│
├── backend/
│   ├── app.py                  # Application entry point
│   ├── wsgi.py                 # WSGI entry for deployment
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variables template
│   │
│   ├── src/
│   │   ├── api.py              # Flask API routes
│   │   ├── recommendation_engine.py  # Scoring algorithm
│   │   ├── rag_pipeline.py     # RAG pipeline (Mistral + FAISS)
│   │   ├── mistral_client.py   # Mistral API client
│   │   ├── faiss_retriever.py  # FAISS vector search
│   │   ├── document_processor.py    # Document chunking
│   │   ├── csv_loader.py       # CSV data loader
│   │   ├── cache_manager.py    # Response caching
│   │   ├── config.py           # Configuration
│   │   ├── prompt_builder.py   # Prompt templates
│   │   └── schemas.py          # Pydantic validation
│   │
│   ├── data/
│   │   └── documents/
│   │       ├── ideas_enriched.csv   # 999 enriched startup ideas
│   │       └── *.txt, *.md, *.json  # Knowledge base documents
│   │
│   └── tests/
│       └── unit/               # 301 unit tests
│
├── render.yaml                 # Render deployment config
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Mistral API key ([Get one here](https://console.mistral.ai/))

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/BizBot.git
cd BizBot
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Mistral API key
MISTRAL_API_KEY=your_api_key_here
```

### 4. Run the application

```bash
python app.py
```

### 5. Open in browser

Navigate to **http://localhost:5000**

---

## 🎮 How to Use

### Find Your Perfect Business Idea
1. Click **"Find My Perfect Idea"** on the landing page
2. Complete the 7-step questionnaire:
   - Your skills (Tech, Marketing, Sales, etc.)
   - Experience level
   - Preferred industries
   - Business model (B2B/B2C/Both)
   - Starting budget
   - Time commitment
   - Professional network strength
3. View your personalized matches ranked by **match percentage**

### Analyze Your Own Idea
1. Click **"I Already Have an Idea"** on the landing page
2. Type your business idea description
3. Get AI-powered analysis including:
   - Viability score
   - SWOT analysis
   - Market fit assessment
   - Risk identification
   - Actionable suggestions

---

## 🔧 API Endpoints

### `POST /api/match` — Get Personalized Recommendations

```bash
curl -X POST http://localhost:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["technology", "marketing"],
    "experience_level": "intermediate",
    "preferred_industries": ["AI/ML", "SaaS"],
    "business_model": "B2C",
    "starting_capital": 10000,
    "time_commitment": "full_time",
    "network_strength": "moderate",
    "desired_income": 100000,
    "top_n": 5
  }'
```

**Response:**
```json
{
  "matches": [
    {
      "rank": 1,
      "match_score": 0.83,
      "match_percentage": "83%",
      "idea_text": "AI-powered resume screening tool...",
      "domain": "AI/ML",
      "business_model": "B2C",
      "difficulty": "Medium",
      "scalability": "High",
      "estimated_cost_bucket": "1000-10000",
      "required_skills": "Python, ML, NLP",
      "explanation": "Strong match because...",
      "score_breakdown": {
        "domain_match": 0.95,
        "skill_overlap": 0.80,
        "experience_fit": 0.75,
        "scalability_fit": 0.90,
        "business_model_match": 1.0,
        "network_leverage": 0.60,
        "cost_fit": 1.0
      }
    }
  ],
  "total_ideas_scored": 999,
  "profile_summary": "intermediate entrepreneur interested in AI/ML, SaaS"
}
```

### `POST /api/analyze` — Analyze a Business Idea

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "business_idea": "A subscription service for organic pet food delivery"
  }'
```

### `GET /health` — Health Check

```bash
curl http://localhost:5000/health
```

---

## 🧪 Running Tests

```bash
cd backend
.\venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

**Test Coverage: 301 tests**

| Module | Tests |
|--------|-------|
| Schemas & Validation | 47 |
| Flask API Server | 34 |
| CSV Loader | 29 |
| Recommendation Engine | 12 |
| RAG Pipeline | 30+ |
| Mistral Client | 25+ |
| Document Processor | 40+ |
| FAISS Retriever | 30+ |
| Other modules | 50+ |

---

## 🧠 How the Scoring Works

BizBot uses a **7-component weighted scoring algorithm**:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Domain Match | 25% | Do preferred industries align? |
| Skill Overlap | 25% | Do your skills match required skills? |
| Experience Fit | 15% | Does your experience level match difficulty? |
| Scalability Fit | 10% | Does scalability match income goals? |
| Business Model | 10% | Does B2B/B2C preference align? |
| Network Leverage | 5% | Can your network help this idea? |
| Cost Fit | 10% | Is the idea within your budget? |

**All scoring is deterministic** — the same profile always produces the same results. Mistral AI is only used AFTER scoring to generate natural language explanations.

---

## 🌐 Deployment (Render)

### One-click deploy

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` configuration
5. Add environment variable: `MISTRAL_API_KEY`
6. Click **Deploy**

### Manual deploy config

| Setting | Value |
|---------|-------|
| Runtime | Python 3.10 |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && gunicorn wsgi:app --bind 0.0.0.0:$PORT` |
| Root Directory | `/` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (Vanilla) |
| Backend | Python, Flask |
| AI/LLM | Mistral AI (mistral-small-latest) |
| Vector Search | FAISS (facebook/faiss) |
| Embeddings | Mistral Embed (1024-dim) |
| Validation | Pydantic v2 |
| Data Processing | Pandas |
| Deployment | Render, Gunicorn |

---

## 📊 Data Pipeline

```
ideas.csv (1000 raw ideas)
    ↓ enrich_ideas.py (Mistral AI classification)
ideas_enriched.csv (999 structured ideas)
    ↓ Each idea has:
    ├── domain (FinTech, HealthTech, AI/ML, etc.)
    ├── business_model (B2B, B2C, Both)
    ├── estimated_cost_bucket (<1000, 1000-10000, etc.)
    ├── difficulty (Low, Medium, High)
    ├── scalability (Low, Medium, High)
    ├── required_skills (comma-separated)
    ├── target_customer (description)
    └── short_summary (refined description)
```

---

## 👥 Team

Built as a college project demonstrating AI-powered recommendation systems.

---

## 📄 License

This project is for educational purposes.
