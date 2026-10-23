# Member 3: AI Forecasting & Agentic Curriculum Recommendation Service

**Specialist Track:** Temporal Trend Analysis, Time-Series Demand Forecasting & Stateful Curriculum Evaluation  
**Branch:** `feature/ai-forecasting`  
**Package:** `services/ai_forecasting/`  

---

## 1. Module Overview & Responsibilities

The **AI Forecasting & Agentic Curriculum Recommendation** module projects industry technical skill demand over 6 to 12 months, classifies skills into deterministic trajectory states (`Emerging`, `Core/Stable`, `Declining`, `Niche`), ingests academic university syllabi (PDF / Text), evaluates course coverage against industry role benchmarks, and generates actionable curriculum recommendations adhering strictly to **Contract B**.

### Key Architectural Capabilities
1. **Three Pluggable Data Modes**:
   - `synthetic_demo`: Fully deterministic, mathematically modeled 24-month skill trajectories for demo & automated testing without database dependencies.
   - `csv`: Ingestion and validation of relational trio exports (`job_postings.csv`, `job_skills.csv`, `skill_taxonomy.csv`) or pre-aggregated monthly CSVs.
   - `live_database`: Non-intrusive dynamic SQL aggregation querying Member 1's PostgreSQL schema (`job_postings -> job_skills -> skill_taxonomy`) without requiring any new database tables.
2. **Defensible Time-Series Forecasting**:
   - Scikit-learn Ridge regression with linear/quadratic trends and cyclical annual harmonics.
   - Dynamic widening prediction intervals ($z \cdot \sigma_h$) with strict non-negative demand clamps.
   - Time-ordered holdout evaluation for MAE and RMSE (no random splitting; sample limitations explicitly explained for $N < 12$).
3. **Deterministic Trajectory Classifier**:
   - Categorizes skills into `Emerging`, `Core/Stable`, `Declining`, `Niche`, or `Insufficient Data` using verifiable statistical thresholds.
4. **Syllabus Document Parsing**:
   - Robust PDF (`pypdf`) and plain-text extraction with file size checks and scanned PDF detection (alerting when OCR is required).
   - Deterministic extraction of course title, units, topics, and candidate skill mentions.
5. **Stateful Agentic Workflow (LangGraph)**:
   - Coordinated 6-node state machine for input validation, parsing, benchmark matching, gap scoring, and recommendation synthesis.
   - Local LLM adapter with short timeouts and deterministic fallback ensuring 100% demo uptime.
6. **Strict Schema Conformance**:
   - Exposes `POST /api/v1/forecast` and `POST /api/v1/curriculum/analyze` matching exact project contracts.

---

## 2. Directory Structure

```text
services/ai_forecasting/
├── __init__.py
├── config.py                 # Pydantic BaseSettings configuration
├── schemas.py                # Contract B and forecasting Pydantic models
├── routers.py                # FastAPI APIRouter exposing endpoints
├── standalone_app.py         # Independent development and demo entry point
├── data/
│   ├── interfaces.py         # DataSourceAdapter protocol
│   ├── aggregator.py         # Missing-month zero-filling & data validation
│   ├── synthetic_generator.py# Deterministic synthetic time-series generator
│   ├── csv_loader.py         # Relational & pre-aggregated CSV loader
│   └── db_adapter.py         # PostgreSQL dynamic aggregation adapter
├── forecasting/
│   ├── interfaces.py         # ForecastingEngine base protocol
│   └── engine.py             # Sklearn Ridge model with holdout evaluation
├── classification/
│   └── trajectory.py         # Deterministic trajectory classifier
├── syllabus/
│   ├── extractor.py          # PDF/Text extraction & scanned PDF detection
│   └── parser.py             # Regex & canonical skill parser
├── curriculum/
│   ├── benchmarks.py         # Target role benchmarks & fallback lookup
│   └── gap_analyzer.py       # Coverage score & missing skill calculator
├── agents/
│   ├── llm_adapter.py        # Local LLM adapter with deterministic fallback
│   └── workflow.py           # LangGraph state machine workflow
├── fixtures/                 # Sample CSVs, syllabus text & PDF fixtures
│   ├── job_postings.csv      # 301 postings baseline
│   ├── job_skills.csv        # 4 skill links baseline
│   ├── skill_taxonomy.csv    # 4 skills taxonomy
│   ├── monthly_skill_demand.csv # Pre-aggregated test fixture
│   ├── sample_syllabus.txt   # Distributed Systems sample syllabus
│   └── sample_syllabus.pdf   # Compiled PDF test fixture
└── tests/                    # Comprehensive unit & integration tests (49 tests)
    ├── test_phase1_schemas.py
    ├── test_phase2_data.py
    ├── test_phase3_forecasting.py
    ├── test_phase4_classification.py
    ├── test_phase5_syllabus.py
    ├── test_phase6_gap_analysis.py
    ├── test_phase7_workflow.py
    └── test_phase8_api.py
```

---

## 3. Quick Start & Demonstration

### Step 1: Run the Automated Test Suite
Verify that all 49 automated tests pass:
```powershell
python -m pytest services/ai_forecasting/tests/ -v
```

### Step 2: Start the Standalone Demo Server
Run the Member 3 standalone service on port 8003:
```powershell
python -m uvicorn services.ai_forecasting.standalone_app:app --host 127.0.0.1 --port 8003 --reload
```
Interactive Swagger documentation is available at:
👉 **`http://127.0.0.1:8003/docs`**

---

## 4. API Request & Response Examples

### Endpoint A: `POST /api/v1/forecast`

#### Request (Demo Mode):
```bash
curl -X POST "http://127.0.0.1:8003/api/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "synthetic_demo",
    "horizon_months": 6,
    "skills": ["FastAPI", "Python", "jQuery", "Rust"]
  }'
```

#### Response (Snippet):
```json
{
  "skill_forecasts": [
    {
      "skill_name": "FastAPI",
      "horizon_months": 6,
      "last_historical_month": "2026-10",
      "forecast_origin_month": "2026-10",
      "forecast": [
        {"year_month": "2026-11", "predicted_count": 120.2, "lower_bound": 104.4, "upper_bound": 135.9},
        {"year_month": "2026-12", "predicted_count": 123.4, "lower_bound": 107.3, "upper_bound": 139.5}
      ],
      "trajectory": {
        "label": "Emerging",
        "growth_rate": 1.429,
        "recent_vs_earlier_ratio": 2.429,
        "summary": "Emerging skill: strong upward growth of 142.9%...",
        "data_sufficiency": true
      },
      "metrics": {
        "mae": 17.78,
        "rmse": 18.27,
        "evaluation_samples": 3,
        "notes": "Time-ordered holdout evaluation (3 months). No random splitting."
      },
      "status": "success",
      "limitations": []
    }
  ],
  "data_source": "synthetic_demo",
  "forecast_origin": "2026-10",
  "generated_at": "2026-10-22T13:29:00Z",
  "warnings": []
}
```

---

### Endpoint B: `POST /api/v1/curriculum/analyze` (Contract B)

#### Request (JSON Payload):
```bash
curl -X POST "http://127.0.0.1:8003/api/v1/curriculum/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "syllabus_text": "Course: Distributed Systems & Cloud Computing\nUnit 1: Linux Basics\nUnit 2: Docker Containers\nUnit 3: REST APIs",
    "target_role": "Cloud Engineer",
    "use_llm": false
  }'
```

#### Response (Exact Contract B):
```json
{
  "course_name": "Distributed Systems & Cloud Computing",
  "coverage_score": 42.9,
  "covered_skills": [
    "Docker",
    "Linux",
    "REST APIs"
  ],
  "missing_skills": [
    "AWS",
    "FastAPI",
    "Kubernetes",
    "gRPC"
  ],
  "recommendations": [
    {
      "unit": "Unit 2: Docker Containers",
      "action": "Add a hands-on Kubernetes orchestration module covering Pods, Deployments, and Service networking alongside Docker.",
      "priority": "High"
    },
    {
      "unit": "Unit 3: REST APIs",
      "action": "Introduce gRPC and Protocol Buffers alongside REST APIs for high-performance inter-service communication.",
      "priority": "High"
    },
    {
      "unit": "Unit 3: REST APIs",
      "action": "Incorporate modern asynchronous API development using FastAPI and Pydantic validation.",
      "priority": "Medium"
    },
    {
      "unit": "Unit 2: Docker Containers",
      "action": "Integrate AWS core cloud infrastructure concepts (EC2, S3, IAM) with hands-on lab deployments.",
      "priority": "Medium"
    }
  ]
}
```

#### Uploading a PDF File (`POST /api/v1/curriculum/analyze/upload`):
```bash
curl -X POST "http://127.0.0.1:8003/api/v1/curriculum/analyze/upload" \
  -F "file=@services/ai_forecasting/fixtures/sample_syllabus.pdf" \
  -F "target_role=Cloud Engineer" \
  -F "use_llm=false"
```

---

## 5. Integration Guide for Member 1 (Central Gateway)

When branches are merged, Member 1 can mount Member 3's router into `app/main.py` with two lines of code:

```python
# In app/main.py:
from services.ai_forecasting.routers import router as ai_forecasting_router

# Mount under API prefix
app.include_router(ai_forecasting_router, prefix="/api/v1")
```

### Live Database Configuration:
Member 1's existing SQLAlchemy models (`JobPosting`, `JobSkill`, `SkillTaxonomy`) connect cleanly to Member 3's `LiveDatabaseAdapter`:
```python
# Member 3 queries:
# job_postings (posting_date) -> job_skills -> skill_taxonomy (canonical_name)
# No new monthly table required!
```

---

## 6. Local LLM Configuration (Phase 7)

By default, the service operates in **deterministic fallback mode**, ensuring that tests and demonstrations run immediately without requiring Ollama or external models.

To activate a live local LLM:
1. Start Ollama locally:
   ```bash
   ollama run llama3:8b
   ```
2. Set environment variables in `.env`:
   ```bash
   AI_FORECASTING_LOCAL_LLM_PROVIDER=ollama
   AI_FORECASTING_LOCAL_LLM_BASE_URL=http://localhost:11434
   AI_FORECASTING_LOCAL_LLM_MODEL=llama3:8b
   AI_FORECASTING_LOCAL_LLM_TIMEOUT=5.0
   ```
3. Set `"use_llm": true` in requests to `POST /api/v1/curriculum/analyze`. If Ollama is offline or times out, the service automatically falls back to deterministic recommendations without returning an error.
