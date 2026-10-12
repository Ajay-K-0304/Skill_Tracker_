# CONTEXT.md

## Project Overview

**Skill Requirement Evolution Tracker (Industry 4.0 AI System)** is a 4th-year B.Tech CSE/IT Major Project designed to bridge the academia-industry curriculum gap. The system continuously ingests tech job listings from public APIs and historical datasets, extracts changing technical skill requirements using Natural Language Processing (NLP), forecasts skill demand trajectories (6–12 months) using time-series machine learning, and uses an agentic workflow to evaluate university syllabi against market trends to generate automated curriculum gap analysis reports and course update recommendations.

* **Project Type:** Final Year Major Project (CSE/IT)
* **Team Size:** 4 Members (Fully Decoupled / Independent Parallel Tracks)
* **Timeline:** 3 Months
* **Core Technology Stack:** FastAPI (Backend), React 18 + Vite + Tailwind CSS (Frontend), PostgreSQL + `pgvector` (Database), spaCy (NLP), Prophet / Scikit-learn (Time-Series ML), LangChain + LangGraph (Agentic AI Workflows), Pandas & NumPy (Data Processing).
* **SDG Alignment:** SDG 4 (Quality Education), SDG 9 (Industry, Innovation, and Infrastructure), SDG 17 (Partnerships for the Goals).

---

## Architecture & Module Breakdown

### Module 1: Industry Data Ingestion Module (Member 1)

* **Directory:** `/services/ingestion/`
* **Branch:** `feature/data-ingestion`
* **Core Functionality:** Automated batch ingestion of historical Kaggle tech job datasets and daily streaming ingestion via Adzuna / Arbeitnow REST APIs. Performs HTML stripping, text cleaning, metadata validation, and database storage.
* **Key Tech:** Python 3.11+, PostgreSQL, SQLAlchemy ORM, Pandas, HTTPX, BeautifulSoup4, FastAPI.

### Module 2: Skill Ontology & NLP Extraction Module (Member 2)

* **Directory:** `/services/nlp_engine/`
* **Branch:** `feature/nlp-ontology`
* **Core Functionality:** Named Entity Recognition (NER) pipeline scanning job descriptions to extract tech skills. Maps raw/variant skill names (e.g., `"React.js"`, `"ReactJS"`) to canonical taxonomy IDs (`"React"`). Computes skill co-occurrence matrices for tech stack relationships.
* **Key Tech:** spaCy (`en_core_web_sm` / `_md`, `EntityRuler`), Python (`itertools`, `collections.Counter`), Pandas.

### Module 3: Temporal Trend Analysis & Agentic Syllabus Module (Member 3)

* **Directory:** `/services/ai_forecasting/`
* **Branch:** `feature/ai-forecasting`
* **Core Functionality:** Time-series forecasting (Prophet / Scikit-learn) projecting 6-month and 12-month skill trajectories into 4 categories (`Emerging`, `Core/Stable`, `Declining`, `Niche`). Operates a stateful LangGraph + LangChain agent that parses PDF syllabi (`pdfplumber`), maps taught topics against industry market benchmarks, and generates structured gap analysis scorecards and recommendations.
* **Key Tech:** Meta Prophet, Statsmodels, LangChain, LangGraph, Pydantic v2, `pdfplumber` / PyPDF.

### Module 4: Analytics Dashboard & UI Module (Member 4)

* **Directory:** `/frontend/`
* **Branch:** `feature/react-dashboard`
* **Core Functionality:** Interactive web analytics dashboard. Features role-based skill heatmaps, interactive Recharts multi-line growth curves, skill co-occurrence network graphs, drag-and-drop syllabus PDF uploader, gap scorecard views, and PDF/CSV report exports.
* **Key Tech:** React 18, Vite, Tailwind CSS, Recharts / Chart.js, `@xyflow/react` (React Flow), `react-dropzone`.

---

## Strict Interface Contracts (Decoupling Protocol)

To maintain zero runtime dependencies between team members during development, all modules interact strictly via predefined JSON schemas.

### Contract A: Extracted Skills Output (Member 2 $\rightarrow$ Member 1 & 3)

```json
{
  "job_id": "12345",
  "extracted_skills": ["Python", "FastAPI", "Docker"],
  "posting_date": "2026-10-01"
}

```

### Contract B: Syllabus Gap Analysis Output (Member 3 $\rightarrow$ Member 4)

```json
{
  "course_name": "Distributed Systems & Cloud Computing",
  "coverage_score": 71.4,
  "covered_skills": ["Docker", "Linux", "REST APIs"],
  "missing_skills": ["Kubernetes", "gRPC", "FastAPI"],
  "recommendations": [
    {
      "unit": "Unit 3: Service Communication",
      "action": "Introduce gRPC and Protocol Buffers alongside REST APIs.",
      "priority": "High"
    }
  ]
}

```

---

## Project Repository Layout

```text
skill-requirement-evolution-tracker/
├── CONTEXT.md               # Project Context and Architecture Specification
├── contracts/               # Shared JSON schemas & sample responses
│   ├── skills_schema.json
│   └── syllabus_report.json
├── frontend/                # Member 4 Workspace (React + Vite + Tailwind)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── mock/            # Local mock JSON files for isolated dev
│   │   └── services/api.js  # Abstraction layer with VITE_USE_MOCK toggle
├── services/
│   ├── ingestion/           # Member 1 Workspace (ETL & Database)
│   ├── nlp_engine/          # Member 2 Workspace (spaCy NLP & Taxonomy)
│   └── ai_forecasting/      # Member 3 Workspace (Prophet & LangGraph Agent)
├── database/                # PostgreSQL scripts & Alembic migrations
├── main.py                  # Central FastAPI Gateway (Integration Phase)
├── requirements.txt         # Global Python dependencies
└── README.md

```

---

## Database Schema (PostgreSQL)

* **`job_postings`**: `id` (UUID), `source` (VARCHAR), `external_id` (VARCHAR), `title` (VARCHAR), `company` (VARCHAR), `location` (VARCHAR), `raw_description` (TEXT), `cleaned_description` (TEXT), `posted_date` (DATE).
* **`skill_taxonomy`**: `id` (UUID), `canonical_name` (VARCHAR), `category` (VARCHAR), `aliases` (TEXT[]).
* **`job_skills`**: `job_id` (UUID, FK), `skill_id` (UUID, FK), `confidence_score` (FLOAT).
* **`skill_monthly_stats`**: `id` (UUID), `skill_id` (UUID, FK), `year_month` (DATE), `mention_count` (INT), `total_jobs_analyzed` (INT), `frequency_ratio` (FLOAT).
* **`syllabus_reports`**: `id` (UUID), `course_name` (VARCHAR), `target_role` (VARCHAR), `extracted_skills` (JSONB), `missing_skills` (JSONB), `coverage_score` (FLOAT), `recommendations` (JSONB).

---

## Integration Strategy

1. **Months 1–3 (Isolated Development):** Each member builds exclusively inside their folder and branch (`feature/*`). Member 4 uses mock JSON files (`VITE_USE_MOCK=true`) for UI development.
2. **Month 4 (Final Merge & Integration):**
* Members merge their branches into `main` via GitHub Pull Requests.
* Member 1 routes imports from `/services/ingestion/`, `/services/nlp_engine/`, and `/services/ai_forecasting/` into `main.py` (FastAPI).
* Member 4 switches `VITE_USE_MOCK=false` to connect React components directly to FastAPI REST endpoints.