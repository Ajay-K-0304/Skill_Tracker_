"""
Standalone Development & Demo Gateway for Member 3 Module.
Allows running and demonstrating AI Forecasting & Agentic Curriculum Recommendation
independently without requiring Member 1's backend or database.

Usage:
    uvicorn services.ai_forecasting.standalone_app:app --host 127.0.0.1 --port 8003 --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.ai_forecasting.config import settings, configure_module_logging
from services.ai_forecasting.routers import router as ai_forecasting_router

# Ensure logging handlers and formatters are attached
configure_module_logging()
logger = logging.getLogger("services.ai_forecasting")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    has_groq_key = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY.strip()) > 5)
    logger.info(
        f"Member 3 AI Forecasting service ready. Provider: '{settings.LOCAL_LLM_PROVIDER}', "
        f"Model: '{settings.LOCAL_LLM_MODEL}', Groq configured: {has_groq_key}, Fallback enabled: {settings.ENABLE_LLM_FALLBACK}"
    )
    yield


app = FastAPI(
    title="Member 3: AI Forecasting & Agentic Curriculum Recommendation Service",
    description=(
        "Specialized sub-service for 6-to-12-month skill demand projection, "
        "deterministic trajectory classification, and automated syllabus gap evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Enable CORS for local dev testing (e.g. Member 4 frontend on vite localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the router under /api/v1
app.include_router(ai_forecasting_router, prefix="/api/v1")


@app.get("/", summary="Root Status")
async def root():
    return {
        "service": "AI Forecasting & Curriculum Recommendation",
        "member": "Member 3",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/forecast",
            "POST /api/v1/curriculum/analyze",
            "POST /api/v1/curriculum/analyze/upload",
            "GET /api/v1/health",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.ai_forecasting.standalone_app:app", host="127.0.0.1", port=8003, reload=True)
