import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.historical import ingest_historical_csv
from app.services.scraper import fetch_and_ingest_live_jobs

# Add these new imports for Step 9
from app.schemas.contract_a import ContractA
from app.services.skill_ingestion import ingest_contract_a_batch
from app.services.analytics import get_top_skills, get_skill_trend

router = APIRouter()

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    return {"status": "ok", "message": "FastAPI Server is running and Database session is active."}

@router.post("/ingest/historical")
async def trigger_historical_ingestion(db: AsyncSession = Depends(get_db)):
    file_path = "data/raw/sample_jobs.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {file_path}")
    result = await ingest_historical_csv(file_path, db)
    return {"message": "Batch ingestion complete", "metrics": result}

@router.post("/ingest/live")
async def trigger_live_ingestion(db: AsyncSession = Depends(get_db)):
    result = await fetch_and_ingest_live_jobs(db)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"message": "Live ingestion complete", "metrics": result}

# Add this new endpoint for Step 9
@router.post("/ingest/skills")
async def ingest_nlp_skills(contracts: List[ContractA], db: AsyncSession = Depends(get_db)):
    """Webhook for Member 2 to submit processed Contract A data."""
    try:
        metrics = await ingest_contract_a_batch(db, contracts)
        return {"message": "Skills successfully linked", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add these new endpoints at the bottom of the file

@router.get("/analytics/top-skills")
async def fetch_top_skills(limit: int = 10, db: AsyncSession = Depends(get_db)):
    """Endpoint for Member 4's React dashboard to display the most in-demand skills."""
    data = await get_top_skills(db, limit)
    return {"data": data}

@router.get("/analytics/trend/{skill_name}")
async def fetch_skill_trend(skill_name: str, db: AsyncSession = Depends(get_db)):
    """Endpoint for Member 3's forecasting models to get time-series data."""
    data = await get_skill_trend(db, skill_name)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for skill: {skill_name}")
    return {"skill": skill_name, "trend": data}