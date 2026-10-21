from pydantic import BaseModel, Field
from typing import List
from datetime import date

class ContractA(BaseModel):
    job_id: str = Field(..., description="Unique posting identifier")
    extracted_skills: List[str] = Field(..., description="List of canonical skills")
    posting_date: date = Field(..., description="Date of the job posting in YYYY-MM-DD format")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "12345",
                "extracted_skills": ["Docker", "FastAPI", "Python"],
                "posting_date": "2026-10-01"
            }
        }