import httpx
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.schemas import JobPosting
from app.services.etl_pipeline import clean_html, generate_job_hash

ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"

async def fetch_and_ingest_live_jobs(db: AsyncSession):
    """Fetches fresh job postings from Arbeitnow API and ingests them."""
    total_inserted = 0
    total_skipped = 0

    # Use HTTPX for asynchronous API requests
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(ARBEITNOW_API_URL, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            return {"error": f"Failed to fetch live jobs: {str(e)}"}

    jobs = data.get("data", [])

    for job in jobs:
        title = job.get("title", "")
        company = job.get("company_name", "")
        location = job.get("location", "")
        raw_description = job.get("description", "")
        
        # 1. Generate deterministic hash for deduplication
        job_hash = generate_job_hash(title, company, location)
        
        # 2. Check if this job already exists in the database
        stmt = select(JobPosting).where(JobPosting.job_hash == job_hash)
        result = await db.execute(stmt)
        existing_job = result.scalar_one_or_none()

        if existing_job:
            total_skipped += 1
            continue

        # 3. Clean HTML (Ready to be passed to NLP later)
        _ = clean_html(raw_description)

        # 4. Parse Metadata
        job_id = str(job.get("slug", f"live-{job_hash[:10]}"))
        
        # 5. Insert new job posting
        new_job = JobPosting(
            job_id=job_id,
            job_hash=job_hash,
            posting_date=datetime.utcnow().date()
        )
        db.add(new_job)
        total_inserted += 1

    # Commit transaction after processing the entire page of results
    await db.commit()
    return {"inserted": total_inserted, "skipped_duplicates": total_skipped}