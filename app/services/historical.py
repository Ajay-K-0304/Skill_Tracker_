import pandas as pd
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.schemas import JobPosting
from app.services.etl_pipeline import clean_html, generate_job_hash

async def ingest_historical_csv(file_path: str, db: AsyncSession, chunk_size: int = 5000):
    """Reads a CSV in chunks, deduplicates, and inserts raw postings into PostgreSQL."""
    total_inserted = 0
    total_skipped = 0

    # Read CSV in chunks to optimize memory usage
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Replace NaNs with empty strings
        chunk = chunk.fillna("")

        for _, row in chunk.iterrows():
            title = str(row.get("title", ""))
            company = str(row.get("company", ""))
            location = str(row.get("location", ""))
            date_str = str(row.get("date", ""))
            raw_description = str(row.get("description", ""))

            # 1. Generate deterministic hash for deduplication
            job_hash = generate_job_hash(title, company, location)

            # 2. Check if this job already exists in the database
            stmt = select(JobPosting).where(JobPosting.job_hash == job_hash)
            result = await db.execute(stmt)
            existing_job = result.scalar_one_or_none()

            if existing_job:
                total_skipped += 1
                continue

            # 3. Clean HTML (Ready to be passed to Member 2's NLP pipeline later)
            _ = clean_html(raw_description) 

            # 4. Parse Date and generate missing Job IDs
            job_id = str(row.get("job_id", f"kaggle-{job_hash[:10]}"))
            try:
                posting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                posting_date = datetime.utcnow().date()

            # 5. Insert new job posting
            new_job = JobPosting(
                job_id=job_id,
                job_hash=job_hash,
                posting_date=posting_date
            )
            db.add(new_job)
            total_inserted += 1

        # Commit transaction after processing each chunk
        await db.commit()

    return {"inserted": total_inserted, "skipped_duplicates": total_skipped}