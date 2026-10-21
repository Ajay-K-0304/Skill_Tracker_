from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from app.models.schemas import JobPosting, SkillTaxonomy, JobSkill

async def get_top_skills(db: AsyncSession, limit: int = 10):
    """Fetches the most frequently requested skills across all job postings."""
    stmt = (
        select(
            SkillTaxonomy.canonical_name, 
            func.count(JobSkill.job_posting_id).label("job_count")
        )
        .join(JobSkill, SkillTaxonomy.id == JobSkill.skill_id)
        .group_by(SkillTaxonomy.canonical_name)
        .order_by(desc("job_count"))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [{"skill": row.canonical_name, "count": row.job_count} for row in result.all()]

async def get_skill_trend(db: AsyncSession, skill_name: str):
    """Fetches time-series daily frequency trend for a specific skill."""
    stmt = (
        select(
            JobPosting.posting_date, 
            func.count(JobPosting.id).label("job_count")
        )
        .join(JobSkill, JobPosting.id == JobSkill.job_posting_id)
        .join(SkillTaxonomy, JobSkill.skill_id == SkillTaxonomy.id)
        .where(SkillTaxonomy.canonical_name == skill_name)
        .group_by(JobPosting.posting_date)
        .order_by(JobPosting.posting_date)
    )
    result = await db.execute(stmt)
    return [{"date": row.posting_date, "count": row.job_count} for row in result.all()]