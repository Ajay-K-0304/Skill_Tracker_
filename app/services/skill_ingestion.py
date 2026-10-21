from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.schemas import SkillTaxonomy, JobSkill, JobPosting
from app.schemas.contract_a import ContractA

async def ingest_contract_a_batch(db: AsyncSession, contracts: list[ContractA]):
    """Processes Contract A outputs and populates the skill taxonomy and job-skill relationships."""
    total_skills_added = 0
    total_links_added = 0

    for contract in contracts:
        # 1. Fetch the corresponding job posting using job_id
        stmt = select(JobPosting).where(JobPosting.job_id == contract.job_id)
        result = await db.execute(stmt)
        job_posting = result.scalar_one_or_none()

        if not job_posting:
            continue  # Skip if the job was not ingested properly beforehand

        for skill_name in contract.extracted_skills:
            # 2. Check if the skill already exists in the Taxonomy
            skill_stmt = select(SkillTaxonomy).where(SkillTaxonomy.canonical_name == skill_name)
            skill_result = await db.execute(skill_stmt)
            skill_obj = skill_result.scalar_one_or_none()

            # 3. If it is a new skill, add it to the Taxonomy
            if not skill_obj:
                skill_obj = SkillTaxonomy(canonical_name=skill_name, category="Uncategorized")
                db.add(skill_obj)
                await db.flush()  # Flush pushes the object to the DB to get the ID without committing the transaction
                total_skills_added += 1

            # 4. Check if this specific Job-Skill link already exists
            link_stmt = select(JobSkill).where(
                JobSkill.job_posting_id == job_posting.id,
                JobSkill.skill_id == skill_obj.id
            )
            link_result = await db.execute(link_stmt)
            existing_link = link_result.scalar_one_or_none()

            # 5. Link the skill to the job posting
            if not existing_link:
                new_link = JobSkill(job_posting_id=job_posting.id, skill_id=skill_obj.id)
                db.add(new_link)
                total_links_added += 1

    # Commit all changes to the database at once for efficiency
    await db.commit()
    return {"new_taxonomy_skills": total_skills_added, "job_skill_links_created": total_links_added}