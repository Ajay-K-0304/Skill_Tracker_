from sqlalchemy import Column, Integer, String, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base

class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    job_hash = Column(String, unique=True, index=True, nullable=False)
    posting_date = Column(Date, nullable=False)
    
    skills = relationship("JobSkill", back_populates="job", cascade="all, delete-orphan")

class SkillTaxonomy(Base):
    __tablename__ = "skill_taxonomy"

    id = Column(Integer, primary_key=True, index=True)
    canonical_name = Column(String, unique=True, index=True, nullable=False)
    category = Column(String, nullable=True) 
    
    jobs = relationship("JobSkill", back_populates="skill")

class JobSkill(Base):
    __tablename__ = "job_skills"

    id = Column(Integer, primary_key=True, index=True)
    job_posting_id = Column(Integer, ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skill_taxonomy.id", ondelete="CASCADE"), nullable=False)

    __table_args__ = (UniqueConstraint("job_posting_id", "skill_id", name="_job_skill_uc"),)

    job = relationship("JobPosting", back_populates="skills")
    skill = relationship("SkillTaxonomy", back_populates="jobs")