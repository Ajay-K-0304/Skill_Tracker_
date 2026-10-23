"""
Skill-gap and coverage analysis service.
Computes covered skills, missing skills, and coverage percentage
against industry role benchmarks with case-insensitive normalization.
"""

from typing import List, Set, Tuple


def _canonicalize_skill_name(name: str) -> str:
    """Standardizes casing and trims whitespace for canonical comparison."""
    clean = name.strip()
    # Normalize common abbreviations and acronyms
    norm_map = {
        "rest api": "REST APIs",
        "rest apis": "REST APIs",
        "restful api": "REST APIs",
        "grpc": "gRPC",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        "sql": "SQL",
        "aws": "AWS",
    }
    low = clean.lower()
    if low in norm_map:
        return norm_map[low]
    # For standard Camel/Pascal/Title names
    if low in ["fastapi", "react", "tailwind", "python", "docker", "linux", "kubernetes", "git", "kafka"]:
        return {
            "fastapi": "FastAPI",
            "react": "React",
            "tailwind": "Tailwind",
            "python": "Python",
            "docker": "Docker",
            "linux": "Linux",
            "kubernetes": "Kubernetes",
            "git": "Git",
            "kafka": "Kafka",
        }[low]
    return clean


def analyze_skill_gap(
    syllabus_skills: List[str],
    benchmark_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Computes coverage score and separates covered skills from missing skills.

    Formula:
        coverage_score = (covered unique benchmark skills / total unique benchmark skills) * 100

    Returns:
        Tuple of (coverage_score, covered_skills, missing_skills)

    Raises:
        ValueError: If benchmark_skills is empty.
    """
    if not benchmark_skills:
        raise ValueError("Target role benchmark must contain at least one skill.")

    # Deduplicate and canonicalize benchmark skills
    clean_benchmark: List[str] = []
    benchmark_lookup: Set[str] = set()

    for s in benchmark_skills:
        c_name = _canonicalize_skill_name(s)
        low = c_name.lower()
        if low not in benchmark_lookup:
            benchmark_lookup.add(low)
            clean_benchmark.append(c_name)

    # Normalize syllabus skills
    syllabus_set: Set[str] = {_canonicalize_skill_name(s).lower() for s in syllabus_skills if s.strip()}

    covered: List[str] = []
    missing: List[str] = []

    for b_skill in clean_benchmark:
        if b_skill.lower() in syllabus_set:
            covered.append(b_skill)
        else:
            missing.append(b_skill)

    total_benchmark = len(clean_benchmark)
    if total_benchmark == 0:
        raise ValueError("Target role benchmark must contain at least one valid skill.")

    coverage_score = round((len(covered) / float(total_benchmark)) * 100.0, 1)

    return coverage_score, sorted(covered), sorted(missing)
