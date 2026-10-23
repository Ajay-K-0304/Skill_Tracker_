"""
Target role skill benchmarks.
Provides clearly labelled mock benchmark fallbacks for demo and testing,
while maintaining a clean interface to connect to Member 1's DB or Member 2's ontology later.
"""

from typing import Dict, List, Optional, Tuple

ROLE_BENCHMARKS: Dict[str, List[str]] = {
    "Cloud Engineer": [
        "Docker",
        "Linux",
        "REST APIs",
        "Kubernetes",
        "gRPC",
        "FastAPI",
        "AWS",
    ],
    "Distributed Systems Engineer": [
        "Docker",
        "Linux",
        "REST APIs",
        "Kubernetes",
        "gRPC",
        "FastAPI",
    ],
    "Full Stack Developer": [
        "Python",
        "FastAPI",
        "React",
        "Tailwind",
        "SQL",
        "Git",
    ],
    "Backend Developer": [
        "Python",
        "FastAPI",
        "Docker",
        "SQL",
        "REST APIs",
        "Git",
    ],
    "Data Engineer": [
        "Python",
        "SQL",
        "Docker",
        "Kafka",
        "Linux",
    ],
}


def get_target_role_benchmark(
    role_name: str,
    override: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Retrieves required skills for a target role.
    Returns:
        Tuple of (skill_list, source_label)
    """
    if override is not None:
        return override, "user_override"

    # Match normalized key
    for role_key, skills in ROLE_BENCHMARKS.items():
        if role_key.lower() == role_name.strip().lower():
            return skills, "mock_role_benchmark"

    # Fallback default if unknown role
    return ROLE_BENCHMARKS["Cloud Engineer"], "mock_role_benchmark_fallback"
