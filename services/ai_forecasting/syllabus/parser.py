"""
Deterministic syllabus text parser.
Extracts course titles, units, topics, learning objectives, and candidate technical skills.
"""

import re
from typing import Dict, List, Optional, Set
from services.ai_forecasting.schemas import ParsedSyllabus, SyllabusUnit

# Standard canonical skills dictionary with surface alias patterns
CANONICAL_SKILL_PATTERNS: Dict[str, List[str]] = {
    "Docker": [r"\bdocker\b", r"\bcontainerization\b", r"\bcontainers\b"],
    "Linux": [r"\blinux\b", r"\bunix\b", r"\bbash\b", r"\bshell scripting\b"],
    "REST APIs": [r"\brest\s*apis?\b", r"\brestful\b", r"\bhttp\s*apis?\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "gRPC": [r"\bgrpc\b", r"\bprotocol buffers?\b", r"\bprotobuf\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Python": [r"\bpython(?:3)?\b"],
    "React": [r"\breact(?:\.js)?\b", r"\breactjs\b"],
    "Tailwind": [r"\btailwind(?:\s*css)?\b"],
    "SQL": [r"\bsql\b", r"\bpostgresql\b", r"\bmysql\b", r"\brelational databases?\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bversion control\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b", r"\bcloud infrastructure\b"],
    "Microservices": [r"\bmicroservices?\b", r"\bservice-oriented architecture\b"],
    "CI/CD": [r"\bci[/-]cd\b", r"\bcontinuous integration\b"],
    "GraphQL": [r"\bgraphql\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "Redis": [r"\bredis\b"],
    "Kafka": [r"\bkafka\b", r"\bevent streaming\b"],
}


def extract_skills_from_text(text: str) -> List[str]:
    """Identifies canonical skills mentioned in text using regex boundary matching."""
    text_lower = text.lower()
    matched_skills: Set[str] = set()

    for canonical, patterns in CANONICAL_SKILL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                matched_skills.add(canonical)
                break

    return sorted(matched_skills)


def parse_syllabus_text(raw_text: str) -> ParsedSyllabus:
    """
    Parses unstructured syllabus text into structured units, topics, and skills.
    """
    clean_text = raw_text.strip()
    if not clean_text:
        return ParsedSyllabus(
            course_name="Untitled Course",
            units=[],
            extracted_skills=[],
            raw_text_length=0,
            extraction_notes="Empty text input provided.",
        )

    lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]

    # 1. Identify course name
    course_name = "Untitled Course"
    notes = []

    course_match = re.search(
        r"(?i)(?:course(?:\s+name|\s+title)?|subject)\s*[:\-]\s*([^\n\r]+)", clean_text
    )
    if course_match:
        course_name = course_match.group(1).strip()
    else:
        # Check first line if it looks like a title
        if lines and len(lines[0]) < 80 and not re.match(r"(?i)^(unit|module|chapter)", lines[0]):
            course_name = lines[0]
        else:
            notes.append("Course name header not explicitly identified; defaulted to Untitled Course.")

    # 2. Extract units/modules
    unit_pattern = re.compile(
        r"(?i)(?:^|\n)\s*(unit|module|chapter)\s*([0-9ivx]+)[:\s\-\–]*(.*)",
        re.MULTILINE,
    )

    matches = list(unit_pattern.finditer(clean_text))
    units: List[SyllabusUnit] = []

    if matches:
        for i, match in enumerate(matches):
            unit_kind = match.group(1).capitalize()
            unit_num = match.group(2).upper()
            unit_title_tail = match.group(3).strip()

            if unit_title_tail:
                unit_title = f"{unit_kind} {unit_num}: {unit_title_tail}"
            else:
                unit_title = f"{unit_kind} {unit_num}"

            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
            unit_body = clean_text[start_pos:end_pos].strip()

            # Extract topics and learning objectives within unit body
            topics = []
            objectives = []

            for line in unit_body.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Check for objective
                if re.search(r"(?i)(objective|outcome|learn|understand|ability to)", line):
                    clean_obj = re.sub(r"^[-*•\d\.\)]+\s*", "", line).strip()
                    objectives.append(clean_obj)
                else:
                    # Clean bullet markers
                    clean_topic = re.sub(r"^[-*•\d\.\)]+\s*", "", line).strip()
                    if clean_topic:
                        # Split by commas or semicolons if multiple topics on one line
                        sub_parts = [p.strip() for p in re.split(r"[,;]", clean_topic) if len(p.strip()) > 2]
                        if len(sub_parts) > 1:
                            topics.extend(sub_parts)
                        else:
                            topics.append(clean_topic)

            units.append(
                SyllabusUnit(
                    unit_title=unit_title,
                    topics=topics,
                    learning_objectives=objectives,
                )
            )
    else:
        # No explicit units found - treat entire content as single module
        notes.append("No explicit Unit/Module delimiters found in document.")
        units.append(
            SyllabusUnit(
                unit_title="General Course Content",
                topics=[ln for ln in lines[1:10] if len(ln) > 3],
                learning_objectives=[],
            )
        )

    # 3. Extract candidate skills
    all_extracted_skills = extract_skills_from_text(clean_text)

    return ParsedSyllabus(
        course_name=course_name,
        units=units,
        extracted_skills=all_extracted_skills,
        raw_text_length=len(clean_text),
        extraction_notes="; ".join(notes) if notes else None,
    )
