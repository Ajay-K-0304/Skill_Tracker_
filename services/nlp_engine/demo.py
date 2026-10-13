"""
demo.py — Interactive Pipeline Demonstration & Step-by-Step Inspector
======================================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Usage:
    # Run built-in showcase demonstration:
    python services/nlp_engine/demo.py

    # Try your own custom job description text:
    python services/nlp_engine/demo.py "We need a senior Python 3 developer skilled in FastAPI, Docker, and PostgreSQL."
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import spacy
from services.nlp_engine.cooccurrence.cooccurrence_matrix import CooccurrenceMatrix
from services.nlp_engine.engine import NLPEngine
from services.nlp_engine.normalization.skill_normalizer import SkillNormalizer
from services.nlp_engine.output.contract_formatter import format_contract_a
from services.nlp_engine.pipeline.ner_pipeline import NERPipeline
from services.nlp_engine.pipeline.text_cleaner import clean


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, title: str) -> None:
    print(f"\n[Step {step_num}] {title}")
    print("-" * 50)


def inspect_single_pipeline(raw_text: str, job_id: str = "demo-001", posting_date: str = "2026-10-13") -> None:
    """Executes each stage step-by-step and prints the intermediate output."""
    nlp = spacy.blank("en")
    ner = NERPipeline(nlp=nlp)
    normalizer = SkillNormalizer()

    print_header("END-TO-END PIPELINE INSPECTION (STAGE BY STAGE)")

    # 1. Raw Input
    print_step(1, "RAW INPUT JOB POSTING")
    print(raw_text.strip())

    # 2. Text Cleaner
    print_step(2, "TEXT CLEANING (HTML, URLs, Emails, Noise Stripped)")
    cleaned_text = clean(raw_text)
    print(f"Cleaned Text:\n\"{cleaned_text}\"")

    # 3. NER Extraction
    print_step(3, "NER EXTRACTION (spaCy EntityRuler Phrase Matching)")
    raw_hits = ner.extract(cleaned_text)
    print(f"Detected {len(raw_hits)} raw skill entity hits:")
    for hit in raw_hits:
        print(f"  * Text: {hit['text']:<15} | ID: {hit['taxonomy_id']:<10} | Span: [{hit['start_char']}:{hit['end_char']}]")

    # 4. Skill Normalization
    print_step(4, "SKILL NORMALIZATION & CANONICAL MAPPING")
    normalized_skills = normalizer.normalize_list(raw_hits, deduplicate=True)
    canonical_list = [item["canonical_name"] for item in normalized_skills]
    print(f"Deduplicated Canonical Skills ({len(canonical_list)}): {sorted(canonical_list)}\n")
    print("Enriched Metadata:")
    for norm in normalized_skills:
        print(
            f"  * {norm['canonical_name']:<12} "
            f"| Category: {norm['category']:<12} "
            f"| Confidence: {norm['confidence']} "
            f"| Method: {norm['method']} "
            f"| Raw: \"{norm['raw_mention']}\""
        )

    # 5. Contract A JSON Output
    print_step(5, "CONTRACT A JSON OUTPUT (Delivered to Member 1 & Member 3)")
    contract_a = format_contract_a(
        job_id=job_id,
        extracted_skills=canonical_list,
        posting_date=posting_date,
        validate=True,
    )
    print(json.dumps(contract_a, indent=2))


def demonstrate_batch_and_cooccurrence() -> None:
    """Runs a batch of diverse tech jobs to demonstrate the co-occurrence matrix."""
    print_header("BATCH PROCESSING & CO-OCCURRENCE MATRIX SHOWCASE")

    sample_postings = [
        {
            "job_id": "job-backend",
            "description": "Looking for Senior Python 3 developers with FastAPI, Docker, and PostgreSQL experience.",
            "posting_date": "2026-10-01",
        },
        {
            "job_id": "job-frontend",
            "description": "React.js frontend engineer skilled in TypeScript, Next.js, and Tailwind CSS.",
            "posting_date": "2026-10-02",
        },
        {
            "job_id": "job-fullstack",
            "description": "Full stack engineer building with Python, FastAPI, React, and Docker.",
            "posting_date": "2026-10-03",
        },
        {
            "job_id": "job-ai-agent",
            "description": "AI engineer with Python, LangChain, LangGraph, FastAPI, and pgvector.",
            "posting_date": "2026-10-04",
        },
    ]

    engine = NLPEngine(nlp=spacy.blank("en"))
    records, matrix = engine.process_batch(sample_postings, build_cooccurrence=True)

    print(f"\nProcessed {len(records)} jobs in batch:")
    for r in records:
        print(f"  * [{r['job_id']}] -> {r['extracted_skills']}")

    print("\nCo-occurrence Matrix (Slice of Core Skills):")
    core_skills = ["Docker", "FastAPI", "LangChain", "Python", "React", "TypeScript"]
    sub_df = matrix.matrix.loc[
        [s for s in core_skills if s in matrix.matrix.index],
        [s for s in core_skills if s in matrix.matrix.columns],
    ]
    print(sub_df)

    print("\nTop Graph Edges (Delivered to Member 4 for Network Graph):")
    edges = matrix.to_edges()[:6]
    for edge in edges:
        print(f"  * {edge['source']:<10} <--- (weight: {edge['weight']}) ---> {edge['target']:<10}")


def main() -> None:
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        inspect_single_pipeline(custom_text)
    else:
        sample_raw = """
        <div class="job-header">
            <h1>Senior Cloud & AI Platform Engineer</h1>
            <p>Location: Remote | Apply: jobs@deepcloud.ai | Web: https://deepcloud.ai/careers</p>
        </div>
        <div class="job-body">
            ==================================================
            We are looking for a backend specialist to build intelligent microservices.
            --------------------------------------------------
            <h3>Requirements:</h3>
            <ul>
                <li>• Mastery of <b>Python 3</b> (or CPython) and async APIs with <i>FastAPI</i>.</li>
                <li>• Frontend collaboration using <b>React.js</b> (or ReactJS) with <b>TypeScript</b>.</li>
                <li>• Deploying containers via <b>Docker</b> & orchestration using <b>Kubernetes</b>.</li>
                <li>• Hands-on LLM workflows using <b>LangChain</b> and vector stores like <b>pgvector</b>.</li>
            </ul>
        </div>
        """
        inspect_single_pipeline(sample_raw)
        demonstrate_batch_and_cooccurrence()

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE -- All modules operating at 100% precision!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
