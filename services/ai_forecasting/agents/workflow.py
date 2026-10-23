"""
Stateful AI Curriculum Recommendation Workflow.
Built using LangGraph state machines, coordinating input validation, syllabus parsing,
target-role benchmark retrieval, deterministic skill-gap calculation, and recommendation synthesis.
Guarantees schema-validated Contract B output.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langgraph.graph import StateGraph, END

from services.ai_forecasting.agents.llm_adapter import LocalLLMAdapter
from services.ai_forecasting.curriculum.benchmarks import get_target_role_benchmark
from services.ai_forecasting.curriculum.gap_analyzer import analyze_skill_gap
from services.ai_forecasting.schemas import ContractB, ParsedSyllabus, RecommendationItem
from services.ai_forecasting.syllabus.parser import parse_syllabus_text

logger = logging.getLogger(__name__)


class CurriculumWorkflowState(TypedDict):
    """Explicit workflow state container for the LangGraph pipeline."""
    raw_text: str
    target_role: str
    benchmark_override: Optional[List[str]]
    use_llm: bool
    parsed_syllabus: Optional[ParsedSyllabus]
    benchmark_skills: List[str]
    benchmark_source: str
    coverage_score: float
    covered_skills: List[str]
    missing_skills: List[str]
    recommendations: List[RecommendationItem]
    recommendation_source: str
    contract_b_output: Optional[ContractB]
    errors: List[str]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def validate_input_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    raw_text = state.get("raw_text", "").strip()
    if not raw_text:
        return {"errors": state.get("errors", []) + ["Syllabus input text is empty."]}
    return {"errors": []}


def extract_syllabus_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    raw_text = state.get("raw_text", "")
    parsed = parse_syllabus_text(raw_text)
    return {"parsed_syllabus": parsed}


def retrieve_benchmark_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    role = state.get("target_role", "Cloud Engineer")
    override = state.get("benchmark_override")
    skills, source = get_target_role_benchmark(role, override=override)
    return {"benchmark_skills": skills, "benchmark_source": source}


def analyze_gap_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    parsed = state.get("parsed_syllabus")
    syllabus_skills = parsed.extracted_skills if parsed else []
    benchmark_skills = state.get("benchmark_skills", [])

    try:
        score, covered, missing = analyze_skill_gap(syllabus_skills, benchmark_skills)
        return {
            "coverage_score": score,
            "covered_skills": covered,
            "missing_skills": missing,
        }
    except Exception as e:
        return {
            "coverage_score": 0.0,
            "covered_skills": [],
            "missing_skills": benchmark_skills,
            "errors": state.get("errors", []) + [str(e)],
        }


async def generate_recommendations_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    parsed = state.get("parsed_syllabus")
    course_name = parsed.course_name if parsed else "Course"
    units = parsed.units if parsed else []
    missing_skills = state.get("missing_skills", [])
    role = state.get("target_role", "Cloud Engineer")
    use_llm = state.get("use_llm", False)

    adapter = LocalLLMAdapter()
    recs, source = await adapter.generate_recommendations(
        course_name=course_name,
        units=units,
        missing_skills=missing_skills,
        target_role=role,
        use_llm=use_llm,
    )
    logger.info(
        f"Workflow node 'generate_recommendations': synthesized {len(recs)} recommendations "
        f"(source='{source}', missing_skills={len(missing_skills)})"
    )
    return {"recommendations": recs, "recommendation_source": source}


def validate_and_finalize_node(state: CurriculumWorkflowState) -> Dict[str, Any]:
    parsed = state.get("parsed_syllabus")
    course_name = parsed.course_name if parsed else "Untitled Course"
    coverage_score = state.get("coverage_score", 0.0)
    covered_skills = state.get("covered_skills", [])
    missing_skills = state.get("missing_skills", [])
    recommendations = state.get("recommendations", [])

    contract_b = ContractB(
        course_name=course_name,
        coverage_score=coverage_score,
        covered_skills=covered_skills,
        missing_skills=missing_skills,
        recommendations=recommendations,
    )
    logger.info(
        f"Workflow node 'validate_and_finalize': generated Contract B for '{course_name}' "
        f"(score={coverage_score}%, recs={len(recommendations)})"
    )

    return {"contract_b_output": contract_b}


# ---------------------------------------------------------------------------
# Workflow Construction
# ---------------------------------------------------------------------------

def build_curriculum_graph():
    workflow = StateGraph(CurriculumWorkflowState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("extract_syllabus", extract_syllabus_node)
    workflow.add_node("retrieve_benchmark", retrieve_benchmark_node)
    workflow.add_node("analyze_gap", analyze_gap_node)
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    workflow.add_node("validate_and_finalize", validate_and_finalize_node)

    workflow.set_entry_point("validate_input")
    workflow.add_edge("validate_input", "extract_syllabus")
    workflow.add_edge("extract_syllabus", "retrieve_benchmark")
    workflow.add_edge("retrieve_benchmark", "analyze_gap")
    workflow.add_edge("analyze_gap", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "validate_and_finalize")
    workflow.add_edge("validate_and_finalize", END)

    return workflow.compile()


# Module-level compiled graph runner
_compiled_app = build_curriculum_graph()


async def run_curriculum_workflow_with_state(
    raw_text: str,
    target_role: str = "Cloud Engineer",
    benchmark_override: Optional[List[str]] = None,
    use_llm: bool = False,
) -> Tuple[ContractB, CurriculumWorkflowState]:
    """
    Executes the end-to-end curriculum recommendation workflow and returns both
    the Contract B output and the final workflow state container.
    """
    initial_state: CurriculumWorkflowState = {
        "raw_text": raw_text,
        "target_role": target_role,
        "benchmark_override": benchmark_override,
        "use_llm": use_llm,
        "parsed_syllabus": None,
        "benchmark_skills": [],
        "benchmark_source": "",
        "coverage_score": 0.0,
        "covered_skills": [],
        "missing_skills": [],
        "recommendations": [],
        "recommendation_source": "",
        "contract_b_output": None,
        "errors": [],
    }

    final_state = await _compiled_app.ainvoke(initial_state)

    if final_state.get("contract_b_output"):
        return final_state["contract_b_output"], final_state

    # Fallback safety in case of graph fault
    fallback_contract = ContractB(
        course_name="Untitled Course",
        coverage_score=0.0,
        covered_skills=[],
        missing_skills=[],
        recommendations=[],
    )
    return fallback_contract, final_state


async def run_curriculum_workflow(
    raw_text: str,
    target_role: str = "Cloud Engineer",
    benchmark_override: Optional[List[str]] = None,
    use_llm: bool = False,
) -> ContractB:
    """
    Executes the end-to-end curriculum recommendation workflow.
    Guarantees Contract B response format.
    """
    contract_b, _ = await run_curriculum_workflow_with_state(
        raw_text=raw_text,
        target_role=target_role,
        benchmark_override=benchmark_override,
        use_llm=use_llm,
    )
    return contract_b
