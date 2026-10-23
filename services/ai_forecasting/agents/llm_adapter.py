"""
Local LLM Adapter with resilient deterministic fallback.
Interfaces with local model runtimes (Ollama, vLLM, LMStudio) via HTTP,
and guarantees valid Contract B recommendation outputs even when the local model is offline.
"""

import json
import logging
import re
from typing import Any, List, Optional, Tuple
import httpx

from services.ai_forecasting.config import settings
from services.ai_forecasting.schemas import PriorityType, RecommendationItem, SyllabusUnit

logger = logging.getLogger(__name__)


# Knowledge map for deterministic curriculum actions and target unit matching
SKILL_ACTION_KNOWLEDGE_BASE = {
    "kubernetes": {
        "unit_keywords": ["container", "infrastructure", "deployment", "virtualization", "devops", "cloud"],
        "action": "Add a hands-on Kubernetes orchestration module covering Pods, Deployments, and Service networking alongside Docker.",
        "default_priority": "High",
    },
    "grpc": {
        "unit_keywords": ["communication", "protocol", "api", "service", "networking", "distributed"],
        "action": "Introduce gRPC and Protocol Buffers alongside REST APIs for high-performance inter-service communication.",
        "default_priority": "High",
    },
    "fastapi": {
        "unit_keywords": ["api", "web", "backend", "service", "python", "communication"],
        "action": "Incorporate modern asynchronous API development using FastAPI and Pydantic validation.",
        "default_priority": "Medium",
    },
    "aws": {
        "unit_keywords": ["cloud", "infrastructure", "deployment", "virtualization", "system"],
        "action": "Integrate AWS core cloud infrastructure concepts (EC2, S3, IAM) with hands-on lab deployments.",
        "default_priority": "Medium",
    },
    "react": {
        "unit_keywords": ["frontend", "web", "ui", "interface", "client"],
        "action": "Incorporate component-based frontend architecture and state management using React.",
        "default_priority": "High",
    },
    "tailwind": {
        "unit_keywords": ["frontend", "ui", "css", "styling", "web"],
        "action": "Adopt utility-first responsive styling workflows with Tailwind CSS.",
        "default_priority": "Low",
    },
    "kafka": {
        "unit_keywords": ["communication", "data", "streaming", "message", "queue", "distributed"],
        "action": "Introduce Apache Kafka event-driven architecture and distributed pub/sub pipelines.",
        "default_priority": "High",
    },
    "sql": {
        "unit_keywords": ["database", "data", "storage", "backend", "persistence"],
        "action": "Enhance relational database schema design, indexing, and complex query optimization with SQL.",
        "default_priority": "High",
    },
}


class LocalLLMAdapter:
    """Configurable adapter for cloud (Groq) and local (Ollama) LLM inference with deterministic fallback."""

    _UNSET = object()

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        enable_fallback: Optional[bool] = None,
        api_key: Any = _UNSET,
    ):
        self.provider = (provider or settings.LOCAL_LLM_PROVIDER or "groq").lower().strip()
        self.base_url = (base_url or settings.LOCAL_LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LOCAL_LLM_MODEL
        self.timeout = timeout if timeout is not None else settings.LOCAL_LLM_TIMEOUT
        self.enable_fallback = enable_fallback if enable_fallback is not None else settings.ENABLE_LLM_FALLBACK

        # Safely extract and strip API key without ever storing raw untrimmed whitespace
        if api_key is not self._UNSET:
            self.api_key = api_key.strip() if (api_key and isinstance(api_key, str) and api_key.strip()) else None
        else:
            raw_key = settings.GROQ_API_KEY
            self.api_key = raw_key.strip() if (raw_key and raw_key.strip()) else None

    def _match_best_unit(self, skill: str, units: List[SyllabusUnit]) -> str:
        """Finds the existing syllabus unit most relevant to the given missing skill."""
        if not units:
            return "General Curriculum Updates"

        skill_lower = skill.lower()
        rule = SKILL_ACTION_KNOWLEDGE_BASE.get(skill_lower)
        keywords = rule["unit_keywords"] if rule else [skill_lower]

        best_unit = units[-1].unit_title  # fallback to last unit
        best_score = 0

        for unit in units:
            text_to_search = f"{unit.unit_title} {' '.join(unit.topics)}".lower()
            score = sum(1 for kw in keywords if kw in text_to_search)
            if score > best_score:
                best_score = score
                best_unit = unit.unit_title

        return best_unit

    def _generate_deterministic_recommendations(
        self,
        missing_skills: List[str],
        units: List[SyllabusUnit],
    ) -> List[RecommendationItem]:
        """Synthesizes deterministic recommendations for missing skills."""
        recommendations: List[RecommendationItem] = []

        for idx, skill in enumerate(missing_skills):
            skill_lower = skill.lower()
            best_unit = self._match_best_unit(skill, units)

            if skill_lower in SKILL_ACTION_KNOWLEDGE_BASE:
                entry = SKILL_ACTION_KNOWLEDGE_BASE[skill_lower]
                action = entry["action"]
                priority: PriorityType = entry["default_priority"]
            else:
                action = f"Introduce practical modules and laboratory exercises covering {skill} industry patterns."
                priority = "High" if idx == 0 else "Medium"

            recommendations.append(
                RecommendationItem(
                    unit=best_unit,
                    action=action,
                    priority=priority,
                )
            )

        return recommendations

    async def _invoke_groq(
        self,
        course_name: str,
        units: List[SyllabusUnit],
        missing_skills: List[str],
        target_role: str,
    ) -> List[RecommendationItem]:
        """Invokes Groq API via official AsyncGroq client."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured or is empty.")

        unit_titles = [u.unit_title for u in units]
        prompt = (
            f"You are an expert academic curriculum reviewer evaluating the course '{course_name}'.\n"
            f"Target industry role: '{target_role}'.\n"
            f"Existing course units: {unit_titles}.\n"
            f"Missing industry skills to introduce: {missing_skills}.\n\n"
            "Respond ONLY with a JSON object containing a 'recommendations' list of objects. Each object must have:\n"
            "- 'unit': exactly which existing syllabus unit title to modify (or 'General Course Enhancement')\n"
            "- 'action': concrete modification or lab to introduce\n"
            "- 'priority': exactly 'High', 'Medium', or 'Low'\n"
            "Example format: {\"recommendations\": [{\"unit\": \"Unit 1\", \"action\": \"...\", \"priority\": \"High\"}]}"
        )

        from groq import AsyncGroq

        client = AsyncGroq(api_key=self.api_key, timeout=self.timeout)
        chat_completion = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert curriculum design assistant. You strictly return JSON objects matching the user schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw_content = chat_completion.choices[0].message.content or ""
        parsed_json = json.loads(raw_content)

        items_list = parsed_json.get("recommendations", parsed_json) if isinstance(parsed_json, dict) else parsed_json
        if not isinstance(items_list, list):
            raise ValueError(f"Unexpected JSON structure from Groq: {type(parsed_json)}")

        validated_items: List[RecommendationItem] = []
        default_unit = units[0].unit_title if units else "General Course Enhancement"
        for item in items_list:
            if isinstance(item, dict) and "action" in item:
                priority = item.get("priority", "Medium")
                if priority not in ("High", "Medium", "Low"):
                    priority = "Medium"
                validated_items.append(
                    RecommendationItem(
                        unit=item.get("unit") or default_unit,
                        action=item["action"],
                        priority=priority,
                    )
                )

        if not validated_items:
            raise ValueError("Groq returned empty recommendations list.")

        return validated_items

    async def _invoke_ollama(
        self,
        course_name: str,
        units: List[SyllabusUnit],
        missing_skills: List[str],
        target_role: str,
    ) -> List[RecommendationItem]:
        """Invokes local Ollama daemon via HTTP."""
        unit_titles = [u.unit_title for u in units]
        prompt = (
            f"You are an academic curriculum reviewer. Course: '{course_name}'. "
            f"Target industry role: '{target_role}'. Existing units: {unit_titles}. "
            f"Missing industry skills to introduce: {missing_skills}. "
            "Respond ONLY with a JSON array of objects. Each object must contain:\n"
            "- 'unit': exactly which existing syllabus unit title to modify\n"
            "- 'action': concrete update description\n"
            "- 'priority': 'High', 'Medium', or 'Low'\n"
            "Do not include explanation text."
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_output = data.get("response", "")
            parsed_json = json.loads(raw_output)

            items_list = parsed_json.get("recommendations", parsed_json) if isinstance(parsed_json, dict) else parsed_json
            if not isinstance(items_list, list):
                raise ValueError("Ollama did not return a JSON array.")

            validated_items = [
                RecommendationItem(
                    unit=item.get("unit", units[0].unit_title if units else "General"),
                    action=item["action"],
                    priority=item.get("priority", "Medium") if item.get("priority") in ("High", "Medium", "Low") else "Medium",
                )
                for item in items_list
                if isinstance(item, dict) and "action" in item
            ]
            if not validated_items:
                raise ValueError("Ollama returned empty recommendations.")
            return validated_items

    async def generate_recommendations(
        self,
        course_name: str,
        units: List[SyllabusUnit],
        missing_skills: List[str],
        target_role: str = "Cloud Engineer",
        use_llm: bool = False,
    ) -> Tuple[List[RecommendationItem], str]:
        """
        Generates curriculum recommendations for missing skills.
        Returns:
            Tuple of (List[RecommendationItem], source_label)
            source_label is 'groq', 'ollama', or 'deterministic_fallback'
        """
        if not missing_skills:
            return [], "no_gaps"

        if not use_llm:
            logger.info("Generating deterministic curriculum recommendations (use_llm=False).")
            return (
                self._generate_deterministic_recommendations(missing_skills, units),
                "deterministic_fallback",
            )

        # Log provider invocation without exposing secrets
        logger.info(
            f"Initiating LLM curriculum recommendation synthesis (provider: '{self.provider}', model: '{self.model}')"
        )

        try:
            if self.provider == "groq":
                items = await self._invoke_groq(course_name, units, missing_skills, target_role)
                logger.info(f"Successfully synthesized {len(items)} recommendations via Groq API (model: '{self.model}').")
                return items, "groq"
            elif self.provider == "ollama":
                items = await self._invoke_ollama(course_name, units, missing_skills, target_role)
                logger.info(f"Successfully synthesized {len(items)} recommendations via Ollama (model: '{self.model}').")
                return items, "ollama"
            elif self.provider == "mock":
                items = self._generate_deterministic_recommendations(missing_skills, units)
                return items, "mock_llm"
            else:
                logger.warning(f"Unrecognized LLM provider '{self.provider}'. Falling back to deterministic generation.")
        except Exception as e:
            # Mask any potential sensitive details in error message
            import re
            err_msg = str(e)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
            err_msg = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", err_msg)
            logger.warning(
                f"LLM synthesis failed for provider '{self.provider}' ({type(e).__name__}: {err_msg[:150]}). "
                f"Fallback policy enabled: {self.enable_fallback}."
            )

        if self.enable_fallback:
            logger.info("Applying deterministic fallback recommendation rules (source='deterministic_fallback').")
            return (
                self._generate_deterministic_recommendations(missing_skills, units),
                "deterministic_fallback",
            )

        raise RuntimeError(f"LLM synthesis failed for provider '{self.provider}' and fallback is disabled.")
