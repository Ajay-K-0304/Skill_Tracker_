"""
Phase 7 Unit Tests: Stateful Curriculum Recommendation Agentic Workflow.
"""

import asyncio
from pathlib import Path
import pytest
from services.ai_forecasting.agents.workflow import (
    run_curriculum_workflow,
    run_curriculum_workflow_with_state,
)
from services.ai_forecasting.agents.llm_adapter import LocalLLMAdapter
from services.ai_forecasting.schemas import ContractB, SyllabusUnit


def test_workflow_end_to_end_sample_syllabus():
    async def _run():
        txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        contract_b = await run_curriculum_workflow(
            raw_text=raw_text,
            target_role="Cloud Engineer",
            use_llm=False,
        )

        assert isinstance(contract_b, ContractB)
        assert contract_b.course_name == "Distributed Systems & Cloud Computing"
        assert contract_b.coverage_score > 0.0

        # Covered skills
        assert "Docker" in contract_b.covered_skills
        assert "Linux" in contract_b.covered_skills
        assert "REST APIs" in contract_b.covered_skills

        # Missing skills
        assert "Kubernetes" in contract_b.missing_skills
        assert "gRPC" in contract_b.missing_skills

        # Recommendations
        assert len(contract_b.recommendations) >= 2
        rec_units = [r.unit for r in contract_b.recommendations]
        rec_actions = [r.action for r in contract_b.recommendations]

        # Verify unit targeting
        assert any("Unit 3" in u or "Communication" in u for u in rec_units)
        assert any("gRPC" in a for a in rec_actions)
        assert any("Kubernetes" in a for a in rec_actions)

    asyncio.run(_run())


def test_workflow_exact_contract_b_benchmark():
    async def _run():
        txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 6-skill benchmark where Docker, Linux, REST APIs are covered (3/6 = 50.0%)
        benchmark = ["Docker", "Linux", "REST APIs", "Kubernetes", "gRPC", "FastAPI"]

        contract_b = await run_curriculum_workflow(
            raw_text=raw_text,
            target_role="Custom",
            benchmark_override=benchmark,
            use_llm=False,
        )

        assert contract_b.coverage_score == 50.0
        assert sorted(contract_b.covered_skills) == ["Docker", "Linux", "REST APIs"]
        assert sorted(contract_b.missing_skills) == ["FastAPI", "Kubernetes", "gRPC"]

        # Validate recommendations schema
        for rec in contract_b.recommendations:
            assert rec.priority in ["High", "Medium", "Low"]
            assert len(rec.action) > 10

    asyncio.run(_run())


def test_llm_adapter_offline_graceful_fallback():
    async def _run():
        adapter = LocalLLMAdapter(
            base_url="http://localhost:99999",  # Non-existent endpoint
            timeout=0.5,
            enable_fallback=True,
        )
        units = [
            SyllabusUnit(unit_title="Unit 3: Service Communication", topics=["REST APIs"]),
        ]
        recs, source = await adapter.generate_recommendations(
            course_name="Distributed Systems",
            units=units,
            missing_skills=["gRPC"],
            target_role="Cloud Engineer",
            use_llm=True,  # Request LLM with offline endpoint
        )
        assert source == "deterministic_fallback"
        assert len(recs) == 1
        assert recs[0].unit == "Unit 3: Service Communication"
        assert "gRPC" in recs[0].action

    asyncio.run(_run())


def test_groq_missing_api_key_deterministic_fallback():
    async def _run():
        adapter = LocalLLMAdapter(
            provider="groq",
            api_key="",  # Empty API key
            enable_fallback=True,
        )
        units = [
            SyllabusUnit(unit_title="Unit 2: Containerization and Infrastructure", topics=["Docker"]),
        ]
        recs, source = await adapter.generate_recommendations(
            course_name="Distributed Systems",
            units=units,
            missing_skills=["Kubernetes"],
            target_role="Cloud Engineer",
            use_llm=True,
        )
        # Missing key must trigger deterministic fallback without raising
        assert source == "deterministic_fallback"
        assert len(recs) == 1
        assert "Kubernetes" in recs[0].action

    asyncio.run(_run())


def test_groq_missing_api_key_fallback_disabled_raises():
    async def _run():
        adapter = LocalLLMAdapter(
            provider="groq",
            api_key=None,
            enable_fallback=False,
        )
        units = [SyllabusUnit(unit_title="Unit 1")]
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate_recommendations(
                course_name="Distributed Systems",
                units=units,
                missing_skills=["Kubernetes"],
                use_llm=True,
            )
        assert "fallback is disabled" in str(exc_info.value)

    asyncio.run(_run())


def test_groq_successful_mocked_response():
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _run():
        adapter = LocalLLMAdapter(
            provider="groq",
            api_key="gsk_mock_valid_key",
            model="llama-3.3-70b-versatile",
            enable_fallback=True,
        )
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "recommendations": [
                {
                    "unit": "Unit 2: Containerization and Infrastructure",
                    "action": "Introduce hands-on Kubernetes orchestration labs covering Pods and Deployments.",
                    "priority": "High",
                }
            ]
        })
        mock_completion.choices = [mock_choice]

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_groq_cls.return_value = mock_client

            units = [SyllabusUnit(unit_title="Unit 2: Containerization and Infrastructure")]
            recs, source = await adapter.generate_recommendations(
                course_name="Distributed Systems",
                units=units,
                missing_skills=["Kubernetes"],
                target_role="Cloud Engineer",
                use_llm=True,
            )

            assert source == "groq"
            assert len(recs) == 1
            assert recs[0].unit == "Unit 2: Containerization and Infrastructure"
            assert "Kubernetes orchestration" in recs[0].action
            assert recs[0].priority == "High"

    asyncio.run(_run())


def test_groq_api_failure_deterministic_fallback():
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _run():
        adapter = LocalLLMAdapter(
            provider="groq",
            api_key="gsk_mock_key",
            enable_fallback=True,
        )

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_client = MagicMock()
            # Simulate Groq 401 Unauthorized / connection error
            mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq 401: Invalid API Key"))
            mock_groq_cls.return_value = mock_client

            units = [SyllabusUnit(unit_title="Unit 3: Service Communication")]
            recs, source = await adapter.generate_recommendations(
                course_name="Distributed Systems",
                units=units,
                missing_skills=["gRPC"],
                use_llm=True,
            )

            assert source == "deterministic_fallback"
            assert len(recs) == 1
            assert "gRPC" in recs[0].action

    asyncio.run(_run())


def test_workflow_invokes_groq_when_use_llm_true_mocked():
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _run():
        txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "recommendations": [
                {
                    "unit": "Unit 2: Containerization and Infrastructure",
                    "action": "Deploy microservices on Kubernetes clusters.",
                    "priority": "High",
                },
                {
                    "unit": "Unit 3: Service Communication and Protocols",
                    "action": "Integrate gRPC inter-service RPC pipelines.",
                    "priority": "High",
                },
            ]
        })
        mock_completion.choices = [mock_choice]

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_groq_cls.return_value = mock_client

            with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", "gsk_test_key"):
                contract_b, state = await run_curriculum_workflow_with_state(
                    raw_text=raw_text,
                    target_role="Cloud Engineer",
                    use_llm=True,
                )

                assert isinstance(contract_b, ContractB)
                assert contract_b.course_name == "Distributed Systems & Cloud Computing"
                assert len(contract_b.recommendations) >= 2
                assert state.get("recommendation_source") == "groq"

    asyncio.run(_run())


def test_workflow_groq_failure_records_deterministic_fallback():
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _run():
        txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("Groq 429: Rate Limit Exceeded")
            )
            mock_groq_cls.return_value = mock_client

            with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", "gsk_test_key"):
                contract_b, state = await run_curriculum_workflow_with_state(
                    raw_text=raw_text,
                    target_role="Cloud Engineer",
                    use_llm=True,
                )

                assert isinstance(contract_b, ContractB)
                assert state.get("recommendation_source") == "deterministic_fallback"
                assert len(contract_b.recommendations) > 0

    asyncio.run(_run())


def test_ollama_provider_retained_as_optional():
    async def _run():
        adapter = LocalLLMAdapter(
            provider="ollama",
            base_url="http://localhost:11434",
            timeout=0.1,
            enable_fallback=True,
        )
        assert adapter.provider == "ollama"
        units = [SyllabusUnit(unit_title="Unit 1")]
        # Offline ollama will trigger fallback
        recs, source = await adapter.generate_recommendations(
            course_name="Course",
            units=units,
            missing_skills=["FastAPI"],
            use_llm=True,
        )
        assert source == "deterministic_fallback"
        assert len(recs) == 1
        assert "FastAPI" in recs[0].action

    asyncio.run(_run())


def test_groq_logging_redacts_api_key_and_logs_info_warning(caplog):
    import logging
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _run():
        caplog.set_level(logging.INFO)
        secret_key = "gsk_super_secret_test_key_xyz987"
        adapter = LocalLLMAdapter(
            provider="groq",
            api_key=secret_key,
            enable_fallback=True,
        )

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_client = MagicMock()
            # Simulate Groq error that echoes back the secret key
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError(f"Unauthorized connection using token {secret_key}")
            )
            mock_groq_cls.return_value = mock_client

            units = [SyllabusUnit(unit_title="Unit 3: Security")]
            recs, source = await adapter.generate_recommendations(
                course_name="Security Course",
                units=units,
                missing_skills=["Kubernetes"],
                use_llm=True,
            )

            assert source == "deterministic_fallback"
            assert len(recs) > 0

            # Verify that logs contain the fallback warning and do NOT leak the secret key
            all_logs = caplog.text
            assert "LLM synthesis failed for provider 'groq'" in all_logs
            assert "Applying deterministic fallback recommendation rules" in all_logs
            assert secret_key not in all_logs
            assert "[REDACTED_API_KEY]" in all_logs

    asyncio.run(_run())


