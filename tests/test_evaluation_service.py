import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

EVALUATION_PATH = Path(__file__).resolve().parents[1] / "services" / "evaluation"
sys.path.insert(0, str(EVALUATION_PATH))


def _install_service_import_stubs():
    if importlib.util.find_spec("fastapi") is None:
        fastapi = types.ModuleType("fastapi")

        class FakeFastAPI:
            def __init__(self, **_kwargs):
                pass

            def mount(self, *_args, **_kwargs):
                pass

            def get(self, *_args, **_kwargs):
                return lambda function: function

            def post(self, *_args, **_kwargs):
                return lambda function: function

        class FakeBackgroundTasks:
            def add_task(self, *_args, **_kwargs):
                pass

        fastapi.FastAPI = FakeFastAPI
        fastapi.BackgroundTasks = FakeBackgroundTasks
        fastapi.Query = lambda default=False, **_kwargs: default
        sys.modules["fastapi"] = fastapi

    if importlib.util.find_spec("dotenv") is None:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv

    ragas_eval = types.ModuleType("ragas_eval")
    ragas_eval.run_ragas_evaluation = AsyncMock(return_value={
        "context_precision": 0.9,
        "context_recall": 0.8,
        "faithfulness": 0.95,
        "answer_correctness": 0.88,
        "answer_relevancy": 0.87,
    })
    sys.modules["ragas_eval"] = ragas_eval


_install_service_import_stubs()

import evaluator


class EvaluationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_evaluation_collects_all_five_layers(self):
        class Response:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        class FakeAsyncClient:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def get(self, url):
                if url.endswith("/quality-report"):
                    return Response({"total_documents_checked": 2})
                if url.endswith("/stats"):
                    return Response({"reliability": {"p95_latency_ms": 250}})
                if url.endswith("/feedback"):
                    return Response([{"is_positive": True}, {"is_positive": False}])
                if url.endswith("/messages"):
                    return Response([
                        {"query": "Where is the policy?", "response": "See the policy."},
                        {"query": "Need assistance?", "response": "Please contact HR."},
                    ])
                raise AssertionError(f"Unexpected service request: {url}")

        original_client = evaluator.httpx.AsyncClient
        evaluator.httpx.AsyncClient = FakeAsyncClient
        try:
            result = await evaluator.run_full_evaluation()
        finally:
            evaluator.httpx.AsyncClient = original_client

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            set(result["layers"]),
            {
                "layer1_data_quality",
                "layer2_retrieval",
                "layer3_generation",
                "layer4_reliability",
                "layer5_business",
            },
        )
        self.assertEqual(result["layers"]["layer1_data_quality"]["total_documents_checked"], 2)
        self.assertEqual(result["layers"]["layer4_reliability"]["p95_latency_ms"], 250)
        self.assertEqual(result["layers"]["layer5_business"]["csat"]["percent"], 50.0)
        self.assertEqual(result["layers"]["layer5_business"]["escalation_rate"], 0.5)
        self.assertFalse(evaluator.eval_in_progress)


if __name__ == "__main__":
    unittest.main()
