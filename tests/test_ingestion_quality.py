import unittest
from unittest.mock import patch
from types import SimpleNamespace

from services.ingestion.quality import (
    check_chunk_quality,
    check_document_freshness,
    check_duplicate,
    check_index_coverage,
)


class ChunkQualityTests(unittest.TestCase):
    def test_complete_text_over_minimum_length_has_full_score(self):
        result = check_chunk_quality({
            "content": ("Useful policy information " * 5).strip() + ".",
            "is_complete": True,
        })

        self.assertEqual(result, {"score": 1.0, "issues": []})

    def test_empty_and_incomplete_short_chunks_are_flagged(self):
        empty_result = check_chunk_quality({"content": "   "})
        short_result = check_chunk_quality({
            "content": "Short text",
            "is_complete": False,
        })

        self.assertEqual(empty_result, {"score": 0.0, "issues": ["Chunk is empty"]})
        self.assertEqual(short_result["score"], 0.5)
        self.assertEqual(len(short_result["issues"]), 2)

    def test_duplicate_detection_normalizes_case_and_whitespace(self):
        hashes = set()

        self.assertFalse(check_duplicate("Company policy text", hashes))
        self.assertTrue(check_duplicate("  COMPANY   policy\ntext  ", hashes))
        self.assertEqual(len(hashes), 1)

    def test_missing_document_path_is_reported_as_fresh(self):
        result = check_document_freshness("missing-document-for-quality-test.txt")

        self.assertTrue(result["is_fresh"])
        self.assertEqual(result["days_old"], 0.0)
        self.assertIn("modified_at", result)


class IndexCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_coverage_counts_retrievable_document_ids(self):
        class FakeQdrantClient:
            def scroll(self, collection_name, scroll_filter, limit):
                doc_id = scroll_filter["must"][0]["match"]["value"]
                points = [object()] if doc_id == "indexed" else []
                return points, None

        with patch(
            "services.ingestion.quality.random.sample",
            return_value=["indexed", "missing"],
        ):
            coverage = await check_index_coverage(
                FakeQdrantClient(),
                "documents",
                ["indexed", "missing"],
            )

        self.assertEqual(coverage, 0.5)

    async def test_empty_index_check_is_treated_as_full_coverage(self):
        self.assertEqual(await check_index_coverage(None, "documents", []), 1.0)


if __name__ == "__main__":
    unittest.main()
