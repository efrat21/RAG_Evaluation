import asyncio
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

INGESTION_PATH = Path(__file__).resolve().parents[1] / "services" / "ingestion"
sys.path.insert(0, str(INGESTION_PATH))


def _install_optional_dependency_stubs():
    if importlib.util.find_spec("fitz") is None:
        sys.modules["fitz"] = types.ModuleType("fitz")
    if importlib.util.find_spec("docx") is None:
        docx = types.ModuleType("docx")
        docx.Document = lambda _path: None
        sys.modules["docx"] = docx

    if importlib.util.find_spec("langchain") is None:
        langchain = types.ModuleType("langchain")
        text_splitter = types.ModuleType("langchain.text_splitter")

        class SingleChunkSplitter:
            def __init__(self, **_kwargs):
                pass

            def split_text(self, text):
                return [text] if text else []

        text_splitter.RecursiveCharacterTextSplitter = SingleChunkSplitter
        langchain.text_splitter = text_splitter
        sys.modules["langchain"] = langchain
        sys.modules["langchain.text_splitter"] = text_splitter

    if importlib.util.find_spec("langchain_ollama") is None:
        ollama = types.ModuleType("langchain_ollama")

        class MissingChatOllama:
            def __init__(self, **_kwargs):
                raise AssertionError("Tests must replace the Ollama model client")

        ollama.ChatOllama = MissingChatOllama
        sys.modules["langchain_ollama"] = ollama


_install_optional_dependency_stubs()

import chunker
import golden_dataset
import parsers


class DocumentParserTests(unittest.TestCase):
    def test_markdown_parser_reads_frontmatter_and_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.md"
            path.write_text(
                '---\ntitle: "Benefits Policy"\nauthor: HR\n---\n'
                "# Leave\nEmployees receive paid leave.\n",
                encoding="utf-8",
            )

            parsed = parsers.parse_document(str(path))

        self.assertEqual(parsed.doc_type, "markdown")
        self.assertEqual(parsed.title, "Benefits Policy")
        self.assertEqual(parsed.metadata["author"], "HR")
        self.assertEqual(parsed.metadata["sections"], ["Leave"])
        self.assertIn("Employees receive paid leave.", parsed.content)
        self.assertFalse(parsed.parse_errors)

    def test_html_parser_removes_navigation_and_scripts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "help.html"
            path.write_text(
                "<html><head><title>Help</title><script>secret()</script></head>"
                "<body><nav>Navigation</nav><h1>Support</h1>"
                "<p>Open a helpdesk ticket.</p></body></html>",
                encoding="utf-8",
            )

            parsed = parsers.parse_document(str(path))

        self.assertEqual(parsed.title, "Help")
        self.assertIn("Support", parsed.content)
        self.assertNotIn("Navigation", parsed.content)
        self.assertNotIn("secret()", parsed.content)
        self.assertEqual(parsed.metadata["sections"], ["Support"])

    def test_missing_file_returns_parse_error_instead_of_raising(self):
        parsed = parsers.parse_document("file-that-does-not-exist.txt")

        self.assertEqual(parsed.doc_type, "unknown")
        self.assertEqual(parsed.content, "")
        self.assertTrue(parsed.parse_errors)


class ChunkingTests(unittest.TestCase):
    def test_empty_document_returns_no_chunks(self):
        document = SimpleNamespace(content="", metadata={}, source_path="policy.txt")

        self.assertEqual(chunker.chunk_document(document, "doc-1"), [])

    def test_chunk_carries_document_and_section_metadata(self):
        document = SimpleNamespace(
            content="# Benefits\nEmployees receive paid leave.",
            metadata={"sections": ["Benefits"], "author": "HR"},
            source_path="benefits.md",
            title="Benefits",
            doc_type="markdown",
        )

        chunks = chunker.chunk_document(document, "doc-1")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "doc-1")
        self.assertEqual(chunks[0].section, "Benefits")
        self.assertEqual(chunks[0].metadata["author"], "HR")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].char_count, len(chunks[0].content))


class GoldenDatasetTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_writes_valid_jsonl_with_source_context(self):
        response = SimpleNamespace(
            content='```json\n[{"question": "When is leave available?", '
            '"answer": "Leave is available after onboarding."}]\n```'
        )
        fake_llm = SimpleNamespace(ainvoke=unittest.mock.AsyncMock(return_value=response))
        chunk = SimpleNamespace(
            content="Employees may request leave after completing onboarding.",
            title="Benefits",
            doc_id="hr-1",
            chunk_id="chunk-1",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "golden" / "golden.jsonl"
            with patch.object(golden_dataset, "ChatOllama", return_value=fake_llm):
                records = await golden_dataset.generate_golden_dataset(
                    chunks=[chunk],
                    llm_model="test-model",
                    ollama_url="http://unused",
                    output_path=str(output_path),
                    min_chunk_length=1,
                )

            saved_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records, saved_records)
        self.assertEqual(records[0]["source_doc_id"], "hr-1")
        self.assertEqual(records[0]["ground_truth_context"], chunk.content)

    async def test_no_chunks_returns_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            records = await golden_dataset.generate_golden_dataset(
                chunks=[],
                llm_model="unused",
                ollama_url="http://unused",
                output_path=str(Path(temp_dir) / "golden.jsonl"),
            )

        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
