"""
Semantic chunking with hierarchical metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import hashlib
import uuid
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    chunk_id: str  # uuid
    doc_id: str
    content: str
    title: str
    section: str
    chunk_index: int
    source_path: str
    doc_type: str
    char_count: int
    is_complete: bool  # False if content seems truncated mid-sentence
    metadata: Dict[str, Any] = field(default_factory=dict)


def _find_active_section(chunk_text: str, doc_content: str, doc_sections: List[str], search_pos: int) -> str:
    """
    Determines the most relevant section for the chunk.
    1. Looks for any section header explicitly inside the chunk.
    2. Fallback to the latest section header appearing before search_pos in the full document.
    3. Fallback to 'Introduction' or 'General'.
    """
    # Check if a known section header is contained in the first 200 chars of the chunk
    chunk_prefix = chunk_text[:300]
    for sec in doc_sections:
        if sec in chunk_prefix:
            return sec

    # Check for markdown-style headers or numbered sections inside the chunk
    header_match = re.search(r'^(?:#{1,6}\s+|(?:\d+\.|\d+\.\d+)\s+|[A-Z0-9\s_-]{3,}:)\s*([^\n]+)', chunk_text, re.MULTILINE)
    if header_match:
        found_header = header_match.group(1).strip()
        if len(found_header) < 100:
            return found_header

    # Search backwards from search_pos in doc_content for the nearest known section
    if doc_sections and search_pos > 0:
        preceding_text = doc_content[:search_pos]
        best_sec = None
        best_pos = -1
        for sec in doc_sections:
            pos = preceding_text.rfind(sec)
            if pos > best_pos:
                best_pos = pos
                best_sec = sec
        if best_sec:
            return best_sec

    return doc_sections[0] if doc_sections else "General"


def _check_completeness(chunk_text: str) -> bool:
    """Check if chunk ends cleanly with sentence-ending punctuation or formatting."""
    cleaned = chunk_text.strip()
    if not cleaned:
        return False
    # Sentence terminators, closing quotes/brackets, code closing braces
    terminators = ('.', '!', '?', ':', ';', '"', "'", '”', '’', ')', ']', '}', '`')
    return cleaned.endswith(terminators)


def chunk_document(parsed_doc, doc_id: str) -> List[DocumentChunk]:
    """
    Chunk a ParsedDocument into DocumentChunks.
    - Use RecursiveCharacterTextSplitter(chunk_size=768*4, chunk_overlap=100)
    - Detect section from surrounding headers
    - Set is_complete based on whether chunk ends with sentence terminator
    - Generate chunk_id as uuid
    """
    if not parsed_doc or not parsed_doc.content:
        return []

    chunk_size = 768 * 4  # 3072 characters (~768 tokens)
    chunk_overlap = 100

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True
    )

    raw_chunks = splitter.split_text(parsed_doc.content)
    if not raw_chunks:
        return []

    sections = parsed_doc.metadata.get("sections", [])
    chunks: List[DocumentChunk] = []

    last_pos = 0
    total_chunks = len(raw_chunks)

    for idx, chunk_text in enumerate(raw_chunks):
        # Locate chunk in original text to identify context and section
        found_pos = parsed_doc.content.find(chunk_text[:50], last_pos)
        if found_pos != -1:
            search_pos = found_pos
            last_pos = found_pos
        else:
            search_pos = last_pos

        section_name = _find_active_section(
            chunk_text=chunk_text,
            doc_content=parsed_doc.content,
            doc_sections=sections,
            search_pos=search_pos
        )

        is_complete = _check_completeness(chunk_text)
        content_hash = hashlib.md5(chunk_text.strip().encode("utf-8")).hexdigest()

        chunk_meta = {
            **parsed_doc.metadata,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "content_hash": content_hash,
            "source_path": parsed_doc.source_path,
            "doc_type": parsed_doc.doc_type,
            "title": parsed_doc.title,
            "section": section_name,
        }

        chunk_obj = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            content=chunk_text,
            title=parsed_doc.title,
            section=section_name,
            chunk_index=idx,
            source_path=parsed_doc.source_path,
            doc_type=parsed_doc.doc_type,
            char_count=len(chunk_text),
            is_complete=is_complete,
            metadata=chunk_meta
        )
        chunks.append(chunk_obj)

    return chunks
