"""
Document parsers for the ingestion pipeline.
Supports: PDF, DOCX, Markdown, plain text, HTML, code files.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import re
import unicodedata
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup


@dataclass
class ParsedDocument:
    content: str
    title: str
    source_path: str
    doc_type: str  # pdf, docx, markdown, html, code, text
    page_count: Optional[int]
    metadata: Dict[str, Any]  # section headers, creation date, author, etc.
    parse_quality_score: float  # 0-1, based on content density and structure
    parse_errors: List[str] = field(default_factory=list)


def _compute_quality_score(
    content: str,
    page_count: Optional[int] = None,
    doc_type: str = "text"
) -> float:
    """
    Computes a parse quality score from 0.0 to 1.0.
    Flags:
    - Low score if garbled characters > 5%
    - Low score if empty / whitespace lines > 20%
    - Low score if very long lines (> 500 chars) > 50%
    - For PDF, factors in character density (char_count / page_count / 500)
    """
    if not content or not content.strip():
        return 0.0

    total_chars = len(content)
    lines = content.splitlines()
    total_lines = max(1, len(lines))

    # 1. Garbled characters ratio
    # Characters that are non-printable and not standard whitespace (tab, newline, carriage return)
    # or Unicode replacement character \ufffd
    garbled_count = 0
    for ch in content:
        if ch == '\ufffd':
            garbled_count += 1
        elif ord(ch) < 32 and ch not in ('\t', '\n', '\r'):
            garbled_count += 1
        elif unicodedata.category(ch) in ('Cs', 'Co', 'Cn'):  # Surrogate, Private Use, Unassigned
            garbled_count += 1

    garbled_ratio = garbled_count / max(1, total_chars)

    # 2. Empty or whitespace-only lines ratio
    empty_lines = sum(1 for line in lines if not line.strip())
    empty_line_ratio = empty_lines / total_lines

    # 3. Very long lines (> 500 chars without punctuation/breaks)
    very_long_lines = sum(1 for line in lines if len(line) > 500)
    long_line_ratio = very_long_lines / total_lines

    # Base quality score
    if doc_type == "pdf" and page_count and page_count > 0:
        # Expected density around 500 chars per page
        density = (total_chars / page_count) / 500.0
        score = min(1.0, max(0.1, density))
    else:
        score = 1.0

    # Penalties
    if garbled_ratio > 0.05:
        score *= max(0.1, 1.0 - (garbled_ratio * 5.0))
    if empty_line_ratio > 0.20:
        score *= max(0.2, 1.0 - (empty_line_ratio - 0.20) * 1.5)
    if long_line_ratio > 0.50:
        score *= max(0.2, 1.0 - (long_line_ratio - 0.50) * 1.5)

    if total_chars < 50:
        score *= 0.5

    return round(float(min(1.0, max(0.0, score))), 3)


def _detect_text_sections(lines: List[str]) -> List[str]:
    """Detect section headers in generic text lines."""
    sections = []
    heading_patterns = [
        re.compile(r'^(?:#+|\d+(\.\d+)*\s+|[A-Z0-9\s_-]{3,}:)'),
        re.compile(r'^[A-Z\s]{4,}$'),
    ]
    for line in lines:
        cleaned = line.strip()
        if 3 <= len(cleaned) <= 100:
            for pat in heading_patterns:
                if pat.match(cleaned):
                    sections.append(cleaned)
                    break
    return sections


def parse_pdf(file_path: str) -> ParsedDocument:
    """Parse PDF document using PyMuPDF (fitz)."""
    parse_errors = []
    title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
    content_parts = []
    sections = []
    page_count = 0
    author = ""
    creation_date = ""

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        meta = doc.metadata or {}
        if meta.get("title") and meta["title"].strip():
            title = meta["title"].strip()
        author = meta.get("author", "")
        creation_date = meta.get("creationDate", "")

        for page_idx in range(page_count):
            page = doc[page_idx]
            text = page.get_text("text")
            if text:
                content_parts.append(text)
                # Extract potential headers from page lines
                for line in text.splitlines():
                    cleaned = line.strip()
                    if cleaned and len(cleaned) < 80 and (cleaned.isupper() or cleaned.istitle()):
                        if cleaned not in sections:
                            sections.append(cleaned)
        doc.close()
    except Exception as e:
        parse_errors.append(f"PDF parsing error: {str(e)}")

    content = "\n\n".join(content_parts).strip()
    quality = _compute_quality_score(content, page_count=page_count, doc_type="pdf")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="pdf",
        page_count=page_count,
        metadata={
            "sections": sections[:30],
            "author": author,
            "creation_date": creation_date,
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_docx(file_path: str) -> ParsedDocument:
    """Parse DOCX document using python-docx."""
    parse_errors = []
    title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
    paragraphs_text = []
    sections = []
    author = ""
    creation_date = ""

    try:
        doc = Document(file_path)
        core_props = doc.core_properties
        if core_props.title and core_props.title.strip():
            title = core_props.title.strip()
        if core_props.author:
            author = core_props.author
        if core_props.created:
            creation_date = str(core_props.created)

        # Extract paragraphs
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style_name = (p.style.name if p.style else "").lower()
            if "heading" in style_name:
                sections.append(text)
            paragraphs_text.append(text)

        # Extract tables
        for table in doc.tables:
            table_lines = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_lines.append(" | ".join(row_cells))
            if table_lines:
                paragraphs_text.append("\n".join(table_lines))

    except Exception as e:
        parse_errors.append(f"DOCX parsing error: {str(e)}")

    content = "\n\n".join(paragraphs_text).strip()
    quality = _compute_quality_score(content, doc_type="docx")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="docx",
        page_count=max(1, len(content) // 2500) if content else 0,
        metadata={
            "sections": sections[:30],
            "author": author,
            "creation_date": creation_date,
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_markdown(file_path: str) -> ParsedDocument:
    """Parse Markdown file, stripping frontmatter and extracting section headers."""
    parse_errors = []
    title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
    content = ""
    sections = []
    author = ""
    creation_date = ""

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        # Strip YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, re.DOTALL)
        if frontmatter_match:
            fm_text = frontmatter_match.group(1)
            for line in fm_text.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip().strip('"\'')
                    if key == "title" and val:
                        title = val
                    elif key == "author" and val:
                        author = val
                    elif key in ("date", "created") and val:
                        creation_date = val
            content = raw_text[frontmatter_match.end():]
        else:
            content = raw_text

        # Extract markdown headers
        for line in content.splitlines():
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                header_title = header_match.group(2).strip()
                sections.append(header_title)
                if not frontmatter_match and header_match.group(1) == '#':
                    title = header_title

    except Exception as e:
        parse_errors.append(f"Markdown parsing error: {str(e)}")

    content = content.strip()
    quality = _compute_quality_score(content, doc_type="markdown")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="markdown",
        page_count=max(1, len(content) // 2000) if content else 0,
        metadata={
            "sections": sections[:30],
            "author": author,
            "creation_date": creation_date,
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_html(file_path: str) -> ParsedDocument:
    """Parse HTML file using BeautifulSoup, stripping navigation, footers, and scripts."""
    parse_errors = []
    title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
    content = ""
    sections = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "html.parser")

        # Strip navigation, footer, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Extract title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Extract sections from heading tags
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            h_text = h.get_text().strip()
            if h_text and h_text not in sections:
                sections.append(h_text)

        content = soup.get_text(separator="\n\n")
        # Collapse excessive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content).strip()

    except Exception as e:
        parse_errors.append(f"HTML parsing error: {str(e)}")

    quality = _compute_quality_score(content, doc_type="html")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="html",
        page_count=max(1, len(content) // 2000) if content else 0,
        metadata={
            "sections": sections[:30],
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_code(file_path: str) -> ParsedDocument:
    """Parse code files, extracting docstrings, comments, classes, and function signatures."""
    parse_errors = []
    filename = os.path.basename(file_path)
    title = filename
    sections = []
    content = ""

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        signatures = []
        for line in lines:
            line_str = line.strip()
            # Python, JS, TS, Go, Java, Rust definitions
            if re.match(r'^(async\s+def|def|class|function|export\s+function|type|interface|pub\s+fn|fn|func)\s+([A-Za-z0-9_]+)', line_str):
                signatures.append(line_str)
                if len(sections) < 30:
                    sections.append(line_str)

        content = "".join(lines).strip()
    except Exception as e:
        parse_errors.append(f"Code parsing error: {str(e)}")

    quality = _compute_quality_score(content, doc_type="code")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="code",
        page_count=max(1, len(content) // 1500) if content else 0,
        metadata={
            "sections": sections[:30],
            "language": os.path.splitext(file_path)[1].lstrip('.'),
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_text(file_path: str) -> ParsedDocument:
    """Parse plain text documents with encoding fallback."""
    parse_errors = []
    title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
    content = ""
    sections = []

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    text_read = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text_read = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if text_read is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text_read = f.read()
            parse_errors.append("Encoding fallback to utf-8 replacement")
        except Exception as e:
            parse_errors.append(f"Failed to read file: {str(e)}")
            text_read = ""

    content = text_read.strip() if text_read else ""
    lines = content.splitlines()

    # Detect title if first non-empty line looks like a title
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            if len(cleaned) < 100 and not cleaned.endswith('.'):
                title = cleaned
            break

    sections = _detect_text_sections(lines)
    quality = _compute_quality_score(content, doc_type="text")

    return ParsedDocument(
        content=content,
        title=title,
        source_path=file_path,
        doc_type="text",
        page_count=max(1, len(content) // 2000) if content else 0,
        metadata={
            "sections": sections[:30],
            "char_count": len(content),
        },
        parse_quality_score=quality,
        parse_errors=parse_errors
    )


def parse_document(file_path: str) -> ParsedDocument:
    """Auto-detect file type and parse."""
    if not os.path.isfile(file_path):
        return ParsedDocument(
            content="",
            title=os.path.basename(file_path),
            source_path=file_path,
            doc_type="unknown",
            page_count=0,
            metadata={},
            parse_quality_score=0.0,
            parse_errors=[f"File not found: {file_path}"]
        )

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return parse_docx(file_path)
    elif ext in (".md", ".markdown"):
        return parse_markdown(file_path)
    elif ext in (".html", ".htm", ".xhtml"):
        return parse_html(file_path)
    elif ext in (
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".rs", ".sql", ".sh",
        ".bash", ".yaml", ".yml", ".json", ".xml"
    ):
        return parse_code(file_path)
    else:
        return parse_text(file_path)
