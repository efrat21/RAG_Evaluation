"""
Generates a golden Q&A dataset from ingested documents using an LLM.
Output: data/golden_dataset/golden.jsonl
Format per line: {"question": str, "ground_truth_answer": str, "ground_truth_context": str, "source_doc_id": str, "source_title": str}
"""

import json
import os
import random
import re
import logging
from typing import List, Dict, Any
from langchain_ollama import ChatOllama

logger = logging.getLogger("golden_dataset")
logging.basicConfig(level=logging.INFO)

GENERATION_PROMPT = """You are creating a test dataset for a RAG knowledge assistant.
Given the following document excerpt, generate {n_questions} diverse question-answer pairs.
The questions should be realistic questions an employee would ask.
The answers must be based ONLY on the provided text.

Document title: {title}
Document excerpt:
{content}

Return a JSON array of objects with keys: "question", "answer".
Return ONLY the JSON array, no other text."""


def _clean_json_response(raw_text: str) -> str:
    """Strip markdown code block tags and extract JSON array."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Search for first '[' and last ']'
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    return text


async def generate_golden_dataset(
    chunks: list,  # List of DocumentChunk
    llm_model: str,
    ollama_url: str,
    output_path: str,
    n_questions_per_chunk: int = 3,
    max_chunks: int = 50,
    min_chunk_length: int = 200,
) -> list:
    """
    For each chunk (sample up to max_chunks):
    1. Send to LLM to generate n_questions Q&A pairs
    2. Parse JSON response
    3. Write to output_path as JSONL
    Returns list of all generated Q&A records.
    """
    if not chunks:
        logger.warning("No chunks provided for golden dataset generation.")
        return []

    # Filter by minimum length
    qualifying_chunks = [
        c for c in chunks
        if len(getattr(c, "content", "")) >= min_chunk_length
    ]

    if not qualifying_chunks:
        logger.warning(f"No chunks meet the minimum length threshold of {min_chunk_length} chars.")
        qualifying_chunks = chunks

    # Shuffle before sampling
    shuffled_chunks = list(qualifying_chunks)
    random.shuffle(shuffled_chunks)
    sampled_chunks = shuffled_chunks[:max_chunks]

    logger.info(f"Generating golden Q&A dataset from {len(sampled_chunks)} sampled chunks using {llm_model} at {ollama_url}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    llm = ChatOllama(
        model=llm_model,
        base_url=ollama_url,
        temperature=0.2,
    )

    all_qa_records: List[Dict[str, Any]] = []

    # Open file for appending or writing
    with open(output_path, "w", encoding="utf-8") as out_file:
        for idx, chunk in enumerate(sampled_chunks):
            content = getattr(chunk, "content", "")
            title = getattr(chunk, "title", "Company Document")
            doc_id = getattr(chunk, "doc_id", "unknown_doc")

            prompt_content = GENERATION_PROMPT.format(
                n_questions=n_questions_per_chunk,
                title=title,
                content=content[:2500]  # Avoid token limit overflow
            )

            try:
                # Call Ollama asynchronously
                response = await llm.ainvoke(prompt_content)
                response_text = response.content if hasattr(response, "content") else str(response)
                cleaned_json = _clean_json_response(response_text)
                qa_list = json.loads(cleaned_json)

                if isinstance(qa_list, list):
                    for item in qa_list:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            record = {
                                "question": item["question"].strip(),
                                "ground_truth_answer": item["answer"].strip(),
                                "ground_truth_context": content,
                                "source_doc_id": doc_id,
                                "source_title": title,
                            }
                            all_qa_records.append(record)
                            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_file.flush()

                logger.info(f"Chunk {idx + 1}/{len(sampled_chunks)}: Generated {len(qa_list) if isinstance(qa_list, list) else 0} Q&As")

            except Exception as e:
                logger.warning(f"Failed to generate Q&A for chunk {getattr(chunk, 'chunk_id', idx)}: {e}")
                continue

    logger.info(f"Golden dataset generation complete. Total QA pairs generated: {len(all_qa_records)}. Saved to {output_path}")
    return all_qa_records
