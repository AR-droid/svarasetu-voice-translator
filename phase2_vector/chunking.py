#!/usr/bin/env python3
"""
Phase-2: Chunking
-----------------
Clean, section-aware, sentence-safe chunking for RAG pipelines.
"""

import os
import re
import json
import argparse
import logging
from typing import List

# ---------------- CONFIG ---------------- #

CHUNK_SIZE = 3500        # characters (~800–1000 tokens)
OVERLAP_SIZE = 400       # characters
MIN_CHUNK_SIZE = 500     # avoid junk chunks

OUTPUT_DIR = "phase2_vector"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.json")

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- HELPERS ---------------- #

def load_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("Input file is empty.")
    return text


def normalize_section_title(raw: str) -> str:
    """Remove Wikipedia markup like == ==="""
    return re.sub(r"=+", "", raw).strip()


def split_by_sections(text: str):
    """
    Split text using Wikipedia-style headers.
    """
    pattern = r"(==+.*?==+)"
    parts = re.split(pattern, text)

    sections = []
    current_title = "Introduction"
    buffer = ""

    for part in parts:
        if re.match(r"==+.*?==+", part):
            if buffer.strip():
                sections.append((current_title, buffer.strip()))
            current_title = normalize_section_title(part)
            buffer = ""
        else:
            buffer += part

    if buffer.strip():
        sections.append((current_title, buffer.strip()))

    return sections


def sentence_chunk(text: str) -> List[str]:
    """
    Chunk text safely using sentence boundaries + overlap.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= CHUNK_SIZE:
            current += " " + sent if current else sent
        else:
            chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    # Add overlap
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap = chunks[i - 1][-OVERLAP_SIZE:]
            chunk = overlap + " " + chunk
        final_chunks.append(chunk.strip())

    # Merge small trailing chunks
    cleaned = []
    for chunk in final_chunks:
        if cleaned and len(chunk) < MIN_CHUNK_SIZE:
            cleaned[-1] += " " + chunk
        else:
            cleaned.append(chunk)

    return cleaned


# ---------------- PIPELINE ---------------- #

def run_chunking(input_file: str):
    logging.info("Starting Phase-2: Chunking")

    text = load_text_file(input_file)
    sections = split_by_sections(text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []
    base_name = os.path.splitext(os.path.basename(input_file))[0].lower()

    for s_idx, (section_title, section_text) in enumerate(sections):
        chunks = sentence_chunk(section_text)

        for c_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{base_name}_s{s_idx}_c{c_idx}",
                "document": base_name,
                "section": section_title,
                "text": chunk
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logging.info(f"Chunking completed: {len(all_chunks)} chunks")
    logging.info(f"Output written to {OUTPUT_FILE}")

    print("\n✅ Phase-2 Chunking Complete")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Output: {OUTPUT_FILE}\n")


# ---------------- CLI ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2: Chunk Wikipedia article into RAG-ready segments"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Phase-1 cleaned text file"
    )

    args = parser.parse_args()
    run_chunking(args.input)


if __name__ == "__main__":
    main()
