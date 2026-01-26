#!/usr/bin/env python3
"""
Phase-2: Chunking (Final Clean Version)
--------------------------------------
Creates high-quality, sentence-aware, section-preserving
chunks suitable for embeddings and vector databases.
"""

import os
import re
import json
import argparse
import logging
from typing import List

# ================= CONFIG ================= #

CHUNK_SIZE = 3500        # characters (~800–1000 tokens)
OVERLAP_SIZE = 400       # preserves semantic continuity
MIN_CHUNK_LENGTH = 150

EXCLUDED_SECTIONS = {
    "references",
    "sources",
    "external links",
    "see also",
    "notes",
    "further reading"
}

OUTPUT_DIR = "phase2_vector"
OUTPUT_FILE = "chunks.json"

# ================= LOGGING ================= #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= HELPERS ================= #

def load_text(filepath: str) -> str:
    logging.info(f"Loading input file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("Input file is empty.")
    return text


def clean_section_title(title: str) -> str:
    title = re.sub(r"[=]+", "", title)
    title = title.strip()
    if not title or len(title) > 80:
        return "General"
    return title


def split_into_sections(text: str):
    """
    Split Wikipedia text into sections using headers.
    """
    pattern = r"\n==\s*(.*?)\s*==\n"
    parts = re.split(pattern, text)

    sections = []
    current_section = "Introduction"
    buffer = ""

    for part in parts:
        if part.strip() and buffer == "":
            buffer = part
        elif part.strip() and buffer:
            sections.append((current_section, buffer.strip()))
            current_section = part.strip()
            buffer = ""
        else:
            buffer += part

    if buffer.strip():
        sections.append((current_section, buffer.strip()))

    return sections


def clean_section_text(text: str) -> str:
    """
    Remove Wikipedia artifacts and headers from section body.
    """
    text = re.sub(r"\n=+.*?=+\n", "\n", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def sentence_tokenize(text: str) -> List[str]:
    """
    Lightweight sentence tokenizer (no external deps).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_sentences(sentences: List[str]) -> List[str]:
    """
    Sentence-aware chunking with overlap.
    """
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append(current_chunk.strip())

            # overlap
            overlap = current_chunk[-OVERLAP_SIZE:]
            current_chunk = overlap + " " + sentence

    if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:
        chunks.append(current_chunk.strip())

    return chunks


# ================= PIPELINE ================= #

def run_chunking(input_file: str):
    logging.info("Starting Phase-2 Chunking")

    raw_text = load_text(input_file)
    sections = split_into_sections(raw_text)

    all_chunks = []
    document_name = os.path.splitext(os.path.basename(input_file))[0].lower()
    section_index = 0

    for raw_title, section_body in sections:
        section_title = clean_section_title(raw_title)

        if section_title.lower() in EXCLUDED_SECTIONS:
            continue

        cleaned_text = clean_section_text(section_body)
        if len(cleaned_text) < MIN_CHUNK_LENGTH:
            continue

        sentences = sentence_tokenize(cleaned_text)
        chunks = chunk_sentences(sentences)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{document_name}_s{section_index}_c{chunk_index}",
                "document": document_name,
                "section": section_title,
                "text": chunk
            })

        section_index += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logging.info(f"Chunking completed: {len(all_chunks)} chunks")
    logging.info(f"Saved to {output_path}")

    print("\n✅ Phase-2 Chunking Complete")
    print(f"   Clean chunks created: {len(all_chunks)}")
    print(f"   Output: {output_path}\n")


# ================= CLI ================= #

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2: Clean, sentence-aware chunking for RAG"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Phase-1 cleaned Wikipedia text file"
    )

    args = parser.parse_args()
    run_chunking(args.input)


if __name__ == "__main__":
    main()
