#!/usr/bin/env python3
"""
Task-1: Data Collection with Ambiguity Handling
----------------------------------------------
1. Takes topic as command-line argument
2. Uses Wikipedia Search API (top-K)
3. Detects ambiguity and selects canonical article
4. Fetches FULL article text via Extracts API
5. Stores clean text in .txt file
"""

import argparse
import requests
import logging
import os
import re
from time import sleep

# ---------------- CONFIG ---------------- #

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "RAG-Voice-System/1.0 (Educational Project)"
}

TIMEOUT = 5
RETRIES = 3
TOP_K = 3
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- FUNCTIONS ---------------- #

def search_wikipedia(topic: str) -> str:
    """
    Search Wikipedia and resolve ambiguity by selecting
    the most relevant canonical article.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": TOP_K,
        "format": "json"
    }

    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(
                WIKI_API,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT
            )
            response.raise_for_status()

            results = response.json()["query"]["search"]

            if not results:
                raise ValueError("No search results found.")

            logging.info("Top Wikipedia search results:")
            for i, r in enumerate(results, 1):
                logging.info(f"{i}. {r['title']}")

            selected = resolve_ambiguity(results)
            logging.info(f"Selected canonical article: {selected}")
            return selected

        except Exception as e:
            logging.warning(f"Search attempt {attempt} failed: {e}")
            sleep(1)

    raise RuntimeError("Wikipedia search failed after retries.")


def resolve_ambiguity(results: list) -> str:
    """
    Resolves ambiguity by:
    - Skipping disambiguation pages
    - Selecting top-ranked canonical article
    """
    for result in results:
        title = result["title"].lower()
        snippet = result.get("snippet", "").lower()

        if "may refer to" in snippet:
            continue
        if "disambiguation" in title:
            continue

        return result["title"]

    # Fallback: return top-ranked result
    return results[0]["title"]


def fetch_full_article(title: str) -> str:
    """
    Fetch FULL Wikipedia article text using Extracts API.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json"
    }

    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(
                WIKI_API,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT
            )
            response.raise_for_status()

            pages = response.json()["query"]["pages"]
            page = next(iter(pages.values()))

            if "extract" not in page or not page["extract"]:
                raise ValueError("Empty article extract.")

            logging.info("Full article text fetched successfully.")
            return page["extract"]

        except Exception as e:
            logging.warning(f"Fetch attempt {attempt} failed: {e}")
            sleep(1)

    raise RuntimeError("Failed to fetch article text.")


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def save_text(text: str, title: str) -> None:
    filename = title.replace(" ", "_") + ".txt"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Saved article text → {path}")
    logging.info(f"Character count: {len(text)}")


def run_pipeline(topic: str) -> None:
    logging.info(f"Starting data collection for topic: '{topic}'")

    title = search_wikipedia(topic)
    article_text = fetch_full_article(title)
    cleaned_text = clean_text(article_text)
    save_text(cleaned_text, title)

    logging.info("Task-1 completed successfully.")


# ---------------- MAIN ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Phase-1: Wikipedia Data Collection with Ambiguity Handling"
    )
    parser.add_argument(
        "topic",
        type=str,
        help="Topic to search on Wikipedia"
    )

    args = parser.parse_args()

    try:
        run_pipeline(args.topic)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
