
import os
import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ---------------- CONFIG ---------------- #

CHUNKS_FILE = "phase2_vector/chunks.json"
VECTOR_DB_DIR = "phase2_vector/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- PIPELINE ---------------- #

def load_chunks(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_documents(chunks):
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                    "document": chunk.get("document", "unknown")
                }
            )
        )

    return documents


def create_vector_db(documents):
    logging.info("Initializing embedding model")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    logging.info("Creating FAISS vector store")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def save_vector_db(vectorstore, documents):
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    vectorstore.save_local(VECTOR_DB_DIR)

    metadata_path = os.path.join(VECTOR_DB_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            [doc.metadata for doc in documents],
            f,
            indent=2,
            ensure_ascii=False
        )

    logging.info(f"Vector DB saved at: {VECTOR_DB_DIR}")


def main():
    logging.info("Starting Phase-2 Task-2: Embeddings + FAISS")

    chunks = load_chunks(CHUNKS_FILE)
    documents = build_documents(chunks)

    vectorstore = create_vector_db(documents)
    save_vector_db(vectorstore, documents)

    logging.info("✅ Phase-2 Task-2 completed successfully")
    print("\n✅ Vector Database created")
    print(f"📁 Location: {VECTOR_DB_DIR}")
    print(f"📦 Total vectors: {len(documents)}\n")


if __name__ == "__main__":
    main()
