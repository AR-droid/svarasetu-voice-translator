# Indic Voice-Enabled RAG System

This project is an end-to-end voice-enabled conversational system built for the AI4Bharat Hiring Challenge. It integrates Automatic Speech Recognition (ASR), Translation, Vector Retrieval (RAG), and Large Language Models (LLM) to answer voice-based questions.

## Features
- **Task 1: Data Collection**: Wikipedia scraper with ambiguity handling.
- **Task 2: Vector DB**: Section-aware chunking and FAISS vector storage.
- **Task 3: ASR Deployment**: FastAPI server running AI4Bharat's `indic-conformer-600m-multilingual` model.
- **Task 4: Translation**: English translation using Sarvam's official Python SDK.
- **Task 5: RAG Pipeline**: End-to-end integration of all components.
- **Bonus: Chat UI**: Gradio-based web interface for seamless interaction.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- At least 10GB of free disk space (for the ASR model and dependencies).
- API Keys: Hugging Face (with access to AI4Bharat models) and Sarvam AI.

### 2. Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_token
SARVAM_API_KEY=your_sarvam_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## How to Run

### Step 1: Data Collection & Vector DB
```bash
# Collect data (Example: Cricket)
python phase1_data/datacollection.py "Cricket"

# Chunk and create embeddings
python phase2_vector/chunking.py --input data/raw/Cricket.txt
python phase2_vector/embeddings.py
```

### Step 2: Deploy ASR API
```bash
# This starts the ASR server on port 8000
python -m uvicorn phase3_asr.asr_api:app --host 0.0.0.0 --port 8000
```

### Step 3: Run the Chat UI
```bash
# This starts the Gradio interface
python app.py
```
Visit `http://localhost:7860` in your browser.

---

## Observations & Implementation Details

### 1. Chunking Strategy (Task 2)
- **Chunk Size**: `3500 chars` (~800-1000 tokens). This size is chosen to provide sufficient context to the LLM while staying within common context windows. It ensures that complex Wikipedia sections are captured in their entirety.
- **Overlap**: `400 chars`. Overlapping chunks prevent context loss at boundaries (e.g., a sentence being cut in half) and improve retrieval quality for questions that might bridge two chunks.
- **Justification**: Using section-aware chunking (splitting by Wikipedia headers) ensures that topics are logically grouped before being subdivided, preserving semantic coherence.

### 2. Vector Database (Task 2)
- **Tool**: FAISS (Facebook AI Similarity Search).
- **Benefits**: Extremely fast, lightweight (runs locally), and integrates perfectly with LangChain/HuggingFace embeddings.
- **Drawbacks**: Unlike managed solutions (Qdrant/Pinecone), it is not naturally persistent/scalable as a standalone service without extra management code.

### 3. Challenges Faced
- **Disk Space**: The `indic-conformer` model is quite large (~2.5GB including ONNX assets). Downloading these model files on a full disk caused IO errors, requiring significant cleanup.
- **SDK Compatibility**: Initial attempts to use the REST API for Sarvam translation hit header/authorization issues (403). Switching to the official `sarvamai` Python SDK resolved these and handled language code formatting (e.g., `hi-IN` instead of just `hi`) correctly.
- **ASR Model Loading**: The model requires `onnxruntime` and `torch`, which are heavy dependencies that require specific versions to work smoothly on Mac Silicon (M1/M2/M3).

---

## Technical Constraints & Handling
- **Ambiguity Detection**: The Wikipedia search script includes a `resolve_ambiguity` function to filter out "may refer to" pages and select the most relevant canonical article.
- **Wait for ASR**: The pipeline handles cases where the ASR server might be slow to respond during initial cold start (model loading).
- **LLM Provider**: Switched to Groq with `llama-3.3-70b-versatile` for extremely low-latency and high-quality responses.

---
*Disclaimer: This project is strictly for evaluation purposes as part of the AI4Bharat Hiring Challenge.*
