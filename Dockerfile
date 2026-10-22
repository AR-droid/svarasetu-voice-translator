FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-voicenote.txt .
RUN pip install --no-cache-dir -r requirements-voicenote.txt

COPY voicenote/ voicenote/

# HF_TOKEN (with access to ai4bharat/IndicF5) and SARVAM_API_KEY come from the environment.
ENV HF_HOME=/app/.cache/huggingface
EXPOSE 7860 8000

# Web demo by default; for the REST API run:
#   uvicorn voicenote.server:app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "voicenote.app"]
