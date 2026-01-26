#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
cd /Users/anjanarangarajan/Desktop/RAG-VOICE-SYSTEM

echo "Stopping existing processes..."
pkill -f "uvicorn"
pkill -f "app.py"
sleep 2

echo "Starting ASR API..."
./venv/bin/python -m uvicorn phase3_asr.asr_api:app --host 0.0.0.0 --port 8000 > asr_startup.log 2>&1 &
ASR_PID=$!

echo "Starting Gradio UI..."
./venv/bin/python app.py > ui_startup.log 2>&1 &
UI_PID=$!

echo "ASR PID: $ASR_PID"
echo "UI PID: $UI_PID"

sleep 10
echo "Check ASR Log:"
cat asr_startup.log
echo "Check UI Log:"
cat ui_startup.log
