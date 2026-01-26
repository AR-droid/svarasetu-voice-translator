import os
import uuid
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from transformers import AutoModel

# ---------------- LOAD ENV ---------------- #

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found. Set it in .env")

# ---------------- CONFIG ---------------- #

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SR = 16000

# ---------------- APP ---------------- #

app = FastAPI(
    title="IndicConformer ASR API",
    description="Multilingual ASR using AI4Bharat IndicConformer (CTC decoding)",
    version="1.0"
)

# ---------------- LOAD MODEL ---------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN
).to(device)

model.eval()

# ---------------- HELPERS ---------------- #

def load_audio(file_path: str):
    from pydub import AudioSegment
    import numpy as np
    
    # Load with pydub (handles mp3, wav, etc.)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    
    # Convert to tensor
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # Normalize to [-1, 1]
    samples /= (1 << (8 * audio.sample_width - 1))
    
    wav = torch.from_numpy(samples).unsqueeze(0)
    return wav.to(device)

# ---------------- API ---------------- #

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(..., description="ISO language code (e.g., ta, hi, te)")
):
    # Normalize language code (e.g., ta-IN -> ta)
    norm_lang = language.split('-')[0]
    print(f"Incoming lang: {language}, Normalized lang: {norm_lang}")
    
    temp_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    waveform = load_audio(temp_path)

    with torch.no_grad():
        transcription = model(
            waveform,
            norm_lang,
            "ctc"
        )

    os.remove(temp_path)

    return {
        "language": norm_lang,
        "transcription": transcription
    }

# ---------------- HEALTH ---------------- #

@app.get("/")
def health_check():
    return {"status": "ASR API running"}
