"""REST API: POST a voice note, get back the translated note in the speaker's voice."""

import base64
import io
import os
import tempfile
from functools import lru_cache

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from voicenote.languages import LANGUAGES
from voicenote.pipeline import VoiceNoteTranslator, build_default_pipeline

app = FastAPI(
    title="Voice Note Translator",
    description="Indic voice-note translation that keeps the speaker's voice "
    "(IndicConformer ASR, Indic-to-Indic MT, IndicF5 voice cloning).",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_pipeline() -> VoiceNoteTranslator:
    return build_default_pipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/languages")
def languages():
    return [{"code": lang.code, "name": lang.name} for lang in LANGUAGES.values()]


@app.post("/translate")
async def translate(
    audio: UploadFile = File(..., description="Voice note (wav, ogg/opus, mp3, m4a, webm)"),
    src_lang: str = Form(..., examples=["ta"]),
    tgt_lang: str = Form(..., examples=["hi"]),
):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await audio.read())
        try:
            result = get_pipeline().translate_file(path, src_lang, tgt_lang)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        os.remove(path)

    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format="WAV")
    return {
        "src_lang": result.src_lang,
        "tgt_lang": result.tgt_lang,
        "transcript": result.transcript,
        "translation": result.translation,
        "reference_text": result.reference_text,
        "input_seconds": round(result.input_seconds, 2),
        "output_seconds": round(result.output_seconds, 2),
        "timings": result.timings,
        "real_time_factor": round(result.real_time_factor, 3),
        "audio_wav_base64": base64.b64encode(buffer.getvalue()).decode(),
    }
