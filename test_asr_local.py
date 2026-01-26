import torch
import os
from phase3_asr.asr_api import load_audio, model
from dotenv import load_dotenv

load_dotenv()

def test_local_asr():
    audio_path = "shubh_tts_audio.mp3"
    lang = "hi" # Normalized
    
    print(f"Testing local ASR with {audio_path}...")
    waveform = load_audio(audio_path)
    
    with torch.no_grad():
        transcription = model(
            waveform,
            lang,
            "ctc"
        )
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    test_local_asr()
