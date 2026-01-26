import torch
from pydub import AudioSegment
import numpy as np
import os

TARGET_SR = 16000

def load_audio(file_path: str):
    print(f"Loading {file_path}...")
    try:
        # Load with pydub (handles mp3, wav, etc.)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        
        # Convert to tensor
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1]
        samples /= (1 << (8 * audio.sample_width - 1))
        
        wav = torch.from_numpy(samples).unsqueeze(0)
        print(f"Successfully loaded. Shape: {wav.shape}")
        return wav
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

if __name__ == "__main__":
    file = "shubh_tts_audio.mp3"
    if os.path.exists(file):
        load_audio(file)
    else:
        print(f"File {file} not found!")
