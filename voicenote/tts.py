"""Zero-shot voice-cloning TTS with AI4Bharat IndicF5."""

import os
import re

import numpy as np

MODEL_ID = "ai4bharat/IndicF5"
TTS_SR = 24000

# Sentence enders across Indic scripts: danda, double danda, and Latin punctuation.
_SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks short enough for one F5 generation.

    F5 generates a fixed-length mel span per call, so very long inputs are
    synthesised sentence by sentence and joined.
    """
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


class IndicF5TTS:
    sample_rate = TTS_SR

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN")
        # IndicF5's remote __init__ builds the vocoder and loads its own weights.
        # Recent transformers run __init__ under from_pretrained on the meta device,
        # which breaks that, so the class is constructed directly instead.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=token)
        model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_id, token=token)
        self.model = model_cls(config).to(self.device)
        self.model.eval()

    def _generate(self, text: str, ref_audio_path: str, ref_text: str) -> np.ndarray:
        audio = np.asarray(self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32).reshape(-1)

    def synthesize(self, text: str, ref_audio_path: str, ref_text: str, pause_s: float = 0.25) -> np.ndarray:
        """Speak text in the voice of the reference clip; returns float32 audio at 24 kHz."""
        pause = np.zeros(int(pause_s * self.sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for chunk in split_sentences(text):
            if pieces:
                pieces.append(pause)
            pieces.append(self._generate(chunk, ref_audio_path, ref_text))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
