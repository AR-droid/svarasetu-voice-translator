import numpy as np
import pytest

from voicenote.audio import ASR_SR
from voicenote.pipeline import VoiceNoteTranslator


class FakeASR:
    def __init__(self):
        self.calls = []

    def transcribe(self, wav, lang):
        self.calls.append((len(wav), lang))
        return f"{lang} words {len(wav)}"


class FakeTranslator:
    def translate(self, text, src_lang, tgt_lang):
        return f"[{src_lang}->{tgt_lang}] {text}"


class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synthesize(self, text, ref_audio_path, ref_text):
        import soundfile as sf

        ref, sr = sf.read(ref_audio_path)
        self.calls.append({"text": text, "ref_text": ref_text, "ref_seconds": len(ref) / sr})
        return np.zeros(self.sample_rate, dtype=np.float32)


def tone(seconds, sr=ASR_SR, freq=220.0, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def fakes():
    asr, tts = FakeASR(), FakeTTS()
    return asr, tts, VoiceNoteTranslator(asr, FakeTranslator(), tts)
