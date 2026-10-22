"""End-to-end voice-note translation: ASR -> translation -> voice-cloning TTS."""

import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from voicenote.audio import ASR_SR, load_audio, select_reference, write_wav
from voicenote.languages import get_language


@dataclass
class TranslationResult:
    src_lang: str
    tgt_lang: str
    transcript: str
    translation: str
    reference_text: str
    audio: np.ndarray
    sample_rate: int
    input_seconds: float
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def output_seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def real_time_factor(self) -> float:
        """Processing time per second of input audio (below 1 is faster than real time)."""
        return self.total_seconds / self.input_seconds if self.input_seconds else float("nan")


class VoiceNoteTranslator:
    def __init__(self, asr, translator, tts, reference_seconds: float = 8.0):
        self.asr = asr
        self.translator = translator
        self.tts = tts
        self.reference_seconds = reference_seconds

    def translate_file(self, audio_path: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        return self.translate(load_audio(audio_path, ASR_SR), src_lang, tgt_lang)

    def translate(self, wav_16k: np.ndarray, src_lang: str, tgt_lang: str) -> TranslationResult:
        src, tgt = get_language(src_lang), get_language(tgt_lang)
        if src.code == tgt.code:
            raise ValueError("Source and target languages must differ")
        timings: dict[str, float] = {}

        @contextmanager
        def timed(stage: str):
            start = time.perf_counter()
            yield
            timings[stage] = round(time.perf_counter() - start, 3)

        with timed("asr"):
            transcript = self.asr.transcribe(wav_16k, src.code)
        if not transcript:
            raise ValueError("No speech recognised in the voice note")

        with timed("translation"):
            translation = self.translator.translate(transcript, src.code, tgt.code)

        with timed("reference"):
            reference = select_reference(wav_16k, ASR_SR, self.reference_seconds)
            # F5 needs the exact words spoken in the prompt clip; reuse the full
            # transcript when the clip is the whole note, otherwise re-transcribe it.
            if len(reference) >= len(wav_16k) * 0.95:
                reference_text = transcript
            else:
                reference_text = self.asr.transcribe(reference, src.code)

        fd, ref_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(ref_path, reference, ASR_SR)
            with timed("tts"):
                audio = self.tts.synthesize(translation, ref_path, reference_text)
        finally:
            os.remove(ref_path)

        return TranslationResult(
            src_lang=src.code,
            tgt_lang=tgt.code,
            transcript=transcript,
            translation=translation,
            reference_text=reference_text,
            audio=audio,
            sample_rate=self.tts.sample_rate,
            input_seconds=len(wav_16k) / ASR_SR,
            timings=timings,
        )


def build_default_pipeline() -> VoiceNoteTranslator:
    """Load the production models (needs HF_TOKEN with IndicF5 access; GPU recommended)."""
    from dotenv import load_dotenv

    load_dotenv()
    from voicenote.asr import IndicConformerASR
    from voicenote.translate import build_translator
    from voicenote.tts import IndicF5TTS

    return VoiceNoteTranslator(IndicConformerASR(), build_translator(), IndicF5TTS())
