"""Evaluation metrics: round-trip CER, translation chrF, and speaker similarity."""

import unicodedata

import numpy as np


def normalize(text: str) -> str:
    """NFC-normalise, drop punctuation and symbols (including danda), collapse whitespace.

    Filters by Unicode category rather than regex \\w, which would also strip
    the combining vowel signs and viramas that Indic scripts depend on.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    import jiwer

    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.cer(reference, hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.wer(reference, hypothesis))


def chrf(reference: str, hypothesis: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_chrf(hypothesis, [reference]).score)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class SpeakerEncoder:
    """ECAPA-TDNN speaker embeddings (SpeechBrain, VoxCeleb-trained), 16 kHz input."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", device: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})

    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        wav = self.torch.from_numpy(np.ascontiguousarray(wav_16k, dtype=np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            return self.model.encode_batch(wav).squeeze().cpu().numpy()
