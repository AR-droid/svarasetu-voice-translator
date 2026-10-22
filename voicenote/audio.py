"""Audio loading, resampling and reference-clip selection (numpy only, no torch)."""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ASR_SR = 16000


def load_audio(path: str, target_sr: int = ASR_SR) -> np.ndarray:
    """Load any common audio file as mono float32 at target_sr."""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        # Formats libsndfile can't read (m4a, some opus) go through ffmpeg via librosa.
        import librosa

        wav, sr = librosa.load(path, sr=None, mono=True)
    return resample(wav.astype(np.float32), sr, target_sr)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    g = gcd(orig_sr, target_sr)
    return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32)


def frame_energy(wav: np.ndarray, sr: int, frame_ms: int = 30) -> np.ndarray:
    hop = int(sr * frame_ms / 1000)
    n = len(wav) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = wav[: n * hop].reshape(n, hop)
    return np.sqrt((frames**2).mean(axis=1) + 1e-12)


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -40.0, pad_ms: int = 150) -> np.ndarray:
    """Drop leading and trailing silence relative to the loudest frame."""
    frame_ms = 30
    energy = frame_energy(wav, sr, frame_ms)
    if energy.size == 0:
        return wav
    db = 20 * np.log10(energy / energy.max())
    voiced = np.flatnonzero(db > threshold_db)
    if voiced.size == 0:
        return wav
    hop = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * hop - pad)
    end = min(len(wav), (voiced[-1] + 1) * hop + pad)
    return wav[start:end]


def select_reference(wav: np.ndarray, sr: int, max_seconds: float = 8.0) -> np.ndarray:
    """Pick the most voiced window of at most max_seconds as the voice-cloning prompt.

    F5-style cloning works best with a short, clean prompt (roughly 3-10 s), so long
    notes are cut to the window with the most speech energy.
    """
    wav = trim_silence(wav, sr)
    max_len = int(max_seconds * sr)
    if len(wav) <= max_len:
        return wav
    hop = sr // 10
    energy = np.square(wav)
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    starts = np.arange(0, len(wav) - max_len + 1, hop)
    scores = cumulative[starts + max_len] - cumulative[starts]
    best = int(starts[int(np.argmax(scores))])
    return wav[best : best + max_len]


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)
