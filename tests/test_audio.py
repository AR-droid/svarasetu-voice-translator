import numpy as np

from conftest import tone
from voicenote.audio import ASR_SR, load_audio, resample, select_reference, trim_silence, write_wav


def test_resample_length():
    wav = tone(1.0, sr=48000)
    assert len(resample(wav, 48000, ASR_SR)) == ASR_SR


def test_trim_silence_removes_edges():
    wav = np.concatenate([np.zeros(ASR_SR), tone(1.0), np.zeros(ASR_SR)])
    trimmed = trim_silence(wav, ASR_SR)
    assert 1.0 <= len(trimmed) / ASR_SR <= 1.4


def test_select_reference_picks_loudest_window():
    quiet, loud = tone(10.0, amp=0.01), tone(5.0, amp=0.8)
    wav = np.concatenate([quiet, loud, quiet])
    ref = select_reference(wav, ASR_SR, max_seconds=5.0)
    assert len(ref) == 5 * ASR_SR
    assert np.abs(ref).max() > 0.5


def test_short_note_kept_whole():
    wav = tone(3.0)
    assert len(select_reference(wav, ASR_SR, max_seconds=8.0)) == len(trim_silence(wav, ASR_SR))


def test_load_audio_downmixes_and_resamples(tmp_path):
    stereo = np.stack([tone(1.0, sr=44100), tone(1.0, sr=44100)], axis=1)
    path = tmp_path / "note.wav"
    write_wav(str(path), stereo, 44100)
    wav = load_audio(str(path))
    assert wav.ndim == 1 and abs(len(wav) - ASR_SR) <= 1
