import pytest

from conftest import tone


def test_short_note_reuses_transcript(fakes):
    asr, tts, pipeline = fakes
    result = pipeline.translate(tone(4.0), "ta", "hi")
    assert result.translation.startswith("[ta->hi]")
    assert len(asr.calls) == 1
    assert tts.calls[0]["ref_text"] == result.transcript
    assert set(result.timings) == {"asr", "translation", "reference", "tts"}
    assert result.sample_rate == 24000 and result.output_seconds == 1.0


def test_long_note_retranscribes_reference_clip(fakes):
    asr, tts, pipeline = fakes
    result = pipeline.translate(tone(20.0), "ta", "hi")
    assert len(asr.calls) == 2
    assert tts.calls[0]["ref_seconds"] == pytest.approx(8.0, abs=0.01)
    assert result.reference_text != result.transcript


def test_same_language_rejected(fakes):
    _, _, pipeline = fakes
    with pytest.raises(ValueError, match="differ"):
        pipeline.translate(tone(2.0), "hi", "hi-IN")
