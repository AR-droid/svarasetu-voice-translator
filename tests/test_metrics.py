import numpy as np
import pytest

from voicenote.eval.metrics import cer, cosine, normalize


def test_normalize_drops_danda_keeps_matras():
    assert normalize("मैं ठीक हूँ।") == "मैं ठीक हूँ"


def test_cer_identical_after_normalisation():
    assert cer("வணக்கம், நண்பா!", "வணக்கம் நண்பா") == 0.0


def test_cer_counts_errors():
    assert 0 < cer("abcd", "abxd") < 1


def test_cosine():
    assert cosine(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert abs(cosine(np.array([1, 0]), np.array([0, 1]))) < 1e-9
