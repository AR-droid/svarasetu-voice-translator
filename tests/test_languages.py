import pytest

from voicenote.languages import LANGUAGES, get_language


def test_accepts_common_forms():
    assert get_language("ta").name == "Tamil"
    assert get_language("hi-IN").code == "hi"
    assert get_language("Telugu").code == "te"
    assert get_language("od-IN").code == "or"


def test_rejects_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        get_language("fr")


def test_eleven_indicf5_languages():
    assert len(LANGUAGES) == 11
