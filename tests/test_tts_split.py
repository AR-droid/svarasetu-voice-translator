from voicenote.tts import split_sentences


def test_splits_on_danda_and_punctuation():
    text = "नमस्ते। आप कैसे हैं? मैं ठीक हूँ."
    assert split_sentences(text) == ["नमस्ते।", "आप कैसे हैं?", "मैं ठीक हूँ."]


def test_long_sentence_split_on_spaces():
    chunks = split_sentences("शब्द " * 100, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == ["शब्द"] * 100
