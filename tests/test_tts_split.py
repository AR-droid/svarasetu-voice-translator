from voicenote.tts import split_sentences


def test_splits_on_danda_and_punctuation():
    text = "नमस्ते। आप कैसे हैं? मैं ठीक हूँ."
    assert split_sentences(text) == ["नमस्ते।", "आप कैसे हैं?", "मैं ठीक हूँ."]


def test_long_sentence_split_on_spaces():
    chunks = split_sentences("शब्द " * 100, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == ["शब्द"] * 100


def test_checkpoint_keys_match_through_torch_compile_wrapper():
    from voicenote.tts import match_checkpoint_keys

    mapping = match_checkpoint_keys(
        ["ema_model.transformer.w", "vocoder.head.w", "unused.x"],
        ["ema_model.transformer.w", "vocoder._orig_mod.head.w", "vocoder._orig_mod.backbone.w"],
    )
    assert mapping == {
        "ema_model.transformer.w": "ema_model.transformer.w",
        "vocoder.head.w": "vocoder._orig_mod.head.w",
    }
