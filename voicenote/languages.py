"""Languages supported end to end: recognised by IndicConformer, translated by
Sarvam / IndicTrans2, and synthesised by IndicF5."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # IndicConformer / ISO 639-1 style code
    name: str
    sarvam: str  # Sarvam API code
    indictrans: str  # IndicTrans2 FLORES-style code


LANGUAGES = {
    lang.code: lang
    for lang in [
        Language("as", "Assamese", "as-IN", "asm_Beng"),
        Language("bn", "Bengali", "bn-IN", "ben_Beng"),
        Language("gu", "Gujarati", "gu-IN", "guj_Gujr"),
        Language("hi", "Hindi", "hi-IN", "hin_Deva"),
        Language("kn", "Kannada", "kn-IN", "kan_Knda"),
        Language("ml", "Malayalam", "ml-IN", "mal_Mlym"),
        Language("mr", "Marathi", "mr-IN", "mar_Deva"),
        Language("or", "Odia", "od-IN", "ory_Orya"),
        Language("pa", "Punjabi", "pa-IN", "pan_Guru"),
        Language("ta", "Tamil", "ta-IN", "tam_Taml"),
        Language("te", "Telugu", "te-IN", "tel_Telu"),
    ]
}


def get_language(code: str) -> Language:
    """Look up a language by code, accepting forms like 'ta', 'ta-IN' or 'Tamil'."""
    key = code.strip().split("-")[0].lower()
    if key == "od":
        key = "or"
    if key in LANGUAGES:
        return LANGUAGES[key]
    for lang in LANGUAGES.values():
        if lang.name.lower() == code.strip().lower():
            return lang
    supported = ", ".join(LANGUAGES)
    raise ValueError(f"Unsupported language '{code}'. Supported: {supported}")
