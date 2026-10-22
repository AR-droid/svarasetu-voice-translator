"""Voice-preserving voice-note translator for Indian languages.

Speak a note in one Indian language; hear it in another, in your own voice.
Pipeline: IndicConformer ASR -> Indic-to-Indic translation -> IndicF5 voice cloning.
"""

from voicenote.languages import LANGUAGES, get_language

__all__ = ["LANGUAGES", "get_language"]
