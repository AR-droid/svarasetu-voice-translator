"""Indic-to-Indic text translation.

Two backends behind one interface:
- SarvamTranslator: hosted API (sarvam-translate:v1), no GPU needed.
- IndicTrans2Translator: self-hosted AI4Bharat IndicTrans2 indic-indic model.
"""

import os
from typing import Protocol

from voicenote.languages import get_language


class Translator(Protocol):
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str: ...


class SarvamTranslator:
    def __init__(self, api_key: str | None = None, model: str = "sarvam-translate:v1"):
        from sarvamai import SarvamAI

        api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set")
        self.client = SarvamAI(api_subscription_key=api_key)
        self.model = model

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        options = {"mode": "modern-colloquial"} if self.model == "mayura:v1" else {}
        response = self.client.text.translate(
            input=text,
            source_language_code=get_language(src_lang).sarvam,
            target_language_code=get_language(tgt_lang).sarvam,
            model=self.model,
            **options,
        )
        return response.translated_text.strip()


class IndicTrans2Translator:
    MODEL_ID = "ai4bharat/indictrans2-indic-indic-dist-320M"

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.processor import IndicProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, token=token
        ).to(self.device)
        self.model.eval()
        self.processor = IndicProcessor(inference=True)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = get_language(src_lang).indictrans
        tgt = get_language(tgt_lang).indictrans
        batch = self.processor.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True
        ).to(self.device)
        with self.torch.no_grad():
            tokens = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5)
        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.processor.postprocess_batch(decoded, lang=tgt)[0].strip()


def build_translator(backend: str | None = None) -> Translator:
    backend = (backend or os.getenv("TRANSLATOR", "sarvam")).lower()
    if backend == "sarvam":
        return SarvamTranslator()
    if backend == "indictrans2":
        return IndicTrans2Translator()
    raise ValueError(f"Unknown translator backend '{backend}' (use 'sarvam' or 'indictrans2')")
