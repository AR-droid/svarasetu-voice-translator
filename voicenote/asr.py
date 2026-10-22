"""Speech recognition with AI4Bharat IndicConformer (22 Indian languages)."""

import os

import numpy as np

from voicenote.languages import get_language

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerASR:
    def __init__(self, model_id: str = MODEL_ID, decoding: str = "ctc", device: str | None = None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.decoding = decoding
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, token=os.getenv("HF_TOKEN")
        ).to(self.device)
        self.model.eval()

    def transcribe(self, wav_16k: np.ndarray, lang: str) -> str:
        code = get_language(lang).code
        if len(wav_16k) < 1600:  # under 0.1 s: nothing to recognise, and the preprocessor fails on it
            return ""
        wav = self.torch.from_numpy(np.ascontiguousarray(wav_16k, dtype=np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            text = self.model(wav.to(self.device), code, self.decoding)
        return str(text).strip()
