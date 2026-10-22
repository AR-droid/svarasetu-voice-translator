"""Zero-shot voice-cloning TTS with AI4Bharat IndicF5."""

import os
import re

import numpy as np

MODEL_ID = "ai4bharat/IndicF5"
TTS_SR = 24000

# Sentence enders across Indic scripts: danda, double danda, and Latin punctuation.
_SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks short enough for one F5 generation.

    F5 generates a fixed-length mel span per call, so very long inputs are
    synthesised sentence by sentence and joined.
    """
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def match_checkpoint_keys(checkpoint_keys, model_keys) -> dict[str, str]:
    """Map checkpoint tensor names onto model parameter names.

    torch.compile wraps modules and inserts "_orig_mod." into parameter names,
    so names are compared with that segment removed.
    """
    by_plain = {key.replace("_orig_mod.", ""): key for key in model_keys}
    return {
        key: by_plain[key.replace("_orig_mod.", "")]
        for key in checkpoint_keys
        if key.replace("_orig_mod.", "") in by_plain
    }


class IndicF5TTS:
    sample_rate = TTS_SR

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN")
        # IndicF5's remote __init__ builds the Vocos vocoder on real tensors. Recent
        # transformers run __init__ under from_pretrained on the meta device, which
        # breaks that, so the class is constructed directly and the trained weights
        # (model.safetensors) are loaded here, as from_pretrained would have done.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=token)
        model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_id, token=token)
        self.model = model_cls(config)
        self.load_report = self._load_weights(model_id, token)
        self.model.to(self.device).eval()

    def _load_weights(self, model_id: str, token: str | None) -> dict[str, int]:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        checkpoint = load_file(hf_hub_download(model_id, "model.safetensors", token=token))
        model_state = self.model.state_dict()
        mapping = match_checkpoint_keys(checkpoint, model_state)
        if not mapping:
            raise RuntimeError(
                f"None of the {len(checkpoint)} tensors in {model_id}/model.safetensors match the model; "
                f"checkpoint keys look like {list(checkpoint)[:3]}, model keys like {list(model_state)[:3]}"
            )
        self.model.load_state_dict({mapping[k]: v for k, v in checkpoint.items() if k in mapping}, strict=False)
        report = {
            "checkpoint_tensors": len(checkpoint),
            "loaded": len(mapping),
            "model_tensors_not_in_checkpoint": len(set(model_state) - set(mapping.values())),
        }
        print(f"IndicF5 weights: {report}")
        return report

    def _generate(self, text: str, ref_audio_path: str, ref_text: str) -> np.ndarray:
        audio = np.asarray(self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32).reshape(-1)

    def synthesize(self, text: str, ref_audio_path: str, ref_text: str, pause_s: float = 0.25) -> np.ndarray:
        """Speak text in the voice of the reference clip; returns float32 audio at 24 kHz."""
        pause = np.zeros(int(pause_s * self.sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for chunk in split_sentences(text):
            if pieces:
                pieces.append(pause)
            pieces.append(self._generate(chunk, ref_audio_path, ref_text))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
