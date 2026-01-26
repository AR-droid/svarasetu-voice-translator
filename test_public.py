from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
    print("Successfully loaded public model config.")
except Exception as e:
    print(f"Failed: {e}")
