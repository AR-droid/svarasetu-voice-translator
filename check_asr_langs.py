import os
import torch
from dotenv import load_dotenv
from transformers import AutoModel

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

def check_langs():
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    if hasattr(model, 'language_masks'):
        print("Supported Languages:", list(model.language_masks.keys()))
    else:
        print("Model does not have language_masks attribute.")

if __name__ == "__main__":
    check_langs()
