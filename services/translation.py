import os
from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def translate_to_english(text: str, source_language: str = "auto") -> str:
    """
    Translates input text to English using Sarvam's Official SDK.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in environment.")

    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        response = client.text.translate(
            input=text,
            source_language_code=source_language,
            target_language_code="en-IN",
            model="mayura:v1"
        )
        # Note: SDK returns a TranslateResponse object
        return response.translated_text
    except Exception as e:
        print(f"Translation Error with SDK: {e}")
        return text

if __name__ == "__main__":
    sample_text = "नमस्ते, आप कैसे हैं?"
    # Using 'hi-IN' as per API requirements seen in previous error
    translated = translate_to_english(sample_text, "hi-IN")
    print(f"Original: {sample_text}")
    print(f"Translated: {translated}")
