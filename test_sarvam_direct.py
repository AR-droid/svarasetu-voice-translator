from sarvamai import SarvamAI

# DEFINITIVE KEY TEST
KEY = "sk_pg47a1pz_Z5wukrX59ynth9oN0Tn2aESB"

client = SarvamAI(api_subscription_key=KEY)
try:
    response = client.text.translate(
        input="வணக்கம்",
        source_language_code="ta-IN",
        target_language_code="en-IN",
        model="mayura:v1"
    )
    print(f"Success: {response.translated_text}")
except Exception as e:
    print(f"Failed with key {KEY[:5]}...: {e}")
