from services.translation import translate_to_english

test_cases = [
    {"text": "नमस्ते, आप कैसे हैं?", "lang": "hi-IN"},
    {"text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "lang": "ta-IN"},
    {"text": "హలో, మీరు ఎలా ఉన్నారు?", "lang": "te-IN"},
    {"text": "Bonjour, comment ça va ?", "lang": "auto"}
]

print("--- Sarvam AI Translation Test ---")
for case in test_cases:
    print(f"\nSource ({case['lang']}): {case['text']}")
    translated = translate_to_english(case['text'], case['lang'])
    print(f"English: {translated}")
print("\n--- Test Complete ---")
