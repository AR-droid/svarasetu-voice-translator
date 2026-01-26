import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile" 
LLM_BASE_URL = "https://api.groq.com/openai/v1"

def test_groq():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found!")
        return

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=GROQ_API_KEY
    )

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.2,
            max_tokens=100
        )
        print(f"Groq Response: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"Groq Error: {e}")

if __name__ == "__main__":
    test_groq()
