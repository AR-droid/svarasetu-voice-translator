from dotenv import load_dotenv
import requests
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
url = "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual/resolve/main/config.json"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Successfully reached HF and authenticated.")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
