import os
import pydantic
import langchain
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

with open("diag.log", "w") as f:
    f.write(f"Pydantic Version: {pydantic.__version__}\n")
    f.write(f"Langchain Version: {langchain.__version__}\n")
    f.write(f"SARVAM_API_KEY from env: {os.getenv('SARVAM_API_KEY')[:10]}...\n")
    
    try:
        client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
        f.write("Sarvam Client Init: SUCCESS\n")
    except Exception as e:
        f.write(f"Sarvam Client Init: FAILED - {e}\n")

print("Diagnostics complete.")
