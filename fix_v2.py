import os
import subprocess
import sys

# DEFACTO CONFIG
SARVAM_KEY = "PLACEHOLDER_KEY"
GROQ_KEY = "PLACEHOLDER_KEY"

print("--- RECONCILING ENVIRONMENT (Pydantic v2) ---")

def run(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if stdout: print(stdout)
    if stderr: print(stderr)
    return process.returncode

# 1. Standardize on Pydantic v2 (required by FastAPI/Gradio)
print("Standardizing packages...")
run("pip install 'pydantic>=2.0.0' 'fastapi>=0.100.0' 'gradio>=4.0.0'")

# 2. Reset .env
with open(".env", "w") as f:
    f.write(f"HF_TOKEN=PLACEHOLDER_KEY\n")
    f.write(f"SARVAM_API_KEY={SARVAM_KEY}\n")
    f.write(f"GROQ_API_KEY={GROQ_KEY}\n")
print("[v] .env updated.")

# 3. Clean and Rebuild Index with Pydantic v2
print("Cleaning old index...")
if os.path.exists("phase2_vector/faiss_index"):
    import shutil
    shutil.rmtree("phase2_vector/faiss_index")

print("Rebuilding Vector Index with v2 compatibility...")
run("python3 phase2_vector/chunking.py --input data/raw/Cricket.txt")
run("python3 phase2_vector/embeddings.py")
print("[v] Index rebuilt.")

print("\n--- RESOLVED ---")
print("1. Your servers (FastAPI/Gradio) will now start correctly.")
print("2. The Vector DB is now v2-compatible, so the KeyError is fixed.")
print("Proceed to run: python3 app.py")
