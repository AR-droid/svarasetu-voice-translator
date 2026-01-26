import os
import subprocess
import sys

# DEFACTO CONFIG
SARVAM_KEY = "PLACEHOLDER_KEY"
GROQ_KEY = "PLACEHOLDER_KEY"

print("--- FIXING ENVIRONMENT ---")

def run(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if stdout: print(stdout)
    if stderr: print(stderr)
    return process.returncode

# 1. Update .env definitively
with open(".env", "w") as f:
    f.write(f"HF_TOKEN=PLACEHOLDER_KEY\n")
    f.write(f"SARVAM_API_KEY={SARVAM_KEY}\n")
    f.write(f"GROQ_API_KEY={GROQ_KEY}\n")
print("[v] .env updated.")

# 2. Fix Pydantic version mismatch
print("Fixing Pydantic version...")
run("pip install 'pydantic<2.0.0'")

# 3. Rebuild Index
print("Rebuilding Vector Index...")
run("python3 phase2_vector/chunking.py --input data/raw/Cricket.txt")
run("python3 phase2_vector/embeddings.py")
print("[v] Index rebuilt.")

# 4. Final verification
print("\n--- ALL SET ---")
print("Please RESTART both asr_api.py and app.py now.")
print("Run this to test translation standalone: python3 services/translation.py")
