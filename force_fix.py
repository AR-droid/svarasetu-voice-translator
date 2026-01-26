import os
import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout + res.stderr

with open("fix_log.txt", "w") as f:
    f.write("--- Starting Fix ---\n")
    f.write(run("./venv/bin/pip install 'pydantic<2.0.0'"))
    f.write("\n--- Rebuilding Index ---\n")
    f.write(run("./venv/bin/python phase2_vector/chunking.py --input data/raw/Cricket.txt"))
    f.write(run("./venv/bin/python phase2_vector/embeddings.py"))
    f.write("\n--- Verifying Index ---\n")
    f.write(run("ls -la phase2_vector/faiss_index"))
    f.write("\n--- Checking Env ---\n")
    f.write(run("cat .env"))
    f.write("\n--- TEST SARVAM ---\n")
    # Setting the key directly in the test to verify key validity
    f.write(run("./venv/bin/python test_sarvam_direct.py"))
