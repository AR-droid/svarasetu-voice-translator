#!/bin/bash
echo "Verifying environment..." > output.log
./venv/bin/python --version >> output.log 2>&1
ls -l phase2_vector/faiss_index >> output.log 2>&1
echo "Done" >> output.log
