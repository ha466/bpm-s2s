#!/bin/bash
# Usage: ./run_smoke.sh bpm|ar|bpm_k8|bpm_gru   (~7 min each on 1 vCPU; resumes from runs/<name>/partial.pt)
cd "$(dirname "$0")"
python3 scripts/train.py --config configs/smoke_${1}.yaml --out runs/${1} >> runs_${1}.log 2>&1
