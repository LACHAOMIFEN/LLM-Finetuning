#!/usr/bin/env bash
# Run within an existing GPU allocation. Preserve logs if either stage fails.
set -euo pipefail
cd /cluster/home/sjtu_bkb/LLM-Finetuning
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HOME="$PWD/data/hf_cache"
export PYTHONUNBUFFERED=1
run_name="${1:-tnews_qwen15b_lora_20260930}"
epochs="${2:-1}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Use letters, digits, underscores, and hyphens in run name" >&2
    exit 2
fi
if [[ ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Epochs must be a positive integer" >&2
    exit 2
fi
if [[ -e "outputs/${run_name}" || -e "outputs/${run_name}_smoke" ]]; then
    echo "Run name already exists; use a new name" >&2
    exit 2
fi
test -f models/Qwen2.5-1.5B-Instruct/download_manifest.json
.venv/bin/python -u scripts/train_tnews.py --smoke --output "outputs/${run_name}_smoke" 2>&1 | tee "logs/${run_name}_smoke.log"
.venv/bin/python -u scripts/train_tnews.py --epochs "$epochs" --output "outputs/${run_name}" 2>&1 | tee "logs/${run_name}.log"
