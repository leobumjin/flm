#!/usr/bin/env bash
set -euo pipefail

# Usage: bash demo.sh  (interactive Hugging Face model selection)
# Local checkpoint: bash demo.sh CHECKPOINT [flm|fmlm] [lm1b|owt] [Hydra overrides...]
# Example: STEPS=1 BATCH_SIZE=1 bash demo.sh checkpoints/lm1b_fmlm.ckpt fmlm lm1b
# This generates unconditional text; no prompt is required.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash demo.sh [1|2|3|4]"
    echo "Or: bash demo.sh CHECKPOINT [flm|fmlm] [lm1b|owt] [Hydra overrides...]"
    echo "Environment: STEPS, BATCH_SIZE, SEED, OUTPUT_PATH, PYTHON_BIN, CUDA_VISIBLE_DEVICES"
    exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
HF_REPO=""
if [[ $# -eq 0 || "${1:-}" =~ ^[1-4]$ ]]; then
    if [[ $# -eq 0 ]]; then
        printf 'Select a Hugging Face model:\n  1) FMLM / LM1B (128 tokens, default 1 step)\n  2) FMLM / OpenWebText (1024 tokens, default 1 step)\n  3) FLM / LM1B (128 tokens, default 1024 steps)\n  4) FLM / OpenWebText (1024 tokens, default 1024 steps)\n'
        read -r -p 'Model [1-4]: ' CHOICE
    else
        CHOICE="$1"
        shift
    fi
    case "${CHOICE}" in
        1) ALGO=fmlm; DATASET=lm1b; HF_REPO=david3684/FMLM-B-LM1B ;;
        2) ALGO=fmlm; DATASET=owt; HF_REPO=david3684/FMLM-B-OWT ;;
        3) ALGO=flm; DATASET=lm1b; HF_REPO=david3684/FLM-B-LM1B ;;
        4) ALGO=flm; DATASET=owt; HF_REPO=david3684/FLM-B-OWT ;;
        *) echo 'Select a number from 1 to 4.' >&2; exit 1 ;;
    esac
else
    CKPT_PATH="$(realpath -m -- "$1")"
    shift
    ALGO="${1:-flm}"
    if [[ $# -gt 0 ]]; then shift; fi
    DATASET="${1:-lm1b}"
    if [[ $# -gt 0 ]]; then shift; fi
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi
fi
case "${ALGO}" in
    flm) DEFAULT_STEPS=1024; DOUBLE_TEMB=False ;;
    fmlm) DEFAULT_STEPS=1; DOUBLE_TEMB=True ;;
    *) echo "Algorithm must be flm or fmlm." >&2; exit 1 ;;
esac
case "${DATASET}" in
    lm1b) DATA_CONFIG=lm1b-wrap; LENGTH=128; GAMMA=0.8 ;;
    owt) DATA_CONFIG=openwebtext-split; LENGTH=1024; GAMMA=1.0 ;;
    *) echo "Dataset must be lm1b or owt." >&2; exit 1 ;;
esac
if [[ "${ALGO}" == "flm" ]]; then GAMMA=0.0; fi

cd -- "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python not found. Run bash setup_venv.sh first, or set PYTHON_BIN." >&2
    exit 1
fi
"${PYTHON_BIN}" - <<'PYCHECK'
import importlib.util
import sys
required = ['torch', 'huggingface_hub', 'flash_attn', 'lightning', 'safetensors']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit('Missing dependencies: ' + ', '.join(missing)
             + '\nRun: INSTALL_FLASH_ATTN=1 bash setup_venv.sh')
import torch
if not torch.cuda.is_available():
    sys.exit('CUDA GPU is required for this demo. Check CUDA_VISIBLE_DEVICES and PyTorch.')
PYCHECK
if [[ -n "${HF_REPO}" ]]; then
    echo "Downloading ${HF_REPO} (cached files are reused)..."
    CKPT_PATH="$("${PYTHON_BIN}" - "${HF_REPO}" <<'PYDOWNLOAD'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["config.json", "model.safetensors"]))
PYDOWNLOAD
)"
fi
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/samples/demo_${DATASET}_${ALGO}_$(date +%Y%m%d_%H%M%S)_$$.json}"
OUTPUT_PATH="$(realpath -m -- "${OUTPUT_PATH}")"
mkdir -p -- "$(dirname -- "${OUTPUT_PATH}")"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -u -m main \
    "mode=${MODE:-sample_eval}" \
    "seed=${SEED:-1}" \
    model=small "model.length=${LENGTH}" \
    "data=${DATA_CONFIG}" "algo=${ALGO}" \
    "eval.checkpoint_path='${CKPT_PATH}'" \
    loader.batch_size=1 "loader.eval_batch_size=${BATCH_SIZE:-1}" \
    sampling.num_sample_batches=1 "sampling.steps=${STEPS:-${DEFAULT_STEPS}}" \
    "sampling.gamma=${GAMMA}" "algo.double_temb=${DOUBLE_TEMB}" \
    algo.learnable_loss_weighting=False eval.disable_ema=False \
    eval.compute_generative_perplexity=False \
    "eval.generated_samples_path='${OUTPUT_PATH}'" \
    hydra.job.chdir=False +wandb.offline=true "$@"

"${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
for index, text in enumerate(result["generated_seqs"], start=1):
    print(f"\n--- Sample {index} ---\n{text}")
print(f"\nSaved to: {sys.argv[1]}")
PY
