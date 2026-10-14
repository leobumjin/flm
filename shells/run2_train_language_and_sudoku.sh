#!/usr/bin/env bash
# Train both compact tutorial models and write reusable checkpoints.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
cd -- "${PROJECT_DIR}"

printf '\033[1;36m\n▶ Run 2A · fine-tune paper-title edit\033[0m\n'
DIT_USE_COMPILE=0 "${PYTHON_BIN}" "${PROJECT_DIR}/tutorials/paper_edit.py" \
  --steps "${LANGUAGE_STEPS:-10000}" --batch-size "${LANGUAGE_BATCH_SIZE:-2}" \
  --print-every "${LANGUAGE_PRINT_EVERY:-1000}" --progress-every "${LANGUAGE_PROGRESS_EVERY:-0}" \
  --edit-ratio "${EDIT_RATIO:-0.20}" \
  --output-dir "${PROJECT_DIR}/outputs/run2_language" \
  --checkpoint "${PROJECT_DIR}/outputs_large_files/run2_fmlm_paper_edit.pt"

printf '\033[1;36m\n▶ Run 2B · train Sudoku masked model\033[0m\n'
"${PYTHON_BIN}" "${PROJECT_DIR}/tutorials/masking_tutorial.py" sudoku --mode train \
  --steps "${SUDOKU_STEPS:-800}" --batch-size "${BATCH_SIZE:-256}" --seed "${SEED:-1}" \
  --sudoku-size "${SUDOKU_SIZE:-4}" --output-dir "${PROJECT_DIR}/outputs/run2_sudoku" \
  --figure "${PROJECT_DIR}/figures/run2_sudoku_training.svg" \
  --checkpoint "${PROJECT_DIR}/outputs_large_files/run2_sudoku.pt"
"${PYTHON_BIN}" "${PROJECT_DIR}/tutorials/present_results.py" run3 \
  --output-dir "${PROJECT_DIR}/outputs/run2_sudoku" --figure "${PROJECT_DIR}/figures/run2_sudoku_training.svg"
