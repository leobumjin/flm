#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

# flash-attn requires a compatible CUDA/PyTorch build environment.
# Enable its installation with: INSTALL_FLASH_ATTN=1 bash setup_venv.sh
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    "${VENV_DIR}/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation
fi

printf '\nSetup complete. Activate the environment with:\n'
printf 'source %q\n' "${VENV_DIR}/bin/activate"
