#!/usr/bin/env bash
# Run the pretrained FMLM/LM1B checkpoint at fixed denoising-step counts.
# Usage: bash shells/run1_checkpoint_eval.sh
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/outputs/run1_checkpoint_eval"
STEP_LIST="${STEP_LIST:-1 2 4 8 16}"
SEED="${SEED:-1}"
MODEL_CHOICE="${MODEL_CHOICE:-1}"  # demo.sh option 1: FMLM / LM1B
CLOZE_PROMPT="${CLOZE_PROMPT:-What is the capital of Korea? Answer: [MASK].}"

run_case() {
    local name="$1" mode="$2" prompt_arg="${3:-}"
    local case_dir="${OUTPUT_DIR}/${name}"
    local figure="${PROJECT_DIR}/figures/run1_${name}.svg"
    mkdir -p "${case_dir}"
    printf 'steps\telapsed_ms\n' > "${case_dir}/timings.tsv"
    for steps in ${STEP_LIST}; do
    if [[ ! "${steps}" =~ ^[1-9][0-9]*$ ]]; then
        echo "STEP_LIST must contain positive integers; got: ${steps}" >&2
        exit 2
    fi
    printf '\033[1;36m\n▶ %s · %s step(s)\033[0m\n' "${name}" "${steps}"
    started_ns="$(date +%s%N)"
    run_log="$(mktemp "${TMPDIR:-/tmp}/flm-run1.XXXXXX")"
    if [[ "${mode}" == "cloze_eval" ]]; then
        if ! MODE="${mode}" STEPS="${steps}" SEED="${SEED}" OUTPUT_PATH="${case_dir}/steps_${steps}.json" \
            bash "${PROJECT_DIR}/demo.sh" "${MODEL_CHOICE}" "+eval.cloze_prompt='${CLOZE_PROMPT}'" >"${run_log}" 2>&1; then
            sed -n '1,240p' "${run_log}" >&2
            rm -f "${run_log}"
            exit 1
        fi
    else
        if ! MODE="${mode}" STEPS="${steps}" SEED="${SEED}" OUTPUT_PATH="${case_dir}/steps_${steps}.json" \
            bash "${PROJECT_DIR}/demo.sh" "${MODEL_CHOICE}" >"${run_log}" 2>&1; then
            sed -n '1,240p' "${run_log}" >&2
            rm -f "${run_log}"
            exit 1
        fi
    fi
    rm -f "${run_log}"
    printf '%s\t%s\n' "${steps}" "$(( ($(date +%s%N) - started_ns) / 1000000 ))" \
        >> "${case_dir}/timings.tsv"
    done
    "${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/tutorials/present_results.py" run1 \
        --output-dir "${case_dir}" --figure "${figure}"
}

run_case language sample_eval
run_case factual cloze_eval yes
