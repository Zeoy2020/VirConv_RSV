#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TOOLS_DIR}/.." && pwd)"

MODE="${1:-rsv}"
GPU_ID="${GPU_ID:-0}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
PROFILE_MAX_ITERS="${PROFILE_MAX_ITERS:-0}"
EVAL_TAG="${EVAL_TAG:-latency_breakdown}"
EXTRA_TAG="${EXTRA_TAG:-default}"
SAVE_JSON="${SAVE_JSON:-1}"
DATA_PATH_OVERRIDE="${DATA_PATH_OVERRIDE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

case "${MODE}" in
    baseline|base|virconv)
        CFG_FILE="cfgs/models/kitti/VirConv-L.yaml"
        CKPT_DIR="${ROOT_DIR}/output/models/kitti/VirConv-L/${EXTRA_TAG}/ckpt"
        ;;
    rsv|virconv_rsv)
        CFG_FILE="cfgs/models/kitti/VirConv-L-RSV.yaml"
        CKPT_DIR="${ROOT_DIR}/output/models/kitti/VirConv-L-RSV/${EXTRA_TAG}/ckpt"
        ;;
    *)
        echo "Unsupported mode: ${MODE}"
        echo "Usage: bash tools/scripts/profile_virconv_latency.sh [baseline|rsv]"
        exit 1
        ;;
esac

CKPT_PATH="${CKPT_PATH:-$(find "${CKPT_DIR}" -maxdepth 1 -name 'checkpoint_epoch_*.pth' | sort | tail -n 1)}"
if [[ -z "${CKPT_PATH}" ]]; then
    echo "No checkpoint found under ${CKPT_DIR}"
    exit 1
fi

detect_kitti_data_path() {
    local candidates=()
    if [[ -n "${DATA_PATH_OVERRIDE}" ]]; then
        candidates+=("${DATA_PATH_OVERRIDE}")
    fi
    candidates+=(
        "${ROOT_DIR}/data/kitti"
        "/data/zhouyi/kitti"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/kitti_infos_val.pkl" && -f "${candidate}/ImageSets/val.txt" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! DATA_PATH="$(detect_kitti_data_path)"; then
    echo "Unable to locate a valid KITTI data root."
    echo "Set DATA_PATH_OVERRIDE=/abs/path/to/kitti and rerun."
    exit 1
fi

SAVE_JSON_FLAG=()
if [[ "${SAVE_JSON}" == "1" ]]; then
    SAVE_JSON_FLAG=(--profile_save_json)
fi

if [[ -n "${CONDA_ENV_NAME}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        # shellcheck disable=SC1091
        source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Unable to locate conda.sh for CONDA_ENV_NAME=${CONDA_ENV_NAME}"
        exit 1
    fi
    conda activate "${CONDA_ENV_NAME}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
mkdir -p "${MPLCONFIGDIR}"

cd "${TOOLS_DIR}"

echo "Mode         : ${MODE}"
echo "Config       : ${CFG_FILE}"
echo "Checkpoint   : ${CKPT_PATH}"
echo "Data Root    : ${DATA_PATH}"
echo "GPU          : ${GPU_ID}"
echo "Warmup       : ${WARMUP_ITERS}"
echo "MeasureIters : ${PROFILE_MAX_ITERS} (0 means all remaining batches)"
echo "Eval Tag     : ${EVAL_TAG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python test.py \
    --cfg_file "${CFG_FILE}" \
    --ckpt "${CKPT_PATH}" \
    --batch_size 1 \
    --extra_tag "${EXTRA_TAG}" \
    --eval_tag "${EVAL_TAG}" \
    --profile_latency_breakdown \
    --profile_warmup_iters "${WARMUP_ITERS}" \
    --profile_max_iters "${PROFILE_MAX_ITERS}" \
    "${SAVE_JSON_FLAG[@]}" \
    --set DATA_CONFIG.DATA_PATH "${DATA_PATH}"
