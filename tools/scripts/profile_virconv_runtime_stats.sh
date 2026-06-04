#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TOOLS_DIR}/.." && pwd)"

TARGET="${1:-baseline}"
GPU_ID="${GPU_ID:-0}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"
PROFILE_MAX_ITERS="${PROFILE_MAX_ITERS:-100}"
EXTRA_TAG="${EXTRA_TAG:-default}"
EVAL_TAG="${EVAL_TAG:-runtime_stats_w${WARMUP_ITERS}_m${PROFILE_MAX_ITERS}_bs1_gpu${GPU_ID}}"
WORKERS="${WORKERS:-0}"
SAVE_JSON="${SAVE_JSON:-1}"
DATA_PATH_OVERRIDE="${DATA_PATH_OVERRIDE:-}"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_PATH="${CKPT_PATH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-RSV}"

resolve_cfg_file() {
    local target="$1"
    local abs_path=""

    if [[ "${target}" == *.yaml ]]; then
        if [[ "${target}" = /* ]]; then
            abs_path="$(realpath "${target}")"
        elif [[ -f "${target}" ]]; then
            abs_path="$(realpath "${target}")"
        elif [[ -f "${TOOLS_DIR}/${target}" ]]; then
            abs_path="$(realpath "${TOOLS_DIR}/${target}")"
        elif [[ -f "${ROOT_DIR}/${target}" ]]; then
            abs_path="$(realpath "${ROOT_DIR}/${target}")"
        else
            echo "Config file not found: ${target}" >&2
            return 1
        fi

        if [[ "${abs_path}" == "${TOOLS_DIR}/"* ]]; then
            echo "${abs_path#${TOOLS_DIR}/}"
        else
            echo "${abs_path}"
        fi
        return 0
    fi

    case "${target}" in
        baseline|base|virconv|virconv_l)
            echo "cfgs/models/kitti/VirConv-L.yaml"
            ;;
        rsv|virconv_rsv|virconv_l_rsv)
            echo "cfgs/models/kitti/VirConv-L-RSV.yaml"
            ;;
        rsv_l|virconv_rsv_l|virconv_l_rsv_l)
            echo "cfgs/models/kitti/VirConv-L-RSV-L.yaml"
            ;;
        rsv_dbv|virconv_rsv_dbv|virconv_l_rsv_dbv)
            echo "cfgs/models/kitti/VirConv-L-RSV-DBV.yaml"
            ;;
        rsv_l_dbv|virconv_rsv_l_dbv|virconv_l_rsv_l_dbv)
            echo "cfgs/models/kitti/VirConv-L-RSV-L-DBV.yaml"
            ;;
        *)
            if [[ -f "${TOOLS_DIR}/cfgs/models/kitti/${target}.yaml" ]]; then
                echo "cfgs/models/kitti/${target}.yaml"
            else
                echo "Unsupported target or config basename: ${target}" >&2
                echo "Usage: bash tools/scripts/profile_virconv_runtime_stats.sh [baseline|rsv|rsv_l|rsv_dbv|rsv_l_dbv|cfg.yaml]" >&2
                return 1
            fi
            ;;
    esac
}

find_latest_ckpt() {
    local ckpt_dir="$1"
    local preferred_ckpt=""
    local fallback_ckpt=""

    if [[ ! -d "${ckpt_dir}" ]]; then
        return 1
    fi

    preferred_ckpt="$(find "${ckpt_dir}" -maxdepth 1 -name 'checkpoint_epoch_*.pth' | sort -V | tail -n 1)"
    if [[ -n "${preferred_ckpt}" ]]; then
        echo "${preferred_ckpt}"
        return 0
    fi

    fallback_ckpt="$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.pth' | sort -V | tail -n 1)"
    if [[ -n "${fallback_ckpt}" ]]; then
        echo "${fallback_ckpt}"
        return 0
    fi
    return 1
}

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

activate_conda_env() {
    if [[ -z "${CONDA_ENV_NAME}" ]]; then
        return 0
    fi
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
        return 0
    fi

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
        echo "Unable to locate conda.sh for CONDA_ENV_NAME=${CONDA_ENV_NAME}" >&2
        exit 1
    fi
    conda activate "${CONDA_ENV_NAME}"
}

CFG_FILE="$(resolve_cfg_file "${TARGET}")"
MODEL_TAG="$(basename "${CFG_FILE}" .yaml)"

if [[ -z "${CKPT_DIR}" ]]; then
    CKPT_DIR="${ROOT_DIR}/output/models/kitti/${MODEL_TAG}/${EXTRA_TAG}/ckpt"
fi
if [[ -z "${CKPT_PATH}" ]]; then
    CKPT_PATH="$(find_latest_ckpt "${CKPT_DIR}" || true)"
fi
if [[ -z "${CKPT_PATH}" ]]; then
    echo "No checkpoint found under ${CKPT_DIR}" >&2
    echo "Set CKPT_PATH=/abs/path/to/checkpoint.pth and rerun." >&2
    exit 1
fi

if ! DATA_PATH="$(detect_kitti_data_path)"; then
    echo "Unable to locate a valid KITTI data root." >&2
    echo "Set DATA_PATH_OVERRIDE=/abs/path/to/kitti and rerun." >&2
    exit 1
fi

SAVE_JSON_FLAG=()
if [[ "${SAVE_JSON}" == "1" ]]; then
    SAVE_JSON_FLAG=(--profile_save_json)
fi

activate_conda_env

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "${MPLCONFIGDIR}"

cd "${TOOLS_DIR}"

echo "Target       : ${TARGET}"
echo "Config       : ${CFG_FILE}"
echo "Checkpoint   : ${CKPT_PATH}"
echo "Data Root    : ${DATA_PATH}"
echo "GPU          : ${GPU_ID}"
echo "Batch Size   : 1"
echo "Workers      : ${WORKERS}"
echo "Warmup       : ${WARMUP_ITERS}"
echo "MeasureIters : ${PROFILE_MAX_ITERS} (0 means all validation batches)"
echo "Extra Tag    : ${EXTRA_TAG}"
echo "Eval Tag     : ${EVAL_TAG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python test.py \
    --cfg_file "${CFG_FILE}" \
    --ckpt "${CKPT_PATH}" \
    --batch_size 1 \
    --workers "${WORKERS}" \
    --extra_tag "${EXTRA_TAG}" \
    --eval_tag "${EVAL_TAG}" \
    --profile_runtime_stats \
    --profile_warmup_iters "${WARMUP_ITERS}" \
    --profile_max_iters "${PROFILE_MAX_ITERS}" \
    "${SAVE_JSON_FLAG[@]}" \
    --set DATA_CONFIG.DATA_PATH "${DATA_PATH}"
