#!/usr/bin/env bash
set -euo pipefail

# Robot-DIFT Stage I on DROID with the paper protocol (docs/STAGE1_PROTOCOL.md).
# Every training setting is a default of train_droid_auto.py; this launcher only
# sets resources, paths, and the global batch. Extra arguments are forwarded to
# train_droid_auto.py, e.g. `--num_epochs 2` for a smoke test (ROBOT_DIFT_SMOKE=1).
#
# Usage: bash scripts/release/stage1_paper_protocol.sh [--dry-run] [train_droid_auto.py args...]

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_ROOT=${WORK:-$PWD}
GPUS=${ROBOT_DIFT_GPUS:-8}
PER_GPU_BATCH=${ROBOT_DIFT_PER_GPU_BATCH:-32}
ACCUMULATION_STEPS=${ROBOT_DIFT_ACCUMULATION_STEPS:-1}
SEED=${ROBOT_DIFT_SEED:-0}
SMOKE=${ROBOT_DIFT_SMOKE:-0}
RUN_NAME=${ROBOT_DIFT_RUN_NAME:-robot_dift_stage1}
SHUFFLE_BUFFER_SIZE=${ROBOT_DIFT_SHUFFLE_BUFFER_SIZE:-50000}
SAVE_FREQ=${ROBOT_DIFT_SAVE_FREQ:-10000}
SAVE_STEPS=${ROBOT_DIFT_SAVE_STEPS:-1000 5000}
DATA_ROOT=${ROBOT_DIFT_DATA_ROOT:-${WORK_ROOT}/datasets/robot_dift}
MODEL_DIR=${ROBOT_DIFT_MODEL_DIR:-${WORK_ROOT}/datasets/robot_dift/pretrained/sd2-1-base}
CLIP_MODEL=${ROBOT_DIFT_CLIP_MODEL:-${WORK_ROOT}/datasets/robot_dift/pretrained/clip-vit-b32/ViT-B-32.pt}
OUT_ROOT=${ROBOT_DIFT_OUTPUT_ROOT:-${WORK_ROOT}/checkpoints/${RUN_NAME}}
OCTO_ROOT=${ROBOT_DIFT_OCTO_ROOT:-${WORK_ROOT}/src/octo_85b83}
MASTER_PORT=${MASTER_PORT:-29530}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

for value in "$GPUS" "$PER_GPU_BATCH" "$ACCUMULATION_STEPS" "$SHUFFLE_BUFFER_SIZE" "$SAVE_FREQ"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU count, per-GPU batch, accumulation, shuffle buffer, and save frequency must be positive integers" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ROBOT_DIFT_SEED must be a non-negative integer" >&2
    exit 2
fi
if [[ "$SMOKE" != 1 && $((GPUS * PER_GPU_BATCH * ACCUMULATION_STEPS)) -ne 256 ]]; then
    echo "The paper's effective global batch is 256 (GPUS x PER_GPU_BATCH x ACCUMULATION_STEPS);" \
         "set ROBOT_DIFT_SMOKE=1 for a smaller startup check" >&2
    exit 2
fi
if [[ "$#" -gt 0 && "$SMOKE" != 1 && "${ROBOT_DIFT_ABLATION:-0}" != 1 ]]; then
    echo "Extra train_droid_auto.py arguments change the paper protocol. Set ROBOT_DIFT_SMOKE=1" \
         "for a startup check or ROBOT_DIFT_ABLATION=1 for a named ablation run." >&2
    exit 2
fi

# Octo hashes the TFDS data_dir into its statistics cache key. Resolve a
# symlinked alias to the physical DROID root so every run reuses one cache.
if [[ -d "${DATA_ROOT}/droid" ]]; then
    DATA_ROOT=$(dirname "$(realpath "${DATA_ROOT}/droid")")
fi

if [[ "$DRY_RUN" != 1 ]]; then
    for required in "${DATA_ROOT}/droid" "${MODEL_DIR}/model_index.json" "$CLIP_MODEL"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing Stage-I prerequisite: $required" >&2
            exit 2
        fi
    done
fi
if [[ -n "${ROBOT_DIFT_RESUME_FROM:-}" ]]; then
    if [[ ! "$ROBOT_DIFT_RESUME_FROM" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
        echo "ROBOT_DIFT_RESUME_FROM must be an absolute checkpoint path without whitespace" >&2
        exit 2
    fi
    if [[ "$DRY_RUN" != 1 && ! -f "$ROBOT_DIFT_RESUME_FROM" ]]; then
        echo "Missing resume checkpoint: $ROBOT_DIFT_RESUME_FROM" >&2
        exit 2
    fi
fi

TORCHRUN_BIN=${ROBOT_DIFT_TORCHRUN:-$(command -v torchrun || true)}
if [[ -z "$TORCHRUN_BIN" ]]; then
    echo "torchrun not found; activate the training environment or set ROBOT_DIFT_TORCHRUN" >&2
    exit 2
fi

export WORK="$WORK_ROOT"
# The VAE, text encoder, Teacher, and Student all load this one SD2.1 snapshot.
export ROBOT_DIFT_MODEL_DIR="$MODEL_DIR"
export ROBOT_DIFT_CLEANDIFT_MODEL_REPO="$MODEL_DIR"
export ROBOT_DIFT_CLIP_MODEL="$CLIP_MODEL"
export HF_HOME=${HF_HOME:-${WORK_ROOT}/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-${WORK_ROOT}/cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/${USER:-$(id -un)}/xdg}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [[ -d "${OCTO_ROOT}/octo" ]]; then
    export PYTHONPATH="${REPO_ROOT}/droid_policy_learning:${OCTO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    else
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
    fi
fi

read -r -a SAVE_STEP_ARGS <<< "$SAVE_STEPS"
CMD=(
    "$TORCHRUN_BIN"
    --nproc_per_node="$GPUS"
    --master_port="$MASTER_PORT"
    train_droid_auto.py
    --name "${RUN_NAME}_seed${SEED}"
    --seed "$SEED"
    --data_path "$DATA_ROOT"
    --clip_model "$CLIP_MODEL"
    --batch_size "$PER_GPU_BATCH"
    --gradient_accumulation_steps "$ACCUMULATION_STEPS"
    --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
    --checkpoint_dir "${OUT_ROOT}/policy"
    --save_cleandift_dir "${OUT_ROOT}/encoder"
    --save_freq "$SAVE_FREQ"
    --require_encoder_export
    --no_wandb
)
if [[ "${#SAVE_STEP_ARGS[@]}" -gt 0 && "$SMOKE" != 1 ]]; then
    CMD+=(--save_steps "${SAVE_STEP_ARGS[@]}")
fi
if [[ "$GPUS" -gt 1 ]]; then
    CMD+=(--use_ddp)
fi
if [[ -n "${ROBOT_DIFT_RESUME_FROM:-}" ]]; then
    CMD+=(--resume_from "$ROBOT_DIFT_RESUME_FROM")
fi
CMD+=("$@")

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
# Replace this shell before the multi-day job so edits to this file cannot be
# read back at a shifted offset while training runs.
exec "${CMD[@]}"
