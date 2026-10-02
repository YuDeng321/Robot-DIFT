#!/usr/bin/env bash
set -euo pipefail

# One shared LIBERO-10 Stage-II candidate policy. The paper specifies 100
# training epochs and 50 evaluation rollouts per task, but does not fully
# specify the LIBERO-10 training split. This recipe assumes all 50 available
# demonstrations per task, as in the paper's fixed-encoder controls.
# The candidate readout has explicit CLIP-query/FPN/RoPE conventions; its
# checkpoints are not interchangeable with raw-dense (legacy readout) runs.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_ROOT=${WORK:-$PWD}
CONDA_ENV=${ROBOT_DIFT_CONDA_ENV:-${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v python)")")}}
MODEL_DIR=${ROBOT_DIFT_MODEL_DIR:-${WORK_ROOT}/datasets/robot_dift/pretrained/sd2-1-base}
CLIP_PATH=${ROBOT_DIFT_CLIP_MODEL:-${WORK_ROOT}/datasets/robot_dift/pretrained/clip-vit-b32/ViT-B-32.pt}
DATASET_PATH=${ROBOT_DIFT_LIBERO_ROOT:-${WORK_ROOT}/datasets/libero/libero_10}
LIBERO_SOURCE=${ROBOT_DIFT_LIBERO_SOURCE:-${WORK_ROOT}/src/LIBERO}
STAGE1_ENCODER=${ROBOT_DIFT_STAGE1_ENCODER:-/path/to/stage1/encoder/checkpoint-300000-ema}
BATCH_SIZE=${ROBOT_DIFT_STAGE2_BATCH_SIZE:-8}
READOUT_CONFIG=${ROBOT_DIFT_STAGE2_READOUT_CONFIG:-robot_dift_paper_candidate}

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROBOT_DIFT_STAGE2_BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
case "$READOUT_CONFIG" in
    robot_dift_paper_candidate|robot_dift_compact_candidate) ;;
    *) echo "Invalid ROBOT_DIFT_STAGE2_READOUT_CONFIG: $READOUT_CONFIG" >&2; exit 2 ;;
esac
# Student input resolution: 256 is the paper setting. Other multiples of 64 run
# the frozen Student at a different scale (a resolution ablation).
IMAGE_SIZE=${ROBOT_DIFT_STAGE2_IMAGE_SIZE:-256}
if ! [[ "$IMAGE_SIZE" =~ ^[1-9][0-9]*$ ]] || (( IMAGE_SIZE % 64 != 0 || IMAGE_SIZE > 1024 )); then
    echo "ROBOT_DIFT_STAGE2_IMAGE_SIZE must be a positive multiple of 64 up to 1024" >&2
    exit 2
fi
IMAGE_TAG=
if [[ "$IMAGE_SIZE" != 256 ]]; then
    IMAGE_TAG=_img${IMAGE_SIZE}
fi

export ROBOT_DIFT_STAGE1_ENCODER="$STAGE1_ENCODER"
export ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT=${ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT:-$STAGE1_ENCODER}
export ROBOT_DIFT_CLIP_MODEL="$CLIP_PATH"
export ROBOT_DIFT_LIBERO_SOURCE="$LIBERO_SOURCE"
export ROBOT_DIFT_MODEL_DIR="$MODEL_DIR"
export WORK="$WORK_ROOT"
export PYTHONPATH="${LIBERO_SOURCE}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=${WANDB_MODE:-offline}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/robot_dift_numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/robot_dift_mpl_cache}
export ROBOT_DIFT_POLICY_OUTPUT_DIR=${ROBOT_DIFT_POLICY_OUTPUT_DIR:-${WORK_ROOT}/runs/stage2_libero10_${READOUT_CONFIG}${IMAGE_TAG}_${ROBOT_DIFT_RUN_ID:-local}}

if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$ROBOT_DIFT_POLICY_OUTPUT_DIR"
    for required in \
        "${CONDA_ENV}/bin/python" \
        "${MODEL_DIR}/model_index.json" \
        "$CLIP_PATH" \
        "$DATASET_PATH" \
        "${STAGE1_ENCODER}/metadata.json" \
        "${STAGE1_ENCODER}/timestep.bin"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing LIBERO Stage-II prerequisite: $required" >&2
            exit 2
        fi
    done
    STUDENT_WEIGHT_COUNT=0
    for weight_name in diffusion_pytorch_model.bin model.safetensors diffusion_pytorch_model.safetensors; do
        if [[ -f "${STAGE1_ENCODER}/unet/${weight_name}" ]]; then
            STUDENT_WEIGHT_COUNT=$((STUDENT_WEIGHT_COUNT + 1))
        fi
    done
    if [[ "$STUDENT_WEIGHT_COUNT" -ne 1 ]]; then
        echo "Expected exactly one Stage-I Student UNet weight file under ${STAGE1_ENCODER}/unet" >&2
        exit 2
    fi
    "${CONDA_ENV}/bin/python" - "$STAGE1_ENCODER/metadata.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
if metadata.get("readout") != "paper" and metadata.get("fusion_mode") != "global_to_fine":
    raise SystemExit("Stage-II candidate adapter export requires paper-readout or global_to_fine Stage-I weights")
PY
    if [[ ! -d "${LIBERO_SOURCE}/libero/libero" ]]; then
        echo "LIBERO source tree is missing: $LIBERO_SOURCE" >&2
        exit 2
    fi
    "${CONDA_ENV}/bin/python" - "$DATASET_PATH" "$LIBERO_SOURCE" <<'PY'
import sys
from pathlib import Path

import h5py

dataset, source = map(Path, sys.argv[1:])
files = sorted(dataset.glob("*_demo.hdf5"))
if len(files) != 10:
    raise SystemExit(f"Expected 10 LIBERO-10 task HDF5 files under {dataset}; found {len(files)}")
for path in files:
    task = path.name.removesuffix("_demo.hdf5")
    bddl = source / "libero" / "libero" / "bddl_files" / "libero_10" / f"{task}.bddl"
    if not bddl.is_file():
        raise SystemExit(f"Missing task language BDDL file: {bddl}")
    with h5py.File(path) as handle:
        demos = handle["data"]
        if len(demos) < 50:
            raise SystemExit(f"Expected at least 50 demonstrations for {task}; found {len(demos)}")
        if any(int(demo.attrs["num_samples"]) > 600 for demo in demos.values()):
            raise SystemExit(f"Demonstration longer than configured max_len_data=600: {path}")
PY
    if ! "${CONDA_ENV}/bin/python" -c 'import torch; import simulation.libero_sim; assert torch.cuda.is_available()' >/dev/null; then
        echo "LIBERO simulator import or CUDA preflight failed" >&2
        exit 2
    fi
fi

CMD=(
    "${CONDA_ENV}/bin/python" run.py
    --config-name=libero_config
    "hydra.sweep.subdir=robot_dift_${SLURM_JOB_ID:-$$}"
    agents=droid_diffusion_agent
    agents/model=droid/droid_diffusion_unet_stage2
    "agents/obs_encoders=${READOUT_CONFIG}"
    "agents.obs_encoders.pretrained_fusion_checkpoint=${ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT}"
    "agents.language_encoders.model_name=${CLIP_PATH}"
    agent_name=droid_diffusion
    task_suite=libero_10
    "dataset_path=${DATASET_PATH}"
    traj_per_task=50
    obs_seq_len=2
    +pred_seq_len=16
    act_seq_len=8
    window_size=17
    obs_tokens=2
    "train_batch_size=${BATCH_SIZE}"
    num_workers=0
    epoch=100
    sim_eval_every_n_epochs=0
    simulation.num_episode=50
    "group=robot_dift_libero10_stage2_${READOUT_CONFIG}"
)

if [[ -n "$IMAGE_TAG" ]]; then
    CMD+=( "agents.obs_encoders.resize_shape=[${IMAGE_SIZE},${IMAGE_SIZE}]" )
fi

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
"${CMD[@]}"
"${CONDA_ENV}/bin/python" scripts/release/export_stage2_adapter.py \
    "$ROBOT_DIFT_POLICY_OUTPUT_DIR/last_model.pth" "$ROBOT_DIFT_POLICY_OUTPUT_DIR/adapter" \
    --stage1-checkpoint "$STAGE1_ENCODER" \
    --clip-model "$CLIP_PATH"
