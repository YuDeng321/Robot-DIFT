#!/usr/bin/env bash
set -euo pipefail

# One-task Stage-II candidate with explicit CLIP-query/RoPE/cross-view-max
# readout conventions, plus the paper's 2/16/8 horizons, frozen Student,
# trainable task readout, 1D U-Net policy, 100 epochs, and 50 final rollouts.
# The paper omits several readout design choices; label results as a
# candidate until measured and compared.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [--dry-run] [RoboCasaTaskName]" >&2
    exit 2
fi
TASK=${1:-CoffeePressButton}
if [[ ! "$TASK" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid RoboCasa task name: $TASK" >&2
    exit 2
fi
READOUT_CONFIG=${ROBOT_DIFT_STAGE2_READOUT_CONFIG:-robot_dift_paper_candidate}
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
SCENE_PRESET=${ROBOT_DIFT_ROBOCASA_EVAL_SCENE_PRESET:-default}
case "$SCENE_PRESET" in
    default) SCENE_STYLE_IDS= ;;
    development_style_9) SCENE_STYLE_IDS='[9]' ;;
    final_style_10) SCENE_STYLE_IDS='[10]' ;;
    heldout_styles_9_10) SCENE_STYLE_IDS='[9,10]' ;;
    *)
        echo "Invalid ROBOT_DIFT_ROBOCASA_EVAL_SCENE_PRESET: $SCENE_PRESET" >&2
        exit 2
        ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_ROOT=${WORK:-$PWD}
CONDA_ENV=${ROBOT_DIFT_CONDA_ENV:-${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v python)")")}}
MODEL_DIR=${ROBOT_DIFT_MODEL_DIR:-${WORK_ROOT}/datasets/robot_dift/pretrained/sd2-1-base}
CLIP_PATH=${ROBOT_DIFT_CLIP_MODEL:-${WORK_ROOT}/datasets/robot_dift/pretrained/clip-vit-b32/ViT-B-32.pt}
DATASET_PATH=${ROBOT_DIFT_ROBOCASA_ROOT:-${WORK_ROOT}/datasets/robot_dift/robocasa/v0.1/single_stage}
export ROBOT_DIFT_STAGE1_ENCODER=${ROBOT_DIFT_STAGE1_ENCODER:-/path/to/stage1/encoder/checkpoint-300000-ema}
export ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT=${ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT:-$ROBOT_DIFT_STAGE1_ENCODER}
export ROBOT_DIFT_CLIP_MODEL="$CLIP_PATH"
export ROBOT_DIFT_MODEL_DIR="$MODEL_DIR"
export WORK="$WORK_ROOT"
export WANDB_MODE=${WANDB_MODE:-offline}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/robot_dift_numba_${ROBOT_DIFT_RUN_ID:-local}}
export ROBOT_DIFT_POLICY_OUTPUT_DIR=${ROBOT_DIFT_POLICY_OUTPUT_DIR:-${WORK_ROOT}/runs/stage2_robocasa_${TASK}_${READOUT_CONFIG}${IMAGE_TAG}_${ROBOT_DIFT_RUN_ID:-local}}
export ROBOT_DIFT_EPISODE_LOG_PATH=${ROBOT_DIFT_EPISODE_LOG_PATH:-${ROBOT_DIFT_POLICY_OUTPUT_DIR}/sim_episodes.jsonl}
export ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE=${ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE:-1}
export ROBOT_DIFT_SAVE_SIM_EMA=${ROBOT_DIFT_SAVE_SIM_EMA:-1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$NUMBA_CACHE_DIR"
    mkdir -p "$ROBOT_DIFT_POLICY_OUTPUT_DIR"
    for required in \
        "${CONDA_ENV}/bin/python" \
        "${MODEL_DIR}/model_index.json" \
        "${CLIP_PATH}" \
        "${ROBOT_DIFT_STAGE1_ENCODER}/metadata.json" \
        "${ROBOT_DIFT_STAGE1_ENCODER}/timestep.bin" \
        "${DATASET_PATH}"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing Stage-II prerequisite: $required" >&2
            exit 2
        fi
    done
    if [[ ! -e "${ROBOT_DIFT_STAGE1_ENCODER}/unet/diffusion_pytorch_model.bin" \
        && ! -e "${ROBOT_DIFT_STAGE1_ENCODER}/unet/model.safetensors" \
        && ! -e "${ROBOT_DIFT_STAGE1_ENCODER}/unet/diffusion_pytorch_model.safetensors" ]]; then
        echo "Missing Stage-II Student UNet weights under ${ROBOT_DIFT_STAGE1_ENCODER}/unet" >&2
        exit 2
    fi
    "${CONDA_ENV}/bin/python" - "$ROBOT_DIFT_STAGE1_ENCODER/metadata.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
if metadata.get("readout") != "paper" and metadata.get("fusion_mode") != "global_to_fine":
    raise SystemExit("Stage-II candidate adapter export requires paper-readout or global_to_fine Stage-I weights")
PY
fi

CMD=(
    "${CONDA_ENV}/bin/python" run.py
    --config-name=robocasa_config
    "hydra.sweep.dir=${ROBOT_DIFT_POLICY_OUTPUT_DIR}/hydra"
    hydra.sweep.subdir=run
    agents=droid_diffusion_agent
    agents/model=droid/droid_diffusion_unet_stage2
    "agents/obs_encoders=${READOUT_CONFIG}"
    "agents.obs_encoders.pretrained_fusion_checkpoint=${ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT}"
    "agents.language_encoders.model_name=${CLIP_PATH}"
    agent_name=droid_diffusion
    "env_name=[${TASK}]"
    "dataset_path=${DATASET_PATH}"
    obs_seq_len=2
    pred_seq_len=16
    act_seq_len=8
    obs_tokens=2
    epoch=100
    sim_eval_every_n_epochs=0
    simulation.num_episode=50
    +isolated_sim_eval_episodes=50
    +skip_final_sim=True
    "group=robot_dift_robocasa_stage2_${READOUT_CONFIG}"
)
if [[ "${ROBOT_DIFT_STAGE2_CACHE_GIB:-0}" != 0 && "${ROBOT_DIFT_STAGE2_CACHE_GIB:-0}" != 0.0 ]]; then
    CMD+=(+trainset.return_frame_ids=True)
fi
if [[ -n "$SCENE_STYLE_IDS" ]]; then
    # Demonstration metadata contains styles 0..8 and 11; validate live resets
    # before using styles 9 and 10 for any held-out success claim.
    CMD+=( "simulation.style_ids=${SCENE_STYLE_IDS}" "+scene_eval_preset=${SCENE_PRESET}" )
fi

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
"${CONDA_ENV}/bin/python" scripts/release/eval_stage2_isolated.py \
    --config "$ROBOT_DIFT_POLICY_OUTPUT_DIR/hydra/run/.hydra/config.yaml" \
    --full-checkpoint "$ROBOT_DIFT_POLICY_OUTPUT_DIR/last_model.pth" \
    --episodes 50 --training-epoch 100 \
    --output "$ROBOT_DIFT_EPISODE_LOG_PATH"
"${CONDA_ENV}/bin/python" scripts/release/export_stage2_adapter.py \
    "$ROBOT_DIFT_POLICY_OUTPUT_DIR/last_model.pth" "$ROBOT_DIFT_POLICY_OUTPUT_DIR/adapter" \
    --stage1-checkpoint "$ROBOT_DIFT_STAGE1_ENCODER" \
    --clip-model "$CLIP_PATH"
