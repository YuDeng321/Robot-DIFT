#!/usr/bin/env bash
set -euo pipefail

# Reproducible frozen-Student raw-dense policy control. This is the measured
# BESo/Mamba development interface, separate from the paper-style 1D U-Net
# candidate. Supply one Student checkpoint and reuse one policy directory for
# its training and evaluation.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ "$#" -ne 2 || ( "$1" != train && "$1" != eval ) ]]; then
    echo "Usage: $0 [--dry-run] train|eval RoboCasaTaskName" >&2
    exit 2
fi
MODE=$1
TASK=$2
if [[ ! "$TASK" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid RoboCasa task name: $TASK" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_ROOT=${WORK:-$PWD}
CONDA_ENV=${ROBOT_DIFT_CONDA_ENV:-${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v python)")")}}
STUDENT=${ROBOT_DIFT_STAGE1_ENCODER:-}
MODEL_REPO=${ROBOT_DIFT_MODEL_DIR:-${WORK_ROOT}/datasets/robot_dift/pretrained/sd2-1-base}
CLIP_MODEL=${ROBOT_DIFT_CLIP_MODEL:-${WORK_ROOT}/datasets/robot_dift/pretrained/clip-vit-b32/ViT-B-32.pt}
DATASET=${ROBOT_DIFT_ROBOCASA_ROOT:-${WORK_ROOT}/datasets/robot_dift/robocasa/v0.1/single_stage}
POLICY_DIR=${ROBOT_DIFT_POLICY_DIR:-${WORK_ROOT}/runs/stage2_rawdense_${TASK}}
HYDRA_DIR=${ROBOT_DIFT_HYDRA_DIR:-${POLICY_DIR}/hydra}
EPISODE_LOG=${ROBOT_DIFT_EPISODE_LOG_PATH:-${POLICY_DIR}/sim_episodes.jsonl}
EPOCHS=${ROBOT_DIFT_STAGE2_EPOCHS:-50}
EVAL_EPISODES=${ROBOT_DIFT_EVAL_EPISODES:-50}
GPUS=${ROBOT_DIFT_STAGE2_GPUS:-4}
DEMOS=${ROBOT_DIFT_DEMOS_PER_TASK:-50}
QUERIES=${ROBOT_DIFT_QUERIES_PER_CAMERA:-8}
FUSION_MODE=${ROBOT_DIFT_STAGE2_FUSION_MODE:-s2fpn}
PER_GPU_BATCH=${ROBOT_DIFT_STAGE2_BATCH_SIZE:-2}
STYLE_IDS=${ROBOT_DIFT_SCENE_STYLE_IDS:-}
CAMERAS=3
OBS_TOKENS=$((CAMERAS * QUERIES))

for number in "$EPOCHS" "$EVAL_EPISODES" "$GPUS" "$DEMOS" "$QUERIES" "$PER_GPU_BATCH"; do
    if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then
        echo "Epochs, episodes, GPUs, demonstrations, queries, and batch must be positive integers" >&2
        exit 2
    fi
done
if [[ -n "$STYLE_IDS" && ! "$STYLE_IDS" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
    echo "ROBOT_DIFT_SCENE_STYLE_IDS must look like [9] or [9,10]" >&2
    exit 2
fi
if [[ "$FUSION_MODE" != s2fpn && "$FUSION_MODE" != global_to_fine && "$FUSION_MODE" != concat ]]; then
    echo "ROBOT_DIFT_STAGE2_FUSION_MODE must be s2fpn, global_to_fine, or concat" >&2
    exit 2
fi

export WORK="$WORK_ROOT"
export ROBOT_DIFT_CLEANDIFT_DROID_FT_CKPT="$STUDENT"  # legacy encoder config alias
export ROBOT_DIFT_CLEANDIFT_MODEL_REPO="$MODEL_REPO"  # legacy encoder config alias
export ROBOT_DIFT_POLICY_OUTPUT_DIR="$POLICY_DIR"
export ROBOT_DIFT_EPISODE_LOG_PATH="$EPISODE_LOG"
export ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE=1
export ROBOT_DIFT_CLIP_SIM_ACTIONS=1
export ROBOT_DIFT_CLIP_ROLLOUT_ACTIONS=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/robot_dift_numba}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/robot_dift_matplotlib}

common=(
    run.py --config-name=robocasa_config
    agents=beso_agent agent_name=beso_mamba
    agents/model=beso/beso_dec_mamba
    agents/obs_encoders=cleandift_droid_ft_raw_dense_queries
    "agents.language_encoders.model_name=${CLIP_MODEL}"
    agents.if_dift_language=True
    latent_dim=512 "obs_tokens=${OBS_TOKENS}" scaler_type=minmax
    num_workers=0 sim_eval_every_n_epochs=0
    simulation.max_step_per_episode=720
    "+trainset.max_demos_per_task=${DEMOS}"
    "+valset.max_demos_per_task=${DEMOS}"
    +save_sim_videos=False
    "env_name=[${TASK}]" "dataset_path=${DATASET}"
    "agents.obs_encoders.rgb_model.fpn_num_queries=${QUERIES}"
    "+agents.obs_encoders.rgb_model.fusion_mode=${FUSION_MODE}"
    "seed=42"
)
if [[ -n "$STYLE_IDS" ]]; then
    common+=("simulation.style_ids=${STYLE_IDS}")
fi

if [[ "$MODE" == train ]]; then
    cmd=("${CONDA_ENV}/bin/torchrun" "--nproc_per_node=${GPUS}" "--master_port=${ROBOT_DIFT_MASTER_PORT:-29577}"
        "${common[@]}" "epoch=${EPOCHS}" "train_batch_size=${PER_GPU_BATCH}"
        gradient_accumulation_steps=1 "simulation.num_episode=${EVAL_EPISODES}"
        +skip_final_sim=True group=robot_dift_rawdense
        "hydra.sweep.dir=${HYDRA_DIR}" "hydra.sweep.subdir=train")
else
    cmd=("${CONDA_ENV}/bin/python" "${common[@]}" epoch=0 train_batch_size=1
        "simulation.num_episode=${EVAL_EPISODES}"
        "agents.ckpt_path=${POLICY_DIR}/last_model.pth"
        group=robot_dift_rawdense_eval
        "hydra.sweep.dir=${HYDRA_DIR}" "hydra.sweep.subdir=eval")
fi

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
for required in "${CONDA_ENV}/bin/python" "${CONDA_ENV}/bin/torchrun" \
    "${STUDENT}/metadata.json" "${MODEL_REPO}/model_index.json" "$CLIP_MODEL" "$DATASET"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing raw-dense policy prerequisite: $required" >&2
        exit 2
    fi
done
if [[ "$MODE" == eval && ! -f "${POLICY_DIR}/last_model.pth" ]]; then
    echo "Missing trained policy: ${POLICY_DIR}/last_model.pth" >&2
    exit 2
fi
if [[ "$MODE" == eval && -e "$EPISODE_LOG" && "${ROBOT_DIFT_ALLOW_EXISTING_EPISODE_LOG:-0}" != 1 ]]; then
    echo "Episode log already exists; set a new ROBOT_DIFT_EPISODE_LOG_PATH or explicitly allow resumption" >&2
    exit 2
fi
mkdir -p "$POLICY_DIR" "$HYDRA_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd "$REPO_ROOT"
exec "${cmd[@]}"
