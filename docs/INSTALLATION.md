# Installation

Choose the environment for the part of Robot-DIFT you need. Extracting frozen Student features does not require DROID, TensorFlow, Mamba, a simulator, or the downstream CLIP ViT-B/32 readout.

| Use | Dependencies |
| --- | --- |
| Extract raw `us3/us6/us8` maps | PyTorch + `requirements-inference.txt`, Student checkpoint, local SD2.1 assets |
| Train a downstream policy | PyTorch + `requirements.txt`, local `droid_policy_learning`, benchmark packages and demonstrations |
| Train Stage I on DROID | Full policy environment + `requirements-stage1.txt`, pinned Octo data loader and DROID TFDS |

Run the commands below from the repository root. Use Python 3.10 and a separate environment for this project.

## 1. Frozen encoder

```bash
git clone https://github.com/YuDeng321/Robot-DIFT.git
cd Robot-DIFT
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.4 example; select a PyTorch build compatible with your machine.
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-inference.txt

python -c 'from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor; print("Student loader import OK")'
```

The repository uses source imports: run examples from its root, or add the checkout to `PYTHONPATH`. The encoder loader can run on CPU with float32; GPU execution is recommended for practical throughput. The CUDA default is bfloat16, so use a compatible GPU or explicitly select float32/float16 as appropriate for your device.

### Model assets

Download the Robot-DIFT encoder checkpoint from [Google Drive](https://drive.google.com/drive/folders/1PHKWcls-hYn8TFP8ovZmwNwqBNiDakg0), extract it to a local directory, and set:

```bash
export ROBOT_DIFT_STAGE1_ENCODER=/path/to/robot-dift-encoder
export ROBOT_DIFT_MODEL_DIR=/path/to/sd2-1-base
export TORCHDYNAMO_DISABLE=1
```

Extract maps for several camera images with one shared Student:

```bash
python examples/extract_features.py \
  --checkpoint "$ROBOT_DIFT_STAGE1_ENCODER" \
  --model-repo "$ROBOT_DIFT_MODEL_DIR" \
  --images /path/to/front.png /path/to/wrist.png \
  --prompt "insert the pin" --output outputs/features.pt
```

The example resizes RGB images to 256 × 256, prints the map dimensions, and saves a CPU tensor dictionary. The Python API accepts an explicit `input_range` (`uint8`, `zero_one`, or `minus_one_one`) and does not resize inputs implicitly. Use the checkpoint's training prompt convention for task comparisons.

The raw Student loader needs these external SD2.1 components:

```text
sd2-1-base/
├── model_index.json
├── tokenizer/
├── text_encoder/
└── vae/
```

Stage-I training additionally needs the source `unet/` and scheduler configuration. Obtain the complete pinned snapshot using the source recorded in [`sd21_source.json`](../configs/release/sd21_source.json):

```bash
python - <<'PY'
import json
import os
from huggingface_hub import snapshot_download

with open("configs/release/sd21_source.json", encoding="utf-8") as stream:
    source = json.load(stream)
snapshot_download(
    repo_id=source["repo_id"],
    revision=source["revision"],
    local_dir=os.environ["ROBOT_DIFT_MODEL_DIR"],
)
PY
```

The raw encoder uses SD2.1's bundled text conditioner. The separate CLIP ViT-B/32 checkpoint is required only for the policy/readout paths; its URL and SHA-256 are in [`clip_source.json`](../configs/release/clip_source.json). Keep the text prompt, VAE posterior mode, preprocessing, and asset revision consistent with the encoder metadata. See the [encoder contract](REPRODUCIBILITY.md#encoder-contract).

## 2. Policy training and benchmark evaluation

Start from the PyTorch environment above, then install the policy requirements and the bundled robomimic-derived package together:

```bash
python -m pip install -r requirements.txt -e ./droid_policy_learning
python -m pip check
```

Installing the local package with its dependencies supplies its logging and data utilities, including `psutil`, `tqdm`, `tensorboard`, and `tensorboardX`. A separate upstream robomimic installation should not take precedence over the bundled package.

The full requirements include `mamba-ssm` for the BESO/Mamba controls. That extension can require a compatible CUDA toolkit and compiler; it is not part of the frozen-encoder installation. Install [RoboCasa](https://github.com/robocasa/robocasa) or [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and their demonstration assets following the respective upstream instructions. Record benchmark revisions, MuJoCo/rendering dependencies, and evaluation scene settings with each run.

Set the relevant paths before using the Stage-II launchers:

```bash
export WORK=/path/to/work
export ROBOT_DIFT_CLIP_MODEL=/path/to/ViT-B-32.pt
export ROBOT_DIFT_ROBOCASA_ROOT=/path/to/robocasa/single_stage
# For LIBERO instead:
export ROBOT_DIFT_LIBERO_ROOT=/path/to/libero-10
```

The [reproducibility guide](REPRODUCIBILITY.md#frozen-student-transfer) contains the launch commands and required comparison controls.

## 3. DROID Stage I

Stage I uses the TensorFlow/RLDS data path in addition to the policy environment:

```bash
python -m pip install -r requirements-stage1.txt -e ./droid_policy_learning \
  'dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972'

export WORK=/path/to/work
export ROBOT_DIFT_OCTO_ROOT="$WORK/src/octo"
mkdir -p "$WORK/src"
git clone https://github.com/octo-models/octo.git "$ROBOT_DIFT_OCTO_ROOT"
git -C "$ROBOT_DIFT_OCTO_ROOT" checkout 85b83fc19657ab407a7f56558a5384ae56fe453b
export PYTHONPATH="$PWD/droid_policy_learning:$ROBOT_DIFT_OCTO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -c 'from octo.data.dataset import make_interleaved_dataset; from robomimic.config import config_factory; print("Stage-I data imports OK")'
python -m pip check
```

The Octo source is used for data loading, rather than its JAX policy. The pinned `dlimp` revision comes from [Octo's requirements at the selected commit](https://github.com/octo-models/octo/blob/85b83fc19657ab407a7f56558a5384ae56fe453b/requirements.txt). The Stage-I launcher adds `ROBOT_DIFT_OCTO_ROOT` to `PYTHONPATH` when it contains `octo/`.

Acquire DROID from the [dataset project](https://droid-dataset.github.io/) and set `ROBOT_DIFT_DATA_ROOT` to the TFDS parent directory containing `droid/`. Keep dataset and statistics caches outside the checkout. Before allocating a training job, resolve the configuration and verify assets as described in [Stage-I protocol](STAGE1_PROTOCOL.md).

## Environment records and licenses

The development configuration recorded Python 3.10, PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`, diffusers `0.37.0`, transformers `5.4.0`, and safetensors `0.7.0`. Stage I additionally used TensorFlow `2.15.0` and TFDS `4.9.2`. These pins describe the current implementation environment; they are not a recovered environment lock for the paper's original run. Save `python -m pip freeze`, CUDA/driver versions, GPU model, source commit, and benchmark revisions for a new experiment.

Robot-DIFT's root [license](../LICENSE) applies to its own code. The bundled robomimic-derived code retains its [upstream license](../droid_policy_learning/LICENSE). CleanDIFT/Diffusers-derived source retains upstream attribution; external model weights and datasets keep their respective licenses and access terms. Obtain those assets from their original projects.
