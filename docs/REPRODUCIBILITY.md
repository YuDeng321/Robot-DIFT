# Robot-DIFT reproducibility guide

This guide describes the encoder interface, training inputs, and evidence needed to compare Robot-DIFT checkpoints. The [paper](https://arxiv.org/abs/2602.11934) defines the scientific claims. [`STAGE1_PROTOCOL.md`](STAGE1_PROTOCOL.md) distinguishes paper-specified settings from the implementation choices made here. A configuration check establishes agreement with the repository's specification; it does not establish original-run provenance or reproduce the paper's benchmark results.

## Encoder contract

| Component | Stage I | Downstream tasks |
| --- | --- | --- |
| SD2.1 Teacher | Frozen; provides noise-conditioned alignment targets | Absent |
| Clean-input Student U-Net | The default preset initializes from SD2.1 and adapts on DROID; alternative initialization must be recorded | Frozen and shared by all cameras |
| Alignment adapters | Training only | Absent |
| VAE and text components | External SD2.1 assets | External SD2.1 assets |
| Feature maps | `us3`, `us6`, `us8` are the release interface | The same raw maps are exposed to every task |
| S2-FPN + CLIP readout | Trains with the DROID policy | Each task trains its readout; Stage-I initialization uses fusion by default, `full` optional |

The Student processes camera views with the **same weights**. Three cameras increase activation memory, not the Student parameter count. Keep the Stage-I Student fixed when comparing policy readouts or tasks.

Raw extraction accepts RGB tensors `[B, 3, H, W]`. The policy paths resize to 256 × 256 and normalize to `[-1, 1]`; preserve that preprocessing, camera convention, and caption when comparing weights. At 256 × 256, this implementation exposes `us3` with 1280 channels at 8 × 8, `us6` with 1280 channels at 16 × 16, and `us8` with 640 channels at 16 × 16. Tap names identify the implemented decoder outputs, including their upsampling convention; they do not guarantee three distinct spatial resolutions.

The checkpoint records the learned Student timestep and VAE posterior mode. New default exports use posterior `mode` for deterministic extraction. Older exports without the field are interpreted as `sample` and remain stochastic; do not silently override their metadata when comparing them.

- `agents/encoders/robot_dift_stage1_encoder.py`: Stage-I multi-view encoder (shared Student, Teacher alignment, readout).
- `agents/encoders/robot_dift_paper_readout.py`: the S2-FPN + CLIP readout used in both stages. Operator details the manuscript leaves open are fixed there.
- `agents/encoders/cleandift_img_encoder.py`: the Student/Teacher module; `readout="legacy"` selects the built-in S2-FPN/query head used by the raw-dense Stage-II control.
- `agents/encoders/robot_dift_student_feature_extractor.py`: the public frozen Student for downstream use.

## Inputs and environment

Follow [`INSTALLATION.md`](INSTALLATION.md) for the separate frozen-encoder and full-training environments. Raw feature extraction uses `requirements-inference.txt` and does not require TensorFlow, Mamba, or benchmark simulators. The upstream requirements inside `droid_policy_learning` describe its parent project; install the bundled package alongside this repository's constraints. Record the exact environment, CUDA, driver, GPU model, source commit, and benchmark revisions for every run.

| Variable | Resource |
| --- | --- |
| `WORK` | Writable root for logs and outputs |
| `ROBOT_DIFT_DATA_ROOT` | DROID TFDS root containing `droid/` |
| `ROBOT_DIFT_MODEL_DIR` | SD2.1 Diffusers snapshot containing `model_index.json` (Teacher, Student init, VAE, text encoder) |
| `ROBOT_DIFT_OCTO_ROOT` | Pinned Octo checkout containing `octo/`, required by the DROID loader |
| `ROBOT_DIFT_CLIP_MODEL` | Local CLIP ViT-B/32 file for the readout queries and the policy language goal (both stages) |
| `ROBOT_DIFT_STAGE1_ENCODER` | Exported Student checkpoint for Stage II |
| `ROBOT_DIFT_ROBOCASA_ROOT` | RoboCasa demonstration root |
| `ROBOT_DIFT_LIBERO_ROOT` | LIBERO-10 demonstration root |
| `ROBOT_DIFT_POLICY_OUTPUT_DIR` or `ROBOT_DIFT_POLICY_DIR` | Stage-II policy output directory, depending on launcher |

Pretrained model identifiers, revisions, and checksums are in `configs/release/sd21_source.json` and `configs/release/clip_source.json`. DROID Stage I uses [Octo](https://github.com/octo-models/octo) commit `85b83fc19657ab407a7f56558a5384ae56fe453b` for RLDS input; the trainer logs the TFDS version and post-filter trajectory and transition counts. Record dataset version, filters, splits, statistics, action normalization, and any preprocessing changes with the run. Install benchmark simulators separately.

## Train and compare

1. Run `bash scripts/release/stage1_paper_protocol.sh --dry-run` and `python scripts/release/check_paper_protocol.py --check-assets` before a GPU run. The checker resolves the launcher's training configuration. The default preset should match its specification; save the resolved config and explicitly document differences for an ablation. Checks do not inspect a completed training run or checkpoint gradients.
2. For a one-GPU startup check, run `ROBOT_DIFT_SMOKE=1 ROBOT_DIFT_GPUS=1 ROBOT_DIFT_PER_GPU_BATCH=2 ROBOT_DIFT_ACCUMULATION_STEPS=1 bash scripts/release/stage1_paper_protocol.sh --num_epochs 20 --save_freq 20`, then `python scripts/release/check_student_artifact.py <encoder>/checkpoint-20-ema --model-repo "$ROBOT_DIFT_MODEL_DIR"`.
3. The Stage-I preset uses 300,000 optimizer steps and global batch 256. The original training resource configuration is **8 GPUs × batch 32 per GPU × accumulation 1**. Accumulation is available to adapt resources; record it separately rather than presenting it as the original configuration. Every schedule (learning rate, alignment weight, EMA, saves) counts optimizer steps, and full policy checkpoints resume optimizer, scheduler, and EMA state. The RLDS iterator is rebuilt on resume, so segmented runs are not bitwise identical to uninterrupted training.
4. Watch the released interface during training: `Alignment/raw_cosine_us3|us6|us8`, the per-map alignment terms, `Alignment_Weight`, per-group gradient norms, and `Nonfinite_Gradient_Skips`. A lower action loss or projected Teacher alignment loss alone does not establish a better encoder.
5. Compare checkpoint candidates at the **deployed** `us3/us6/us8` interface: multi-scale correspondence, object-mask and contact-point probes, and frozen-Student downstream success. Include matched SD2.1 and public CleanDIFT baselines where applicable. `scripts/release/interpolate_student.py` adds candidates between the initialization and a trained Student. Test Student reload, finite maps, the checkpoint's expected determinism, and export integrity.
6. Compare policies with the same demonstrations, readout, seeds, episode count, and scene split. Train readout/policy separately for each task while keeping the Student frozen. The paper Stage-II presets use the same readout as Stage I; the raw-dense launcher uses the legacy S2-FPN route. Do not mix their checkpoints in a matched comparison.

### Is the DROID-adapted Student better than SD2.1 DIFT?

`scripts/probes/compare_students.py` compares a trained Student with its initialization. By default, `interpolate_student.py --alpha 0` writes an SD2.1 initialization candidate so both use the same loader, text conditioning, and VAE mode. For a Student initialized from another checkpoint, pass `--init-checkpoint`; otherwise the reference is incorrect. Probes use identical inputs and produce paired bootstrap differences.

```bash
python scripts/probes/compare_students.py \
  --checkpoint /path/to/encoder/checkpoint-300000-ema --alpha 0.5 \
  --descriptor-images /path/to/robocasa_descriptor_images \
  --contact-images /path/to/contact_points --mask-images /path/to/object_masks \
  --droid-root "$ROBOT_DIFT_DATA_ROOT" --output-dir /path/to/student_comparison
```

Add `--extra LABEL=PATH` for other Students (for example an earlier run) and `--dry-run` to list the steps first; finished steps are reused. `summary.md` collects the results:

| Question | Probe | Read |
| --- | --- | --- |
| Finer spatial detail? | Correspondence under known warps; cross-demo contact points (RoboCasa) | PCK@4/8 px up, pixel error down, per map `us3/us6/us8` |
| Object-level structure? | Leave-one-demo-out target masks (RoboCasa) | Average precision and ROC-AUC up |
| More control-relevant information? | `droid_multiview_probe.py` on held-out DROID failure episodes: grouped-CV ridge from grid-pooled maps to end-effector position and gripper now, and end-effector displacement and gripper 8 steps ahead | ΔR² with an episode-bootstrap CI, for each camera alone and for all cameras |
| How are the cameras combined? | Same probe, Stage-I paper readout | Share of max-pooled query channels each camera wins, overall and with the gripper closed; how far the policy token moves without each camera (in units of its spread across frames) |

The DROID frames are held out but come from the training domain; the RoboCasa probes measure transfer. An interpolated Student (`--alpha`) doing better on RoboCasa but worse on DROID is evidence of a possible adaptation/generalization trade-off, not proof of its cause. Frozen-Student Stage-II success remains the deciding test. For a comparison of two Students, keep readout architecture and initialization, demonstrations, training budget, prompts, and policy seeds matched.

## Frozen-Student transfer

The RoboCasa preset trains a 1D U-Net diffusion policy with horizons 2/16/8, a trainable S2-FPN/readout, 100 epochs, and 50 final evaluation episodes. The Student remains frozen. These numerical settings follow the paper; CLIP variant, exact readout operators, readout initialization scope, and DDIM inference-step count are reconstruction choices.

```bash
export ROBOT_DIFT_STAGE1_ENCODER=/path/to/robot-dift-encoder
export ROBOT_DIFT_ROBOCASA_ROOT=/path/to/robocasa/single_stage
export ROBOT_DIFT_POLICY_OUTPUT_DIR="$WORK/runs/robocasa/CoffeePressButton/seed42"

bash scripts/release/stage2_robocasa_paper_protocol.sh --dry-run CoffeePressButton
bash scripts/release/stage2_robocasa_paper_protocol.sh CoffeePressButton
```

The Stage-II launcher's resource default is 8 GPUs × batch 4 × accumulation 8, giving global batch 256. This is a resource adaptation for the current implementation. Use `ROBOT_DIFT_STAGE2_GPUS`, `ROBOT_DIFT_STAGE2_PER_GPU_BATCH`, and `ROBOT_DIFT_STAGE2_ACCUMULATION_STEPS` to change the allocation while preserving the declared batch. The output-directory name does not set the training seed; inspect and explicitly fix the resolved Hydra configuration when constructing seed comparisons.

`ROBOT_DIFT_STAGE1_READOUT_SCOPE=fusion` initializes the FPN portion of the task readout; `full` initializes the entire readout. This is a comparison variable. The `stage2_robocasa_rawdense.sh` BESO/Mamba control uses a legacy readout; its checkpoints are not interchangeable with the paper-readout preset.

Save the resolved Hydra config, selected encoder/policy checksums, train/validation/final scene splits, policy training seeds, and per-episode evaluation records. Verify that paired policies use the same actual reset states and camera setup, rather than relying on a matching seed label alone. Missing or failed rollout records are failures of evaluation, not zero-success observations. Confirm promising methods across policy training seeds and use tasks or scenes excluded from method selection for final transfer claims.

The public feature probes live in `scripts/probes/`. `scripts/release/check_stage1_descriptor_gate.py`, `check_stage1_stage2_transfer.py`, and `check_stage1_promotion_report.py` provide structured checks. A release claim should be tied to complete episode logs and a declared checkpoint-selection rule; Figure 1 and the abstract report the paper's experiments, not an automatic guarantee for every newly trained weight.

## Artifact checks

Run `scripts/release/export_encoder_release.py` on the selected EMA Student and then `verify_encoder_release.py` against the same external SD2.1 source:

```bash
python scripts/release/export_encoder_release.py \
  --checkpoint /path/to/stage1/encoder/checkpoint-300000-ema \
  --model-repo "$ROBOT_DIFT_MODEL_DIR" --output /path/to/robot-dift-encoder
python scripts/release/verify_encoder_release.py \
  --package /path/to/robot-dift-encoder --model-repo "$ROBOT_DIFT_MODEL_DIR"
```

The export contains the Student U-Net, its timestep, the deploy head (the Stage-I readout), metadata, and a SHA-256 manifest. It excludes the frozen Teacher, training policy, and alignment adapters. Raw extraction does not load the deploy head. Runtime use still needs the pinned SD2.1 VAE and text components. Metadata records training settings, VAE posterior mode, source git revision, and a hash of original training metadata. Local data, checkpoint, model, and log paths are replaced with portable asset variables in the public package.

The verifier checks packaged file hashes and the external `model_index.json` hash. Verify the external component weight checksums against `sd21_source.json` as well: the model-index check alone does not certify VAE or text weights. Compare features before and after export/reload on fixed images, prompts, precision, and preprocessing before publishing a weight.

`release/source_manifest.json` lists every public source file. `scripts/release/validate_source_manifest.py` checks the selected paths and refuses data or checkpoint types. The `.gitignore` additionally excludes local data, logs, checkpoints, model weights, and credentials from normal Git operations.
