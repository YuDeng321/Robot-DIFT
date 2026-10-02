# Stage-I training protocol and implementation choices

This page maps the paper's Stage-I settings to the implementation and states the reconstruction choices needed to run it. The [paper](https://arxiv.org/abs/2602.11934) specifies the method and experimental claims; `configs/release/paper_stage1_protocol.json` is the repository's machine-readable specification. `scripts/release/check_paper_protocol.py` compares that specification with the configuration resolved by `train_droid_auto.py` for the release launcher.

A passing check validates configuration agreement. It does not recover the original run, verify actual gradient paths, certify a checkpoint's training history, or reproduce downstream success. The exact readout and several optimizer/data settings are not uniquely specified by the manuscript.

## Launch the default preset

`train_droid_auto.py` uses the paper-derived preset plus the implementation choices below. `scripts/release/stage1_paper_protocol.sh` sets resources, paths, and the global batch, and refuses extra training arguments unless the run is marked as a smoke test (`ROBOT_DIFT_SMOKE=1`) or an ablation (`ROBOT_DIFT_ABLATION=1`). Follow [installation](INSTALLATION.md) before launching.

The original Stage-I resource configuration is **8 GPUs × per-GPU batch 32 × accumulation 1 = global batch 256**. The launcher defaults and this example use that configuration:

```bash
export WORK=/path/to/work
export ROBOT_DIFT_DATA_ROOT=/path/to/droid-tfds-root
export ROBOT_DIFT_MODEL_DIR=/path/to/sd2-1-base
export ROBOT_DIFT_CLIP_MODEL=/path/to/ViT-B-32.pt
export ROBOT_DIFT_OCTO_ROOT=/path/to/octo
export ROBOT_DIFT_GPUS=8
export ROBOT_DIFT_PER_GPU_BATCH=32
export ROBOT_DIFT_ACCUMULATION_STEPS=1

bash scripts/release/stage1_paper_protocol.sh --dry-run
python scripts/release/check_paper_protocol.py --check-assets
bash scripts/release/stage1_paper_protocol.sh
```

If available memory or GPU count requires a different microbatch, gradient accumulation can preserve global batch 256. For example, 4 GPUs × batch 16 × accumulation 4 is a resource adaptation, rather than the original training setup. Record both microbatch and accumulation in the resolved config and profile memory before choosing them. Camera/history expansion increases processed image count beyond the trajectory microbatch.

## Paper statement → implementation

| Paper | Setting | Where |
| --- | --- | --- |
| §2.1, §S5.1.2 | Frozen SD2.1 U-Net Teacher sees `x_t` at one uniformly sampled `t ∈ {1..999}` per view and step | `StableFeatureAligner.sample_timesteps` / `alignment_terms` (`--cleandift_num_t_bins 1`, `t_min 1`, `t_max 1000` exclusive) |
| §2.1, §S5.1.2 | Clean-input Student initialized as an exact copy of the SD2.1 U-Net | `StableFeatureAligner.__init__` loads the pipeline in float32; `--cleandift_student_init sd_teacher` |
| §S4.2, Fig. S1 | `L_align` = uniform sum over 11 maps (`mid`, `us1`–`us10`) of `1 − cos` | `reduce_alignment_loss_terms(..., "sum")`; `--cleandift_alignment_feature_keys` |
| §2.1, §S4.2 | Training-only, timestep-conditioned alignment adapters, dropped at deployment | `--cleandift_alignment_apply_feature_adapter` (default on); the release export drops `adapters.bin` |
| §2.1, Eq. 4 | `L = L_policy + λ(s) L_align`, λ linear 0.1 → 0.001 over 150k steps, then constant | `alignment_weight_at_step` with `min_decay_factor 0.01`, `power 1`, absolute step clock |
| §S5.1.3, Tab. S1 | 300k steps, Adam, lr 1e-4, linear schedule, global batch 256 | `train_droid_auto.py` defaults; the launcher enforces GPUs × batch × accumulation = 256 |
| Tab. S1 | EMA power 0.75 | `EMAModel` over trainable tensors; inverse-gamma and cap are implementation choices |
| §S5.1.3, Tab. S1 | 256 × 256 images, no augmentation | `observation.image_dim`, no observation randomizer |
| Tab. S1, §S5.1.1 | 1D U-Net diffusion policy, horizons 2/16/8, DDIM | robomimic `DiffusionPolicyUNet` with `ConditionalUnet1D` (widths from `algo.unet`) |
| §2.2, §S4.3–S4.4 | Coarse-to-fine S2-FPN over `us3/us6/us8` with a CLIP-conditioned readout into visual tokens | `RobotDIFTStage1Encoder` + `RobotDIFTPaperReadout`; exact readout operators are reconstruction choices |
| §2.2 | The same Student weights process every camera view | One shared Student for all configured cameras; views are max-pooled inside the readout |
| Abstract, §2 | Deterministic, single-pass Student at deployment | Raw `us3/us6/us8` at the learned Student timestep; this implementation uses VAE posterior `mode` |

## Choices the paper leaves open

These choices are fixed for the current preset. The specification records them so a new run can be compared; a paper table omitting an option is not evidence that the original run used its default value.

- Linear LR end factor 0.1; no warmup.
- Weight decay 0 for policy, Student, and readout; LR multipliers 1 for Student and readout. The paper reports Adam and LR 1e-4 but does not specify these group settings or weight decay.
- EMA inverse-gamma 1 without a decay cap (same as the Stage-II preset).
- VAE posterior `mode`, rather than sampling the posterior.
- Cameras: DROID wrist (`hand_camera_left`) and both exterior left views.
- CLIP ViT-B/32 text tower, shared by the readout queries and the policy's language goal in both stages.
- Readout: FPN width 256, model width 256, 8 heads, one Transformer layer, MLP 1024–512, 512-d output (the Stage-II defaults).
- Readout operators: Conv/GroupNorm/GELU FPN blocks, LayerNorm then Linear adapters, 77 CLIP token positions, 2D RoPE on visual keys with unrotated text queries, independent cross-attention per camera, and max aggregation before the Transformer. See `robot_dift_paper_readout.py` for masking and operator order; the paper does not fix all these details.
- Learned Student timestep initialized at 261 (CleanDIFT).
- bfloat16 autocast with float32 weights; gradient clipping at 1.0.
- 50k-frame shuffle buffer per rank. Octo shuffles encoded frames before decoding; actual host-memory use depends on encoded image sizes, cameras, and sequence length.
- Eight action-noise samples per policy example.
- The Student trains from the first step (`--student_freeze_steps 0`, `--student_lr_warmup_steps 0`).
- No policy proprioception input in this preset. Record modality changes explicitly.

The manuscript describes a frozen Student+S2-FPN deployment backbone in the main text, while the Stage-II appendix describes training the readout on a frozen Student. This implementation follows the appendix for task transfer. Initializing only FPN fusion from Stage I (`fusion`) or the full readout (`full`) is an explicit choice, not a recovered original-run setting.

## Implementation notes

- **Precision.** The Teacher and the initial Student are float32 copies of the SD2.1 weights. A trainable Student keeps float32 master weights and runs under bfloat16 autocast. VAE latents, the noise mixing, and the adapter time embedding stay in float32, so every Teacher timestep from 1 to 999 is represented exactly. If the GPU lacks bfloat16, the trainer stops; `--amp_dtype float16` (with a gradient scaler) must be chosen explicitly.
- **One Student pass.** The current frame's Student pass feeds both the readout and the alignment loss, and one pass serves every Teacher timestep bin. History frames skip the final up block when `us10` is not requested. Each distinct caption is encoded once per batch.
- **Optimizer step.** Gradients are synchronized, unscaled, clipped, and checked on device with a single host sync. A non-finite step is skipped on every rank and counted in `Nonfinite_Gradient_Skips`. EMA tracks only trainable tensors.
- **Inputs and assets.** Every configured camera must be present in each RLDS batch. The VAE, Teacher, and Student initialization come from one SD2.1 snapshot (`ROBOT_DIFT_MODEL_DIR`); a conflicting `ROBOT_DIFT_CLEANDIFT_MODEL_REPO` raises. Missing or partial Student weights raise when loading.
- **Readouts.** `--stage1_readout legacy` trains the built-in CleanDIFT S2-FPN/query head instead of the paper readout; the raw-dense Stage-II control uses that head.

## Train, validate, and export

1. **Preflight.** `bash scripts/release/stage1_paper_protocol.sh --dry-run`, then `python scripts/release/check_paper_protocol.py --check-assets`. For a startup check on one GPU, run `ROBOT_DIFT_SMOKE=1 ROBOT_DIFT_GPUS=1 ROBOT_DIFT_PER_GPU_BATCH=2 ROBOT_DIFT_ACCUMULATION_STEPS=1 bash scripts/release/stage1_paper_protocol.sh --num_epochs 20 --save_freq 20`, and confirm that `checkpoint-20-ema` passes `scripts/release/check_student_artifact.py`.
2. **Train.** The full default preset targets 300k steps, global batch 256. First run short controlled comparisons to justify the training recipe. Resume with `ROBOT_DIFT_RESUME_FROM=/path/to/model_epoch_N.pth`; the λ, LR, and EMA clocks continue from the restored step. A Student-only export is not a full training-resume checkpoint.
3. **Monitor the released interface, not only losses.** TensorBoard logs `Alignment/raw_cosine_us3|us6|us8` (raw Student vs. Teacher before adapters), per-map alignment terms, λ, per-group gradient norms, and `Nonfinite_Gradient_Skips`. Raw-cosine changes describe agreement with Teacher targets; they need not predict robot-task performance. Check the actual BC/alignment gradients reaching the Student and parameter changes when altering the training objective.
4. **Select checkpoints on transfer.** Export candidates are written every 10k steps plus steps 1k and 5k. Compare EMA snapshots with the correspondence, object-mask, and contact-point probes in `scripts/probes/` and with frozen-Student Stage-II success; `scripts/probes/compare_students.py` runs the probes against the SD2.1 initialization in one command (see the reproducibility guide). Lower action loss alone is not evidence of a better encoder.
5. **Release.** Run `export_encoder_release.py` on the selected `checkpoint-N-ema`, then `verify_encoder_release.py` and check feature parity after reloading. The package holds the Student U-Net, its timestep, the candidate readout (deploy head), and metadata with training configuration and source revision. Stage II can initialize only S2-FPN fusion (`ROBOT_DIFT_STAGE1_READOUT_SCOPE=fusion`, default) or the full readout (`full`). Raw extraction loads neither the readout nor the Teacher. See [artifact checks](REPRODUCIBILITY.md#artifact-checks).

Alignment weight follows the configured absolute step clock. At step 5,000 of the 300k preset, λ is approximately 0.0967, so a short run mostly tests the initial alignment regime. Compressing the schedule can answer a mechanism question, but is a different experiment. Sum versus mean changes gradient scale, and Adam does not necessarily change parameter updates proportionally. In an alignment-only Student variant, lowering λ does not introduce a policy gradient; measure the actual update rather than assuming an anneal releases robot adaptation.

## Optional ablations

Three switches target the balance between robot-domain adaptation and the SD2.1 prior. All are off by default, so the release launcher and the protocol check are unchanged. Compare each against a paper-protocol run with the same seed, steps, probes, and Stage-II tasks.

**Student freeze and warmup (Stage I).** `--student_freeze_steps N` keeps the Student U-Net and its timestep fixed for the first N optimizer steps while the readout, policy, and alignment adapters train. `--student_lr_warmup_steps M` then ramps the Student LR linearly over M steps. Only the Student optimizer group is scaled, for one step at a time, so the LR schedule, λ, and EMA clocks are unaffected. TensorBoard logs `Student_LR_Scale`; compare `Alignment/raw_cosine_us*` over the first steps with and without the freeze.

```bash
ROBOT_DIFT_ABLATION=1 ROBOT_DIFT_RUN_NAME=stage1_freeze2k_warmup3k \
  bash scripts/release/stage1_paper_protocol.sh --student_freeze_steps 2000 --student_lr_warmup_steps 3000
```

**Weight interpolation (after training).** `scripts/release/interpolate_student.py` writes checkpoints with U-Net weights `(1 − α)·init + α·Student` and the learned timestep interpolated the same way (WiSE-FT). `α = 1` is the trained Student and `α = 0` its SD2.1 initialization; a Student initialized from another checkpoint needs `--init-checkpoint`. The outputs load in the probes, `check_student_artifact.py`, Stage II, and `export_encoder_release.py`; the deploy head is copied from the trained checkpoint, and Stage II retrains the readout per task.

```bash
python scripts/release/interpolate_student.py \
  --checkpoint /path/to/encoder/checkpoint-300000-ema --model-repo "$ROBOT_DIFT_MODEL_DIR" \
  --alpha 0.25 0.5 0.75 1.0 --output-root /path/to/interpolated
```

**Student resolution (Stage II).** `ROBOT_DIFT_STAGE2_IMAGE_SIZE` (a multiple of 64, default 256) sets the frozen Student's input size in the Stage-II launchers through the Hydra override `agents.obs_encoders.resize_shape`, which is saved with the run and used again by `eval_stage2_checkpoint.py`. At 384 the maps are 12×12 and 24×24 instead of 8×8 and 16×16; the readout weights do not depend on the map size, so Stage-I initialization still applies. Student activations grow about 2.3×, so lower the per-GPU batch and raise accumulation to keep the global batch. The paper preset requires `ROBOT_DIFT_ABLATION=1` for any size other than 256, and non-default sizes get an `_img<size>` suffix in the default output directory.

```bash
ROBOT_DIFT_ABLATION=1 ROBOT_DIFT_STAGE2_IMAGE_SIZE=384 \
  ROBOT_DIFT_STAGE2_PER_GPU_BATCH=2 ROBOT_DIFT_STAGE2_ACCUMULATION_STEPS=16 \
  bash scripts/release/stage2_robocasa_paper_protocol.sh CoffeePressButton
```

The RoboCasa Stage-II launchers save the final EMA policy after 100 epochs, then load it in a fresh Python process for each of the 50 evaluation rollouts. The saved Hydra configuration is under `<policy-output>/hydra/run/.hydra/config.yaml`; per-episode records are preserved under `<policy-output>/sim_episodes_isolated/`, and the validated combined records are written to `<policy-output>/sim_episodes.jsonl`. An interrupted evaluation can resume from completed episodes by rerunning `eval_stage2_isolated.py` with the same arguments.

Use a declared short-run budget to screen variants, with matched initialization, data exposure, actual Student updates, readout, and downstream policy budget. Repeat promising methods across policy training seeds and evaluate held-out tasks/scenes before committing to a long run. With 50 independent rollouts, the single-policy standard error near 50% success is about 7 percentage points; paired episode differences and variation across trained policies must also be reported. There is no universal 10-point threshold that establishes or rules out improvement.
