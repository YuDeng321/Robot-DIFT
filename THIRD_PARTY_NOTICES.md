# Third-party notices

Robot-DIFT includes adapted third-party source. The [MIT license](LICENSE) covers
Robot-DIFT's original contributions; upstream notices and license texts are
retained below.

| Upstream project | Bundled source | License text |
| --- | --- | --- |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | `droid_policy_learning/robomimic/` | [Upstream MIT license](droid_policy_learning/LICENSE) |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | `agents/encoders/cleandift/src/min_sd15.py`, `min_sd21.py` | [Apache License 2.0](licenses/Apache-2.0.txt) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | `agents/models/beso/models/networks/clip.py`, `agents/models/beso/utils/clip_tokenizer.py`, and the bundled BPE vocabulary | [Upstream MIT license](licenses/OpenAI-CLIP-MIT.txt) |

## CleanDIFT and diffusion feature extraction

The CleanDIFT implementation under `agents/encoders/cleandift/` is adapted from
[CompVis/CleanDIFT](https://github.com/CompVis/cleandift). Robot-DIFT adds robot
training integration and feature extraction for downstream control. Its
original source attribution and file-level notices remain applicable; this
notice does not assign a repository-wide license to upstream CleanDIFT code.

The minimal SD1.5 and SD2.1 U-Net implementations identify Diffusers as their
source and retain attribution to Simo Ryu and Nick Stracke. Robot-DIFT adapts
these implementations for selected decoder feature outputs and deployment.

## Other retained notices

The learning-rate scheduler sources under
`agents/models/beso/utils/lr_schedulers/` retain their MIT notices for
Soohwan Kim, Sangchun Ha, and Soyoung Cho.

Depth utilities in `agents/encoders/cleandift/src/depth.py` retain links to the
[DINOv2](https://github.com/facebookresearch/dinov2) sources from which they
were adapted.

Pretrained model files and datasets are external assets. Their upstream
licenses and terms accompany those assets.
