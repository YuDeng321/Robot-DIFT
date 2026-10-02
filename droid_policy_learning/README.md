# DROID Stage I training integration

This directory contains the [robomimic](https://robomimic.github.io/) training fork used for Robot-DIFT's DROID Stage I. Its upstream MIT license is preserved in [`LICENSE`](LICENSE). Robot-DIFT-specific changes include the diffusion Student/Teacher encoder, alignment losses, distributed gradient handling, and checkpoint export.

Install the top-level Robot-DIFT environment, then install this package without replacing its dependency versions:

```bash
python -m pip install -r requirements.txt
python -m pip install -e droid_policy_learning --no-deps
```

Stage I loads DROID in RLDS/TFDS format through [Octo](https://github.com/octo-models/octo) at commit `85b83fc19657ab407a7f56558a5384ae56fe453b`. Use the top-level [`README.md`](../README.md) and [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for inputs, launchers, encoder contract, and evaluation criteria.
