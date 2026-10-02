"""Catch imports from a sibling checkout when robomimic loads the encoder first."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robomimic_encoder_import_uses_this_checkout():
    script = """
import inspect
from pathlib import Path
from robomimic.models.cleandift_backbone import CleanDIFTImgEncoder
expected = Path.cwd() / 'agents' / 'encoders' / 'cleandift_img_encoder.py'
actual = Path(inspect.getfile(CleanDIFTImgEncoder)).resolve()
assert actual == expected, (actual, expected)
assert 'readout' in inspect.signature(CleanDIFTImgEncoder.__init__).parameters
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "droid_policy_learning"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
