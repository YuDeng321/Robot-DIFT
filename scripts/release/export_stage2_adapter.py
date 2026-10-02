#!/usr/bin/env python3
"""Export a policy adapter from a trusted local Stage-II full checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from release.stage2_adapter import export_stage2_adapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    args = parser.parse_args()
    result = export_stage2_adapter(
        args.full_checkpoint,
        args.output_dir,
        stage1_checkpoint=args.stage1_checkpoint,
        clip_model=args.clip_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
